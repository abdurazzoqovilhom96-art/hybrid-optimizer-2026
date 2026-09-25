"""Main experiment driver.

Runs a benchmark suite under **its own official protocol** and writes raw results
incrementally, so a long run can be interrupted and resumed without losing work.
Analysis is a separate step (``analyze.py``): a bug in a table must never destroy
hours of compute.

The protocol -- number of runs, evaluation budget, dimensions -- comes from the
suite, not from the command line. The original study used 30 runs and 3000*D
evaluations, which matches no competition and makes its numbers incomparable
with any published table. Overrides exist but are recorded in the manifest.

Usage
-----
    python run_all.py --suite cec2017 --dims 10 --runs 5 --jobs 4   # quick check
    python run_all.py --suite cec2017 --dims 10 30 --jobs 20        # the gate
    python run_all.py --suite legacy --jobs 20                      # regression
    python run_all.py --suite cec2017 --dims 10 30 --jobs 20 --resume

On a 16-core / 24-thread machine ``--jobs 20`` is a good setting. Each worker is
pinned to a single BLAS thread; without that pinning NumPy's internal threads
oversubscribe the CPU and more jobs makes the run *slower*.
"""

from __future__ import annotations

# Must precede the NumPy import in every worker: loky spawns fresh processes
# that inherit this environment.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from temoa.registry import ALL_ALGORITHMS, LEGACY_SWARM, MODERN, WEAKENED
from temoa.suites import SUITES, make_suite_problem
from temoa.tracker import Tracker

SEED_BASE = 42          # algorithm streams
NOISE_SEED = 99         # noise streams: matched across algorithms within a run
N_POINTS = 100          # convergence-curve checkpoints

RAW_COLUMNS = ["Algorithm", "Function", "Dimension", "Run", "Error", "Seconds", "FES", "Overrun"]


def run_task(alg_name, fid, dim, run_id, max_fes, suite, legacy_track, alg_index):
    """One (algorithm, function, dimension, run). Returns a row and a curve."""
    problem = make_suite_problem(
        suite, fid, dim,
        noise_rng=np.random.default_rng([NOISE_SEED, dim, run_id]),
        legacy_track=legacy_track)
    tracker = Tracker(problem, max_fes, N_POINTS)
    rng = np.random.default_rng([SEED_BASE, dim, run_id, alg_index])

    t0 = time.perf_counter()
    ALL_ALGORITHMS[alg_name](tracker, dim, (problem.lb, problem.ub), max_fes, rng)
    seconds = time.perf_counter() - t0

    curve, error = tracker.finalize()
    row = {"Algorithm": alg_name, "Function": SUITES[suite].label(fid), "Dimension": dim,
           "Run": run_id, "Error": error, "Seconds": seconds,
           "FES": tracker.fes, "Overrun": tracker.overrun}
    return row, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description="TEMOA benchmark study")
    ap.add_argument("--suite", choices=sorted(SUITES), default="cec2017",
                    help="benchmark suite; its official protocol is used unless overridden")
    ap.add_argument("--dims", type=int, nargs="+", default=None,
                    help="override the suite's dimensions")
    ap.add_argument("--runs", type=int, default=None, help="override the suite's run count")
    ap.add_argument("--fes-per-dim", type=int, default=None,
                    help="override the suite's budget with this multiple of D")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--legacy-track", choices=["rotated", "shift"], default="rotated",
                    help="only for --suite legacy: 'shift' reproduces the original study")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tier", choices=["modern", "all", "legacy"], default="modern",
                    help="'modern' = the comparison that counts; 'all' adds the swarm "
                         "baselines and the original study's weakened configurations")
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--functions", nargs="+", default=None,
                    help="official function ids (CEC) or names (legacy)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="smallest dimension, 3 runs, tiny budget")
    args = ap.parse_args(argv)

    spec = SUITES[args.suite]

    dims = args.dims or list(spec.dims)
    runs = args.runs or spec.runs
    fes_of = (lambda d: args.fes_per_dim * d) if args.fes_per_dim else spec.max_fes
    out = Path(args.out or f"results_{args.suite}")

    if args.smoke:
        dims, runs = [min(spec.dims)], 3
        fes_of = lambda d: 1000 * d          # noqa: E731
        out = Path(str(out) + "_smoke")

    if args.tier == "modern":
        algos = list(MODERN)
    elif args.tier == "legacy":
        algos = list(LEGACY_SWARM) + list(WEAKENED)
    else:
        algos = list(ALL_ALGORITHMS)
    if args.algos:
        algos = args.algos
    unknown = [a for a in algos if a not in ALL_ALGORITHMS]
    if unknown:
        ap.error(f"unknown algorithms: {unknown}. Available: {list(ALL_ALGORITHMS)}")

    if args.functions:
        funcs = [int(f) if args.suite != "legacy" else f for f in args.functions]
    else:
        funcs = list(spec.function_ids)

    overrides = {k: v for k, v in
                 {"dims": args.dims, "runs": args.runs, "fes_per_dim": args.fes_per_dim}.items()
                 if v is not None}
    if not spec.protocol_verified:
        print(f"[!] {args.suite}: protocol NOT verified against a primary source.")
        print(f"[!] {spec.note}")
        print("[!] Do not quote numbers from this suite until it is verified.")
    if overrides or args.smoke:
        print(f"[!] protocol overridden: {overrides or 'smoke'} -- results are NOT "
              f"comparable with published {args.suite} tables.")

    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    raw_csv = out / "raw" / "results.csv"

    done = set()
    if args.resume and raw_csv.exists():
        prev = pd.read_csv(raw_csv, float_precision="round_trip")
        done = set(map(tuple, prev[["Algorithm", "Function", "Dimension", "Run"]].to_numpy()))
        print(f"[resume] {len(done)} runs already recorded in {raw_csv}")
    elif raw_csv.exists():
        raw_csv.unlink()

    manifest = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "suite": args.suite,
        "suite_protocol_verified": spec.protocol_verified,
        "suite_note": spec.note,
        "protocol_overrides": overrides or ({"smoke": True} if args.smoke else {}),
        "dims": dims, "runs": runs,
        "max_fes_per_dim": {int(d): int(fes_of(d)) for d in dims},
        "tier": args.tier, "algorithms": algos,
        "functions": [spec.label(f) for f in funcs],
        "legacy_track": args.legacy_track if args.suite == "legacy" else None,
        "seed_base": SEED_BASE, "noise_seed": NOISE_SEED,
        "seed_scheme": "algorithm rng = default_rng([SEED_BASE, dim, run, alg_index]); "
                       "noise rng = default_rng([NOISE_SEED, dim, run]) (matched across algorithms)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    alg_index = {a: i for i, a in enumerate(sorted(ALL_ALGORITHMS))}
    total = len(algos) * len(funcs) * len(dims) * runs
    print(f"[*] suite={args.suite} tier={args.tier} algorithms={len(algos)} "
          f"functions={len(funcs)} dims={dims} runs={runs}")
    print(f"[*] budget: " + ", ".join(f"{d}D={fes_of(d):,} FES" for d in dims))
    print(f"[*] {total} runs total, {max(0, total - len(done))} to do, jobs={args.jobs}")

    t_start = time.perf_counter()
    completed = len(done)

    for dim in dims:
        max_fes = int(fes_of(dim))
        curve_path = out / "curves" / f"curves_{dim}D.npz"
        curves = dict(np.load(curve_path)) if (args.resume and curve_path.exists()) else {}

        for fid in funcs:
            label = spec.label(fid)
            todo = [(a, r) for a in algos for r in range(runs)
                    if (a, label, dim, r) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            results = Parallel(n_jobs=args.jobs, backend="loky")(
                delayed(run_task)(a, fid, dim, r, max_fes, args.suite, args.legacy_track,
                                  alg_index[a])
                for a, r in todo)

            pd.DataFrame([row for row, _ in results])[RAW_COLUMNS].to_csv(
                raw_csv, mode="a", header=not raw_csv.exists(), index=False)
            for (a, r), (_, curve) in zip(todo, results):
                curves[f"{a}|{label}|{r}"] = curve
            np.savez_compressed(curve_path, **curves)

            completed += len(todo)
            elapsed = time.perf_counter() - t_start
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / rate if rate > 0 else float("nan")
            print(f"  {dim:3d}D {label:8s} {len(todo):5d} runs in {time.perf_counter()-t0:7.1f}s "
                  f"| {completed}/{total} | ETA {eta/60:6.1f} min", flush=True)

    print(f"[*] finished in {(time.perf_counter()-t_start)/60:.1f} min -> {raw_csv}")
    print(f"[*] next: python analyze.py --out {out}")


if __name__ == "__main__":      # required on Windows (loky spawns processes)
    main()
