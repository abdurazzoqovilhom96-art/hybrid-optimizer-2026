"""Main experiment driver.

Runs the full protocol and writes raw results incrementally, so a long run can
be interrupted and resumed without losing work. Analysis is deliberately a
separate step (``analyze.py``): a bug in a table or a figure must never destroy
hours of compute.

Usage (Windows or Linux)
------------------------
    python run_all.py --smoke                       # 10D, 3 runs, ~3 minutes
    python run_all.py --dims 30 50 100 --runs 30 --jobs 20
    python run_all.py --dims 30 50 100 --runs 30 --jobs 20 --resume

On a 16-core / 24-thread machine ``--jobs 20`` is a good setting: it leaves
headroom for the OS and keeps memory use well under control. Each worker is
pinned to a single BLAS thread, otherwise NumPy's internal threads oversubscribe
the CPU and the run gets *slower* as jobs increase.
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

from temoa.problems import FUNC_NAMES
from temoa.problems import make_problem
from temoa.registry import ALGORITHMS, ALL_ALGORITHMS
from temoa.tracker import Tracker

SEED_BASE = 42          # algorithm streams
NOISE_SEED = 99         # noise streams: matched across algorithms within a run
N_POINTS = 100          # convergence-curve checkpoints

RAW_COLUMNS = ["Algorithm", "Function", "Dimension", "Run", "Error", "Seconds", "FES", "Overrun"]


def run_task(alg_name, func_name, dim, run_id, max_fes, suite, alg_index):
    """One (algorithm, function, dimension, run). Returns a row and a curve."""
    problem = make_problem(func_name, dim, suite=suite,
                           noise_rng=np.random.default_rng([NOISE_SEED, dim, run_id]))
    tracker = Tracker(problem, max_fes, N_POINTS)
    rng = np.random.default_rng([SEED_BASE, dim, run_id, alg_index])

    t0 = time.perf_counter()
    ALL_ALGORITHMS[alg_name](tracker, dim, (problem.lb, problem.ub), max_fes, rng)
    seconds = time.perf_counter() - t0

    curve, error = tracker.finalize()
    row = {"Algorithm": alg_name, "Function": func_name, "Dimension": dim,
           "Run": run_id, "Error": error, "Seconds": seconds,
           "FES": tracker.fes, "Overrun": tracker.overrun}
    return row, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description="TEMOA benchmark study")
    ap.add_argument("--dims", type=int, nargs="+", default=[30, 50, 100])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--suite", choices=["rotated", "shift"], default="rotated",
                    help="'rotated' = CEC-style (default); 'shift' = the original study's suite")
    ap.add_argument("--fes-per-dim", type=int, default=3000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--algos", nargs="+", default=None,
                    help="subset of algorithm names; default = the main line-up")
    ap.add_argument("--with-weakened", action="store_true",
                    help="also run PSO_orig and HHO_orig (the original study's configurations)")
    ap.add_argument("--functions", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="10D, 3 runs, reduced budget")
    args = ap.parse_args(argv)

    if args.smoke:
        args.dims, args.runs, args.fes_per_dim = [10], 3, 1000
        args.out = args.out + "_smoke"

    algos = args.algos or list(ALGORITHMS)
    if args.with_weakened and not args.algos:
        algos = list(ALL_ALGORITHMS)
    unknown = [a for a in algos if a not in ALL_ALGORITHMS]
    if unknown:
        ap.error(f"unknown algorithms: {unknown}. Available: {list(ALL_ALGORITHMS)}")
    funcs = args.functions or FUNC_NAMES

    out = Path(args.out)
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
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "suite": args.suite,
        "dims": args.dims, "runs": args.runs, "fes_per_dim": args.fes_per_dim,
        "algorithms": algos, "functions": funcs,
        "seed_base": SEED_BASE, "noise_seed": NOISE_SEED,
        "seed_scheme": "algorithm rng = default_rng([SEED_BASE, dim, run, alg_index]); "
                       "noise rng = default_rng([NOISE_SEED, dim, run]) (matched across algorithms)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    alg_index = {a: i for i, a in enumerate(sorted(ALL_ALGORITHMS))}
    total = len(algos) * len(funcs) * len(args.dims) * args.runs
    print(f"[*] suite={args.suite}  algorithms={len(algos)}  functions={len(funcs)}  "
          f"dims={args.dims}  runs={args.runs}")
    print(f"[*] {total} runs total, {max(0, total - len(done))} to do, jobs={args.jobs}")

    t_start = time.perf_counter()
    completed = len(done)

    for dim in args.dims:
        max_fes = args.fes_per_dim * dim
        curves = {}
        curve_path = out / "curves" / f"curves_{dim}D.npz"
        if args.resume and curve_path.exists():
            curves = dict(np.load(curve_path))

        for func in funcs:
            todo = [(a, r) for a in algos for r in range(args.runs)
                    if (a, func, dim, r) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            results = Parallel(n_jobs=args.jobs, backend="loky")(
                delayed(run_task)(a, func, dim, r, max_fes, args.suite, alg_index[a])
                for a, r in todo)

            rows = [row for row, _ in results]
            pd.DataFrame(rows)[RAW_COLUMNS].to_csv(
                raw_csv, mode="a", header=not raw_csv.exists(), index=False)
            for (a, r), (_, curve) in zip(todo, results):
                curves[f"{a}|{func}|{r}"] = curve
            np.savez_compressed(curve_path, **curves)

            completed += len(todo)
            elapsed = time.perf_counter() - t_start
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / rate if rate > 0 else float("nan")
            print(f"  {dim:3d}D {func:16s} {len(todo):5d} runs in {time.perf_counter()-t0:7.1f}s "
                  f"| {completed}/{total} | ETA {eta/60:6.1f} min", flush=True)

    print(f"[*] finished in {(time.perf_counter()-t_start)/60:.1f} min -> {raw_csv}")
    print(f"[*] next: python analyze.py --out {args.out}")


if __name__ == "__main__":      # required on Windows (loky spawns processes)
    main()
