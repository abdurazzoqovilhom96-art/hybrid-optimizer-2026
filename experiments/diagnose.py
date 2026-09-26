"""Why does L-SHADE-DGR lose on F4 and F5, and what is the cost of the fix?

The D=10 gate at 51 runs put this algorithm first overall, but the losses are
not spread evenly -- they are concentrated, and two of them are qualitative
rather than marginal:

    F4  Shifted Rotated Rosenbrock   ours 3.99   CMA-ES 0.00   A12 = 0.186
    F5  Shifted Rotated Rastrigin    ours 1.99   IPOP    0.00   A12 = 0.181

Everything else we lose by a factor of one or two, which is noise next to this.
Theory says where to look: DE/rand/1/bin is invariant under diagonal affine
maps but *not* under rotation, because binomial crossover selects coordinates
in a fixed basis while a rotation mixes them. CMA-ES is fully affine invariant.
F4 and F5 are both f(R(x - o)). So the losses are where the theory predicts.

The eigen crossover exists precisely to restore rotation invariance
approximately (it is not ours -- LSHADE-cnEpSin and EA4eig got there first), so
the question is not "what is missing" but "which of our own components is
preventing the one that should help". This script answers that by flipping one
flag at a time and by reading the algorithm's own instrumentation.

TWO MEASUREMENTS, AND WHY BOTH ARE NEEDED
-----------------------------------------
1. **Error per configuration.** Which flag, flipped, recovers F4 and F5.
2. **The cost of that flip elsewhere.** A component that hurts on F4 was added
   for a reason; the diversity guard was added because instrumentation measured
   66% of all fitness gain going to an operator that was collapsing the
   population. A fix that recovers F4 and loses the hybrid class is not a fix,
   so the winning functions are measured in the same run.

Running it on the losing functions alone would produce a confident wrong answer.

    python experiments/diagnose.py --runs 15 --jobs 20
    python experiments/diagnose.py --runs 5 --jobs 4 --quick   # smoke, ~10 min
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.algorithms.lshade_dgr import LSHADE_DGR as G        # noqa: E402
from temoa.algorithms.modern import BIPOP_CMAES, IPOP_CMAES, jSO  # noqa: E402
from temoa.registry import algorithm_seed                       # noqa: E402
from temoa.stats import paired_wilcoxon, vargha_delaney_a12     # noqa: E402
from temoa.suites import make_suite_problem                     # noqa: E402
from temoa.tracker import Tracker                               # noqa: E402

SEED_BASE = 42
BASELINE = "default"

#: One flag flipped per configuration, so any difference is attributable to it.
CONFIGS = {
    BASELINE:              lambda o, d, b, m, r: G(o, d, b, m, r),
    # -- the eigen crossover: present, but is it reaching the run? ------------
    "P_EIG=0.9 fixed":     lambda o, d, b, m, r: G(o, d, b, m, r, P_EIG=0.9, ADAPT_EIG=False),
    "P_EIG=0.5 fixed":     lambda o, d, b, m, r: G(o, d, b, m, r, P_EIG=0.5, ADAPT_EIG=False),
    "P_EIG=0 (no eigen)":  lambda o, d, b, m, r: G(o, d, b, m, r, P_EIG=0.0, ADAPT_EIG=False),
    "no eigen gate":       lambda o, d, b, m, r: G(o, d, b, m, r, EIGEN_GATE=False),
    # -- our own additions, each suspect --------------------------------------
    "no diversity guard":  lambda o, d, b, m, r: G(o, d, b, m, r, DIV_GUARD=False),
    "guard thresh 1e-6":   lambda o, d, b, m, r: G(o, d, b, m, r, DIV_THRESH=1e-6),
    "no restart":          lambda o, d, b, m, r: G(o, d, b, m, r, RESTART=False),
    "OPS=(0,) pure pbest": lambda o, d, b, m, r: G(o, d, b, m, r, OPS=(0,)),
    "OPS=(0,1) no levy":   lambda o, d, b, m, r: G(o, d, b, m, r, OPS=(0, 1)),
    "POP_FACTOR=18":       lambda o, d, b, m, r: G(o, d, b, m, r, POP_FACTOR=18),
    # -- references ------------------------------------------------------------
    "jSO":                 lambda o, d, b, m, r: jSO(o, d, b, m, r),
    "BIPOP_CMAES":         lambda o, d, b, m, r: BIPOP_CMAES(o, d, b, m, r),
    "IPOP_CMAES":          lambda o, d, b, m, r: IPOP_CMAES(o, d, b, m, r),
}

#: Measured from the gate. Both groups are required: see the module docstring.
LOSING = {4: "Rosenbrock (rot)", 5: "Rastrigin (rot)", 7: "Lunacek Bi-Rastrigin",
          24: "Composition 4", 25: "Composition 5"}
WINNING = {11: "Hybrid 1", 14: "Hybrid 4", 16: "Hybrid 6",
           19: "Hybrid 9", 10: "Schwefel (rot)"}


def task(cfg, fid, dim, run, max_fes):
    problem = make_suite_problem("cec2017", fid, dim)
    tracker = Tracker(problem, max_fes, 10)
    rng = np.random.default_rng([SEED_BASE, dim, run, algorithm_seed(cfg)])
    t0 = time.perf_counter()
    CONFIGS[cfg](tracker, dim, (problem.lb, problem.ub), max_fes, rng)
    return {"Config": cfg, "Function": f"F{fid}", "Run": run,
            "Error": tracker.finalize()[1], "Seconds": time.perf_counter() - t0,
            "FES": tracker.fes, "Overrun": tracker.overrun}


def internals(fid, dim, max_fes, runs, **kw):
    """Read the algorithm's own instrumentation on one function.

    Answers, with numbers rather than argument: how often is the eigen gate
    open, where does adaptation drive P_EIG, and how much of the run does the
    diversity guard spend overriding the operator distribution.
    """
    rows = []
    for run in range(runs):
        log = []
        problem = make_suite_problem("cec2017", fid, dim)
        tracker = Tracker(problem, max_fes, 10)
        G(tracker, dim, (problem.lb, problem.ub), max_fes,
          np.random.default_rng([SEED_BASE, dim, run, algorithm_seed(BASELINE)]),
          logger=log, **kw)
        if not log:
            continue
        gen = pd.DataFrame([{k: v for k, v in e.items()
                             if k in ("t", "P_EIG", "eig_ok", "guard", "diversity",
                                      "pop_size", "n_eig_used", "improved")} for e in log])
        rows.append({
            "generations": len(gen),
            "eig gate open %": 100 * gen["eig_ok"].mean(),
            "guard active %": 100 * gen["guard"].mean(),
            "P_EIG mean": gen["P_EIG"].mean(),
            "P_EIG final": gen["P_EIG"].iloc[-1],
            "P_EIG min": gen["P_EIG"].min(),
            "diversity final": gen["diversity"].iloc[-1],
            "improved/gen": gen["improved"].mean(),
        })
    return pd.DataFrame(rows).mean().to_frame().T if rows else pd.DataFrame()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--fes-per-dim", type=int, default=10_000)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", default="results_diagnose")
    ap.add_argument("--quick", action="store_true",
                    help="losing functions only -- a smoke test, not a diagnosis")
    args = ap.parse_args(argv)

    dim, runs = args.dim, args.runs
    max_fes = args.fes_per_dim * dim
    funcs = dict(LOSING) if args.quick else {**LOSING, **WINNING}
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    if args.quick:
        print("[!] --quick measures only the functions we lose. A component that")
        print("[!] recovers F4 and destroys the hybrid class would look like a fix.")

    total = len(CONFIGS) * len(funcs) * runs
    print(f"[*] {len(CONFIGS)} configurations x {len(funcs)} functions x {runs} runs "
          f"= {total} runs at {max_fes:,} FES, D={dim}")

    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=1)(
        delayed(task)(c, f, dim, r, max_fes)
        for c in CONFIGS for f in funcs for r in range(runs))
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw_diagnose.csv", index=False)
    print(f"[*] {len(df)} runs in {(time.perf_counter()-t0)/60:.1f} min")

    bad = df[df["Overrun"] != 0]
    if len(bad):
        print(f"[X] {len(bad)} runs exceeded their budget -- the table below is not usable")
        print(bad.head().to_string())
        return 1

    order = [f"F{f}" for f in funcs]
    piv = df.pivot_table(index="Config", columns="Function",
                         values="Error", aggfunc="median")[order]
    piv = piv.reindex([c for c in CONFIGS if c in piv.index])

    print("\n" + "=" * 100)
    print(f"MEDIAN ERROR  ({runs} runs, D={dim}, {max_fes:,} FES)")
    print(f"  losing  : {'  '.join(f'F{k}={v}' for k, v in LOSING.items())}")
    if not args.quick:
        print(f"  winning : {'  '.join(f'F{k}={v}' for k, v in WINNING.items())}")
    print("=" * 100)
    print(piv.to_string(float_format=lambda x: f"{x:.4e}"))
    piv.to_csv(out / "tables" / "median_by_config.csv")

    # -- each configuration against the baseline, on both groups ------------------
    print("\n" + "=" * 100)
    print(f"EFFECT OF EACH FLAG vs '{BASELINE}'  (paired Wilcoxon, Vargha-Delaney A12)")
    print("  A12 < 0.5 means the configuration is BETTER than the baseline")
    print("=" * 100)
    lines = []
    for cfg in CONFIGS:
        if cfg == BASELINE or cfg not in set(df["Config"]):
            continue
        for fid in funcs:
            f = f"F{fid}"

            def errors(which, _f=f):
                sel = df[(df.Config == which) & (df.Function == _f)]
                return sel.sort_values("Run")["Error"].to_numpy()

            a, b = errors(cfg), errors(BASELINE)
            if len(a) != len(b) or len(a) == 0:
                continue
            lines.append({"Config": cfg, "Function": f,
                          "group": "losing" if fid in LOSING else "winning",
                          "median_cfg": float(np.median(a)),
                          "median_base": float(np.median(b)),
                          "A12": vargha_delaney_a12(a, b),
                          "p_wilcoxon": paired_wilcoxon(a, b)})
    eff = pd.DataFrame(lines)
    eff.to_csv(out / "tables" / "effect_vs_baseline.csv", index=False)

    sig = eff[(eff.p_wilcoxon < 0.05)]
    better = sig[sig.A12 < 0.5]
    worse = sig[sig.A12 > 0.5]
    print("\n-- flags that HELP (significantly better than the baseline)")
    print(better.sort_values(["group", "A12"]).to_string(index=False) if len(better) else "  none")
    print("\n-- flags that HURT (significantly worse than the baseline)")
    print(worse.sort_values(["group", "A12"], ascending=[True, False]).to_string(index=False)
          if len(worse) else "  none")

    print("\n-- net effect per flag: how many functions it helps vs hurts, by group")
    net = (sig.assign(dir=np.where(sig.A12 < 0.5, "helps", "hurts"))
              .groupby(["Config", "group", "dir"]).size().unstack(fill_value=0))
    print(net.to_string() if len(net) else "  nothing reached significance")

    # -- the algorithm's own instrumentation -----------------------------------
    print("\n" + "=" * 100)
    print("INTERNALS on the two functions that matter (baseline configuration)")
    print("=" * 100)
    ins = []
    for fid in ([4, 5] + ([11] if not args.quick else [])):
        row = internals(fid, dim, max_fes, min(runs, 5))
        if len(row):
            row.insert(0, "Function", f"F{fid}")
            ins.append(row)
    if ins:
        tab = pd.concat(ins, ignore_index=True)
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        tab.to_csv(out / "tables" / "internals.csv", index=False)

    (out / "manifest.json").write_text(json.dumps({
        "dim": dim, "runs": runs, "max_fes": max_fes,
        "configs": list(CONFIGS), "losing": LOSING, "winning": WINNING,
        "seed_scheme": "default_rng([SEED_BASE, dim, run, algorithm_seed(config)])",
        "note": "diagnostic, not the competition protocol; 51 runs are needed "
                "before any of this is quoted as a result",
    }, indent=2), encoding="utf-8")
    print(f"\n[ok] tables -> {out}/tables")
    print("[!] This is a diagnostic at "
          f"{runs} runs. Nothing here is a result until it is re-measured at 51.")
    return 0


if __name__ == "__main__":       # required on Windows (loky spawns processes)
    sys.exit(main())
