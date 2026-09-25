"""Parameter sensitivity.

The original study reports one parameter set and no sensitivity analysis. The
population factor turns out to matter enormously: at 30D, raising V10's
population from 6*D to 18*D changes RotatedElliptic by a factor of 6600 while
costing a factor of 21 on Rosenbrock.

    python experiments/sensitivity.py --dim 30 --runs 15 --jobs 20
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.algorithms.temoa_v10 import TEMOA_V10_HYBRID as V10   # noqa: E402
from temoa.algorithms.temoa_v11 import TEMOA_V11 as V11          # noqa: E402
from temoa.problems import make_problem                      # noqa: E402
from temoa.tracker import Tracker                            # noqa: E402

SWEEPS = {
    "V10.POP_FACTOR":  (V10, "POP_FACTOR", [3, 6, 12, 18, 25]),
    "V10.P_MIN":       (V10, "P_MIN",      [0.0, 0.02, 0.05, 0.10, 0.20]),
    "V10.LS_FRACTION": (V10, "LS_FRACTION", [0.0, 0.05, 0.10, 0.20]),
    "V11.POP_FACTOR":  (V11, "POP_FACTOR", [6, 12, 18, 25]),
    "V11.DIV_THRESH":  (V11, "DIV_THRESH", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
    "V11.DIV_BOOST":   (V11, "DIV_BOOST",  [0.3, 0.5, 0.7, 0.9]),
}


def task(sweep, value, func, dim, run, max_fes, suite):
    alg, param, _ = SWEEPS[sweep]
    pr = make_problem(func, dim, suite=suite, noise_rng=np.random.default_rng([99, dim, run]))
    tr = Tracker(pr, max_fes, 10)
    alg(tr, dim, (pr.lb, pr.ub), max_fes, np.random.default_rng([42, dim, run]), **{param: value})
    return {"Sweep": sweep, "Value": value, "Function": func, "Run": run,
            "Error": tr.finalize()[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--suite", default="rotated")
    ap.add_argument("--fes-per-dim", type=int, default=3000)
    ap.add_argument("--out", default="results/sensitivity")
    ap.add_argument("--functions", nargs="+",
                    default=["Schwefel", "NoisyRastrigin", "RotatedElliptic", "Rosenbrock"])
    a = ap.parse_args()

    max_fes = a.fes_per_dim * a.dim
    jobs = [(s, v, f, r) for s, (_, _, vals) in SWEEPS.items()
            for v in vals for f in a.functions for r in range(a.runs)]
    rows = Parallel(n_jobs=a.jobs, backend="loky")(
        delayed(task)(s, v, f, a.dim, r, max_fes, a.suite) for s, v, f, r in jobs)

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"sensitivity_raw_{a.dim}D.csv", index=False)

    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.3e}")
    for s in SWEEPS:
        piv = (df[df["Sweep"] == s]
               .pivot_table(index="Value", columns="Function", values="Error", aggfunc="median"))
        print(f"\n=== {s} ({a.dim}D, {a.runs} runs, median error) ===")
        print(piv.to_string())
        piv.to_csv(out / f"sensitivity_{s.replace('.', '_')}_{a.dim}D.csv")
    print(f"\n[*] -> {out}")


if __name__ == "__main__":
    main()
