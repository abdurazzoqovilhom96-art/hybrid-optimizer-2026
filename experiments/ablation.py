"""Component ablation: what does each part of the hybrid actually contribute?

Every configuration is the same code path with a flag flipped, so any difference
is attributable to that component alone.

    python experiments/ablation.py --dim 30 --runs 15 --jobs 20
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

from temoa.algorithms.modern import jSO                      # noqa: E402
from temoa.algorithms.temoa_v10 import TEMOA_V10_HYBRID as V10   # noqa: E402
from temoa.algorithms.temoa_v11 import TEMOA_V11 as V11          # noqa: E402
from temoa.problems import make_problem                      # noqa: E402
from temoa.tracker import Tracker                            # noqa: E402

CONFIGS = {
    # --- V10 operator portfolio -------------------------------------------
    "V10 full (0,1,2,3)":     lambda o, d, b, m, r: V10(o, d, b, m, r),
    "V10 op0 only":           lambda o, d, b, m, r: V10(o, d, b, m, r, OPS=(0,)),
    "V10 op0+op1 leader":     lambda o, d, b, m, r: V10(o, d, b, m, r, OPS=(0, 1)),
    "V10 op0+op2 spiral":     lambda o, d, b, m, r: V10(o, d, b, m, r, OPS=(0, 2)),
    "V10 op0+op3 levy":       lambda o, d, b, m, r: V10(o, d, b, m, r, OPS=(0, 3)),
    "V10 drop op2":           lambda o, d, b, m, r: V10(o, d, b, m, r, OPS=(0, 1, 3)),
    # --- V10 other components ---------------------------------------------
    "V10 no eigen":           lambda o, d, b, m, r: V10(o, d, b, m, r, P_EIG=0.0, ADAPT_EIG=False),
    "V10 no ES tail":         lambda o, d, b, m, r: V10(o, d, b, m, r, LS_FRACTION=0.0),
    "V10 no CR floor":        lambda o, d, b, m, r: V10(o, d, b, m, r, CR_FLOOR=False),
    # --- V11 and its own ablations ----------------------------------------
    "V11 full":               lambda o, d, b, m, r: V11(o, d, b, m, r),
    "V11 no diversity guard": lambda o, d, b, m, r: V11(o, d, b, m, r, DIV_GUARD=False),
    "V11 no eigen gate":      lambda o, d, b, m, r: V11(o, d, b, m, r, EIGEN_GATE=False),
    "V11 with op2 back":      lambda o, d, b, m, r: V11(o, d, b, m, r, OPS=(0, 1, 2, 3)),
    "V11 no eigen":           lambda o, d, b, m, r: V11(o, d, b, m, r, P_EIG=0.0, ADAPT_EIG=False),
    # --- reference ---------------------------------------------------------
    "jSO (reference core)":   lambda o, d, b, m, r: jSO(o, d, b, m, r),
}


def task(cfg, func, dim, run, max_fes, suite):
    pr = make_problem(func, dim, suite=suite, noise_rng=np.random.default_rng([99, dim, run]))
    tr = Tracker(pr, max_fes, 20)
    CONFIGS[cfg](tr, dim, (pr.lb, pr.ub), max_fes, np.random.default_rng([42, dim, run]))
    return {"Config": cfg, "Function": func, "Run": run, "Error": tr.finalize()[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--suite", default="rotated")
    ap.add_argument("--fes-per-dim", type=int, default=3000)
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--functions", nargs="+",
                    default=["Schwefel", "NoisyRastrigin", "Ackley", "Griewank",
                             "RotatedElliptic", "Rosenbrock", "BentCigar", "Zakharov"])
    a = ap.parse_args()

    max_fes = a.fes_per_dim * a.dim
    rows = Parallel(n_jobs=a.jobs, backend="loky")(
        delayed(task)(c, f, a.dim, r, max_fes, a.suite)
        for c in CONFIGS for f in a.functions for r in range(a.runs))

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"ablation_raw_{a.dim}D.csv", index=False)

    piv = df.pivot_table(index="Config", columns="Function", values="Error", aggfunc="median")
    piv = piv.reindex(list(CONFIGS))[a.functions]
    piv.to_csv(out / f"ablation_median_{a.dim}D.csv")
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.3e}")
    print(f"\n=== ABLATION, {a.dim}D, {a.runs} runs, median error ===")
    print(piv.to_string())
    print(f"\n[*] -> {out}")


if __name__ == "__main__":
    main()
