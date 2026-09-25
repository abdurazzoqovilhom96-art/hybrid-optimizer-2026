"""Does ``temoa/`` reproduce the original ``hybrid 2026.py``?

Every criticism in ``reports/TAHLIL_UZ.md`` rests on the ported code behaving
like the original. This check proves it on two levels:

1. **The landscapes are bit-identical.** ``--suite shift`` rebuilds exactly the
   problem instances the original constructs (same shift vectors, same sign
   flips, same rotation), so nothing in the comparison comes from a different
   benchmark.
2. **The algorithm is indistinguishable.** The port replaces the global RNG with
   an injected ``Generator``, so the two cannot agree bit-for-bit; they must
   agree in distribution. Measured at 30D over 10 runs, they do (all p > 0.4).

Slow (a few minutes), so it is kept out of ``test_all.py``:

    python tests/test_port_fidelity.py
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from temoa.algorithms.temoa_v10 import TEMOA_V10_HYBRID as PORT   # noqa: E402
from temoa.problems import f_star, make_problem                   # noqa: E402
from temoa.tracker import Tracker                                 # noqa: E402

FUNCTIONS = ["Ackley", "Schwefel", "Rosenbrock", "RotatedElliptic"]
DIM, RUNS = 30, 10
MAX_FES = 3000 * DIM
ALPHA = 0.05


def load_original():
    spec = importlib.util.spec_from_file_location("orig", ROOT / "hybrid 2026.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # safe: its experiment is __main__-guarded
    return mod


def check_landscapes(orig) -> bool:
    print("--- landscape equality: original make_problem vs suite='shift' ---")
    ok = True
    rng = np.random.default_rng(0)
    for f in FUNCTIONS:
        _, o_true, o_lb, o_ub = orig.make_problem(f, DIM)
        pr = make_problem(f, DIM, suite="shift")
        X = rng.uniform(pr.lb, pr.ub, (300, DIM))
        worst = max(abs(o_true(x) - pr.true_obj(x)) / max(1.0, abs(o_true(x))) for x in X)
        good = (o_lb, o_ub) == (pr.lb, pr.ub) and worst == 0.0
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {f:16s} max relative difference = {worst:.3e}")
    return ok


def _run_original(orig, func, run):
    np.random.seed(42 + 1000 * DIM + run)
    obj, true_obj, lb, ub = orig.make_problem(func, DIM)
    tr = orig.Tracker(obj, MAX_FES, 10)
    orig.TEMOA_V10_HYBRID(tr, DIM, (lb, ub), MAX_FES)
    tr.finalize()
    return "original", func, float(true_obj(tr.best_x)) - f_star(func, DIM)


def _run_port(func, run):
    pr = make_problem(func, DIM, suite="shift")
    tr = Tracker(pr, MAX_FES, 10)
    PORT(tr, DIM, (pr.lb, pr.ub), MAX_FES, np.random.default_rng([42, DIM, run]))
    return "port", func, tr.finalize()[1]


def check_algorithm(orig, jobs=-1) -> bool:
    res = Parallel(n_jobs=jobs)(
        [delayed(_run_original)(orig, f, r) for f in FUNCTIONS for r in range(RUNS)]
        + [delayed(_run_port)(f, r) for f in FUNCTIONS for r in range(RUNS)])
    by = {}
    for which, f, v in res:
        by.setdefault((f, which), []).append(v)

    print(f"\n--- TEMOA_V10: original vs port ({RUNS} runs, {DIM}D, shift suite) ---")
    print(f"{'function':18s}{'original':>14s}{'port':>14s}{'ratio':>8s}{'p':>8s}")
    ok = True
    for f in FUNCTIONS:
        a, b = by[(f, "original")], by[(f, "port")]
        ma, mb = float(np.median(a)), float(np.median(b))
        try:
            p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            p = 1.0
        good = p >= ALPHA
        ok &= good
        print(f"{f:18s}{ma:14.4e}{mb:14.4e}{(mb + 1e-300) / (ma + 1e-300):8.2f}{p:8.3f}"
              f"{'' if good else '   <-- DIFFERS'}")
    return ok


def main() -> int:
    orig = load_original()
    ok = check_landscapes(orig)
    ok &= check_algorithm(orig)
    print("\n" + ("PORT FIDELITY CONFIRMED: the port matches the original."
                  if ok else
                  "PORT DIFFERS FROM THE ORIGINAL -- do not trust the comparison."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
