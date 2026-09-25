"""Are the competitors real?

A comparison is only honest if the competitors actually work. A wrapper that
silently mis-drives CMA-ES, or an algorithm that quietly spends more evaluations
than its rivals, produces a table that favours us for the wrong reason -- the
exact fault this project was started to correct.

These checks are behavioural, not cosmetic: each one would fail if the wrapper
were driving a different algorithm than its name claims.

Run with ``pytest tests/test_competitor_validation.py`` or directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.registry import ALL_ALGORITHMS                # noqa: E402
from temoa.suites import make_suite_problem             # noqa: E402
from temoa.tracker import Tracker                       # noqa: E402

DIM = 10
BUDGET = 30_000


def _run(alg, fid, budget=BUDGET, seed=3, dim=DIM):
    p = make_suite_problem("cec2017", fid, dim)
    tr = Tracker(p, budget, 10)
    alg(tr, dim, (p.lb, p.ub), budget, np.random.default_rng(seed))
    return tr


def test_no_algorithm_exceeds_its_evaluation_budget():
    """cma.fmin2 checks maxfevals between generations and overruns by ~11 calls;
    the budget guard in modern.py must absorb that."""
    for name, alg in ALL_ALGORITHMS.items():
        tr = _run(alg, 1)
        assert tr.overrun == 0, f"{name} overran the budget by {tr.overrun} evaluations"
        assert tr.fes <= BUDGET, f"{name} used {tr.fes} of {BUDGET}"


def test_every_algorithm_returns_a_finite_score():
    for name, alg in ALL_ALGORITHMS.items():
        score = _run(alg, 1).finalize()[1]
        assert np.isfinite(score), f"{name} returned {score}"


def test_cma_family_solves_the_rotated_ill_conditioned_function():
    """CEC'2017 F1 is the shifted and rotated Bent Cigar, condition number 1e6.

    A correctly driven CMA-ES is rotation-invariant and must solve it at 10D.
    If a wrapper fails here it is not running CMA-ES.
    """
    for name in ("CMAES", "IPOP_CMAES", "BIPOP_CMAES"):
        err = _run(ALL_ALGORITHMS[name], 1).finalize()[1]
        assert err < 1e-6, f"{name} only reached {err:.3e} on rotated Bent Cigar"


def test_sep_cma_es_really_is_the_separable_variant():
    """The diagonal variant cannot follow a rotation, so it must FAIL where the
    full-covariance ones succeed. If it passed, ``CMA_diagonal`` is not in effect
    and sepCMAES would be a mislabelled duplicate of CMAES."""
    full = _run(ALL_ALGORITHMS["CMAES"], 1).finalize()[1]
    sep = _run(ALL_ALGORITHMS["sepCMAES"], 1).finalize()[1]
    assert full < 1e-6, f"the full-covariance control failed first ({full:.3e})"
    assert sep > full, (f"sepCMAES ({sep:.3e}) did no worse than full CMA-ES "
                        f"({full:.3e}) on a rotated problem -- CMA_diagonal is not active")


def test_adaptive_de_reaches_high_precision_on_a_unimodal_function():
    for name in ("jSO", "LSHADE"):
        err = _run(ALL_ALGORITHMS[name], 1).finalize()[1]
        assert err < 1e-4, f"{name} only reached {err:.3e} on rotated Bent Cigar"


def test_every_competitor_is_reproducible_from_its_seed():
    for name, alg in ALL_ALGORITHMS.items():
        a = _run(alg, 4, budget=6000, seed=11).finalize()[1]
        b = _run(alg, 4, budget=6000, seed=11).finalize()[1]
        assert a == b, f"{name} is not reproducible: {a} vs {b}"


def test_cma_wrappers_are_three_distinct_algorithms():
    """CMAES and IPOP differ only in incpopsize (1 vs 2, and cma defaults to 2);
    BIPOP differs from IPOP only in the bipop flag. A missing override would make
    two of them the same algorithm under different names.

    F25 is a composition function, where the restart policy actually decides the
    outcome, and the budget has to be large enough for BIPOP's second regime to
    engage -- measured: identical to IPOP below ~20000 evaluations, distinct from
    50000 upward. That is correct BIPOP behaviour, not a wiring fault.
    """
    scores = {n: _run(ALL_ALGORITHMS[n], 25, budget=60_000, seed=1).finalize()[1]
              for n in ("CMAES", "IPOP_CMAES", "BIPOP_CMAES")}
    assert scores["CMAES"] != scores["IPOP_CMAES"], \
        f"CMAES == IPOP_CMAES ({scores['CMAES']:.6e}) -- incpopsize override missing"
    assert scores["BIPOP_CMAES"] != scores["IPOP_CMAES"], \
        f"BIPOP == IPOP ({scores['IPOP_CMAES']:.6e}) -- bipop flag missing"


def test_every_algorithm_spends_its_whole_budget():
    """With a fixed restart count, restart CMA-ES exhausts its restarts and stops
    early -- measured at 27410 of 100000 evaluations, 73% unspent. An algorithm
    that cannot spend its budget is handicapped, and the resulting table would
    favour us for the wrong reason."""
    budget = 40_000
    for name, alg in ALL_ALGORITHMS.items():
        tr = _run(alg, 25, budget=budget, seed=2)
        used = tr.fes / budget
        assert used > 0.98, f"{name} used only {100*used:.1f}% of its evaluation budget"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                    # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
