"""The noise mechanism: does the maths hold, and does the code implement it?

Every tolerance here was fixed by simulation before the test was written, not
tuned until it passed. Where a default could be weakened without anything
obviously breaking, a test pins it -- ``m=3, alpha=0.3`` fails
``test_smoothed_estimate_is_accurate_enough`` (measured P(error>20%) = 0.234
against a 0.05 bar), which is why the defaults are ``m=5, alpha=0.15``.

Run with ``pytest tests/test_noise.py`` or directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.algorithms.temoa_v12 import TEMOA_V12                 # noqa: E402
from temoa.noise import (NOISE_LEVELS, NoiseEstimator,           # noqa: E402
                         NoisyProblem, credit_contamination,
                         false_acceptance_rate, reference_scale)
from temoa.suites import make_suite_problem                      # noqa: E402
from temoa.tracker import Tracker                                # noqa: E402

SIGMA = 0.37


# ----------------------------------------------------------------- the maths
def test_observed_gap_has_twice_the_noise_variance():
    """delta~ = delta + (eps_x - eps_u), so Var[eta] = 2 sigma^2."""
    rng = np.random.default_rng(0)
    eta = rng.normal(0, SIGMA, 2_000_000) - rng.normal(0, SIGMA, 2_000_000)
    assert abs(eta.std() - SIGMA * np.sqrt(2)) / (SIGMA * np.sqrt(2)) < 0.01


def test_false_acceptance_is_one_half_at_zero_true_gap():
    rng = np.random.default_rng(1)
    eta = rng.normal(0, SIGMA * np.sqrt(2), 2_000_000)
    assert abs((eta > 0).mean() - 0.5) < 0.01


def test_expected_observed_gap_of_a_worthless_trial():
    """E[delta~ | accepted, delta=0] = E[eta | eta>0] = 2 sigma / sqrt(pi).

    This is the bias the mechanism exists to remove: a trial that improves
    nothing still contributes ~1.13 sigma of weight to the memory update.
    """
    rng = np.random.default_rng(2)
    eta = rng.normal(0, SIGMA * np.sqrt(2), 4_000_000)
    empirical = eta[eta > 0].mean()
    theoretical = 2 * SIGMA / np.sqrt(np.pi)
    assert abs(empirical - theoretical) / theoretical < 0.01, (empirical, theoretical)


def test_credit_threshold_bounds_the_false_inclusion_rate():
    """N1 keeps a zero-gap trial out of the credit set with probability Phi(tau)."""
    rng = np.random.default_rng(3)
    eta = rng.normal(0, SIGMA * np.sqrt(2), 4_000_000)
    for tau in (0.5, 1.0, 1.5, 2.0):
        empirical = (eta > tau * SIGMA * np.sqrt(2)).mean()
        assert abs(empirical - (1 - norm.cdf(tau))) < 0.005, (tau, empirical)


# ------------------------------------------------------------ the estimator
def test_variance_estimate_is_unbiased():
    """sigma^2_gen = (1/2m) sum (f~1 - f~2)^2 is unbiased for sigma^2."""
    rng = np.random.default_rng(4)
    m, reps = 5, 200_000
    d = rng.normal(0, SIGMA, (reps, m)) - rng.normal(0, SIGMA, (reps, m))
    est = (d**2).sum(axis=1) / (2 * m)
    assert abs(est.mean() - SIGMA**2) / SIGMA**2 < 0.01


def _smoothed_errors(m, alpha, gens=300, reps=1500, burn=150, seed=5):
    rng = np.random.default_rng(seed)
    var = np.full(reps, SIGMA**2)
    kept = []
    for g in range(gens):
        d = rng.normal(0, SIGMA, (reps, m)) - rng.normal(0, SIGMA, (reps, m))
        var = (1 - alpha) * var + alpha * (d**2).sum(axis=1) / (2 * m)
        if g >= burn:
            kept.append(var.copy())
    sig = np.sqrt(np.concatenate(kept))
    return np.abs(sig - SIGMA) / SIGMA


def test_smoothed_estimate_is_accurate_enough():
    """The default (m=5, alpha=0.15) must keep P(relative error > 20%) below 5%."""
    err = _smoothed_errors(5, 0.15)
    p_bad = float(np.mean(err > 0.2))
    assert p_bad < 0.05, f"m=5, alpha=0.15 gave P(error>20%) = {p_bad:.3f}"


def test_the_weaker_setting_is_genuinely_rejected():
    """Guards the defaults: m=3, alpha=0.3 is too loose (measured 0.234)."""
    p_bad = float(np.mean(_smoothed_errors(3, 0.30) > 0.2))
    assert p_bad > 0.10, ("m=3, alpha=0.3 no longer looks bad; re-check the "
                          f"defaults (got {p_bad:.3f})")


def test_estimator_matches_a_known_sigma_end_to_end():
    est = NoiseEstimator(m=5, alpha=0.15)
    rng = np.random.default_rng(6)
    for _ in range(200):
        est.update_from_pairs(rng.normal(0, SIGMA, 5) - rng.normal(0, SIGMA, 5))
    assert abs(est.sigma - SIGMA) / SIGMA < 0.25, est.sigma
    assert abs(est.gap_sigma - est.sigma * np.sqrt(2)) < 1e-12


def test_estimator_is_zero_before_any_update():
    est = NoiseEstimator()
    assert est.sigma == 0.0 and est.gap_sigma == 0.0


# --------------------------------------------------------- the noisy problem
def test_noisy_problem_keeps_a_noise_free_true_objective():
    base = make_suite_problem("cec2017", 1, 10)
    p = NoisyProblem(base, 1e-2, np.random.default_rng(7), dim=10)
    x = np.asarray(base.o, dtype=float)
    assert p.true_obj(x) == base.true_obj(x)
    assert p.kind == "noisy"
    assert p.f_star == base.f_star and (p.lb, p.ub) == (base.lb, base.ub)


def test_noise_level_scales_with_the_problem():
    base = make_suite_problem("cec2017", 1, 10)
    s = reference_scale(base, 10)
    assert s > 0 and np.isfinite(s)
    for rel in (1e-3, 1e-2):
        p = NoisyProblem(base, rel, np.random.default_rng(8), dim=10)
        assert abs(p.sigma - rel * s) < 1e-9


def test_zero_noise_level_is_exactly_deterministic():
    base = make_suite_problem("cec2017", 4, 10)
    p = NoisyProblem(base, 0.0, np.random.default_rng(9), dim=10)
    x = np.asarray(base.o, dtype=float) + 0.1
    assert p(x) == p(x) == base.true_obj(x)


def test_observed_noise_matches_the_requested_level():
    base = make_suite_problem("cec2017", 1, 10)
    p = NoisyProblem(base, 1e-2, np.random.default_rng(10), dim=10)
    x = np.asarray(base.o, dtype=float)
    vals = np.array([p(x) for _ in range(20000)])
    assert abs(vals.std() - p.sigma) / p.sigma < 0.05


# ------------------------------------------------------------- diagnostics
def test_contamination_counters():
    true_gaps = np.array([1.0, -1.0, 0.5, -0.2, 0.0])
    accepted = np.array([True, True, True, False, True])
    # accepted with true gap <= 0: indices 1 and 4 -> 2 of 4
    assert abs(false_acceptance_rate(true_gaps, accepted) - 0.5) < 1e-12
    credit = np.array([True, False, True, False, False])
    assert credit_contamination(true_gaps, credit) == 0.0


# ------------------------------------------------------------------- V12
def test_v12_respects_the_budget_including_extra_evaluations():
    """N0 and N3 buy evaluations; they must be charged like any other."""
    budget = 8000
    for kw in ({}, {"REEVAL_R": 5, "REEVAL_Q": 3}, {"NOISE_M": 0}, {"RESTART": False}):
        base = make_suite_problem("cec2017", 4, 10)
        p = NoisyProblem(base, 1e-2, np.random.default_rng(11), dim=10)
        tr = Tracker(p, budget, 10)
        TEMOA_V12(tr, 10, (p.lb, p.ub), budget, np.random.default_rng(12), **kw)
        assert tr.overrun == 0, f"{kw}: overran by {tr.overrun}"
        assert tr.fes <= budget, f"{kw}: used {tr.fes} of {budget}"
        assert np.isfinite(tr.finalize()[1])


def test_v12_spends_almost_all_of_its_budget():
    budget = 8000
    base = make_suite_problem("cec2017", 4, 10)
    p = NoisyProblem(base, 1e-2, np.random.default_rng(13), dim=10)
    tr = Tracker(p, budget, 10)
    TEMOA_V12(tr, 10, (p.lb, p.ub), budget, np.random.default_rng(14))
    assert tr.fes / budget > 0.98, f"used only {100 * tr.fes / budget:.1f}%"


def test_v12_is_reproducible_from_its_seed():
    def run():
        base = make_suite_problem("cec2017", 4, 10)
        p = NoisyProblem(base, 1e-2, np.random.default_rng(15), dim=10)
        tr = Tracker(p, 5000, 10)
        TEMOA_V12(tr, 10, (p.lb, p.ub), 5000, np.random.default_rng(16))
        return tr.finalize()[1]
    assert run() == run()


def test_credit_filter_actually_removes_trials():
    """With noise present, N1 must shrink the credit set below the accepted set;
    without it, the two coincide. A no-op filter would silently pass every other
    test in this file."""
    base = make_suite_problem("cec2017", 4, 10)
    out = {}
    for tau in (0.0, 1.0):
        log = []
        p = NoisyProblem(base, 1e-2, np.random.default_rng(17), dim=10)
        tr = Tracker(p, 6000, 10)
        TEMOA_V12(tr, 10, (p.lb, p.ub), 6000, np.random.default_rng(18),
                  CREDIT_TAU=tau, logger=log)
        out[tau] = sum(int(e["credit"].sum()) for e in log)
    assert out[1.0] < out[0.0], f"credit filter is a no-op: {out}"


def test_noise_levels_are_ordered_and_start_at_zero():
    assert NOISE_LEVELS[0] == 0.0
    assert list(NOISE_LEVELS) == sorted(NOISE_LEVELS)


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
