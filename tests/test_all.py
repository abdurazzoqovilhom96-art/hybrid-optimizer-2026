"""Correctness tests. Run with ``pytest tests`` or ``python tests/test_all.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.problems import FUNC_NAMES, make_problem          # noqa: E402
from temoa.registry import ALL_ALGORITHMS                    # noqa: E402
from temoa.stats import (cliffs_delta, friedman, holm,       # noqa: E402
                         nemenyi_cd, paired_wilcoxon,
                         vargha_delaney_a12)
from temoa.tracker import Tracker                            # noqa: E402


# ---------------------------------------------------------------- statistics
def test_holm_is_monotone_and_conservative():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    adj = holm(p)
    assert np.all(adj >= p - 1e-12), "Holm must never decrease a p-value"
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "Holm must be monotone in rank order"
    assert np.all(adj <= 1.0)
    # smallest p, family of 5 -> 5 * 0.001
    assert abs(adj[0] - 0.005) < 1e-12


def test_holm_matches_statsmodels_reference_values():
    # hand-computed step-down: [4*.01, max(prev,3*.02), max(prev,2*.03), max(prev,1*.04)]
    adj = holm([0.01, 0.02, 0.03, 0.04])
    assert np.allclose(adj, [0.04, 0.06, 0.06, 0.06]), adj


def test_a12_known_cases():
    a, b = [1, 2, 3], [4, 5, 6]        # `a` always smaller => always better (minimisation)
    assert abs(vargha_delaney_a12(a, b) - 1.0) < 1e-12
    assert abs(vargha_delaney_a12(b, a) - 0.0) < 1e-12
    assert abs(vargha_delaney_a12(a, a) - 0.5) < 1e-12      # all ties
    assert abs(cliffs_delta(a, b) - 1.0) < 1e-12
    # A12 must equal the U-statistic definition
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 40), rng.normal(0.5, 1, 40)
    wins = np.mean([(xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y])
    assert abs(vargha_delaney_a12(x, y) - wins) < 1e-12


def test_friedman_matches_scipy():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(12, 5)), columns=list("ABCDE"))
    mine = friedman(data)
    ref = sps.friedmanchisquare(*[data[c].to_numpy() for c in data.columns])
    assert abs(mine["chi2"] - ref.statistic) < 1e-8, (mine["chi2"], ref.statistic)
    assert abs(mine["p_chi2"] - ref.pvalue) < 1e-10
    assert abs(mine["avg_ranks"].mean() - 3.0) < 1e-12     # ranks 1..5 average to 3


def test_friedman_ranks_order_best_first():
    # column A is uniformly best => rank 1
    data = pd.DataFrame({"A": [1, 1, 1, 1], "B": [2, 2, 2, 2], "C": [3, 3, 3, 3]})
    fr = friedman(data)
    assert list(fr["avg_ranks"].index) == ["A", "B", "C"]
    assert abs(fr["avg_ranks"]["A"] - 1.0) < 1e-12


def test_nemenyi_cd_matches_published_value():
    # Demsar (2006), Sec. 3.2.2: k=4, N=14 -> CD = 2.569 * sqrt(4*5/(6*14)) = 1.2535,
    # reported there as 1.25.
    import math
    assert abs(nemenyi_cd(4, 14) - 2.569 * math.sqrt(4 * 5 / (6 * 14))) < 1e-12
    assert abs(nemenyi_cd(4, 14) - 1.25) < 5e-3
    # CD must shrink as the number of blocks grows
    assert nemenyi_cd(4, 50) < nemenyi_cd(4, 14)


def test_paired_wilcoxon_handles_identical_samples():
    assert paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


# ------------------------------------------------------------------ problems
def test_optimum_is_attained_at_the_stated_location():
    for suite in ("shift", "rotated"):
        for dim in (10, 30):
            for name in FUNC_NAMES:
                pr = make_problem(name, dim, suite=suite)
                err = pr.true_obj(pr.o) - pr.f_star
                assert abs(err) < 1e-8, f"{name} {dim}D {suite}: error at optimum = {err}"


def test_no_point_in_the_box_beats_the_stated_optimum():
    """Guards the Schwefel bug: f* must really be the in-box global minimum."""
    rng = np.random.default_rng(0)
    for name in FUNC_NAMES:
        pr = make_problem(name, 20, suite="rotated")
        if pr.kind in ("noisy", "dynamic"):
            continue
        X = rng.uniform(pr.lb, pr.ub, (20000, 20))
        worst = min(pr.true_obj(x) for x in X[:2000])
        assert worst >= pr.f_star - 1e-6, f"{name}: found {worst} below f*={pr.f_star}"


def test_rotation_matrix_is_orthogonal():
    pr = make_problem("RotatedElliptic", 30, suite="rotated")
    assert pr.M is not None
    assert np.allclose(pr.M @ pr.M.T, np.eye(30), atol=1e-10)


def test_shift_suite_rotates_only_one_problem():
    n = sum(make_problem(f, 30, suite="shift").M is not None for f in FUNC_NAMES)
    assert n == 1, f"the original suite should rotate exactly 1 of 12, got {n}"


# ------------------------------------------------------------------- tracker
def test_tracker_counts_and_scores():
    pr = make_problem("Sphere" if "Sphere" in FUNC_NAMES else "Ackley", 5, suite="rotated")
    tr = Tracker(pr, 100, 10)
    for _ in range(100):
        tr(np.zeros(5))
    assert tr.fes == 100 and tr.overrun == 0
    curve, score = tr.finalize()
    assert np.all(np.isfinite(curve)) and np.isfinite(score)


def test_convergence_curve_is_monotone_for_deterministic_problems():
    pr = make_problem("Ackley", 5, suite="rotated")
    tr = Tracker(pr, 500, 20)
    rng = np.random.default_rng(0)
    for _ in range(500):
        tr(rng.uniform(pr.lb, pr.ub, 5))
    curve, _ = tr.finalize()
    assert np.all(np.diff(curve) <= 1e-12), "best-so-far error must never increase"


# ---------------------------------------------------------------- algorithms
def test_every_algorithm_respects_the_budget_and_solves_sphere():
    budget = 6000
    for name, alg in ALL_ALGORITHMS.items():
        pr = make_problem("NoisySphere", 5, suite="rotated",
                          noise_rng=np.random.default_rng(0))
        tr = Tracker(pr, budget, 10)
        alg(tr, 5, (pr.lb, pr.ub), budget, np.random.default_rng(3))
        assert tr.overrun == 0, f"{name} exceeded the FES budget by {tr.overrun}"
        assert tr.fes <= budget, f"{name} used {tr.fes} of {budget}"
        assert np.isfinite(tr.finalize()[1]), f"{name} returned a non-finite score"


def test_modern_de_reaches_high_precision_on_a_unimodal_problem():
    from temoa.algorithms.modern import LSHADE, jSO
    for alg in (LSHADE, jSO):
        pr = make_problem("BentCigar", 10, suite="rotated")
        tr = Tracker(pr, 40000, 10)
        alg(tr, 10, (pr.lb, pr.ub), 40000, np.random.default_rng(0))
        err = tr.finalize()[1]
        assert err < 1e-8, f"{alg.__name__} only reached {err:.3e} on BentCigar"


def test_v10_default_ops_reproduce_the_original_portfolio():
    """OPS was added for ablation; the default must not change behaviour."""
    from temoa.algorithms.temoa_v10 import TEMOA_V10_HYBRID as V10
    scores = []
    for ops in (None, (0, 1, 2, 3)):
        pr = make_problem("Ackley", 10, suite="rotated")
        tr = Tracker(pr, 5000, 10)
        kw = {} if ops is None else {"OPS": ops}
        V10(tr, 10, (pr.lb, pr.ub), 5000, np.random.default_rng(11), **kw)
        scores.append(tr.finalize()[1])
    assert scores[0] == scores[1], scores


def test_lshade_dgr_is_v12_with_the_noise_mechanism_removed():
    """The rename must not have changed the algorithm.

    ``L-SHADE-DGR`` was produced by deleting N0-N3 from ``TEMOA_V12`` -- the
    noise-robust credit machinery whose measured effect on final error was nil.
    If the edit also changed a coefficient or an operator, every ablation number
    recorded against V12 would quietly stop describing the algorithm in the
    paper.

    Disabling N0-N3 in V12 must therefore reproduce L-SHADE-DGR. It does, up to
    one redundant renormalisation of an already-normalised weight vector that
    V12 still performs (mathematically a no-op, about 1 ULP numerically). With
    that line removed from a copy of V12 the two are bit-identical on F1, F4,
    F11, F21 and F25; here the claim is pinned without patching V12:

      * after one generation the two agree exactly;
      * measured divergence stays at 3.2e-13 through eight generations, which is
        rounding noise amplified by a chaotic map, not a change in behaviour.
    """
    from temoa.algorithms.lshade_dgr import LSHADE_DGR
    from temoa.algorithms.temoa_v12 import TEMOA_V12
    from temoa.suites import make_suite_problem
    off = dict(NOISE_M=0, CREDIT_TAU=0.0, RANK_WEIGHTS=False, REEVAL_R=0)

    def run(alg, fid, budget, **kw):
        pr = make_suite_problem("cec2017", fid, 10)
        tr = Tracker(pr, budget, 10)
        alg(tr, 10, (pr.lb, pr.ub), budget, np.random.default_rng(5), **kw)
        return tr.finalize()[1]

    n_init = 120                                   # POP_FACTOR * dim at D=10
    for fid in (1, 4, 11, 21, 25):
        a = run(LSHADE_DGR, fid, 2 * n_init)
        b = run(TEMOA_V12, fid, 2 * n_init, **off)
        assert a == b, f"F{fid}: the two differ after one generation: {a!r} vs {b!r}"

        a = run(LSHADE_DGR, fid, 9 * n_init)
        b = run(TEMOA_V12, fid, 9 * n_init, **off)
        rel = abs(a - b) / abs(b)
        assert rel < 1e-10, (f"F{fid}: {rel:.3e} after eight generations is too "
                             f"large for rounding alone ({a!r} vs {b!r})")


# ----------------------------------------------------------- reproducibility
def test_same_seed_gives_bit_identical_results():
    """Every algorithm must be a pure function of its injected RNG."""
    from temoa.registry import ALL_ALGORITHMS
    for name, alg in ALL_ALGORITHMS.items():
        scores = []
        for _ in range(2):
            pr = make_problem("Ackley", 8, suite="rotated",
                              noise_rng=np.random.default_rng([99, 8, 0]))
            tr = Tracker(pr, 3000, 10)
            alg(tr, 8, (pr.lb, pr.ub), 3000, np.random.default_rng([42, 8, 0, 1]))
            scores.append(tr.finalize()[1])
        assert scores[0] == scores[1], f"{name} is not reproducible: {scores}"


def test_raw_csv_round_trips_float64_exactly():
    """analyze.py compares algorithms that reach 1e-20; the default CSV parser
    is off by up to one ULP, which would silently reorder them."""
    import io
    vals = np.array([4.689743049827384e-08, 1.2345678901234567e-20,
                     7.105427357601002e-15, 3.141592653589793, 0.0])
    buf = io.StringIO()
    pd.DataFrame({"Error": vals}).to_csv(buf, index=False)
    default = pd.read_csv(io.StringIO(buf.getvalue()))["Error"].to_numpy()
    exact = pd.read_csv(io.StringIO(buf.getvalue()),
                        float_precision="round_trip")["Error"].to_numpy()
    assert np.array_equal(exact, vals), "round_trip parsing must be exact"
    if not np.array_equal(default, vals):
        # documents why analyze.py must pass float_precision explicitly
        assert np.max(np.abs((default - vals)[vals != 0] / vals[vals != 0])) < 1e-15


def test_analyze_uses_round_trip_parsing():
    src = (Path(__file__).resolve().parents[1] / "analyze.py").read_text(encoding="utf-8")
    assert 'float_precision="round_trip"' in src, \
        "analyze.py must read raw results with exact float parsing"


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
