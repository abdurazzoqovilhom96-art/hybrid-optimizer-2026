"""L-SHADE-DGR: the complete study in one file.

L-SHADE-DGR is L-SHADE with a Diversity Guard and Restart. The name declares the
lineage on purpose: this is an L-SHADE/jSO variant, and a reviewer should learn
that from the title rather than discover it. The 2025 surveys that found 26
published metaheuristics structurally identical to earlier ones are a direct
warning against names that hide their ancestry.

WHAT IS INHERITED, AND FROM WHOM. Nothing here is presented as new except where
stated.

    current-to-pbest-w/1 with archive, success-history F/CR   L-SHADE, jSO
    linear population size reduction (LPSR)                   L-SHADE
    eigenbasis crossover                                      LSHADE-cnEpSin,
                                                              EA4eig, L-SRTDE
    Levy-flight operator                                      HHO
    IPOP-style restart                                        Auger & Hansen 2005

The eigen crossover in particular is not ours. EA4eig, the CEC'2022 winner, is
itself an operator portfolio with an eigenbasis crossover, which makes it the
structurally closest rival and the reason "portfolio plus eigen crossover" is
never claimed as a contribution.

WHAT THIS FILE IS. The study used to be a package (``temoa/``) plus three
drivers. Everything that the comparison actually runs on now lives here: the
statistics, the FES accounting, the CEC suites, the algorithm, the six rivals,
the experiment driver, the analysis and the component diagnostic. Deliberately
left out, because the project's own methodology excludes them from every table:

  * the weak swarm baselines (PSO/GWO/WOA-class). Beating a 2014 swarm
    metaheuristic establishes nothing about a modern adaptive DE.
  * our own earlier versions (TEMOA V10-V12). They are ancestors, not rivals.
  * the legacy twelve-function suite and its noise/drift machinery. It matches no
    competition protocol, so its numbers are not comparable with anything
    published.

USAGE

    python lshade_dgr_study.py run                    # the gate, all threads
    python lshade_dgr_study.py run --dims 10          # D=10 only
    python lshade_dgr_study.py run --resume           # continue an interrupted run
    python lshade_dgr_study.py analyze                # tables, statistics, figures
    python lshade_dgr_study.py diagnose --runs 31     # the component diagnostic
    python lshade_dgr_study.py all --dims 10          # run, then analyze

FUNCTION NUMBERING -- READ THIS BEFORE COMPARING WITH ANY PUBLISHED TABLE.
``opfunu`` ships 29 CEC'2017 classes named F1..F29, but the official suite is
F1..F30. opfunu has already dropped the unstable F2 (withdrawn after the
competition) and renumbered, so opfunu Fk = official F(k+1) for k >= 2. This file
exposes **official** ids everywhere -- tables, CSVs and figures say F4 where the
official suite says F4 -- because an off-by-one against published results would
be invisible and would invalidate every comparison. Errors, never raw objective
values, are what gets reported and compared; f(x*) == f* holds exactly for every
function, so f(x) - f* is the correct comparable quantity.

ON PARALLELISM AND REPRODUCIBILITY. Each run's random stream is derived from
``(SEED_BASE, dim, run, algorithm_seed(name))`` alone. It does not depend on the
worker count, on the order results come back, or on which other algorithms are
registered. Changing ``--jobs`` therefore cannot change a single reported number;
it only changes how long the run takes. This is worth stating because the driver
runs every task through one worker pool and collects results out of order.
"""

from __future__ import annotations

# Must precede the NumPy import in every worker: loky spawns fresh processes that
# inherit this environment. Without the pinning, NumPy's internal threads
# oversubscribe the CPU and more workers makes the run slower, not faster.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import functools
import hashlib
import inspect
import json
import math
import platform
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats as sps

SEED_BASE = 42              # algorithm streams
SEED_SCHEME_VERSION = 2     # v1 seeded from the position in sorted(ALGORITHMS)
N_POINTS = 100              # convergence-curve checkpoints
CEC_ERROR_FLOOR = 1e-8      # competition convention: errors below this count as 0
ALPHA = 0.05

RAW_COLUMNS = ["Algorithm", "Function", "Dimension", "Run", "Error", "Seconds",
               "FES", "Overrun"]


# ==========================================================================
# 1. STATISTICS
# ==========================================================================
# What the original study did, and why each item is changed:
#
# *Friedman over pooled dimensions.* The original builds one Friedman test over
# 36 blocks = 12 functions x 3 dimensions. The same function at 30D, 50D and
# 100D is not an independent block -- the three share a landscape, a shift vector
# and a rotation -- so pooling inflates the block count and makes the p-value
# anti-conservative. Here Friedman runs per dimension, and the Iman-Davenport F
# is reported alongside chi-square because Friedman's chi-square is known to be
# conservative for small block counts.
#
# *No post-hoc.* The original reports a Friedman p-value and average ranks but
# never tests which pairs differ. Here: Holm's step-down procedure on the
# normal-approximation pairwise rank statistic (Demsar's recommendation when one
# algorithm is the control), plus the Nemenyi critical difference for CD
# diagrams.
#
# *Unpaired test on paired data.* The original applies Mann-Whitney U. Every
# algorithm sees the identical problem instance with matched seeds, so the runs
# are paired and the Wilcoxon signed-rank test applies and has more power. Both
# are computed; signed-rank is primary.
#
# *Partial multiplicity control.* The original applies Holm within each
# (function, dimension) cell, i.e. many separate small families, with no control
# over the tests actually performed. Here Holm is applied both within the cell
# and over the whole family, and both are reported.
#
# *No effect size.* A p-value says a difference exists, not that it matters.
# Added: Vargha-Delaney A12 and Cliff's delta.

# Studentized range / sqrt(2), alpha = 0.05 and 0.10 (Demsar 2006, Table 5).
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
_Q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693,
        8: 2.780, 9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030}


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values. Monotone, in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def friedman(perf: pd.DataFrame) -> dict:
    """Friedman test on a blocks x algorithms performance matrix (lower = better).

    Returns chi-square, Iman-Davenport F, both p-values, and the average ranks
    (rank 1 = best).
    """
    vals = perf.to_numpy(dtype=float)
    N, k = vals.shape
    ranks = np.apply_along_axis(sps.rankdata, 1, vals)
    R = ranks.mean(axis=0)

    chi2 = 12.0 * N / (k * (k + 1)) * (np.sum(R**2) - k * (k + 1) ** 2 / 4.0)
    p_chi2 = sps.chi2.sf(chi2, k - 1)

    denom = N * (k - 1) - chi2
    if denom > 0:
        F = (N - 1) * chi2 / denom
        p_F = sps.f.sf(F, k - 1, (k - 1) * (N - 1))
    else:                       # chi2 saturated: complete separation
        F, p_F = np.inf, 0.0

    return {"N_blocks": N, "k_algorithms": k, "chi2": float(chi2),
            "p_chi2": float(p_chi2), "iman_davenport_F": float(F), "p_F": float(p_F),
            "avg_ranks": pd.Series(R, index=perf.columns).sort_values()}


def nemenyi_cd(k: int, N: int, alpha: float = ALPHA) -> float:
    """Critical difference for the Nemenyi post-hoc (CD diagrams)."""
    table = _Q05 if alpha == 0.05 else _Q10
    if k not in table:
        raise ValueError(f"no tabulated q for k={k}; tabulated k = 2..12")
    return table[k] * np.sqrt(k * (k + 1) / (6.0 * N))


def friedman_posthoc_holm(avg_ranks: pd.Series, N: int, control: str) -> pd.DataFrame:
    """Holm-adjusted pairwise comparisons of every algorithm against a control."""
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    others = [a for a in avg_ranks.index if a != control]
    z = np.array([(avg_ranks[control] - avg_ranks[a]) / se for a in others])
    p = 2.0 * sps.norm.sf(np.abs(z))
    return pd.DataFrame({
        "comparison": [f"{control} vs {a}" for a in others],
        "rank_diff": [avg_ranks[control] - avg_ranks[a] for a in others],
        "z": z, "p_raw": p, "p_holm": holm(p),
    }).sort_values("p_holm").reset_index(drop=True)


def vargha_delaney_a12(a, b) -> float:
    """P(a < b) + 0.5 P(a = b) for minimisation: **A12 > 0.5 means `a` is better**.

    Computed from the Mann-Whitney U statistic, so it is O(n log n) and exact for
    ties. The orientation is the one thing about this function that is easy to
    get backwards, and getting it backwards silently swaps every "helps" and
    "hurts" verdict downstream, so ``test_a12_orientation`` pins it.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    # rank-sum of `a` when ranking b-then-a ascending; smaller values rank first
    ranks = sps.rankdata(np.concatenate([a, b]))
    r_a = ranks[:na].sum()
    u_a = r_a - na * (na + 1) / 2.0          # # of pairs where a > b (ties at 0.5)
    return float(1.0 - u_a / (na * nb))      # invert: lower error is better


def a12_magnitude(a12: float) -> str:
    d = abs(a12 - 0.5)
    if np.isnan(a12):
        return "n/a"
    if d < 0.056:
        return "negligible"
    if d < 0.138:
        return "small"
    if d < 0.214:
        return "medium"
    return "large"


def cliffs_delta(a, b) -> float:
    """2*A12 - 1, on [-1, 1]; positive means `a` is better (minimisation)."""
    return 2.0 * vargha_delaney_a12(a, b) - 1.0


def paired_wilcoxon(a, b) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when all differences are zero."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length samples")
    if np.allclose(a, b):
        return 1.0
    try:
        return float(sps.wilcoxon(a, b, zero_method="wilcox",
                                  alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def mannwhitney(a, b) -> float:
    """Two-sided Mann-Whitney U p-value (the original study's test)."""
    try:
        p = sps.mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        return 1.0
    return 1.0 if np.isnan(p) else float(p)


def descriptive(values) -> dict:
    """best / worst / mean / std / median / IQR -- the full set journals expect."""
    v = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    return {"best": float(v.min()), "worst": float(v.max()), "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "iqr": float(q3 - q1)}


# ==========================================================================
# 2. FES ACCOUNTING
# ==========================================================================

class Tracker:
    """One tracker wraps one problem for one run.

    Every algorithm sees only the tracker, so the evaluation budget is counted
    identically for all of them and no algorithm can buy extra evaluations. CEC
    functions are deterministic and static, so best-so-far is the correct metric
    and the score is ``true_obj(best_x) - f*`` with the competition's 1e-8 floor.
    """

    def __init__(self, problem, max_fes: int, n_points: int = N_POINTS):
        self.problem = problem
        self.max_fes = int(max_fes)
        self.fes = 0
        self.best_f = np.inf
        self.best_x = None
        self.overrun = 0

        self.checkpoints = np.linspace(max_fes / n_points, max_fes, n_points).astype(int)
        self.curve = np.full(n_points, np.nan)
        self._k = 0

        self._f_star = problem.f_star
        self._error_floor = float(getattr(problem, "error_floor", 0.0) or 0.0)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        f = self.problem(x)
        self.fes += 1

        if self.fes > self.max_fes:
            self.overrun += 1
            return f

        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()

        while self._k < len(self.checkpoints) and self.fes >= self.checkpoints[self._k]:
            self.curve[self._k] = self._curve_value()
            self._k += 1
        return f

    def _floor(self, err: float) -> float:
        return 0.0 if (self._error_floor and err < self._error_floor) else err

    def _curve_value(self) -> float:
        return self._floor(float(self.best_f - self._f_star))

    def finalize(self) -> tuple[np.ndarray, float]:
        """Return (convergence curve, final score). Both are error values."""
        if self._k < len(self.curve):
            self.curve[self._k:] = self._curve_value()
        return self.curve, self.score()

    def score(self) -> float:
        if self.best_x is None:
            return float("inf")
        return self._floor(float(self.problem.true_obj(self.best_x) - self._f_star))


# ==========================================================================
# 3. BENCHMARK SUITES
# ==========================================================================

@dataclass(frozen=True)
class SuiteSpec:
    name: str
    dims: tuple[int, ...]
    runs: int
    max_fes: Callable[[int], int]
    function_ids: tuple           # official ids
    error_floor: float
    protocol_verified: bool
    note: str = ""
    extra: dict = field(default_factory=dict)

    def label(self, fid) -> str:
        return f"F{fid}" if isinstance(fid, int) else str(fid)


def _cec2017_ids() -> tuple[int, ...]:
    """Official ids present in opfunu: F1 and F3..F30."""
    return (1,) + tuple(range(3, 31))


SUITES: dict[str, SuiteSpec] = {
    "cec2017": SuiteSpec(
        name="cec2017",
        dims=(10, 30, 50, 100),
        runs=51,
        max_fes=lambda d: 10_000 * d,
        function_ids=_cec2017_ids(),
        error_floor=CEC_ERROR_FLOOR,
        protocol_verified=True,
        note="CEC'2017 bound-constrained: 51 runs, MaxFES = 10000*D, "
             "D in {10,30,50,100}, errors below 1e-8 reported as 0. "
             "F2 excluded (withdrawn as unstable), leaving F1 and F3..F30.",
    ),
    "cec2022": SuiteSpec(
        name="cec2022",
        dims=(10, 20),
        runs=30,
        max_fes=lambda d: 200_000 if d == 10 else 1_000_000,
        function_ids=tuple(range(1, 13)),
        error_floor=CEC_ERROR_FLOOR,
        protocol_verified=False,
        note="CEC'2022 bound-constrained: 12 functions, D in {10,20}, 30 runs -- "
             "these three are confirmed. The MaxFES values are NOT yet confirmed "
             "against the competition technical report and must be before any "
             "number from this suite is quoted. Pass --fes-per-dim to override.",
    ),
}


def _opfunu_classes(suite: str) -> dict[int, type]:
    """opfunu index -> class, for one CEC suite."""
    if suite == "cec2017":
        from opfunu.cec_based import cec2017 as mod
        tag, year = "cec2017", "2017"
    elif suite == "cec2022":
        from opfunu.cec_based import cec2022 as mod
        tag, year = "cec2022", "2022"
    else:
        raise ValueError(f"not a CEC suite: {suite!r}")
    out = {}
    for cname, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__.endswith(tag):
            out[int(cname[1:cname.index(year)])] = obj
    return out


def official_to_opfunu(suite: str, official_id: int) -> int:
    """Map an official CEC function id onto opfunu's index."""
    if suite == "cec2022":
        return official_id
    if suite != "cec2017":
        raise ValueError(f"not a CEC suite: {suite!r}")
    if official_id == 1:
        return 1
    if official_id == 2:
        raise ValueError("official CEC'2017 F2 was withdrawn and is not available")
    if not 3 <= official_id <= 30:
        raise ValueError(f"CEC'2017 has F1..F30, got F{official_id}")
    return official_id - 1


class CECProblem:
    """One CEC function at one dimension."""

    kind = "plain"

    def __init__(self, suite: str, official_id: int, dim: int,
                 error_floor: float | None = None):
        cls = _opfunu_classes(suite)[official_to_opfunu(suite, official_id)]
        self._f = cls(ndim=dim)
        self.suite, self.official_id, self.dim = suite, official_id, dim
        self.name = f"F{official_id}"
        self.lb = float(np.min(self._f.lb))
        self.ub = float(np.max(self._f.ub))
        self.f_star = float(self._f.f_global)
        self.o = np.asarray(self._f.x_global, dtype=float)
        self.error_floor = (SUITES[suite].error_floor if error_floor is None
                            else error_floor)
        self.description = str(getattr(cls, "name", "") or "")

    def true_obj(self, x) -> float:
        return float(self._f.evaluate(np.asarray(x, dtype=float)))

    def __call__(self, x) -> float:
        return float(self._f.evaluate(np.asarray(x, dtype=float)))


def make_suite_problem(suite: str, fid, dim: int):
    """Build one problem. ``fid`` is an official id."""
    return CECProblem(suite, int(fid), dim)


# ==========================================================================
# 4. SHARED OPERATORS
# ==========================================================================

def levy_flight(rng: np.random.Generator, shape, beta: float = 1.5):
    """Mantegna's algorithm for Levy-stable steps."""
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0.0, sigma_u, shape)
    v = rng.normal(0.0, 1.0, shape)
    return u / np.abs(v) ** (1 / beta)


def rank_index(rng, n: int, k: int):
    """``k`` draws from linear rank weights over a fitness-sorted population.

    Index ``i`` (0 = best) carries weight ``n - i``, so the best individual is
    drawn ``n`` times as often as the worst and the worst still has non-zero
    mass. This is L-SHADE-RSP's selective pressure; its reference writes the
    weights as ``3.0 * (NInds - i)``, but the constant cancels under
    normalisation, so it is omitted rather than carried as decoration.

    Sampled by the closed form of the rank CDF. With ``y = n - i`` in ``1..n``,
    ``P(Y <= y) = y(y+1) / (n(n+1))``, which inverts to the expression below --
    one square root per draw, no cumulative weight vector. That matters because
    LPSR changes ``n`` on almost every generation, so a cumsum could never be
    cached, and ``rng.choice(..., p=...)`` re-validates and re-cumsums the
    weights on every call. All three approaches consume exactly ``k`` uniforms,
    so the choice of sampler cannot change the length of the random stream.
    """
    u = rng.random(k)
    y = np.ceil((np.sqrt(1.0 + 4.0 * u * n * (n + 1.0)) - 1.0) * 0.5)
    # u == 0 would give y == 0 and an out-of-range index; the clip also absorbs
    # rounding at the top end.
    return np.clip(n - np.maximum(y, 1.0), 0, n - 1).astype(np.int64)


def union_rank_index(rng, n_pop: int, n_arc: int, k: int):
    """``k`` draws over ``pop + archive``, rank-weighted inside the population.

    Indices ``0..n_pop-1`` are population members and ``n_pop..n_pop+n_arc-1``
    are archive members, matching the ``vstack`` the caller builds.

    Drawing uniformly over the union -- what the unranked path does -- already
    picks an archive member with probability ``n_arc / (n_arc + n_pop)``. That
    Bernoulli is reproduced explicitly here so that switching the flag changes
    the weighting *within* the population and nothing else. Archive members carry
    no fitness rank, so they stay uniform among themselves, as in the reference.
    """
    out = rank_index(rng, n_pop, k)
    if n_arc:
        take = rng.random(k) < n_arc / (n_arc + n_pop)
        out = np.where(take, n_pop + rng.integers(0, n_arc, k), out)
    return out


# ==========================================================================
# 5. THE ALGORITHM UNDER STUDY
# ==========================================================================
# WHAT MEASUREMENT CHANGED. Every deviation from the original hybrid came from an
# ablation, not from taste.
#
#   removed  WOA spiral operator   dominated on every function tested: adding it
#                                  to the pbest mutation was worse than the pbest
#                                  mutation alone, everywhere
#   removed  (1+1)-ES tail         no measurable contribution; on two functions
#                                  the results were bit-identical with and
#                                  without it, so 5% of the budget went back to
#                                  the main loop
#   removed  noise-robust credit   the theory held -- credit-set contamination
#                                  fell 8-9x -- but the final error did not move.
#                                  The F/CR memory is not what limits performance
#                                  under noise; selection error is
#   gated    eigen crossover       only when the covariance has more samples than
#                                  dimensions. LPSR drives the population below D
#                                  long before the end, and a 100x100 covariance
#                                  estimated from 25 points is mostly noise
#   lowered  P_MIN 0.05 -> 0.02    a 0.05 floor guarantees every operator 5% of
#                                  every generation for the whole run, so a
#                                  dominated operator can never be switched off
#   added    diversity guard       an operator that collapses the population
#                                  earns the largest immediate credit while
#                                  causing the collapse that loses the run;
#                                  instrumentation measured 66% of all fitness
#                                  gain going to such an operator on Schwefel. No
#                                  credit scheme built on immediate improvement
#                                  can see this, so the guard measures diversity
#                                  directly
#   added    restart               the CEC'2017 gate measured this family weakest
#                                  on the composition class, which is where
#                                  restarts decide the outcome
#
# ON NORMALISATION. DE/rand/1/bin is invariant under any diagonal affine change
# of variables: the mutation is built from differences, which scale with the
# coordinates, and binomial crossover acts coordinate-wise. So normalising groups
# of variables to a common box buys this algorithm nothing. It is standard
# preparation, stated for reproducibility, and is not a contribution.

def LSHADE_DGR(obj_func, dim, bounds, max_fes, rng,
               POP_FACTOR=12, POP_MAX=500, POP_UNCAP=False, POP_RULE="linear",
               RESTART_POP_MULT=4.0, P_MIN=0.02, P_EIG=0.4,
               ADAPT_EIG=True, ARC_RATE=1.0, CR_FLOOR=True,
               OPS=(0, 1, 3), OPS_DROP_ABOVE=None,
               EIGEN_GATE=True, EIGEN_GATE_EXTRA=0.0,
               EIG_MEMORY=0.0, ALIGN_TAU=0.0,
               CHANNEL_DECOUPLE=False, P_EIG_RULE="success", SELECTION="uniform",
               CREDIT_RULE="rate", DIV_PRICE=0.0,
               DIV_GUARD=True, DIV_THRESH=1e-4, DIV_BOOST=0.5,
               RESTART=True, RESTART_BUDGET_FRAC=0.2, RESTART_GROWTH=2.0,
               logger=None):
    lb, ub = bounds
    OPS = tuple(OPS)
    # DIMENSION-CONDITIONAL PORTFOLIO. Three rounds of screening and two full
    # gates say the same thing from both sides: the operator portfolio earns its
    # keep at D=10 and costs at D=30 and above. In the k=8 line-up the recorded
    # algorithm leads the Composition class at D=10 (3.300) where `core-only`
    # is fifth (4.400), and `core-only` leads it at D=30 (2.150) and D=50
    # (2.900) where the recorded algorithm is fourth and sixth. Hybrid moves the
    # same way, 2.650 -> 2.050 at D=10 but 4.700 -> 2.000 at D=50. The two are
    # not in tension: a portfolio needs room to differentiate between its
    # operators, and what the leader step costs grows with dimension while what
    # it buys does not.
    #
    # Above the threshold the portfolio collapses to the core operator alone.
    # The default None is no threshold at all, so every recorded row keeps its
    # meaning; placed before K_OPS and LEVY_SLOT because both are derived from
    # OPS and a later switch would leave them describing the wrong portfolio.
    #
    # WHAT THIS IS, AND WHAT IT IS NOT. On this suite the parameter has exactly
    # two reachable settings: D runs over {10, 30, 50}, so every threshold in
    # [10, 30) gives one algorithm and every threshold in [30, 50) gives
    # another. It is not a continuous knob that could be overfitted. But the
    # choice between those settings was read off our own D=10 gate, which the
    # binding criterion also reads, so it is a post-hoc design choice and the
    # report states it as one. In particular the criterion's "D=10 not made
    # worse in any class" is satisfied here *by construction* -- at D=10 this
    # is the recorded algorithm, bit for bit -- and not by a measured
    # improvement. Claiming otherwise would be the one error that invalidates
    # the study.
    if OPS_DROP_ABOVE is not None and dim > OPS_DROP_ABOVE:
        OPS = (0,)
    N_min, H_SIZE = 4, 6
    K_OPS = len(OPS)
    LEVY_SLOT = OPS.index(3) if 3 in OPS else None
    span = float(ub - lb)

    # POP_MAX has two jobs: it caps the initial size here, and it is the ceiling
    # on restart growth below. They are separated into two names so that lifting
    # one does not silently move the other -- raising the initial size past a
    # POP_MAX that stays put would make the restart `min` shrink the population
    # instead of growing it, which is the opposite of an IPOP restart.
    # POP_MAX = 500 binds from D = 42, so at D = 50 this carries 500 individuals
    # where jSO carries 25*ln(D)*sqrt(D) = 692 and L-SHADE carries 18*D = 900 --
    # the smallest population of any DE rival, and narrowest at exactly the
    # dimension where this family loses the hybrid class. POP_UNCAP lifts it;
    # POP_RULE offers L-SHADE-RSP's sizing as an alternative to be measured.
    N_base = (int(round(POP_FACTOR * dim)) if POP_RULE == "linear"
              else int(round(75 * dim ** (2.0 / 3.0))))
    N_init = max(40, N_base) if POP_UNCAP else int(np.clip(N_base, 40, POP_MAX))
    pop_ceiling = int(RESTART_POP_MULT * N_init) if POP_UNCAP else POP_MAX
    # P_EIG is a parameter that the success rule reassigns. The working value is
    # kept in its own name so the argument survives as the starting point, and
    # deliberately NOT reset per epoch: a restart re-initialises op_quality,
    # op_prob, eig_quality and B, but the eigen probability has always carried
    # over, and every row in results_cec2017/ was produced that way.
    p_eig = float(P_EIG)
    fes = 0

    def evaluate(x):
        nonlocal fes
        fes += 1
        return obj_func(x)

    # -- one epoch per (re)start ----------------------------------------------
    while fes < max_fes:
        pop_size = N_init
        pop = lb + rng.random((pop_size, dim)) * (ub - lb)
        fitness = np.empty(pop_size)
        for i in range(pop_size):
            if fes >= max_fes:
                return
            fitness[i] = evaluate(pop[i])

        M_F = np.full(H_SIZE, 0.3)
        M_CR = np.full(H_SIZE, 0.8)
        M_F[-1], M_CR[-1] = 0.9, 0.9          # jSO: last memory cell frozen
        k_mem = 0
        archive = np.empty((0, dim))
        op_quality = np.full(K_OPS, 0.5)
        op_prob = np.full(K_OPS, 1.0 / K_OPS)
        eig_quality = np.array([0.5, 0.5])
        B = np.eye(dim)
        # Reset with the basis it feeds: a restart re-initialises the
        # population, so a covariance accumulated from the previous epoch's
        # would describe a distribution that no longer exists.
        C_cum = None
        epoch_fes0 = fes
        stalled = 0

        while fes < max_fes:
            t = fes / max_fes
            order = np.argsort(fitness)
            pop, fitness = pop[order], fitness[order]
            idx = np.arange(pop_size)

            # eigenbasis only where the covariance is actually estimable
            #
            # WHERE THE THRESHOLD BELONGS. rank(C) <= n_samples - 1, so
            # n_samples > dim is what keeps the basis identified -- but being
            # identified is not the same as being usable. For n samples of p
            # variables the sample eigenvalues of isotropic data spread over
            # [(1-sqrt(p/n))^2, (1+sqrt(p/n))^2] (Marchenko-Pastur), so as
            # p/n -> 1 the estimate approaches pure noise. At D=50 the gate as
            # recorded admits the basis at n_samples=51, where that interval is
            # [1.0e-4, 3.96] -- a spurious anisotropy of about 38800x. A usable
            # estimate needs p/n << 1, not p/n < 1. EIGEN_GATE_EXTRA raises the
            # threshold to (1 + extra) * dim; at 0.0 the condition is exactly
            # the recorded `n_samples > dim`, since (1.0 + 0.0) * dim is dim.
            n_samples = max(2, pop_size // 2)
            eig_ok = (not EIGEN_GATE) or (n_samples > (1.0 + EIGEN_GATE_EXTRA) * dim)
            cond_hat = cond_adj = align = float("nan")
            mp_valid, eig_branch = False, "gate closed"
            if eig_ok:
                C = np.cov(pop[:n_samples], rowvar=False)
                # CUMULATIVE BASIS ESTIMATE. The covariance above is recomputed
                # from scratch every generation, so its effective sample size is
                # whatever n_samples happens to be -- and by the note above that
                # is not enough for most of the run. An exponentially weighted
                # mean of the per-generation covariances raises it: with weight
                # a = 1 - EIG_MEMORY on the newest generation the squared
                # weights sum to a/(2-a), so the effective sample size is
                # n_samples * (2-a)/a, i.e. 9x at EIG_MEMORY=0.8. Successive
                # populations overlap heavily, so that factor is an upper bound
                # rather than a promise; what it cannot do is make the estimate
                # noisier. Branching rather than scaling by (1 - a): at
                # EIG_MEMORY=0.0 this leaves C bit-identical, while 0.0 * C_cum
                # would be NaN if C_cum ever held an inf.
                if EIG_MEMORY > 0.0 and np.all(np.isfinite(C)):
                    eig_a = 1.0 - EIG_MEMORY
                    C_cum = (C if C_cum is None
                             else (1.0 - eig_a) * C_cum + eig_a * C)
                    C = C_cum
                # BASIS MISALIGNMENT. Computed whenever it is logged as well as
                # when the rule below uses it, so the statistic can be read off
                # a default run without enabling the rule -- the same
                # arrangement as `rho`.
                if (P_EIG_RULE == "align" or logger is not None) and np.all(np.isfinite(C)):
                    c_norm = float(np.linalg.norm(C))
                    align = (float(np.linalg.norm(C - np.diag(np.diag(C)))) / c_norm
                             if c_norm > 0.0 else 0.0)
                w = None
                if np.all(np.isfinite(C)):
                    try:
                        # The eigenvalues used to be discarded. The condition
                        # rules below are read straight off this decomposition,
                        # so they cost nothing extra.
                        w, B = np.linalg.eigh(C)
                        eig_branch = "ok"
                    except np.linalg.LinAlgError:
                        w, eig_branch = None, "eigh failed"
                else:
                    eig_branch = "C not finite"

                # CONDITION-DRIVEN BASIS ADAPTATION. The success rule asks "did
                # the eigen crossover win last generation"; this asks "is the
                # landscape ill-conditioned enough to need one", which is the
                # question the operator is actually for.
                #
                # The Marchenko-Pastur term is load-bearing, not decoration. A
                # sample covariance of p variables from n points is ill
                # conditioned even when the truth is a sphere: at p=10, n=12 the
                # null condition number is about 483, so an uncorrected rule
                # reports P_EIG = 0.63 on a perfect sphere. Dividing by the null
                # asks whether the data are worse conditioned than chance alone
                # would make them. LPSR shrinks n over the run, so without the
                # correction the rule drifts upward on every function -- which is
                # why condition_raw stays available and gets measured rather than
                # argued about.
                if w is not None and P_EIG_RULE in ("condition_mp", "condition_raw"):
                    lam_max = float(w[-1])          # eigh returns them ascending
                    if not np.isfinite(lam_max) or lam_max <= 0.0:
                        # A collapsed or rank-dead covariance carries no
                        # direction, and the ratio below would divide by zero.
                        # Hold the current value instead of inventing one.
                        eig_branch = "degenerate"
                    else:
                        # Floor lam_min *relative* to lam_max. An absolute floor
                        # would make the condition number depend on the units of
                        # x, in an algorithm that is otherwise invariant to a
                        # diagonal change of variables.
                        lam_min = max(float(w[0]), lam_max * dim * np.finfo(float).eps)
                        cond_hat = lam_max / lam_min
                        # EIGEN_GATE guarantees n > p, but it can be switched off,
                        # and then n_samples == dim exactly at pop_size 2*dim and
                        # 2*dim+1 -- values LPSR walks through on every run. There
                        # 1 - sqrt(p/n) is exactly 0.0. Degrade to the raw rule
                        # rather than divide by it.
                        ratio = dim / n_samples
                        mp_valid = ratio < 1.0 - 1e-12
                        if P_EIG_RULE == "condition_mp" and mp_valid:
                            s = math.sqrt(ratio)
                            cond_null = ((1.0 + s) / (1.0 - s)) ** 2
                        else:
                            cond_null = 1.0
                        cond_adj = max(cond_hat / cond_null, 1.0)
                        p_eig = float(np.clip(
                            1.0 - 1.0 / math.log10(max(cond_adj, 10.0)), 0.1, 0.9))
                        eig_branch = "condition"

                # MISALIGNMENT, NOT CONDITIONING. The rules above adapt on
                # kappa(C), and kappa does not measure the quantity that breaks
                # binomial crossover. Two counterexamples settle it:
                # C = diag(1, 1e6) has kappa = 1e6 and its eigenbasis IS the
                # coordinate basis, so rotating into it gains exactly nothing;
                # C = [[1, 0.99], [0.99, 1]] has kappa = 199 and its eigenbasis
                # sits 45 degrees off the axes, where coordinate-wise crossover
                # is maximally wrong. What separates them is how much of C lies
                # off the diagonal,
                #
                #     align = ||C - diag(C)||_F / ||C||_F   in [0, 1)
                #
                # which is 0 exactly when C is diagonal -- exactly when the
                # eigenbasis is a signed permutation of the coordinate axes, and
                # binomial crossover is already equivariant under those, so
                # there is nothing to gain. This is the measured explanation for
                # why condition_mp and condition_raw were both worse than the
                # success rule at D=50: they adapt on the wrong quantity.
                #
                # A HARD SWITCH, NOT A PROBABILITY. Equivariance is
                # all-or-nothing. Crossover in the eigenbasis is exactly
                # rotation-equivariant, coordinate crossover is not, and a
                # p-mixture of the two is equivariant only at p = 1, because the
                # mixture's equivariance forces the second component's. Both
                # adaptive rules clip p to [0.1, 0.9], so neither can ever reach
                # the only value at which the property holds; that is a
                # structural fact about the parameter, not a tuning failure.
                #
                # NOT CLAIMED AS NEW. Covariance-based crossover is EA4eig,
                # L-SRTDE and LSHADE-cnEpSin; rotation invariance through a
                # learned basis is CMA-ES. Whether off-diagonal mass has been
                # used as the switching signal is unverified -- see
                # reports/PRIOR_ART.md, and do not call this new until it is.
                if w is not None and P_EIG_RULE == "align" and np.isfinite(align):
                    p_eig = 1.0 if align >= ALIGN_TAU else 0.0
                    eig_branch = "align"

            # parameters: success-history with jSO's constraint schedule
            r = rng.integers(0, H_SIZE, pop_size)
            CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
            if CR_FLOOR:
                if t < 0.25:
                    CR = np.maximum(CR, 0.7)
                elif t < 0.5:
                    CR = np.maximum(CR, 0.6)
            F = M_F[r] + 0.1 * rng.standard_cauchy(pop_size)
            bad = F <= 0
            while np.any(bad):
                F[bad] = M_F[r[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
                bad = F <= 0
            F = np.minimum(F, 1.0)
            if t < 0.6:
                F = np.minimum(F, 0.7)
            Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
            Fc, Fwc = F[:, None], Fw[:, None]

            p_num = max(2, int(round((0.25 - 0.125 * t) * pop_size)))
            x_pbest = pop[rng.integers(0, p_num, pop_size)]
            union_pop = np.vstack([pop, archive]) if len(archive) else pop
            n_union = len(union_pop)
            if SELECTION == "rank":
                # RANK-BASED SELECTIVE PRESSURE (L-SHADE-RSP). F11 is lost 4.6x
                # at D=50 with the diversity guard never firing, so neither the
                # guard nor the basis can be the cause there; what is left is
                # where the difference vector comes from.
                #
                # Only the weighting inside the population changes; the whole
                # mixture -- not just the index -- is redrawn on a clash, which
                # keeps the rejection semantics identical to the unranked path
                # and stops a retry leaking a uniform draw.
                n_arc = n_union - pop_size

                # The reference ranks Rand1 and Rand2 and leaves prand uniform
                # inside the top-p subset, so x_pbest above is untouched. r1 needs
                # its own rejection: the uniform arm gets i != r1 for free from
                # the modular offset, but a rank draw does not, and the clash loop
                # below only tests r2.
                r1 = rank_index(rng, pop_size, pop_size)
                bad, tries = r1 == idx, 0
                while np.any(bad) and tries < 100:
                    r1[bad] = rank_index(rng, pop_size, int(bad.sum()))
                    bad, tries = r1 == idx, tries + 1
                r2 = union_rank_index(rng, pop_size, n_arc, pop_size)
                clash, guard = (r2 == idx) | (r2 == r1), 0
                while np.any(clash) and guard < 100:
                    r2[clash] = union_rank_index(rng, pop_size, n_arc,
                                                 int(clash.sum()))
                    clash = (r2 == idx) | (r2 == r1)
                    guard += 1
            else:
                r1 = (idx + rng.integers(1, pop_size, pop_size)) % pop_size
                r2 = rng.integers(0, n_union, pop_size)
                clash = (r2 == idx) | (r2 == r1)
                guard = 0
                while np.any(clash) and guard < 100:
                    r2[clash] = rng.integers(0, n_union, int(clash.sum()))
                    clash = (r2 == idx) | (r2 == r1)
                    guard += 1
            diff = pop[r1] - union_pop[r2]
            X_lead = np.array([0.5, 0.3, 0.2]) @ pop[:3]

            # -- diversity guard ---------------------------------------------
            # Credit assignment cannot see a population collapse, because the
            # operator that causes it is also the one producing the largest
            # immediate gains. Measuring the spread directly can.
            diversity = float(np.mean(np.std(pop, axis=0))) / span
            probs = op_prob
            guard_active = False
            if DIV_GUARD and LEVY_SLOT is not None and K_OPS > 1:
                guard_active = diversity < DIV_THRESH and t < 0.95
                if guard_active:
                    probs = np.full(K_OPS, (1.0 - DIV_BOOST) / (K_OPS - 1))
                    probs[LEVY_SLOT] = DIV_BOOST

            ops = np.asarray(OPS)[rng.choice(K_OPS, pop_size, p=probs)]
            V = np.empty_like(pop)
            m = ops == 0                       # current-to-pbest-w/1 (jSO core)
            if np.any(m):
                V[m] = pop[m] + Fwc[m] * (x_pbest[m] - pop[m]) + Fc[m] * diff[m]
            m = ops == 1                       # translation-invariant leader step
            if np.any(m):
                V[m] = pop[m] + Fc[m] * (X_lead - pop[m]) + Fc[m] * diff[m]
            m = ops == 3                       # Levy-flight escape
            if np.any(m):
                L = np.clip(levy_flight(rng, (int(m.sum()), dim)), -5.0, 5.0)
                V[m] = pop[m] + Fc[m] * (x_pbest[m] - pop[m]) + Fc[m] * L * diff[m]

            cross = rng.random((pop_size, dim)) < CR[:, None]
            cross[idx, rng.integers(0, dim, pop_size)] = True
            U = np.where(cross, V, pop)
            if eig_ok:
                use_eig = rng.random(pop_size) < p_eig
                if np.any(use_eig):
                    xe, ve = pop[use_eig] @ B, V[use_eig] @ B
                    U[use_eig] = np.where(cross[use_eig], ve, xe) @ B.T
            else:
                use_eig = np.zeros(pop_size, dtype=bool)

            low, high = U < lb, U > ub         # midpoint-target repair (L-SHADE)
            U[low] = ((lb + pop) / 2.0)[low]
            U[high] = ((ub + pop) / 2.0)[high]

            n_eval = min(pop_size, max_fes - fes)
            fit_U = np.full(pop_size, np.inf)
            for i in range(n_eval):
                fit_U[i] = evaluate(U[i])

            improved = fit_U < fitness
            accept = fit_U <= fitness

            if np.any(improved):
                archive = np.vstack([archive, pop[improved]])
                df = fitness[improved] - fit_U[improved]
                w = df / df.sum()
                S_CR = CR[improved]
                den = np.sum(w * S_CR)
                mcr = np.sum(w * S_CR**2) / den if den > 0 else 0.0
                S_F = F[improved]
                dF = np.sum(w * S_F)
                if dF > 0:
                    M_F[k_mem] = (M_F[k_mem] + np.sum(w * S_F**2) / dF) / 2.0
                M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
                k_mem = (k_mem + 1) % (H_SIZE - 1)

            rho = np.full(K_OPS, np.nan)
            rho_bar = float("nan")
            if CREDIT_RULE == "diversity" or logger is not None:
                centroid = pop.mean(axis=0)
                d_par = ((pop - centroid) ** 2).sum(axis=1)
                d_off = ((U - centroid) ** 2).sum(axis=1)
                num, den = np.zeros(K_OPS), np.zeros(K_OPS)
                for slot, k in enumerate(OPS):
                    mk = (ops == k) & accept
                    if np.any(mk):
                        num[slot] = d_off[mk].sum()
                        den[slot] = d_par[mk].sum()
                if den.sum() > 0:
                    rho = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                    # The reference is the MOST expansive operator this
                    # generation, not the pooled ratio. Pooling was measured and
                    # is dead: the pooled value is dominated by whichever
                    # operator touched the most spread, every per-operator ratio
                    # then sits above it, and the resulting price is ~0 for all
                    # of them -- with the leader step penalised least of the
                    # three, the exact opposite of what is wanted. Measured
                    # against the most expansive operator instead, the leader
                    # step is priced 1.6-16.9x higher than the core operator on
                    # every function tried, which is the separation the rule
                    # needs to act on.
                    finite = rho[np.isfinite(rho)]
                    if finite.size:
                        rho_bar = float(finite.max())

            if logger is not None:
                # P_EIG and op_quality are what the F4/F5 diagnosis turns on: the
                # question is whether success-driven adaptation switches the eigen
                # crossover off on exactly the rotated, ill-conditioned problems
                # where it is the one thing that could help.
                logger.append({"fes": fes, "t": t, "diversity": diversity,
                               "guard": guard_active, "op_prob": op_prob.copy(),
                               "eig_ok": eig_ok, "pop_size": pop_size,
                               "P_EIG": float(p_eig),
                               "n_samples": int(n_samples),
                               "align": float(align),
                               "cond_hat": float(cond_hat),
                               # log10, because _diag_internals aggregates with
                               # mean() and a mean over 1 .. 1e10 is not a statistic
                               "log10_cond_adj": (float(np.log10(cond_adj))
                                                  if np.isfinite(cond_adj) else float("nan")),
                               "mp_valid": bool(mp_valid),
                               "rho": rho.copy(),
                               "rho_bar": float(rho_bar),
                               "eig_branch": eig_branch,
                               "op_quality": op_quality.copy(),
                               "eig_quality": eig_quality.copy(),
                               "n_eig_used": int(use_eig.sum()),
                               "improved": int(improved.sum()),
                               "best": float(min(fitness.min(), fit_U.min()))})

            # credit for the operator portfolio; frozen while the guard overrides
            # the distribution, so forced Levy trials do not distort the estimates
            #
            # WHAT THE CREDIT SIGNAL CAN AND CANNOT SEE. The "rate" rule scores an
            # operator by how often it improves a parent. That cannot separate an
            # operator making many tiny improvements from one making few large
            # ones, and the two are not equally valuable. Measured at D=50: the
            # leader step holds 52% of the budget on F15 and F16 -- the two worst
            # losses in the gate -- because moving a point toward the weighted
            # top-3 centroid almost always improves it slightly, while collapsing
            # the spread that the rest of the search needs. Removing that operator
            # outright improves the F12 median 72-fold and F16 19-fold.
            #
            # The "gain" rule scores mean improvement per trial instead, which is
            # exactly the quantity the rate rule discards. Two properties matter:
            # dividing by the trial count keeps an operator from earning credit
            # merely for holding a large share of the budget, which would be a
            # feedback loop; and normalising across operators within a generation
            # makes the signal scale-free, so it does not drift as improvements
            # shrink over the run or change magnitude between functions.
            # DIVERSITY PRICE. Both rules above score an operator by the good it
            # does this generation, and the measurements say that is exactly why
            # neither can see the leader step: it genuinely produces the largest
            # immediate improvements, more often AND bigger, while pulling the
            # population onto the weighted top-3 centroid. An operator like that
            # wins every immediate metric and loses the run.
            #
            # What it spends, rather than what it earns, is the population's
            # spread. For the trials an operator got accepted, compare the
            # squared distances of its offspring to the centroid against those of
            # the parents they replaced:
            #
            #     rho_k = sum_{i in A_k} ||u_i - m||^2 / sum_{i in A_k} ||x_i - m||^2
            #
            # Acceptance is fitness-based, so on a converging population every
            # operator shows rho < 1 late in the run -- an absolute threshold
            # would penalise all of them equally and carry no information. The
            # ratio is therefore taken against the generation's pooled rho, which
            # cancels the common convergence trend and leaves only the difference
            # BETWEEN operators:
            #
            #     c_k = max(0, 1 - rho_k / rho_bar)
            #
            # The credit is then the existing signal minus a price on that
            # contraction. At DIV_PRICE = 0 this is bit-identical to "rate", so
            # the new term is one parameter on top of the recorded algorithm
            # rather than a different algorithm.
            if not guard_active:
                if CREDIT_RULE == "gain":
                    gain = np.where(improved, fitness - fit_U, 0.0)
                    per_trial = np.zeros(K_OPS)
                    for slot, k in enumerate(OPS):
                        mk = ops == k
                        n_k = int(mk.sum())
                        if n_k:
                            per_trial[slot] = gain[mk].sum() / n_k
                    scale = per_trial.sum()
                    if scale > 0:
                        for slot, k in enumerate(OPS):
                            if np.any(ops == k):
                                op_quality[slot] = (0.7 * op_quality[slot]
                                                    + 0.3 * per_trial[slot] / scale)
                    # scale == 0 means no operator improved anything this
                    # generation. That is an absence of evidence, not evidence
                    # against every operator, so the estimates are left alone.
                elif CREDIT_RULE == "diversity":
                    usable = np.isfinite(rho_bar) and rho_bar > 0
                    for slot, k in enumerate(OPS):
                        mk = ops == k
                        if np.any(mk):
                            rate_k = improved[mk].mean()
                            # No accepted trial this generation means no evidence
                            # about what the operator spent, not evidence that it
                            # spent nothing -- the price is then zero.
                            price = (max(0.0, 1.0 - rho[slot] / rho_bar)
                                     if usable and np.isfinite(rho[slot]) else 0.0)
                            op_quality[slot] = (0.7 * op_quality[slot]
                                                + 0.3 * max(0.0, rate_k - DIV_PRICE * price))
                else:
                    for slot, k in enumerate(OPS):
                        mk = ops == k
                        if np.any(mk):
                            op_quality[slot] = 0.7 * op_quality[slot] + 0.3 * improved[mk].mean()
                q_sum = op_quality.sum()
                op_prob = P_MIN + (1.0 - K_OPS * P_MIN) * (
                    op_quality / q_sum if q_sum > 0 else np.full(K_OPS, 1.0 / K_OPS))

            # CROSS-CHANNEL INTERFERENCE. The eigen probability is a separate
            # adaptive channel that happens to live inside the same `if`. The
            # guard exists to override the *operator* distribution; freezing
            # basis adaptation as well is a side effect nobody chose. It is not
            # harmless: the guard is active 44.5% of the budget on F15 and 17.9%
            # on F16 at D=50, the two worst losses in the gate, so the channel
            # that is supposed to answer rotation is switched off for half the
            # run on exactly the functions where it is needed. CHANNEL_DECOUPLE
            # lifts this half out while the operator half stays frozen.
            #
            # The two `if`s are deliberately separate rather than one compound
            # condition: with the flag off they are exactly equivalent, and
            # keeping them apart is what makes the split auditable.
            if (not guard_active) or CHANNEL_DECOUPLE:
                # ADAPT_EIG gates the success rule only -- P_EIG_RULE is
                # authoritative. Without this the "fixed" rule would keep
                # adapting, since ADAPT_EIG defaults to True.
                if ADAPT_EIG and eig_ok and P_EIG_RULE == "success":
                    for k, mk in enumerate([~use_eig, use_eig]):
                        if np.any(mk):
                            eig_quality[k] = 0.7 * eig_quality[k] + 0.3 * improved[mk].mean()
                    e_sum = eig_quality.sum()
                    p_eig = (float(np.clip(eig_quality[1] / e_sum, 0.1, 0.9))
                             if e_sum > 0 else 0.5)

            pop[accept], fitness[accept] = U[accept], fit_U[accept]
            stalled = 0 if np.any(improved) else stalled + 1

            # LPSR measured over this epoch's own share of the budget
            epoch_span = max(1, max_fes - epoch_fes0)
            frac = (fes - epoch_fes0) / epoch_span
            new_size = max(N_min, int(round(N_init + (N_min - N_init) * frac)))
            if new_size < pop_size:
                keep = np.argsort(fitness)[:new_size]
                pop, fitness = pop[keep], fitness[keep]
                pop_size = new_size
            arc_max = int(round(ARC_RATE * pop_size))
            if len(archive) > arc_max:
                archive = archive[rng.choice(len(archive), arc_max, replace=False)]

            if RESTART:
                budget_left = (max_fes - fes) / max_fes
                collapsed = diversity < DIV_THRESH or pop_size <= N_min
                if collapsed and stalled > 0 and budget_left > RESTART_BUDGET_FRAC:
                    grown = int(min(pop_ceiling, round(N_init * RESTART_GROWTH)))
                    # An epoch that cannot afford to evaluate its own initial
                    # population spends the whole tail drawing uniform random
                    # points, and no budget test can see it: the initialisation
                    # loop returns mid-population, so Overrun stays 0 and FES
                    # still reaches the budget. Only reachable once the absolute
                    # cap is lifted, so the check is confined to that arm and
                    # the recorded behaviour is untouched.
                    if POP_UNCAP and grown > 0.1 * (max_fes - fes):
                        grown = N_init
                    N_init = grown
                    break                      # outer loop reinitialises


# ==========================================================================
# 6. THE RIVALS
# ==========================================================================
# The original study compares its hybrid against DE/rand/1/bin (1995) and five
# swarm metaheuristics (2014-2019). It does not compare against any modern
# adaptive DE, even though its own header names L-SHADE and jSO as the sources of
# its core. That makes the headline claim untestable: beating DE/rand/1/bin does
# not show that the hybridisation contributes anything.
#
# ``LSHADE``  Tanabe & Fukunaga, CEC 2014. N_init = 18*D, H = 6, p = 0.11,
#             archive rate 2.6, current-to-pbest/1 with archive, binomial
#             crossover, midpoint repair, LPSR, weighted Lehmer means with the
#             terminal-CR rule.
# ``jSO``     Brest et al., CEC 2017. The tightest possible comparator:
#             L-SHADE-DGR's operator 0, its F/CR constraint schedule, its
#             weighted F, its frozen last memory cell, its p-schedule and its
#             boundary repair are all jSO. Any gap between the two is
#             attributable to exactly the extra operators, the eigen crossover,
#             the diversity guard and the restart.

TERMINAL = -1.0     # L-SHADE's "CR memory is dead" marker


def _sample_F(rng, M_F, r, n):
    F = M_F[r] + 0.1 * rng.standard_cauchy(n)
    bad = F <= 0
    while np.any(bad):
        F[bad] = M_F[r[bad]] + 0.1 * rng.standard_cauchy(int(bad.sum()))
        bad = F <= 0
    return np.minimum(F, 1.0)


def _distinct_r1_r2(rng, pop_size, n_union, idx):
    r1 = (idx + rng.integers(1, pop_size, pop_size)) % pop_size
    r2 = rng.integers(0, n_union, pop_size)
    clash = (r2 == idx) | (r2 == r1)
    guard = 0
    while np.any(clash) and guard < 100:
        r2[clash] = rng.integers(0, n_union, int(clash.sum()))
        clash = (r2 == idx) | (r2 == r1)
        guard += 1
    return r1, r2


def LSHADE(obj_func, dim, bounds, max_fes, rng, POP_FACTOR=18, ARC_RATE=2.6,
           P_BEST=0.11):
    lb, ub = bounds
    N_init = int(POP_FACTOR * dim)
    N_min, H = 4, 6
    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k_mem = 0
    archive = np.empty((0, dim))

    while fes < max_fes:
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        r = rng.integers(0, H, pop_size)
        CR = np.where(M_CR[r] == TERMINAL, 0.0,
                      np.clip(rng.normal(np.where(M_CR[r] == TERMINAL, 0.0, M_CR[r]),
                                         0.1), 0.0, 1.0))
        F = _sample_F(rng, M_F, r, pop_size)
        Fc = F[:, None]

        p_num = max(2, int(round(P_BEST * pop_size)))
        x_pbest = pop[rng.integers(0, p_num, pop_size)]
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r1, r2 = _distinct_r1_r2(rng, pop_size, len(union_pop), idx)

        V = pop + Fc * (x_pbest - pop) + Fc * (pop[r1] - union_pop[r2])

        cross = rng.random((pop_size, dim)) < CR[:, None]
        cross[idx, rng.integers(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)

        low, high = U < lb, U > ub
        U[low] = ((lb + pop) / 2.0)[low]
        U[high] = ((ub + pop) / 2.0)[high]

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        if np.any(improved):
            archive = np.vstack([archive, pop[improved]])
            df = fitness[improved] - fit_U[improved]
            w = df / df.sum()
            S_CR, S_F = CR[improved], F[improved]
            if M_CR[k_mem] == TERMINAL or S_CR.max() == 0.0:
                M_CR[k_mem] = TERMINAL
            else:
                M_CR[k_mem] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
            M_F[k_mem] = np.sum(w * S_F**2) / np.sum(w * S_F)
            k_mem = (k_mem + 1) % H

        accept = fit_U <= fitness
        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]


def jSO(obj_func, dim, bounds, max_fes, rng, ARC_RATE=1.0):
    lb, ub = bounds
    N_init = int(round(25.0 * np.log(dim) * np.sqrt(dim)))
    N_min, H = 4, 5
    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H, 0.3)
    M_CR = np.full(H, 0.8)
    M_F[-1], M_CR[-1] = 0.9, 0.9      # frozen last cell
    k_mem = 0
    archive = np.empty((0, dim))
    p_max, p_min = 0.25, 0.125

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        r = rng.integers(0, H, pop_size)
        CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
        if t < 0.25:
            CR = np.maximum(CR, 0.7)
        elif t < 0.5:
            CR = np.maximum(CR, 0.6)
        F = _sample_F(rng, M_F, r, pop_size)
        if t < 0.6:
            F = np.minimum(F, 0.7)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        Fc, Fwc = F[:, None], Fw[:, None]

        p = p_max - (p_max - p_min) * t
        p_num = max(2, int(round(p * pop_size)))
        x_pbest = pop[rng.integers(0, p_num, pop_size)]
        union_pop = np.vstack([pop, archive]) if len(archive) else pop
        r1, r2 = _distinct_r1_r2(rng, pop_size, len(union_pop), idx)

        V = pop + Fwc * (x_pbest - pop) + Fc * (pop[r1] - union_pop[r2])

        cross = rng.random((pop_size, dim)) < CR[:, None]
        cross[idx, rng.integers(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)

        low, high = U < lb, U > ub
        U[low] = ((lb + pop) / 2.0)[low]
        U[high] = ((ub + pop) / 2.0)[high]

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        if np.any(improved):
            archive = np.vstack([archive, pop[improved]])
            df = fitness[improved] - fit_U[improved]
            w = df / df.sum()
            S_CR, S_F = CR[improved], F[improved]
            den = np.sum(w * S_CR)
            mcr = np.sum(w * S_CR**2) / den if den > 0 else 0.0
            M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
            M_F[k_mem] = (M_F[k_mem] + np.sum(w * S_F**2) / np.sum(w * S_F)) / 2.0
            k_mem = (k_mem + 1) % (H - 1)

        accept = fit_U <= fitness
        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]


def L_SHADE_RSP(obj_func, dim, bounds, max_fes, rng, ENFORCE_DISTINCT=False):
    """L-SHADE-RSP (Stanovov, Akhmedova & Semenkin, CEC 2018) -- CEC'2018 winner.

    jSO's direct successor, and the reason it is in the line-up: this study is
    level with jSO, so the honest question is whether it is also level with jSO's
    successor.

    PROVENANCE. Transcribed from the authors' own reference implementation,
    ``main.cpp`` in github.com/VladimirStanovov/LSHADE-RSP, not from a prose
    description. What the reference actually does, and where it departs from jSO:

      rank-based selection   ``FitTemp[i] = 3.0*(NInds-i)`` on the fitness-sorted
                             population, fed to a discrete distribution. The
                             discrete distribution normalises, so the 3 cancels
                             exactly and is not a greediness knob -- what the
                             weights actually say is that the best individual is
                             ``NInds`` times as likely as the worst and that the
                             pressure is linear in rank, not exponential. It is
                             kept here only to match the reference line by line;
                             ``rank_index`` omits it for the same reason.
      who is ranked          ``Rand1`` and ``Rand2`` only. ``prand`` is
                             ``Indexes[IntRandom(psizeval)]`` -- uniform inside
                             the top-p subset, exactly as in jSO. Reading the
                             paper's phrase "rank-based selective pressure" as
                             applying to pbest as well would change the algorithm.
      p schedule             ``psize = (p/2)*(1 + FEval/MaxFEval)`` with p = 0.17,
                             so the elite fraction *grows* from 0.085 to 0.17.
                             jSO's shrinks from 0.25 to 0.125. Opposite direction.
      population             ``NPinit = int(75 * D^(2/3))``, N_min = 4, linear
                             reduction. jSO uses ``25*ln(D)*sqrt(D)``.
      memories               H = 5, M_F = 0.3, M_CR = 0.8, and **no frozen last
                             cell** -- jSO freezes one at 0.9/0.9; this does not.
      archive                rate 1.0, and the second difference vector comes from
                             the archive with probability |A|/(|A|+N).
      F, CR, Fw              jSO's schedules unchanged: F capped at 0.7 while
                             progress < 0.6; CR floored at 0.7 then 0.6; Fw =
                             0.7F / 0.8F / 1.2F at progress 0.2 and 0.4.

    ``ENFORCE_DISTINCT`` is the one point not confirmed against the reference: the
    lines quoted above sample r1 and r2 without a distinctness loop, so that is
    what runs by default. The flag exists so validation against the published
    table can test the alternative instead of arguing about it.
    """
    lb, ub = bounds
    N_init = int(75 * dim ** (2.0 / 3.0))
    N_min, H = 4, 5
    P_PARAM, ARC_RATE, K_RANK = 0.17, 1.0, 3.0

    pop_size = N_init
    pop = lb + rng.random((pop_size, dim)) * (ub - lb)
    fitness = np.array([obj_func(ind) for ind in pop])
    fes = pop_size

    M_F = np.full(H, 0.3)
    M_CR = np.full(H, 0.8)
    k_mem = 0
    archive = np.empty((0, dim))

    while fes < max_fes:
        t = fes / max_fes
        order = np.argsort(fitness)
        pop, fitness = pop[order], fitness[order]
        idx = np.arange(pop_size)

        # -- rank-based selective pressure: linear weights, best first --------
        ranks = K_RANK * (pop_size - idx)
        pr = ranks / ranks.sum()

        r = rng.integers(0, H, pop_size)
        CR = np.clip(rng.normal(M_CR[r], 0.1), 0.0, 1.0)
        if t < 0.25:
            CR = np.maximum(CR, 0.7)
        if t < 0.5:
            CR = np.maximum(CR, 0.6)
        F = _sample_F(rng, M_F, r, pop_size)
        F = np.where((t < 0.6) & (F > 0.7), 0.7, F)
        Fw = F * (0.7 if t < 0.2 else 0.8 if t < 0.4 else 1.2)
        Fc, Fwc = F[:, None], Fw[:, None]

        # elite fraction grows over the run, unlike jSO's
        p_num = max(1, int(pop_size * (P_PARAM / 2.0) * (1.0 + t)))
        x_pbest = pop[rng.integers(0, p_num, pop_size)]

        r1 = rng.choice(pop_size, pop_size, p=pr)
        r2_pop = rng.choice(pop_size, pop_size, p=pr)
        if ENFORCE_DISTINCT:
            for _ in range(100):
                bad = (r1 == idx) | (r2_pop == idx) | (r1 == r2_pop)
                if not np.any(bad):
                    break
                r1[bad] = rng.choice(pop_size, int(bad.sum()), p=pr)
                r2_pop[bad] = rng.choice(pop_size, int(bad.sum()), p=pr)

        # the second difference vector: archive with probability |A|/(|A|+N)
        n_arc = len(archive)
        second = pop[r2_pop]
        if n_arc:
            from_arc = rng.random(pop_size) < n_arc / (n_arc + pop_size)
            if np.any(from_arc):
                second = second.copy()
                second[from_arc] = archive[rng.integers(0, n_arc, int(from_arc.sum()))]

        V = pop + Fwc * (x_pbest - pop) + Fc * (pop[r1] - second)

        cross = rng.random((pop_size, dim)) < CR[:, None]
        cross[idx, rng.integers(0, dim, pop_size)] = True
        U = np.where(cross, V, pop)

        low, high = U < lb, U > ub
        U[low] = ((lb + pop) / 2.0)[low]
        U[high] = ((ub + pop) / 2.0)[high]

        n_eval = min(pop_size, max_fes - fes)
        fit_U = np.full(pop_size, np.inf)
        for i in range(n_eval):
            fit_U[i] = obj_func(U[i])
        fes += n_eval

        improved = fit_U < fitness
        if np.any(improved):
            archive = np.vstack([archive, pop[improved]])
            df = fitness[improved] - fit_U[improved]
            w = df / df.sum()
            S_CR, S_F = CR[improved], F[improved]
            den_cr, den_f = np.sum(w * S_CR), np.sum(w * S_F)
            mcr = np.sum(w * S_CR**2) / den_cr if den_cr > 0 else 0.0
            mf = np.sum(w * S_F**2) / den_f if den_f > 0 else M_F[k_mem]
            M_CR[k_mem] = (M_CR[k_mem] + mcr) / 2.0
            M_F[k_mem] = (M_F[k_mem] + mf) / 2.0
            k_mem = (k_mem + 1) % H          # every cell cycles; none is frozen

        accept = fit_U <= fitness
        pop[accept], fitness[accept] = U[accept], fit_U[accept]

        new_size = max(N_min, int(round(N_init + (N_min - N_init) * fes / max_fes)))
        if new_size < pop_size:
            keep = np.argsort(fitness)[:new_size]
            pop, fitness = pop[keep], fitness[keep]
            pop_size = new_size
        arc_max = int(round(ARC_RATE * pop_size))
        if len(archive) > arc_max:
            archive = archive[rng.choice(len(archive), arc_max, replace=False)]


# -- CMA-ES family ---------------------------------------------------------
# L-SHADE-DGR's one measured strength is its eigenbasis crossover on
# ill-conditioned rotated problems. That is precisely CMA-ES's home ground: it
# adapts the full covariance and is invariant to rotation by construction.
# Leaving it out, as the original study did, makes the strength claim untestable.
#
# These are thin wrappers around Nikolaus Hansen's own ``cma`` package rather
# than reimplementations. A faulty reimplementation of a competitor errs in our
# own favour, which is the exact fault this project criticises.
#
# Restarts are capped by the budget, not by a restart count. With restarts=9 a
# plain restart CMA-ES exhausts its restarts and stops early -- measured: 27410
# of a 100000 evaluation budget, i.e. 73% unspent. That would hand our algorithm
# a win it did not earn. cma stops at maxfevals regardless, so a high cap simply
# means the budget governs.
_MAX_RESTARTS = 200


class _BudgetGuard:
    """Caps evaluations at ``max_fes`` exactly; later calls never reach the tracker.

    ``cma.fmin2`` checks ``maxfevals`` between generations, so it overruns by a
    few evaluations (measured: 3011 against a budget of 3000). This guard makes
    the budget exact, so every algorithm gets the same number of evaluations.
    """

    def __init__(self, obj_func, max_fes: int):
        self._obj, self._max_fes = obj_func, int(max_fes)
        self.used = 0
        self._last = 1e30

    def __call__(self, x):
        if self.used >= self._max_fes:
            return self._last          # search is over; keeps the scale sane
        self.used += 1
        self._last = self._obj(x)
        return self._last


def _cma_driver(obj_func, dim, bounds, max_fes, rng, *, restarts, bipop,
                incpopsize=2, diagonal=False, sigma_frac=0.3):
    import cma

    lb, ub = bounds
    guard = _BudgetGuard(obj_func, max_fes)
    opts = {
        "bounds": [lb, ub],
        "maxfevals": max_fes,
        "seed": int(rng.integers(1, 2**31 - 1)),
        "verbose": -9, "verb_log": 0, "verb_disp": 0,   # verb_log=0: writes no files
        "CMA_diagonal": bool(diagonal),
    }
    # A callable x0 is re-evaluated on every restart, so each restart starts from
    # a fresh uniform point rather than repeating the same basin.
    cma.fmin2(guard, lambda: list(rng.uniform(lb, ub, dim)),
              sigma_frac * (ub - lb), options=opts,
              restarts=restarts, incpopsize=incpopsize, bipop=bipop)


def CMAES(obj_func, dim, bounds, max_fes, rng, sigma_frac=0.3):
    """Restart CMA-ES: a fresh uniform start whenever it converges, popsize fixed.

    ``incpopsize=1`` keeps the population constant across restarts. Without it
    this would be IPOP -- ``cma`` defaults to ``incpopsize=2``.
    """
    _cma_driver(obj_func, dim, bounds, max_fes, rng,
                restarts=_MAX_RESTARTS, bipop=False, incpopsize=1,
                sigma_frac=sigma_frac)


def IPOP_CMAES(obj_func, dim, bounds, max_fes, rng, sigma_frac=0.3):
    """IPOP-CMA-ES (Auger & Hansen 2005): population doubles on each restart."""
    _cma_driver(obj_func, dim, bounds, max_fes, rng,
                restarts=_MAX_RESTARTS, bipop=False, sigma_frac=sigma_frac)


def BIPOP_CMAES(obj_func, dim, bounds, max_fes, rng, sigma_frac=0.3):
    """BIPOP-CMA-ES (Hansen 2009): alternates large- and small-population regimes.

    The strongest general-purpose baseline for multimodal and ill-conditioned
    continuous problems, and the one L-SHADE-DGR has to answer for.
    """
    _cma_driver(obj_func, dim, bounds, max_fes, rng,
                restarts=_MAX_RESTARTS, bipop=True, sigma_frac=sigma_frac)


def sepCMAES(obj_func, dim, bounds, max_fes, rng, sigma_frac=0.3):
    """Separable CMA-ES (Ros & Hansen 2008): diagonal covariance, O(D) per sample."""
    _cma_driver(obj_func, dim, bounds, max_fes, rng,
                restarts=_MAX_RESTARTS, bipop=False, incpopsize=1, diagonal=True,
                sigma_frac=sigma_frac)


# ==========================================================================
# 7. THE LINE-UP
# ==========================================================================

#: The algorithm under study.
TARGET = "L-SHADE-DGR"

#: What runs, and what appears in every table. One line-up only: the weak swarm
#: baselines were removed outright, so there is no tier to choose.
ALGORITHMS: dict[str, Callable] = {
    "L-SHADE-DGR": LSHADE_DGR,
    "LSHADE":      LSHADE,
    "jSO":         jSO,
    "BIPOP_CMAES": BIPOP_CMAES,
    "IPOP_CMAES":  IPOP_CMAES,
    "CMAES":       CMAES,
    "sepCMAES":    sepCMAES,
}

#: Required rivals with no implementation here, and none is written from the
#: papers' prose: a rival reconstructed from a description errs in our own
#: favour, which is the exact fault this study criticises. Their authors'
#: reference code was not found.
UNIMPLEMENTED_RIVALS = ("LSHADE-cnEpSin", "NL-SHADE-RSP")

#: Implemented in full, from the authors' own reference, but deliberately not
#: registered: what licenses a rival to appear in a published table is matching
#: that rival's published CEC'2017 medians, and those tables are not in the
#: repository yet. "Implemented" and "validated" are different claims, and
#: conflating them in a warning is how an unvalidated competitor slips into a
#: results table.
UNVALIDATED_RIVALS = ("L-SHADE-RSP",)

#: Everything still missing from the line-up, whatever the reason.
PLANNED_RIVALS = UNIMPLEMENTED_RIVALS + UNVALIDATED_RIVALS


def algorithm_seed(name: str) -> int:
    """A seed component that depends only on the algorithm's name.

    The driver used to seed each algorithm from its position in
    ``sorted(ALGORITHMS)``. That made every algorithm's random stream depend on
    which *other* algorithms happened to be registered: removing a competitor
    silently re-seeded every remaining one, so a resumed run would have mixed two
    seed schemes inside a single results file, and no result recorded before the
    line-up changed could be reproduced after it.

    Deriving the component from the name alone removes that coupling. Adding or
    dropping a competitor now leaves every other competitor's streams untouched,
    which is what makes a partial re-run comparable with an earlier one.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def target_defaults() -> dict:
    """The target's hyperparameters, as ``repr`` strings, for the manifest.

    The driver calls ``ALGORITHMS[name]`` with no keyword arguments, so these
    defaults are not configuration around the algorithm -- they *are* the
    algorithm. Recording them is what lets ``--resume`` refuse to append rows
    produced by different behaviour to a file that says otherwise.
    """
    return {k: repr(v.default)
            for k, v in inspect.signature(LSHADE_DGR).parameters.items()
            if v.default is not inspect.Parameter.empty and k != "logger"}


# ==========================================================================
# 8. THE EXPERIMENT DRIVER
# ==========================================================================
# The protocol -- number of runs, evaluation budget, dimensions -- comes from the
# suite, not from the command line. The original study used 30 runs and 3000*D
# evaluations, which matches no competition and makes its numbers incomparable
# with any published table. Overrides exist but are recorded in the manifest.
#
# ON THE WORKER POOL. An earlier driver created a fresh ``Parallel`` inside the
# function loop, so a 29-function gate built 29 executors and spawned workers 29
# times over. On Windows that exhausted the process resources part-way through:
# the run died at function 17 of 29 with ``OSError: [WinError 1450] insufficient
# system resources``, after 72 minutes. Here every task in the whole study goes
# through **one** pool, created once, and results are consumed as a generator so
# they can be written to disk while the rest are still running. That makes the
# full thread count safe to use, which is the point.

FLUSH_ROWS = 250        # append rows to the CSV this often
FLUSH_CURVES = 1000     # rewrite the curve archive this often (whole-file write)


def _dep_version(module: str) -> str:
    """Version of an installed dependency, or why it could not be read."""
    try:
        import importlib
        return str(getattr(importlib.import_module(module), "__version__", "unknown"))
    except Exception as e:                                      # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def _task(alg_name, fid, label, dim, run_id, max_fes, suite, alg_code):
    """One (algorithm, function, dimension, run). Returns a row and a curve."""
    problem = make_suite_problem(suite, fid, dim)
    tracker = Tracker(problem, max_fes, N_POINTS)
    rng = np.random.default_rng([SEED_BASE, dim, run_id, alg_code])

    t0 = time.perf_counter()
    ALGORITHMS[alg_name](tracker, dim, (problem.lb, problem.ub), max_fes, rng)
    seconds = time.perf_counter() - t0

    curve, error = tracker.finalize()
    row = {"Algorithm": alg_name, "Function": label, "Dimension": dim,
           "Run": run_id, "Error": error, "Seconds": seconds,
           "FES": tracker.fes, "Overrun": tracker.overrun}
    return row, curve


def _generator_mode() -> str:
    """``generator_unordered`` where joblib supports it, else ``generator``.

    Unordered consumption lets a finished task be written while slower ones are
    still running, which keeps memory flat. Results carry their own identity, so
    order is irrelevant.
    """
    for mode in ("generator_unordered", "generator"):
        try:
            Parallel(n_jobs=1, return_as=mode)
            return mode
        except (TypeError, ValueError):
            continue
    return "list"


def cmd_run(args) -> int:
    spec = SUITES[args.suite]
    dims = args.dims or list(spec.dims)
    runs = args.runs or spec.runs
    fes_of = (lambda d: args.fes_per_dim * d) if args.fes_per_dim else spec.max_fes
    out = Path(args.out or f"results_{args.suite}")

    algos = args.algos or list(ALGORITHMS)
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        print(f"[X] unknown algorithms: {unknown}. Available: {list(ALGORITHMS)}")
        return 2

    funcs = [int(f) for f in args.functions] if args.functions else list(spec.function_ids)

    overrides = {k: v for k, v in
                 {"dims": args.dims, "runs": args.runs,
                  "fes_per_dim": args.fes_per_dim,
                  "functions": args.functions, "algos": args.algos}.items()
                 if v is not None}
    if not spec.protocol_verified:
        print(f"[!] {args.suite}: protocol NOT verified against a primary source.")
        print(f"[!] {spec.note}")
        print("[!] Do not quote numbers from this suite until it is verified.")
    if overrides:
        print(f"[!] protocol overridden: {overrides} -- results are NOT comparable "
              f"with published {args.suite} tables.")

    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "curves").mkdir(parents=True, exist_ok=True)
    raw_csv = out / "raw" / "results.csv"

    # -- resume, or refuse to destroy an existing run -----------------------
    done: set[tuple] = set()
    prev_manifest: dict = {}
    if args.resume:
        # Refuse to mix seed schemes. A results file written under a different
        # scheme holds different random streams for the same algorithm, and
        # nothing in it could be reproduced.
        old = out / "manifest.json"
        prev_manifest = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
        prev_version = prev_manifest.get("seed_scheme_version", 1) if prev_manifest else None
        if prev_version is None:
            print(f"[X] --resume needs an existing {old}; there is none. "
                  f"Drop --resume to start this run.")
            return 2
        if prev_version != SEED_SCHEME_VERSION:
            print(f"[X] {raw_csv} was written under seed scheme v{prev_version}, "
                  f"this is v{SEED_SCHEME_VERSION}; the runs are not comparable. "
                  f"Use a fresh --out directory.")
            return 2
        # Refuse to pool cells measured under different run counts. A paired test
        # across a 51-run cell and a 31-run cell is not a paired test, and the
        # mismatch would be invisible in the merged file.
        prev_runs = int(prev_manifest.get("runs", runs))
        if prev_runs != runs:
            print(f"[X] {raw_csv} was measured with {prev_runs} runs per cell, "
                  f"this invocation asks for {runs}. Pooling them would break "
                  f"every paired test. Use a fresh --out directory.")
            return 2
        # Refuse to pool cells measured under different hyperparameters. The two
        # checks above cover the random streams and the protocol; this one covers
        # the algorithm itself. ALGORITHMS[name] is called with no keyword
        # arguments and the raw CSV records no hyperparameters, so without this a
        # changed default would append rows from a *different* algorithm under the
        # same name, and nothing in the file would show it.
        # The benchmark library defines the landscapes. opfunu 1.0.1 and 1.0.4
        # disagree on 13 of the 29 CEC'2017 functions, so rows measured under
        # two versions are not one experiment. The manifest already says two
        # files are only poolable if this line matches; this enforces it.
        prev_opfunu = prev_manifest.get("opfunu")
        now_opfunu = _dep_version("opfunu")
        if prev_opfunu and prev_opfunu != now_opfunu:
            print(f"[X] {raw_csv} was measured with opfunu {prev_opfunu}, this "
                  f"environment has {now_opfunu}. Those define different "
                  f"functions; the rows are not poolable. Use a fresh --out.")
            return 2
        for key, now_v in (("numpy", np.__version__),
                           ("python", sys.version.split()[0])):
            was_v = prev_manifest.get(key)
            if was_v and was_v != now_v:
                print(f"[!] {key} changed since these rows were recorded: "
                      f"{was_v} -> {now_v}. Floating-point results may differ; "
                      f"this is a warning, not a refusal.")

        # COMPARE ONLY WHAT THE MANIFEST RECORDED. An earlier version of this
        # guard compared the whole dict, which made every *new* parameter look
        # like drift: adding a flag that defaults to the recorded behaviour
        # changed `target_defaults()` and the guard then refused to resume its
        # own results file. That is the wrong unit of comparison. A parameter
        # that did not exist when the rows were written cannot have been set
        # differently then; what has to hold is that its default reproduces the
        # recorded behaviour, and that is a statement about behaviour, which the
        # pinned-output tests check directly and a dict comparison never could.
        prev_defaults = prev_manifest.get("target_defaults")
        now_defaults = target_defaults()
        if prev_defaults:
            drift = {k: (v, now_defaults.get(k, "<removed>"))
                     for k, v in prev_defaults.items()
                     if now_defaults.get(k, "<removed>") != v}
            if drift:
                print(f"[X] {TARGET} was measured with different defaults than "
                      f"this code has. Resuming would mix two algorithms under "
                      f"one name.")
                for key, (was, now) in sorted(drift.items()):
                    print(f"      {key}: recorded {was} -> now {now}")
                print(f"    Use a fresh --out directory, or restore the defaults.")
                return 2
            added = sorted(set(now_defaults) - set(prev_defaults))
            if added:
                print(f"[resume] {len(added)} parameter(s) added since these rows "
                      f"were recorded: {', '.join(added)}")
                print(f"[resume] their defaults must reproduce the recorded "
                      f"behaviour; that is pinned by the reproduction tests, not "
                      f"by this manifest.")
        if raw_csv.exists():
            recorded = pd.read_csv(raw_csv, float_precision="round_trip")
            done = set(map(tuple, recorded[["Algorithm", "Function", "Dimension",
                                            "Run"]].to_numpy()))
            print(f"[resume] {len(done)} runs already recorded in {raw_csv}")
    elif raw_csv.exists():
        # An earlier driver deleted this file silently, which is how 72 minutes of
        # compute came within one keystroke of being lost. Deleting is now explicit.
        if not args.fresh:
            print(f"[X] {raw_csv} already holds results. Pass --resume to continue "
                  f"it, or --fresh to discard it and start over.")
            return 2
        raw_csv.unlink()
        for p in (out / "curves").glob("curves_*.npz"):
            p.unlink()
        print(f"[!] --fresh: discarded {raw_csv} and its curves")

    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)
    mode = _generator_mode()

    # -- provenance across invocations ---------------------------------------
    # A gate is built up over several invocations: one dimension at a time, and
    # later a second pass that adds algorithms. The manifest has to describe the
    # whole file, not just the call that happened to write it last, or a reader
    # would see "dims: [50]" sitting next to results for three dimensions. Each
    # invocation is also kept verbatim in ``history`` so the overrides that were
    # in force for any given block stay recoverable.
    labels_now = [spec.label(f) for f in funcs]
    all_dims = sorted(set(prev_manifest.get("dims", [])) | set(dims))
    all_algos = ([a for a in ALGORITHMS if a in set(prev_manifest.get("algorithms", [])) | set(algos)]
                 + [a for a in list(prev_manifest.get("algorithms", [])) + algos
                    if a not in ALGORITHMS])
    all_algos = list(dict.fromkeys(all_algos))
    all_funcs = function_order(set(prev_manifest.get("functions", [])) | set(labels_now))
    all_fes = {int(k): int(v) for k, v in (prev_manifest.get("max_fes_per_dim") or {}).items()}
    all_fes.update({int(d): int(fes_of(d)) for d in dims})
    history = list(prev_manifest.get("history", []))
    if prev_manifest and not history:
        # The existing block was written before ``history`` existed. Reconstruct
        # its entry from the fields it does carry rather than letting a resume
        # silently drop the provenance of everything already in the file.
        history = [{
            "started": prev_manifest.get("started", "unknown"),
            "dims": prev_manifest.get("dims", []),
            "algorithms": prev_manifest.get("algorithms", []),
            "functions": prev_manifest.get("functions", []),
            "jobs": prev_manifest.get("jobs"),
            "resumed": False,
            "protocol_overrides": prev_manifest.get("protocol_overrides", {}),
            "note": "reconstructed on resume; this block predates history tracking",
        }]
    history = history + [{
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dims": dims, "algorithms": algos, "functions": labels_now,
        "jobs": jobs, "resumed": bool(args.resume),
        "protocol_overrides": overrides,
        # The live signature at this invocation. The top-level target_defaults
        # records what the *earliest* rows were measured with and is never
        # rewritten; this is how a later reader sees what the code looked like
        # when each batch of rows was added.
        "target_defaults_now": target_defaults(),
        "opfunu": _dep_version("opfunu"),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }]

    manifest = {
        "started": prev_manifest.get("started", time.strftime("%Y-%m-%d %H:%M:%S")),
        "last_written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "jobs": jobs,
        "numpy": np.__version__, "pandas": pd.__version__,
        # The benchmark library's version is part of the result, not of the
        # environment trivia: opfunu 1.0.1 and 1.0.4 define 13 of the 29 CEC'2017
        # functions differently and 1.0.1 returns NaN for F18 and F30 at D=10. Two
        # result files are only poolable if this line matches in both.
        "opfunu": _dep_version("opfunu"),
        "cma": _dep_version("cma"),
        "scipy": _dep_version("scipy"),
        "suite": args.suite,
        "suite_protocol_verified": spec.protocol_verified,
        "suite_note": spec.note,
        "protocol_overrides": overrides,
        "dims": all_dims, "runs": runs,
        "max_fes_per_dim": all_fes,
        "algorithms": all_algos,
        "functions": all_funcs,
        "history": history,
        "planned_rivals_not_yet_implemented": list(PLANNED_RIVALS),
        "seed_base": SEED_BASE,
        "seed_scheme_version": SEED_SCHEME_VERSION,
        "seed_scheme": "algorithm rng = default_rng([SEED_BASE, dim, run, "
                       "algorithm_seed(name)]), algorithm_seed = blake2b-8(name) "
                       "mod (2**31-1). CEC problems are deterministic, so there is "
                       "no noise stream. The stream does not depend on the worker "
                       "count or on the order results are collected, so --jobs "
                       "cannot change any reported number. Version 1 used the "
                       "algorithm's index in sorted(ALGORITHMS), which re-seeded "
                       "every algorithm whenever the line-up changed; results "
                       "recorded under version 1 are NOT resumable here.",
        "algorithm_seeds": {a: int(algorithm_seed(a)) for a in all_algos},
        # The hyperparameters the target was measured with. The driver calls the
        # algorithm with no keyword arguments, so these defaults ARE the algorithm
        # and belong in the record of what produced these rows. --resume refuses to
        # append to a file whose recorded defaults differ from the running code.
        #
        # This is deliberately NOT refreshed from the live signature. It records
        # what the earliest rows in this file were measured with, and rewriting it
        # would quietly relabel them as having been produced by whatever the code
        # says today -- including flags that did not exist when they were run. The
        # live signature goes into `history` instead, per invocation.
        "target": TARGET,
        "target_defaults": prev_manifest.get("target_defaults") or target_defaults(),
        **({"target_defaults_provenance": prev_manifest["target_defaults_provenance"]}
           if "target_defaults_provenance" in prev_manifest else {}),
        # What each registered algorithm was actually called with, so a name in
        # the CSV can still be resolved to a configuration years later.
        "algorithm_overrides": {
            name: {k: repr(v) for k, v in getattr(fn, "keywords", {}).items()}
            for name, fn in ALGORITHMS.items()
            if getattr(fn, "keywords", None)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    alg_code = {a: algorithm_seed(a) for a in algos}
    total = len(algos) * len(funcs) * len(dims) * runs
    tasks = [(a, fid, spec.label(fid), dim, r, int(fes_of(dim)))
             for dim in dims for fid in funcs for a in algos for r in range(runs)
             if (a, spec.label(fid), dim, r) not in done]

    print(f"[*] suite={args.suite} algorithms={len(algos)} functions={len(funcs)} "
          f"dims={dims} runs={runs}")
    print(f"[*] budget: " + ", ".join(f"{d}D={int(fes_of(d)):,} FES" for d in dims))
    print(f"[*] {total} runs total, {len(tasks)} to do, jobs={jobs} ({mode})")
    if not tasks:
        print("[*] nothing to do")
        return 0

    # -- curves: one archive per dimension, loaded on resume ---------------
    curves = {}
    for d in dims:
        path = out / "curves" / f"curves_{d}D.npz"
        curves[d] = dict(np.load(path)) if (args.resume and path.exists()) else {}

    buf: list[dict] = []
    t_start = time.perf_counter()
    # Progress is reported against THIS invocation's work, not against the file.
    # Counting the rows a resume inherited against a total that only covers the
    # new dimension printed "11853/10353" and a negative ETA, which is worse than
    # no estimate: the ETA is the number the operator plans around.
    n_before, n_this = len(done), 0
    since_curve_flush = 0

    def flush(force=False):
        nonlocal buf, since_curve_flush
        if buf:
            pd.DataFrame(buf)[RAW_COLUMNS].to_csv(
                raw_csv, mode="a", header=not raw_csv.exists(), index=False)
            buf = []
        if force or since_curve_flush >= FLUSH_CURVES:
            for d in dims:
                if curves[d]:
                    # Write beside the target and rename, rather than over it.
                    # savez_compressed truncates first, so a crash part-way
                    # through leaves a corrupt archive -- and this archive is
                    # what every convergence figure reads, for a run that can
                    # take a day. os.replace is atomic on Windows and POSIX.
                    final = out / "curves" / f"curves_{d}D.npz"
                    # The temp name must itself end in .npz: savez appends the
                    # extension when it is missing, so "curves_10D.npz.tmp"
                    # would be written as "curves_10D.npz.tmp.npz" and the
                    # rename below would move a file that does not exist.
                    tmp = final.with_name(f"{final.stem}.tmp.npz")
                    np.savez_compressed(tmp, **curves[d])
                    os.replace(tmp, final)
            since_curve_flush = 0

    payload = (delayed(_task)(a, fid, label, dim, r, mf, args.suite, alg_code[a])
               for a, fid, label, dim, r, mf in tasks)

    try:
        # One pool for the entire study. Workers are created once and reused for
        # every task, which is what keeps Windows from running out of handles.
        with Parallel(n_jobs=jobs, backend="loky", return_as=mode) as parallel:
            for row, curve in parallel(payload):
                buf.append(row)
                curves[row["Dimension"]][
                    f'{row["Algorithm"]}|{row["Function"]}|{row["Run"]}'] = curve
                n_this += 1
                since_curve_flush += 1
                if len(buf) >= FLUSH_ROWS:
                    flush()
                    elapsed = time.perf_counter() - t_start
                    rate = n_this / max(elapsed, 1e-9)
                    eta = (len(tasks) - n_this) / rate if rate > 0 else float("nan")
                    print(f"  {n_this}/{len(tasks)} this run "
                          f"({n_before + n_this} in file) | {rate*60:6.1f} runs/min "
                          f"| ETA {eta/60:6.1f} min", flush=True)
    finally:
        # Whatever happened, everything already computed reaches the disk.
        flush(force=True)

    print(f"[*] finished in {(time.perf_counter()-t_start)/60:.1f} min -> {raw_csv}")
    print(f"[*] next: python {Path(__file__).name} analyze --out {out}")
    return 0


# ==========================================================================
# 9. ANALYSIS
# ==========================================================================
# Separate from the run on purpose: analysis is cheap and gets re-run many times,
# the experiment is expensive and gets run once. A bug in a table must never
# destroy hours of compute.

def function_order(labels):
    """Natural order: F1, F3, F4, ..., F30 -- not the string order F1, F10, F3.

    CEC labels sort wrongly as strings, which would put F10 before F3 in every
    table and invite a misreading.
    """
    def key(lbl):
        m = re.fullmatch(r"F(\d+)", str(lbl))
        return (0, int(m.group(1)), "") if m else (1, 0, str(lbl))
    return sorted(labels, key=key)


def load_results(out: Path) -> pd.DataFrame:
    # float_precision="round_trip" is required: the default CSV parser is off by
    # up to one ULP, which matters when algorithms are separated at 1e-20.
    df = pd.read_csv(out / "raw" / "results.csv", float_precision="round_trip")
    # a resumed run can append a duplicate block; keep the last of each key
    return df.drop_duplicates(subset=["Algorithm", "Function", "Dimension", "Run"],
                              keep="last")


def table_descriptive(df, tdir):
    rows = []
    for (f, d, a), g in df.groupby(["Function", "Dimension", "Algorithm"]):
        rows.append({"Function": f, "Dimension": d, "Algorithm": a,
                     **descriptive(g["Error"].to_numpy())})
    out = pd.DataFrame(rows)
    order = {f: i for i, f in enumerate(function_order(out["Function"].unique()))}
    out = (out.assign(_o=out["Function"].map(order))
              .sort_values(["Dimension", "_o", "median"]).drop(columns="_o"))
    out.to_csv(tdir / "descriptive_stats.csv", index=False)
    return out


def median_pivot(df, dim):
    sel = df[df["Dimension"] == dim]
    piv = sel.pivot_table(index="Function", columns="Algorithm", values="Error",
                          aggfunc="median")
    return piv.reindex(function_order(piv.index))


def success_pivot(df, dim, floor=CEC_ERROR_FLOOR):
    """P(error <= floor) in %.

    On a bimodal function this is the statistic and the median is not: a run
    either finds the global basin and reaches the floor or settles on a local
    one, so the median reports one of the two modes and moves discontinuously as
    soon as the rate crosses 50%.
    """
    sel = df[df["Dimension"] == dim].assign(solved=lambda d: d["Error"] <= floor)
    piv = 100 * sel.pivot_table(index="Function", columns="Algorithm",
                                values="solved", aggfunc="mean")
    return piv.reindex(function_order(piv.index))


#: The official CEC'2017 function classes. Class-level ranks are what the paper
#: reports -- a full 29 x 10 table does not fit a 12-page limit -- and the pass
#: criterion for any mechanism change is stated per class, because a change that
#: recovers a rotated function by sacrificing the hybrid class is not a fix.
FUNCTION_CLASSES: dict[str, tuple[int, ...]] = {
    "Unimodal":          (1, 3),
    "Simple multimodal": (4, 5, 6, 7, 8, 9, 10),
    "Hybrid":            tuple(range(11, 21)),
    "Composition":       tuple(range(21, 31)),
}


def class_of(fid: int) -> str:
    for cname, ids in FUNCTION_CLASSES.items():
        if fid in ids:
            return cname
    return "unclassified"


def class_ranks(df, dim, algos, control=TARGET):
    """Average Friedman rank within each official CEC'2017 class.

    Ranks are taken per function over the median error -- the same statistic the
    omnibus test consumes -- then averaged inside the class.

    Two deliberate refusals, both about not over-claiming:

    *No p-value on a tiny class.* Nemenyi's critical difference grows as
    ``1/sqrt(N)``; at ``N = 2`` it exceeds the entire rank range, so nothing in a
    two-function class can separate. ``separable`` is then False by construction
    and no test is reported.

    *Place counts strict betters, not sort position.* On the unimodal class every
    algorithm that reaches the 1e-8 floor ties at the same average rank, and
    reading a winner off the sort order would invent a first place that the data
    does not contain. ``tied_at_place`` reports how many share the position.
    """
    piv = median_pivot(df, dim)
    present = [a for a in algos if a in piv.columns]
    if not present:
        return pd.DataFrame(), pd.DataFrame()
    piv = piv[present]

    wide, summary = {}, []
    for cname, ids in FUNCTION_CLASSES.items():
        labels = [f"F{i}" for i in ids if f"F{i}" in piv.index]
        sub = piv.loc[labels].dropna()
        if sub.empty:
            continue
        ranks = np.apply_along_axis(sps.rankdata, 1, sub.to_numpy(dtype=float))
        avg = pd.Series(ranks.mean(axis=0), index=sub.columns).sort_values()
        wide[cname] = avg

        n = len(sub)
        try:
            cd = nemenyi_cd(len(present), n)
        except ValueError:
            cd = float("nan")
        gap = float(avg.iloc[1] - avg.iloc[0]) if len(avg) > 1 else float("nan")

        row = {"Class": cname, "n_functions": n,
               "leader": avg.index[0], "leader_rank": float(avg.iloc[0]),
               "gap_1_2": gap, "CD": cd,
               "separable": bool(np.isfinite(cd) and np.isfinite(gap) and gap > cd)}
        if control in avg.index:
            v = float(avg[control])
            arr = avg.to_numpy(dtype=float)
            row["control"] = control
            row["control_rank"] = v
            row["control_place"] = int((arr < v - 1e-12).sum()) + 1
            row["tied_at_place"] = int((np.abs(arr - v) <= 1e-12).sum())
        summary.append(row)

    wide_df = pd.DataFrame(wide).T.reindex(columns=present)
    wide_df.index.name = "Class"
    return wide_df, pd.DataFrame(summary)


def report_class_ranks(df, tdir, control, algos):
    """Write and print the per-class rank tables for every dimension present."""
    for dim in sorted(df["Dimension"].unique()):
        wide, summary = class_ranks(df, dim, algos, control)
        if wide.empty:
            continue
        wide.to_csv(tdir / f"class_ranks_{dim}D.csv")
        summary.to_csv(tdir / f"class_summary_{dim}D.csv", index=False)
        print(f"\n=== {dim}D average Friedman rank by CEC'2017 class "
              f"(1 = best) ===")
        print(wide.round(3).to_string())
        print(f"\n--- {dim}D class summary "
              f"(separable = leader's margin exceeds the Nemenyi CD) ---")
        print(summary.to_string(index=False))
        for _, r in summary.iterrows():
            if r["n_functions"] < 3:
                print(f"    [!] {r['Class']}: {int(r['n_functions'])} functions -- "
                      f"no critical difference is reportable at this block count, "
                      f"so no winner is claimed here.")
            elif not r["separable"]:
                print(f"    [!] {r['Class']}: margin {r['gap_1_2']:.3f} < "
                      f"CD {r['CD']:.3f} -- the leader is NOT statistically "
                      f"separable from the runner-up.")


def stats_per_dimension(df, tdir, control, algos):
    lines, cd_data = [], {}
    for dim in sorted(df["Dimension"].unique()):
        present = [a for a in algos if a in median_pivot(df, dim).columns]
        piv = median_pivot(df, dim)[present].dropna()
        if piv.empty:
            continue
        fr = friedman(piv)
        ranks = fr["avg_ranks"]
        cd_data[dim] = (ranks, fr["N_blocks"])

        lines.append(f"=== {dim}D  ({fr['N_blocks']} functions, "
                     f"{fr['k_algorithms']} algorithms) ===")
        lines.append(f"Friedman chi2 = {fr['chi2']:.4f}, p = {fr['p_chi2']:.4e}")
        lines.append(f"Iman-Davenport F = {fr['iman_davenport_F']:.4f}, "
                     f"p = {fr['p_F']:.4e}")
        try:
            cd = nemenyi_cd(fr["k_algorithms"], fr["N_blocks"], ALPHA)
            lines.append(f"Nemenyi critical difference (alpha={ALPHA}) = {cd:.4f}")
        except ValueError as e:
            lines.append(f"Nemenyi CD unavailable: {e}")
        lines.append("Average ranks (1 = best):")
        lines.append(ranks.to_string())

        ph = friedman_posthoc_holm(ranks, fr["N_blocks"], control)
        ph.to_csv(tdir / f"posthoc_holm_{dim}D.csv", index=False)
        lines.append(f"\nHolm post-hoc, control = {control}:")
        lines.append(ph.to_string(index=False))
        lines.append("")

    (tdir / "friedman_per_dimension.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return cd_data


def pairwise(df, tdir, control, algos):
    """Signed-rank + Mann-Whitney + A12, Holm within cell and over the whole family."""
    competitors = [a for a in algos if a != control]
    rows = []
    for dim in sorted(df["Dimension"].unique()):
        for f in function_order(df["Function"].unique()):
            sel = df[(df["Dimension"] == dim) & (df["Function"] == f)]
            ctl = sel[sel["Algorithm"] == control].sort_values("Run")["Error"].to_numpy()
            if ctl.size == 0:
                continue
            cell = []
            for c in competitors:
                cmp_ = sel[sel["Algorithm"] == c].sort_values("Run")["Error"].to_numpy()
                if cmp_.size != ctl.size:
                    continue
                # A12 with the control first: > 0.5 means the control is better.
                a12 = vargha_delaney_a12(ctl, cmp_)
                cell.append({"Dimension": dim, "Function": f, "Control": control,
                             "Competitor": c,
                             "median_control": float(np.median(ctl)),
                             "median_competitor": float(np.median(cmp_)),
                             "p_signedrank": paired_wilcoxon(ctl, cmp_),
                             "p_mannwhitney": mannwhitney(ctl, cmp_),
                             "A12_control_better": a12,
                             "A12_magnitude": a12_magnitude(a12)})
            if not cell:
                continue
            adj = holm([r["p_signedrank"] for r in cell])
            for r, p in zip(cell, adj):
                r["p_holm_cell"] = p
            rows.extend(cell)

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # multiplicity control over the entire family of tests actually performed
    res["p_holm_family"] = holm(res["p_signedrank"].to_numpy())

    def verdict(r):
        if r["p_holm_cell"] >= ALPHA:
            return "="
        return "+" if r["median_control"] < r["median_competitor"] else "-"

    res["outcome"] = res.apply(verdict, axis=1)
    res.to_csv(tdir / "pairwise_tests.csv", index=False)

    wtl = (res.groupby(["Competitor", "outcome"]).size().unstack(fill_value=0)
           .reindex(columns=["+", "=", "-"], fill_value=0))
    wtl.to_csv(tdir / "win_tie_loss.csv")
    print(f"\n{control} vs competitors (+ win / = tie / - loss), "
          f"Holm within cell, alpha={ALPHA}:")
    print(wtl.to_string())
    return res


def fes_to_target(out: Path, tdir: Path, algos, dims, floor=CEC_ERROR_FLOOR):
    """How fast each algorithm reaches the floor, not just whether it got there.

    WHY THIS TABLE EXISTS. The competition metric is the error at MaxFES, and on
    the unimodal class it saturates: every algorithm but one reaches the 1e-8
    floor on both functions, so the per-function ranks tie and the class rank is
    the same number for seven of eight algorithms. That is a property of the
    measure, not a finding about the algorithms, and no algorithmic change can
    move it. The only honest way to say anything about that class is to measure
    the budget each algorithm spends getting there.

    THIS DOES NOT REPLACE THE PROTOCOL. The competition tables are unchanged and
    remain the basis of every claim; this is reported beside them. Changing the
    headline metric after seeing the headline metric saturate would be fitting
    the measure to the result.

    TWO NUMBERS PER CELL, BECAUSE ONE WOULD LIE. A median budget computed only
    over the runs that reached the floor flatters an algorithm that reaches it
    rarely, so the success rate is reported with it and neither is quoted alone.
    Runs that never reach it are censored, not imputed.

    RESOLUTION IS COARSE AND IS STATED. The convergence curves hold N_POINTS
    checkpoints over the whole budget, so a crossing is located to within
    MaxFES/N_POINTS evaluations -- 1000 at D=10. Good enough to separate
    algorithms that differ by a factor; not good enough to rank near-ties, and
    the column records the step so a reader can see which they are looking at.
    """
    rows = []
    for dim in dims:
        path = out / "curves" / f"curves_{dim}D.npz"
        if not path.exists():
            continue
        max_fes = int(budget_of(out, dim))
        with np.load(path) as z:
            step = None
            per = {}
            for key in z.files:
                alg, fid, _run = key.split("|")
                if alg not in algos:
                    continue
                curve = z[key]
                if step is None:
                    step = max_fes / len(curve)
                hit = np.flatnonzero(curve <= floor)
                per.setdefault((alg, fid), []).append(
                    float((hit[0] + 1) * step) if hit.size else np.nan)
        for (alg, fid), vals in per.items():
            v = np.asarray(vals, dtype=float)
            reached = v[np.isfinite(v)]
            rows.append({"Algorithm": alg, "Function": fid, "Dimension": dim,
                         "runs": int(v.size),
                         "success_rate": float(reached.size / v.size),
                         "median_fes_to_target": (float(np.median(reached))
                                                  if reached.size else np.nan),
                         "resolution_fes": float(step) if step else np.nan})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows)
    t.to_csv(tdir / "fes_to_target.csv", index=False)
    for dim in sorted(t["Dimension"].unique()):
        sub = t[t["Dimension"] == dim]
        piv = sub.pivot_table(index="Function", columns="Algorithm",
                              values="median_fes_to_target")
        piv = piv.reindex(function_order(sub["Function"].unique()))
        piv.to_csv(tdir / f"fes_to_target_{dim}D.csv")
    uni = [f"F{k}" for k in FUNCTION_CLASSES["Unimodal"]]
    u = t[t["Function"].isin(uni)]
    if len(u):
        print("\nFES to reach the 1e-8 floor on the unimodal class, where the "
              "error-at-MaxFES metric saturates (median over the runs that "
              "reached it; success rate in brackets):")
        for dim in sorted(u["Dimension"].unique()):
            s = u[u["Dimension"] == dim]
            agg = s.groupby("Algorithm").agg(
                fes=("median_fes_to_target", "median"),
                sr=("success_rate", "mean")).sort_values("fes")
            print(f"  D={dim} (resolution {s['resolution_fes'].iloc[0]:.0f} FES)")
            for alg, r in agg.iterrows():
                fes = "never" if not np.isfinite(r.fes) else f"{r.fes:,.0f}"
                print(f"    {alg:34s} {fes:>12s}   [{r.sr:.0%}]")
    return t


def _aos_task(fid, dim, run, max_fes, kw):
    """One instrumented run: the per-operator budget share and the outcome."""
    problem = make_suite_problem("cec2017", fid, dim)
    tracker = Tracker(problem, max_fes, N_POINTS)
    log: list = []
    rng = np.random.default_rng([SEED_BASE, dim, run, ABLATE_SALT])
    LSHADE_DGR(tracker, dim, (problem.lb, problem.ub), max_fes, rng,
               logger=log, **kw)
    err = tracker.finalize()[1]
    if not log:
        return None
    # The share the selector actually asked for, averaged over the run. Guarded
    # generations are excluded: there the distribution used is the guard's
    # override, which is not what op_prob holds.
    free = [e for e in log if not e["guard"]]
    if not free:
        return None
    share = np.vstack([np.asarray(e["op_prob"], dtype=float) for e in free]).mean(axis=0)
    return {"Function": f"F{fid}", "Dimension": dim, "Run": run,
            # Carried out of the task rather than inferred from the task
            # order: results come back unordered, so position says nothing
            # about which configuration produced them.
            "leader": bool(1 in kw["OPS"]),
            "Error": err, "generations": len(log),
            "guarded_fraction": 1.0 - len(free) / len(log),
            **{f"share_op{k}": float(share[i]) for i, k in enumerate(kw["OPS"])}}


def aos_correlation(tdir, functions, dim, runs, jobs, fes_per_dim=None,
                    fdir=None):
    """Does the selector spend most where spending costs most?

    THE CENTRAL MECHANISM CLAIM OF THIS STUDY, PLOTTED. The recorded algorithm
    carries three operators and divides the budget between them by immediate
    improvement. One of them -- the leader step -- makes the largest immediate
    gains per trial and loses the run anyway, so a credit rule built on
    immediate improvement cannot see it and hands it a large share precisely
    where it does most damage. That is an assertion about a correlation, so it
    is measured as one.

    WHAT IS ON EACH AXIS, AND WHY BOTH COME FROM THE SAME INVOCATION. The x
    axis is the share of the budget the selector gave the leader step on a
    function, read from the algorithm's own instrumentation. The y axis is how
    much that function's error improves when the operator is removed, measured
    here rather than read from an earlier table, so the two axes cannot drift
    apart across runs or configurations.

    Generations where the diversity guard was active are excluded from the
    share: there the distribution actually sampled is the guard's override, not
    op_prob, and averaging the two together would report a share that was never
    used.
    """
    max_fes = (int(fes_per_dim * dim) if fes_per_dim
               else int(SUITES["cec2017"].max_fes(dim)))
    with_leader = dict(POP_UNCAP=True, OPS=(0, 1, 3))
    without = dict(POP_UNCAP=True, OPS=(0, 3))
    tasks = [(f, dim, r, max_fes, kw)
             for kw in (with_leader, without)
             for f in functions for r in range(runs)]
    print(f"  AOS correlation: 2 configurations x {len(functions)} functions "
          f"x {runs} runs at D={dim}")
    rows = []
    with Parallel(n_jobs=jobs, backend="loky",
                  return_as=_generator_mode()) as parallel:
        for row in parallel(delayed(_aos_task)(*t) for t in tasks):
            if row is not None:
                rows.append(row)
    if not rows:
        print("  [!] no instrumented generations; nothing to correlate")
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d.to_csv(tdir / "aos_runs.csv", index=False)
    on = d[d["leader"]].groupby("Function").agg(
        share_leader=("share_op1", "mean"),
        err_with=("Error", "median"),
        guarded=("guarded_fraction", "mean"))
    off = d[~d["leader"]].groupby("Function")["Error"].median().rename("err_without")
    t = on.join(off).dropna()
    # How much removing the operator buys, in orders of magnitude. The floor
    # keeps a solved function from producing an infinite ratio.
    floor = CEC_ERROR_FLOOR
    t["log10_gain"] = np.log10(np.maximum(t["err_with"], floor)
                               / np.maximum(t["err_without"], floor))
    t = t.reset_index()
    t.to_csv(tdir / "aos_correlation.csv", index=False)
    if len(t) >= 3:
        rho, p = sps.spearmanr(t["share_leader"], t["log10_gain"])
        print(f"  Spearman rho = {rho:.3f}, p = {p:.4g} over {len(t)} functions "
              f"(positive = the more budget the selector gave the operator, the "
              f"more its removal gains)")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(t["share_leader"], t["log10_gain"], s=42,
                       color="#E45756", zorder=3)
            for _, r in t.iterrows():
                ax.annotate(r["Function"], (r["share_leader"], r["log10_gain"]),
                            fontsize=7, xytext=(4, 3),
                            textcoords="offset points")
            ax.axhline(0.0, color="#333", lw=1)
            ax.set_xlabel("Mean budget share given to the leader step")
            ax.set_ylabel("log10( error with / error without )")
            ax.set_title(f"Adaptive operator selection, D={dim}:  "
                         f"Spearman rho = {rho:.3f}, p = {p:.3g}")
            ax.grid(alpha=0.25, zorder=0)
            fig.tight_layout()
            out_png = Path(fdir or tdir) / f"aos_correlation_{dim}D.png"
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_png, dpi=200)
            plt.close(fig)
        except Exception as e:                      # pragma: no cover
            print(f"  [!] the correlation plot was not drawn: {e}")
    else:
        print(f"  [!] only {len(t)} functions paired; no correlation reported")
    return t


def runtime_table(df, tdir):
    rt = df.groupby(["Algorithm", "Dimension"])["Seconds"].agg(["mean", "std"]).reset_index()
    rt.to_csv(tdir / "runtime_seconds.csv", index=False)
    piv = rt.pivot(index="Algorithm", columns="Dimension", values="mean")
    print("\nMean wall-clock seconds per run:")
    print(piv.round(2).to_string())
    return rt


def budget_of(out: Path, dim: int, default: int = 10_000) -> int:
    """FES budget for one dimension, from the manifest.

    The manifest stores ``max_fes_per_dim`` as a mapping of dimension to the
    total budget. An earlier analyzer looked for a ``fes_per_dim`` key that the
    driver never wrote, so every convergence figure silently fell back to
    3000*D and mislabelled its own x-axis.
    """
    mf = out / "manifest.json"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        per_dim = m.get("max_fes_per_dim") or {}
        for key in (str(dim), dim):
            if key in per_dim:
                return int(per_dim[key])
    return int(default * dim)


def figures(df, out, algos, control, fdir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = Path(fdir) if fdir else out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    cdir = out / "curves"

    for dim in sorted(df["Dimension"].unique()):
        path = cdir / f"curves_{dim}D.npz"
        if not path.exists():
            continue
        data = np.load(path)
        total = budget_of(out, dim)
        for f in function_order(df["Function"].unique()):
            plt.figure(figsize=(9, 5.5))
            plotted = False
            for a in algos:
                keys = [k for k in data.files if k.startswith(f"{a}|{f}|")]
                if not keys:
                    continue
                arr = np.vstack([data[k] for k in keys])
                # median, not mean: robust to outlier runs
                med = np.median(arr, axis=0)
                fes_axis = np.linspace(total / len(med), total, len(med))
                plt.plot(fes_axis, np.maximum(med, 1e-300), label=a,
                         linewidth=2.5 if a == control else 1.2)
                plotted = True
            if not plotted:
                plt.close()
                continue
            plt.yscale("log")
            plt.xlabel("Function evaluations (FES)")
            plt.ylabel("Median error  f(x) - f*")
            plt.title(f"{f} ({dim}D)")
            plt.legend(fontsize=8, ncol=2)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fdir / f"Conv_{f}_{dim}D.png", dpi=200)
            plt.close()


def cd_diagram(cd_data, out, alpha=ALPHA, fdir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fdir = Path(fdir) if fdir else out / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    for dim, (ranks, n_blocks) in cd_data.items():
        k = len(ranks)
        try:
            cd = nemenyi_cd(k, n_blocks, alpha)
        except ValueError:
            continue
        fig, ax = plt.subplots(figsize=(9, 0.42 * k + 1.8))
        y = np.arange(k)[::-1]
        ax.barh(y, ranks.to_numpy(), color="#4C78A8", height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(ranks.index)
        best = ranks.min()
        ax.axvline(best, color="#333", lw=1)
        ax.axvline(best + cd, color="#E45756", ls="--", lw=1.4,
                   label=f"critical difference = {cd:.2f}")
        ax.set_xlabel("Average Friedman rank (1 = best)")
        ax.set_title(f"{dim}D  -  Nemenyi CD, alpha={alpha}, N={n_blocks}")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fdir / f"CD_{dim}D.png", dpi=200)
        plt.close(fig)


def cmd_analyze(args) -> int:
    out = Path(args.out or "results_cec2017")
    # A restricted line-up writes into its own directory. `tables/` holds the
    # canonical comparison and is the file the published class ranks are read
    # from; an exploratory `--algos` run must not be able to overwrite it with
    # a different k, because every rank, Holm family and critical difference in
    # there is a function of how many algorithms were compared.
    restricted = bool(args.algos) and not getattr(args, "include_unlisted", False)
    suffix = f"_k{len(args.algos)}" if restricted else ""
    tdir, figdir = out / f"tables{suffix}", out / f"figures{suffix}"
    tdir.mkdir(parents=True, exist_ok=True)

    if not (out / "raw" / "results.csv").exists():
        print(f"[X] no results in {out / 'raw' / 'results.csv'}")
        return 2

    df = load_results(out)
    # --algos is a FILTER, not just an ordering. It used to be the latter: the
    # second line below re-added every name found in the CSV, so once a results
    # file held extra algorithms there was no way to reproduce the original
    # line-up's tables again. That matters because a Friedman rank, a Holm
    # family and a Nemenyi critical difference are all functions of k -- adding
    # four internal variants to a seven-algorithm comparison silently changes
    # every number the paper quotes. Reproducing the k = 7 tables has to stay
    # possible, so an explicit --algos now restricts the data.
    present = [a for a in ALGORITHMS if a in set(df["Algorithm"])]
    present += [a for a in sorted(df["Algorithm"].unique()) if a not in present]
    if args.algos:
        unknown = [a for a in args.algos if a not in set(df["Algorithm"])]
        if unknown:
            print(f"[X] not in the results file: {unknown}. Present: {present}")
            return 2
        algos = list(args.algos)
        if not getattr(args, "include_unlisted", False):
            df = df[df["Algorithm"].isin(algos)].copy()
            print(f"[*] restricted to {len(algos)} algorithms: {', '.join(algos)}")
    else:
        algos = present

    # A DIMENSION ONLY ONE ALGORITHM HAS IS NOT A COMPARISON. Every table below
    # is a statement about a line-up: a Friedman rank, a Holm family and a
    # Nemenyi critical difference are all functions of how many algorithms were
    # compared, and at k = 1 they degenerate rather than fail -- the rank comes
    # out 1.000 in every class, the CD is refused, and what lands on disk is a
    # class-rank table showing a clean sweep that means nothing. That file is a
    # trap for whoever reads it next, so the dimension never reaches it.
    # Dropped here rather than by editing the results file: the rows are real
    # measurements and stay where they were recorded.
    have = df.groupby("Dimension")["Algorithm"].apply(lambda s: set(s.unique()))
    full = [int(d) for d, got in have.items() if got >= set(algos)]
    dropped = sorted(set(int(d) for d in have.index) - set(full))
    if dropped:
        print(f"[!] dimensions {dropped} dropped: not every one of the "
              f"{len(algos)} algorithms has rows there, so no rank, Holm family "
              f"or critical difference can include them. The rows are untouched "
              f"in raw/results.csv.")
        df = df[df["Dimension"].isin(full)].copy()
    if not full:
        print(f"[X] no dimension has rows for all of {algos}")
        return 2

    if args.control not in algos:
        print(f"[X] control {args.control!r} not present. Available: {algos}")
        return 2

    over = df[df["Overrun"] > 0]
    if not over.empty:
        print(f"[!] {len(over)} runs exceeded the FES budget -- investigate "
              f"before reporting:")
        print(over.groupby("Algorithm")["Overrun"].max().to_string())

    incomplete = df.groupby(["Algorithm", "Function", "Dimension"]).size()
    expected = int(incomplete.max()) if len(incomplete) else 0
    short = incomplete[incomplete != expected]
    if len(short):
        print(f"[!] {len(short)} (algorithm, function, dimension) cells have fewer "
              f"than {expected} runs; paired tests skip them:")
        print(short.to_string())

    table_descriptive(df, tdir)
    for dim in sorted(df["Dimension"].unique()):
        median_pivot(df, dim).to_csv(tdir / f"median_error_{dim}D.csv")
        success_pivot(df, dim).round(2).to_csv(tdir / f"success_rate_{dim}D.csv")
    cd_data = stats_per_dimension(df, tdir, args.control, algos)
    report_class_ranks(df, tdir, args.control, algos)
    pairwise(df, tdir, args.control, algos)
    runtime_table(df, tdir)
    fes_to_target(out, tdir, algos,
                  sorted(int(d) for d in df["Dimension"].unique()))
    if not args.no_figures:
        figures(df, out, algos, args.control, fdir=figdir)
        cd_diagram(cd_data, out, fdir=figdir)
    print(f"\n[*] tables -> {tdir}")
    if not args.no_figures:
        print(f"[*] figures -> {figdir}")
    if UNIMPLEMENTED_RIVALS:
        print(f"[!] the line-up is incomplete: "
              f"{', '.join(UNIMPLEMENTED_RIVALS)} are required and not "
              f"implemented here.")
    if UNVALIDATED_RIVALS:
        print(f"[!] {', '.join(UNVALIDATED_RIVALS)} is implemented but not "
              f"validated against its published CEC'2017 table, so it is not "
              f"in the line-up.")
    return 0


# ==========================================================================
# 10. THE COMPONENT DIAGNOSTIC
# ==========================================================================
# Why does L-SHADE-DGR lose on F4 and F5, and what is the cost of the fix?
#
# The losses are not spread evenly -- they are concentrated, and two of them are
# qualitative rather than marginal: F4 (Shifted Rotated Rosenbrock) and F5
# (Shifted Rotated Rastrigin). Theory says where to look: DE/rand/1/bin is
# invariant under diagonal affine maps but *not* under rotation, because binomial
# crossover selects coordinates in a fixed basis while a rotation mixes them.
# CMA-ES is fully affine invariant. F4 and F5 are both f(R(x - o)). So the losses
# are where the theory predicts.
#
# The eigen crossover exists precisely to restore rotation invariance
# approximately, so the question is not "what is missing" but "which of our own
# components is preventing the one that should help". This answers that by
# flipping one flag at a time and by reading the algorithm's own instrumentation.
#
# TWO MEASUREMENTS, AND WHY BOTH ARE NEEDED.
#   1. Error per configuration: which flag, flipped, recovers F4 and F5.
#   2. The cost of that flip elsewhere. A component that hurts on F4 was added
#      for a reason. A fix that recovers F4 and loses the hybrid class is not a
#      fix, so the winning functions are measured in the same run.
# Running it on the losing functions alone would produce a confident wrong answer.

DIAG_BASELINE = "default"

#: One flag flipped per configuration, so any difference is attributable to it.
DIAG_CONFIGS: dict[str, Callable] = {
    DIAG_BASELINE:         lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r),
    # -- the eigen crossover: present, but is it reaching the run? ----------
    "P_EIG=0.9 fixed":     lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, P_EIG=0.9, ADAPT_EIG=False),
    "P_EIG=0.5 fixed":     lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, P_EIG=0.5, ADAPT_EIG=False),
    "P_EIG=0 (no eigen)":  lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, P_EIG=0.0, ADAPT_EIG=False),
    "no eigen gate":       lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, EIGEN_GATE=False),
    # -- our own additions, each suspect -----------------------------------
    "no diversity guard":  lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, DIV_GUARD=False),
    "guard thresh 1e-6":   lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, DIV_THRESH=1e-6),
    "no restart":          lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, RESTART=False),
    "OPS=(0,) pure pbest": lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, OPS=(0,)),
    "OPS=(0,1) no levy":   lambda o, d, b, m, r: LSHADE_DGR(o, d, b, m, r, OPS=(0, 1)),
    # POP_FACTOR alone is inert from D = 42, where POP_MAX clips 18*D back to the
    # baseline's own 500. Run as a bare factor change this cell was the baseline
    # under a different seed, so it read as noise rather than as a no-op. The cap
    # has to come off with it for the cell to mean what its name says.
    "POP_FACTOR=18 uncapped": lambda o, d, b, m, r: LSHADE_DGR(
        o, d, b, m, r, POP_FACTOR=18, POP_UNCAP=True),
    # -- references ---------------------------------------------------------
    "jSO":                 lambda o, d, b, m, r: jSO(o, d, b, m, r),
    "BIPOP_CMAES":         lambda o, d, b, m, r: BIPOP_CMAES(o, d, b, m, r),
    "IPOP_CMAES":          lambda o, d, b, m, r: IPOP_CMAES(o, d, b, m, r),
}

#: Measured from the gate. Both groups are required: see the note above.
DIAG_LOSING = {4: "Rosenbrock (rot)", 5: "Rastrigin (rot)",
               7: "Lunacek Bi-Rastrigin", 24: "Composition 4", 25: "Composition 5"}
DIAG_WINNING = {11: "Hybrid 1", 14: "Hybrid 4", 16: "Hybrid 6",
                19: "Hybrid 9", 10: "Schwefel (rot)"}


def _diag_task(cfg, fid, dim, run, max_fes):
    problem = make_suite_problem("cec2017", fid, dim)
    tracker = Tracker(problem, max_fes, 10)
    rng = np.random.default_rng([SEED_BASE, dim, run, algorithm_seed(cfg)])
    t0 = time.perf_counter()
    DIAG_CONFIGS[cfg](tracker, dim, (problem.lb, problem.ub), max_fes, rng)
    return {"Config": cfg, "Function": f"F{fid}", "Run": run,
            "Error": tracker.finalize()[1], "Seconds": time.perf_counter() - t0,
            "FES": tracker.fes, "Overrun": tracker.overrun}


def _diag_internals(fid, dim, max_fes, runs, **kw):
    """Read the algorithm's own instrumentation on one function.

    Answers, with numbers rather than argument: how often is the eigen gate open,
    where does adaptation drive P_EIG, and how much of the run does the diversity
    guard spend overriding the operator distribution.
    """
    rows = []
    for run in range(runs):
        log = []
        problem = make_suite_problem("cec2017", fid, dim)
        tracker = Tracker(problem, max_fes, 10)
        LSHADE_DGR(tracker, dim, (problem.lb, problem.ub), max_fes,
                   np.random.default_rng([SEED_BASE, dim, run,
                                          algorithm_seed(DIAG_BASELINE)]),
                   logger=log, **kw)
        if not log:
            continue
        gen = pd.DataFrame([{k: v for k, v in e.items()
                             # This whitelist drops anything not named here, so a
                             # new logger field has to be added in three places:
                             # the payload, this filter, and the aggregation below.
                             if k in ("t", "P_EIG", "eig_ok", "guard", "diversity",
                                      "pop_size", "n_eig_used", "improved",
                                      "n_samples", "cond_hat", "log10_cond_adj",
                                      "mp_valid", "eig_branch")}
                            for e in log])
        rows.append({
            "generations": len(gen),
            "eig gate open %": 100 * gen["eig_ok"].mean(),
            "guard active %": 100 * gen["guard"].mean(),
            "P_EIG mean": gen["P_EIG"].mean(),
            "P_EIG final": gen["P_EIG"].iloc[-1],
            "P_EIG min": gen["P_EIG"].min(),
            "P_EIG max": gen["P_EIG"].max(),
            "diversity final": gen["diversity"].iloc[-1],
            "improved/gen": gen["improved"].mean(),
            # Only meaningful under the condition rules; NaN under the others,
            # which is the honest reading rather than a zero.
            "log10 cond_adj mean": gen["log10_cond_adj"].mean(),
            "MP valid %": 100 * gen["mp_valid"].mean(),
            "condition branch %": 100 * (gen["eig_branch"] == "condition").mean(),
            "degenerate branch %": 100 * (gen["eig_branch"] == "degenerate").mean(),
        })
    return pd.DataFrame(rows).mean().to_frame().T if rows else pd.DataFrame()


def cmd_diagnose(args) -> int:
    dim, runs = args.dim, args.runs
    max_fes = args.fes_per_dim * dim
    funcs = dict(DIAG_LOSING) if args.quick else {**DIAG_LOSING, **DIAG_WINNING}
    out = Path(args.out or "results_diagnose")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)

    if args.quick:
        print("[!] --quick measures only the functions we lose. A component that")
        print("[!] recovers F4 and destroys the hybrid class would look like a fix.")

    total = len(DIAG_CONFIGS) * len(funcs) * runs
    print(f"[*] {len(DIAG_CONFIGS)} configurations x {len(funcs)} functions x "
          f"{runs} runs = {total} runs at {max_fes:,} FES, D={dim}, jobs={jobs}")

    t0 = time.perf_counter()
    mode = _generator_mode()
    payload = (delayed(_diag_task)(c, f, dim, r, max_fes)
               for c in DIAG_CONFIGS for f in funcs for r in range(runs))
    rows = []
    with Parallel(n_jobs=jobs, backend="loky", return_as=mode) as parallel:
        for row in parallel(payload):
            rows.append(row)
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{total}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "raw_diagnose.csv", index=False)
    print(f"[*] {len(df)} runs in {(time.perf_counter()-t0)/60:.1f} min")

    bad = df[df["Overrun"] != 0]
    if len(bad):
        print(f"[X] {len(bad)} runs exceeded their budget -- the tables below "
              f"are not usable")
        print(bad.head().to_string())
        return 1

    order = [f"F{f}" for f in funcs]
    piv = df.pivot_table(index="Config", columns="Function",
                         values="Error", aggfunc="median")[order]
    piv = piv.reindex([c for c in DIAG_CONFIGS if c in piv.index])

    print("\n" + "=" * 100)
    print(f"MEDIAN ERROR  ({runs} runs, D={dim}, {max_fes:,} FES)")
    print(f"  losing  : {'  '.join(f'F{k}={v}' for k, v in DIAG_LOSING.items())}")
    if not args.quick:
        print(f"  winning : {'  '.join(f'F{k}={v}' for k, v in DIAG_WINNING.items())}")
    print("=" * 100)
    print(piv.to_string(float_format=lambda x: f"{x:.4e}"))
    piv.to_csv(out / "tables" / "median_by_config.csv")

    df["solved"] = df["Error"] <= CEC_ERROR_FLOOR
    rate = 100 * df.pivot_table(index="Config", columns="Function",
                                values="solved", aggfunc="mean")[order]
    rate = rate.reindex([c for c in DIAG_CONFIGS if c in rate.index])
    if rate.to_numpy().max() > 0:
        print("\n" + "=" * 100)
        print(f"SUCCESS RATE  P(error <= {CEC_ERROR_FLOOR:g}) in %   ({runs} runs)")
        print("  On a bimodal function this is the statistic; the median is not.")
        print("=" * 100)
        print(rate.round(1).to_string())
        rate.round(2).to_csv(out / "tables" / "success_rate_by_config.csv")

    # -- each configuration against the baseline, on both groups ------------
    print("\n" + "=" * 100)
    print(f"EFFECT OF EACH FLAG vs '{DIAG_BASELINE}'  "
          f"(paired Wilcoxon, Vargha-Delaney A12)")
    print("  A12 > 0.5 means the configuration is BETTER than the baseline")
    print("=" * 100)
    lines = []
    for cfg in DIAG_CONFIGS:
        if cfg == DIAG_BASELINE or cfg not in set(df["Config"]):
            continue
        for fid in funcs:
            f = f"F{fid}"

            def errors(which, _f=f):
                sel = df[(df.Config == which) & (df.Function == _f)]
                return sel.sort_values("Run")["Error"].to_numpy()

            a, b = errors(cfg), errors(DIAG_BASELINE)
            if len(a) != len(b) or len(a) == 0:
                continue
            lines.append({"Config": cfg, "Function": f,
                          "group": "losing" if fid in DIAG_LOSING else "winning",
                          "median_cfg": float(np.median(a)),
                          "median_base": float(np.median(b)),
                          "A12_cfg_better": vargha_delaney_a12(a, b),
                          "p_wilcoxon": paired_wilcoxon(a, b)})
    eff = pd.DataFrame(lines)
    eff.to_csv(out / "tables" / "effect_vs_baseline.csv", index=False)

    sig = eff[eff.p_wilcoxon < ALPHA]
    # A12 comes from vargha_delaney_a12(cfg, baseline), so A12 > 0.5 means the
    # configuration is better. Getting this backwards swaps both tables, which is
    # exactly what an earlier version of this diagnostic did.
    better = sig[sig.A12_cfg_better > 0.5]
    worse = sig[sig.A12_cfg_better < 0.5]
    print("\n-- flags that HELP (significantly better than the baseline)")
    print(better.sort_values(["group", "A12_cfg_better"], ascending=[True, False])
          .to_string(index=False) if len(better) else "  none")
    print("\n-- flags that HURT (significantly worse than the baseline)")
    print(worse.sort_values(["group", "A12_cfg_better"]).to_string(index=False)
          if len(worse) else "  none")

    print("\n-- net effect per flag: how many functions it helps vs hurts, by group")
    net = (sig.assign(dir=np.where(sig.A12_cfg_better > 0.5, "helps", "hurts"))
              .groupby(["Config", "group", "dir"]).size().unstack(fill_value=0))
    print(net.to_string() if len(net) else "  nothing reached significance")

    # -- the algorithm's own instrumentation -------------------------------
    print("\n" + "=" * 100)
    print("INTERNALS on the two functions that matter (baseline configuration)")
    print("=" * 100)
    ins = []
    for fid in ([4, 5] + ([11] if not args.quick else [])):
        row = _diag_internals(fid, dim, max_fes, min(runs, 5))
        if len(row):
            row.insert(0, "Function", f"F{fid}")
            ins.append(row)
    if ins:
        tab = pd.concat(ins, ignore_index=True)
        print(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        tab.to_csv(out / "tables" / "internals.csv", index=False)

    (out / "manifest.json").write_text(json.dumps({
        "dim": dim, "runs": runs, "max_fes": max_fes, "jobs": jobs,
        "configs": list(DIAG_CONFIGS), "losing": DIAG_LOSING, "winning": DIAG_WINNING,
        "seed_scheme": "default_rng([SEED_BASE, dim, run, algorithm_seed(config)])",
        "a12_orientation": "vargha_delaney_a12(cfg, baseline); > 0.5 means the "
                           "configuration is better",
        "note": "diagnostic, not the competition protocol; 51 runs are needed "
                "before any of this is quoted as a result",
    }, indent=2), encoding="utf-8")
    print(f"\n[ok] tables -> {out}/tables")
    print(f"[!] This is a diagnostic at {runs} runs. Nothing here is a result "
          f"until it is re-measured at {SUITES['cec2017'].runs}.")
    return 0


# ==========================================================================
# 11. THE FACTORIAL ABLATION
# ==========================================================================
# The diagnostic above flips one flag at a time. That cannot answer the question
# this study now faces, and the reason is mechanical: the diversity guard freezes
# the eigen channel, so "decouple the channels" and "replace the P_EIG rule" are
# not separable one at a time -- switching the rule while the channel is still
# frozen measures the freeze, not the rule. The design has to be factorial.
#
# THREE FACTORS, NOT FOUR. The plan this implements specifies
# {cap} x {decouple} x {P_EIG rule} x {selection} = 2 x 2 x 4 x 2 = 32 cells.
# Twelve of those are the same run twice: under condition_mp, condition_raw and
# fixed the eigen probability is computed outside the credit block, or never, so
# CHANNEL_DECOUPLE cannot reach it. Measured on this machine, the redundant
# cells would cost about 17 wall hours for no information.
#
# That is a claim about the code, so it is proved rather than believed --
# test_decoupling_is_a_no_op_unless_the_rule_is_success asserts the generator
# state, the error, the curve and every logged field are identical, and carries
# a negative control. If it ever fails, the twelve cells go back in.
#
# WHAT THIS CANNOT CONCLUDE. It selects a finalist. It cannot decide adoption:
# the binding criterion is the Hybrid class rank at D=50 *against the rivals*
# together with the D=10 first place, and neither this function subset nor this
# pair of dimensions contains them. That is measured on the full gate
# afterwards, and the criterion is not weakened to let a mechanism through.

#: A fixed stream component, shared by every cell: common random numbers, so a
#: difference in outcome is attributable to the flags rather than to
#: initialisation luck and the paired tests below are genuinely paired. Cells
#: that size their population the same way therefore start from the identical
#: initial population; the capped and uncapped arms draw the same numbers but
#: lay out more of them above D = 42, which is exactly what that factor varies.
#: The component diagnostic keeps its own per-config seeding, so nothing already
#: measured with it changes.
ABLATE_SALT = algorithm_seed("ablation-crn-v1")

#: The factor levels. Each is the keyword override that produces it, so the
#: baseline is the empty override and the cells are generated rather than listed
#: -- a hand-written list of 20 is where a typo becomes a silent third factor.
ABLATE_FACTORS: dict[str, dict[str, dict]] = {
    "pop": {
        "capped":            {},
        "uncapped":          {"POP_UNCAP": True},
    },
    "eig": {
        "success-coupled":   {},
        "success-decoupled": {"CHANNEL_DECOUPLE": True},
        "condition_mp":      {"P_EIG_RULE": "condition_mp"},
        "condition_raw":     {"P_EIG_RULE": "condition_raw"},
        # The one fixed value the diagnostic measured as helping at D=10, kept
        # as the reference arm: it bounds how much the adaptation is worth.
        "fixed0.9":          {"P_EIG_RULE": "fixed", "P_EIG": 0.9},
    },
    "sel": {
        "uniform":           {},
        "rank":              {"SELECTION": "rank"},
    },
}


def _ablate_cells() -> dict[str, dict]:
    """The full crossing of the factor levels, as name -> keyword overrides."""
    cells = {}
    for pop, a in ABLATE_FACTORS["pop"].items():
        for eig, b in ABLATE_FACTORS["eig"].items():
            for sel, c in ABLATE_FACTORS["sel"].items():
                cells[f"{pop} | {eig} | {sel}"] = {**a, **b, **c}
    return cells


ABLATE_CONFIGS: dict[str, dict] = _ablate_cells()

#: All flags off: the algorithm exactly as results_cec2017/ measured it.
ABLATE_BASELINE = "capped | success-coupled | uniform"

#: The hybrid class, which is where the losses are, plus the two simple
#: multimodals that a fix could quietly break. Measuring only the functions we
#: lose would produce a confident wrong answer, the same trap the component
#: diagnostic documents.
ABLATE_FUNCTIONS = (4, 5) + tuple(range(11, 21))
ABLATE_DIMS = (30, 50)

ABLATE_COLUMNS = ["Config", "Function", "Dimension", "Run", "Error", "Seconds",
                  "FES", "Overrun"]


def ablate_factor_of(cfg: str) -> dict:
    """Split a cell name back into its factor levels, for the marginal table."""
    pop, eig, sel = (part.strip() for part in cfg.split("|"))
    return {"pop": pop, "eig": eig, "sel": sel}


def _ablate_task(cfg, fid, dim, run, max_fes):
    problem = make_suite_problem("cec2017", fid, dim)
    tracker = Tracker(problem, max_fes, 10)
    rng = np.random.default_rng([SEED_BASE, dim, run, ABLATE_SALT])
    t0 = time.perf_counter()
    LSHADE_DGR(tracker, dim, (problem.lb, problem.ub), max_fes, rng,
               **ABLATE_CONFIGS[cfg])
    return {"Config": cfg, "Function": f"F{fid}", "Dimension": int(dim),
            "Run": int(run), "Error": tracker.finalize()[1],
            "Seconds": time.perf_counter() - t0, "FES": tracker.fes,
            "Overrun": tracker.overrun}


def ablate_report(df: pd.DataFrame, tdir: Path, runs: int) -> None:
    """Every table the ablation produces. Separated from the run on purpose."""
    cfgs = [c for c in ABLATE_CONFIGS if c in set(df["Config"])]
    order = function_order(df["Function"].unique())
    hybrid = [f for f in order if 11 <= int(f[1:]) <= 20]

    # -- 1. medians, split by group so a fix that trades one for the other shows
    for dim in sorted(df["Dimension"].unique()):
        sub = df[df["Dimension"] == dim]
        piv = sub.pivot_table(index="Config", columns="Function",
                              values="Error", aggfunc="median")[order]
        piv = piv.reindex([c for c in cfgs if c in piv.index])
        # Summary columns only for the groups actually present: a partial run,
        # or a smoke test on one function, must still produce a table.
        simple = [f for f in ("F4", "F5") if f in piv.columns]
        summary = 0
        for label, members in (("F4/F5 median", simple), ("hybrid median", hybrid)):
            if members:
                piv.insert(0, label, piv[members].median(axis=1))
                summary += 1
        piv.to_csv(tdir / f"ablate_median_{dim}D.csv")
        print(f"\n{'=' * 100}\nMEDIAN ERROR, D={dim} ({runs} runs)\n{'=' * 100}")
        shown = piv.iloc[:, :summary] if summary else piv
        print(shown.to_string(float_format=lambda x: f"{x:.4e}"))

    # -- 2. Friedman across the cells, per dimension
    #    No CD diagram: nemenyi_cd is defined for k <= 12 and there are 20 cells.
    #    Saying so is better than omitting the diagram silently.
    lines = []
    for dim in sorted(df["Dimension"].unique()):
        sub = df[df["Dimension"] == dim]
        piv = sub.pivot_table(index="Function", columns="Config",
                              values="Error", aggfunc="median")
        present = [c for c in cfgs if c in piv.columns]
        piv = piv[present].dropna()
        if piv.empty or len(present) < 3:
            continue
        fr = friedman(piv)
        lines += [f"D={dim}: N={fr['N_blocks']} functions, k={fr['k_algorithms']} cells",
                  f"  chi2={fr['chi2']:.4f} p={fr['p_chi2']:.3e}  "
                  f"Iman-Davenport F={fr['iman_davenport_F']:.4f} p={fr['p_F']:.3e}",
                  "  average rank (1 = best):",
                  fr["avg_ranks"].to_string(), ""]
        print(f"\n{'=' * 100}\nFRIEDMAN OVER THE {len(present)} CELLS, D={dim}"
              f"\n{'=' * 100}")
        print(fr["avg_ranks"].to_string())
        print(f"  [!] no critical-difference diagram: Nemenyi's CD is tabulated "
              f"for k <= 12 and this design has {len(present)} cells. Holm below "
              f"is the post-hoc.")
    (tdir / "ablate_friedman.txt").write_text("\n".join(lines), encoding="utf-8")

    # -- 3. each cell against the baseline, Holm within cell and over the family
    rows = []
    for dim in sorted(df["Dimension"].unique()):
        for f in order:
            sel = df[(df["Dimension"] == dim) & (df["Function"] == f)]
            base = sel[sel["Config"] == ABLATE_BASELINE].sort_values("Run")["Error"].to_numpy()
            if base.size == 0:
                continue
            cell = []
            for c in cfgs:
                if c == ABLATE_BASELINE:
                    continue
                arr = sel[sel["Config"] == c].sort_values("Run")["Error"].to_numpy()
                if arr.size != base.size:
                    continue
                # A12 with the cell first, so > 0.5 means the CELL is better.
                # Getting this backwards swaps every verdict in the table, which
                # is what an earlier version of the component diagnostic did.
                a12 = vargha_delaney_a12(arr, base)
                cell.append({"Dimension": dim, "Function": f, "Config": c,
                             "group": "hybrid" if f in hybrid else "simple multimodal",
                             "median_cfg": float(np.median(arr)),
                             "median_base": float(np.median(base)),
                             "p_signedrank": paired_wilcoxon(arr, base),
                             "A12_cfg_better": a12,
                             "A12_magnitude": a12_magnitude(a12)})
            if not cell:
                continue
            for r, p in zip(cell, holm([r["p_signedrank"] for r in cell])):
                r["p_holm_cell"] = p
            rows.extend(cell)

    eff = pd.DataFrame(rows)
    if eff.empty:
        print("\n[!] no cell could be compared against the baseline")
        return
    eff["p_holm_family"] = holm(eff["p_signedrank"].to_numpy())
    eff["outcome"] = np.where(
        eff["p_holm_cell"] >= ALPHA, "=",
        np.where(eff["A12_cfg_better"] > 0.5, "+", "-"))
    eff.to_csv(tdir / "ablate_effect_vs_baseline.csv", index=False)

    print(f"\n{'=' * 100}\nEACH CELL vs '{ABLATE_BASELINE}'"
          f"\n  + better / = no difference / - worse, Holm within cell, "
          f"alpha={ALPHA}\n  A12 > 0.5 means the CELL is better"
          f"\n{'=' * 100}")
    for dim in sorted(eff["Dimension"].unique()):
        wtl = (eff[eff["Dimension"] == dim]
               .groupby(["Config", "group", "outcome"]).size()
               .unstack(fill_value=0).reindex(columns=["+", "=", "-"], fill_value=0))
        print(f"\n-- D={dim}")
        print(wtl.to_string())
        wtl.to_csv(tdir / f"ablate_win_tie_loss_{dim}D.csv")

    # -- 4. the marginal effect of each factor, which is why this is factorial
    #    Each level's mean rank over the other factors: an interaction between
    #    the cap and decoupling is invisible one flag at a time.
    marg = []
    for dim in sorted(df["Dimension"].unique()):
        sub = df[df["Dimension"] == dim]
        piv = sub.pivot_table(index="Function", columns="Config",
                              values="Error", aggfunc="median")
        present = [c for c in cfgs if c in piv.columns]
        piv = piv[present].dropna()
        if piv.empty:
            continue
        ranks = pd.DataFrame(
            np.apply_along_axis(sps.rankdata, 1, piv.to_numpy(dtype=float)),
            index=piv.index, columns=piv.columns)
        mean_rank = ranks.mean(axis=0)
        for factor in ABLATE_FACTORS:
            for level in ABLATE_FACTORS[factor]:
                members = [c for c in present
                           if ablate_factor_of(c)[factor] == level]
                if members:
                    marg.append({"Dimension": dim, "factor": factor,
                                 "level": level, "cells": len(members),
                                 "mean_rank": float(mean_rank[members].mean())})
    mt = pd.DataFrame(marg)
    mt.to_csv(tdir / "ablate_marginal_effects.csv", index=False)
    print(f"\n{'=' * 100}\nMARGINAL EFFECT OF EACH FACTOR"
          f"\n  mean rank over the other factors, 1 = best of the "
          f"{len(cfgs)} cells\n{'=' * 100}")
    for dim in sorted(mt["Dimension"].unique()):
        print(f"\n-- D={dim}")
        print(mt[mt["Dimension"] == dim]
              .pivot(index=["factor", "level"], columns=[], values="mean_rank")
              .to_string(float_format=lambda x: f"{x:.3f}"))


def cmd_ablate(args) -> int:
    out = Path(args.out or "results_ablate")
    tdir, raw_dir = out / "tables", out / "raw"
    tdir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / "results.csv"

    dims = args.dims or list(ABLATE_DIMS)
    runs = args.runs if args.runs else SUITES["cec2017"].runs
    funcs = [int(f) for f in args.functions] if args.functions else list(ABLATE_FUNCTIONS)
    cfgs = args.configs or list(ABLATE_CONFIGS)
    unknown = [c for c in cfgs if c not in ABLATE_CONFIGS]
    if unknown:
        print(f"[X] unknown configurations: {unknown}")
        print(f"    available:\n      " + "\n      ".join(ABLATE_CONFIGS))
        return 2
    fes_of = ((lambda d: args.fes_per_dim * d) if args.fes_per_dim
              else SUITES["cec2017"].max_fes)
    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)

    done: set[tuple] = set()
    prev: dict = {}
    if args.resume:
        old = out / "manifest.json"
        prev = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
        if not prev:
            print(f"[X] --resume needs an existing {old}; there is none.")
            return 2
        prev_runs = int(prev.get("runs", runs))
        if prev_runs != runs:
            print(f"[X] {raw_csv} holds {prev_runs} runs per cell, this asks for "
                  f"{runs}. Pooling them would break every paired test.")
            return 2
        prev_defaults = prev.get("target_defaults")
        if prev_defaults is not None and prev_defaults != target_defaults():
            print(f"[X] {raw_csv} was measured with different algorithm defaults; "
                  f"resuming would mix two algorithms in one file.")
            return 2
        if raw_csv.exists():
            rec = pd.read_csv(raw_csv, float_precision="round_trip")
            done = set(map(tuple, rec[["Config", "Function", "Dimension",
                                       "Run"]].to_numpy()))
            print(f"[resume] {len(done)} runs already recorded in {raw_csv}")
    elif raw_csv.exists():
        print(f"[X] {raw_csv} already holds results. Pass --resume to continue "
              f"it, or --fresh to discard it and start over.")
        return 2
    if args.fresh and raw_csv.exists():
        raw_csv.unlink()
        done = set()

    tasks = [(c, fid, dim, r, int(fes_of(dim)))
             for dim in dims for fid in funcs for c in cfgs for r in range(runs)
             if (c, f"F{fid}", dim, r) not in done]
    total = len(cfgs) * len(funcs) * len(dims) * runs

    # The manifest describes the whole accumulated file, not this invocation.
    manifest = {
        "started": prev.get("started", time.strftime("%Y-%m-%d %H:%M:%S")),
        "last_written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "jobs": jobs, "numpy": np.__version__, "pandas": pd.__version__,
        "opfunu": _dep_version("opfunu"), "scipy": _dep_version("scipy"),
        "dims": sorted(set(prev.get("dims", [])) | set(int(d) for d in dims)),
        "runs": runs,
        "functions": function_order(set(prev.get("functions", []))
                                    | {f"F{f}" for f in funcs}),
        "max_fes_per_dim": {**prev.get("max_fes_per_dim", {}),
                            **{str(int(d)): int(fes_of(d)) for d in dims}},
        "configs": sorted(set(prev.get("configs", [])) | set(cfgs),
                          key=lambda c: list(ABLATE_CONFIGS).index(c)
                          if c in ABLATE_CONFIGS else 999),
        "factors": {k: list(v) for k, v in ABLATE_FACTORS.items()},
        "baseline": ABLATE_BASELINE,
        "seed_base": SEED_BASE, "ablate_salt": int(ABLATE_SALT),
        "seed_scheme": "default_rng([SEED_BASE, dim, run, ABLATE_SALT]) -- the "
                       "identical stream for every configuration (common random "
                       "numbers), so the paired tests are genuinely paired. "
                       "Cells with the same initial population size therefore "
                       "start from the identical population; the capped and "
                       "uncapped arms draw from the same stream but size that "
                       "population differently above D = 42, which is the point "
                       "of that factor.",
        "a12_orientation": "vargha_delaney_a12(cell, baseline); > 0.5 means the "
                           "CELL is better",
        "target_defaults": target_defaults(),
        "note": "Selects a finalist. It cannot decide adoption: the binding "
                "criterion is the Hybrid class rank at D=50 against the rivals "
                "plus the D=10 first place, which this function subset and pair "
                "of dimensions do not contain.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    print(f"[*] {len(cfgs)} configurations x {len(funcs)} functions x "
          f"{len(dims)} dimensions x {runs} runs = {total} runs")
    print(f"[*] {len(tasks)} to do in this invocation, jobs={jobs}")
    if not tasks:
        print("[*] nothing to run; reporting on what is recorded")
    else:
        buf, t0 = [], time.perf_counter()

        def flush(force=False):
            if buf and (force or len(buf) >= FLUSH_ROWS):
                pd.DataFrame(buf)[ABLATE_COLUMNS].to_csv(
                    raw_csv, mode="a", header=not raw_csv.exists(), index=False)
                buf.clear()

        payload = (delayed(_ablate_task)(c, fid, dim, r, fes)
                   for c, fid, dim, r, fes in tasks)
        n = 0
        try:
            with Parallel(n_jobs=jobs, backend="loky",
                          return_as=_generator_mode()) as parallel:
                for row in parallel(payload):
                    buf.append(row)
                    n += 1
                    if n % 200 == 0:
                        el = time.perf_counter() - t0
                        eta = el / n * (len(tasks) - n) / 60
                        print(f"  {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                              f"~{eta:.0f} min left", flush=True)
                    flush()
        finally:
            flush(force=True)
        print(f"[*] {n} runs in {(time.perf_counter()-t0)/60:.1f} min -> {raw_csv}")

    if not raw_csv.exists():
        print("[X] nothing was recorded")
        return 1
    df = pd.read_csv(raw_csv, float_precision="round_trip")
    df = df.drop_duplicates(subset=["Config", "Function", "Dimension", "Run"],
                            keep="last")
    bad = df[df["Overrun"] != 0]
    if len(bad):
        print(f"[X] {len(bad)} runs exceeded their budget; the tables would not "
              f"be usable")
        print(bad.head().to_string())
        return 1
    short = (df.groupby(["Config", "Function", "Dimension"]).size()
               .loc[lambda s: s != runs])
    if len(short):
        print(f"[!] {len(short)} cells do not hold {runs} runs; the tables below "
              f"cover what is recorded. Use --resume to finish them.")
        print(short.head().to_string())

    ablate_report(df, tdir, runs)
    print(f"\n[ok] tables -> {tdir}")
    print(f"[!] This selects a finalist. Adoption is decided on the full gate: "
          f"the Hybrid class rank at D=50 against the rivals must improve on "
          f"3.700 AND the D=10 first place (2.776) must not be lost.")
    return 0


# ==========================================================================
# 12. TESTS
# ==========================================================================
# This study used to carry a six-file pytest suite. Two of those files do not
# apply any more, and that is stated here rather than quietly dropped:
#
#   tests/test_noise.py          19 tests on the legacy noisy/dynamic problem
#                                machinery. That suite matches no competition
#                                protocol and is not part of the CEC study.
#   tests/test_port_fidelity.py  pinned "TEMOA_V12 minus N0-N3 reproduces
#                                L-SHADE-DGR". It needs the ancestors, which are
#                                deliberately outside the line-up.
#
# So the count here is smaller than the 72 the old plan quotes, and the smaller
# number is the honest one. Everything applicable was carried over and re-pointed
# at the CEC suites. Four tests are new and each pins a bug that was actually
# found in this codebase rather than a hypothetical one: the inverted A12
# orientation, the per-class rank tie artefact, the manifest budget key, and the
# opfunu version floor that silently redefines 13 of 29 functions.

_TESTS: list = []


def _test(slow: bool = False):
    """Register a test. ``slow`` ones run only under ``--full``."""
    def deco(fn):
        fn.slow = slow
        _TESTS.append(fn)
        return fn
    return deco


def _cec(fid, dim=10):
    return make_suite_problem("cec2017", fid, dim)


def _run_alg(alg, fid, budget, dim=10, seed=0, **kw):
    """One short run; returns the final error."""
    pr = _cec(fid, dim)
    tr = Tracker(pr, budget, N_POINTS)
    alg(tr, dim, (pr.lb, pr.ub), budget, np.random.default_rng(seed), **kw)
    return tr, tr.finalize()[1]


# ------------------------------------------------------------- statistics
@_test()
def test_holm_is_monotone_and_conservative():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    adj = holm(p)
    assert np.all(adj >= p - 1e-12), "Holm must never decrease a p-value"
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12), "Holm must be monotone"
    assert np.all(adj <= 1.0)
    assert abs(adj[0] - 0.005) < 1e-12          # smallest p, family of 5


@_test()
def test_holm_matches_reference_values():
    # step-down: [4*.01, max(prev,3*.02), max(prev,2*.03), max(prev,1*.04)]
    adj = holm([0.01, 0.02, 0.03, 0.04])
    assert np.allclose(adj, [0.04, 0.06, 0.06, 0.06]), adj


@_test()
def test_a12_known_cases():
    a, b = [1, 2, 3], [4, 5, 6]     # `a` always smaller => always better
    assert abs(vargha_delaney_a12(a, b) - 1.0) < 1e-12
    assert abs(vargha_delaney_a12(b, a) - 0.0) < 1e-12
    assert abs(vargha_delaney_a12(a, a) - 0.5) < 1e-12
    assert abs(cliffs_delta(a, b) - 1.0) < 1e-12
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 40), rng.normal(0.5, 1, 40)
    wins = np.mean([(xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y])
    assert abs(vargha_delaney_a12(x, y) - wins) < 1e-12


@_test()
def test_a12_orientation_is_not_inverted():
    """NEW. An earlier diagnostic read A12 backwards and swapped every verdict.

    The orientation is the one thing about this statistic that is easy to invert
    silently: both directions produce plausible-looking tables. Pinned against an
    unambiguous case and against the documented contract.
    """
    better, worse = np.zeros(20), np.ones(20)
    assert vargha_delaney_a12(better, worse) > 0.5, (
        "A12 > 0.5 must mean the FIRST sample is better for minimisation")
    assert vargha_delaney_a12(worse, better) < 0.5
    assert "A12 > 0.5 means `a` is better" in (vargha_delaney_a12.__doc__ or ""), \
        "the docstring states the contract; keep them in step"


@_test()
def test_friedman_matches_scipy():
    rng = np.random.default_rng(1)
    data = pd.DataFrame(rng.normal(size=(12, 5)), columns=list("ABCDE"))
    mine = friedman(data)
    ref = sps.friedmanchisquare(*[data[c].to_numpy() for c in data.columns])
    assert abs(mine["chi2"] - ref.statistic) < 1e-8, (mine["chi2"], ref.statistic)
    assert abs(mine["p_chi2"] - ref.pvalue) < 1e-10
    assert abs(mine["avg_ranks"].mean() - 3.0) < 1e-12


@_test()
def test_friedman_ranks_order_best_first():
    data = pd.DataFrame({"A": [1, 1, 1, 1], "B": [2, 2, 2, 2], "C": [3, 3, 3, 3]})
    fr = friedman(data)
    assert list(fr["avg_ranks"].index) == ["A", "B", "C"]
    assert abs(fr["avg_ranks"]["A"] - 1.0) < 1e-12


@_test()
def test_nemenyi_cd_matches_published_value():
    # Demsar (2006) Sec. 3.2.2: k=4, N=14 -> 2.569*sqrt(20/84) = 1.2535, quoted 1.25
    assert abs(nemenyi_cd(4, 14) - 2.569 * math.sqrt(4 * 5 / (6 * 14))) < 1e-12
    assert abs(nemenyi_cd(4, 14) - 1.25) < 5e-3
    assert nemenyi_cd(4, 50) < nemenyi_cd(4, 14), "CD must shrink as blocks grow"


@_test()
def test_paired_wilcoxon_handles_identical_samples():
    assert paired_wilcoxon([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


# ----------------------------------------------------------------- suites
@_test()
def test_official_numbering_maps_onto_opfunu():
    assert official_to_opfunu("cec2017", 1) == 1
    for off in range(3, 31):
        assert official_to_opfunu("cec2017", off) == off - 1
    assert official_to_opfunu("cec2022", 7) == 7


@_test()
def test_withdrawn_f2_is_refused_not_silently_remapped():
    try:
        official_to_opfunu("cec2017", 2)
    except ValueError:
        return
    raise AssertionError("official CEC'2017 F2 was withdrawn; it must be refused")


@_test()
def test_official_f3_is_zakharov_not_rosenbrock():
    """The off-by-one that would invalidate every published comparison."""
    assert "Zakharov" in _cec(3).description, _cec(3).description
    assert "Rosenbrock" in _cec(4).description, _cec(4).description
    assert "Rastrigin" in _cec(5).description, _cec(5).description


@_test()
def test_cec2017_covers_f1_and_f3_to_f30():
    ids = SUITES["cec2017"].function_ids
    assert ids[0] == 1 and 2 not in ids
    assert list(ids[1:]) == list(range(3, 31))
    assert len(ids) == 29


@_test()
def test_every_cec_function_attains_f_star_at_its_optimum():
    """Also the opfunu version floor: 1.0.1 returns NaN here for F18 and F30."""
    for off in SUITES["cec2017"].function_ids:
        pr = _cec(off)
        err = pr.true_obj(pr.o) - pr.f_star
        assert np.isfinite(err), f"F{off}: f(x*) is not finite -- opfunu too old?"
        assert abs(err) < 1e-8, f"F{off}: error at optimum = {err}"


@_test(slow=True)
def test_every_cec_function_attains_f_star_at_d30():
    for off in SUITES["cec2017"].function_ids:
        pr = _cec(off, 30)
        err = pr.true_obj(pr.o) - pr.f_star
        assert np.isfinite(err) and abs(err) < 1e-8, f"F{off} D=30: {err}"


@_test()
def test_opfunu_is_new_enough_to_define_f18_and_f30():
    """NEW. opfunu 1.0.1 partitions F18 as [.1,.2,.2,.2,.3]; at D=10 the first
    block is 1-dimensional and ``elliptic_func`` divides by ``ndim-1 == 0``.
    Thirteen of 29 functions differ between 1.0.1 and 1.0.4."""
    import opfunu
    parts = tuple(int(x) for x in opfunu.__version__.split(".")[:3])
    assert parts >= (1, 0, 4), (
        f"opfunu {opfunu.__version__} redefines 13 of 29 CEC'2017 functions and "
        f"yields NaN on F18/F30 at D=10; requirements.txt pins >= 1.0.4")
    for off in (18, 30):
        pr = _cec(off)
        assert np.isfinite(pr.true_obj(pr.o)), f"F{off} evaluates to NaN"


@_test()
def test_cec2022_biases_match_the_official_values():
    expected = {1: 300.0, 2: 400.0, 3: 600.0, 12: 2700.0}
    for fid, bias in expected.items():
        assert make_suite_problem("cec2022", fid, 10).f_star == bias


@_test()
def test_cec_bounds_are_the_official_box():
    for off in (1, 4, 15, 30):
        pr = _cec(off)
        assert (pr.lb, pr.ub) == (-100.0, 100.0), (off, pr.lb, pr.ub)


@_test()
def test_protocols_match_the_competition_rules():
    s = SUITES["cec2017"]
    assert s.runs == 51 and s.error_floor == 1e-8 and s.protocol_verified
    assert s.max_fes(10) == 100_000 and s.max_fes(30) == 300_000
    assert s.dims == (10, 30, 50, 100)
    c22 = SUITES["cec2022"]
    assert not c22.protocol_verified, "CEC'2022 MaxFES is still unconfirmed"


@_test()
def test_error_floor_zeroes_below_but_not_above():
    pr = _cec(1)
    tr = Tracker(pr, 10, N_POINTS)
    tr.best_f, tr.best_x = pr.f_star + 1e-12, pr.o.copy()
    assert tr._floor(1e-12) == 0.0
    assert tr._floor(1e-7) == 1e-7, "an error above the floor must survive"


@_test()
def test_cec_problem_satisfies_the_tracker_interface():
    pr = _cec(4)
    for attr in ("lb", "ub", "f_star", "kind", "error_floor", "true_obj"):
        assert hasattr(pr, attr), attr
    assert pr.kind == "plain"
    assert isinstance(pr(np.zeros(10)), float)


@_test()
def test_function_labels_sort_naturally_in_tables():
    assert function_order(["F10", "F3", "F1", "F30", "F4"]) == \
        ["F1", "F3", "F4", "F10", "F30"]


# ---------------------------------------------------------------- tracker
@_test()
def test_tracker_counts_and_scores():
    pr = _cec(1)
    tr = Tracker(pr, 100, 10)
    for _ in range(100):
        tr(np.zeros(10))
    assert tr.fes == 100 and tr.overrun == 0
    curve, score = tr.finalize()
    assert np.all(np.isfinite(curve)) and np.isfinite(score)


@_test()
def test_tracker_charges_an_overrun_and_does_not_score_it():
    pr = _cec(1)
    tr = Tracker(pr, 10, 10)
    for _ in range(15):
        tr(np.zeros(10))
    assert tr.fes == 15 and tr.overrun == 5, (tr.fes, tr.overrun)


@_test()
def test_convergence_curve_is_monotone_for_deterministic_problems():
    pr = _cec(4)
    tr = Tracker(pr, 500, 20)
    rng = np.random.default_rng(0)
    for _ in range(500):
        tr(rng.uniform(pr.lb, pr.ub, 10))
    curve, _ = tr.finalize()
    assert np.all(np.diff(curve) <= 1e-12), "best-so-far error must never increase"


# ------------------------------------------------------------- algorithms
@_test()
def test_no_algorithm_exceeds_its_evaluation_budget():
    budget = 6000
    for name, alg in ALGORITHMS.items():
        tr, err = _run_alg(alg, 4, budget, seed=3)
        assert tr.overrun == 0, f"{name} overran by {tr.overrun}"
        assert tr.fes <= budget, f"{name} used {tr.fes} of {budget}"
        assert np.isfinite(err), f"{name} returned a non-finite score"


@_test()
def test_every_algorithm_spends_its_whole_budget():
    """A competitor that stops early would be handed a loss it did not earn."""
    budget = 20_000
    for name, alg in ALGORITHMS.items():
        tr, _ = _run_alg(alg, 4, budget, seed=1)
        assert tr.fes / budget > 0.98, f"{name} used only {tr.fes}/{budget}"


@_test()
def test_modern_de_reaches_high_precision_on_a_unimodal_problem():
    for name in ("LSHADE", "jSO", TARGET):
        _, err = _run_alg(ALGORITHMS[name], 1, 40_000, seed=0)
        assert err < 1e-8, f"{name} only reached {err:.3e} on F1 (Bent Cigar)"


@_test()
def test_cma_family_solves_the_rotated_ill_conditioned_function():
    """The invariance claim is tested on CMA-ES's own ground."""
    for name in ("CMAES", "IPOP_CMAES", "BIPOP_CMAES"):
        _, err = _run_alg(ALGORITHMS[name], 1, 30_000, seed=0)
        assert err < 1e-8, f"{name} only reached {err:.3e} on F1"


@_test()
def test_sep_cma_es_really_is_the_separable_variant():
    """sep-CMA-ES is the control that isolates rotation: it must be clearly worse
    than full CMA-ES on a rotated ill-conditioned function, or it is not diagonal."""
    _, full = _run_alg(ALGORITHMS["CMAES"], 1, 20_000, seed=2)
    _, sep = _run_alg(ALGORITHMS["sepCMAES"], 1, 20_000, seed=2)
    assert sep > full, f"sep-CMA-ES ({sep:.3e}) should trail CMA-ES ({full:.3e}) on F1"


@_test()
def test_cma_wrappers_are_three_distinct_algorithms():
    """CMAES and IPOP differ only in incpopsize (1 vs 2, and cma defaults to 2);
    BIPOP differs from IPOP only in the bipop flag. A missing override would make
    two of them the same algorithm under different names.

    F25 is a composition function, where the restart policy decides the outcome,
    and the budget has to be large enough for BIPOP's second regime to engage --
    measured: identical to IPOP below ~20000 evaluations, distinct from 50000
    upward. Below that they agree bit-for-bit, and that is correct BIPOP
    behaviour, not a wiring fault, so a cheap version of this test would report a
    collapse that is not there.
    """
    for seed in (1, 2):
        c = _run_alg(ALGORITHMS["CMAES"], 25, 20_000, seed=seed)[1]
        i = _run_alg(ALGORITHMS["IPOP_CMAES"], 25, 20_000, seed=seed)[1]
        assert c != i, (f"CMAES == IPOP_CMAES on F25 seed {seed} ({c:.6e}) -- "
                        f"the incpopsize=1 override is missing")
    assert ALGORITHMS["BIPOP_CMAES"] is not ALGORITHMS["IPOP_CMAES"]


@_test(slow=True)
def test_bipop_second_regime_engages_and_separates_it_from_ipop():
    """BIPOP's small-population regime only engages after the first restart, so
    below that point BIPOP and IPOP are the same algorithm bit-for-bit -- correct
    behaviour, not a wiring fault.

    Measured in this environment (D=10, 3 seeds, fraction of seeds where the two
    differ): F25 20k 0/3, F25 30k 1/3, F25 50k 1/3, F21 20k 1/3, F21 30k 2/3,
    F21 50k 3/3. In the completed 100k-FES gate they agree on only 2% of F25
    runs. So the honest assertion is "differs on at least one seed": requiring
    every seed to differ would make the test fail on a coincidence rather than on
    a fault, which is exactly what a cheaper version of this test did.
    """
    diffs = 0
    for seed in (1, 2, 3):
        i = _run_alg(ALGORITHMS["IPOP_CMAES"], 21, 50_000, seed=seed)[1]
        b = _run_alg(ALGORITHMS["BIPOP_CMAES"], 21, 50_000, seed=seed)[1]
        diffs += int(i != b)
    assert diffs > 0, ("BIPOP never diverged from IPOP over three seeds at "
                       "50000 evaluations -- the bipop flag is not reaching cma")


@_test()
def test_same_seed_gives_bit_identical_results():
    for name in ALGORITHMS:
        a = _run_alg(ALGORITHMS[name], 4, 4000, seed=5)[1]
        b = _run_alg(ALGORITHMS[name], 4, 4000, seed=5)[1]
        assert a == b, f"{name} is not reproducible: {a!r} vs {b!r}"


#: What ``LSHADE_DGR`` was measured with when ``results_cec2017/`` was recorded.
#: Every row in that file was produced by these values, and the driver calls the
#: algorithm with no keyword arguments at all, so a changed default silently
#: rewrites the meaning of 31,059 already-recorded rows. The two tests below make
#: that impossible to do by accident: one pins the defaults, the other pins the
#: numbers they produce.
TARGET_RECORDED_DEFAULTS = {
    "POP_FACTOR": 12, "POP_MAX": 500, "P_MIN": 0.02, "P_EIG": 0.4,
    "ADAPT_EIG": True, "ARC_RATE": 1.0, "CR_FLOOR": True, "OPS": (0, 1, 3),
    "EIGEN_GATE": True, "DIV_GUARD": True, "DIV_THRESH": 1e-4, "DIV_BOOST": 0.5,
    "RESTART": True, "RESTART_BUDGET_FRAC": 0.2, "RESTART_GROWTH": 2.0,
}


@_test()
def test_the_target_keeps_the_defaults_its_results_were_measured_with():
    """A new flag must default to the behaviour already in ``results_cec2017/``.

    Adding a flag is safe; changing what an existing one defaults to is not,
    because the driver passes no keyword arguments and the raw CSV records no
    hyperparameters. Both halves are checked: no recorded default may drift, and
    no *new* parameter may appear without being declared here, which is what
    forces the author of a new flag to state its default deliberately.
    """
    params = inspect.signature(LSHADE_DGR).parameters
    for key, want in TARGET_RECORDED_DEFAULTS.items():
        assert key in params, f"{key} was removed from LSHADE_DGR"
        got = params[key].default
        assert got == want, f"default {key} drifted: {got!r} != recorded {want!r}"

    declared = set(TARGET_RECORDED_DEFAULTS) | {"logger"}
    added = [k for k, v in params.items()
             if v.default is not inspect.Parameter.empty and k not in declared]
    # Compared by repr, not by ==: `0.0 == False` is True in Python, so a
    # value-based allowlist would silently accept any new flag defaulting to
    # zero, which is exactly the kind of accidental pass this test exists to
    # prevent.
    neutral = {"False", "'success'", "'uniform'", "'linear'", "'rate'",
               "None", "4.0", "0.0"}
    for key in added:
        assert repr(params[key].default) in neutral, \
            (f"new flag {key}={params[key].default!r} does not default to the "
             f"behaviour results_cec2017/ was measured with")


@_test()
def test_the_target_reproduces_the_numbers_its_results_were_measured_with():
    """Pinned outputs, not just pinned inputs.

    The defaults test cannot see a change to the *body* of the algorithm. These
    three literals were recorded from the code that produced ``results_cec2017/``;
    any edit that moves them has changed the algorithm, whatever its flags say.
    Three cells rather than one: F4 exercises the rotated unimodal path, F11 a
    hybrid, and D=30 a dimension where the population is larger than ``dim``, so
    the eigen gate is open.

    Compared to a relative tolerance rather than bit for bit, because float64
    reductions are not portable: the same source file, hash-identical, returns
    different last digits under a different BLAS and CPU. Measured on the
    Windows machine that recorded the literals against a Linux/AMD EPYC run,
    the three cells drift by 2.0e-11, 6.0e-10 and 6.2e-9. A genuine change to
    the algorithm moves them by 2.8e-1 to 3.6e-1 (flipping ``CR_FLOOR``,
    ``P_EIG`` or ``OPS`` on the F4 D=10 cell), because the trajectory diverges
    chaotically from the first differing selection. The gap between the two is
    seven and a half orders of magnitude, so 1e-6 sits 160x above the largest
    platform drift and 280000x below the smallest real edit, and the guard
    keeps the discrimination it was written for.
    """
    for fid, budget, dim, seed, want in (
            (4, 4000, 10, 5, 6.680478750061297),
            (11, 4000, 10, 5, 29.75348084491884),
            (4, 6000, 30, 3, 74.35420153038649)):
        got = _run_alg(LSHADE_DGR, fid, budget, dim=dim, seed=seed)[1]
        assert abs(got - want) <= 1e-6 * abs(want), \
            (f"L-SHADE-DGR changed on F{fid} D={dim}: {got!r} != recorded "
             f"{want!r}. results_cec2017/ was measured with the old behaviour.")


def _dgr_log(fid=4, dim=10, budget=20_000, seed=1, **kw):
    """Run the target with instrumentation on and hand back the generation log."""
    pr = _cec(fid, dim)
    tr = Tracker(pr, budget, N_POINTS)
    log: list = []
    LSHADE_DGR(tr, dim, (pr.lb, pr.ub), budget,
               np.random.default_rng(seed), logger=log, **kw)
    return tr, log


@_test()
def test_the_gate_resumes_the_whole_line_up_after_an_interruption():
    """The defect this pins cost nothing yet only because it was caught first.

    An earlier gate chose between "measure the line-up" and "measure only the
    candidate" by asking whether the results file existed. Interrupt a multi-day
    run once and the file exists, so the next attempt measures only the
    candidate and leaves the rivals permanently incomplete -- and it fails in a
    way that reads like a missing baseline rather than a half-finished run.

    So: the gate must never restrict the line-up with --algos, and must pass
    --resume exactly when a manifest is there to resume against. cmd_run is
    stubbed out because what is being tested is the decision, not the search.
    """
    import tempfile
    real = cmd_run
    # stage_gate registers the winner, which mutates a module-level registry.
    # Restored below: a test that leaks into ALGORITHMS makes every later test
    # that counts the line-up depend on the order tests happen to run in.
    registry = dict(ALGORITHMS)
    try:
        def argv_for(make_manifest: bool):
            d = Path(tempfile.mkdtemp()) / "res"
            (d / "raw").mkdir(parents=True)
            pd.DataFrame([{"Algorithm": TARGET, "Function": "F1",
                           "Dimension": 10, "Run": 0, "Error": 0.0,
                           "Seconds": 1.0, "FES": 1, "Overrun": 0}]).to_csv(
                d / "raw" / "results.csv", index=False)
            if make_manifest:
                (d / "manifest.json").write_text("{}", encoding="utf-8")
            seen = {}
            cmd_run_spy = lambda a: (seen.update(vars(a)) or 1)
            globals()["cmd_run"] = cmd_run_spy
            state = {"stages": {"select": {
                "winner": "dim-cond",
                "config": {"POP_UNCAP": True, "OPS_DROP_ABOVE": 10}}}}
            stage_gate(state, 2, d)
            return seen

        fresh = argv_for(False)
        assert fresh.get("resume") is False, \
            "a study with no manifest cannot be resumed; cmd_run refuses that"
        assert not fresh.get("algos"), (
            f"the gate restricted the line-up to {fresh.get('algos')}; it must "
            f"measure every algorithm so a study can be built from nothing")
        assert sorted(fresh.get("dims") or []) == sorted(STUDY_DIMS)

        again = argv_for(True)
        assert again.get("resume") is True, (
            "the gate did not resume an existing study; an interrupted "
            "multi-day run would restart from zero or skip the rivals")
        assert not again.get("algos"), (
            f"after an interruption the gate narrowed to {again.get('algos')}; "
            f"the rivals' missing cells would never be filled")
    finally:
        globals()["cmd_run"] = real
        ALGORITHMS.clear()
        ALGORITHMS.update(registry)


@_test()
def test_the_analysis_drops_a_dimension_not_every_algorithm_has():
    """A dimension only one algorithm reached is not a comparison.

    At k = 1 the statistics degenerate rather than fail: the class rank comes
    out 1.000 everywhere, the critical difference is refused, and what lands on
    disk is a table showing a clean sweep that means nothing. This was measured
    once -- 125 core-hours of D=100 rows no rival had -- so the guard exists and
    this pins it. The rows themselves are never touched; only the analysis
    declines to read them.
    """
    import tempfile
    out = Path(tempfile.mkdtemp()) / "study"
    (out / "raw").mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    pair = [TARGET, "jSO"]
    for alg in pair:
        for dim in (10, 30):
            for fid in (1, 3, 4, 5):
                for run in range(3):
                    rows.append({"Algorithm": alg, "Function": f"F{fid}",
                                 "Dimension": dim, "Run": run,
                                 "Error": float(rng.random()), "Seconds": 1.0,
                                 "FES": 100, "Overrun": 0})
    # one algorithm, one extra dimension: comparable with nothing
    for fid in (1, 3, 4, 5):
        for run in range(3):
            rows.append({"Algorithm": TARGET, "Function": f"F{fid}",
                         "Dimension": 100, "Run": run,
                         "Error": float(rng.random()), "Seconds": 1.0,
                         "FES": 100, "Overrun": 0})
    pd.DataFrame(rows).to_csv(out / "raw" / "results.csv", index=False)
    rc = cmd_analyze(build_parser().parse_args(
        ["analyze", "--out", str(out), "--algos", *pair,
         "--control", TARGET, "--no-figures"]))
    assert rc == 0, f"the analysis refused a usable results file: {rc}"
    tdir = out / f"tables_k{len(pair)}"
    assert (tdir / "class_ranks_10D.csv").exists()
    assert not (tdir / "class_ranks_100D.csv").exists(), (
        "a class-rank table was written for a dimension only one algorithm "
        "has rows at; at k=1 every class ranks 1.000 and a later reader would "
        "be right to mistake that for a win")
    # and the measurements are untouched where they were recorded
    back = pd.read_csv(out / "raw" / "results.csv")
    assert (back["Dimension"] == 100).sum() == 12, \
        "the analysis edited the results file instead of declining to read it"


@_test()
def test_fes_to_target_counts_budget_not_error():
    """The supplementary metric reads the curves and censors honestly.

    Two synthetic runs: one crosses the floor halfway through the budget, one
    never does. The median must be taken over the run that crossed, the success
    rate must be 0.5, and the crossing must be located to the checkpoint grid
    rather than invented between points.
    """
    import tempfile
    out = Path(tempfile.mkdtemp()) / "study"
    (out / "curves").mkdir(parents=True)
    (out / "raw").mkdir(parents=True)
    n = 100
    crossed = np.concatenate([np.full(50, 1.0), np.full(n - 50, 1e-12)])
    never = np.full(n, 1.0)
    np.savez_compressed(out / "curves" / "curves_10D.npz",
                        **{f"{TARGET}|F1|0": crossed, f"{TARGET}|F1|1": never})
    pd.DataFrame([{"Algorithm": TARGET, "Function": "F1", "Dimension": 10,
                   "Run": 0, "Error": 0.0, "Seconds": 1.0,
                   "FES": 1, "Overrun": 0}]).to_csv(
        out / "raw" / "results.csv", index=False)
    t = fes_to_target(out, out, [TARGET], [10])
    assert len(t) == 1, t
    r = t.iloc[0]
    assert r["runs"] == 2 and r["success_rate"] == 0.5, (
        f"censoring is wrong: {r['runs']} runs, {r['success_rate']} success")
    # The budget comes from budget_of, which is the same source the curves were
    # written against; asserting against a literal here would pin this test to a
    # fixture assumption rather than to the behaviour.
    step = int(budget_of(out, 10)) / n
    assert r["median_fes_to_target"] == 51 * step, (
        f"the crossing was not located on the checkpoint grid: "
        f"{r['median_fes_to_target']} != {51 * step}")
    assert r["resolution_fes"] == step, "the resolution must be reported"


@_test()
def test_the_dimension_conditional_portfolio_is_the_configuration_it_claims():
    """The claim that makes this variant cost one gate instead of three stages.

    `OPS_DROP_ABOVE=10` with `POP_UNCAP=True` is asserted to be the recorded
    algorithm at D=10 and `core-only` at D=30 and D=50 -- bit for bit, not
    approximately. That is what licenses skipping the screening: there is
    nothing left to screen, because every dimension the suite runs has already
    been measured under one of those two configurations.

    POP_UNCAP is inert below D=42 (12*D clears 500 only from D=42), which is
    what makes the D=10 and D=30 halves hold; above it the cap binds and the
    arm must match core-only, which carries POP_UNCAP too.

    If this test fails, the variant is a new algorithm at some dimension and
    has to go through the screening like any other.
    """
    recorded, core = {}, dict(POP_UNCAP=True, OPS=(0,))
    cond = dict(POP_UNCAP=True, OPS_DROP_ABOVE=10)
    for fid, dim, want_same_as, label in ((11, 10, recorded, "the recorded algorithm"),
                                          (4, 10, recorded, "the recorded algorithm"),
                                          (11, 30, core, "core-only"),
                                          (4, 30, core, "core-only")):
        budget = 4000 if dim == 10 else 6000
        got = _run_alg(LSHADE_DGR, fid, budget, dim=dim, seed=5, **cond)[1]
        ref = _run_alg(LSHADE_DGR, fid, budget, dim=dim, seed=5, **want_same_as)[1]
        assert got == ref, (
            f"OPS_DROP_ABOVE=10 is not {label} on F{fid} at D={dim}: "
            f"{got!r} != {ref!r}")
    # And above the threshold it must differ from the recorded algorithm, or
    # the switch never fires and the arm is the baseline under another name.
    dropped = _run_alg(LSHADE_DGR, 11, 6000, dim=30, seed=5, **cond)[1]
    base = _run_alg(LSHADE_DGR, 11, 6000, dim=30, seed=5)[1]
    assert dropped != base, (
        "OPS_DROP_ABOVE=10 produced the recorded behaviour at D=30; the "
        "threshold never fired")


@_test()
def test_the_eigen_channel_is_rotation_equivariant_only_at_p_one():
    """The claim the whole eigen argument rests on, checked in this code.

    Crossover in the covariance eigenbasis is exactly rotation-equivariant:
    under x -> Rx the covariance becomes R C R^T, whose eigenvector matrix is
    R B, and B [P_S B^T v + (I - P_S) B^T x] carries the rotation straight
    through. Coordinate-wise crossover is not, because a general rotation does
    not map a coordinate subspace to a coordinate subspace. A p-mixture of the
    two is therefore equivariant only at p = 1 -- which both adaptive rules are
    barred from reaching, since each clips p to [0.1, 0.9].

    So this asserts both halves: equivariance holds at P_EIG=1.0 and fails at
    0.9. The second half is the one that matters, because it is what makes the
    clip a structural bar rather than a tuning detail.

    A tolerance, not an equality: eigh(R C R^T) is not computed as R eigh(C) in
    floating point. One generation, not a trajectory -- over a run the uniform
    initialisation inside a box and the coordinate-wise bound repair are both
    basis-dependent, and neither is what this test is about.
    """
    rng_setup = np.random.default_rng(12345)
    dim, n = 12, 60
    # Well inside the box, so the midpoint repair cannot fire and confound this.
    X = rng_setup.normal(0.0, 3.0, (n, dim))
    R = np.linalg.qr(rng_setup.normal(size=(dim, dim)))[0]

    w = np.linalg.eigvalsh(np.cov(X[:max(2, n // 2)], rowvar=False))
    gaps = np.diff(np.sort(w))
    assert gaps.min() > 1e-8, (
        f"the covariance has near-degenerate eigenvalues (min gap {gaps.min():.2e}); "
        f"eigh may then return any basis of the eigenspace and equivariance is "
        f"not claimed there")

    def one_generation(pop, p_eig):
        """The eigen crossover exactly as the algorithm applies it."""
        r = np.random.default_rng(777)
        n_samples = max(2, len(pop) // 2)
        _, B = np.linalg.eigh(np.cov(pop[:n_samples], rowvar=False))
        V = pop + 0.5 * (pop[r.permutation(len(pop))] - pop)
        cross = r.random(pop.shape) < 0.5
        U = np.where(cross, V, pop)
        use = r.random(len(pop)) < p_eig
        if np.any(use):
            xe, ve = pop[use] @ B, V[use] @ B
            U[use] = np.where(cross[use], ve, xe) @ B.T
        return U

    for p_eig, equivariant in ((1.0, True), (0.9, False)):
        U = one_generation(X, p_eig)
        U_rot = one_generation(X @ R.T, p_eig)
        err = np.linalg.norm(U_rot - U @ R.T) / np.linalg.norm(U)
        if equivariant:
            assert err < 1e-10, (
                f"the eigen channel is not rotation-equivariant at p={p_eig}: "
                f"relative error {err:.3e}. Theorem aside, the code must show it.")
        else:
            assert err > 1e-6, (
                f"a p={p_eig} mixture came out equivariant (error {err:.3e}). "
                f"Either the coordinate channel is unreachable at this p or the "
                f"test no longer measures the mixture.")


@_test()
def test_the_misalignment_statistic_measures_alignment_not_conditioning():
    """align and kappa answer different questions, and only one is the right one.

    A diagonal covariance can be arbitrarily ill conditioned while its
    eigenbasis is already the coordinate basis, so rotating into it gains
    nothing; a well-conditioned one can sit 45 degrees off the axes, where
    coordinate crossover is maximally wrong. The condition rules adapt on
    kappa, which is why they were measured worse than the success rule; this
    pins the statistic that does separate the two cases.
    """
    def align_of(C):
        return float(np.linalg.norm(C - np.diag(np.diag(C)))) / float(np.linalg.norm(C))

    ill_but_aligned = np.diag([1.0, 1e6])
    mild_but_rotated = np.array([[1.0, 0.99], [0.99, 1.0]])

    k_aligned = np.linalg.cond(ill_but_aligned)
    k_rotated = np.linalg.cond(mild_but_rotated)
    assert k_aligned > 1e5 > k_rotated, "the counterexamples lost their point"

    assert align_of(ill_but_aligned) == 0.0, (
        "a diagonal covariance must score 0: its eigenbasis is the coordinate "
        "basis up to signed permutation, which binomial crossover already "
        "respects")
    assert align_of(mild_but_rotated) > 0.5, (
        "a covariance whose eigenbasis is 45 degrees off the axes must score "
        "high, whatever its condition number says")
    assert align_of(np.eye(5)) == 0.0


@_test()
def test_the_new_eigen_flags_are_bit_exact_no_ops_at_their_defaults():
    """Passing a new flag at its default must change nothing at all.

    The pinned-literal test would catch a changed default, but not a branch
    that is logically inert while still consuming a random draw or perturbing
    an accumulator. This compares a run with the flags stated explicitly
    against one with them omitted, bit for bit, which is the property the
    recorded 31,059 rows depend on.
    """
    base = _run_alg(LSHADE_DGR, 11, 4000, dim=10, seed=3)[1]
    same = _run_alg(LSHADE_DGR, 11, 4000, dim=10, seed=3,
                    EIGEN_GATE_EXTRA=0.0, EIG_MEMORY=0.0, ALIGN_TAU=0.0)[1]
    assert base == same, (
        f"the new eigen flags are not inert at their defaults: {base!r} != "
        f"{same!r}. results_cec2017/ was measured without them.")
    # And each one must actually do something when it is turned on, or the
    # screening would be comparing an arm with itself.
    moved = _run_alg(LSHADE_DGR, 11, 4000, dim=10, seed=3, EIG_MEMORY=0.8)[1]
    assert moved != base, "EIG_MEMORY=0.8 changed nothing; the branch is dead"
    gated = _run_alg(LSHADE_DGR, 11, 4000, dim=10, seed=3, EIGEN_GATE_EXTRA=2.0)[1]
    assert gated != base, "EIGEN_GATE_EXTRA=2.0 changed nothing; the branch is dead"


@_test()
def test_p_eig_actually_moves_under_the_success_rule():
    """Guards the instrumentation, not just the algorithm.

    ``P_EIG`` is a parameter that the success rule reassigns. If the working
    value and the logged value ever drift apart, ``_diag_internals`` reports a
    constant and the diagnostic would "prove" that adaptation never moves the
    eigen probability -- which is the exact claim the study is investigating.
    """
    _, log = _dgr_log()
    open_gens = [e for e in log if e["eig_ok"]]
    assert open_gens, "the eigen gate never opened; the test measures nothing"
    seen = {e["P_EIG"] for e in log}
    assert len(seen) > 1, (
        f"P_EIG never moved under the success rule: {seen}. The logger is "
        f"probably reading the parameter instead of the working value.")


@_test()
def test_the_condition_rules_compute_the_ratio_they_claim_to():
    """Tie the published formula to what the algorithm actually did.

    Every generation logs ``cond_hat``, ``n_samples``, ``mp_valid`` and
    ``log10_cond_adj``, so the Marchenko-Pastur step can be re-derived from the
    log and compared against the value the run used. This checks the arithmetic
    on real states rather than on a hand-made matrix.
    """
    dim = 10
    for rule in ("condition_mp", "condition_raw"):
        _, log = _dgr_log(dim=dim, P_EIG_RULE=rule)
        checked = 0
        for e in log:
            if e["eig_branch"] != "condition":
                continue
            if rule == "condition_mp" and e["mp_valid"]:
                s = math.sqrt(dim / e["n_samples"])
                cond_null = ((1.0 + s) / (1.0 - s)) ** 2
            else:
                cond_null = 1.0
            want = math.log10(max(e["cond_hat"] / cond_null, 1.0))
            assert abs(e["log10_cond_adj"] - want) < 1e-9, \
                f"{rule}: logged {e['log10_cond_adj']} != re-derived {want}"
            checked += 1
        assert checked > 20, f"{rule}: only {checked} condition generations"


@_test()
def test_the_mp_correction_is_not_a_no_op():
    """If the null correction changed nothing there would be no claim to make."""
    tr_mp, log_mp = _dgr_log(P_EIG_RULE="condition_mp")
    tr_raw, _ = _dgr_log(P_EIG_RULE="condition_raw")
    active = [e for e in log_mp
              if e["eig_branch"] == "condition" and e["mp_valid"]]
    assert active, "the MP correction was never active; the test measures nothing"
    a, b = tr_mp.finalize()[1], tr_raw.finalize()[1]
    assert a != b, (
        f"condition_mp and condition_raw gave the identical error {a!r}; the "
        f"null correction is not reaching the run")


@_test()
def test_the_condition_rule_survives_n_samples_equal_to_dim():
    """The divide-by-zero that would have killed a whole ablation.

    ``1 - sqrt(p/n)`` is exactly 0.0 when ``n_samples == dim``, which happens at
    ``pop_size`` equal to ``2*dim`` and ``2*dim + 1`` -- values LPSR walks
    through on every run. EIGEN_GATE normally keeps ``n > p``, but it is a flag,
    and with it off the condition rule meets that state. Inside a loky worker the
    exception would take the entire run with it.
    """
    for dim, budget in ((10, 20_000), (50, 60_000)):
        tr, log = _dgr_log(dim=dim, budget=budget,
                           EIGEN_GATE=False, P_EIG_RULE="condition_mp")
        assert tr.overrun == 0
        hit = [e for e in log if e["n_samples"] == dim]
        assert hit, f"D={dim}: n_samples never equalled dim; the test is vacuous"
        for e in hit:
            assert not e["mp_valid"], \
                f"D={dim}: the MP correction claimed validity at n == p"
            assert np.isfinite(e["log10_cond_adj"]) or e["eig_branch"] != "condition"


def _dgr_paired(seed=7, fid=4, dim=10, budget=20_000, **kw):
    """One instrumented run plus the generator, so draw consumption is checkable."""
    pr = _cec(fid, dim)
    tr = Tracker(pr, budget, N_POINTS)
    rng = np.random.default_rng(seed)
    log: list = []
    LSHADE_DGR(tr, dim, (pr.lb, pr.ub), budget, rng, logger=log, **kw)
    return tr, log, rng


def _logs_are_identical(la, lb):
    """Every logged field equal, arrays included. Returns the first difference."""
    if len(la) != len(lb):
        return f"generation counts differ: {len(la)} vs {len(lb)}"
    for g, (a, b) in enumerate(zip(la, lb)):
        if set(a) != set(b):
            return f"gen {g}: different keys"
        for k in a:
            x, y = a[k], b[k]
            same = (np.array_equal(x, y, equal_nan=True)
                    if isinstance(x, np.ndarray)
                    else (x == y or (isinstance(x, float) and isinstance(y, float)
                                     and np.isnan(x) and np.isnan(y))))
            if not same:
                return f"gen {g}: {k} differs, {x!r} vs {y!r}"
    return None


@_test()
def test_decoupling_is_a_no_op_unless_the_rule_is_success():
    """What licenses running 20 ablation cells instead of 32.

    The factorial {cap} x {decouple} x {P_EIG rule} x {selection} has 32 cells,
    but under condition_mp, condition_raw and fixed the eigen probability is
    computed outside the credit block, or never, so CHANNEL_DECOUPLE cannot
    reach it. Twelve of the 32 cells are then the same run twice, at about 17
    wall hours of compute for no information.

    That is a claim about the code, so it is proved rather than asserted. It can
    only hold because the flag consumes no random draws and because the *value*
    of P_EIG cannot change stream length either -- ``rng.random(pop_size)`` draws
    pop_size uniforms whatever the threshold. The operator channel is not like
    this: levy_flight draws 2*m*dim normals, so op_prob does move the stream,
    which is why only the eigen half can be decoupled cleanly.

    Comparing the generator states is the strongest single check: equal states
    prove identical draw counts *and* identical values.
    """
    # The guard must actually fire, or the whole test is vacuous. Raising the
    # threshold alone also trips `collapsed` and restarts constantly, so the
    # restart is switched off to keep one epoch under study.
    forced = dict(DIV_THRESH=1e-1, RESTART=False)
    fired = sum(e["guard"] for e in _dgr_paired(**forced,
                                                P_EIG_RULE="condition_mp")[1])
    assert fired > 50, f"the guard fired {fired} times; the test proves nothing"

    for rule in ("condition_mp", "condition_raw", "fixed"):
        tr_a, log_a, rng_a = _dgr_paired(**forced, P_EIG_RULE=rule,
                                         CHANNEL_DECOUPLE=False)
        tr_b, log_b, rng_b = _dgr_paired(**forced, P_EIG_RULE=rule,
                                         CHANNEL_DECOUPLE=True)
        assert rng_a.bit_generator.state == rng_b.bit_generator.state, \
            f"{rule}: decoupling changed the random stream"
        curve_a, err_a = tr_a.finalize()
        curve_b, err_b = tr_b.finalize()
        assert err_a == err_b, f"{rule}: {err_a!r} != {err_b!r}"
        assert np.array_equal(curve_a, curve_b, equal_nan=True), \
            f"{rule}: the convergence curves differ"
        diff = _logs_are_identical(log_a, log_b)
        assert diff is None, f"{rule}: {diff}"

    # Negative control: under the success rule the flag must change something,
    # or it is simply not wired up and the three results above are meaningless.
    tr_a, _, _ = _dgr_paired(**forced, P_EIG_RULE="success",
                             CHANNEL_DECOUPLE=False)
    tr_b, _, _ = _dgr_paired(**forced, P_EIG_RULE="success",
                             CHANNEL_DECOUPLE=True)
    assert tr_a.finalize()[1] != tr_b.finalize()[1], (
        "CHANNEL_DECOUPLE changed nothing under the success rule either -- the "
        "flag is not reaching the credit block")


def _epoch_initial_sizes(log):
    """The population size each epoch started at.

    LPSR only ever shrinks within an epoch, so a size that goes *up* between two
    consecutive generations is a restart boundary.
    """
    sizes = [e["pop_size"] for e in log]
    return [sizes[0]] + [b for a, b in zip(sizes, sizes[1:]) if b > a]


@_test()
def test_a_restart_never_shrinks_the_population():
    """The defect that lifting the cap would otherwise have introduced.

    POP_MAX had two jobs: the initial size, and the ceiling on restart growth.
    Raising the initial size past a ceiling that stayed at 500 would have made
    ``min(500, 2*600)`` return 500, so the first restart would *shrink* the
    population from 600 to 500 -- the opposite of the IPOP restart this is
    modelled on, and invisible in any error table.
    """
    # D = 10 with a threshold above the initial diversity (~0.29 for a uniform
    # box) is the configuration that restarts reliably; at D = 50 the `stalled`
    # condition is rarely met inside a test-sized budget, so the cap's effect at
    # that dimension is measured by the sizing test below instead.
    for uncap in (False, True):
        _, log = _dgr_log(dim=10, budget=40_000, POP_UNCAP=uncap,
                          DIV_THRESH=0.5, RESTART=True)
        starts = _epoch_initial_sizes(log)
        assert len(starts) > 1, \
            f"POP_UNCAP={uncap}: no restart happened; nothing is tested"
        for a, b in zip(starts, starts[1:]):
            assert b >= a, (
                f"POP_UNCAP={uncap}: a restart shrank the population {a} -> "
                f"{b}; the growth ceiling is below the initial size")
        ceiling = 4.0 * starts[0] if uncap else 500
        assert starts[-1] <= ceiling, \
            f"POP_UNCAP={uncap}: growth passed its ceiling, {starts[-1]} > {ceiling}"


@_test()
def test_lifting_the_cap_restores_the_population_the_rivals_use():
    """The cap binds from D = 42, which is the measured defect it is there for."""
    def n_init(dim, **kw):
        return _epoch_initial_sizes(
            _dgr_log(dim=dim, budget=40 * dim, RESTART=False, **kw)[1])[0]

    assert n_init(30) == n_init(30, POP_UNCAP=True) == 360, \
        "below D = 42 the cap does not bind, so the flag must change nothing"
    assert n_init(50) == 500, "the recorded behaviour is the capped 500"
    assert n_init(50, POP_UNCAP=True) == 600, \
        "uncapped, D = 50 should be POP_FACTOR * dim = 600"
    assert n_init(50, POP_UNCAP=True, POP_RULE="rsp") == 1018, \
        "POP_RULE='rsp' should give L-SHADE-RSP's 75 * dim ** (2/3)"


@_test()
def test_rank_index_matches_the_linear_weights_it_claims():
    """The closed form is an exact inverse CDF, so measure it, not trust it.

    Weight ``n - i`` means the best individual is drawn ``n`` times as often as
    the worst, and the worst keeps non-zero mass. A sampler that silently
    produced a different curve -- or an out-of-range index at ``u == 0`` -- would
    change the algorithm while looking like L-SHADE-RSP in the source.
    """
    rng = np.random.default_rng(0)
    for n, k in ((9, 400_000), (2, 200_000), (200, 400_000)):
        draws = rank_index(rng, n, k)
        assert draws.min() >= 0 and draws.max() <= n - 1, \
            f"n={n}: index out of range [{draws.min()}, {draws.max()}]"
        emp = np.bincount(draws, minlength=n) / k
        want = (n - np.arange(n)) / (np.arange(1, n + 1).sum())
        # 4 sigma of the binomial standard error, plus a floor for tiny cells
        tol = 4.0 * np.sqrt(want * (1 - want) / k) + 1e-4
        assert np.all(np.abs(emp - want) <= tol), \
            f"n={n}: worst deviation {np.abs(emp - want).max():.2e}"

    # u == 0 is the boundary the clip exists for. y = n - i, so the smallest y
    # is the *worst* index; without the clip y would round to 0 and the index
    # would be n, one past the end of the population.
    class _Zero:
        def random(self, size):
            return np.zeros(size)

    got = rank_index(_Zero(), 10, 3).tolist()
    assert got == [9, 9, 9], f"u == 0 should give the worst index, got {got}"


@_test()
def test_rank_selection_keeps_the_archive_probability_of_the_uniform_path():
    """Switching the flag must change the weighting and nothing else.

    A uniform draw over pop+archive already takes an archive member with
    probability |A|/(|A|+N). If the ranked path did not reproduce that, the
    ablation cell would be testing two changes at once and no conclusion about
    selective pressure could be drawn from it.
    """
    rng = np.random.default_rng(1)
    for n_pop, n_arc in ((100, 100), (900, 300), (4, 4), (50, 0)):
        k = 400_000
        draws = union_rank_index(rng, n_pop, n_arc, k)
        got = float(np.mean(draws >= n_pop))
        want = n_arc / (n_arc + n_pop)
        tol = 4.0 * math.sqrt(max(want * (1 - want), 1e-12) / k) + 1e-4
        assert abs(got - want) <= tol, \
            f"N={n_pop} |A|={n_arc}: archive probability {got:.4f} != {want:.4f}"
        # ...and the archive half stays uniform among itself, as in the reference
        if n_arc > 1:
            arc = draws[draws >= n_pop] - n_pop
            counts = np.bincount(arc, minlength=n_arc)
            assert counts.min() > 0, "some archive members are unreachable"


@_test()
def test_rank_selection_never_picks_the_parent_as_r1():
    """The uniform path gets ``r1 != i`` free from its modular offset.

    A rank draw does not, and the clash loop only tests ``r2``, so without its
    own rejection the ranked cell would emit degenerate mutations whose
    difference vector is built from the parent itself.
    """
    for dim, budget in ((10, 8_000), (30, 20_000)):
        tr, log = _dgr_log(fid=11, dim=dim, budget=budget, SELECTION="rank")
        assert tr.overrun == 0 and log, f"D={dim}: the run did not complete"
    # The rejection is only observable through its effect, so the arrangement is
    # pinned in the source: r1 is redrawn until it differs from the parent.
    src = Path(__file__).read_text(encoding="utf-8")
    needle = "r1[bad] = rank_index(" + "rng, pop_size, int(bad.sum()))"
    assert needle in src, "the r1 self-exclusion loop is gone"


# ------------------------------------------------------------- pipeline
@_test()
def test_the_tuning_split_is_disjoint_from_everything_it_is_judged_on():
    """The one methodological error that would invalidate the whole study.

    DIV_PRICE is chosen by measurement, so it must not be chosen on the
    functions the choice is later judged by. Making the two sets disjoint in
    code is what turns that from a thing to remember into a thing that cannot
    be got wrong.
    """
    assert not set(TUNE_FUNCTIONS) & set(SCREEN_FUNCTIONS), (
        f"the tuning split overlaps the screening set: "
        f"{sorted(set(TUNE_FUNCTIONS) & set(SCREEN_FUNCTIONS))}")
    # The binding criterion is read on the hybrid class; tuning must not touch it.
    assert not set(TUNE_FUNCTIONS) & set(range(11, 21)), \
        "the tuning split overlaps the hybrid class the criterion is read on"
    assert TUNE_FUNCTIONS and SCREEN_FUNCTIONS


@_test()
def test_every_screening_arm_sets_real_parameters():
    """A typo in an arm's override would silently screen the baseline twice."""
    params = inspect.signature(LSHADE_DGR).parameters
    for name, kw in SCREEN_ARMS.items():
        assert kw, f"arm {name!r} overrides nothing; it would duplicate baseline"
        for key in kw:
            assert key in params, f"arm {name!r} sets unknown parameter {key!r}"
    assert len(set(map(lambda d: tuple(sorted(d.items())),
                       SCREEN_ARMS.values()))) == len(SCREEN_ARMS), \
        "two screening arms are the same configuration under different names"


@_test()
def test_selection_refuses_an_arm_that_does_not_beat_the_baseline():
    """Taking the best candidate is not the same as finding an improvement.

    Three things are pinned here, because the selection rule reads two classes
    and each half can fail on its own:

      1. If no arm beats the baseline on the target class, nothing is promoted.
         The best of a losing field is still losing, and promoting it would
         spend hours of gate compute measuring a regression.
      2. An arm that does beat it, without damaging the guard class, is
         promoted.
      3. An arm that beats the target *by* damaging the guard class is refused
         outright -- it has traded the class we lead for the class we lag,
         which is not an improvement however large the target movement is.

    The third is the half that is easy to lose in a refactor, because it only
    fires on an arm that looks good in the column the eye goes to first.
    """
    import tempfile
    d = Path(tempfile.mkdtemp()) / "pipe"
    (d / "screen").mkdir(parents=True)

    def write(spec):
        """spec: {config: (error on the target class, error on the guard class)}"""
        rows = []
        for cfg, (e_target, e_guard) in spec.items():
            for cls, err in ((SELECT_TARGET_CLASS, e_target),
                             (SELECT_GUARD_CLASS, e_guard)):
                for f in FUNCTION_CLASSES[cls]:
                    for r in range(3):
                        rows.append({"Config": cfg, "Function": f"F{f}",
                                     "Dimension": 50, "Run": r, "Error": err,
                                     "Seconds": 0.0, "FES": 1, "Overrun": 0})
        pd.DataFrame(rows).to_csv(d / "screen" / "results.csv", index=False)
        return {"stages": {"screen": {}}}

    # 1. every arm worse on the target class
    state = write({"baseline": (1.0, 1.0), "eig-p1": (2.0, 1.0),
                   "align": (3.0, 1.0)})
    assert stage_select(state, d) == 3, "a losing field must not be promoted"
    assert state["stages"]["select"]["winner"] is None

    # 2. a genuine winner: better on the target, no worse on the guard
    state = write({"baseline": (1.0, 1.0), "eig-p1": (0.5, 1.0),
                   "align": (3.0, 1.0)})
    assert stage_select(state, d) == 0
    assert state["stages"]["select"]["winner"] == "eig-p1"

    # 3. better on the target *and* worse on the guard -- refused, and named
    state = write({"baseline": (1.0, 1.0), "eig-p1": (0.5, 2.0)})
    assert stage_select(state, d) == 3, (
        "an arm that buys the target class with the guard class was promoted; "
        "the guard filter is a filter, not a tie-break")
    assert state["stages"]["select"]["winner"] is None
    assert "eig-p1" in state["stages"]["select"]["refused_on_guard"]


@_test()
def test_the_pipeline_skips_finished_stages():
    """A day-long run has to survive being restarted."""
    import tempfile
    d = Path(tempfile.mkdtemp()) / "pipe"
    _save_state(d / "state.json", {"stages": {"preflight": {"done": True}}})
    args = build_parser().parse_args(
        ["experiment", "--out", str(d), "--only-stages", "preflight"])
    assert cmd_experiment(args) == 0
    state = _pipeline_state(d / "state.json")
    assert state["stages"]["preflight"]["done"] is True


# ------------------------------------------------------------- ablation
@_test()
def test_the_ablation_is_the_factorial_it_claims_to_be():
    """The cells are generated, so what needs checking is the generator."""
    expected = (len(ABLATE_FACTORS["pop"]) * len(ABLATE_FACTORS["eig"])
                * len(ABLATE_FACTORS["sel"]))
    assert len(ABLATE_CONFIGS) == expected == 20, \
        f"expected 20 cells, got {len(ABLATE_CONFIGS)}"
    assert ABLATE_BASELINE in ABLATE_CONFIGS
    assert ABLATE_CONFIGS[ABLATE_BASELINE] == {}, (
        "the baseline cell must be the algorithm exactly as results_cec2017/ "
        f"measured it, but it overrides {ABLATE_CONFIGS[ABLATE_BASELINE]}")

    params = inspect.signature(LSHADE_DGR).parameters
    for name, kw in ABLATE_CONFIGS.items():
        for key in kw:
            assert key in params, f"cell {name!r} sets unknown parameter {key}"
        assert ablate_factor_of(name), f"cell {name!r} does not split into factors"
    # Every level must appear in exactly half (pop, sel) or a fifth (eig) of the
    # cells, or the crossing is not balanced and the marginal table would lie.
    for factor, levels in ABLATE_FACTORS.items():
        counts = {lvl: sum(1 for c in ABLATE_CONFIGS
                           if ablate_factor_of(c)[factor] == lvl)
                  for lvl in levels}
        assert len(set(counts.values())) == 1, \
            f"factor {factor} is unbalanced: {counts}"


@_test()
def test_the_ablation_seeds_every_cell_identically():
    """Common random numbers, which is what makes the paired tests paired.

    Proved behaviourally rather than by reading the seed expression: at D = 10
    the population cap does not bind, so the capped and uncapped cells are the
    same algorithm. If the configuration name reached the seed -- as it does in
    the component diagnostic -- they would differ anyway.
    """
    fes = 6_000
    a = _ablate_task("capped | success-coupled | uniform", 11, 10, 3, fes)
    b = _ablate_task("uncapped | success-coupled | uniform", 11, 10, 3, fes)
    assert a["Error"] == b["Error"], (
        f"{a['Error']!r} != {b['Error']!r}: at D=10 the cap does not bind, so "
        f"these cells differ only if the config name reached the seed")

    # ...and a cell that really is different must not collide with it.
    c = _ablate_task("capped | condition_mp | rank", 11, 10, 3, fes)
    assert c["Error"] != a["Error"], "two genuinely different cells agree exactly"


@_test(slow=True)
def test_no_ablation_cell_exceeds_its_budget():
    """One over-budget cell invalidates the whole table, so check all 20.

    Both dimensions the ablation runs at, because the uncapped arm changes the
    population size and the restart ceiling with dimension, and the budget
    accounting is what those touch.
    """
    for dim in ABLATE_DIMS:
        fes = 40 * dim
        for cfg in ABLATE_CONFIGS:
            row = _ablate_task(cfg, 11, dim, 0, fes)
            assert row["Overrun"] == 0, f"{cfg} at D={dim} overran its budget"
            assert row["FES"] <= fes, f"{cfg} at D={dim}: {row['FES']} > {fes}"


@_test()
def test_ablate_resumes_only_what_is_missing():
    """A 29-hour run has to survive a crash without recomputing what it had."""
    import tempfile
    d = Path(tempfile.mkdtemp()) / "abl"
    argv = ["ablate", "--dims", "10", "--runs", "2", "--fes-per-dim", "400",
            "--functions", "11", "--jobs", "2", "--out", str(d),
            "--configs", ABLATE_BASELINE, "capped | condition_mp | rank"]
    assert cmd_ablate(build_parser().parse_args(argv)) == 0
    raw = d / "raw" / "results.csv"
    first = pd.read_csv(raw, float_precision="round_trip")
    assert len(first) == 4, f"expected 4 rows, got {len(first)}"

    # Drop one row and resume: exactly that row comes back, and identically,
    # which is only true if the stream does not depend on what else ran.
    gone = first.iloc[[1]]
    first.drop(index=first.index[1]).to_csv(raw, index=False)
    assert cmd_ablate(build_parser().parse_args(argv + ["--resume"])) == 0
    back = pd.read_csv(raw, float_precision="round_trip")
    key = ["Config", "Function", "Dimension", "Run"]
    assert len(back) == 4 and not back.duplicated(subset=key).any()
    merged = back.merge(gone, on=key, suffixes=("_new", "_old"))
    assert len(merged) == 1 and merged["Error_new"].iloc[0] == merged["Error_old"].iloc[0], \
        "the refilled row differs from the one it replaced"

    # Resuming a complete file must add nothing at all.
    before = raw.read_bytes()
    assert cmd_ablate(build_parser().parse_args(argv + ["--resume"])) == 0
    assert raw.read_bytes() == before, "a complete resume rewrote the results"


# -------------------------------------------------------------- line-up
@_test()
def test_the_target_algorithm_is_registered():
    assert TARGET in ALGORITHMS


@_test()
def test_algorithm_seed_depends_only_on_the_name():
    """Seeding from a position in a sorted list re-seeds every algorithm whenever
    the line-up changes, which silently invalidates a resumed run."""
    assert algorithm_seed("jSO") == algorithm_seed("jSO")
    assert algorithm_seed(TARGET) == 1311180535, "the recorded value must not drift"


@_test()
def test_algorithm_seeds_do_not_collide():
    seeds = {a: algorithm_seed(a) for a in ALGORITHMS}
    assert len(set(seeds.values())) == len(seeds), seeds


@_test()
def test_the_driver_does_not_seed_from_a_sorted_index():
    """Guards the fix: seeding from ``sorted(ALGORITHMS)`` couples every
    algorithm's random stream to which other algorithms are registered, so adding
    a rival silently re-seeds the rest and no earlier row can be reproduced.

    The needle is assembled at run time. Spelling it as one literal would place it
    in this file, and since the test greps its own source it would then match
    itself -- which is exactly what a first version of this test did. The
    manifest's ``seed_scheme`` note also quotes the v1 pattern in prose, so the
    check has to be on the code form, not on the bare name.
    """
    needle = "enumerate(" + "sorted(ALGORITHMS))"
    src = Path(__file__).read_text(encoding="utf-8")
    assert needle not in src, \
        "the driver seeds from the line-up again; adding a rival re-seeds the rest"
    assert "algorithm_seed(a) for a in algos" in src, \
        "the driver must take each algorithm's seed component from its name"
    assert "SEED_SCHEME_VERSION = 2" in src


@_test()
def test_changing_the_line_up_does_not_reseed_the_other_algorithms():
    """The behavioural form of the check above, and the one that actually matters:
    a rival's rows must be reproducible after the line-up around it changes."""
    import tempfile
    root = Path(tempfile.mkdtemp())
    common = ["run", "--dims", "10", "--runs", "2", "--fes-per-dim", "500",
              "--jobs", "2", "--functions", "1"]
    got = {}
    for tag, algos in (("pair", [TARGET, "jSO"]),
                       ("trio", [TARGET, "jSO", "sepCMAES"])):
        out = root / tag
        assert cmd_run(build_parser().parse_args(
            common + ["--algos", *algos, "--out", str(out)])) == 0
        d = pd.read_csv(out / "raw" / "results.csv", float_precision="round_trip")
        got[tag] = (d[d.Algorithm == "jSO"].sort_values("Run")["Error"]
                    .reset_index(drop=True))
    assert got["pair"].equals(got["trio"]), (
        "jSO's results changed when another algorithm was added to the line-up; "
        "its random stream is coupled to the registry")


@_test()
def test_planned_rivals_are_declared_and_not_silently_counted():
    for name in PLANNED_RIVALS:
        assert name not in ALGORITHMS, f"{name} is declared planned but registered"
    assert len(ALGORITHMS) == 1 + 6, \
        f"the line-up is ours + 6 implemented rivals, got {list(ALGORITHMS)}"
    # An unvalidated rival must be declared as implemented-but-unvalidated, not
    # as missing. Claiming L-SHADE-RSP is "not implemented" while 130 lines of it
    # sit in section 6 is the kind of stale warning that stops being read.
    assert "L-SHADE-RSP" in UNVALIDATED_RIVALS, \
        "L-SHADE-RSP is implemented in section 6; it is unvalidated, not missing"
    assert not set(UNIMPLEMENTED_RIVALS) & set(UNVALIDATED_RIVALS)
    for name in UNVALIDATED_RIVALS:
        fn = name.replace("-", "_")
        assert fn in globals() and callable(globals()[fn]), \
            f"{name} is declared implemented but {fn} is not defined"


# ------------------------------------------------------- analysis and IO
@_test()
def test_class_ranks_place_counts_strict_betters_not_sort_position():
    """NEW. On the unimodal class every algorithm that reaches the floor ties.
    Reading a winner off the sort order invents a first place the data lacks."""
    df = pd.DataFrame([
        {"Algorithm": a, "Function": f, "Dimension": 10, "Run": r,
         "Error": 0.0, "Seconds": 0.0, "FES": 1, "Overrun": 0}
        for a in ("L-SHADE-DGR", "jSO", "LSHADE") for f in ("F1", "F3")
        for r in range(3)])
    wide, summary = class_ranks(df, 10, ["L-SHADE-DGR", "jSO", "LSHADE"])
    row = summary[summary.Class == "Unimodal"].iloc[0]
    assert row["control_place"] == 1 and row["tied_at_place"] == 3, row.to_dict()
    assert not row["separable"], "a 2-function class can never be separable"


@_test()
def test_class_ranks_refuses_a_critical_difference_it_cannot_support():
    """Nemenyi CD at N=2 exceeds the whole rank range, so nothing separates."""
    assert nemenyi_cd(7, 2) > 6.0
    assert FUNCTION_CLASSES["Unimodal"] == (1, 3)
    assert len(FUNCTION_CLASSES["Hybrid"]) == 10
    assert sum(len(v) for v in FUNCTION_CLASSES.values()) == 29


@_test()
def test_budget_of_reads_the_key_the_driver_actually_writes():
    """NEW. The analyser used to look for ``fes_per_dim``, which no driver ever
    wrote, so every convergence figure silently fell back to 3000*D."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "manifest.json").write_text(
        json.dumps({"max_fes_per_dim": {"10": 100000, "30": 300000}}),
        encoding="utf-8")
    assert budget_of(d, 10) == 100_000
    assert budget_of(d, 30) == 300_000
    assert budget_of(Path(tempfile.mkdtemp()), 10) == 100_000   # default 1e4*D


@_test()
def test_raw_csv_round_trips_float64_exactly():
    import tempfile
    vals = [1.9899181141865938, 3.986579112347158, 1e-300, 0.0]
    p = Path(tempfile.mkdtemp()) / "r.csv"
    pd.DataFrame({"Error": vals}).to_csv(p, index=False)
    back = pd.read_csv(p, float_precision="round_trip")["Error"].tolist()
    assert back == vals, f"{back} != {vals}"


@_test()
def test_analyze_uses_round_trip_parsing():
    src = Path(__file__).read_text(encoding="utf-8")
    assert src.count('float_precision="round_trip"') >= 2, (
        "the default CSV parser is off by up to one ULP, which matters when "
        "algorithms are separated at 1e-20")


# ------------------------------------------------------------- the driver
@_test()
def test_the_driver_refuses_to_resume_across_a_seed_scheme_change():
    import tempfile
    d = Path(tempfile.mkdtemp()) / "out"
    (d / "raw").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"seed_scheme_version": 1}),
                                     encoding="utf-8")
    (d / "raw" / "results.csv").write_text("Algorithm\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["run", "--dims", "10", "--runs", "1", "--functions", "1",
         "--jobs", "1", "--resume", "--out", str(d)])
    assert cmd_run(args) == 2, "a v1 results file must not be resumed under v2"


@_test()
def test_the_driver_refuses_to_resume_across_a_hyperparameter_change():
    """The seed check covers the streams; this covers the algorithm.

    ``ALGORITHMS[name]`` is called with no keyword arguments and the raw CSV
    records no hyperparameters, so a changed default would append rows from a
    different algorithm under the same name, invisibly. The recorded manifest is
    the only place that difference can be seen.
    """
    import tempfile
    d = Path(tempfile.mkdtemp()) / "out"
    (d / "raw").mkdir(parents=True)
    stale = dict(target_defaults())
    stale["POP_MAX"] = repr(9999)          # what a careless edit would look like
    (d / "manifest.json").write_text(
        json.dumps({"seed_scheme_version": SEED_SCHEME_VERSION, "runs": 1,
                    "target_defaults": stale}), encoding="utf-8")
    # A real header, not a stub: the accepting branch below reaches the point
    # where the driver reads the four key columns back out of the file.
    (d / "raw" / "results.csv").write_text(",".join(RAW_COLUMNS) + "\n",
                                           encoding="utf-8")
    argv = ["run", "--dims", "10", "--runs", "1", "--functions", "1",
            "--jobs", "1", "--fes-per-dim", "200", "--algos", TARGET,
            "--resume", "--out", str(d)]
    assert cmd_run(build_parser().parse_args(argv)) == 2, \
        "results measured under different defaults must not be resumed"

    # ...and the same file with the true defaults is accepted, so the guard is
    # refusing the mismatch rather than refusing everything.
    (d / "manifest.json").write_text(
        json.dumps({"seed_scheme_version": SEED_SCHEME_VERSION, "runs": 1,
                    "target_defaults": target_defaults()}), encoding="utf-8")
    assert cmd_run(build_parser().parse_args(argv)) == 0, \
        "a matching manifest must still resume"


@_test(slow=True)
def test_recorded_gate_rows_still_reproduce():
    """The strongest guard available: re-run cells and compare to the CSV.

    The defaults test pins the inputs and the literal test pins three short runs.
    This pins the thing that actually matters -- that the algorithm in this file
    is still the one that produced ``results_cec2017/``, at full competition
    budget. Six cells across three function classes; a change anywhere in the
    mutation, selection, credit or restart path moves at least one of them.

    Skipped rather than failed when the results directory is absent, so a fresh
    clone is not blocked by data it does not have.
    """
    out = Path("results_cec2017")
    raw = out / "raw" / "results.csv"
    if not raw.exists():
        return
    df = pd.read_csv(raw, float_precision="round_trip")
    sel = df[(df.Algorithm == TARGET) & (df.Dimension == 10)]
    if sel.empty:
        return
    max_fes = int(budget_of(out, 10))
    code = algorithm_seed(TARGET)
    for fid, run in ((4, 0), (4, 7), (11, 0), (16, 3), (15, 11), (30, 2)):
        rec = sel[(sel.Function == f"F{fid}") & (sel.Run == run)]["Error"]
        if not len(rec):
            continue
        pr = _cec(fid, 10)
        tr = Tracker(pr, max_fes, N_POINTS)
        LSHADE_DGR(tr, 10, (pr.lb, pr.ub), max_fes,
                   np.random.default_rng([SEED_BASE, 10, run, code]))
        got = tr.finalize()[1]
        assert got == float(rec.iloc[0]), (
            f"F{fid} run {run} no longer reproduces: {got!r} != recorded "
            f"{float(rec.iloc[0])!r}. The algorithm has changed since the gate.")


@_test()
def test_the_driver_refuses_to_overwrite_results_without_fresh():
    import tempfile
    d = Path(tempfile.mkdtemp()) / "out"
    (d / "raw").mkdir(parents=True)
    (d / "raw" / "results.csv").write_text("Algorithm\n", encoding="utf-8")
    args = build_parser().parse_args(
        ["run", "--dims", "10", "--runs", "1", "--functions", "1",
         "--jobs", "1", "--out", str(d)])
    assert cmd_run(args) == 2, "an existing results file must not be deleted silently"


@_test()
def test_the_manifest_describes_the_whole_file_not_the_last_invocation():
    """A gate is built one dimension at a time. If the manifest only recorded the
    call that wrote it last, a reader would see ``dims: [30]`` beside results for
    two dimensions, and the recorded budget for the missing dimension would be
    gone -- which is what ``budget_of`` reads to label every convergence figure.
    """
    import tempfile
    out = Path(tempfile.mkdtemp()) / "acc"
    common = ["run", "--runs", "1", "--functions", "1", "--fes-per-dim", "200",
              "--jobs", "2", "--algos", "jSO", "--out", str(out)]
    assert cmd_run(build_parser().parse_args(common + ["--dims", "10"])) == 0
    assert cmd_run(build_parser().parse_args(
        common + ["--dims", "30", "--resume"])) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["dims"] == [10, 30], m["dims"]
    assert set(m["max_fes_per_dim"]) == {"10", "30"}, m["max_fes_per_dim"]
    assert len(m["history"]) == 2 and m["history"][1]["resumed"] is True
    assert budget_of(out, 10) == 2000 and budget_of(out, 30) == 6000


@_test()
def test_resume_refuses_to_pool_different_run_counts():
    import tempfile
    out = Path(tempfile.mkdtemp()) / "mix"
    common = ["run", "--dims", "10", "--functions", "1", "--fes-per-dim", "200",
              "--jobs", "2", "--algos", "jSO", "--out", str(out)]
    assert cmd_run(build_parser().parse_args(common + ["--runs", "2"])) == 0
    assert cmd_run(build_parser().parse_args(
        common + ["--runs", "3", "--resume"])) == 2, \
        "a 2-run cell and a 3-run cell must not be pooled"


@_test(slow=True)
def test_splitting_the_run_by_function_changes_nothing():
    """No seed contains a function or shard term, so a gate split across jobs and
    a single-process gate must agree bit-for-bit. If they ever did not, the two
    tables would disagree and neither would look wrong on its face."""
    import tempfile
    root = Path(tempfile.mkdtemp())
    common = ["run", "--dims", "10", "--runs", "2", "--fes-per-dim", "1000",
              "--jobs", "2", "--algos", TARGET, "jSO"]
    frames = []
    for fid in ("1", "4"):
        out = root / f"shard{fid}"
        assert cmd_run(build_parser().parse_args(
            common + ["--functions", fid, "--out", str(out)])) == 0
        frames.append(pd.read_csv(out / "raw" / "results.csv",
                                  float_precision="round_trip"))
    single = root / "single"
    assert cmd_run(build_parser().parse_args(
        common + ["--functions", "1", "4", "--out", str(single)])) == 0
    b = pd.read_csv(single / "raw" / "results.csv", float_precision="round_trip")

    key = ["Algorithm", "Function", "Dimension", "Run"]
    a = pd.concat(frames).sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    assert a[key].equals(b[key]), "sharded and single runs cover different cells"
    assert a["Error"].equals(b["Error"]), (
        "sharded results differ from single-process results:\n"
        + a.merge(b, on=key, suffixes=("_shard", "_single"))
           .query("Error_shard != Error_single").to_string())


# --------------------------------------------------------------- harness
def cmd_test(args) -> int:
    only = getattr(args, "only", None)
    full = getattr(args, "full", False)
    selected = [t for t in _TESTS
                if (full or not t.slow) and (not only or only in t.__name__)]
    skipped = len(_TESTS) - len(selected)

    print(f"[*] {len(selected)} tests"
          + (f", {skipped} skipped (slow -- pass --full)" if skipped else ""))
    failures, t0 = [], time.perf_counter()
    for t in selected:
        ts = time.perf_counter()
        try:
            t()
            print(f"  ok    {t.__name__}  ({time.perf_counter()-ts:.2f}s)", flush=True)
        except AssertionError as e:
            failures.append((t.__name__, str(e) or "assertion failed"))
            print(f"  FAIL  {t.__name__}: {e}", flush=True)
        except Exception as e:                                  # noqa: BLE001
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}", flush=True)

    print(f"\n[*] {len(selected) - len(failures)}/{len(selected)} passed in "
          f"{time.perf_counter()-t0:.1f}s")
    if failures:
        print(f"[X] {len(failures)} failed:")
        for name, msg in failures:
            print(f"    {name}: {msg}")
        return 1
    print("[ok] every test passed")
    return 0


def cmd_selfcheck(args) -> int:
    """The fast subset, kept as a pre-flight before a long run."""
    args.full = False
    args.only = None
    return cmd_test(args)


# ==========================================================================
# 13. THE END-TO-END EXPERIMENT
# ==========================================================================
# One command, start to finish: `python lshade_dgr_study.py experiment`.
#
# WHY A PIPELINE AND NOT A SCRIPT. The stages here cost between a second and
# nine hours, and the long ones have already been interrupted once by a machine
# restart. So every stage is idempotent and resumable: it records that it
# finished in `state.json`, re-running skips what is done, and a stage that was
# cut off mid-way resumes from its own results file rather than from the start.
# Nothing here recomputes a row that exists.
#
# WHY THE STAGES ARE ORDERED THIS WAY. Each one is a gate on the next, so a
# failure stops the pipeline instead of propagating into a day of compute:
#
#   preflight  the environment the recorded rows were measured under. opfunu
#              1.0.1 and 1.0.4 define 13 of the 29 functions differently, so
#              this is part of the result, not trivia.
#   test       the full correctness suite, including the three that pin the
#              target's recorded behaviour bit-for-bit. If the algorithm has
#              drifted, every comparison below is meaningless.
#   tune       DIV_PRICE, on the COMPOSITION functions F21-F30 -- disjoint from
#              the functions every later stage measures on. Tuning a parameter
#              on the test set is the one methodological error that would
#              invalidate the whole study, so the split is structural, not a
#              matter of remembering.
#   screen     the candidate arms at D=50 only, where the collapse is. D=30 is
#              skipped deliberately: POP_UNCAP is a measured no-op below D=42,
#              and the screening's job is to choose, not to publish.
#   select     one winner, by a rule fixed before the numbers exist.
#   gate       the winner on the full 29-function protocol, resumed into the
#              existing results, so it is compared against the six rivals under
#              exactly the seeds and budgets they were measured with.
#   analyze    twice: the published k=7 line-up untouched, and k=8 with the
#              candidate. A rank, a Holm family and a critical difference are
#              all functions of k, so the two must not be mixed.
#   report     the binding criterion, applied and written down whichever way it
#              falls.

#: Where the pipeline keeps its own state. The results directories are the
#: expensive artefacts and are never owned by the pipeline.
PIPELINE_DIR = Path("results_pipeline")

#: Where the measurements live. The gate, the analysis and the report all read
#: and write here, and every one of them takes it as a parameter so a complete
#: study can be built in a directory of its own -- which is what makes the
#: numbers in a paper reproducible by someone who has only this file and no
#: directory they are expected to already possess.
RESULTS_DIR = Path("results_cec2017")

#: The dimensions a study is built at. D=100 is excluded deliberately, not
#: forgotten: no CEC'2017 paper this study compares against reports it, it cost
#: 125 core-hours for one algorithm when it was measured once, and the project
#: records it as a stated limitation.
STUDY_DIMS = (10, 30, 50)

#: The candidate arms, measured at D=50. Each is a keyword override of the
#: target; the empty one is the recorded algorithm and is read from the gate
#: rather than recomputed.
#:
#: THE FIRST ROUND ANSWERED ONE QUESTION. Is the leader step harmful: yes,
#: decisively -- hybrid class rank at D=50 went 4.700 -> 2.200 and the arm beat
#: the recorded algorithm on 10 of 10 hybrid functions under Holm. What it left
#: is a candidate that leads the two heterogeneous classes and sits fifth of
#: eight on Simple multimodal, and that gap is not a mystery. Binomial crossover
#: is equivariant only under signed permutations of the coordinate axes, F4, F5
#: and F7 are all f(R(x - o)) for a general rotation R, and the eigen crossover
#: that exists to restore equivariance cannot deliver it at any probability the
#: adaptive rules can reach, because both clip to [0.1, 0.9] and a mixture of an
#: equivariant channel with a non-equivariant one is equivariant only at p = 1.
#: These arms test that explanation and the three things that follow from it.
#:
#: Two arms repeat configurations already measured (no-leader, core-only). They
#: re-establish the reference point inside this screening's own function set,
#: which now covers the whole Simple multimodal class, so nothing is compared
#: across two different sets of cells.
SCREEN_ARMS: dict[str, dict] = {
    # -- the recorded fix, as the reference point --------------------------
    "no-leader":           dict(POP_UNCAP=True, OPS=(0, 3)),
    "no-leader-no-guard":  dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False),
    # -- the rotation hypothesis. p = 1 is the only equivariant setting, and
    #    p = 0.9 is the sharpest control there is for it: one step away, and
    #    the only step that matters. ----------------------------------------
    "eig-p1":              dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                                P_EIG_RULE="fixed", P_EIG=1.0),
    "eig-p09":             dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                                P_EIG_RULE="fixed", P_EIG=0.9),
    # -- and the three things that make the basis worth rotating into ------
    "eig-p1-rsp":          dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                                P_EIG_RULE="fixed", P_EIG=1.0, POP_RULE="rsp"),
    "eig-p1-mem":          dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                                P_EIG_RULE="fixed", P_EIG=1.0, EIG_MEMORY=0.8),
    "eig-p1-gate3":        dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                                P_EIG_RULE="fixed", P_EIG=1.0,
                                EIGEN_GATE_EXTRA=2.0),
    "align":               dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                                P_EIG_RULE="align"),
    # -- controls, each for a claim that would otherwise stay a guess ------
    #    core-only:  the Levy operator measured 0 better / 12 no-difference /
    #      0 worse against no-leader-no-guard at 15 runs, and the per-function
    #      oracle over the two arms is 1.01x, so it is a candidate for removal
    #      on parsimony rather than on performance.
    #    no-restart: measured identical to no-leader at D=50, rank and count.
    #    cr-nofloor: CR_FLOOR raises CR in the first half of the budget, the
    #      coordinate-basis deficit is proportional to (1 - CR), so it
    #      interacts with every eigen arm above. PRIOR_ART section 2 records
    #      disabling it as an improvement (2685 -> 2350) that was never
    #      measured at competition protocol.
    "core-only":           dict(POP_UNCAP=True, OPS=(0,)),
    "no-restart":          dict(POP_UNCAP=True, OPS=(0, 3), RESTART=False),
    "cr-nofloor":          dict(POP_UNCAP=True, OPS=(0, 3), CR_FLOOR=False),
    # -- the dimension-conditional portfolio ------------------------------
    #    Listed here so its configuration lives in the source rather than only
    #    in a pipeline state file, but it is DEGENERATE AT SCREEN_DIM: above
    #    the threshold it is bit-identical to `core-only`, which is the arm
    #    immediately above. Screening the two at D=50 would compare an arm with
    #    itself. The difference lives at D=10, where it is bit-identical to the
    #    recorded algorithm instead -- both halves are asserted by
    #    test_the_dimension_conditional_portfolio_is_the_configuration_it_claims.
    #    That is why this arm goes straight to the gate: every dimension the
    #    suite runs has already been measured under one of the two
    #    configurations it reduces to, so there is nothing left for a screening
    #    to decide. The gate still has to run, because algorithm_seed derives
    #    the random stream from the algorithm's name and the recorded rows
    #    therefore cannot simply be relabelled.
    "dim-cond":            dict(POP_UNCAP=True, OPS_DROP_ABOVE=10),
    #    The population rule, measured ON ITS OWN. L-SHADE-RSP's sizing was
    #    tried once before, in an arm that also carried P_EIG=1.0, and that
    #    change was refuted -- so the population half has never been measured
    #    unconfounded. It is worth one arm: at D=50 this study carries the
    #    smallest population of any DE rival in the line-up, 600 against jSO's
    #    692 and L-SHADE's 900, and the rsp rule gives 1018.
    "dim-cond-rsp":        dict(POP_UNCAP=True, OPS_DROP_ABOVE=10,
                                POP_RULE="rsp"),
}

#: The whole Simple multimodal class, which these arms target, plus the whole
#: hybrid class, which they must not damage. Screening a fix only on the
#: functions it is meant to help produces a confident wrong answer -- the
#: component diagnostic says so in its own header, and it is why both groups
#: are here rather than the two multimodals the first round could afford.
SCREEN_FUNCTIONS = tuple(range(4, 21))
SCREEN_DIM = 50

#: Named before the numbers exist. The arms above were chosen to move the first
#: class; the temptation once the table is on screen is to read whichever class
#: happened to move, and naming both here is what makes that visible.
SELECT_TARGET_CLASS = "Simple multimodal"
SELECT_GUARD_CLASS = "Hybrid"

#: Runs per configuration for the operator-selection correlation. Small on
#: purpose: the quantity being estimated is a mean budget share over hundreds
#: of generations, which is far more stable than a final error, and the
#: correlation is taken over functions rather than over runs.
AOS_RUNS = 5

#: Disjoint from SCREEN_FUNCTIONS and from both classes the selection reads.
TUNE_FUNCTIONS = tuple(range(21, 31))
TUNE_DIM = 50
#: ALIGN_TAU, the misalignment threshold the "align" rule switches on. The
#: statistic lies in [0, 1), and 0.0 switches the eigen channel on
#: unconditionally -- which is the same algorithm as `eig-p1`. That is exactly
#: why it is the low end of the grid: if no positive threshold beats it, the
#: gate is not carrying its weight and the honest report says so.
TUNE_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)


def _pipeline_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _stage_banner(n: int, total: int, name: str, note: str = "") -> None:
    print(f"\n{'=' * 78}")
    print(f"STAGE {n}/{total}  {name}" + (f"   -- {note}" if note else ""))
    print("=" * 78, flush=True)


def stage_preflight(state: dict, out=None) -> int:
    """The environment is part of the result, so it is checked, not assumed."""
    ok = True
    py_ver = sys.version.split()[0]
    print(f"  python  {py_ver}")
    if not py_ver.startswith("3.14"):
        print(f"  [X] the recorded rows were measured under 3.14; bare `python` "
              f"resolves to a 3.12 install with no numpy on this machine.")
        ok = False
    opf = _dep_version("opfunu")
    print(f"  opfunu  {opf}")
    if opf < "1.0.4":
        print(f"  [X] opfunu must be >= 1.0.4: 1.0.1 defines 13 of the 29 "
              f"CEC'2017 functions differently and returns NaN for F18 and F30 "
              f"at D=10.")
        ok = False
    for mod, ver in (("numpy", np.__version__), ("pandas", pd.__version__),
                     ("scipy", _dep_version("scipy")), ("cma", _dep_version("cma"))):
        print(f"  {mod:7s} {ver}")
    gate = Path(out) if out is not None else Path(RESULTS_DIR)
    if not (gate / "raw" / "results.csv").exists():
        # Not an error. An empty results directory is how a complete
        # study is built from nothing, and the gate measures the whole
        # line-up when it finds one.
        print(f"  gate    {gate} is empty; the line-up will be measured "
              f"from nothing")
    else:
        n = len(pd.read_csv(gate / "raw" / "results.csv",
                            float_precision="round_trip"))
        print(f"  gate    {n} recorded rows")
        backup = gate.with_name(gate.name + "_BACKUP")
        print(f"  backup  {'present' if backup.exists() else 'MISSING'}")
        if not backup.exists():
            print(f"  [!] no backup of {n} rows. This is hours of compute "
                  f"with no version control behind it; copy it before the "
                  f"gate stage.")
    state["stages"]["preflight"] = {"ok": ok, "python": py_ver, "opfunu": opf}
    return 0 if ok else 1


def stage_test(args) -> int:
    """The correctness suite, including the recorded-behaviour pins."""
    t_args = argparse.Namespace(full=True, only=None)
    return cmd_test(t_args)


def _arm_task(name, kw, fid, dim, run, max_fes):
    problem = make_suite_problem("cec2017", fid, dim)
    tracker = Tracker(problem, max_fes, 10)
    rng = np.random.default_rng([SEED_BASE, dim, run, ABLATE_SALT])
    t0 = time.perf_counter()
    LSHADE_DGR(tracker, dim, (problem.lb, problem.ub), max_fes, rng, **kw)
    return {"Config": name, "Function": f"F{fid}", "Dimension": int(dim),
            "Run": int(run), "Error": tracker.finalize()[1],
            "Seconds": time.perf_counter() - t0, "FES": tracker.fes,
            "Overrun": tracker.overrun}


def _run_arms(arms, functions, dim, runs, out_csv, jobs, label, fes_per_dim=None):
    """Common random numbers across arms, incremental flush, resumable."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if out_csv.exists():
        rec = pd.read_csv(out_csv, float_precision="round_trip")
        done = set(map(tuple, rec[["Config", "Function", "Dimension",
                                   "Run"]].to_numpy()))
        print(f"  [resume] {len(done)} runs already recorded")
    max_fes = int(fes_per_dim * dim) if fes_per_dim else int(SUITES["cec2017"].max_fes(dim))
    tasks = [(n, arms[n], f, dim, r, max_fes)
             for n in arms for f in functions for r in range(runs)
             if (n, f"F{f}", dim, r) not in done]
    total = len(arms) * len(functions) * runs
    print(f"  {label}: {len(arms)} arms x {len(functions)} functions x {runs} "
          f"runs = {total}; {len(tasks)} to do, jobs={jobs}")
    if not tasks:
        return 0
    buf, t0, n = [], time.perf_counter(), 0

    def flush(force=False):
        if buf and (force or len(buf) >= FLUSH_ROWS):
            pd.DataFrame(buf)[ABLATE_COLUMNS].to_csv(
                out_csv, mode="a", header=not out_csv.exists(), index=False)
            buf.clear()

    try:
        with Parallel(n_jobs=jobs, backend="loky",
                      return_as=_generator_mode()) as parallel:
            for row in parallel(delayed(_arm_task)(*t) for t in tasks):
                buf.append(row)
                n += 1
                if n % 200 == 0:
                    el = time.perf_counter() - t0
                    print(f"    {n}/{len(tasks)}  {el/60:.1f} min elapsed, "
                          f"~{el/n*(len(tasks)-n)/60:.0f} min left", flush=True)
                flush()
    finally:
        flush(force=True)
    print(f"  {n} runs in {(time.perf_counter()-t0)/60:.1f} min -> {out_csv}")
    return 0


def stage_tune(state, jobs, runs, fes=None, pdir=None) -> int:
    """Choose ALIGN_TAU on functions no later stage measures on.

    The screening and both selection classes live on F4-F20. This tunes on the
    composition class instead, so the value is not chosen by looking at the
    numbers it will later be judged by. That is the whole point and it is
    enforced by the constants, not by discipline.
    """
    csv = (pdir or PIPELINE_DIR) / "tune" / "results.csv"
    arms = {f"tau={v}": dict(POP_UNCAP=True, OPS=(0, 3), DIV_GUARD=False,
                             P_EIG_RULE="align", ALIGN_TAU=v)
            for v in TUNE_GRID}
    rc = _run_arms(arms, TUNE_FUNCTIONS, TUNE_DIM, runs, csv, jobs, "tune", fes)
    if rc:
        return rc
    d = pd.read_csv(csv, float_precision="round_trip")
    piv = d.pivot_table(index="Config", columns="Function", values="Error",
                        aggfunc="median")
    ranks = pd.DataFrame(
        np.apply_along_axis(sps.rankdata, 0, piv.to_numpy(dtype=float)),
        index=piv.index, columns=piv.columns).mean(axis=1).sort_values()
    print("\n  mean rank over the composition class (1 = best):")
    print("   " + ranks.to_string().replace("\n", "\n   "))
    best = float(str(ranks.index[0]).split("=")[1])
    print(f"\n  [ok] ALIGN_TAU = {best}, chosen on F21-F30 and now frozen.")
    if best == 0.0:
        print(f"      tau = 0.0 switches the eigen channel on unconditionally, "
              f"which is the same algorithm as the `eig-p1` arm. If that is "
              f"what wins here, the threshold is not carrying its weight and "
              f"the report says so rather than dressing it up.")
    state["stages"]["tune"] = {"align_tau": best,
                               "grid": list(TUNE_GRID),
                               "functions": list(TUNE_FUNCTIONS),
                               "mean_ranks": {str(k): float(v)
                                              for k, v in ranks.items()}}
    return 0


def stage_screen(state, jobs, runs, fes=None, pdir=None) -> int:
    """Every candidate arm at D=50, against the recorded algorithm."""
    tau = state["stages"].get("tune", {}).get("align_tau", 0.0)
    arms = {k: dict(v) for k, v in SCREEN_ARMS.items()}
    arms["align"]["ALIGN_TAU"] = tau
    arms["baseline"] = {}
    csv = (pdir or PIPELINE_DIR) / "screen" / "results.csv"
    rc = _run_arms(arms, SCREEN_FUNCTIONS, SCREEN_DIM, runs, csv, jobs, "screen", fes)
    if rc:
        return rc
    d = pd.read_csv(csv, float_precision="round_trip")
    bad = d[d["Overrun"] != 0]
    if len(bad):
        print(f"  [X] {len(bad)} runs exceeded the budget; the tables are not usable")
        return 1
    state["stages"]["screen"] = {"arms": list(arms), "align_tau": tau,
                                 "rows": int(len(d))}
    return 0


def _screen_table(csv: Path, runs: int):
    """Mean rank per arm on the target and guard classes, and each vs baseline.

    Two classes, not one. The arms are chosen to move SELECT_TARGET_CLASS, and
    an arm that moves it by damaging SELECT_GUARD_CLASS has not found anything
    -- it has traded one class for another, which the gate would then measure as
    a regression after hours of compute. Both columns are computed here so the
    selection rule can require both, and both class names are module constants
    fixed before the numbers existed.
    """
    d = pd.read_csv(csv, float_precision="round_trip")
    target = [f"F{k}" for k in FUNCTION_CLASSES[SELECT_TARGET_CLASS]]
    guard = [f"F{k}" for k in FUNCTION_CLASSES[SELECT_GUARD_CLASS]]
    piv = d.pivot_table(index="Function", columns="Config", values="Error",
                        aggfunc="median")
    ranks = pd.DataFrame(
        np.apply_along_axis(sps.rankdata, 1, piv.to_numpy(dtype=float)),
        index=piv.index, columns=piv.columns)
    summary = pd.DataFrame({
        "mean_rank_all": ranks.mean(axis=0),
        "mean_rank_target": ranks.loc[[f for f in target if f in ranks.index]].mean(axis=0),
        "mean_rank_guard": ranks.loc[[f for f in guard if f in ranks.index]].mean(axis=0),
        "median_target": piv.loc[[f for f in target if f in piv.index]].median(axis=0),
        "median_guard": piv.loc[[f for f in guard if f in piv.index]].median(axis=0),
    })
    rows = []
    for cfg in piv.columns:
        if cfg == "baseline":
            continue
        ps, a12s, better = [], [], 0
        for f in piv.index:
            A = d[(d.Config == cfg) & (d.Function == f)].sort_values("Run")["Error"].to_numpy()
            B = d[(d.Config == "baseline") & (d.Function == f)].sort_values("Run")["Error"].to_numpy()
            if A.size != B.size or A.size == 0:
                continue
            a12 = vargha_delaney_a12(A, B)
            a12s.append(a12)
            ps.append(paired_wilcoxon(A, B))
            better += a12 > 0.5
        if ps:
            rows.append({"Config": cfg, "funcs_better": better,
                         "funcs_total": len(ps), "mean_A12": float(np.mean(a12s)),
                         "n_sig_holm": int((holm(np.array(ps)) < ALPHA).sum())})
    return summary.sort_values("mean_rank_target"), pd.DataFrame(rows)


def stage_select(state, pdir=None) -> int:
    """One winner, by a rule fixed before the numbers existed.

    Primary: mean Friedman rank over SELECT_TARGET_CLASS at D=50, the class
    these arms were built to move. Ties are broken toward the arm with fewer
    changes from the recorded algorithm, because a simpler configuration that
    measures the same is the one that can be defended.

    Second, and a hard filter rather than a tie-break: an arm whose
    SELECT_GUARD_CLASS rank is worse than the baseline's is refused outright,
    however well it does on the target. Trading the class we lead for the class
    we lag is not an improvement, and the gate would take hours to say so. This
    is strictly harder than the rule it replaces, which read one class alone.
    """
    pdir = pdir or PIPELINE_DIR
    csv = pdir / "screen" / "results.csv"
    summary, effects = _screen_table(csv, state["stages"]["screen"])
    print(f"\n  arms by mean rank over {SELECT_TARGET_CLASS!r} "
          f"(1 = best; guard class is {SELECT_GUARD_CLASS!r}):")
    print("   " + summary.to_string(float_format=lambda x: f"{x:.4f}")
          .replace("\n", "\n   "))
    print("\n  each arm against the recorded algorithm:")
    print("   " + effects.to_string(index=False, float_format=lambda x: f"{x:.4f}")
          .replace("\n", "\n   "))
    (pdir / "screen" / "summary.csv").write_text(
        summary.to_csv(), encoding="utf-8")
    effects.to_csv(pdir / "screen" / "effects.csv", index=False)

    ordered = [c for c in summary.index if c != "baseline"]
    if not ordered:
        print("  [X] no arm to select")
        return 1
    if "baseline" not in summary.index:
        print("  [X] the recorded algorithm is missing from the screening; "
              "there is nothing to select against")
        return 1
    base_target = float(summary.loc["baseline", "mean_rank_target"])
    base_guard = float(summary.loc["baseline", "mean_rank_guard"])

    # THE GUARD CLASS IS A FILTER, NOT A TIE-BREAK. Applied before the target
    # is read at all, so an arm cannot buy its way past it with a large enough
    # win elsewhere.
    kept = [c for c in ordered
            if summary.loc[c, "mean_rank_guard"] <= base_guard + 1e-9]
    refused = [c for c in ordered if c not in kept]
    if refused:
        print(f"\n  refused on the guard class {SELECT_GUARD_CLASS!r} "
              f"(baseline {base_guard:.4f}):")
        for c in refused:
            print(f"    {c:22s} guard {summary.loc[c, 'mean_rank_guard']:.4f}"
                  f"   target {summary.loc[c, 'mean_rank_target']:.4f}")

    # AN ARM MUST BEAT THE RECORDED ALGORITHM TO BE PROMOTED. Taking the best
    # of the candidates is not the same as finding an improvement: if every arm
    # is worse, the best of them is still worse, and promoting it would spend
    # hours of gate compute to measure a regression. "No candidate improved on
    # what we already have" is a result, so it is reported as one and the
    # pipeline stops here rather than continuing.
    beats = [c for c in kept
             if summary.loc[c, "mean_rank_target"] < base_target - 1e-9]
    if not beats:
        best_seen = ordered[0]
        print(f"\n  [X] no arm improved on the recorded algorithm on "
              f"{SELECT_TARGET_CLASS!r} without damaging "
              f"{SELECT_GUARD_CLASS!r} (baseline target {base_target:.4f}, "
              f"guard {base_guard:.4f}; best candidate {best_seen!r} at "
              f"{summary.loc[best_seen, 'mean_rank_target']:.4f}).")
        print(f"      That is the measured answer, not a failure of the run. "
              f"Nothing is promoted to the gate.")
        state["stages"]["select"] = {
            "winner": None,
            "target_class": SELECT_TARGET_CLASS,
            "guard_class": SELECT_GUARD_CLASS,
            "baseline_target_rank": base_target,
            "baseline_guard_rank": base_guard,
            "refused_on_guard": refused,
            "best_candidate": best_seen,
            "best_candidate_rank": float(summary.loc[best_seen, "mean_rank_target"]),
            "verdict": "no candidate improved the target class without "
                       "damaging the guard class"}
        return 3

    n_changes = {c: len(SCREEN_ARMS.get(c, {})) for c in beats}
    best_rank = min(float(summary.loc[c, "mean_rank_target"]) for c in beats)
    tied = [c for c in beats
            if summary.loc[c, "mean_rank_target"] <= best_rank + 1e-9]
    winner = min(tied, key=lambda c: (n_changes.get(c, 99), c))
    print(f"\n  [ok] winner: {winner!r}  ({SELECT_TARGET_CLASS} rank "
          f"{summary.loc[winner, 'mean_rank_target']:.4f} vs baseline "
          f"{base_target:.4f}; guard {summary.loc[winner, 'mean_rank_guard']:.4f} "
          f"vs {base_guard:.4f})")
    if len(tied) > 1:
        print(f"       tied with {[c for c in tied if c != winner]}; broken "
              f"toward fewer changes from the recorded algorithm")
    state["stages"]["select"] = {
        "winner": winner, "config": SCREEN_ARMS.get(winner, {}),
        "target_class": SELECT_TARGET_CLASS,
        "guard_class": SELECT_GUARD_CLASS,
        "target_rank": float(summary.loc[winner, "mean_rank_target"]),
        "guard_rank": float(summary.loc[winner, "mean_rank_guard"]),
        "baseline_target_rank": base_target,
        "baseline_guard_rank": base_guard,
        "refused_on_guard": refused,
        "tied": tied}
    return 0


def _register_winner(state) -> str:
    """Put the winner in ALGORITHMS under a name that records what it is."""
    sel = state["stages"]["select"]
    name = f"L-SHADE-DGR2 ({sel['winner']})"
    if name not in ALGORITHMS:
        cfg = dict(sel["config"])
        tau = state["stages"].get("tune", {}).get("align_tau")
        if cfg.get("P_EIG_RULE") == "align" and tau is not None:
            cfg["ALIGN_TAU"] = tau
        ALGORITHMS[name] = functools.partial(LSHADE_DGR, **cfg)
    return name


def stage_gate(state, jobs, out=None, dims=None, fes=None) -> int:
    """The whole line-up on the full protocol, in one results directory.

    ONE PATH, NOT TWO, AND THE REASON MATTERS. An earlier version chose between
    "measure everything" and "measure only the candidate" by asking whether the
    results file existed. That is the wrong question, and on a multi-day run it
    is a trap: interrupt the gate once -- a Windows update restart inside the
    00:00-06:00 window is enough -- and the file exists on the next attempt, so
    the incremental path is taken, only the candidate is measured, and the
    rivals are left permanently incomplete. Worse, it fails quietly: the
    incremental path asks which dimensions every rival already covers, finds
    none, and returns an error that reads like a missing baseline rather than a
    half-finished run.

    So there is one path. The target is always the whole line-up at the study
    dimensions, and --resume is passed whenever a manifest exists. cmd_run then
    skips every cell already recorded, which covers all three cases with the
    same code: a complete study is built from nothing, an interrupted one is
    finished, and a round that only adds a candidate measures only what is
    missing. Nothing is restricted with --algos, so nothing is reported as a
    protocol override either.

    D=100 is excluded deliberately, not forgotten: no CEC'2017 paper this study
    compares against reports it, it cost 125 core-hours for one algorithm when
    it was measured once, and the project records it as a stated limitation.
    """
    name = _register_winner(state)
    print(f"  registered {name!r} -> {ALGORITHMS[name].keywords}")
    out = Path(out) if out is not None else Path(RESULTS_DIR)
    todo = [int(d) for d in (dims or STUDY_DIMS)]
    manifest = out / "manifest.json"
    argv = ["run", "--out", str(out), "--dims", *[str(d) for d in todo],
            "--jobs", str(jobs)]
    if manifest.exists():
        # Only with a manifest: cmd_run refuses --resume without one, and the
        # first invocation of a fresh study is the one that writes it.
        argv.insert(1, "--resume")
        print(f"  resuming {out}; recorded cells are skipped")
    else:
        print(f"  {out} is new: the whole line-up will be measured")
    # Only ever set for a smoke test of the wiring. cmd_run prints its own
    # "protocol overridden -- NOT comparable" banner when it is, so a shortened
    # run cannot be mistaken later for a measurement.
    if fes:
        argv += ["--fes-per-dim", str(fes)]
    print(f"  line-up ({len(ALGORITHMS)}): {', '.join(ALGORITHMS)}")
    print(f"  dimensions: {todo}")
    rc = cmd_run(build_parser().parse_args(argv))
    if rc:
        return rc

    # THE GATE IS NOT DONE UNTIL EVERY CELL IS THERE. Checked rather than
    # assumed, because a stage marked done is never revisited and a silently
    # short gate would be read as a result.
    want_runs = SUITES["cec2017"].runs
    df = pd.read_csv(out / "raw" / "results.csv",
                     usecols=["Algorithm", "Function", "Dimension", "Run"])
    df = df[df["Dimension"].isin(todo)]
    sizes = df.groupby(["Algorithm", "Function", "Dimension"]).size()
    n_funcs = len(SUITES["cec2017"].function_ids)
    expected = len(ALGORITHMS) * n_funcs * len(todo)
    short = sizes[sizes < want_runs]
    if len(sizes) < expected or len(short):
        print(f"  [X] the gate is incomplete: {len(sizes)} of {expected} cells "
              f"present, {len(short)} of them with fewer than {want_runs} runs. "
              f"Re-run to continue; nothing downstream is allowed to read a "
              f"partial gate as a result.")
        return 1
    state["stages"]["gate"] = {"algorithm": name,
                               "dimensions": todo,
                               "algorithms": list(ALGORITHMS),
                               "cells": int(len(sizes)),
                               "runs_per_cell": int(want_runs)}
    return 0


def stage_analyze(state, out=None) -> int:
    """Twice: the published line-up untouched, and k=8 with the candidate."""
    name = state["stages"]["gate"]["algorithm"]
    _register_winner(state)
    out = str(out if out is not None else RESULTS_DIR)
    published = [a for a in ALGORITHMS if a != name]
    print("  k=7: the published line-up, written to tables/ unchanged")
    rc = cmd_analyze(build_parser().parse_args(
        ["analyze", "--out", out, "--algos", *published, "--no-figures"]))
    if rc:
        return rc
    print(f"\n  k=8: the same six rivals plus {name!r}")
    rc = cmd_analyze(build_parser().parse_args(
        ["analyze", "--out", out, "--algos", *published, name,
         "--control", name]))
    if rc:
        return rc
    state["stages"]["analyze"] = {"k7": f"{out}/tables_k7",
                                  "k8": f"{out}/tables_k8"}
    return 0


def stage_aos(state, jobs, out=None, fes=None) -> int:
    """The correlation the mechanism section of the paper rests on.

    Its own stage rather than part of the analysis, because it measures
    configurations that are not in the line-up: the recorded three-operator
    algorithm and the same algorithm without the leader step. The gate compares
    finished algorithms; this explains why one of them was changed.
    """
    out = Path(out) if out is not None else Path(RESULTS_DIR)
    tdir = out / "tables_k8"
    tdir.mkdir(parents=True, exist_ok=True)
    # Tables with the tables, the figure with the figures.
    t = aos_correlation(tdir, SCREEN_FUNCTIONS, SCREEN_DIM,
                        AOS_RUNS, jobs, fes, fdir=out / "figures_k8")
    state["stages"]["aos"] = {"functions": list(SCREEN_FUNCTIONS),
                              "dimension": SCREEN_DIM, "runs": AOS_RUNS,
                              "paired": int(len(t))}
    return 0


def stage_report(state, pdir=None, out=None) -> int:
    """The binding criterion, applied and written down whichever way it falls."""
    name = state["stages"]["gate"]["algorithm"]
    out = Path(out) if out is not None else Path(RESULTS_DIR)
    k8 = out / "tables_k8"
    lines = ["# Result", "",
             f"Candidate: `{name}`",
             f"Configuration: `{state['stages']['select']['config']}`", ""]
    verdict = {}
    for dim in (10, 30, 50):
        f = k8 / f"class_ranks_{dim}D.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f, index_col=0)
        if name not in t.columns or TARGET not in t.columns:
            continue
        lines += [f"## D={dim} (k=8, recomputed)", "",
                  "| Class | recorded L-SHADE-DGR | candidate | change |",
                  "|---|---|---|---|"]
        for cls in t.index:
            was, now = float(t.loc[cls, TARGET]), float(t.loc[cls, name])
            arrow = "better" if now < was else ("same" if now == was else "worse")
            lines.append(f"| {cls} | {was:.3f} | {now:.3f} | {arrow} |")
            verdict[(dim, cls)] = (was, now)
        lines.append("")

    hyb50 = verdict.get((50, "Hybrid"))
    d10 = [(c, v) for (d, c), v in verdict.items() if d == 10]
    ok_hybrid = hyb50 is not None and hyb50[1] < hyb50[0]
    ok_d10 = all(now <= was + 1e-9 for _, (was, now) in d10) if d10 else False
    lines += ["## Binding criterion", "",
              f"- Hybrid class rank at D=50 improves: "
              f"**{'yes' if ok_hybrid else 'no'}**"
              + (f" ({hyb50[0]:.3f} -> {hyb50[1]:.3f})" if hyb50 else ""),
              f"- D=10 not made worse in any class: "
              f"**{'yes' if ok_d10 else 'no'}**", "",
              f"**Verdict: {'ADOPT' if (ok_hybrid and ok_d10) else 'REJECT'}**", "",
              "Both conditions are required. A rejection is a measured negative "
              "result and is reported as one; the criterion is not weakened to "
              "let a mechanism through.", ""]

    # THE SAME CRITERION, READ ON THE EVIDENCE INSTEAD OF ON THE DIRECTION.
    #
    # The condition above compares class ranks and calls any increase "worse",
    # with no null model at all. That is not strictness, it is a missing
    # baseline, and it is demonstrably broken: at D=10 the `dim-cond` candidate
    # IS the recorded algorithm -- OPS_DROP_ABOVE=10 does not fire at dim=10
    # and POP_UNCAP is inert below D=42 -- differing only in the random streams,
    # because algorithm_seed derives those from the algorithm's name. Measured
    # across all 29 functions at D=10, the two are indistinguishable: 0 of 29
    # differ under Holm, the smallest adjusted p is 1.0000, mean A12 is 0.5022.
    # Yet their Composition class rank differs by 0.800, and the condition
    # above reads that as "worse". So it rejects the recorded algorithm when
    # compared with itself, and both rounds it has rejected may have been
    # rejected on nothing.
    #
    # The fix is not a looser threshold, which would be the forbidden move. It
    # is to read the condition on the quantity this study already calls the
    # evidence: the per-function Holm tests. Section 3 of the handoff states it
    # plainly -- class ranks show direction, the per-function tests are the
    # evidence -- and the criterion was simply not written that way.
    #
    # BOTH VERDICTS ARE REPORTED, always, and the rank-based one keeps its
    # place above. Replacing it silently after it refused a candidate is
    # indistinguishable from fitting the criterion to the result, so nothing is
    # replaced: the reader gets the original, the corrected one, and the reason
    # they differ.
    pw = out / "tables_k8" / "pairwise_tests.csv"
    ok_d10_sig, sig_detail = None, []
    if pw.exists():
        p = pd.read_csv(pw)
        p = p[(p["Dimension"] == 10) & (p["Competitor"] == TARGET)
              & (p["Control"] == name)]
        if len(p):
            p = p.assign(Class=p["Function"].map(
                {f"F{i}": c for c, ids in FUNCTION_CLASSES.items() for i in ids}))
            worse = p[p["outcome"] == "-"]
            ok_d10_sig = worse.empty
            for cls in FUNCTION_CLASSES:
                n_cls = int((p["Class"] == cls).sum())
                n_bad = int((worse["Class"] == cls).sum()) if len(worse) else 0
                if n_cls:
                    sig_detail.append(f"| {cls} | {n_bad} of {n_cls} |")
    if ok_d10_sig is not None:
        lines += ["## The same criterion, read on the per-function tests", "",
                  "The condition above has no null model: it calls any increase "
                  "in a class rank \"worse\". Measured on this study's own data, "
                  "two gates of one algorithm at D=10 move a class rank by up "
                  "to 0.800 while not one of 29 functions differs under Holm, "
                  "so the rank condition can reject an algorithm compared with "
                  "itself. Read instead on the evidence -- functions where the "
                  "candidate is significantly worse than the recorded algorithm "
                  "at D=10, Holm-corrected within each cell:", "",
                  "| Class | significantly worse |", "|---|---|"] + sig_detail + [
                  "",
                  f"- D=10 not significantly worse in any class: "
                  f"**{'yes' if ok_d10_sig else 'no'}**", "",
                  f"**Verdict on the evidence: "
                  f"{'ADOPT' if (ok_hybrid and ok_d10_sig) else 'REJECT'}**", "",
                  "Both verdicts are reported and neither is withdrawn. If they "
                  "disagree, the disagreement is the finding: it says the "
                  "movement the rank condition saw was not large enough to be "
                  "detected on any single function.", ""]

    # THE TARGET CLASS IS REPORTED SEPARATELY AND CANNOT EXCUSE THE CRITERION.
    # The criterion above is the one this study registered before any of these
    # arms existed, and it is unchanged. What the current round of arms was
    # built to move is a different class, so it gets its own line rather than
    # being folded into the verdict -- an arm that moves the target and fails
    # the criterion has still failed the criterion, and one that passes the
    # criterion without moving the target has not done the thing it was built
    # for. Two answers, neither able to stand in for the other.
    tgt50 = verdict.get((50, SELECT_TARGET_CLASS))
    ok_target = tgt50 is not None and tgt50[1] < tgt50[0]
    lines += [f"## Target of this round: {SELECT_TARGET_CLASS}", "",
              f"- {SELECT_TARGET_CLASS} class rank at D=50 improves: "
              f"**{'yes' if ok_target else 'no'}**"
              + (f" ({tgt50[0]:.3f} -> {tgt50[1]:.3f})" if tgt50 else
                 " (not measured)"), "",
              "Reported beside the criterion, not inside it. This line records "
              "whether the rotation-equivariance argument delivered what it "
              "predicted; the verdict above records whether the candidate is "
              "adopted. They are allowed to disagree, and if they do, both are "
              "reported as they fell.", "",
              "Class ranks are compared inside one k=8 analysis, never against "
              "the published k=7 numbers: a Friedman rank, a Holm family and a "
              "Nemenyi critical difference are all functions of how many "
              "algorithms were compared."]
    path = (pdir or PIPELINE_DIR) / "RESULT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-14:]))
    print(f"\n  [ok] written to {path}")
    state["stages"]["report"] = {"adopt": bool(ok_hybrid and ok_d10),
                                 "adopt_on_evidence": (None if ok_d10_sig is None
                                                       else bool(ok_hybrid and ok_d10_sig)),
                                 "target_class": SELECT_TARGET_CLASS,
                                 "target_improved": bool(ok_target)}
    return 0


def _write_no_candidate_report(state, pdir) -> None:
    """A negative result is written down in the same place a positive one is."""
    sel = state["stages"]["select"]
    lines = [
        "# Result: no candidate was promoted", "",
        f"The screening measured every candidate arm against the recorded "
        f"algorithm at D=50 under common random numbers, ranked on the "
        f"**{sel['target_class']}** class and filtered on the "
        f"**{sel['guard_class']}** class. None improved the target without "
        f"damaging the guard.", "",
        f"- recorded algorithm, mean {sel['target_class']} rank: "
        f"**{sel['baseline_target_rank']:.4f}** "
        f"(guard {sel['baseline_guard_rank']:.4f})",
        f"- best candidate (`{sel['best_candidate']}`): "
        f"**{sel['best_candidate_rank']:.4f}**", "",
    ] + ([
        f"- refused on the guard class: "
        f"{', '.join('`' + c + '`' for c in sel['refused_on_guard'])}", "",
    ] if sel.get("refused_on_guard") else []) + [
        "The best of a set of candidates is not an improvement if every one of "
        "them is worse. Promoting it would have spent hours of gate compute to "
        "measure a regression, so the pipeline stopped here.", "",
        "This is a measured negative result and is reported as one. The "
        "screening data is in `screen/results.csv`, the per-arm comparison in "
        "`screen/effects.csv`.",
    ]
    path = Path(pdir) / "RESULT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_experiment(args) -> int:
    """Run every stage, in order, resuming whatever is already done."""
    pdir = Path(args.out) if getattr(args, "out", None) else PIPELINE_DIR
    rdir = Path(getattr(args, "results", None) or RESULTS_DIR)
    state_path = pdir / "state.json"
    state = _pipeline_state(state_path)
    jobs = args.jobs if args.jobs and args.jobs > 0 else (os.cpu_count() or 1)
    runs = args.runs or SUITES["cec2017"].runs
    only = set(args.only_stages or [])
    print(f"[*] pipeline state -> {pdir}")
    print(f"[*] measurements   -> {rdir}"
          + ("" if (rdir / "raw" / "results.csv").exists()
             else "   (empty: the whole line-up will be measured from nothing)"))
    stages = [
        ("preflight", lambda: stage_preflight(state, rdir), "environment and baseline"),
        ("test", lambda: stage_test(args), "the full correctness suite"),
        ("tune", lambda: stage_tune(state, jobs, runs, args.fes_per_dim, pdir),
         "ALIGN_TAU on F21-F30"),
        ("screen", lambda: stage_screen(state, jobs, runs, args.fes_per_dim, pdir),
         f"{len(SCREEN_ARMS)+1} arms at D={SCREEN_DIM}"),
        ("select", lambda: stage_select(state, pdir), "one winner"),
        ("gate", lambda: stage_gate(state, jobs, rdir, None, args.fes_per_dim),
         "the line-up on 29 functions"),
        ("analyze", lambda: stage_analyze(state, rdir), "k=7 and k=8"),
        ("aos", lambda: stage_aos(state, jobs, rdir, args.fes_per_dim),
         "the operator-selection correlation"),
        ("report", lambda: stage_report(state, pdir, rdir), "the binding criterion"),
    ]
    t0 = time.perf_counter()
    for i, (name, fn, note) in enumerate(stages, 1):
        if only and name not in only:
            continue
        if state["stages"].get(name, {}).get("done") and not args.redo:
            print(f"\n[skip] stage {i}/{len(stages)} {name} -- already done "
                  f"(pass --redo to force)")
            continue
        _stage_banner(i, len(stages), name, note)
        rc = fn()
        if rc == 3 and name == "select":
            # A measured "nothing improved", not a failure. Stopping here is the
            # correct outcome: promoting the least-bad arm would spend hours of
            # gate compute to measure a regression.
            state["stages"].setdefault(name, {})["done"] = True
            _save_state(state_path, state)
            _write_no_candidate_report(state, pdir)
            print(f"\n[ok] the pipeline stopped after {name!r} with a negative "
                  f"result, which is recorded in {pdir / 'RESULT.md'}.")
            return 0
        if rc:
            print(f"\n[X] stage {name!r} failed with {rc}. Nothing downstream "
                  f"ran. Fix it and re-run; finished stages are skipped.")
            _save_state(state_path, state)
            return rc
        state["stages"].setdefault(name, {})["done"] = True
        _save_state(state_path, state)
    print(f"\n{'=' * 78}")
    print(f"[ok] every stage finished in {(time.perf_counter()-t0)/3600:.2f} h")
    print(f"     state   -> {state_path}")
    print(f"     result  -> {pdir / 'RESULT.md'}")
    print("=" * 78)
    return 0


# ==========================================================================
# 14. COMMAND LINE
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")

    def add_run(p):
        p.add_argument("--suite", choices=sorted(SUITES), default="cec2017")
        p.add_argument("--dims", type=int, nargs="+", default=None,
                       help="override the suite's dimensions")
        p.add_argument("--runs", type=int, default=None,
                       help="override the suite's run count")
        p.add_argument("--fes-per-dim", type=int, default=None,
                       help="override the suite's budget")
        p.add_argument("--jobs", type=int, default=None,
                       help="worker processes; default = every thread. Cannot "
                            "change any reported number, only the wall clock.")
        p.add_argument("--out", default=None)
        p.add_argument("--algos", nargs="+", default=None)
        p.add_argument("--functions", nargs="+", default=None,
                       help="official function ids")
        p.add_argument("--resume", action="store_true",
                       help="continue an interrupted run, keeping what is recorded")
        p.add_argument("--fresh", action="store_true",
                       help="discard an existing results file and start over")

    p_run = sub.add_parser("run", help="run the benchmark suite")
    add_run(p_run)

    def add_analyze(p):
        p.add_argument("--out", default=None)
        p.add_argument("--control", default=TARGET)
        p.add_argument("--algos", nargs="+", default=None)
        p.add_argument("--no-figures", action="store_true",
                       help="tables and statistics only")
        p.add_argument("--include-unlisted", action="store_true",
                       help="with --algos, also keep algorithms found in the "
                            "results file but not named. Off by default, so "
                            "--algos restricts k and the statistics stay "
                            "comparable with a previously published line-up.")

    p_an = sub.add_parser("analyze", help="tables, statistics and figures")
    add_analyze(p_an)

    p_diag = sub.add_parser("diagnose", help="the component diagnostic")
    p_diag.add_argument("--dim", type=int, default=10)
    p_diag.add_argument("--runs", type=int, default=31)
    p_diag.add_argument("--fes-per-dim", type=int, default=10_000)
    p_diag.add_argument("--jobs", type=int, default=None)
    p_diag.add_argument("--out", default=None)
    p_diag.add_argument("--quick", action="store_true",
                        help="losing functions only -- a smoke test, not a diagnosis")

    p_abl = sub.add_parser("ablate", help="the factorial component ablation")
    p_abl.add_argument("--dims", type=int, nargs="+", default=None,
                       help=f"default {list(ABLATE_DIMS)}")
    p_abl.add_argument("--runs", type=int, default=None,
                       help="default is the suite's 51")
    p_abl.add_argument("--functions", nargs="+", default=None,
                       help=f"official ids; default {list(ABLATE_FUNCTIONS)}")
    p_abl.add_argument("--configs", nargs="+", default=None,
                       help="cell names; default is the whole factorial")
    p_abl.add_argument("--fes-per-dim", type=int, default=None)
    p_abl.add_argument("--jobs", type=int, default=None)
    p_abl.add_argument("--out", default=None)
    p_abl.add_argument("--resume", action="store_true",
                       help="continue an interrupted ablation")
    p_abl.add_argument("--fresh", action="store_true",
                       help="discard an existing results file and start over")

    p_exp = sub.add_parser("experiment",
                           help="the whole study, start to finish, resumable")
    p_exp.add_argument("--jobs", type=int, default=None)
    p_exp.add_argument("--runs", type=int, default=None,
                       help="override the suite's 51 runs (screening only; "
                            "the gate always uses the recorded count)")
    p_exp.add_argument("--only-stages", nargs="+", default=None,
                       help="run only these stages")
    p_exp.add_argument("--out", default=None,
                       help="pipeline state directory; default results_pipeline")
    p_exp.add_argument("--results", default=None,
                       help="where the measurements live; default "
                            "results_cec2017. Point it at an empty directory to "
                            "build a complete study from nothing -- the gate "
                            "then measures every algorithm in the line-up, not "
                            "only the candidate.")
    p_exp.add_argument("--fes-per-dim", type=int, default=None,
                       help="shrink the budget for a smoke test of the whole "
                            "pipeline; results are NOT comparable")
    p_exp.add_argument("--redo", action="store_true",
                       help="re-run stages already marked done")

    p_test = sub.add_parser("test", help="the correctness suite")
    p_test.add_argument("--full", action="store_true",
                        help="include the slow tests (D=30 optima, run sharding)")
    p_test.add_argument("--only", default=None,
                        help="run only tests whose name contains this substring")

    sub.add_parser("selfcheck",
                   help="the fast subset -- a pre-flight before a long run")

    p_all = sub.add_parser("all", help="run, then analyze")
    add_run(p_all)
    p_all.add_argument("--control", default=TARGET)
    p_all.add_argument("--no-figures", action="store_true")

    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not args.command:
        ap.print_help()
        return 0
    if args.command == "run":
        return cmd_run(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "diagnose":
        return cmd_diagnose(args)
    if args.command == "ablate":
        return cmd_ablate(args)
    if args.command == "experiment":
        return cmd_experiment(args)
    if args.command == "test":
        return cmd_test(args)
    if args.command == "selfcheck":
        return cmd_selfcheck(args)
    if args.command == "all":
        rc = cmd_run(args)
        if rc != 0:
            return rc
        args.out = args.out or f"results_{args.suite}"
        return cmd_analyze(args)
    ap.print_help()
    return 0


if __name__ == "__main__":      # required on Windows (loky spawns processes)
    sys.exit(main())
