"""Statistical machinery for the comparison.

What the original file does, and why each item is changed:

*Friedman over pooled dimensions.* The original builds one Friedman test over
36 blocks = 12 functions x 3 dimensions. The same function at 30D, 50D and 100D
is not an independent block -- the three share a landscape, a shift vector and a
rotation -- so pooling them inflates the block count and makes the p-value
anti-conservative. Here Friedman is run **per dimension** (12 blocks each), and
the Iman-Davenport F statistic is reported alongside chi-square because
Friedman's chi-square is known to be conservative for small block counts.

*No post-hoc.* The original reports a Friedman p-value and average ranks but
never tests which pairs differ. Here: Holm's step-down procedure on the
normal-approximation pairwise rank statistic (Demsar's recommendation when one
algorithm is the control), plus the Nemenyi critical difference for CD diagrams.

*Unpaired test on paired data.* The original applies the Mann-Whitney U test.
Every algorithm is run on the identical problem instance with matched seeds, so
the runs are paired and the Wilcoxon **signed-rank** test applies and has more
power. Both are computed; signed-rank is the primary.

*Partial multiplicity control.* The original applies Holm within each
(function, dimension) cell over 6 competitors, i.e. 36 separate families of 6,
with no control over the 216 tests actually performed. Here Holm is applied both
within the cell and over the whole family, and both are reported.

*No effect size.* A p-value says a difference exists, not that it matters. Added:
Vargha-Delaney A12 (probability that a random run of A beats a random run of B)
with the usual negligible/small/medium/large thresholds, and Cliff's delta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

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


def friedman(perf: pd.DataFrame):
    """Friedman test on a blocks x algorithms performance matrix (lower = better).

    Returns a dict with chi-square, Iman-Davenport F, both p-values, and the
    average ranks (rank 1 = best).
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

    return {"N_blocks": N, "k_algorithms": k, "chi2": float(chi2), "p_chi2": float(p_chi2),
            "iman_davenport_F": float(F), "p_F": float(p_F),
            "avg_ranks": pd.Series(R, index=perf.columns).sort_values()}


def nemenyi_cd(k: int, N: int, alpha: float = 0.05) -> float:
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
    """P(a < b) + 0.5 P(a = b) for minimisation: A12 > 0.5 means `a` is better.

    Computed from the Mann-Whitney U statistic, so it is O(n log n) and exact
    for ties.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    # rank-sum of `a` when ranking b-then-a ascending; smaller values rank first
    ranks = sps.rankdata(np.concatenate([a, b]))
    r_a = ranks[:na].sum()
    u_a = r_a - na * (na + 1) / 2.0          # # of pairs where a > b (with ties at 0.5)
    return float(1.0 - u_a / (na * nb))       # invert: lower is better


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
        return float(sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
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
