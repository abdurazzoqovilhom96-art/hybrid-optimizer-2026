"""FES accounting and scoring.

One tracker wraps one problem for one run. Every algorithm sees only the
tracker, so the evaluation budget is counted identically for all of them and no
algorithm can buy extra evaluations.

Scoring is per problem class:

``plain`` / ``flip``
    ``true_obj(best_x) - f*``. Deterministic, so best-so-far is the right metric.

``noisy``
    Best-so-far is *biased* here: once the error drops below the noise amplitude
    the arg-min of the observed values is decided by the noise draw, not by the
    landscape, so "best observed" rewards whoever sampled most often near the
    optimum. Instead the tracker keeps a pool of the ``POOL_SIZE`` best observed
    candidates and, at the end, re-evaluates each ``REEVAL_K`` times and
    recommends the one with the best sample mean. The reported score is
    ``true_obj(recommended) - f*``. The re-evaluations are *not* charged to the
    FES budget, and the rule is identical for every algorithm.

``dynamic``
    Best-so-far against a moving optimum is not a performance measure at all: it
    only records whether a sample happened to be near the optimum when the
    optimum passed by, and it can never get worse. The tracker instead reports
    *windowed offline error* (Branke's offline error with the drift window as the
    change period): the budget is split into windows of ``DRIFT_WINDOW`` FES, the
    minimum error observed within each window is recorded, and the score is the
    mean of those minima. Lower is better, and unlike best-so-far it penalises an
    algorithm that converges and then fails to follow the optimum.
"""

from __future__ import annotations

import numpy as np

from .problems import DRIFT_WINDOW, Problem

POOL_SIZE = 12      # candidates kept for the noisy recommendation rule
REEVAL_K = 25       # re-evaluations per candidate


class Tracker:
    def __init__(self, problem: Problem, max_fes: int, n_points: int = 100):
        self.problem = problem
        self.max_fes = int(max_fes)
        self.fes = 0
        self.best_f = np.inf
        self.best_x = None
        self.overrun = 0

        self.checkpoints = np.linspace(max_fes / n_points, max_fes, n_points).astype(int)
        self.curve = np.full(n_points, np.nan)
        self._k = 0

        # noisy: pool of (observed_f, x)
        self._pool: list[tuple[float, np.ndarray]] = []

        # dynamic: windowed offline error
        self._win_min = np.inf
        self._win_mins: list[float] = []

        self._f_star = problem.f_star
        self._kind = problem.kind

    # ------------------------------------------------------------------
    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        f = self.problem(x)
        self.fes += 1

        if self.fes > self.max_fes:
            self.overrun += 1
            return f

        if f < self.best_f:
            self.best_f, self.best_x = f, x.copy()

        if self._kind == "noisy":
            if len(self._pool) < POOL_SIZE:
                self._pool.append((f, x.copy()))
                self._pool.sort(key=lambda t: t[0])
            elif f < self._pool[-1][0]:
                self._pool[-1] = (f, x.copy())
                self._pool.sort(key=lambda t: t[0])

        if self._kind == "dynamic":
            self._win_min = min(self._win_min, f - self._f_star)
            if self.fes % DRIFT_WINDOW == 0:
                self._win_mins.append(self._win_min)
                self._win_min = np.inf

        # convergence curve
        while self._k < len(self.checkpoints) and self.fes >= self.checkpoints[self._k]:
            self.curve[self._k] = self._curve_value()
            self._k += 1
        return f

    def _curve_value(self) -> float:
        if self._kind == "dynamic":
            seen = self._win_mins + ([self._win_min] if np.isfinite(self._win_min) else [])
            return float(np.mean(seen)) if seen else np.inf
        return float(self.best_f - self._f_star)

    # ------------------------------------------------------------------
    def finalize(self) -> tuple[np.ndarray, float]:
        """Return (convergence curve, final score). Both are error values."""
        if self._k < len(self.curve):
            self.curve[self._k:] = self._curve_value()
        return self.curve, self.score()

    def score(self) -> float:
        if self._kind == "dynamic":
            seen = self._win_mins + ([self._win_min] if np.isfinite(self._win_min) else [])
            return float(np.mean(seen)) if seen else float("inf")

        if self._kind == "noisy" and self._pool:
            # recommendation rule: re-evaluate the pool, take the best sample mean
            means = [np.mean([self.problem(x) for _ in range(REEVAL_K)]) for _, x in self._pool]
            rec = self._pool[int(np.argmin(means))][1]
            return float(self.problem.true_obj(rec) - self._f_star)

        if self.best_x is None:
            return float("inf")
        return float(self.problem.true_obj(self.best_x) - self._f_star)
