"""Controlled additive noise, an online noise-scale estimator, and diagnostics.

This module exists to test one claim, stated precisely in the report and derived
here for the code that implements it.

THE CLAIM
---------
Let the observed objective be ``f~(x) = f(x) + eps`` with ``eps`` iid, mean 0,
variance ``sigma^2``. For a parent ``x`` and trial ``u``:

    true gap      delta      = f(x) - f(u)              (positive = improvement)
    observed gap  delta~     = delta + eta,   eta = eps_x - eps_u,  Var[eta] = 2 sigma^2

SHADE, L-SHADE and jSO accept a trial when ``delta~ > 0`` and then update their
F/CR memory with a Lehmer mean weighted by ``delta~``:

    w_k = delta~_k / sum_j delta~_j ,   M_F <- sum_k w_k F_k^2 / sum_k w_k F_k

Two consequences follow.

1. *False acceptance* (known). ``P(accept) = Phi(delta / (sigma*sqrt2))``, which
   is 1/2 at ``delta = 0`` and stays positive for true worsenings.

2. *Weight bias* (the claim this project tests). The weight uses ``delta~``, not
   ``delta``. Conditioning on acceptance,

       E[delta~ | delta~ > 0, delta = 0] = E[eta | eta > 0] = 2 sigma / sqrt(pi)

   while a genuine improvement with ``delta >> sigma`` contributes ``~ delta``.
   So once true improvements fall to the order of ``sigma`` -- the late,
   refinement phase of a run -- the largest weights belong to the trials that
   were accepted *because* the noise draw was large. The memory then drifts
   toward whatever (F, CR) happened to accompany big noise draws, which carries
   no information.

Existing noise handling (UH-CMA-ES, RA-CMA-ES, resampling, racing) stabilises
the *ranking* before a deterministic selection rule. It does not address a
``delta~``-proportional credit weight, because CMA-ES has no such weight -- it
uses rank-based weights. The bias is specific to the SHADE family.

Every constant below was chosen by simulation, not by guesswork; see
``tests/test_noise.py``, which fails if the defaults are weakened.
"""

from __future__ import annotations

import numpy as np

#: Relative noise levels for the study. 0.0 is the noise-free control.
NOISE_LEVELS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


def reference_scale(problem, dim: int, n: int = 2000, seed: int = 12345) -> float:
    """Standard deviation of ``f`` over uniform samples in the box.

    The noise level is expressed relative to this, so ``rel_sigma`` means the
    same thing on every function regardless of its raw magnitude. It is computed
    from a fixed seed, so it is a property of the problem instance, not of a run.
    """
    rng = np.random.default_rng([seed, dim])
    xs = rng.uniform(problem.lb, problem.ub, (n, dim))
    vals = np.array([problem.true_obj(x) for x in xs], dtype=float)
    vals = vals[np.isfinite(vals)]
    s = float(np.std(vals))
    return s if s > 0 else 1.0


class NoisyProblem:
    """Wraps a problem so that every evaluation returns ``f(x) + N(0, sigma^2)``.

    Duck-types the problem interface the tracker expects. ``kind`` is "noisy", so
    the tracker applies its re-evaluation recommendation rule rather than
    trusting the best observed value -- selecting by a noisy value and then
    scoring noise-free is extreme-value selection bias.

    ``true_obj`` stays noise-free, which is what makes the diagnostics possible:
    the true gap of every trial is knowable here even though the algorithm cannot
    see it.
    """

    kind = "noisy"

    def __init__(self, problem, rel_sigma: float, rng: np.random.Generator,
                 dim: int | None = None):
        self.problem = problem
        self.rel_sigma = float(rel_sigma)
        self.lb, self.ub = problem.lb, problem.ub
        self.f_star = problem.f_star
        self.o = getattr(problem, "o", None)
        self.error_floor = getattr(problem, "error_floor", 0.0)
        self.name = f"{getattr(problem, 'name', '?')}@s{rel_sigma:g}"
        dim = dim if dim is not None else len(np.atleast_1d(self.o))
        self.scale = reference_scale(problem, dim)
        self.sigma = self.rel_sigma * self.scale
        self._rng = rng

    def true_obj(self, x) -> float:
        return self.problem.true_obj(x)

    def __call__(self, x) -> float:
        v = self.problem.true_obj(x)
        return v if self.sigma == 0.0 else v + float(self._rng.normal(0.0, self.sigma))


class NoiseEstimator:
    """Online estimate of the noise scale from duplicate evaluations.

    Each generation, ``m`` individuals are evaluated a second time:

        sigma^2_gen = (1 / 2m) * sum_i ( f~1(x_i) - f~2(x_i) )^2

    which is unbiased for ``sigma^2`` because ``Var[f~1 - f~2] = 2 sigma^2``.

    One generation's estimate is very noisy: ``(f~1-f~2)^2 / (2 sigma^2)`` is
    chi-squared with one degree of freedom, so the relative standard deviation of
    ``sigma^2_gen`` is ``sqrt(2/m)`` -- 0.816 at m=3. Exponential smoothing raises
    the effective sample to about ``m (2 - alpha) / alpha``. Simulated over 400
    generations and 4000 repetitions:

        m   alpha   m_eff   rel.SD(sigma_hat)   P(error > 20%)
        3   0.30      17          0.170             0.234   <- too loose
        3   0.15      37          0.116             0.081
        5   0.15      62          0.090             0.025   <- default
        8   0.15      99          0.071             0.005

    The default costs ``m = 5`` extra evaluations per generation, about 2.5-4% of
    a CEC budget, and they are charged to the budget like any other evaluation.

    The mechanism tolerates an imperfect estimate: a 15% error in ``sigma_hat``
    moves the false-inclusion rate at ``tau = 1`` only from 0.159 to 0.125-0.198.
    """

    def __init__(self, m: int = 5, alpha: float = 0.15):
        self.m, self.alpha = int(m), float(alpha)
        self.var: float | None = None          # smoothed sigma^2
        self.n_updates = 0

    def update_from_pairs(self, diffs) -> float:
        """Fold one generation's duplicate-evaluation differences in."""
        d = np.asarray(diffs, dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return self.sigma
        gen_var = float(np.sum(d**2) / (2 * d.size))
        self.var = gen_var if self.var is None else (1 - self.alpha) * self.var + self.alpha * gen_var
        self.n_updates += 1
        return self.sigma

    @property
    def sigma(self) -> float:
        """Estimated noise standard deviation; 0 until the first update."""
        return 0.0 if self.var is None else float(np.sqrt(max(self.var, 0.0)))

    @property
    def gap_sigma(self) -> float:
        """Standard deviation of the observed gap ``delta~`` under a zero true gap.

        ``Var[eta] = 2 sigma^2``, so this is ``sigma * sqrt(2)`` -- the natural
        unit for a threshold on ``delta~``.
        """
        return self.sigma * np.sqrt(2.0)


def false_acceptance_rate(true_gaps, accepted_mask) -> float:
    """Fraction of accepted trials whose *true* gap was not an improvement.

    Only computable in a study where the noise is added by us, which is the point
    of ``NoisyProblem``: the algorithm cannot see this, the experiment can.
    """
    tg = np.asarray(true_gaps, dtype=float)[np.asarray(accepted_mask, dtype=bool)]
    return float(np.mean(tg <= 0)) if tg.size else 0.0


def credit_contamination(true_gaps, credit_mask) -> float:
    """Same quantity, restricted to the trials that entered the credit set.

    This is what the proposed mechanism is supposed to reduce, and it is distinct
    from the acceptance rate: the population update and the memory update are
    decoupled here on purpose.
    """
    return false_acceptance_rate(true_gaps, credit_mask)
