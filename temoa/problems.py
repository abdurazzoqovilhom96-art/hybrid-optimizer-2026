"""Benchmark suite.

Differences from the original ``hybrid 2026.py`` (all deliberate, all documented):

1. **Error values, not raw values.** Every problem exposes ``f_star(dim)``, the
   objective value at the true optimizer, and the harness reports
   ``f(x) - f*``. In the original, Schwefel's ``f*`` is 3.82e-04 (30D) and
   1.27e-03 (100D), so an algorithm that solved it exactly still appeared to
   stall at ~1e-03 on a log axis.

2. **A rotated track.** The original rotates 1 of 12 problems, so the suite is
   almost entirely separable and structurally favours coordinate-wise operators
   (binomial crossover). ``suite="rotated"`` applies an orthogonal rotation
   wherever rotation is meaningful, as CEC suites do. ``suite="shift"``
   reproduces the original transformation exactly.

3. **Dynamic problems get a dynamic metric.** Best-so-far is meaningless against
   a moving optimum. ``DynamicSphere`` is scored by *windowed offline error*
   (see ``tracker.py``).

4. **Noisy problems get a recommendation rule.** Selecting ``best_x`` by its
   noisy value and then scoring it noise-free is extreme-value selection bias:
   below the noise amplitude it rewards whoever sampled most often near the
   optimum. The tracker keeps candidates and re-evaluates them (see
   ``tracker.py``).

Schwefel is *not* rotated or shifted, only sign-flipped, in both tracks. Its
optimizer sits at |z| = 420.9687 per coordinate; an unconstrained shift or
rotation pushes the optimizer outside the box or into the region where the
function's structure changes. CEC handles this with a special wrap; keeping
Schwefel flip-only is the honest simplification and is stated in the report.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Base functions. Each attains its minimum at z_star * ones(dim).
# --------------------------------------------------------------------------


def ackley(x):
    dim = x.shape[-1]
    return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim))
            - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20.0 + np.e)


def bent_cigar(x):
    return x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2)


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def griewank(x):
    dim = x.shape[-1]
    return (np.sum(x**2) / 4000.0
            - np.prod(np.cos(x / np.sqrt(np.arange(1, dim + 1)))) + 1.0)


def schwefel(x):
    dim = x.shape[-1]
    return 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def levy(x):
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * w[-1]) ** 2)
    t2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    return t1 + t2 + t3


def zakharov(x):
    dim = x.shape[-1]
    s1 = np.sum(x**2)
    s2 = np.sum(0.5 * np.arange(1, dim + 1) * x)
    return s1 + s2**2 + s2**4


def elliptic(x):
    dim = x.shape[-1]
    return np.sum(10.0 ** (6 * np.linspace(0, 1, dim)) * x**2)


def rastrigin(x):
    return np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x) + 10.0)


def sphere(x):
    return np.sum(x**2)


def composition(x):
    return 0.5 * ackley(x) + 0.5 * np.sum(x**2)


# --------------------------------------------------------------------------
# Problem table: name -> (base, lb, ub, z_star, kind, rotatable)
#
#   z_star  : each coordinate of the base optimizer
#   kind    : "plain" | "noisy" | "dynamic" | "flip"
#   rotatable: whether an orthogonal rotation is applied in the rotated track
# --------------------------------------------------------------------------

PROBLEMS = {
    "Ackley":          (ackley,      -32.0,  32.0,   0.0,      "plain",   True),
    "BentCigar":       (bent_cigar, -100.0, 100.0,   0.0,      "plain",   True),
    "Composition":     (composition,-100.0, 100.0,   0.0,      "plain",   True),
    "DynamicSphere":   (sphere,     -100.0, 100.0,   0.0,      "dynamic", False),
    "Griewank":        (griewank,   -600.0, 600.0,   0.0,      "plain",   True),
    "Levy":            (levy,        -10.0,  10.0,   1.0,      "plain",   True),
    "NoisyRastrigin":  (rastrigin,    -5.12,  5.12,  0.0,      "noisy",   True),
    "NoisySphere":     (sphere,     -100.0, 100.0,   0.0,      "noisy",   False),
    "Rosenbrock":      (rosenbrock,  -30.0,  30.0,   1.0,      "plain",   True),
    "RotatedElliptic": (elliptic,   -100.0, 100.0,   0.0,      "plain",   True),
    "Schwefel":        (schwefel,   -500.0, 500.0, 420.968746, "flip",    False),
    "Zakharov":        (zakharov,    -10.0,  10.0,   0.0,      "plain",   True),
}

FUNC_NAMES = list(PROBLEMS)

SHIFT_SEED = 20260922      # fixed: the problem instance must not vary across runs
NOISE_AMP = 0.5            # uniform noise on [-NOISE_AMP, +NOISE_AMP]
DRIFT_WINDOW = 2000        # FES; sets both the drift timescale and the offline-error window


def f_star(name: str, dim: int) -> float:
    """Objective value at the true optimizer. Reported errors subtract this."""
    base, _, _, z_star, _, _ = PROBLEMS[name]
    return float(base(np.full(dim, z_star)))


class Problem:
    """A concrete (function, dimension, suite) instance.

    Attributes
    ----------
    obj        : callable, what the optimizer sees (noisy/drifting where applicable)
    true_obj   : callable, the noise-free, non-drifting objective (for scoring)
    f_star     : float, objective value at the optimizer
    kind       : "plain" | "noisy" | "dynamic" | "flip"
    optimum_at : callable(fes) -> ndarray, the optimizer's location (dynamic only)
    """

    def __init__(self, name: str, dim: int, suite: str = "rotated",
                 noise_rng: np.random.Generator | None = None):
        if suite not in ("shift", "rotated"):
            raise ValueError(f"suite must be 'shift' or 'rotated', got {suite!r}")
        base, lb, ub, z_star, kind, rotatable = PROBLEMS[name]
        self.name, self.dim, self.suite, self.kind = name, dim, suite, kind
        self.lb, self.ub = lb, ub
        self.f_star = f_star(name, dim)

        # The problem instance is a deterministic function of (SHIFT_SEED, dim, name)
        # so every algorithm and every run faces the identical landscape.
        inst = np.random.default_rng(SHIFT_SEED + 1000 * dim + FUNC_NAMES.index(name))

        if kind == "flip":
            # Schwefel's optimizer at |z| = 420.9687 is global only INSIDE the box.
            # Adding z_star to the transform would push z outside [-500, 500],
            # where the function keeps descending and the stated optimum is no
            # longer global. So the flip track applies no offset: z = signs * x,
            # and the optimizer sits at x = signs * z_star, inside the box.
            self.signs = inst.choice([-1.0, 1.0], dim)
            self.o = self.signs * z_star
            self._offset = 0.0
        else:
            self.signs = np.ones(dim)
            self.o = inst.uniform(0.8 * lb, 0.8 * ub, dim)
            self._offset = z_star

        # Rotation: always for "RotatedElliptic" (as in the original), for every
        # rotatable problem in the rotated track.
        rotate = rotatable and (suite == "rotated" or name == "RotatedElliptic")
        if rotate:
            q, r = np.linalg.qr(inst.standard_normal((dim, dim)))
            self.M = q * np.sign(np.diag(r))     # Haar-distributed orthogonal matrix
        else:
            self.M = None

        self._base, self._z_star = base, z_star
        self._noise_rng = noise_rng if noise_rng is not None else np.random.default_rng(0)
        self._fes = 0

    # -- geometry -----------------------------------------------------------
    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if self.kind == "flip":
            return self.signs * x                 # optimizer at x = signs * z_star
        z = self.signs * (x - self.o)
        if self.M is not None:
            z = self.M @ z
        return z + self._offset

    def true_obj(self, x):
        """Noise-free, drift-free objective. Used for scoring, never charged FES."""
        return float(self._base(self.transform(x)))

    def _drift(self, fes: int) -> float:
        return 5.0 * np.sin(fes / DRIFT_WINDOW)

    def optimum_at(self, fes: int):
        """Location of the optimizer at a given FES count (dynamic problems)."""
        if self.kind != "dynamic":
            return self.o.copy()
        d = np.full(self.dim, self._drift(fes))
        if self.M is not None:
            d = self.M.T @ d
        return self.o + self.signs * d

    # -- what the optimizer actually calls ----------------------------------
    def __call__(self, x):
        if self.kind == "noisy":
            return (float(self._base(self.transform(x)))
                    + self._noise_rng.uniform(-NOISE_AMP, NOISE_AMP))
        if self.kind == "dynamic":
            self._fes += 1
            return float(self._base(self.transform(x) - self._drift(self._fes)))
        return float(self._base(self.transform(x)))


def make_problem(name: str, dim: int, suite: str = "rotated",
                 noise_rng: np.random.Generator | None = None) -> Problem:
    return Problem(name, dim, suite=suite, noise_rng=noise_rng)
