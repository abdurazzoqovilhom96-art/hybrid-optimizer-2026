"""Shared operators."""
from __future__ import annotations

import math

import numpy as np


def levy_flight(rng: np.random.Generator, shape, beta: float = 1.5):
    """Mantegna's algorithm for Levy-stable steps."""
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0.0, sigma_u, shape)
    v = rng.normal(0.0, 1.0, shape)
    return u / np.abs(v) ** (1 / beta)
