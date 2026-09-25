"""Benchmark suites and their official protocols.

The legacy twelve-function suite is kept for regression testing, but it is not
what a Q1 submission is judged on. The comparison that counts runs on the CEC
bound-constrained suites, under the competition's own protocol, so that results
are directly comparable with the published tables of the competition entrants.

FUNCTION NUMBERING -- READ THIS BEFORE COMPARING WITH ANY PUBLISHED TABLE.

``opfunu`` ships 29 CEC'2017 classes named ``F1..F29``, but the official suite
is ``F1..F30``. opfunu has already dropped the unstable F2 (Shifted and Rotated
Sum of Different Power, withdrawn after the competition) and renumbered, which
was verified here by reading the class descriptions:

    opfunu F1 = "Shifted and Rotated Bent Cigar"       = official F1
    opfunu F2 = "Shifted and Rotated Zakharov"         = official F3
    opfunu F3 = "Shifted and Rotated Rosenbrock"       = official F4
    ...
    opfunu Fk = official F(k+1)  for k >= 2

So opfunu's 29 functions are exactly the standard set {F1, F3..F30}. This module
exposes **official** ids everywhere -- tables, CSVs and figures say F3 where the
official suite says F3 -- because an off-by-one against published results would
be invisible and would invalidate every comparison.

THE OPTIMUM BIAS IS SHIFTED TOO. opfunu sets each CEC'2017 optimum value to
100 * (its own index), so official F3 carries f* = 200 where the official suite
says 300. This is harmless *provided everything is reported as error*, which it
is: ``f(x*) == f*`` holds exactly for every function (tested), so
``f(x) - f*`` is the correct, comparable quantity and is what published CEC
tables report as well. Never compare raw objective values with a published
table -- only errors. CEC'2022 is unaffected: its biases match the official ones
(F1 = 300, F2 = 400, F3 = 600, ..., F12 = 2700).

PROTOCOL PROVENANCE. Each suite records whether its protocol was verified
against a primary source. Anything unverified is flagged in the spec and must be
checked against the competition technical report before the numbers are quoted.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# The legacy suite lives in problems.py and keeps its own protocol.
from .problems import FUNC_NAMES as LEGACY_FUNCS
from .problems import Problem as LegacyProblem

CEC_ERROR_FLOOR = 1e-8      # competition convention: errors below this count as 0


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    dims: tuple[int, ...]
    runs: int
    max_fes: Callable[[int], int]
    function_ids: tuple           # official ids (ints) or names (str) for legacy
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
    "legacy": SuiteSpec(
        name="legacy",
        dims=(30, 50, 100),
        runs=30,
        max_fes=lambda d: 3000 * d,
        function_ids=tuple(LEGACY_FUNCS),
        error_floor=0.0,
        protocol_verified=True,
        note="The original study's twelve classic functions. Regression only; "
             "not a basis for any claim against modern algorithms.",
    ),
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
             "these three are confirmed. The MaxFES values (200000 for D=10, "
             "1000000 for D=20) are NOT yet confirmed against the competition "
             "technical report and must be before any number from this suite is "
             "quoted. Pass --fes-per-dim to override.",
    ),
}


# --------------------------------------------------------------------------
# CEC problem wrapper -- duck-types the legacy Problem so Tracker is unchanged
# --------------------------------------------------------------------------

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
    """One CEC function at one dimension, with the legacy Problem interface.

    ``kind`` is always "plain": CEC functions are deterministic and static, so
    best-so-far is the correct metric and the tracker needs no special handling.
    """

    kind = "plain"

    def __init__(self, suite: str, official_id: int, dim: int, error_floor: float | None = None):
        cls = _opfunu_classes(suite)[official_to_opfunu(suite, official_id)]
        self._f = cls(ndim=dim)
        self.suite, self.official_id, self.dim = suite, official_id, dim
        self.name = f"F{official_id}"
        self.lb = float(np.min(self._f.lb))
        self.ub = float(np.max(self._f.ub))
        self.f_star = float(self._f.f_global)
        self.o = np.asarray(self._f.x_global, dtype=float)
        self.error_floor = (SUITES[suite].error_floor if error_floor is None else error_floor)
        self.description = str(getattr(cls, "name", "") or "")

    def true_obj(self, x) -> float:
        return float(self._f.evaluate(np.asarray(x, dtype=float)))

    def __call__(self, x) -> float:
        return float(self._f.evaluate(np.asarray(x, dtype=float)))


def make_suite_problem(suite: str, fid, dim: int, noise_rng=None, legacy_track: str = "rotated"):
    """Build one problem from any suite. ``fid`` is an official id, or a name for legacy."""
    if suite == "legacy":
        return LegacyProblem(str(fid), dim, suite=legacy_track, noise_rng=noise_rng)
    return CECProblem(suite, int(fid), dim)
