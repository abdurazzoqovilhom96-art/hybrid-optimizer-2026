"""The competitor line-up: eight rivals, each with a documented standing.

**Why only eight.** The original study compared against DE/rand/1/bin and five
swarm metaheuristics. Recent surveys flag that exact pattern as the weak-baseline
problem, and one 2025 analysis found 26 published metaheuristics to be
structurally identical to earlier ones. Beating GWO, WOA, SCA, HHO and PSO
establishes nothing about a modern algorithm, so they were removed outright
rather than demoted to a secondary table.

Every remaining rival has a place in the categories this work competes in:

    L-SHADE        CEC'2014 winner
    jSO            leading CEC'2017 entrant; the core this algorithm is built on
    BIPOP-CMA-ES   the strongest general-purpose continuous baseline
    IPOP-CMA-ES    the standard restart CMA-ES (Auger & Hansen, 2005)
    CMA-ES         the reference for continuous black-box optimisation (Hansen)
    sep-CMA-ES     diagonal variant; the control that isolates rotation handling
    EA4eig         CEC'2022 winner -- compared via its published tables
    L-SRTDE        CEC'2024 winner -- compared via its published tables

**Why the last two are not reimplemented.** EA4eig is an ensemble of four
algorithms (CMA-ES, CoBiDE, a jSO variant, and IDE with eigen crossover), and
L-SRTDE replaces success-history adaptation with a success-rate rule. A faulty
reimplementation of either would err in our own favour -- the precise fault this
project exists to correct. They are compared against their official competition
tables, and only where the protocol matches exactly: 29 functions, D in {10, 30},
51 runs, 10000*D evaluations.

**On the eigen crossover.** It is not ours and is never claimed: EA4eig, the
CEC'2022 winner, and L-SRTDE are built on it, as are LSHADE-cnEpSin and
LSHADE-SPACMA. EA4eig is in fact the structurally closest rival to this work --
it is also an operator portfolio with an eigenbasis crossover -- which is exactly
why "portfolio plus eigen crossover" cannot be presented as a contribution.

``temoa/algorithms/baselines.py`` is kept so the original study can still be
reproduced, but nothing in it is used by any experiment.
"""

from __future__ import annotations

import hashlib

from .algorithms.lshade_dgr import LSHADE_DGR
from .algorithms.modern import (BIPOP_CMAES, CMAES, IPOP_CMAES, LSHADE,
                                sepCMAES, jSO)
from .algorithms.temoa_v10 import TEMOA_V10_HYBRID
from .algorithms.temoa_v11 import TEMOA_V11
from .algorithms.temoa_v12 import TEMOA_V12

#: The algorithm under study. Renamed from TEMOA_Vxx: a version number is not a
#: name, and "TEMOA" was never expanded anywhere. The new name declares its
#: lineage, as jSO, iL-SHADE, LSHADE-cnEpSin and L-SRTDE do -- a reviewer should
#: learn it is an L-SHADE variant from us, not by discovering it.
TARGET = "L-SHADE-DGR"

#: Ours: the algorithm under study plus the earlier versions, kept so the
#: repair history stays measurable.
OURS = {
    "L-SHADE-DGR": LSHADE_DGR,       # the algorithm the paper proposes
    "TEMOA_V10": TEMOA_V10_HYBRID,   # the original, unchanged
    "TEMOA_V11": TEMOA_V11,          # the measured repair
    "TEMOA_V12": TEMOA_V12,          # the noise study, kept as a negative result
}

#: The six rivals run here. See the module docstring for each one's standing.
RIVALS = {
    "LSHADE":      LSHADE,
    "jSO":         jSO,
    "BIPOP_CMAES": BIPOP_CMAES,
    "IPOP_CMAES":  IPOP_CMAES,
    "CMAES":       CMAES,
    "sepCMAES":    sepCMAES,
}

#: The two rivals compared via published competition tables, not run here.
PUBLISHED_ONLY = ("EA4eig", "L-SRTDE")

#: Everything executable. This is the whole line-up: no weak baselines remain.
ALGORITHMS = {**OURS, **RIVALS}
ALL_ALGORITHMS = ALGORITHMS


def algorithm_seed(name: str) -> int:
    """A seed component that depends only on the algorithm's name.

    The driver used to seed each algorithm from its position in
    ``sorted(ALL_ALGORITHMS)``. That made every algorithm's random stream depend
    on which *other* algorithms happened to be registered: removing the six weak
    swarm baselines silently re-seeded every remaining one, so a resumed run
    would have mixed two seed schemes inside a single results file, and no result
    recorded before the line-up changed could be reproduced after it.

    Deriving the component from the name alone removes that coupling. Adding or
    dropping a competitor now leaves every other competitor's streams untouched,
    which is also what makes a partial re-run comparable with an earlier one.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)
