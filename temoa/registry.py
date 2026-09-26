"""The competitor line-up, and what separates a rival from our own history.

**Rivals are other people's algorithms.** The earlier versions of this work --
TEMOA_V10, V11, V12 -- are not rivals, and keeping them in the comparison table
did two things wrong. It let a weak entry (V10 ranked 7.17 of 10) lift everyone
else's average rank for free, and it made the study look as though it were
competing with itself. They are kept in ``ANCESTORS``, where the ablation and
the repair history can still reach them, and they are out of ``ALGORITHMS``.

The line-up is also sharper for it. Nemenyi's critical difference depends on how
many algorithms are compared (measured at N = 29 functions):

    k = 10   CD = 2.5157
    k =  9   CD = 2.2309
    k =  7   CD = 1.6730

Fewer algorithms, a more sensitive test. This does not rescue our case and is
not meant to: at k = 7 the gap from L-SHADE-DGR to jSO is 0.38 against a CD of
1.67. Removing our own versions is methodological hygiene, not a way to
manufacture significance.

**The nine rivals.** Six are implemented here; three are listed in
``PLANNED_RIVALS`` and not yet written.

    CMA-ES         2001  the reference for continuous black-box optimisation
    IPOP-CMA-ES    2005  the standard restart CMA-ES (Auger & Hansen)
    sep-CMA-ES     2008  diagonal variant; the control that isolates rotation
    BIPOP-CMA-ES   2009  the strongest general-purpose continuous baseline
    L-SHADE        2014  CEC'2014 winner
    jSO            2017  leading CEC'2017 entrant; the core this is built on
    LSHADE-cnEpSin 2017  CEC'2017 third place; where the eigen crossover is from
    L-SHADE-RSP    2018  CEC'2018 winner; jSO's direct successor
    NL-SHADE-RSP   2021  CEC'2021 winner; the modern frontier

The newest rival we currently run is jSO, from 2017. In a 2026 paper that is
the weak-baseline problem this project exists to correct, which is why the last
three are required rather than optional.

**Why EA4eig and L-SRTDE are not rivals.** They won CEC'2022 and CEC'2024, and
their published tables are for those suites. No CEC'2017 table at this protocol
-- 29 functions, D = 10, 51 runs -- was found for either, and a comparison
without a common basis is not a comparison. They remain prior art that must be
cited for the eigen crossover, which is theirs and is never claimed here.

**Why the three new rivals are not simply reimplemented and trusted.** A faulty
reimplementation of a competitor errs in our own favour -- the precise fault this
project exists to correct. Each one has to reproduce its published CEC'2017
numbers before it may enter a table; see ``tests/test_competitor_validation.py``.

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

#: Ours. One algorithm: the one the paper proposes.
OURS = {
    "L-SHADE-DGR": LSHADE_DGR,
}

#: Our own earlier versions. Not rivals, and never in a comparison table -- they
#: are here so the ablation can measure what each repair actually bought, and so
#: TEMOA_V12 stays available as the recorded negative result on noisy credit.
ANCESTORS = {
    "TEMOA_V10": TEMOA_V10_HYBRID,   # the original, unchanged
    "TEMOA_V11": TEMOA_V11,          # the measured repair
    "TEMOA_V12": TEMOA_V12,          # + noise mechanism (measured ineffective)
}

#: The rivals implemented here. See the module docstring for each one's standing.
RIVALS = {
    "LSHADE":      LSHADE,
    "jSO":         jSO,
    "BIPOP_CMAES": BIPOP_CMAES,
    "IPOP_CMAES":  IPOP_CMAES,
    "CMAES":       CMAES,
    "sepCMAES":    sepCMAES,
}

#: Required rivals not yet implemented. The line-up is incomplete until these
#: exist and pass validation against their published CEC'2017 tables.
PLANNED_RIVALS = ("LSHADE-cnEpSin", "L-SHADE-RSP", "NL-SHADE-RSP")

#: Compared from published tables: nothing. EA4eig and L-SRTDE won on other
#: suites, so no protocol-matched comparison exists. They are cited prior art.
PUBLISHED_ONLY = ()

#: The study line-up: what runs, and what appears in every table.
ALGORITHMS = {**OURS, **RIVALS}
ALL_ALGORITHMS = ALGORITHMS

#: Every implementation in the repository, including our own earlier versions.
#: Used by the correctness tests, which must cover code that is still callable
#: even when it is not part of the comparison.
EVERY_IMPLEMENTATION = {**ALGORITHMS, **ANCESTORS}


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
    which is also what makes a partial re-run comparable with an earlier one --
    and it is why dropping TEMOA_V10..V12 from the line-up does not invalidate
    the rows already recorded for the seven that remain.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)
