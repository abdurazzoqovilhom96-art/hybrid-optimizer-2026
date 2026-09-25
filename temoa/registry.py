"""The algorithm line-up, in two tiers.

**Why two tiers.** The original study compared only against DE/rand/1/bin and
five swarm metaheuristics. Recent surveys flag exactly that pattern as the weak
baseline problem, and one 2025 analysis found 26 published metaheuristics to be
structurally identical to earlier ones. A comparison that beats only GWO, WOA,
SCA, HHO and PSO establishes nothing about a modern algorithm.

``MODERN`` is therefore the comparison that counts: the adaptive-DE lineage that
TEMOA is built from, plus the CMA-ES family. ``LEGACY_SWARM`` is kept so the
original study's table can be reproduced, but it belongs in a secondary table,
never in the headline claim.

**The CMA-ES family is not optional here.** TEMOA's one measured strength is its
eigenbasis crossover on ill-conditioned rotated problems, and that is CMA-ES's
home ground. Its absence from the original comparison is what made the strength
claim untestable.

**On the eigen-crossover's novelty.** It is not ours and must not be claimed as a
contribution: EA4eig, the CEC'2022 competition winner, and L-SRTDE are built on
it. Those two, and the other competition entrants (NL-SHADE-RSP, NL-SHADE-LBC,
LSHADE-cnEpSin, LSHADE-SPACMA, EBOwithCMAR, APGSK-IMODE), are compared against
via their published result tables rather than reimplemented here -- a faulty
reimplementation of a competitor errs in our own favour, which is the fault this
project exists to correct.
"""

from __future__ import annotations

from .algorithms.baselines import (baseline_DE, baseline_GWO, baseline_HHO,
                                   baseline_HHO_orig, baseline_PSO,
                                   baseline_PSO_orig, baseline_SCA, baseline_WOA)
from .algorithms.modern import (BIPOP_CMAES, CMAES, IPOP_CMAES, LSHADE,
                                sepCMAES, jSO)
from .algorithms.temoa_v10 import TEMOA_V10_HYBRID
from .algorithms.temoa_v11 import TEMOA_V11

TARGET = "TEMOA_V11"

#: Tier 1 -- the comparison a Q1 submission is judged on.
MODERN = {
    "TEMOA_V10":   TEMOA_V10_HYBRID,   # the algorithm under study, unchanged
    "TEMOA_V11":   TEMOA_V11,          # the repaired variant
    "jSO":         jSO,                # TEMOA's own core (CEC'2017 entrant)
    "LSHADE":      LSHADE,             # the other named source of that core
    "BIPOP_CMAES": BIPOP_CMAES,        # strongest general-purpose baseline
    "IPOP_CMAES":  IPOP_CMAES,
    "CMAES":       CMAES,              # restart CMA-ES, population fixed
    "sepCMAES":    sepCMAES,           # separable control: isolates rotation handling
}

#: Tier 2 -- the original study's baselines. Secondary tables only.
LEGACY_SWARM = {
    "DE":  baseline_DE,
    "PSO": baseline_PSO,               # constriction (convergent, standard)
    "GWO": baseline_GWO,
    "WOA": baseline_WOA,
    "SCA": baseline_SCA,
    "HHO": baseline_HHO,               # with the Levy rapid dive, as published
}

#: The two baselines as the original study configured them, kept as controls so
#: the share of its reported margin that came from a weakened baseline can be
#: measured rather than argued about.
WEAKENED = {
    "PSO_orig": baseline_PSO_orig,
    "HHO_orig": baseline_HHO_orig,
}

#: Backwards compatibility with the earlier line-up.
ALGORITHMS = {**MODERN, **LEGACY_SWARM}
ALL_ALGORITHMS = {**MODERN, **LEGACY_SWARM, **WEAKENED}

#: Competitors compared against via published competition tables, not run here.
PUBLISHED_ONLY = (
    "EA4eig", "L-SRTDE", "NL-SHADE-RSP", "NL-SHADE-LBC", "LSHADE-cnEpSin",
    "LSHADE-SPACMA", "EBOwithCMAR", "APGSK-IMODE",
)
