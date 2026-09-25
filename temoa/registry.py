"""The algorithm line-up for the main study."""

from __future__ import annotations

from .algorithms.baselines import (baseline_DE, baseline_GWO, baseline_HHO,
                                   baseline_HHO_orig, baseline_PSO,
                                   baseline_PSO_orig, baseline_SCA, baseline_WOA)
from .algorithms.modern import LSHADE, jSO
from .algorithms.temoa_v10 import TEMOA_V10_HYBRID
from .algorithms.temoa_v11 import TEMOA_V11

TARGET = "TEMOA_V10"

#: Main comparison. Order matters only for reporting.
ALGORITHMS = {
    "TEMOA_V10":  TEMOA_V10_HYBRID,     # the algorithm under study, unchanged
    "TEMOA_V11":  TEMOA_V11,            # the repaired variant
    "jSO":        jSO,                  # TEMOA's own core, minus the hybridisation
    "LSHADE":     LSHADE,               # the other named source of the core
    "DE":         baseline_DE,
    "PSO":        baseline_PSO,         # constriction (convergent, standard)
    "GWO":        baseline_GWO,
    "WOA":        baseline_WOA,
    "SCA":        baseline_SCA,
    "HHO":        baseline_HHO,         # with the Levy rapid dive, as published
}

#: The two baselines as the original study configured them. Run alongside the
#: corrected versions to quantify how much of the reported margin came from the
#: baseline configuration rather than from the hybrid.
WEAKENED = {
    "PSO_orig": baseline_PSO_orig,
    "HHO_orig": baseline_HHO_orig,
}

ALL_ALGORITHMS = {**ALGORITHMS, **WEAKENED}
