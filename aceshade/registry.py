"""Algoritmlar reestri: barcha eksperimentlar shu ro'yxatdan foydalanadi."""
from functools import partial

from .ace import ace_shade
from .baselines import (baseline_LSHADE, baseline_jSO, baseline_LSHADE_cnEpSin,
                        baseline_CMAES)
from .rivals_new import (baseline_LSHADE_RSP, baseline_LSHADE_SPACMA,
                         baseline_NL_SHADE_RSP)

TARGET = "ACE-SHADE"

# B3 uchun asosiy ro'yxat (8 ta): taklif etilayotgan usul + 7 raqobatchi.
ALGORITHMS = {
    "ACE-SHADE":      ace_shade,
    "NL-SHADE-RSP":   baseline_NL_SHADE_RSP,
    "LSHADE-SPACMA":  baseline_LSHADE_SPACMA,
    "LSHADE-cnEpSin": baseline_LSHADE_cnEpSin,
    "LSHADE-RSP":     baseline_LSHADE_RSP,
    "jSO":            baseline_jSO,
    "L-SHADE":        baseline_LSHADE,
    "CMA-ES":         baseline_CMAES,
}

# B4 ablatsiyasi: har bir hissa alohida o'chiriladi.
ABLATION = {
    "ACE-SHADE":            ace_shade,
    "ACE-SHADE-N1-off":     partial(ace_shade, USE_COV=False),
    "ACE-SHADE-N2-off":     partial(ace_shade, USE_ENSEMBLE=False),
    "ACE-SHADE-cmu-pop":    partial(ace_shade, CMU_SOURCE="population"),
}
