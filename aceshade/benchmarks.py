"""CEC-2017 (sozlash) va CEC-2022 (baholash) to'plamlari.

Ikkalasi ham `opfunu` orqali RASMIY siljitish va aylantirish ma'lumotlari
bilan yuklanadi - sintetik generatsiya qilinmaydi. Optimumda har bir
funksiya rasmiy `f_bias` qiymatini qaytaradi, shuning uchun xato
`f(x) - f_bias` ko'rinishida hisoblanadi.

Import paytida `fastops.apply()` chaqiriladi: u opfunu ning sekin
`katsuura_func` ini matematik jihatdan aynan bir xil vektorlashtirilgan
variant bilan almashtiradi (`tests/test_fastops.py` ga qarang).
"""
import numpy as np
from . import fastops

fastops.apply()

# CEC-2022: 12 funksiya, [-100, 100]^D, D = 10 va 20
CEC2022_BUDGET = {10: 200_000, 20: 1_000_000}
CEC2022_DIMS = (10, 20)
CEC2022_FUNCS = tuple(f"F{i}" for i in range(1, 13))

# Rasmiy kategoriyalar (CEC-2022 texnik hisoboti)
CEC2022_CATEGORIES = {
    "bir ekstremumli": ("F1",),
    "asosiy ko'p ekstremumli": ("F2", "F3", "F4", "F5"),
    "gibrid": ("F6", "F7", "F8"),
    "kompozitsiya": ("F9", "F10", "F11", "F12"),
}

# CEC-2017: sozlash to'plami. F2 rasmiy ravishda chiqarib tashlangan
# (beqaror xulq), shuning uchun opfunu da ham yo'q.
CEC2017_DIMS = (10, 30)
CEC2017_TUNING_FUNCS = ("F1", "F3", "F4", "F5", "F6", "F7", "F9", "F10", "F11", "F15")


def _suite(year):
    if year == 2022:
        from opfunu.cec_based import cec2022 as mod
        return mod, 2022
    if year == 2017:
        from opfunu.cec_based import cec2017 as mod
        return mod, 2017
    raise ValueError(f"qo'llab-quvvatlanmaydigan to'plam: {year}")


def make_problem(name, dim, year=2022):
    """`(obj, lb, ub, f_bias)` qaytaradi. `obj(x)` XATO qiymatini beradi."""
    mod, yr = _suite(year)
    cls = getattr(mod, f"{name}{yr}")
    prob = cls(ndim=dim)
    bias = float(prob.f_global)
    lb = np.asarray(prob.lb, dtype=float)
    ub = np.asarray(prob.ub, dtype=float)

    def obj(x):
        return float(prob.evaluate(np.asarray(x, dtype=float))) - bias

    return obj, lb, ub, bias


def category_of(func_name):
    for cat, members in CEC2022_CATEGORIES.items():
        if func_name in members:
            return cat
    return "noma'lum"
