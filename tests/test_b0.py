"""B0 birlik testlari. Hammasi o'tmasa keyingi bosqichga o'tilmaydi.

`docs/ACE_SHADE_PROMPT.md` 6-bo'limdagi ro'yxat.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from aceshade.core import Tracker, midpoint
from aceshade.benchmarks import make_problem, CEC2022_BUDGET
from aceshade.registry import ALGORITHMS, ABLATION

DIM = 10
FES = 4000


def _problem(name="F1", year=2022):
    return make_problem(name, DIM, year)


def test_1_covariance_stays_psd():
    """C simmetrik va musbat yarim aniq bo'lib qoladi."""
    from aceshade import ace as ace_mod
    seen = {"bad": 0, "checks": 0}
    real_eigh = np.linalg.eigh

    def spy(mat):
        seen["checks"] += 1
        if not np.allclose(mat, mat.T, atol=1e-10):
            seen["bad"] += 1
        vals, vecs = real_eigh(mat)
        if vals.min() < -1e-8 * max(abs(vals).max(), 1.0):
            seen["bad"] += 1
        return vals, vecs

    np.linalg.eigh = spy
    try:
        obj, lb, ub, _ = _problem()
        np.random.seed(3)
        ACE = ALGORITHMS["ACE-SHADE"]
        ACE(Tracker(obj, FES), DIM, (lb, ub), FES)
    finally:
        np.linalg.eigh = real_eigh
    assert seen["checks"] > 0, "eigh umuman chaqirilmadi"
    assert seen["bad"] == 0, f"{seen['bad']} ta buzilgan kovariatsiya"
    return seen["checks"]


def test_2_budget_never_exceeded():
    """Hech bir algoritm MaxFES dan oshmaydi."""
    obj, lb, ub, _ = _problem()
    over = {}
    for name, alg in ALGORITHMS.items():
        np.random.seed(11)
        tr = Tracker(obj, FES)
        alg(tr, DIM, (lb, ub), FES)
        if tr.fes > FES:
            over[name] = tr.fes
    assert not over, f"byudjetdan oshdi: {over}"
    return True


def test_3_bounds_respected():
    """Chegara ishlovi barcha nuqtalarni [lb, ub] ichida saqlaydi."""
    obj, lb, ub, _ = _problem()
    worst = {}
    for name, alg in ALGORITHMS.items():
        bad = {"n": 0}

        def guard(x, _f=obj, _bad=bad):
            a = np.asarray(x, dtype=float)
            if np.any(a < lb - 1e-9) or np.any(a > ub + 1e-9):
                _bad["n"] += 1
            return _f(a)

        np.random.seed(5)
        alg(Tracker(guard, FES), DIM, (lb, ub), FES)
        if bad["n"]:
            worst[name] = bad["n"]
    assert not worst, f"chegaradan chiqqan baholashlar: {worst}"
    return True


def test_4_deterministic_under_seed():
    """Bir xil seed -> bir xil natija."""
    obj, lb, ub, _ = _problem()
    diffs = {}
    for name, alg in ALGORITHMS.items():
        outs = []
        for _ in range(2):
            np.random.seed(77)
            tr = Tracker(obj, FES)
            alg(tr, DIM, (lb, ub), FES)
            outs.append(tr.best_f)
        if outs[0] != outs[1]:
            diffs[name] = outs
    assert not diffs, f"takrorlanmadi: {diffs}"
    return True


def test_5_official_optimum_is_zero_error():
    """CEC-2017 va CEC-2022 optimumda xato 0 beradi."""
    from opfunu.cec_based import cec2022, cec2017
    worst = 0.0
    for fi in range(1, 13):
        for d in (10, 20):
            obj, _, _, _ = make_problem(f"F{fi}", d, 2022)
            worst = max(worst, abs(obj(getattr(cec2022, f"F{fi}2022")(ndim=d).x_global)))
    for nm in ("F1", "F3", "F5"):
        obj, _, _, _ = make_problem(nm, 10, 2017)
        worst = max(worst, abs(obj(getattr(cec2017, f"{nm}2017")(ndim=10).x_global)))
    assert worst < 1e-6, f"optimumda xato {worst:.3e}"
    return worst


def test_6_midpoint_repairs_everything():
    """midpoint chegara tuzatishi har doim [lb, ub] ichiga qaytaradi."""
    rng = np.random.default_rng(0)
    lb, ub = np.full(8, -100.0), np.full(8, 100.0)
    for _ in range(500):
        pop = rng.uniform(-100, 100, (16, 8))
        U = rng.uniform(-400, 400, (16, 8))
        R = midpoint(U.copy(), pop, lb, ub)
        assert np.all(R >= lb - 1e-9) and np.all(R <= ub + 1e-9)
    return True


def test_7_lpsr_floor_and_adaptation():
    """Populyatsiya N_min dan past tushmaydi; moslashuvchan ulushlar chegarada qoladi."""
    obj, lb, ub, _ = _problem()
    log = []
    np.random.seed(9)
    ALGORITHMS["ACE-SHADE"](Tracker(obj, FES), DIM, (lb, ub), FES,
                            on_generation=log.append)
    assert log, "kuzatuv nuqtasi ishlamadi"
    sizes = [g["pop_size"] for g in log]
    assert min(sizes) >= 4, f"populyatsiya {min(sizes)} gacha tushdi"
    assert sizes == sorted(sizes, reverse=True), "LPSR monoton kamaymadi"
    for key in ("p_bank", "p_eig", "p_cma"):
        vals = [g[key] for g in log]
        assert all(0.0 <= v <= 1.0 for v in vals), f"{key} chegaradan chiqdi"
    assert all(g["sigma"] > 0 for g in log), "sigma nolga tushdi"
    return f"N: {max(sizes)} -> {min(sizes)}, {len(log)} avlod"


def test_8_ablation_variants_run():
    """Ablatsiya variantlari ishlaydi va bir-biridan farq qiladi."""
    obj, lb, ub, _ = _problem()
    out = {}
    for name, alg in ABLATION.items():
        np.random.seed(21)
        tr = Tracker(obj, FES)
        alg(tr, DIM, (lb, ub), FES)
        out[name] = tr.best_f
    assert len(set(out.values())) > 1, f"variantlar bir xil natija berdi: {out}"
    return out


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            r = fn()
            print(f"  O'TDI  {name:38s} {r if not isinstance(r, bool) else ''}")
        except AssertionError as e:
            failed += 1
            print(f"  XATO   {name:38s} {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test o'tdi")
    sys.exit(1 if failed else 0)
