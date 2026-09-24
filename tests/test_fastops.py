"""Vektorlashtirilgan ildiz funksiyalari asl variantlar bilan bir xilligini tekshiradi."""
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from aceshade import fastops


def test_replacements_match_originals():
    orig = fastops.originals()
    rng = np.random.default_rng(12345)
    worst = {}
    for name, fast in fastops.REPLACEMENTS.items():
        w = 0.0
        for ndim in (2, 5, 10, 20, 30):
            for _ in range(400):
                x = rng.uniform(-100, 100, ndim)
                a, b = orig[name](x), fast(x)
                w = max(w, abs(a - b) / max(abs(a), 1e-300))
        worst[name] = w
        assert w < 1e-12, f"{name}: nisbiy farq {w:.3e}"
    return worst


def test_cec2022_unchanged_after_patch():
    from opfunu.cec_based import cec2022
    rng = np.random.default_rng(7)
    cases = {}
    for fi in range(1, 13):
        for d in (10, 20):
            f = getattr(cec2022, f"F{fi}2022")(ndim=d)
            xs = [rng.uniform(-100, 100, d) for _ in range(4)]
            cases[(fi, d)] = (xs, [f.evaluate(x) for x in xs])
    fastops.apply()
    worst = 0.0
    for (fi, d), (xs, vals) in cases.items():
        f = getattr(cec2022, f"F{fi}2022")(ndim=d)
        for x, v in zip(xs, vals):
            worst = max(worst, abs(f.evaluate(x) - v) / max(abs(v), 1e-300))
    assert worst < 1e-12, f"patchdan keyin farq {worst:.3e}"
    return worst


if __name__ == "__main__":
    w1 = test_replacements_match_originals()
    for k, v in w1.items():
        print(f"  {k:20s} eng katta nisbiy farq {v:.3e}")
    w2 = test_cec2022_unchanged_after_patch()
    print(f"  CEC-2022 patchdan keyin  eng katta nisbiy farq {w2:.3e}")
    print("TESTLAR O'TDI")
