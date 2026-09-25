"""The line-up itself: who is in it, and does the seeding depend on that?

Two faults live here, and neither shows up as a crash.

The first is the weak-baseline problem. Beating PSO, GWO, WOA, SCA and HHO
establishes nothing about a modern algorithm, so those baselines were removed
outright. A test pins their absence, because a single convenience import would
quietly put them back into every table.

The second is subtler. The driver used to seed each algorithm from its position
in ``sorted(ALL_ALGORITHMS)``, so *removing* a competitor re-seeded all the
others. Nothing fails when that happens -- the numbers simply stop being the
numbers that were recorded, and a resumed run silently mixes two random streams
inside one results file.

Run with ``pytest tests/test_registry.py`` or directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.registry import (ALGORITHMS, ALL_ALGORITHMS, OURS,      # noqa: E402
                            PUBLISHED_ONLY, RIVALS, TARGET,
                            algorithm_seed)

ROOT = Path(__file__).resolve().parents[1]

#: Removed outright, not demoted to a secondary table.
WEAK_BASELINES = ("PSO", "GWO", "WOA", "SCA", "HHO", "DE",
                  "PSO_orig", "HHO_orig", "PSO_weak", "HHO_weak")


# ---------------------------------------------------------------- the line-up
def test_the_target_algorithm_is_registered():
    assert TARGET in ALGORITHMS, \
        f"TARGET is {TARGET!r} but the registry has {sorted(ALGORITHMS)}"
    assert TARGET in OURS


def test_there_are_exactly_eight_rivals():
    """Six run here, two compared from their published competition tables."""
    assert len(RIVALS) == 6, sorted(RIVALS)
    assert len(PUBLISHED_ONLY) == 2, PUBLISHED_ONLY
    assert len(RIVALS) + len(PUBLISHED_ONLY) == 8


def test_every_rival_has_a_documented_standing():
    expected = {"LSHADE", "jSO", "BIPOP_CMAES", "IPOP_CMAES", "CMAES", "sepCMAES"}
    assert set(RIVALS) == expected, set(RIVALS) ^ expected
    assert set(PUBLISHED_ONLY) == {"EA4eig", "L-SRTDE"}


def test_no_weak_baseline_is_registered():
    present = [n for n in WEAK_BASELINES if n in ALL_ALGORITHMS]
    assert not present, f"weak baselines are back in the line-up: {present}"


def test_published_only_rivals_are_never_run():
    """EA4eig and L-SRTDE are compared from their official tables. A
    reimplementation of either would err in our favour -- the precise fault this
    project exists to correct -- so they must not be executable here."""
    for name in PUBLISHED_ONLY:
        assert name not in ALL_ALGORITHMS, f"{name} must not be runnable"


def test_ours_and_rivals_do_not_overlap():
    assert not set(OURS) & set(RIVALS)
    assert ALGORITHMS == {**OURS, **RIVALS}
    assert ALL_ALGORITHMS is ALGORITHMS


def test_the_original_baselines_are_kept_but_unused():
    """``baselines.py`` stays so the original study can be reproduced; nothing
    in the experiment path may import it. Checked on the parsed imports, not on
    the text, so the prose that explains the removal cannot trip it."""
    import ast
    for rel in ("temoa/registry.py", "run_all.py", "analyze.py"):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom)
                    else [])
            assert not any("baselines" in m for m in mods), \
                f"{rel} imports baselines.py -- the weak baselines are back"
    assert (ROOT / "temoa" / "algorithms" / "baselines.py").exists(), \
        "baselines.py is the record of the original study; it is kept, not deleted"


# ------------------------------------------------------------- the seeding
def test_algorithm_seed_depends_only_on_the_name():
    """Pinned values. If this test has to be updated, every previously recorded
    result becomes unreproducible -- which is the point of pinning it."""
    pinned = {
        "L-SHADE-DGR": 1311180535,
        "TEMOA_V10": 1963213242,
        "TEMOA_V11": 329255234,
        "TEMOA_V12": 1826345859,
        "LSHADE": 1230746920,
        "jSO": 1666937469,
        "BIPOP_CMAES": 167067236,
        "IPOP_CMAES": 624805395,
        "CMAES": 1234773454,
        "sepCMAES": 1523297831,
    }
    for name, want in pinned.items():
        assert algorithm_seed(name) == want, f"{name}: {algorithm_seed(name)} != {want}"


def test_algorithm_seeds_do_not_collide():
    seeds = {n: algorithm_seed(n) for n in ALL_ALGORITHMS}
    assert len(set(seeds.values())) == len(seeds), seeds


def test_the_driver_does_not_seed_from_a_sorted_index():
    """Guards the fix: seeding from ``sorted(ALL_ALGORITHMS)`` couples every
    algorithm's random stream to which other algorithms are registered."""
    src = (ROOT / "run_all.py").read_text(encoding="utf-8")
    assert "enumerate(sorted(ALL_ALGORITHMS))" not in src, \
        "run_all.py seeds from the line-up again; adding a rival re-seeds the rest"
    assert "algorithm_seed(a) for a in algos" in src, \
        "the driver must take each algorithm's seed component from its name"


def test_the_driver_refuses_to_resume_across_a_seed_scheme_change():
    src = (ROOT / "run_all.py").read_text(encoding="utf-8")
    assert "SEED_SCHEME_VERSION" in src
    assert "seed_scheme_version" in src, "the manifest must record the scheme version"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                    # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
