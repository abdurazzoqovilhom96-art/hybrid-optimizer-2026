"""Suite and protocol correctness.

The costly error this file exists to prevent: ``opfunu`` ships 29 CEC'2017
classes named F1..F29, but the official suite is F1..F30. opfunu has already
dropped the withdrawn F2 and renumbered, so its F2 is the official F3. Comparing
our F5 against a published F5 would then silently compare two different
functions, and nothing downstream would look wrong.

Run with ``pytest tests/test_suites.py`` or ``python tests/test_suites.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temoa.suites import (CEC_ERROR_FLOOR, SUITES, CECProblem,  # noqa: E402
                          make_suite_problem, official_to_opfunu)
from temoa.tracker import Tracker                                # noqa: E402


# ------------------------------------------------------- function numbering
def test_official_numbering_maps_onto_opfunu():
    assert official_to_opfunu("cec2017", 1) == 1
    assert official_to_opfunu("cec2017", 3) == 2      # the off-by-one this guards
    assert official_to_opfunu("cec2017", 30) == 29
    for fid in range(1, 13):
        assert official_to_opfunu("cec2022", fid) == fid


def test_withdrawn_f2_is_refused_not_silently_remapped():
    try:
        official_to_opfunu("cec2017", 2)
    except ValueError as e:
        assert "withdrawn" in str(e).lower()
    else:
        raise AssertionError("official CEC'2017 F2 must be refused, not remapped")


def test_official_f3_is_zakharov_not_rosenbrock():
    """Anchors the mapping against the actual function, not just the index."""
    p3 = CECProblem("cec2017", 3, 10)
    p4 = CECProblem("cec2017", 4, 10)
    assert "zakharov" in p3.description.lower(), p3.description
    assert "rosenbrock" in p4.description.lower(), p4.description
    p1 = CECProblem("cec2017", 1, 10)
    assert "cigar" in p1.description.lower(), p1.description


def test_cec2017_covers_f1_and_f3_to_f30():
    ids = SUITES["cec2017"].function_ids
    assert len(ids) == 29
    assert set(ids) == {1} | set(range(3, 31))
    assert 2 not in ids


# ------------------------------------------------------------ the landscapes
def test_every_cec_function_attains_f_star_at_its_optimum():
    for suite, dims in (("cec2017", (10, 30)), ("cec2022", (10, 20))):
        for fid in SUITES[suite].function_ids:
            for d in dims:
                p = make_suite_problem(suite, fid, d)
                err = p.true_obj(p.o) - p.f_star
                assert abs(err) < 1e-6, f"{suite} F{fid} {d}D: f(x*)-f* = {err:.3e}"


def test_cec2022_biases_match_the_official_values():
    official = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]
    for fid, bias in enumerate(official, start=1):
        got = make_suite_problem("cec2022", fid, 10).f_star
        assert got == float(bias), f"CEC'2022 F{fid}: f* = {got}, official {bias}"


def test_cec2017_bias_follows_opfunu_index_not_the_official_one():
    """opfunu shifted the biases along with the renumbering.

    Official F3 carries f* = 200 here where the official suite says 300. Harmless
    because everything is reported as error and f(x*) == f* holds exactly -- but
    it means raw objective values must never be compared with a published table.
    """
    for fid in (1, 3, 4, 10, 30):
        p = make_suite_problem("cec2017", fid, 10)
        assert p.f_star == 100.0 * official_to_opfunu("cec2017", fid)
        assert abs(p.true_obj(p.o) - p.f_star) < 1e-6


def test_cec_bounds_are_the_official_box():
    for fid in SUITES["cec2017"].function_ids[:5]:
        p = make_suite_problem("cec2017", fid, 10)
        assert (p.lb, p.ub) == (-100.0, 100.0)


# ---------------------------------------------------------------- protocols
def test_protocols_match_the_competition_rules():
    c17 = SUITES["cec2017"]
    assert c17.runs == 51
    assert c17.max_fes(10) == 100_000 and c17.max_fes(30) == 300_000
    assert c17.dims == (10, 30, 50, 100)
    assert c17.error_floor == CEC_ERROR_FLOOR
    assert c17.protocol_verified

    c22 = SUITES["cec2022"]
    assert c22.runs == 30 and c22.dims == (10, 20) and len(c22.function_ids) == 12
    # The budget could not be checked against the technical report; the flag must
    # stay False so run_all.py warns rather than quietly producing a wrong table.
    assert not c22.protocol_verified


def test_legacy_protocol_is_marked_as_not_a_competition():
    legacy = SUITES["legacy"]
    assert legacy.runs == 30 and legacy.max_fes(30) == 90_000
    assert len(legacy.function_ids) == 12


# -------------------------------------------------------------- error floor
def test_error_floor_applies_to_cec_and_not_to_legacy():
    p = make_suite_problem("cec2017", 1, 10)
    tr = Tracker(p, 10, 5)
    tr(p.o)                                  # the exact optimum
    assert tr.finalize()[1] == 0.0, "an error below 1e-8 must be reported as 0"

    # a legacy problem sets no floor, so a tiny error survives
    lp = make_suite_problem("legacy", "Ackley", 10)
    assert getattr(lp, "error_floor", 0.0) == 0.0


def test_error_just_above_the_floor_is_not_zeroed():
    p = make_suite_problem("cec2017", 1, 10)
    tr = Tracker(p, 10, 5)
    x = np.asarray(p.o, dtype=float).copy()
    # Bent Cigar grows as 1e6*z^2 off the first axis, so a small step gives a
    # measurable error well above the floor.
    x[1] += 1e-3
    tr(x)
    err = tr.finalize()[1]
    assert err > CEC_ERROR_FLOOR, f"expected a measurable error, got {err:.3e}"


# ------------------------------------------------------------ interop check
def test_cec_problem_satisfies_the_tracker_interface():
    p = make_suite_problem("cec2017", 5, 10)
    for attr in ("lb", "ub", "f_star", "kind", "true_obj", "o"):
        assert hasattr(p, attr), f"CECProblem is missing {attr!r}"
    assert p.kind == "plain"
    tr = Tracker(p, 50, 5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        tr(rng.uniform(p.lb, p.ub, 10))
    curve, score = tr.finalize()
    assert np.all(np.isfinite(curve)) and np.isfinite(score)
    assert tr.overrun == 0


# ------------------------------------------------------------ presentation
def test_function_labels_sort_naturally_in_tables():
    """As strings, F10 sorts before F3. Every table and figure would then be in
    an order that invites a misreading."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analyze import function_order
    labels = ["F30", "F3", "F10", "F1", "F4", "F20"]
    assert function_order(labels) == ["F1", "F3", "F4", "F10", "F20", "F30"]
    assert sorted(labels) != function_order(labels), "this test would be vacuous"
    # legacy names still come out stable and after the numbered ones
    mixed = function_order(["F3", "Ackley", "F1", "Schwefel"])
    assert mixed == ["F1", "F3", "Ackley", "Schwefel"], mixed


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
