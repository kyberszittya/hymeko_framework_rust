"""Integration tests for the `hymeko_pgraph_py` PyO3 binding.

Validates each public entry point on the canonical book + Pimentel
fixtures, with ground-truth values pinned from
`hymeko_pgraph/tests/{book_validation,pimentel_distractors}.rs`.

Skipped at collection time when the wheel is not installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

hp = pytest.importorskip("hymeko_pgraph_py")

REPO = Path(__file__).resolve().parents[2]
FIXT_EX_3_2 = REPO / "data" / "pgraph" / "Chapter3" / "example3_2.hymeko"
FIXT_EX_6_1 = REPO / "data" / "pgraph" / "Chapter6" / "example6_1.hymeko"
FIXT_PIMENTEL = REPO / "data" / "pgraph" / "Chapter6" / "pimentel_distractors.hymeko"


def _load(path: Path) -> "hp.LoweredPGraph":
    return hp.from_hymeko_text(path.read_text())


# ----------------------------------------------------------------------
# from_hymeko_text + LoweredPGraph
# ----------------------------------------------------------------------

@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_load_example_3_2():
    g = _load(FIXT_EX_3_2)
    # Ex. 3.2 spec: 11 materials, 7 units (book Fig. 3.4 + comment).
    assert g.n_materials() == 11
    assert g.n_units() == 7
    assert set(g.raw_names()) == {"E", "G", "J", "K", "L"}
    assert set(g.product_names()) == {"A"}


def test_load_invalid_raises():
    with pytest.raises(ValueError):
        hp.from_hymeko_text("(this is not a hymeko program")


# ----------------------------------------------------------------------
# maximal_structure_rs
# ----------------------------------------------------------------------

@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_msg_example_3_2_keeps_seven_units():
    g = _load(FIXT_EX_3_2)
    msg = hp.maximal_structure_rs(g)
    # Book Fig. 3.6: MSG has 7 units (the full unit set).
    assert msg.n_units() == 7
    assert msg.unit_names() == [f"O{i}" for i in range(1, 8)]


@pytest.mark.skipif(not FIXT_PIMENTEL.exists(),
                    reason="missing pimentel_distractors fixture")
def test_msg_pimentel_drops_distractors():
    g = _load(FIXT_PIMENTEL)
    msg = hp.maximal_structure_rs(g)
    # Pimentel benchmark: MSG must strip O8 / O9 / O10 (axiom-violating
    # distractors); canonical 7-unit answer stands.
    assert msg.unit_names() == [f"O{i}" for i in range(1, 8)]


# ----------------------------------------------------------------------
# enumerate_ssg_rs
# ----------------------------------------------------------------------

@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_ssg_example_3_2_yields_nineteen_structures():
    g = _load(FIXT_EX_3_2)
    msg = hp.maximal_structure_rs(g)
    sss = hp.enumerate_ssg_rs(g, msg)
    # Book Fig. 3.4: 19 solution-structures via decision-mapping SSG.
    assert len(sss) == 19
    # Each structure is a non-empty subset of the 7 MSG units.
    msg_units = set(msg.unit_names())
    for s in sss:
        assert len(s) >= 1
        assert set(s.unit_names()) <= msg_units


@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_ssg_max_structures_cap_respected():
    g = _load(FIXT_EX_3_2)
    msg = hp.maximal_structure_rs(g)
    capped = hp.enumerate_ssg_rs(g, msg, max_structures=5)
    assert len(capped) == 5


# ----------------------------------------------------------------------
# solve_abb_rs
# ----------------------------------------------------------------------

@pytest.mark.skipif(not FIXT_EX_6_1.exists(),
                    reason="missing example6_1 fixture")
def test_abb_example_6_1_optimum_is_nine():
    g = _load(FIXT_EX_6_1)
    msg = hp.maximal_structure_rs(g)
    sol = hp.solve_abb_rs(g, msg)
    assert sol is not None
    # Book Ex. 6.1: optimum is {O2, O5, O7} cost = 9.
    assert sol.unit_names() == ["O2", "O5", "O7"]
    assert sol.cost == pytest.approx(9.0)
    # The solver should have made some progress; explored > 0 is the
    # only universal contract (exact counts depend on order-of-exploration).
    assert sol.explored > 0


@pytest.mark.skipif(not FIXT_EX_6_1.exists(),
                    reason="missing example6_1 fixture")
def test_abb_explores_cap_respected():
    g = _load(FIXT_EX_6_1)
    msg = hp.maximal_structure_rs(g)
    sol = hp.solve_abb_rs(g, msg, {"max_explored": 1})
    # With cap=1 the solver may return None or a suboptimal incumbent;
    # the contract is "doesn't crash, respects the cap".
    if sol is not None:
        assert sol.explored <= 2  # 1 step beyond the cap is allowed


# ----------------------------------------------------------------------
# solve_top_k_abb_rs
# ----------------------------------------------------------------------

@pytest.mark.skipif(not FIXT_PIMENTEL.exists(),
                    reason="missing pimentel_distractors fixture")
def test_top_3_pimentel_costs_are_9_12_13():
    g = _load(FIXT_PIMENTEL)
    msg = hp.maximal_structure_rs(g)
    top3 = hp.solve_top_k_abb_rs(g, msg, 3)
    assert len(top3) == 3
    costs = [s.cost for s in top3]
    # Pinned in hymeko_pgraph/tests/pimentel_distractors.rs:
    # #1 {O2,O5,O7} cost 9; #2 {O1,O4} 12; #3 {O1,O3} 13.
    assert costs == sorted(costs), "top-K must be cost-ordered"
    assert costs[0] == pytest.approx(9.0)
    assert costs[1] == pytest.approx(12.0)
    assert costs[2] == pytest.approx(13.0)
    assert top3[0].unit_names() == ["O2", "O5", "O7"]
    assert top3[1].unit_names() == ["O1", "O4"]
    assert top3[2].unit_names() == ["O1", "O3"]


@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_top_k_zero_returns_empty():
    g = _load(FIXT_EX_3_2)
    msg = hp.maximal_structure_rs(g)
    assert hp.solve_top_k_abb_rs(g, msg, 0) == []


@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_top_k_consistent_with_ssg_membership():
    g = _load(FIXT_EX_3_2)
    msg = hp.maximal_structure_rs(g)
    sss = {tuple(s.unit_names()) for s in hp.enumerate_ssg_rs(g, msg)}
    top5 = hp.solve_top_k_abb_rs(g, msg, 5)
    assert len(top5) == 5
    for s in top5:
        assert tuple(s.unit_names()) in sss, (
            "top-K solution must appear in the SSG enumeration"
        )


# ----------------------------------------------------------------------
# Regime flag
# ----------------------------------------------------------------------

@pytest.mark.skipif(not FIXT_EX_3_2.exists(),
                    reason="missing example3_2 fixture")
def test_strict_no_excess_is_a_subset_of_canonical():
    g = _load(FIXT_EX_3_2)
    canonical = hp.maximal_structure_rs(g, strict_no_excess=False)
    strict = hp.maximal_structure_rs(g, strict_no_excess=True)
    assert set(strict.unit_names()) <= set(canonical.unit_names()), (
        "NoExcess refinement is a subset of canonical MSG"
    )
