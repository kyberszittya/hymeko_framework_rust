"""Tests for the R11.5++ deliverability pilot gate routing (pure)."""
from hymeko_rl.experiments.r11_5_plus_deliverability_pilot import gate


def _row(*, material: bool = False, ranked_k6: bool = False, recovered: bool = False, deliverable: int = 0,
         safe: bool = True, certified: bool = True, grasps: int = 8) -> dict:
    return {"n_grasps": grasps, "material_dtz_improvement": material, "ranked_k6": ranked_k6,
            "recovered_by_ranking": recovered, "n_deliverable_k6": deliverable, "all_safe": safe, "all_certified": certified}


def test_pass_needs_six_material_and_five_k6() -> None:
    rows = [_row(material=True, ranked_k6=True, deliverable=2) for _ in range(6)] + [_row(deliverable=1) for _ in range(4)]
    assert gate(rows)["verdict"] == "R11_5_PLUS_DELIVERABILITY_RANKED_GRASP_PILOT_PASS"


def test_ranking_gap_when_deliverable_but_under_gate() -> None:
    """Deliverable grasps exist (ranking is load-bearing) but the gate threshold isn't met -> RANKING_GAP."""
    rows = [_row(material=True, ranked_k6=True, recovered=True, deliverable=1) for _ in range(3)] + [_row() for _ in range(7)]
    assert gate(rows)["verdict"] == "CAPTURE_DELIVERABILITY_RANKING_CONTRACT_GAP"


def test_support_insufficient_when_no_deliverable_grasp() -> None:
    """No grasp in any population reaches K6 -> SUPPORT_INSUFFICIENT (do not refine ranking further)."""
    rows = [_row(deliverable=0) for _ in range(10)]
    assert gate(rows)["verdict"] == "CAPTURE_DELIVERABILITY_SUPPORT_INSUFFICIENT"


def test_safety_blocks_pass() -> None:
    rows = [_row(material=True, ranked_k6=True, deliverable=2) for _ in range(6)] + [_row(deliverable=1, safe=False)]
    assert gate(rows)["verdict"] != "R11_5_PLUS_DELIVERABILITY_RANKED_GRASP_PILOT_PASS"
