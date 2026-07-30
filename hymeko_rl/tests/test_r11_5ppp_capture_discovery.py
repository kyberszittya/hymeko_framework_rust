"""Tests for the R11.5+++ capture-discovery gate routing + curve (pure)."""
from hymeko_rl.experiments.r11_5ppp_capture_discovery import BUDGETS, _curve_point, gate


def _row(*, deliverable: int, saturated: bool = False, safe: bool = True, stable: bool = True) -> dict:
    return {"deliverable_at_max": deliverable, "saturated": saturated, "all_safe": safe, "all_stable": stable}


def test_growth_pass_needs_three_of_five() -> None:
    rows = [_row(deliverable=1) for _ in range(3)] + [_row(deliverable=0, saturated=True) for _ in range(2)]
    assert gate(rows)["verdict"] == "R11_5_PLUS_CAPTURE_POPULATION_GROWTH_PASS"


def test_support_limited_when_all_barren_saturate() -> None:
    """< 3 deliverable and every still-barren scenario's curve flattened -> stop, don't burn seeds."""
    rows = [_row(deliverable=0, saturated=True) for _ in range(5)]
    assert gate(rows)["verdict"] == "R11_5_PLUS_CAPTURE_PROPOSAL_SUPPORT_LIMITED"


def test_partial_when_some_deliverable_but_not_saturated() -> None:
    """Some deliverable found (<3) and a barren scenario is still climbing (not saturated) -> PARTIAL (keep going / diversity)."""
    rows = [_row(deliverable=1), _row(deliverable=1)] + [_row(deliverable=0, saturated=False) for _ in range(3)]
    assert gate(rows)["verdict"] == "R11_5_PLUS_CAPTURE_POPULATION_GROWTH_PARTIAL"


def test_curve_point_cumulative_and_marginal() -> None:
    recs = [{"seed": 2, "deliverable": False, "dtz_mm": 40.0, "sig": ("a",)},
            {"seed": 7, "deliverable": True, "dtz_mm": 12.0, "sig": ("b",)},
            {"seed": 15, "deliverable": True, "dtz_mm": 9.0, "sig": ("b",)}]        # same sig as seed 7 -> not unique
    p5 = _curve_point(recs, 5, 0)
    assert p5["certified"] == 1 and p5["deliverable"] == 0 and p5["first_deliverable_seed"] is None
    p10 = _curve_point(recs, 10, p5["deliverable"])
    assert p10["deliverable"] == 1 and p10["first_deliverable_seed"] == 7 and p10["marginal_deliverable"] == 1
    p20 = _curve_point(recs, 20, p10["deliverable"])
    assert p20["deliverable"] == 2 and p20["unique_descriptors"] == 2 and p20["best_dtz_mm"] == 9.0
    assert BUDGETS == (5, 10, 20, 40)
