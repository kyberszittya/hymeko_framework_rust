"""Tests for the R11.5+++ re-A/B 3-arm gate (pure)."""
from hymeko_rl.experiments.r11_5ppp_reab import gate


def _row(*, a0: bool, a1: bool, a2: bool, safe: bool = True, certified: bool = True, grasps: int = 12) -> dict:
    return {"n_grasps": grasps, "a0_k6": a0, "a1_k6": a1, "a2_k6": a2, "all_safe": safe, "all_certified": certified}


def test_pass_needs_eight_a2_and_plus3_over_a0_and_a2_not_worse_than_a1() -> None:
    # A0 = 4/10 (current selection at N=10), A1 = 4/10 (depth alone adds nothing), A2 = 8/10 (ranking on the enlarged bank)
    rows = ([_row(a0=True, a1=True, a2=True) for _ in range(4)]          # A0=A1=A2 K6
            + [_row(a0=False, a1=False, a2=True) for _ in range(4)]      # ranking recovers 4 more -> A2=8
            + [_row(a0=False, a1=False, a2=False) for _ in range(2)])
    g = gate(rows)
    assert g["a0_k6"] == "4/10" and g["a2_k6"] == "8/10" and g["ranking_gain"] == 4 and g["population_depth_gain"] == 0
    assert g["verdict"] == "R11_5_PLUS_DELIVERABILITY_RANKED_ENLARGED_POPULATION_PASS"


def test_insufficient_when_a2_below_eight() -> None:
    rows = [_row(a0=True, a1=True, a2=True) for _ in range(4)] + [_row(a0=False, a1=False, a2=True) for _ in range(3)] \
        + [_row(a0=False, a1=False, a2=False) for _ in range(3)]        # A2 = 7 < 8
    assert gate(rows)["verdict"].endswith("INSUFFICIENT")


def test_insufficient_when_a2_worse_than_a1() -> None:
    """A2 must never be worse than A1 (the ranking includes A1's grasp in the bank) — a violation blocks PASS."""
    rows = [_row(a0=True, a1=True, a2=True) for _ in range(7)] + [_row(a0=False, a1=True, a2=False)] \
        + [_row(a0=False, a1=False, a2=True) for _ in range(2)]         # one scenario A1 K6 but A2 not
    g = gate(rows)
    assert g["a2_not_worse_than_a1"] is False and g["verdict"].endswith("INSUFFICIENT")


def test_plus3_requirement() -> None:
    """A2 >= A0 + 3: if population is already strong (A0=6) A2 must reach >= 9 (also clears >=8)."""
    rows = [_row(a0=True, a1=True, a2=True) for _ in range(6)] + [_row(a0=False, a1=False, a2=True) for _ in range(2)] \
        + [_row(a0=False, a1=False, a2=False) for _ in range(2)]        # A0=6, A2=8 -> A2-A0=2 < 3
    assert gate(rows)["verdict"].endswith("INSUFFICIENT")
