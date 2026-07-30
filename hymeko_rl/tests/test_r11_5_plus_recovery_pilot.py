"""Tests for the R11.5+ recovery-pilot gate logic (pure)."""
from hymeko_rl.experiments.r11_5_plus_recovery_pilot import NEG_X, gate


def _row(cat: str, *, certified: bool = True, recovered: bool = False, safe: bool = True,
         energy: bool = True, two_adds: bool = False) -> dict:
    return {"category": cat, "certified": certified, "recovered": recovered, "safe": safe,
            "energy_complete": energy, "two_stage_adds": two_adds}


def test_gate_pass_needs_six_and_two_negative_x() -> None:
    rows = ([_row(NEG_X, recovered=True) for _ in range(2)]                      # 2 negative-x recovered
            + [_row("INSUFFICIENT_TRANSPORT_PROGRESS", recovered=True) for _ in range(4)]  # +4 = 6 total
            + [_row("CAPTURE_SUPPORT_FAILURE", certified=False) for _ in range(6)])
    assert gate(rows)["verdict"] == "R11_5_PLUS_RESIDUAL_RECOVERY_PILOT_PASS"


def test_gate_insufficient_when_under_six() -> None:
    rows = [_row(NEG_X, recovered=True), _row(NEG_X, recovered=True)] + [_row("INSUFFICIENT_TRANSPORT_PROGRESS")] * 10
    assert gate(rows)["verdict"] == "R11_5_PLUS_PIPELINE_PASS_RESIDUAL_RECOVERY_INSUFFICIENT"   # only 2 recovered


def test_gate_negative_x_clause_vacuous_when_fewer_than_two_available() -> None:
    """With <2 negative-x scenarios in the set, the >=2-negative-x sub-clause must not block a >=6 recovery."""
    rows = [_row("INSUFFICIENT_TRANSPORT_PROGRESS", recovered=True) for _ in range(6)] + [_row("CAPTURE_SUPPORT_FAILURE", certified=False)]
    g = gate(rows)
    assert g["verdict"] == "R11_5_PLUS_RESIDUAL_RECOVERY_PILOT_PASS" and g["negative_x_recovered"] == "0/0"


def test_gate_blocks_on_safety_or_energy() -> None:
    base = [_row(NEG_X, recovered=True) for _ in range(2)] + [_row("INSUFFICIENT_TRANSPORT_PROGRESS", recovered=True) for _ in range(4)]
    assert gate(base + [_row("INSUFFICIENT_TRANSPORT_PROGRESS", safe=False)])["verdict"].endswith("INSUFFICIENT")
    assert gate(base + [_row("INSUFFICIENT_TRANSPORT_PROGRESS", energy=False)])["verdict"].endswith("INSUFFICIENT")
