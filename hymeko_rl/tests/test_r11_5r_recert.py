"""Tests for the R11.5R re-certification teacher gate + B1 dataset row (synthetic records)."""
from hymeko_rl.experiments.r11_5r_robust_teacher import _b1_sample, teacher_gate


def _row(sid: str, split: str, status: str, t0: float = 0.30, t1: float = 0.90) -> dict:
    return {"scenario_id": sid, "split": split, "status": status, "seed": 0, "source": "recovered",
            "t0": {"survival1": t0, "cvar_dtz": 10.0}, "t1": {"survival1": t1, "cvar_dtz": 5.0},
            "chosen_theta": [0.1] * 6, "chosen_wide": status == "WIDE_RECERTIFIED", "x": [0.0] * 30}


def test_teacher_gate_pass() -> None:
    rows = [_row(f"tr{i}", "train", "WIDE_RECERTIFIED") for i in range(40)]      # all wide, big survival gain
    g = teacher_gate(rows)
    assert g["verdict"] == "R11_5R_ROBUST_TEACHER_RECERTIFICATION_PASS"
    assert g["train_wide_frac"] == 1.0 and g["mean_survival_gain_t1_minus_t0"] == 0.6


def test_teacher_gate_limited_when_few_wide() -> None:
    rows = [_row(f"tr{i}", "train", "WIDE_RECERTIFIED") for i in range(5)] \
        + [_row(f"tr{i}", "train", "NARROW_ONLY", t1=0.30) for i in range(35)]   # only 5/40 wide, tiny gain
    g = teacher_gate(rows)
    assert g["verdict"] == "R11_5R_WIDE_BASIN_SUPPORT_LIMITED" and g["train_wide_frac"] == 0.125


def test_teacher_gate_survival_over_k6_scenarios_only() -> None:
    # NO_NOMINAL_K6 rows must NOT dilute the survival gain (survival is meaningless there).
    rows = [_row(f"tr{i}", "train", "WIDE_RECERTIFIED") for i in range(38)] \
        + [_row(f"no{i}", "train", "NO_NOMINAL_K6", t0=0.0, t1=0.0) for i in range(5)]
    g = teacher_gate(rows)
    assert g["mean_survival_gain_t1_minus_t0"] == 0.6 and g["status_counts"]["NO_NOMINAL_K6"] == 5


def test_b1_sample_uses_chosen_theta() -> None:
    s = _b1_sample(_row("s0", "dev", "WIDE_RECERTIFIED"))
    assert s["theta"] == [0.1] * 6 and s["k6"] is True and s["scenario_id"] == "s0" and s["split"] == "dev"
