"""Tests for the R11.4B gate + negative classifier (synthetic eval rows — no rollout)."""
from hymeko_rl.experiments.r11_4b_conditioned_bc import _classify_negative, _held, _rate, bc_gate


def _row(sid: str, split: str, *, mean: bool, nn: bool, ridge: bool, mlp: bool, r2: bool,
         nn_dist: float = 1.0, mlp_safe: bool = True, teacher: bool = True) -> dict:
    def cell(k6: bool, safe: bool = True) -> dict:
        return {"k6": k6, "safe": safe, "dtz_mm": 5.0 if k6 else 40.0}
    return {"scenario_id": sid, "split": split, "source": "recovered", "nn_distance": nn_dist,
            "teacher": cell(teacher), "mean_theta": cell(mean), "nearest_schedule": cell(nn),
            "ridge": cell(ridge), "mlp_bc": cell(mlp, mlp_safe), "frozen_r2": cell(r2)}


def _pass_rows() -> list:
    rows = []
    for i in range(10):                                  # train: MLP 9/10, R2 3/10
        rows.append(_row(f"tr{i}", "train", mean=i < 5, nn=i < 8, ridge=i < 7, mlp=i < 9, r2=i < 3))
    for i in range(4):                                   # dev: MLP 3/4, R2 1/4, best baseline (nn) 2/4
        rows.append(_row(f"dv{i}", "dev", mean=False, nn=i < 2, ridge=i < 1, mlp=i < 3, r2=i < 1))
    for i in range(4):                                   # test: MLP 3/4, R2 1/4, best baseline 2/4
        rows.append(_row(f"te{i}", "test", mean=False, nn=i < 2, ridge=i < 1, mlp=i < 3, r2=i < 1))
    return rows


def test_rate_and_held() -> None:
    rows = _pass_rows()
    assert _rate(rows, "mlp_bc", "train") == 0.9
    assert _held(rows, "mlp_bc") == 0.75            # 6/8 held-out


def test_gate_pass() -> None:
    g = bc_gate(_pass_rows())
    assert g["verdict"] == "R11_4B_CONDITIONED_DELIVERY_BC_PASS"
    assert g["bc_safe"] and g["teacher_theta_reproduces_k6"]


def test_gate_fails_when_bc_not_better_than_r2() -> None:
    rows = _pass_rows()
    for r in rows:                                   # lift R2 to match BC on held-out => BC not strictly better
        if r["split"] in ("dev", "test"):
            r["frozen_r2"] = {"k6": r["mlp_bc"]["k6"], "safe": True, "dtz_mm": 5.0}
    assert bc_gate(rows)["verdict"] != "R11_4B_CONDITIONED_DELIVERY_BC_PASS"


def test_classify_optimization_failure_when_mlp_underfits_train() -> None:
    rows = [_row(f"tr{i}", "train", mean=True, nn=True, ridge=True, mlp=i < 3, r2=False) for i in range(10)]
    rows += [_row(f"dv{i}", "dev", mean=False, nn=False, ridge=False, mlp=False, r2=False) for i in range(2)]
    g = bc_gate(rows)
    assert g["verdict"] == "BC_OPTIMIZATION_FAILURE"      # mlp train 0.3 < ridge/nn/mean train 1.0


def test_classify_representation_when_smooth_regressors_fail_on_train() -> None:
    # ridge & mlp both ~0.4 on train (chaotic/narrow-basin target); 1-NN trivially 1.0 must NOT trigger OPTIMIZATION.
    rows = [_row(f"tr{i}", "train", mean=i < 3, nn=True, ridge=i < 4, mlp=i < 4, r2=False) for i in range(10)]
    rows += [_row(f"dv{i}", "dev", mean=False, nn=False, ridge=False, mlp=False, r2=False) for i in range(2)]
    assert bc_gate(rows)["verdict"] == "BC_REPRESENTATION_INSUFFICIENT"


def test_classify_data_coverage_when_misses_are_far() -> None:
    rows = [_row(f"tr{i}", "train", mean=True, nn=True, ridge=True, mlp=True, r2=False) for i in range(10)]
    rows.append(_row("dv0", "dev", mean=False, nn=False, ridge=False, mlp=False, r2=False, nn_dist=9.0))
    rows.append(_row("dv1", "dev", mean=False, nn=False, ridge=False, mlp=True, r2=False, nn_dist=0.5))
    rates = {b: {"train": 1.0} for b in ("mean_theta", "nearest_schedule", "ridge", "mlp_bc")}   # mlp fits train
    assert _classify_negative(rows, rates) == "BC_DATA_COVERAGE_INSUFFICIENT"   # miss far from train (9.0 > hit 0.5)
