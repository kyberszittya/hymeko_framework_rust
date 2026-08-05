"""Tests for R11.6D causal-audit logic: swap-axis coverage, component-diff, and the recovery summary."""
from hymeko_rl.coin_delivery.handoff_audit import SwapComponent, _NEEDS_METRICS
from hymeko_rl.experiments.r11_6d_handoff_audit import _component_diff, summarize


def test_swap_axes_and_metrics_refresh_set() -> None:
    axes = {c.value for c in SwapComponent}
    assert axes == {"coin_pos", "coin_yaw", "coin_linvel", "coin_spin", "arm_qpos", "arm_qvel", "prev_tau", "zone"}
    # coin-state + zone swaps must refresh the cached planar metrics; arm / prev_tau must NOT need it.
    assert _NEEDS_METRICS == {SwapComponent.COIN_POS, SwapComponent.COIN_YAW, SwapComponent.COIN_LINVEL,
                              SwapComponent.COIN_SPIN, SwapComponent.ZONE}
    assert SwapComponent.ARM_QPOS not in _NEEDS_METRICS and SwapComponent.PREV_TAU not in _NEEDS_METRICS


def _audit(coin_xy, coin_speed=0.1, zone=(0.3, 0.0), arm_qpos=(0.0,) * 4) -> dict:
    return {"coin_xy": list(coin_xy), "coin_yaw": 0.1, "coin_speed": coin_speed, "coin_spin": 0.2,
            "arm_qpos": list(arm_qpos), "arm_qvel": [0.0] * 4, "prev_tau": [0.0] * 4, "zone": list(zone)}


def test_component_diff_magnitudes() -> None:
    dev = _audit((0.10, 0.00), coin_speed=0.30, zone=(0.30, 0.00))
    bank = _audit((0.10, 0.00), coin_speed=0.10, zone=(0.25, 0.00))
    assert _component_diff(dev, bank, SwapComponent.COIN_POS) == 0.0            # identical coin xy
    assert _component_diff(dev, bank, SwapComponent.COIN_LINVEL) == 0.2         # |0.30 - 0.10|
    assert _component_diff(dev, bank, SwapComponent.ZONE) == 50.0              # 0.05 m -> 50 mm


def _row(recovering: list, gap_by_axis: dict, control_k6: bool = True, baseline_k6: bool = False) -> dict:
    axes = ["coin_pos", "coin_yaw", "coin_linvel", "coin_spin", "arm_qpos", "arm_qvel", "prev_tau", "zone"]
    return {"dev": "d", "control": {"k6": control_k6}, "baseline": {"k6": baseline_k6}, "recovering": recovering,
            "counterfactual": {a: {"gap_gain": gap_by_axis.get(a, 0.0)} for a in axes}}


def test_summarize_ranks_dominant_recovering_axis() -> None:
    rows = [_row(["coin_linvel"], {"coin_linvel": 0.3}),
            _row(["coin_linvel", "zone"], {"coin_linvel": 0.28, "zone": 0.31}),
            _row(["coin_linvel"], {"coin_linvel": 0.25})]
    s = summarize(rows)
    assert s["controls_all_k6"] and s["n_diagnosable"] == 3
    assert s["recover_count"]["coin_linvel"] == 3 and s["recover_count"]["zone"] == 1
    assert s["dominant"] == "coin_linvel"


def test_summarize_excludes_blend_artifact_from_diagnosis() -> None:
    # a pair whose baseline already delivers (nearest theta transports) is a blend artifact, not diagnosed.
    rows = [_row([], {}, baseline_k6=True), _row(["coin_linvel"], {"coin_linvel": 0.3})]
    s = summarize(rows)
    assert s["n_diagnosable"] == 1 and s["blend_artifact_dev"] and s["dominant"] == "coin_linvel"


def test_summarize_flags_broken_control() -> None:
    rows = [_row([], {}, control_k6=False)]                                    # a control that fails invalidates the pair
    s = summarize(rows)
    assert not s["controls_all_k6"] and s["dominant"] is not None
