"""Offline anti-farming validation — the eight decoupled counterfactual trajectory classes."""
from __future__ import annotations

from hymeko_rl.eval.reward_repair.anti_farming_validation import TRAJECTORIES, run_anti_farming_validation


def test_eight_trajectory_classes() -> None:
    assert set(TRAJECTORIES) == {"far", "approach", "hover_farm", "bare_contact_farm", "grasp_lift_no_delivery",
                                 "delivery", "success", "proxy_exploit"}


def test_validation_healthy_and_proxy_suppressed(tmp_path) -> None:
    """monitor_aligned passes all five health checks and suppresses the proxy-exploit that farms `original`."""
    s = run_anti_farming_validation(out_dir=tmp_path)
    assert s["verdict"]["healthy"] is True
    tot = {v: {c: s["per_class"][c][v]["total"] for c in TRAJECTORIES} for v in
           ("original", "mw_in_place_off", "monitor_aligned")}
    # original farms the proxy (≈ delivery); monitor_aligned suppresses it far below delivery
    assert tot["original"]["proxy_exploit"] > 0.5 * tot["original"]["delivery"]
    assert tot["monitor_aligned"]["proxy_exploit"] < 0.3 * tot["monitor_aligned"]["delivery"]
    # true delivery beats every farming class; success is strongest
    for f in ("hover_farm", "bare_contact_farm", "proxy_exploit"):
        assert tot["monitor_aligned"]["delivery"] > tot["monitor_aligned"][f]
    assert tot["monitor_aligned"]["success"] == max(tot["monitor_aligned"].values())
    assert (tmp_path / "anti_farming_validation.json").exists()
