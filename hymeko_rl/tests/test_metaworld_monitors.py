"""Unit tests for the MetaWorld coffee-push runtime monitor (synthetic, no MetaWorld dependency).

Proves the monitor framework generalises beyond coin-delivery: the coffee-push template judges the
approach → move-toward-target → reached story on plain position trajectories, reuses the coin-era
``StagnationMonitor`` unchanged (via ``SupportsDistanceSeries``), and its verdict feeds ``export_cip_variables``.
"""
from __future__ import annotations

import pytest

from hymeko_rl.eval.task_monitor import (
    CoffeePushContext,
    CoffeePushMonitor,
    CoffeePushProgressMonitor,
    CoffeePushSuccessMonitor,
    DialTurnContext,
    DialTurnMonitor,
    DialTurnOvershootMonitor,
    DialTurnProgressMonitor,
    DialTurnSuccessMonitor,
    StagnationMonitor,
    export_cip_variables,
)


def _traj(dists, target=(0.0, 0.0), gripper=False, contact=False):
    """Object sits on the +x axis at each given distance from the fixed target → distance == the given value."""
    steps = []
    for d in dists:
        s = {"object_xy": [float(d), 0.0], "target_xy": [float(target[0]), float(target[1])]}
        if gripper:
            s["gripper_xy"] = [float(d), 0.0]
        if contact:
            s["contact"] = True
        steps.append(s)
    return steps


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        CoffeePushProgressMonitor(min_progress=-1.0)
    with pytest.raises(ValueError):
        CoffeePushSuccessMonitor(success_radius=0.0)


def test_empty_trajectory_rejected():
    v = CoffeePushMonitor().evaluate([])
    assert not v.monitor_pass and v.violation_reason == "empty_trajectory"


def test_reaches_target_passes():
    v = CoffeePushMonitor(success_radius=0.05, min_progress=0.01).evaluate(
        _traj([0.30, 0.22, 0.14, 0.08, 0.04, 0.02]))
    assert v.monitor_pass and v.target_reached and v.object_moved_toward_target
    assert v.object_target_distance_final <= 0.05 and v.object_target_distance_delta > 0.0
    assert v.violation_reason == "none"


def test_progress_without_success_is_partial():
    v = CoffeePushMonitor(success_radius=0.05, min_progress=0.01).evaluate(
        _traj([0.30, 0.24, 0.18, 0.14, 0.12]))
    assert not v.monitor_pass                    # not full success …
    assert v.object_moved_toward_target          # … but progress happened (partial)
    assert not v.target_reached
    assert v.violation_reason == "target_not_reached"
    assert v.sub_verdicts["coffee_progress"].passed and not v.sub_verdicts["coffee_success"].passed


def test_no_motion_fails_and_is_stagnation_compatible():
    mon = CoffeePushMonitor(success_radius=0.05, min_progress=0.01,
                            stagnation=StagnationMonitor(window=3, eps=0.005))
    v = mon.evaluate(_traj([0.20] * 8))
    assert not v.monitor_pass and not v.object_moved_toward_target
    assert v.violation_reason == "object_did_not_move_toward_target"
    stag = v.sub_verdicts["stagnation"]          # StagnationMonitor composed on the coffee context
    assert stag.stagnated and stag.stagnation_duration > 0


def test_moves_away_fails():
    v = CoffeePushMonitor(success_radius=0.05, min_progress=0.01).evaluate(
        _traj([0.10, 0.14, 0.20, 0.26, 0.30]))
    assert not v.monitor_pass and not v.object_moved_toward_target
    assert v.object_target_distance_delta < 0.0
    assert v.violation_reason == "object_moved_away_from_target"


def test_noisy_net_improvement_counts_as_progress():
    v = CoffeePushMonitor(success_radius=0.05, min_progress=0.01).evaluate(
        _traj([0.30, 0.34, 0.26, 0.28, 0.20, 0.22, 0.14, 0.10]))   # jitters but nets closer
    assert v.object_moved_toward_target and v.object_target_distance_delta > 0.0
    assert v.sub_verdicts["coffee_progress"].passed
    assert not v.target_reached                  # final 0.10 > 0.05: real progress, not full success


def test_verdict_serializes_all_fields():
    mon = CoffeePushMonitor(stagnation=StagnationMonitor(window=1, eps=0.001))
    d = mon.evaluate(_traj([0.30, 0.10, 0.04], gripper=True, contact=True)).as_dict()
    for k in ("monitor_pass", "monitor_score", "progress_score", "object_target_distance_initial",
              "object_target_distance_final", "object_target_distance_delta", "object_moved_toward_target",
              "target_reached", "violation_reason", "sub_verdicts"):
        assert k in d
    assert set(d["sub_verdicts"]) == {"coffee_progress", "coffee_success", "stagnation"}
    assert "stagnation_duration" in d["sub_verdicts"]["stagnation"]   # StagnationVerdict extra field survives


def test_cip_export_reads_coffee_verdict():
    mon = CoffeePushMonitor(success_radius=0.05, min_progress=0.01,
                            stagnation=StagnationMonitor(window=2, eps=0.005))
    verdict = mon.evaluate(_traj([0.30, 0.20, 0.10, 0.04]))
    exp = export_cip_variables(verdict)
    assert exp.variables["success_monitor_pass"] == verdict.monitor_pass
    assert exp.variables["progress_score"] == verdict.progress_score
    assert "success_monitor_pass" not in exp.missing and "progress_score" not in exp.missing
    assert "stagnation_duration" not in exp.missing   # stagnation composed → sourced (value present)


def test_stagnation_monitor_reused_on_coffee_context():
    # the coin-era StagnationMonitor, unchanged, judges a MetaWorld CoffeePushContext (the generalisation proof)
    ctx = CoffeePushContext.build(_traj([0.20] * 6))
    v = StagnationMonitor(window=2, eps=0.005).evaluate(ctx)
    assert v.stagnated and v.stagnation_duration > 0


# --- dial-turn (angular progress) -----------------------------------------------------------------------------
def _dial(angles, target=0.0, contact=False):
    """Trajectory of scalar dial angles (radians) toward a fixed target angle."""
    steps = []
    for a in angles:
        s = {"dial_angle": float(a), "target_angle": float(target)}
        if contact:
            s["contact"] = True
        steps.append(s)
    return steps


def test_dial_bad_params_rejected():
    with pytest.raises(ValueError):
        DialTurnProgressMonitor(min_rotation=-0.1)
    with pytest.raises(ValueError):
        DialTurnSuccessMonitor(success_tolerance=0.0)
    with pytest.raises(ValueError):
        DialTurnOvershootMonitor(overshoot_tolerance=0.0)
    with pytest.raises(ValueError):
        DialTurnOvershootMonitor(direction="sideways")   # type: ignore[arg-type]  # invalid literal at runtime


def test_dial_empty_trajectory_rejected():
    v = DialTurnMonitor().evaluate([])
    assert not v.monitor_pass and v.violation_reason == "empty_trajectory"


def test_dial_rotates_and_reaches_passes():
    v = DialTurnMonitor().evaluate(_dial([0.50, 0.35, 0.20, 0.08, 0.02], target=0.0))
    assert v.monitor_pass and v.target_reached and v.rotated_toward_target and not v.overshot
    assert v.target_error_final <= 0.05 and v.violation_reason == "none"


def test_dial_progress_without_success_is_partial():
    v = DialTurnMonitor().evaluate(_dial([0.80, 0.60, 0.45, 0.35, 0.30], target=0.0))
    assert not v.monitor_pass and v.rotated_toward_target and not v.target_reached
    assert v.violation_reason == "target_angle_not_reached"


def test_dial_no_rotation_fails_and_is_stagnation_compatible():
    mon = DialTurnMonitor(stagnation=StagnationMonitor(window=3, eps=0.005))
    v = mon.evaluate(_dial([0.50] * 6, target=0.0))
    assert not v.monitor_pass and not v.rotated_toward_target
    assert v.violation_reason == "dial_did_not_rotate_toward_target"
    assert v.sub_verdicts["stagnation"].stagnated   # generic StagnationMonitor over the angular error series


def test_dial_rotates_away_fails():
    v = DialTurnMonitor().evaluate(_dial([0.10, 0.20, 0.35, 0.50, 0.60], target=0.0))
    assert not v.monitor_pass and v.rotated_away_from_target and not v.rotated_toward_target
    assert v.violation_reason == "dial_rotated_away_from_target"


def test_dial_overshoot_detected_and_fails():
    # rotates down past 0 to -0.15 (|.|>overshoot_tol 0.10) then settles at -0.02 (reached): overshoot is the fault
    v = DialTurnMonitor().evaluate(_dial([0.50, 0.20, -0.15, -0.02], target=0.0))
    assert v.overshot and not v.monitor_pass
    assert v.target_reached and v.rotated_toward_target
    assert v.violation_reason == "dial_overshot_target"


def test_dial_angle_wrapping_no_false_failure():
    # dial crosses the +pi/-pi seam (3.14 -> -3.13) toward a target near -pi: must read as smooth approach
    v = DialTurnMonitor().evaluate(_dial([2.90, 3.05, 3.14, -3.13], target=-3.10))
    assert v.monitor_pass and v.target_reached
    assert v.target_error_final < 0.05
    assert v.angle_delta == pytest.approx(0.2532, abs=0.01)   # net +0.25 rad rotation across the seam, not -6.03


def test_dial_verdict_serializes_all_fields():
    mon = DialTurnMonitor(stagnation=StagnationMonitor(window=1, eps=0.001))
    d = mon.evaluate(_dial([0.50, 0.20, 0.02], target=0.0, contact=True)).as_dict()
    for k in ("monitor_pass", "monitor_score", "progress_score", "angle_initial", "angle_final",
              "target_angle", "angle_delta", "target_error_initial", "target_error_final",
              "rotated_toward_target", "rotated_away_from_target", "target_reached", "overshot",
              "violation_reason", "sub_verdicts"):
        assert k in d
    assert set(d["sub_verdicts"]) == {"dial_progress", "dial_success", "dial_overshoot", "stagnation"}


def test_dial_cip_export_reads_verdict():
    mon = DialTurnMonitor(stagnation=StagnationMonitor(window=2, eps=0.005))
    verdict = mon.evaluate(_dial([0.50, 0.30, 0.10, 0.02], target=0.0))
    exp = export_cip_variables(verdict)
    assert exp.variables["success_monitor_pass"] == verdict.monitor_pass
    assert exp.variables["progress_score"] == verdict.progress_score
    assert "success_monitor_pass" not in exp.missing and "progress_score" not in exp.missing
    assert "stagnation_duration" not in exp.missing


def test_no_metaworld_dependency():
    import sys
    # the real MetaWorld package is not required or imported by the monitor module (our submodule key is the
    # dotted path 'hymeko_rl.eval.task_monitor.metaworld', not the top-level 'metaworld' package)
    assert "metaworld" not in sys.modules
    from hymeko_rl.eval.task_monitor import DialTurnMonitor as _D   # importable without MetaWorld installed
    assert _D is not None


def test_stagnation_monitor_reused_on_dial_context():
    # SAME StagnationMonitor, now over an angular-error series (DialTurnContext.dist) — angular generalisation
    ctx = DialTurnContext.build(_dial([0.50] * 6, target=0.0))
    v = StagnationMonitor(window=2, eps=0.005).evaluate(ctx)
    assert v.stagnated and v.stagnation_duration > 0
