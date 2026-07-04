"""The declarative reward spec — terms + weights read from a ``.hymeko`` task profile
drive ``step()``'s reward (no hard-coded ``-dist``).

Covers the pure spec/reader (no MuJoCo) and an equivalence check that the spec-driven
reward reproduces the former ``-dist`` exactly.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hymeko_rl.env.reward import REACH_REWARD, RewardSpec, read_reward_terms

_REPO = Path(__file__).resolve().parents[2]
_TASK = _REPO / "data" / "robotics" / "arm_reach_task.hymeko"


def _env_stub(reach_thresh: float = 0.06) -> SimpleNamespace:
    """A minimal stand-in: the reward terms only read ``reach_thresh`` off the env."""
    return SimpleNamespace(reach_thresh=reach_thresh)


# ── spec (pure) ──────────────────────────────────────────────────────────────
def test_reach_reward_is_negative_distance() -> None:
    env = _env_stub()
    assert REACH_REWARD.terms == (("reach_distance", 1.0),)
    assert REACH_REWARD.evaluate(env, 0.5, np.zeros(4)) == pytest.approx(-0.5)
    assert REACH_REWARD.evaluate(env, 0.0, np.zeros(4)) == pytest.approx(0.0)


def test_weighted_terms_sum() -> None:
    spec = RewardSpec(
        (("reach_distance", 1.0), ("success_bonus", 2.0), ("action_cost", 0.1)))
    env = _env_stub(reach_thresh=0.1)
    # dist 0.05 < thresh → bonus fires; action ‖·‖²=2 → cost −0.2.
    # −0.05 + 2.0·1.0 + 0.1·(−2.0) = 1.75
    assert spec.evaluate(env, 0.05, np.array([1.0, 1.0])) == pytest.approx(1.75)
    # dist 0.5 ≥ thresh → no bonus; zero action → no cost. → −0.5
    assert spec.evaluate(env, 0.5, np.zeros(2)) == pytest.approx(-0.5)


def test_coin_pregrasp_still_gates_on_grasp() -> None:
    """The pre-grasp stillness term penalises the coin's speed only until it is grasped, then 0."""
    from hymeko_rl.env.reward import _term_coin_pregrasp_still
    env = SimpleNamespace(_planar_metrics=SimpleNamespace(disk_speed=0.4), _ever_grasped=False)
    assert _term_coin_pregrasp_still(env, 0.0, np.zeros(4)) == pytest.approx(-0.4)   # not grasped → -speed
    env._ever_grasped = True
    assert _term_coin_pregrasp_still(env, 0.0, np.zeros(4)) == pytest.approx(0.0)     # grasped → off
    assert _term_coin_pregrasp_still(SimpleNamespace(), 0.0, np.zeros(4)) == pytest.approx(0.0)  # non-planar


def test_grasp_deliver_gates_success_on_an_actual_grasp() -> None:
    """Grasp-gated success: +1 only when the coin is in the zone AND was grasped — a knock (in-zone but never
    grasped) earns 0, closing the shortcut the bare in_zone bonus rewarded."""
    from hymeko_rl.env.reward import _term_grasp_deliver
    knock = SimpleNamespace(_planar_metrics=SimpleNamespace(in_zone=True), _ever_grasped=False)
    assert _term_grasp_deliver(knock, 0.0, np.zeros(4)) == pytest.approx(0.0)        # knock → no success
    grasped = SimpleNamespace(_planar_metrics=SimpleNamespace(in_zone=True), _ever_grasped=True)
    assert _term_grasp_deliver(grasped, 0.0, np.zeros(4)) == pytest.approx(1.0)      # grasp + in zone → success
    not_yet = SimpleNamespace(_planar_metrics=SimpleNamespace(in_zone=False), _ever_grasped=True)
    assert _term_grasp_deliver(not_yet, 0.0, np.zeros(4)) == pytest.approx(0.0)      # grasped but not delivered
    assert _term_grasp_deliver(SimpleNamespace(), 0.0, np.zeros(4)) == pytest.approx(0.0)  # non-planar env


def test_unknown_or_empty_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reward term"):
        RewardSpec((("reach_distance", 1.0), ("nope", 1.0)))
    with pytest.raises(ValueError, match="at least one term"):
        RewardSpec(())


# ── reader (the .hymeko bridge) ──────────────────────────────────────────────
def test_read_reward_terms_from_task_profile() -> None:
    """The reader recovers the reaching profile's term + weight — so the .hymeko reward
    and the default Python spec agree."""
    assert read_reward_terms(_TASK) == (("reach_distance", 1.0),)
    assert RewardSpec.from_hymeko(_TASK).terms == (("reach_distance", 1.0),)


def test_reader_parses_weights_and_order(tmp_path: Path) -> None:
    prof = tmp_path / "r.hymeko"
    prof.write_text(
        "p {\n"
        "  @d: rew.reach_distance { weight 0.5; (+ f, - t); }\n"
        "  @b: rew.success_bonus { weight 3.0; }\n"
        "  @s: r.reward_spec { (+ d, + b); }\n"
        "}\n")
    assert read_reward_terms(prof) == (("reach_distance", 0.5), ("success_bonus", 3.0))


def test_reader_parses_arc_weights(tmp_path: Path) -> None:
    """Weights live on the bundle ARC (the hyperedge); the term nodes carry none."""
    prof = tmp_path / "arc.hymeko"
    prof.write_text(
        "p {\n"
        "  @d: rew.reach_distance { (+ f, - t); }\n"
        "  @b: rew.success_bonus {}\n"
        "  @s: r.reward_spec { (+ d 0.5, + b 3.0); }\n"
        "}\n")
    assert read_reward_terms(prof) == (("reach_distance", 0.5), ("success_bonus", 3.0))


def test_arc_weight_overrides_body_then_defaults(tmp_path: Path) -> None:
    """Arc weight wins over a body weight; absent arc weight falls back to body, then to 1.0."""
    prof = tmp_path / "mix.hymeko"
    prof.write_text(
        "p {\n"
        "  @a: rew.reach_distance { weight 9.0; }\n"   # body 9.0 but arc 0.5 -> 0.5 wins
        "  @b: rew.success_bonus { weight 3.0; }\n"    # no arc weight -> body 3.0
        "  @c: rew.action_cost {}\n"                   # neither -> default 1.0
        "  @s: r.reward_spec { (+ a 0.5, + b, + c); }\n"
        "}\n")
    assert read_reward_terms(prof) == (
        ("reach_distance", 0.5), ("success_bonus", 3.0), ("action_cost", 1.0))


def test_reworked_task_profiles_have_arc_weights() -> None:
    """The reworked Galambos + FANUC rewards declare every weight on the bundle arc (regression:
    the values match the pre-rework body weights exactly)."""
    gala = read_reward_terms(_REPO / "data" / "robotics" / "galambos_task.hymeko")
    assert gala == (   # 4-core (approach·both·zone·oob, grasp-fraction 0.615) + the COLLAB extension 2026-07-03:
                       # finger_contact (denser per-fingertip reward) and arm_body_collision (upper-arm-only, the
                       # two-agent coordination constraint that excludes the pinch — not the grasp-killing 2.0).
        ("grasp_approach", 4.0), ("both_contact", 5.0), ("finger_contact", 1.5), ("in_zone", 10.0),
        ("out_of_bounds", 2.0), ("arm_body_collision", 0.5))
    pick = read_reward_terms(_REPO / "data" / "robotics" / "pick_place_task.hymeko")
    assert pick == (
        ("pick_approach", 1.0), ("pick_contact", 0.5), ("pick_lift", 5.0),
        ("pick_place_distance", 1.0), ("pick_place_bonus", 20.0),
        ("pick_approach_penalty", 2.0), ("pick_disturbance", 3.0))


def test_fingers_collision_term() -> None:
    """The narrow finger-finger collision term: -1 only when the two fingertip links touch each other, else 0;
    inert on a non-planar env. (Regression for the 'proper noclash' that replaced the grasp-killing arm-wide one.)"""
    import dataclasses

    import numpy as np

    from hymeko_rl.env.planar_grasp_env import CLEAN_PLANAR
    from hymeko_rl.env.reward import _REWARD_TERMS

    fn = _REWARD_TERMS["fingers_collision"]

    class _E:
        pass

    e = _E()
    assert fn(e, 0.0, np.zeros(2)) == 0.0                                   # no _planar_metrics -> inert
    e._planar_metrics = CLEAN_PLANAR                                        # fingers_self_contact False (default)
    assert fn(e, 0.0, np.zeros(2)) == 0.0
    e._planar_metrics = dataclasses.replace(CLEAN_PLANAR, fingers_self_contact=True)
    assert fn(e, 0.0, np.zeros(2)) == -1.0                                  # fingers crashing -> -1


def test_arm_body_collision_term() -> None:
    """Upper-arm collision (2026-07-03, collab coordination): -1 when the two arms crash at a NON-fingertip body,
    but 0 on the coin-pinch (arms not touching) AND 0 when only the fingertips touch (that is `fingers_collision`'s
    narrow job) — so it adds the two-agent 'don't slam together' constraint WITHOUT fighting the grasp, unlike the
    whole-arm `arm_collision` that at 2.0 drove grasp-fraction 0.615->0.0. Inert on a non-planar env."""
    import dataclasses

    import numpy as np

    from hymeko_rl.env.planar_grasp_env import CLEAN_PLANAR
    from hymeko_rl.env.reward import _REWARD_TERMS

    fn = _REWARD_TERMS["arm_body_collision"]
    assert fn(SimpleNamespace(), 0.0, np.zeros(2)) == 0.0                             # no _planar_metrics -> inert
    assert fn(SimpleNamespace(_planar_metrics=CLEAN_PLANAR), 0.0, np.zeros(2)) == 0.0  # no contact
    crash = dataclasses.replace(CLEAN_PLANAR, arm_self_contact=True)                  # arms crash (non-fingertip)
    assert fn(SimpleNamespace(_planar_metrics=crash), 0.0, np.zeros(2)) == -1.0
    pinch = dataclasses.replace(CLEAN_PLANAR, left_contact=True, right_contact=True)  # grasping: arms NOT touching
    assert fn(SimpleNamespace(_planar_metrics=pinch), 0.0, np.zeros(2)) == 0.0        # THE point: no penalty on a grasp
    fingertip_only = dataclasses.replace(CLEAN_PLANAR, arm_self_contact=True, fingers_self_contact=True)
    assert fn(SimpleNamespace(_planar_metrics=fingertip_only), 0.0, np.zeros(2)) == 0.0   # fingertip pair EXCLUDED


def test_finger_contact_term() -> None:
    """Graded per-fingertip coin contact (2026-07-03): +1 per touching fingertip (0/1/2) — denser than the
    all-or-nothing `both_contact`, so one fingertip on the coin already pays. Inert on a non-planar env."""
    import dataclasses

    import numpy as np

    from hymeko_rl.env.planar_grasp_env import CLEAN_PLANAR
    from hymeko_rl.env.reward import _REWARD_TERMS

    fn = _REWARD_TERMS["finger_contact"]
    assert fn(SimpleNamespace(), 0.0, np.zeros(2)) == 0.0                             # non-planar -> inert
    assert fn(SimpleNamespace(_planar_metrics=CLEAN_PLANAR), 0.0, np.zeros(2)) == 0.0  # no contact -> 0
    one = dataclasses.replace(CLEAN_PLANAR, left_contact=True)
    assert fn(SimpleNamespace(_planar_metrics=one), 0.0, np.zeros(2)) == 1.0          # one fingertip -> +1
    both = dataclasses.replace(CLEAN_PLANAR, left_contact=True, right_contact=True)
    assert fn(SimpleNamespace(_planar_metrics=both), 0.0, np.zeros(2)) == 2.0         # both fingertips -> +2


def test_read_arc_weights_general(tmp_path: Path) -> None:
    """The general arc-weight capability: signed arcs + weights of any named hyperedge."""
    from hymeko_rl.env._profile import read_arc_weights
    prof = tmp_path / "e.hymeko"
    prof.write_text("p {\n  @e: ns.kind { (+ a 4.0, - b, ~ c 0.5); }\n}\n")
    assert read_arc_weights(prof, "e") == [("+", "a", 4.0), ("-", "b", None), ("~", "c", 0.5)]
    with pytest.raises(ValueError, match="no hyperedge"):
        read_arc_weights(prof, "nope")


def test_read_arc_weights_on_reward_bundle() -> None:
    """Reads the weights straight off the Galambos reward hyperedge's arcs."""
    from hymeko_rl.env._profile import read_arc_weights
    arcs = read_arc_weights(_REPO / "data" / "robotics" / "galambos_task.hymeko", "grasp_reward")
    assert ("+", "approach", 4.0) in arcs
    assert ("+", "both", 5.0) in arcs              # SIMPLIFIED 2026-06-28: strong contact reward
    assert ("+", "zone", 10.0) in arcs             # ungated clean delivery (the 4-term reward)
    assert ("+", "oob", 2.0) in arcs
    assert all(sign == "+" for sign, _m, _w in arcs)


def test_reader_rejects_missing_reward_spec(tmp_path: Path) -> None:
    bad = tmp_path / "no_spec.hymeko"
    bad.write_text("p { @d: rew.reach_distance { weight 1.0; (+ f); } }")
    with pytest.raises(ValueError, match="reward_spec"):
        read_reward_terms(bad)


# ── equivalence: spec-driven reward == former -dist ──────────────────────────
def test_env_reward_matches_minus_dist() -> None:
    """``step()``'s reward reproduces the old ``-dist`` exactly (4-DOF arm_world)."""
    from hymeko_rl.env.arm_reach_env import ArmReachEnv

    env = ArmReachEnv(control_mode="torque")
    assert env.reward_spec is REACH_REWARD
    env.reset(seed=4)
    _, reward, _, _, info = env.step(np.zeros(env.n_actions, dtype=np.float32))
    assert reward == pytest.approx(-info["dist"])


def test_both_approach_penalises_the_farther_arm() -> None:
    """`both_approach` = -max(left,right) is the coordination gradient the mean `grasp_approach` lacks:
    with one arm close and one far, the mean is compensable but the max is dominated by the FAR arm.
    Would fail against a mean-based implementation."""
    from hymeko_rl.env.reward import _term_both_approach, _term_grasp_approach

    a = np.zeros(4, dtype=np.float32)
    lopsided = SimpleNamespace(_planar_metrics=SimpleNamespace(left_tip_dist=0.0, right_tip_dist=0.4))
    both_close = SimpleNamespace(_planar_metrics=SimpleNamespace(left_tip_dist=0.2, right_tip_dist=0.2))
    # mean is identical for the two configs (0.2) — cannot tell them apart:
    assert _term_grasp_approach(lopsided, 0.0, a) == pytest.approx(_term_grasp_approach(both_close, 0.0, a))
    # both_approach STRICTLY prefers the simultaneous config (penalises the lagging arm):
    assert _term_both_approach(both_close, 0.0, a) > _term_both_approach(lopsided, 0.0, a)
    assert _term_both_approach(lopsided, 0.0, a) == pytest.approx(-0.4)
    assert _term_both_approach(SimpleNamespace(), 0.0, a) == 0.0   # 0 on a non-planar env


def test_coord_hymeko_adds_only_both_approach() -> None:
    """The A/B variant `.hymeko` is the baseline reward + `both_approach` (weight 4.0), nothing else."""
    base = dict(RewardSpec.from_hymeko(_REPO / "data" / "robotics" / "galambos_task.hymeko").terms)
    coord = dict(RewardSpec.from_hymeko(_REPO / "data" / "robotics" / "galambos_task_coord.hymeko").terms)
    assert set(coord) - set(base) == {"both_approach"}
    assert coord["both_approach"] == pytest.approx(4.0)


def test_terminal_deliver_is_one_shot_on_completion_not_an_annuity() -> None:
    """`terminal_deliver` fires +1 EXACTLY on the step that completes the dwell (reward is evaluated before the
    `_success++`, so that step has `_success == success_steps-1`), and 0 while holding past it — the anti-farming
    property. Would fire every held step (an annuity) under a naive `_success >= success_steps` implementation."""
    from hymeko_rl.env.reward import _term_terminal_deliver

    a = np.zeros(4, dtype=np.float32)
    def env(in_zone: bool, success: int, ss: int = 5) -> SimpleNamespace:
        return SimpleNamespace(_planar_metrics=SimpleNamespace(in_zone=in_zone),
                               _success=success, success_steps=ss)
    assert _term_terminal_deliver(env(True, 4), 0.0, a) == 1.0    # completing step (after ++ = 5 = terminate)
    assert _term_terminal_deliver(env(True, 3), 0.0, a) == 0.0    # not yet
    assert _term_terminal_deliver(env(True, 5), 0.0, a) == 0.0    # HELD past completion — no annuity
    assert _term_terminal_deliver(env(False, 4), 0.0, a) == 0.0   # out of zone
    assert _term_terminal_deliver(object(), 0.0, a) == 0.0         # non-planar env


def test_deliver_hymeko_is_the_oracle_certified_de_annuitized_shape() -> None:
    """The deliver variant is the ORACLE-CERTIFIED de-annuitized shape: the farmable per-step in_zone/both/
    fingertouch annuity is DROPPED; the grasp gradient is carried by both_approach (dense, not farmable) +
    approach, plus the one-shot terminal_deliver. No in_zone/both_contact/finger_contact (all farmable)."""
    dlv = dict(RewardSpec.from_hymeko(_REPO / "data" / "robotics" / "galambos_task_deliver.hymeko").terms)
    assert set(dlv) == {"grasp_approach", "both_approach", "out_of_bounds", "terminal_deliver"}
    assert dlv["terminal_deliver"] == pytest.approx(30.0)
    assert "in_zone" not in dlv and "both_contact" not in dlv and "finger_contact" not in dlv  # no annuity


def test_pbrs_shaping_is_progress_and_telescopes_unfarmable() -> None:
    """Potential-based shaping (Ng-Harada-Russell): F = γΦ(s')-Φ(s). First call inits Φ(s0) (returns 0); a step
    of progress (coin nearer the zone) pays >0; and dipping in-then-out telescopes to ~0 (the γ<1 residual only),
    so it CANNOT be farmed like the per-step in_zone annuity. Would fail against a raw per-step -dist term."""
    from types import SimpleNamespace

    from hymeko_rl.env.reward import _PBRS_GAMMA, _term_zone_progress

    a = np.zeros(4, dtype=np.float32)
    env = SimpleNamespace(_planar_metrics=SimpleNamespace(disk_to_zone=0.20))
    assert _term_zone_progress(env, 0.0, a) == 0.0                      # first call inits Φ(s0), no shaping
    env._planar_metrics = SimpleNamespace(disk_to_zone=0.15)            # progress toward zone
    f_prog = _term_zone_progress(env, 0.0, a)
    assert f_prog == pytest.approx(_PBRS_GAMMA * (-0.15) - (-0.20))     # exact γΦ(s')-Φ(s)
    assert f_prog > 0.0                                                  # progress pays
    env._planar_metrics = SimpleNamespace(disk_to_zone=0.20)           # back out
    f_back = _term_zone_progress(env, 0.0, a)
    assert abs(f_prog + f_back) < 0.01                                  # in->out telescopes ~0 (unfarmable)
    assert _term_zone_progress(object(), 0.0, a) == 0.0                 # non-planar env


def test_pbrs_hymeko_is_task_plus_potential_shaping() -> None:
    dlv = dict(RewardSpec.from_hymeko(_REPO / "data" / "robotics" / "galambos_task_pbrs.hymeko").terms)
    assert set(dlv) == {"terminal_deliver", "out_of_bounds", "zone_progress", "grasp_progress"}
    assert dlv["terminal_deliver"] == pytest.approx(30.0)
