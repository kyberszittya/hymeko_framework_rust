"""Unit tests for the hierarchical formal task monitor (env-free, synthetic trajectories).

Exercises every submonitor's pass/fail path, the root aggregation, and the cross-cutting consistency monitors.
The real-env integration (7-policy evaluation) is the ``scratchpad/monitor_eval.py`` harness reported alongside.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.task_monitor import (
    STAGES,
    AntiExploitMonitor,
    ApproachMonitor,
    ContactMonitor,
    DeliveryMonitor,
    GeometryMonitor,
    MonitorContext,
    MonitorContract,
    PipelineSchemaError,
    PipelineSchemaLedger,
    ProgressMonitor,
    RewardConsistencyMonitor,
    StagnationMonitor,
    StagnationVerdict,
    TaskMonitor,
    TensorContract,
    TensorContractMonitor,
    TransitionField,
    TransitionSchema,
    default_submonitors,
    flat_dim,
)

C = MonitorContract(zone_half=0.05, success_steps=3, near_coin=0.06, body_eps=0.005, progress_eps=0.002)


def step(d, *, lc=False, rc=False, body=False, ltd=1.0, rtd=1.0, in_zone=False, xy=(0.0, 0.0)):
    return {"coin_xy": np.array(xy, float), "coin_vel": np.zeros(2), "dist_to_zone": float(d),
            "left_tip_contact": lc, "right_tip_contact": rc, "arm_body_contact": body,
            "left_tip_dist": ltd, "right_tip_dist": rtd, "in_zone": in_zone}


def good_delivery():
    """Fingertip-dominant delivery: approach → both-fingertip engage → push into zone → dwell k."""
    return [
        step(0.30, ltd=0.20, rtd=0.20), step(0.30, ltd=0.10, rtd=0.10), step(0.30, ltd=0.05, rtd=0.05),
        step(0.24, lc=True, rc=True, ltd=0.02, rtd=0.02), step(0.16, lc=True, rc=True, ltd=0.02, rtd=0.02),
        step(0.08, lc=True, rc=True, ltd=0.02, rtd=0.02),
        step(0.03, lc=True, rc=True, ltd=0.02, rtd=0.02, in_zone=True),
        step(0.02, lc=True, rc=True, ltd=0.02, rtd=0.02, in_zone=True),
        step(0.02, lc=True, rc=True, ltd=0.02, rtd=0.02, in_zone=True),
    ]


def body_shove():
    """Body-only progress drives the coin into the zone; fingertips never approach."""
    return [step(0.30, ltd=0.5, rtd=0.5)] + [
        step(d, body=True, ltd=0.5, rtd=0.5, in_zone=d <= 0.05)
        for d in (0.22, 0.14, 0.06, 0.03, 0.02, 0.02)
    ]


def no_delivery():
    """Coin never reaches the zone."""
    return [step(0.30, ltd=0.5, rtd=0.5) for _ in range(6)]


# --- MonitorContext -------------------------------------------------------------------------------------------
def test_context_progress_attribution():
    ctx = MonitorContext.build(good_delivery(), C)
    assert ctx.ft_prog > 0.25 and ctx.body_prog == 0.0
    assert ctx.two_finger_steps == 6 and ctx.first_two_finger == 4
    assert ctx.max_dwell == 3 and ctx.first_delivery == 9

    ctx_b = MonitorContext.build(body_shove(), C)
    assert ctx_b.body_prog > 0.25 and ctx_b.ft_prog == 0.0 and ctx_b.two_finger_steps == 0


def test_context_empty_is_rejected():
    v = TaskMonitor(C).evaluate([])
    assert not v.monitor_pass and v.violation_reason == "empty_trajectory"


# --- individual submonitors -----------------------------------------------------------------------------------
def test_geometry_pass_and_fail():
    assert GeometryMonitor().evaluate(MonitorContext.build(good_delivery(), C)).passed
    v = GeometryMonitor().evaluate(MonitorContext.build(no_delivery(), C))
    assert not v.passed and "coin_never_reached_zone" in v.violations


def test_approach_fail_when_tips_far():
    v = ApproachMonitor().evaluate(MonitorContext.build(body_shove(), C))
    assert not v.passed and "fingertips_never_approached" in v.violations
    assert ApproachMonitor().evaluate(MonitorContext.build(good_delivery(), C)).passed


def test_contact_requires_both_fingertips():
    v = ContactMonitor().evaluate(MonitorContext.build(body_shove(), C))
    assert not v.passed and "no_both_fingertip_engagement" in v.violations
    good = ContactMonitor().evaluate(MonitorContext.build(good_delivery(), C))
    assert good.passed and good.time_indices  # both-contact steps recorded


def test_progress_pass_and_fail():
    assert ProgressMonitor().evaluate(MonitorContext.build(good_delivery(), C)).passed
    v = ProgressMonitor().evaluate(MonitorContext.build(no_delivery(), C))
    assert not v.passed and "coin_not_closer_to_target" in v.violations


def test_delivery_dwell_rule():
    assert DeliveryMonitor().evaluate(MonitorContext.build(good_delivery(), C)).passed
    v = DeliveryMonitor().evaluate(MonitorContext.build(no_delivery(), C))
    assert not v.passed and "never_delivered" in v.violations


def test_anti_exploit_flags_body_driven():
    v = AntiExploitMonitor().evaluate(MonitorContext.build(body_shove(), C))
    assert not v.passed
    assert "body_driven_delivery" in v.violations
    assert "delivery_without_fingertip_engagement" in v.violations
    assert AntiExploitMonitor().evaluate(MonitorContext.build(good_delivery(), C)).passed


# --- root aggregation -----------------------------------------------------------------------------------------
def test_task_monitor_pass_on_good_delivery():
    v = TaskMonitor(C).evaluate(good_delivery())
    assert v.monitor_pass and v.monitor_score > 0.0 and v.violation_reason == "none"
    assert set(v.sub_verdicts) == {"geometry", "approach", "contact", "progress", "delivery", "anti_exploit"}


def test_task_monitor_fail_on_exploit():
    v = TaskMonitor(C).evaluate(body_shove())
    assert not v.monitor_pass
    assert v.violation_reason == "fingertips_never_approached"  # first gating failure
    assert v.delivery_score > 0.0  # coin did reach the zone — but by the wrong mechanism


def test_task_monitor_fail_on_no_delivery():
    v = TaskMonitor(C).evaluate(no_delivery())
    assert not v.monitor_pass and v.delivery_score < 0.0


# --- stagnation monitor (reward-independent no-progress detector) ---------------------------------------------
def _approach(slope=0.02, n=10, d0=0.30):
    """Monotone approach: distance-to-target decreases by ``slope`` each step (net progress every window)."""
    return [step(max(d0 - slope * i, 0.0)) for i in range(n)]


def _flat(d=0.20, n=10):
    """Distance never changes → zero net progress."""
    return [step(d) for _ in range(n)]


def _oscillating(n=8, lo=0.20, hi=0.30):
    """Distance oscillates lo/hi with zero NET improvement (a one-sided ``toward`` sum would look busy)."""
    return [step(hi if i % 2 else lo) for i in range(n)]


def test_stagnation_rejects_bad_params():
    with pytest.raises(ValueError):
        StagnationMonitor(window=0)
    with pytest.raises(ValueError):
        StagnationMonitor(eps=-1.0)
    with pytest.raises(ValueError):
        StagnationMonitor(min_steps=-1)


def test_stagnation_none_when_progress_improves():
    v = StagnationMonitor(window=3, eps=0.005).evaluate(MonitorContext.build(_approach(), C))
    assert isinstance(v, StagnationVerdict)
    assert not v.stagnated and v.passed and v.stagnation_duration == 0 and v.score > 0.0


def test_stagnation_when_flat():
    v = StagnationMonitor(window=3, eps=0.005).evaluate(MonitorContext.build(_flat(n=10), C))
    assert v.stagnated and not v.passed and v.score < 0.0
    assert v.stagnation_duration == 7   # steps [3, 10): every trailing 3-window is sub-eps
    assert v.violations and v.violations[0].startswith("no_progress_window")
    assert "stagnation_duration" in v.as_dict() and v.as_dict()["stagnated"] is True


def test_stagnation_not_flagged_before_enough_steps():
    # window longer than the trajectory → insufficient evidence, never flags (not a positive pass claim)
    short = StagnationMonitor(window=5, eps=0.005).evaluate(MonitorContext.build(_flat(n=4), C))
    assert not short.stagnated and short.passed and short.stagnation_duration == 0
    # min_steps pushes the earliest flag past the trajectory end → still no flag on a flat rollout
    delayed = StagnationMonitor(window=3, eps=0.005, min_steps=10).evaluate(MonitorContext.build(_flat(n=6), C))
    assert not delayed.stagnated and delayed.passed


def test_stagnation_on_noisy_non_improving():
    # oscillation: NET window progress ~0 → stagnating, even though one-sided per-step motion is large.
    # This is the case a `toward`-sum monitor would MISS; the net-window rule catches it.
    v = StagnationMonitor(window=2, eps=0.01).evaluate(MonitorContext.build(_oscillating(n=8), C))
    assert v.stagnated and v.stagnation_duration == 6   # steps [2, 8): every 2-window returns to its start


def test_stagnation_composes_without_changing_root_verdict():
    base = TaskMonitor(C).evaluate(good_delivery())
    composed = TaskMonitor(C, default_submonitors() + [StagnationMonitor()]).evaluate(good_delivery())
    # gating verdict + score are unchanged: stagnation is NOT a gating submonitor
    assert composed.monitor_pass == base.monitor_pass
    assert composed.monitor_score == base.monitor_score
    # but the stagnation verdict is now reported alongside and serialises its extra fields
    assert "stagnation" in composed.sub_verdicts and "stagnation" not in base.sub_verdicts
    sv = composed.sub_verdicts["stagnation"]
    assert isinstance(sv, StagnationVerdict) and "stagnation_duration" in sv.as_dict()
    assert not sv.stagnated   # default window (20) > the 9-step fixture → insufficient evidence


# --- cross-cutting consistency monitors -----------------------------------------------------------------------
def test_reward_alignment_detects_inversion():
    rc = RewardConsistencyMonitor()
    aligned = rc.check_reward_alignment(
        [{"policy": "a", "total_reward": -100, "monitor_score": 0.5},
         {"policy": "b", "total_reward": -200, "monitor_score": 0.1}])
    assert aligned.passed and aligned.score == pytest.approx(1.0)
    inverted = rc.check_reward_alignment(
        [{"policy": "a", "total_reward": -100, "monitor_score": 0.1},
         {"policy": "b", "total_reward": -200, "monitor_score": 0.5}])
    assert not inverted.passed and inverted.violations
    # direction: reward prefers the higher-reward 'a', monitor prefers 'b'
    assert inverted.violations[0] == "reward prefers a but monitor prefers b"


def test_critic_alignment_detects_bad_ranking():
    rc = RewardConsistencyMonitor()
    # critic Q ranks exploit above dagger, monitor prefers dagger → inversion (direction must be correct)
    v = rc.check_critic_alignment(q_by_action={"dagger": -5.65, "exploit": -4.99},
                                  score_by_action={"dagger": 0.6, "exploit": -0.3})
    assert not v.passed
    assert v.violations[0] == "Q ranks exploit above dagger but monitor prefers dagger"
    ok = rc.check_critic_alignment(q_by_action={"dagger": -4.0, "exploit": -6.0},
                                   score_by_action={"dagger": 0.6, "exploit": -0.3})
    assert ok.passed


def test_tensor_contract_field_order_and_env():
    tc = TensorContractMonitor(TensorContract(obs_dim=20, privileged_dim=5, node_feat_dim=8))
    assert TensorContractMonitor.field_hash(("a", "b")) != TensorContractMonitor.field_hash(("b", "a"))
    agree = tc.verify_stages({"rollout": ("a", "b", "c"), "replay": ("a", "b", "c"), "critic": ("a", "b", "c")})
    assert agree.passed
    drift = tc.verify_stages({"rollout": ("a", "b", "c"), "replay": ("a", "c", "b")})
    assert not drift.passed and drift.violations

    class StubSpace:
        shape = (20,)

    class StubEnv:
        observation_space = StubSpace()
        _FEAT = 8

        def privileged_state(self):
            return np.zeros(5, np.float32)

    assert tc.check_env(StubEnv()).passed
    bad = TensorContractMonitor(TensorContract(obs_dim=99, privileged_dim=5, node_feat_dim=8))
    assert not bad.check_env(StubEnv()).passed


# --- live-pipeline schema guard --------------------------------------------------------------------------------
_PRIV_SCHEMA = TransitionSchema(tuple(TransitionField(n, d) for n, d in [
    ("obs", 20), ("action", 4), ("reward", 1), ("next_obs", 20), ("done", 1), ("priv", 5), ("next_priv", 5)]))
_FULL = [("obs", 20), ("action", 4), ("reward", 1), ("next_obs", 20), ("done", 1), ("priv", 5), ("next_priv", 5)]
_CRIT = [("obs", 20), ("action", 4), ("priv", 5)]
_EVAL = [("obs", 20), ("action", 4)]


def _record_all(led):
    led.record("rollout", _FULL)
    led.record("replay_serialize", _FULL)
    led.record("replay_load", _FULL)
    led.record("critic_train", _CRIT)
    led.record("eval", _EVAL)


def test_flat_dim_batched_and_unbatched():
    assert flat_dim(np.zeros((3, 20)), batched=True) == 20
    assert flat_dim(np.zeros(20), batched=False) == 20
    assert flat_dim(np.zeros((8, 2, 5)), batched=True) == 10   # trailing dims multiplied
    assert flat_dim(np.zeros((7,)), batched=True) == 1         # a (B,) reward/done column → width 1


def test_transition_schema_from_env():
    class Space:
        def __init__(self, shape):
            self.shape = shape

    class Env:
        def __init__(self, shape, pd):
            self.observation_space = Space(shape)
            self.privileged_dim = pd

    s = TransitionSchema.from_env(Env((20,), 5), action_dim=4, priv_enabled=True)
    assert s.names == ("obs", "action", "reward", "next_obs", "done", "priv", "next_priv")
    assert s.fields[0].dim == 20 and s.fields[5].dim == 5
    s2 = TransitionSchema.from_env(Env((4, 5), 0), action_dim=4, priv_enabled=False)
    assert s2.names == ("obs", "action", "reward", "next_obs", "done")   # no priv fields
    assert s2.fields[0].dim == 20                                        # obs flattened 4*5


def test_pipeline_ledger_passes_and_seals():
    led = PipelineSchemaLedger(_PRIV_SCHEMA)
    _record_all(led)
    v = led.verify_or_abort()
    assert v.passed and led.sealed
    assert set(led.recorded) == set(STAGES)
    assert len(set(led.stage_hashes().values())) >= 1


def test_pipeline_ledger_aborts_on_priv_dim_mismatch():
    led = PipelineSchemaLedger(_PRIV_SCHEMA)
    with pytest.raises(PipelineSchemaError, match="priv dim"):
        led.record("critic_train", [("obs", 20), ("action", 4), ("priv", 7)])   # wrong z width


def test_pipeline_ledger_aborts_on_full_stage_field_order():
    led = PipelineSchemaLedger(_PRIV_SCHEMA)
    swapped = [("action", 4), ("obs", 20), ("reward", 1), ("next_obs", 20), ("done", 1), ("priv", 5), ("next_priv", 5)]
    with pytest.raises(PipelineSchemaError):
        led.record("replay_load", swapped)   # full stage must equal canonical exactly


def test_pipeline_ledger_aborts_on_consumer_order():
    led = PipelineSchemaLedger(_PRIV_SCHEMA)
    with pytest.raises(PipelineSchemaError, match="violates canonical order"):
        led.record("critic_train", [("action", 4), ("obs", 20), ("priv", 5)])   # obs must precede action


def test_pipeline_ledger_incomplete_coverage_aborts():
    led = PipelineSchemaLedger(_PRIV_SCHEMA)
    led.record("rollout", _FULL)
    led.record("replay_serialize", _FULL)
    led.record("replay_load", _FULL)
    led.record("critic_train", _CRIT)     # eval never exercised
    assert not led.verify().passed
    with pytest.raises(PipelineSchemaError, match="never exercised"):
        led.verify_or_abort()


def test_pipeline_ledger_rerecord_conflict_aborts():
    led = PipelineSchemaLedger(_PRIV_SCHEMA)
    led.record("critic_train", _CRIT)
    with pytest.raises(PipelineSchemaError, match="changed mid-run"):
        led.record("critic_train", [("obs", 20), ("action", 4)])   # same stage, different schema
