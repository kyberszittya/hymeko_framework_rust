"""Tests for COIN-DELIVERY-DERIVED-SCENARIOS-1 — the reusable scenario framework around the frozen delivery actor.

Covers contact-continuity certification (C0/C1), the kinematic embodiments (K0 byte-unchanged, K1 distal DOF),
scenario config serialization, the registry, and the demo/batch runners. The delivery monitor is the FROZEN
strict monitor throughout — these tests never weaken it. Reasonably short: env builds are ~1 s, rollouts small.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from hymeko_rl.coin_delivery.actors.base import (
    SemanticCommand,
    build_actor_registry,
    orientation_scramble,
)
from hymeko_rl.coin_delivery.monitoring.delivery import DeliveryMonitor
from hymeko_rl.coin_delivery.runners.demo import DemoRunner
from hymeko_rl.coin_delivery.runners.experiment import ExperimentRunner
from hymeko_rl.coin_delivery.scenarios.config import ScenarioConfig
from hymeko_rl.coin_delivery.scenarios.contact_policy import (
    C0Continuous,
    C1Intermittent,
    ContactSample,
)
from hymeko_rl.coin_delivery.scenarios.kinematic_variant import (
    K0Canonical,
    K1DistalOrientation,
    distal_dof_probe,
)
from hymeko_rl.coin_delivery.scenarios.registry import ScenarioRegistry
from hymeko_rl.experiments.pedc_selection import _env
from hymeko_rl.train.coin_delivery_actor import (
    Attribution,
    DeliveryActor,
    Mechanism,
    _valid_delivery,
    actor_action,
    classify_mechanism,
    rollout_delivery,
)

_ALL_SCENARIOS = ("c0k0", "c1k0", "c0k1", "c1k1")


def _sample(t: int, left: bool, right: bool, body: bool = False, prog: float = 0.0) -> ContactSample:
    return ContactSample(t=t, left=left, right=right, body=body, progress=prog)


# ── 1. C0 rejects contact loss beyond tolerance ─────────────────────────────────────────────────────────────────
def test_c0_rejects_contact_loss_beyond_grace() -> None:
    policy = C0Continuous(grace_steps=2)
    established = [_sample(0, True, True), _sample(1, True, True)]
    lost_long = established + [_sample(i, False, False) for i in range(2, 6)]   # 4-step loss > grace 2
    cert = policy.certify(lost_long)
    assert cert.certified is False and cert.max_loss_run == 4
    lost_short = established + [_sample(2, False, False), _sample(3, False, False), _sample(4, True, True)]
    assert policy.certify(lost_short).certified is True                        # 2-step loss within grace
    assert policy.certify([_sample(0, True, False)]).certified is False        # never both-established


# ── 2. C1 accepts impulse → free → recontact ────────────────────────────────────────────────────────────────────
def test_c1_accepts_impulse_free_recontact() -> None:
    trace = [_sample(0, True, True, prog=0.02), _sample(1, True, True, prog=0.01),   # impulse
             _sample(2, False, False, prog=0.01), _sample(3, False, False),          # free
             _sample(4, False, True), _sample(5, True, True, prog=0.01)]             # recontact
    cert = C1Intermittent().certify(trace)
    assert cert.certified is True and cert.n_contact_phases == 2
    assert C1Intermittent().certify([_sample(i, False, False) for i in range(5)]).certified is False  # no impulse


# ── 3. C1 still rejects a body-shove delivery (monitor's rejection, not the contact policy) ──────────────────────
def test_c1_scenario_still_rejects_body_shove() -> None:
    # contact continuity is fine under C1 ...
    trace = [_sample(0, True, True, prog=0.02), _sample(1, False, False, prog=0.01), _sample(2, True, True)]
    assert C1Intermittent().certify(trace).certified is True
    # ... but the FROZEN monitor rejects a body-shove even with a good dwell/settle log
    att = Attribution(0.30, 0.10, 0.50, 0.0, 0.10, 0.10)                        # alpha_body 0.5 > 0.2 threshold
    mech = classify_mechanism(DeliveryActor.A0_SYM_PUSH, att, made_progress=True,
                              initial_success=False, both_frac=0.5)
    assert mech is Mechanism.BODY_SHOVE
    assert _valid_delivery(_trace(), att, mech) is False


def _trace(initial=False, loose=True, dwell=8, vel=0.05):
    """A minimal public-trace stand-in exposing exactly the fields the strict monitor reads (post rollout-unification)."""
    return types.SimpleNamespace(initial_success=initial, loose=loose, best_dwell=dwell, settle_vel=vel)


# ── 4. K0 action mapping unchanged (byte-identical env + exact adapter reproduction) ─────────────────────────────
def test_k0_action_mapping_unchanged() -> None:
    k0env = K0Canonical().build_env()
    assert k0env.action_space.shape == (6,)
    r_k0 = rollout_delivery(k0env, 3, DeliveryActor.A0_SYM_PUSH, max_steps=40)
    r_ref = rollout_delivery(_env(), 3, DeliveryActor.A0_SYM_PUSH, max_steps=40)
    assert r_k0.as_dict() == r_ref.as_dict()                                   # byte-unchanged from today
    env = _env()
    env.reset(seed=1)
    inner = env._env
    a0 = build_actor_registry()["symmetric_push"]
    for t in range(3):
        mapped = K0Canonical().map_command(a0.command(inner, t)).action
        assert np.allclose(mapped, actor_action(inner, t, DeliveryActor.A0_SYM_PUSH).astype(np.float32))


# ── 5. K1 exposes exactly the intended distal-orientation DOF ────────────────────────────────────────────────────
def test_k1_exposes_one_distal_dof_per_fingertip() -> None:
    k1 = K1DistalOrientation()
    assert k1.n_extra_dof == 1 and k1.n_fingertips == 2                        # 1 distal DOF per fingertip
    assert k1.action_dim == 8 and K0Canonical().action_dim == 6
    meta = k1.metadata()
    assert meta.n_extra_dof_per_fingertip == 1 and meta.action_dim == 8


# ── 5b. K0 explicitly rejects pad-orientation commands (no silent reinterpretation) ─────────────────────────────
def test_k0_explicitly_rejects_pad_orientation() -> None:
    cmd = SemanticCommand(push_direction=(0.1, 0.2), aperture=-0.2, contact_preload=0.4,
                          left_pad_orientation=0.5, right_pad_orientation=-0.3)
    m0 = K0Canonical().map_command(cmd)
    assert m0.rejected == ("left_pad_orientation", "right_pad_orientation") and len(m0.action) == 6
    m1 = K1DistalOrientation().map_command(cmd)
    assert m1.rejected == () and len(m1.action) == 8 and np.allclose(m1.action[6:], [0.5, -0.3])


# ── 6. K1 orientation command affects the pad normal (physical DOF built) ────────────────────────────────────────
def test_k1_physical_distal_dof_responds() -> None:
    probe = distal_dof_probe()
    assert probe.built and probe.steps_cleanly, probe.detail
    assert probe.n_extra_actuators == 2                                        # one pad actuator per fingertip
    assert probe.responds and probe.pad_response_rad > 0.1, probe.detail       # pad normal rotates on command


# ── 7. ScenarioConfig serializes + reconstructs identically ─────────────────────────────────────────────────────
def test_scenario_config_roundtrip() -> None:
    for sid in _ALL_SCENARIOS:
        cfg = ScenarioRegistry.get(sid)
        assert ScenarioConfig.from_dict(cfg.to_dict()) == cfg


# ── 8. The supported (K0) factorial scenarios run through the common runner (2-seed smoke) ──────────────────────
def test_factorial_all_scenarios_run() -> None:
    k0_scenarios = ["c0k0", "c1k0"]                                            # K1 is unsupported (build_env raises)
    report = ExperimentRunner().run(k0_scenarios, ["symmetric_push"], [0, 1], scramble=True, max_steps=25)
    assert len(report.per_scenario) == 2 and len(report.cells) == 2
    for cell in report.cells:
        assert cell.n_seeds == 2 and 0.0 <= cell.delivery_rate <= 1.0
        assert cell.correct_minus_scramble is not None                        # correct-vs-scramble computed


def test_k1_build_env_raises_never_silently_k0() -> None:
    """Required regression (defect 4): K1's build_env must FAIL LOUDLY, never fabricate the K0 env."""
    with pytest.raises(NotImplementedError, match="never silently construct K0"):
        K1DistalOrientation().build_env()


# ── 9. The SAME frozen delivery monitor is used across all four scenarios ────────────────────────────────────────
def test_single_frozen_monitor_across_scenarios() -> None:
    assert DeliveryMonitor.rollout_fn is rollout_delivery
    runner = ExperimentRunner()
    assert runner.monitor.rollout_fn is rollout_delivery
    assert DemoRunner().monitor.rollout_fn is rollout_delivery
    # every SUPPORTED (K0) scenario resolves to a config the ONE monitor can evaluate (no per-scenario monitor variant)
    for sid in ("c0k0", "c1k0"):
        assert ScenarioRegistry.get(sid).kinematic_variant.build_env().action_space.shape == (6,)


# ── 10. Demo and batch runners agree on the monitor outcome for the same (scenario, actor, seed) ────────────────
def test_demo_and_batch_consistent() -> None:
    demo = DemoRunner().run("c1k0", "v_plow", 2, max_steps=40)
    monitor = DeliveryMonitor()
    env = ScenarioRegistry.get("c1k0").kinematic_variant.build_env()
    batch = monitor.evaluate(env, 2, DeliveryActor.A1_VPLOW, max_steps=40)
    assert demo.strict_delivery == batch.strict_delivery and demo.mechanism == batch.mechanism


# ── 11. Orientation scramble is capacity-matched ────────────────────────────────────────────────────────────────
def test_orientation_scramble_capacity_matched() -> None:
    rng = np.random.default_rng(0)
    correct6 = np.arange(6, dtype=np.float32)
    correct8 = np.arange(8, dtype=np.float32)
    assert len(orientation_scramble(correct6, rng)) == 6                       # K0: no tail → same length
    scr8 = orientation_scramble(correct8, rng)
    assert len(scr8) == 8 and np.allclose(scr8[:6], correct8[:6])              # K1: head preserved, tail scrambled


# ── 12. Initial-state success is rejected in every scenario (the shared monitor's guard) ────────────────────────
def test_initial_state_success_rejected_every_scenario() -> None:
    good_att = Attribution(0.5, 0.4, 0.0, 0.0, 0.1, 0.2)                       # fingertip 0.9, clean
    for sid in _ALL_SCENARIOS:                                                 # same monitor → holds everywhere
        _ = ScenarioRegistry.get(sid)
        assert _valid_delivery(_trace(initial=False), good_att, Mechanism.SYM_PUSH) is True
        assert _valid_delivery(_trace(initial=True), good_att, Mechanism.SYM_PUSH) is False


# ── extra: alpha_free decomposition is a labelled approximation summing to ~1 under free progress ────────────────
def test_alpha_free_decomposition() -> None:
    from hymeko_rl.coin_delivery.monitoring.attribution import decompose_alpha_free
    trace = [_sample(0, True, True), _sample(1, False, False, prog=0.03),      # free after fingertip
             _sample(2, False, False, body=True), _sample(3, False, False, prog=0.01)]  # free after body
    dec = decompose_alpha_free(trace)
    assert dec.approximate is True
    assert abs(dec.after_fingertip + dec.after_body + dec.unexplained - 1.0) < 1e-6
    assert dec.after_fingertip > 0.0 and dec.after_body > 0.0
