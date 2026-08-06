"""Contract tests for the deployable phase gate (PHASE_GATED_LEARNED_RESIDUAL_TD3, §3/§4).

Prove the hysteresis FSM behaves exactly as the runtime contract requires: transient contact never arms, stable
contact arms after ``arm_after``, single-step dropouts do not disarm, ``disarm_after`` consecutive losses disarm,
re-acquisition re-arms, terminal is absorbing, reset clears state, and the gate never arms before contact.
"""
from __future__ import annotations

import pytest

from hymeko_rl.coin_delivery.coin_phase_gate import GateState, PhaseGate, PhaseGateConfig


def _run(gate: PhaseGate, contacts, terminated=None):
    """Feed a contact sequence; return the list of per-step multipliers."""
    terminated = terminated or [False] * len(contacts)
    return [gate.update(c, t) for c, t in zip(contacts, terminated)]


def test_never_arms_without_contact():
    g = PhaseGate()
    assert _run(g, [False] * 50) == [0.0] * 50
    assert g.state is GateState.EARLY_CONTROL


def test_transient_contact_never_arms():
    # arm_after-1 consecutive contact steps must NOT arm (a transient touch)
    g = PhaseGate(PhaseGateConfig(arm_after=3))
    out = _run(g, [True, True, False, True, True, False])   # never 3 in a row
    assert out == [0.0] * 6
    assert g.state is GateState.EARLY_CONTROL
    assert g.arm_count == 0


def test_stable_contact_arms_after_arm_after():
    g = PhaseGate(PhaseGateConfig(arm_after=3))
    out = _run(g, [True, True, True, True])
    assert out == [0.0, 0.0, 1.0, 1.0]          # arms on the 3rd consecutive contact step
    assert g.state is GateState.LATE_CONTROL_ARMED
    assert g.arm_count == 1


def test_single_step_dropout_does_not_disarm():
    g = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    out = _run(g, [True, True, True, False, True, True])   # 1-step loss then contact back
    assert out[2] == 1.0 and out[3] == 1.0 and out[4] == 1.0
    assert g.state is GateState.LATE_CONTROL_ARMED
    assert g.disarm_count == 0


def test_disarm_after_consecutive_loss():
    g = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    out = _run(g, [True, True, True, False, False, False])
    assert out[2] == 1.0                        # armed
    assert out[4] == 0.0                        # disarmed on 2nd consecutive loss
    assert g.state is GateState.REACQUIRE
    assert g.disarm_count == 1


def test_reacquire_rearms():
    g = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    _run(g, [True, True, True, False, False])   # -> REACQUIRE
    assert g.state is GateState.REACQUIRE
    out = _run(g, [True, True, True])           # fresh 3 consecutive -> re-arm
    assert out[-1] == 1.0
    assert g.state is GateState.LATE_CONTROL_ARMED
    assert g.arm_count == 2


def test_terminal_is_absorbing():
    g = PhaseGate()
    _run(g, [True, True, True])                 # armed
    assert g.update(True, terminated=True) == 0.0
    assert g.state is GateState.TERMINAL
    assert g.update(True, terminated=False) == 0.0   # stays terminal, residual off
    assert g.state is GateState.TERMINAL


def test_chatter_suppressed():
    # alternating touch/no-touch: streak never reaches arm_after -> never arms (no chatter)
    g = PhaseGate(PhaseGateConfig(arm_after=3))
    out = _run(g, [True, False] * 20)
    assert set(out) == {0.0}
    assert g.arm_count == 0


def test_reset_clears_state():
    g = PhaseGate()
    _run(g, [True, True, True])
    assert g.state is GateState.LATE_CONTROL_ARMED
    g.reset()
    assert g.state is GateState.EARLY_CONTROL
    assert g._contact_streak == 0 and g._loss_streak == 0 and g.arm_count == 0


def test_gate_property_matches_update():
    g = PhaseGate()
    _run(g, [True, True, True])
    assert g.gate == 1.0
    _run(g, [False, False])
    assert g.gate == 0.0


def test_contract_sha_deterministic_and_config_sensitive():
    a = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2)).contract_sha256()
    b = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2)).contract_sha256()
    c = PhaseGate(PhaseGateConfig(arm_after=4, disarm_after=2)).contract_sha256()
    assert a == b and a != c
    assert len(a) == 64


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PhaseGateConfig(arm_after=0)
    with pytest.raises(ValueError):
        PhaseGateConfig(disarm_after=0)
