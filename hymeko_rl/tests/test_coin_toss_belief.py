"""Rotor-state VALIDITY suite for the new belief architecture (directive §1-§5,§7). Proves the SO(2) rotor spike
actually WRITES into the recurrent structural state (the earlier z-aligned state was a dead axis), gradients flow,
T4 differs FUNCTIONALLY from T3 under events, and the SO(2) model is honestly commutative (no holonomy claim)."""
from __future__ import annotations

import math

import torch

from hymeko_neuro.graph.embeddings.cayley_rotor import quat_rotate
from hymeko_rl.experiments.exp_coin_toss_belief import (JointBelief, RecurrentHSiKANBelief, _N, so2_rotor_spike)


def _planar(B, val=(1.0, 0.0, 0.0)):
    s = torch.zeros(B, _N, 3); s[..., 0], s[..., 1], s[..., 2] = val
    return s


# ── §1 state-change invariants ───────────────────────────────────────────────────────────────────────────────────
def test_identity_rotor_leaves_any_state_unchanged() -> None:
    S = torch.randn(4, _N, 3); q = so2_rotor_spike(torch.zeros(4, 1), torch.randn(4, 1))   # gate 0 ⇒ identity
    assert torch.allclose(quat_rotate(q[:, None].expand(-1, _N, -1), S), S, atol=1e-6)


def test_event_rotor_changes_nonaxis_state() -> None:
    S = _planar(3)                                              # x-direction, PERPENDICULAR to the z rotor axis
    q = so2_rotor_spike(torch.ones(3, 1), torch.full((3, 1), 0.8))
    Sr = quat_rotate(q[:, None].expand(-1, _N, -1), S)
    assert not torch.allclose(Sr, S, atol=1e-3)                # THE fix: a planar state actually rotates (not dead axis)
    assert torch.allclose(Sr[..., 2], torch.zeros(3, _N), atol=1e-5)             # stays in the x-y plane (z-rotor)
    eff = 2 * math.atan(0.8)                                    # Cayley param: Rodrigues 0.8 → rotation 2·atan(0.8)
    assert torch.allclose(Sr[..., 0], torch.cos(torch.tensor(eff)), atol=1e-4)   # x → cos(2·atan θ)
    assert torch.allclose(Sr[..., 1], torch.sin(torch.tensor(eff)), atol=1e-4)   # y → sin(2·atan θ)


def test_rotor_unit_and_norm_preserving() -> None:
    q = so2_rotor_spike(torch.ones(5, 1), torch.randn(5, 1))
    assert torch.allclose(q.norm(dim=-1), torch.ones(5), atol=1e-5)              # unit rotor
    S = torch.randn(5, _N, 3)
    Sr = quat_rotate(q[:, None].expand(-1, _N, -1), S)
    assert torch.allclose(Sr.norm(dim=-1), S.norm(dim=-1), atol=1e-4)           # sandwich preserves norm (pre-normalize)


def test_no_event_recurrence_preserves_state() -> None:
    m = RecurrentHSiKANBelief(2, 3); m.eval(); S = m.init_state(4)
    with torch.no_grad():
        S1, _ = m(S, torch.randn(4, _N, 2), torch.randn(4, 3), torch.zeros(4, 1), torch.randn(4, 1))
    assert torch.allclose(S1, S, atol=1e-5)                    # no event + zero-init readout ⇒ unchanged


def test_event_recurrence_changes_state() -> None:
    m = RecurrentHSiKANBelief(2, 3); m.eval(); S = m.init_state(4)   # canonical state is now PLANAR (non-axis)
    with torch.no_grad():
        S1, _ = m(S, torch.randn(4, _N, 2), torch.randn(4, 3), torch.ones(4, 1), torch.full((4, 1), 0.6))
    assert not torch.allclose(S1, S, atol=1e-3)                # event genuinely changes the structural state


# ── §2 gradient flow (the dead axis gave zero angle-gradient; the planar fix must not) ───────────────────────────
def test_rotor_gradient_flow() -> None:
    m = RecurrentHSiKANBelief(2, 3)
    S = m.init_state(4).clone().requires_grad_(True)
    angle = torch.full((4, 1), 0.5, requires_grad=True); gate = torch.full((4, 1), 1.0, requires_grad=True)
    S1, z = m(S, torch.randn(4, _N, 2), torch.randn(4, 3), gate, angle)
    tgt = torch.randn_like(S1)                                  # DIRECTION-sensitive loss (||S1||≡1, so a norm loss is
    (((S1 - tgt) ** 2).sum() + (z ** 2).sum()).backward()      # constant and gives 0 readout-grad — use direction)
    l1g = next(p.grad.norm() for p in m.l1.parameters() if p.grad is not None)
    norms = {"angle": float(angle.grad.norm()), "gate": float(gate.grad.norm()), "S_prev": float(S.grad.norm()),
             "readout": float(m.readout.weight.grad.norm()), "hsikan_l1": float(l1g)}
    for k, v in norms.items():
        assert v > 1e-7 and math.isfinite(v), f"dead/zero gradient for {k}: {v}"
    print("grad norms:", {k: round(v, 5) for k, v in norms.items()})


# ── §4 SO(2) semantics: COMMUTATIVE, no holonomy claim ───────────────────────────────────────────────────────────
def test_so2_rotors_commute_no_holonomy() -> None:
    S = _planar(2)
    R1 = so2_rotor_spike(torch.ones(2, 1), torch.full((2, 1), 0.4))
    R2 = so2_rotor_spike(torch.ones(2, 1), torch.full((2, 1), -0.9))
    r1r2 = quat_rotate(R1[:, None].expand(-1, _N, -1), quat_rotate(R2[:, None].expand(-1, _N, -1), S))
    r2r1 = quat_rotate(R2[:, None].expand(-1, _N, -1), quat_rotate(R1[:, None].expand(-1, _N, -1), S))
    assert torch.allclose(r1r2, r2r1, atol=1e-5)               # z-rotors COMMUTE ⇒ SO(2), order-insensitive, NO holonomy


# ── §5 functional T3 vs T4 separation (matched weights) ──────────────────────────────────────────────────────────
def _matched_t3_t4():
    t4 = JointBelief(6, 4, 2, 1, kind="T4"); t3 = JointBelief(6, 4, 2, 1, kind="T3")
    t3.load_state_dict(t4.state_dict())                        # copy weights (use_rotor is config, not a param)
    return t3, t4


def test_t3_t4_identical_when_no_event() -> None:
    t3, t4 = _matched_t3_t4()
    args = (torch.randn(4, 6), torch.randn(4, 4), torch.randn(4, _N, 2), torch.zeros(4, 1), torch.randn(4, 1))
    with torch.no_grad():
        o3, _ = t3.step(t3.init_state(4), *args); o4, _ = t4.step(t4.init_state(4), *args)
    assert torch.allclose(o3, o4, atol=1e-5)                   # no event ⇒ T4 rotor identity ⇒ == T3


def test_t3_t4_differ_under_real_event() -> None:
    t3, t4 = _matched_t3_t4()
    args = (torch.randn(4, 6), torch.randn(4, 4), torch.randn(4, _N, 2), torch.ones(4, 1), torch.full((4, 1), 0.7))
    with torch.no_grad():
        o3, n3 = t3.step(t3.init_state(4), *args); o4, n4 = t4.step(t4.init_state(4), *args)
    assert not torch.allclose(n3["S"], n4["S"], atol=1e-4)     # T4 structural state moved by rotor, T3 did not
    assert not torch.allclose(o3, o4, atol=1e-5)               # functional (not merely configurational) separation


def _roll_struct(angles):
    """Roll the recurrent structural state under an event-angle sequence, zero-init readout + zero inputs ⇒ the state
    is PURE cumulative SO(2) rotor transport (isolates the rotor's order/cumsum semantics)."""
    m = RecurrentHSiKANBelief(2, 3); m.eval(); st = m.init_state(1)
    with torch.no_grad():
        for a in angles:
            st, _ = m(st, torch.zeros(1, _N, 2), torch.zeros(1, 3), torch.ones(1, 1), torch.full((1, 1), a))
    return st


def test_same_cumsum_permutation_is_identical_so2_collapse() -> None:
    a = _roll_struct([0.5, -0.3, 0.8]); b = _roll_struct([0.8, 0.5, -0.3])   # SAME cumulative angle 1.0, permuted
    assert torch.allclose(a, b, atol=1e-4)                     # SO(2): order collapses to cumsum (documented, no holonomy)


def test_different_cumsum_is_distinguishable() -> None:
    a = _roll_struct([0.5, -0.3, 0.8]); b = _roll_struct([0.5, 0.3, 0.8])    # cumsum 1.0 vs 1.6
    assert not torch.allclose(a, b, atol=1e-3)                 # different cumulative angle ⇒ distinguishable T4 states


# ── §7 rotor controls (synthetic) ────────────────────────────────────────────────────────────────────────────────
def test_inverse_spike_consistency_and_long_seq_stability() -> None:
    S = _planar(3)
    Rf = so2_rotor_spike(torch.ones(3, 1), torch.full((3, 1), 0.6))
    Ri = so2_rotor_spike(torch.ones(3, 1), torch.full((3, 1), -0.6))   # inverse spike
    S2 = quat_rotate(Ri[:, None].expand(-1, _N, -1), quat_rotate(Rf[:, None].expand(-1, _N, -1), S))
    assert torch.allclose(S2, S, atol=1e-4)                    # forward then inverse returns to start
    for _ in range(200):                                       # long sequence: norm stays bounded (stability)
        S = quat_rotate(so2_rotor_spike(torch.ones(3, 1), torch.full((3, 1), 0.3))[:, None].expand(-1, _N, -1), S)
    assert torch.allclose(S.norm(dim=-1), torch.ones(3, _N), atol=1e-3)
