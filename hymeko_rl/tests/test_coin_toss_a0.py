"""A0 gate integration invariants (directive §4/§5): channels shape, gate init near-zero (identity), off/on/lstm scale
semantics, recurrence-destruction control resets state. Pure/small — no MuJoCo (rollout identity is smoke-tested)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.experiments.exp_coin_toss_a0 import GateController, _channels, _gate_belief, _PHYS


class _M:
    left_contact, right_contact, in_zone, legality = True, False, False, None
    disk_pos = np.array([0.1, 0.2]); disk_vel = np.array([0.01, -0.02])


class _Env:
    _planar_metrics = _M()


def test_channels_shape() -> None:
    assert _channels(_M(), None).shape == (_PHYS,)


def test_alpha0_is_exact_identity() -> None:
    # §1: g = alpha·sigmoid(logit); alpha=0 ⇒ g≡0 EXACTLY regardless of the (arbitrary) LSTM logit ⇒ exact v16f identity.
    g = GateController("lstm", belief=_gate_belief(), horizon=32, alpha=0.0); g.reset()
    for _ in range(5):
        g.observe(_Env())
    assert g.scale(1, None, None) == 0.0                        # exact 0, not near-zero (identity guarantee)
    assert g._g != 0.0                                          # the LSTM logit is live (state updates) — just gated to 0


def test_off_on_scale() -> None:
    assert GateController("off").scale(1, None, None) == 0.0    # A00 pure v16f
    assert GateController("on").scale(1, None, None) == 1.0     # A01 always-on M0


def test_alpha_scales_deployed_gate() -> None:
    g = GateController("lstm", belief=_gate_belief(), horizon=32, alpha=0.5); g.reset()
    for _ in range(5):
        g.observe(_Env())
    dep = g.scale(1, None, None)
    assert 0.0 <= dep <= 0.5 and abs(dep - 0.5 * g._g) < 1e-6   # deployed g = alpha·sigmoid ∈ [0, alpha]


def test_recurrence_destruction_is_memoryless() -> None:
    # A05 destruction: hidden state re-inited every observe ⇒ the gate is memoryless (same input ⇒ same output,
    # independent of history). Contrast with a normal LSTM gate whose state accumulates.
    g = GateController("lstm", belief=_gate_belief(), horizon=0, destroy="reset_every"); g.reset()
    g.observe(_Env()); g1 = g._g
    for _ in range(4):
        g.observe(_Env())
    assert abs(g._g - g1) < 1e-6                                # memoryless: identical input ⇒ identical gate
