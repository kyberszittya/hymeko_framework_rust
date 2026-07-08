"""Phase-gated hierarchical policy — the learned MLP handles approach, a scripted PhasePushController(θ) handles
the contact push.

`bounded_option_parameter_rl_v0` composes the deployable learned base (E_valselect_v2, a DeterministicMLPMultiActor)
with a bounded 5-parameter scripted push controller: in APPROACH / non-contact the frozen MLP acts (it is the better
approacher + engager); the instant a fingertip touches the coin, the PhasePushController(θ) takes over for the
sustained two-finger push (what it is designed for). This is a per-PHASE handoff — exactly one controller's action
is executed each step — NOT an additive per-step residual. Only θ (the 5 PhasePushController params) is learnable;
the MLP is frozen. The gate reads the privileged contact state z = env.privileged_state() = [left_contact,
right_contact, …], so the phase is decided from the current state before stepping.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class PhaseGatedPolicy:
    """Frozen MLP (approach) + PhasePushController(θ) (contact/push), gated on fingertip contact.

    # Preconditions: ``mlp_actor`` is a frozen ``action_mean(obs)->action`` policy; ``ppc`` exposes
      ``reset()``/``action(env)`` and advances its own FSM from env state. # Invariants: exactly one controller's
      action is returned per step (handoff, not residual); the PPC FSM is advanced every step so it is in the right
      phase at handoff; the MLP is never trained here."""

    def __init__(self, mlp_actor: Any, ppc: Any, *, contact_thresh: float = 0.5) -> None:
        self.mlp = mlp_actor
        self.ppc = ppc
        self.contact_thresh = contact_thresh
        self.last_phase = "APPROACH"

    def reset(self) -> None:
        if hasattr(self.ppc, "reset"):
            self.ppc.reset()
        self.last_phase = "APPROACH"

    def _mlp_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = self.mlp.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
        return a.astype(np.float32)

    def action(self, env: Any) -> np.ndarray:
        z = env.privileged_state()
        any_contact = bool(z[0] > self.contact_thresh or z[1] > self.contact_thresh)
        a_ppc = np.asarray(self.ppc.action(env), dtype=np.float32)   # advance the FSM every step
        if any_contact:
            self.last_phase = "PUSH"
            return a_ppc
        self.last_phase = "APPROACH"
        return self._mlp_action(np.asarray(env.node_features(), dtype=np.float32))


def phase_gated_action_fn(policy: PhaseGatedPolicy):
    """Adapt a (reset) PhaseGatedPolicy to the ``action_fn(env, _obs)`` contract of record_trajectory / eval."""
    def fn(env: Any, _obs: np.ndarray) -> np.ndarray:
        return policy.action(env)
    return fn
