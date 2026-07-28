"""R9 explicit APPROACH→KINETIC handoff-reset contracts (make the hidden mode boundary a first-class online event).

The handoff audit showed the frozen KINETIC entry = the natural handoff advanced by ONE frozen servo KINETIC step — a step that
existed only in the OFFLINE entry-creation (`freeze_kinetic_entry` captures post-step), never in the uninterrupted deploy. That is
not illegitimate as a hybrid controller; the defect was that it was implicit and offline-only. This module makes the reset map
explicit, so it is part of the natural end-to-end controller (no snapshot injection):

  H0  DIRECT_HANDOFF         APPROACH → guard crossing → the KINETIC policy executes the FIRST action immediately
                            (the plain `KineticTemporalResidualController`/`AuthorityUnlockController`; the audit's E1).
  H1  EXPLICIT_HANDOFF_RESET APPROACH → guard crossing → ONE explicit frozen transition-servo step (mode HANDOFF_RESET, its own
                            trace, clear pre/post semantics; the clone does NOT act, its GRU hidden stays fresh) → the KINETIC
                            policy. Reproduces the frozen-entry interface ONLINE, end-to-end.

The two contracts must NOT be mixed within one run. All `8a0c1c7b` modules are imported unchanged; the mixin replicates the frozen
`KineticCloneController.dtau_for_step` (the phase machine must not be double-invoked) and only inserts the HANDOFF_RESET step.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC
from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import AuthorityUnlockController
from hymeko_rl.coin_delivery.theta_option.kinetic_contract import kinetic_observe
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import KineticTemporalResidualController


class ExplicitHandoffResetMixin:
    """Insert ONE explicit frozen transition-servo step (mode HANDOFF_RESET) at the first KINETIC step, before the policy acts.
    Replicates the frozen `KineticCloneController.dtau_for_step` for the APPROACH / KINETIC-policy / brake-release branches (the
    phase machine `_base_targets` may only be called once per step); the only addition is the one-shot reset step. # Postconditions:
    exactly one HANDOFF_RESET precedes the first KINETIC_CLONE; during it the clone does NOT act (hidden stays fresh); |Δτ| ≤ slew;
    the KINETIC policy (clone+R2 or clone+R2+expansion, via `_transport_action`) is otherwise unchanged."""

    def reset(self) -> None:
        super().reset()                                                       # type: ignore[misc]
        self._handoff_reset_done = False
        self.handoff_trace: list[dict[str, Any]] = []

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        bt = self._base_targets(rl, t)                                        # frozen phase machine — ONCE
        if self.phase == KINETIC and not bt["in_release"]:
            if not self._handoff_reset_done:                                  # HANDOFF_RESET: one frozen servo step, no clone act
                self._handoff_reset_done = True
                dtau = self._servo(rl, bt, bt["base_qref"], bt["base_sqz"], self.p.k_v)
                self.handoff_trace.append({"t": int(t), "kind": "HANDOFF_RESET", "dtz_mm": round(bt["dtz"] * 1000, 2),
                                           "dtau": [round(float(x), 5) for x in dtau]})
                self._log_clone(rl, t, "HANDOFF_RESET", bt, None)
                return dtau
            obs, hframe = kinetic_observe(rl, self._obs_hist)                 # the KINETIC policy acts (fresh hidden here)
            self._obs_hist.append(hframe)
            a = self._transport_action(rl, obs)
            dtau = np.clip(a * self.slew, -self.slew, self.slew)
            self._log_clone(rl, t, "KINETIC_CLONE", bt, a)
            return dtau
        dtau = self._servo(rl, bt, bt["base_qref"], bt["base_sqz"], self.p.k_v)   # frozen APPROACH / brake / release
        self._log_clone(rl, t, "FROZEN", bt, None)
        return dtau


class HandoffResetTemporalController(ExplicitHandoffResetMixin, KineticTemporalResidualController):
    """H1 clone + R2 with an explicit online HANDOFF_RESET (the online equivalent of the frozen-entry interface)."""


class HandoffResetUnlockController(ExplicitHandoffResetMixin, AuthorityUnlockController):
    """H1 FULL (clone + R2 + β·expansion) with an explicit online HANDOFF_RESET."""
