"""Deployable phase gate for PHASE_GATED_LEARNED_RESIDUAL_TD3 (2026-07-23).

A small deterministic finite-state machine with explicit hysteresis that decides *when* a learned late-phase
residual is allowed to act. The gate **generates no actions**: it only returns a scalar multiplier ``g_t ∈ {0, 1}``
that multiplies the residual. The early approach/grasp policy (frozen ``pi_0``) is therefore structurally protected —
whenever ``g_t == 0`` the composite action equals ``pi_0`` exactly regardless of the residual network.

Deployability contract (why this is not privileged simulator state)
------------------------------------------------------------------
The only runtime signal the gate consumes is **robot-attributed fingertip contact** — ``left_contact or
right_contact`` — which is the same predicate the canonical delivery certificate, :class:`CoinRL4Dof` (its
``_touched``), and ``eval_bc_delivery`` already read from ``inner._planar_metrics``. Physically it is a gripper
tactile sensor: available on a real robot. The gate uses **no** reset seed, trajectory id, future information,
planner state, ``disk_to_zone`` or any target-relative pose. "Stable grasp" versus "transient contact" is
distinguished purely by a *consecutive-step* counter (transient < ``arm_after`` steps; stable ≥ ``arm_after``),
which needs no privileged information.

States
------
- ``EARLY_CONTROL``      approach / contact acquisition / initial grasp — residual OFF (``g=0``).
- ``LATE_CONTROL_ARMED`` stable robot-attributed contact held ``arm_after`` consecutive steps — residual ON (``g=1``).
- ``REACQUIRE``          armed, then contact lost ``disarm_after`` consecutive steps — residual OFF while the frozen
                         base recovers; re-arms on ``arm_after`` fresh consecutive contact steps.
- ``TERMINAL``           strict K=6 success or episode end — residual OFF, absorbing.

Reset clears all state (:meth:`reset`); the caller must call it on every environment reset.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum


class GateState(str, Enum):
    EARLY_CONTROL = "EARLY_CONTROL"
    LATE_CONTROL_ARMED = "LATE_CONTROL_ARMED"
    REACQUIRE = "REACQUIRE"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True)
class PhaseGateConfig:
    """Preregistered gate counters (§3). Frozen — a change is a new gate with a new SHA.

    # Invariants: ``arm_after >= 1``, ``disarm_after >= 1``. Contact predicate = robot-attributed (left OR right).
    """
    arm_after: int = 3            # consecutive robot-attributed-contact steps to ARM late control
    disarm_after: int = 2         # consecutive complete-contact-loss steps to DISARM (enter REACQUIRE)
    contact_predicate: str = "robot_attributed_left_or_right"

    def __post_init__(self) -> None:
        if self.arm_after < 1 or self.disarm_after < 1:
            raise ValueError("arm_after and disarm_after must be >= 1")


class PhaseGate:
    """Deterministic hysteresis FSM. :meth:`update` advances one step and returns the residual multiplier ``g_t``.

    # Preconditions: ``contact`` is the deployable robot-attributed-contact boolean for the step just executed;
      ``terminated`` is the episode's strict-K6/end flag. # Postconditions: returns 1.0 iff the machine is in
      ``LATE_CONTROL_ARMED`` after this step, else 0.0; the machine is a pure function of the (contact, terminated)
      stream since the last :meth:`reset`. # Invariants: the gate never produces or modifies an action.
    """

    def __init__(self, config: PhaseGateConfig | None = None) -> None:
        self.cfg = config or PhaseGateConfig()
        self.reset()

    def reset(self) -> None:
        self.state = GateState.EARLY_CONTROL
        self._contact_streak = 0
        self._loss_streak = 0
        # diagnostics (not used by the transition logic)
        self.arm_count = 0
        self.disarm_count = 0
        self.steps = 0

    def update(self, contact: bool, terminated: bool = False) -> float:
        """Advance the FSM by one executed step; return ``g_t`` for the *next* action (0 or 1)."""
        self.steps += 1
        if terminated:
            self.state = GateState.TERMINAL
            return 0.0
        if self.state is GateState.TERMINAL:
            return 0.0

        self._contact_streak = self._contact_streak + 1 if contact else 0
        self._loss_streak = self._loss_streak + 1 if not contact else 0

        if self.state in (GateState.EARLY_CONTROL, GateState.REACQUIRE):
            if self._contact_streak >= self.cfg.arm_after:
                self.state = GateState.LATE_CONTROL_ARMED
                self.arm_count += 1
                self._loss_streak = 0
        elif self.state is GateState.LATE_CONTROL_ARMED:
            if self._loss_streak >= self.cfg.disarm_after:
                self.state = GateState.REACQUIRE
                self.disarm_count += 1
                self._contact_streak = 0

        return 1.0 if self.state is GateState.LATE_CONTROL_ARMED else 0.0

    @property
    def gate(self) -> float:
        """Current multiplier without advancing (1.0 only in LATE_CONTROL_ARMED)."""
        return 1.0 if self.state is GateState.LATE_CONTROL_ARMED else 0.0

    # ---- serialization / provenance (§3: record the exact gate contract + SHA-256) ----
    def contract(self) -> dict:
        return {"gate": "coin_phase_gate.PhaseGate", "version": 1, "config": asdict(self.cfg),
                "states": [s.value for s in GateState],
                "transition": {"arm": "contact_streak>=arm_after -> LATE_CONTROL_ARMED (from EARLY/REACQUIRE)",
                               "disarm": "loss_streak>=disarm_after -> REACQUIRE (from ARMED)",
                               "terminal": "terminated -> TERMINAL (absorbing)"},
                "multiplier": {"LATE_CONTROL_ARMED": 1.0, "other": 0.0},
                "deployable_signal": "robot_attributed_contact = left_contact or right_contact",
                "forbidden_inputs": ["seed", "trajectory_id", "future_info", "planner_state", "disk_to_zone"]}

    def contract_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.contract(), sort_keys=True).encode()).hexdigest()


def robot_attributed_contact(inner) -> bool:
    """The canonical deployable contact predicate — identical to ``eval_bc_delivery``'s ``touched`` and
    :class:`CoinRL4Dof`'s ``_touched``. Reads ``inner._planar_metrics.{left_contact,right_contact}`` (tactile)."""
    m = inner._planar_metrics
    return bool(m.left_contact or m.right_contact)
