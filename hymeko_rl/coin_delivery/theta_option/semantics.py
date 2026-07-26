"""STAGE 0 — frozen 6-D torque-θ option semantics for the teacher-to-RL campaign.

The physical campaign (tag ``coin-physical-feasibility-closed``, commit a3459629) is closed: the HyMeKo-structured
slew-admissible 6-D torque option (``forward_displacement.py``) delivers and frozen-K6 settles the coin from all four
certified cradles (development s1,s3; held-out s4,s7). This module *freezes the exact meaning and bounds* of the six θ
components and the canonical action provenance, so the learned actor and the Bellman action use the SAME deployed
semantics. Nothing here re-derives physics; it imports the frozen delivery config and K6 constants and names them.

Canonical action provenance (the anti-aliasing contract):

    actor emits proposal centre θ_0  →  fixed bounded structured search selects θ_exec  →  env executes θ_exec

For replay / Bellman updates the **Bellman action is θ_0**; θ_exec is retained as execution/search provenance and must
never silently replace θ_0 as the learned-action label. `ThetaProvenance` makes that separation structural.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.forward_displacement import ForwardConfig

# ── the CANONICAL frozen delivery option config (the exact ForwardConfig that produced the 4/4 mobile_teacher.json) ──
# lo/hi/init_std/pop/iters reproduced verbatim from horizon_authority_benchmark.deliver_main (commit a3459629).
DELIVERY_CFG = ForwardConfig(
    horizon=60, deliver=True,
    lo=(0.0, 0.0, -0.10, 1.0, 4.0, 0.0),
    hi=(0.25, 0.30, 0.10, 28.0, 48.0, 4.0),
    init_std=(0.08, 0.10, 0.05, 8.0, 12.0, 1.2),
    pop=56, iters=10,
)

DIM = 6
THETA_NAMES = ("squeeze_mag", "forward_mag", "balance", "ramp_steps", "release_step", "brake_gain")
THETA_UNITS = ("N·m (grip incr)", "N·m (push incr)", "N·m (L/R steer incr)", "control steps",
               "control steps", "N·m per m/s (velocity-feedback brake)")
THETA_PHASE = ("PUSH+BRAKE+RELEASE (grip, all phases)", "PUSH", "PUSH (L−R steer)",
               "PUSH↔BRAKE boundary", "BRAKE↔RELEASE boundary", "BRAKE (velocity feedback)")
THETA_INTERP = (
    "grip magnitude: each tip driven toward the coin centre along the inward geom Jacobian; held through PUSH+BRAKE, "
    "relaxed to -0.5·squeeze at RELEASE",
    "forward push magnitude toward the target zone (Jᵀe_par of both tips), ramped over ramp_steps during PUSH",
    "L/R steer: shifts the forward push between arms (bal·(fwd_L − fwd_R)) to null spin / steer cross-track",
    "PUSH lasts t ≤ ramp_steps; the push increment ramps min(1, t/ramp) up over this window",
    "BRAKE lasts ramp_steps < t ≤ release_step; RELEASE is t > release_step",
    "velocity-feedback brake: during BRAKE, Δτ ∝ brake_gain·‖v_coin‖ driving the tips OPPOSITE the coin velocity through "
    "the contacts (a decelerating increment, NOT a timed impulse)",
)


def theta_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Frozen (lo, hi) of the 6-D θ box. # Postconditions: two float64 length-6 arrays, lo < hi componentwise."""
    return np.asarray(DELIVERY_CFG.lo, np.float64), np.asarray(DELIVERY_CFG.hi, np.float64)


class ThetaBox:
    """Affine normaliser between legal θ (in the frozen delivery box) and the network action space ``z ∈ [-1,1]^6``.

    z_i = 2·(θ_i − lo_i)/(hi_i − lo_i) − 1 ;  θ_i = clip((z_i+1)/2·(hi_i−lo_i) + lo_i, lo_i, hi_i).

    # Preconditions: inputs broadcastable to (...,6). # Postconditions: `denorm` output is always a LEGAL θ (clipped to
    the box); `norm∘denorm` is the identity on legal θ; `denorm` clamps z outside [-1,1] back into the box (a Tanh actor
    can never emit an out-of-box option).
    """

    def __init__(self) -> None:
        self.lo, self.hi = theta_bounds()
        self.span = self.hi - self.lo
        if np.any(self.span <= 0):
            raise ValueError("θ box has a non-positive span component")

    def norm(self, theta: np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(theta, np.float64), self.lo, self.hi)
        return np.asarray(2.0 * (t - self.lo) / self.span - 1.0, np.float32)

    def denorm(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(np.asarray(z, np.float64), -1.0, 1.0)
        theta = (z + 1.0) * 0.5 * self.span + self.lo
        return np.asarray(np.clip(theta, self.lo, self.hi), np.float32)

    def clip(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(np.asarray(theta, np.float64), self.lo, self.hi), np.float32)


@dataclass(frozen=True)
class ThetaProvenance:
    """Structural separation of the Bellman action (proposal centre θ_0) from the executed/search-selected θ_exec.

    ``center`` (θ_0) is the ONLY thing used as the learned-action label / Bellman action; ``selected`` (θ_exec) is what
    the fixed bounded search executed — provenance for audit, never a substitute label. ``displacement`` = ‖θ_exec−θ_0‖
    in normalised space (how far the search moved from the proposal)."""

    center: np.ndarray            # θ_0 — the Bellman action
    selected: np.ndarray          # θ_exec — execution/search provenance ONLY
    score: float
    budget: int
    outcome: dict[str, Any]

    def displacement(self, box: "ThetaBox | None" = None) -> float:
        box = box or ThetaBox()
        return float(np.linalg.norm(box.norm(self.selected) - box.norm(self.center)))

    def as_dict(self) -> dict[str, Any]:
        return {"theta_center": [round(float(x), 5) for x in np.asarray(self.center)],
                "theta_selected": [round(float(x), 5) for x in np.asarray(self.selected)],
                "search_displacement_norm": round(self.displacement(), 5),
                "score": float(self.score), "budget": int(self.budget),
                "k6": self.outcome.get("k6_delivered"), "dtz_end": self.outcome.get("dtz_end")}


def option_semantics() -> dict[str, Any]:
    """The machine-readable freeze of the 6-D θ option (Stage 0 artifact ``option_semantics.json``). Records the exact
    meaning, units, bounds, phase, and interpretation of every component plus the option duration, action clipping,
    structured-search perturbation semantics, termination, and the frozen K6 evaluation. Read-only description — building
    it does not run physics."""
    lo, hi = theta_bounds()
    comps = [
        {"index": i, "name": THETA_NAMES[i], "unit": THETA_UNITS[i], "lo": float(lo[i]), "hi": float(hi[i]),
         "phase": THETA_PHASE[i], "interpretation": THETA_INTERP[i]}
        for i in range(DIM)
    ]
    return {
        "contract": "COIN_6D_TORQUE_THETA_OPTION_SEMANTICS_V1",
        "base_tag": "coin-physical-feasibility-closed", "base_commit": "a3459629",
        "dim": DIM, "components": comps,
        "phase_schedule": "PUSH (t ≤ ramp_steps) → BRAKE (ramp_steps < t ≤ release_step, velocity-feedback) → "
                          "RELEASE (t > release_step)",
        "velocity_feedback_brake": "during BRAKE, Δτ ∝ brake_gain·‖v_coin‖ driving the tips opposite the coin velocity "
                                   "through the contacts (decelerating increment, not a timed impulse)",
        "per_step_map": "θ → slew-admissible Δτ_cmd (|Δτ_cmd| ≤ τ̇·dt) from live geometry; a = clip(prev_tau + Δτ_cmd, "
                        "actuator_lo, actuator_hi) then the per-sub-step directional velocity governor (motion contract V4)",
        "action_clipping": {
            "network_action_space": "z ∈ [-1,1]^6 (Tanh actor)",
            "theta_from_action": "affine denormalise to the frozen box then hard clip to [lo,hi] (ThetaBox.denorm)",
            "per_step_dtau_clip": "Δτ_cmd clipped to ±(τ̇·dt) (slew-admissible)",
            "actuator_clip": "a clipped to actuator_ctrlrange before the governor",
        },
        "option_duration": {"horizon_control_steps": DELIVERY_CFG.horizon,
                            "note": "one option = the full PUSH→BRAKE→RELEASE rollout (single continuous trajectory); "
                                    "the episode is a single terminal option (semi-MDP τ = horizon, no bootstrap)"},
        "structured_search": {
            "role": "fixed bounded local search around θ_0; selects θ_exec by the frozen canonical delivery score",
            "perturbation": "Gaussian jitter in normalised z-space with a FROZEN per-component std, denorm-clipped to the box",
            "score": "forward_displacement.score(deliver=True): -dtz_end - lost_penalty·lost_before_release + "
                     "0.02·k6_max_dwell + 0.10·k6_delivered (higher = better)",
            "budget_semantics": "budget = TOTAL candidate evaluations; the centre θ_0 is ALWAYS evaluated + (budget-1) "
                                "jittered neighbours; budget≤1 = centre only (direct execution). FROZEN & identical for "
                                "all states and for the informed/uninformed/oracle conditions",
            "budget_counts_total_candidates": True, "centre_always_evaluated": True,
            "n_jittered_candidates": "budget - 1",
            "provenance": "θ_0 is the Bellman action; θ_exec (search-selected) is provenance only (never a label)",
        },
        "termination_and_k6": {
            "monitor": "frozen K6 (forward_displacement.rollout_primitive / coin_rl_env constants)",
            "CENTER_TOL_m": CENTER_TOL, "SETTLE_VEL_mps": SETTLE_VEL, "HELD_DWELL_steps": HELD_DWELL,
            "predicate": "k6_delivered = (max consecutive steps with dtz ≤ CENTER_TOL ∧ coin_speed < SETTLE_VEL) ≥ "
                         "HELD_DWELL AND touched, with the motion contract held over a single continuous trajectory "
                         "(no re-grip / teleport / coin-state edit)",
            "delivery_success": "delivery_success(m): k6_delivered AND peak_qdot ≤ joint_vel_hard AND peak_coin_speed ≤ "
                                "ee_speed_hard",
        },
        "frozen_split": {"development": ["s1", "s3"], "held_out": ["s4", "s7"],
                        "note": "held-out is eval-only; never used for actor fitting or hyperparameter selection"},
    }
