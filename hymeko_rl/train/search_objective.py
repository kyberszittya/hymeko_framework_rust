"""SearchObjective — the training/optimization objective, kept SEPARATE from the frozen external verifier
(:class:`~hymeko_rl.eval.task_monitor.TaskMonitor`).

The ε-sweep + gradient probe (report 2026-07-07-eps-sweep-gradient-probe) showed a scalar critic ranks whole
policies correctly but its local action-gradient is monitor-misaligned (it raises Q by cutting two-fingertip
contact). The remedy is a VECTOR of monitor-derived component signals a projected-gradient update can respect as
objectives + constraints — NOT one scalar. This module defines those per-step component signals. It is deliberately
a distinct object from ``TaskMonitor``: the monitor stays the final verifier; this is only the learning objective,
so optimizer and verifier are not the same instance (per the branch directive).
"""
from __future__ import annotations

from dataclasses import dataclass

# component signal names (the vector), and the objective / constraint split
COMPONENTS = ("approach", "contact", "progress", "delivery", "antiexploit", "body_progress")
OBJECTIVES = ("delivery", "progress")               # maximise (+ fingertip_progress_ratio at trajectory level)
# constraint name -> direction the component must move ("up" = must not decrease, "down" = must not increase)
CONSTRAINTS = {"contact": "up", "antiexploit": "up", "body_progress": "down"}


@dataclass(frozen=True)
class SearchObjective:
    """Per-step monitor-derived component signals for vector-critic learning (NOT the verifier)."""

    near_coin: float = 0.06
    progress_eps: float = 0.002

    def step_signals(self, *, prev_dist: "float | None", dist: float, min_tip: float, both_contact: bool,
                     fingertip_contact: bool, arm_body_contact: bool, in_zone: bool) -> dict[str, float]:
        """The component signal vector for one transition (all monitor-physical, no policy internals).

        ``progress`` is fingertip-attributed toward-zone displacement; ``body_progress`` is body-only toward-zone
        displacement (a constraint to keep DOWN); ``antiexploit`` is fingertip-minus-body progress."""
        toward = max(0.0, prev_dist - dist) if prev_dist is not None else 0.0
        ft = toward if fingertip_contact else 0.0
        body = toward if (arm_body_contact and not fingertip_contact) else 0.0
        return {
            "approach": max(0.0, 1.0 - min_tip / self.near_coin),   # 1 at contact, 0 far
            "contact": 1.0 if both_contact else 0.0,                # two-fingertip engagement
            "progress": ft,                                         # fingertip-attributed progress
            "delivery": 1.0 if in_zone else 0.0,                   # in target zone
            "antiexploit": ft - body,                              # fingertip minus body progress
            "body_progress": body,                                 # body-only progress (minimise)
        }
