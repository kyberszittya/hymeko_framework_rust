"""The certified CAPTURE -> HANDOFF_RESET -> CONTROLLED_DELIVERY transition, made an explicit first-class event.

The frozen R2 downstream inserts ONE frozen transition-servo step (mode HANDOFF_RESET) at the first KINETIC step, which
physically re-establishes the grasp before any delivery policy acts. This module runs *exactly that* frozen step and
captures the post-reset state, so the frozen-R2 baseline and the target-conditioned teacher **both split from an identical,
physically-created, re-grasped handoff** — an apples-to-apples policy comparison.

This is a physically-executed re-grasp transition, NOT an environment reset: it writes no state directly, loads no
snapshot, does not teleport, runs in the same continuous episode, and uses the same frozen servo code + parameters as the
R2 path. HANDOFF_RESET is a **deployed certified transition** (like RRT), not a teacher-oracle: deterministic, not
scenario-optimized, no per-instance CEM. The transition + its energy are recorded as first-class events; a failure to
re-grasp is the distinct class ``CAPTURE_TO_DELIVERY_REGRASP_FAILURE`` — not a capture / delivery / precontact failure.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.moving_precapture import R2_ALPHA

TRANSITION_KIND = "CAPTURE_TO_DELIVERY_REGRASP"
REGRASP_FAILURE = "CAPTURE_TO_DELIVERY_REGRASP_FAILURE"


def _both_contacts(rl: Any) -> int:
    con = primary_fingertip_contacts(rl)
    return int(con["left"] is not None) + int(con["right"] is not None)


def _coin_state(rl: Any) -> "tuple[np.ndarray, float]":
    pm = rl.inner._planar_metrics
    return np.asarray(pm.disk_pos, np.float64)[:2], float(np.linalg.norm(np.asarray(pm.disk_vel, np.float64)[:2]))


@dataclass(frozen=True)
class HandoffResetOutcome:
    """The certified post-reset handoff + the frozen-R2 baseline that shares it. ``post_reset`` is None on regrasp failure."""

    post_reset: Optional[Any]                    # TransportSnapshot both branches split from (re-grasped)
    regrasp_ok: bool
    regrasp_contacts: int
    reset_step: int
    baseline_k6: bool
    baseline_min_dtz_mm: float
    reset_w_pos: float
    reset_w_neg: float
    coin_ke_before: float
    coin_ke_after: float
    coin_moved_in_reset_mm: float
    post_reset_state_hash: str


def apply_handoff_reset(snap: Any, down: Any) -> HandoffResetOutcome:
    """Run the frozen CAPTURE->HANDOFF_RESET->R2 delivery, capturing the post-reset state the instant the frozen reset
    step latches. Returns the shared post-reset handoff + the frozen-R2 baseline outcome + the reset-phase energetics.

    Preconditions: ``snap`` is a raw post-capture handoff; ``down`` is the FrozenDownstream (``rig['down']``).
    Postconditions: on success, ``post_reset`` is a re-grasped handoff both policies can branch from.
    """
    from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

    coin0, _sp0 = _coin_state(snap.branch())
    ke_before = 0.5 * float(_sp0 ** 2)
    ctrl = HandoffResetTemporalController(snap, CloneActor(down.model, down.norm), down.r2_fn,
                                          ResidualBounds(alpha=R2_ALPHA))
    cap: dict[str, Any] = {}
    dt = float(snap.stack.control_dt)
    acc = {"w_pos": 0.0, "w_neg": 0.0}

    def hook(rl: Any, t: int) -> None:
        d = rl.inner.data
        power = float(np.asarray(d.ctrl[:4], np.float64) @ np.asarray(d.qvel[:4], np.float64))
        if "post_reset" not in cap:                              # accumulate the reset-phase (capture -> post-reset) work
            acc["w_pos"] += max(power, 0.0) * dt
            acc["w_neg"] += max(-power, 0.0) * dt
        if getattr(ctrl, "_handoff_reset_done", False) and "post_reset" not in cap:
            cap["post_reset"] = (copy.deepcopy(rl), np.asarray(d.ctrl[:4], np.float64).copy(), int(t))

    m = _run_delivery(snap, ctrl, down, hook)   # frozen R2 delivery + reset capture (single instrumented run)
    baseline_k6 = bool(delivery_success(m, down.cfg))
    baseline_min_dtz = round(_min_dtz_mm(snap, m), 2)
    if "post_reset" not in cap:
        return HandoffResetOutcome(None, False, 0, -1, baseline_k6, baseline_min_dtz, round(acc["w_pos"], 6),
                                   round(acc["w_neg"], 6), round(ke_before, 6), round(ke_before, 6), 0.0, "")
    rl_pr, prev_tau, rstep = cap["post_reset"]
    coin1, sp1 = _coin_state(rl_pr)
    post = kc.TransportSnapshot.from_live(rl_pr, snap.stack, prev_tau)
    contacts = _both_contacts(post.branch())
    return HandoffResetOutcome(
        post_reset=post, regrasp_ok=contacts >= 2, regrasp_contacts=contacts, reset_step=rstep,
        baseline_k6=baseline_k6, baseline_min_dtz_mm=baseline_min_dtz, reset_w_pos=round(acc["w_pos"], 6),
        reset_w_neg=round(acc["w_neg"], 6), coin_ke_before=round(ke_before, 6),
        coin_ke_after=round(0.5 * float(sp1 ** 2), 6),
        coin_moved_in_reset_mm=round(float(np.linalg.norm(coin1 - coin0)) * 1000, 2),
        post_reset_state_hash=_state_hash(rl_pr, prev_tau))


def _run_delivery(snap: Any, ctrl: Any, down: Any, hook: Any) -> dict[str, Any]:
    """Drive the frozen delivery with the reset-capturing hook (mirrors FrozenDownstream.deliver_with_trace)."""
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    return velocity_rollout(snap, ctrl, down.cfg, frame_hook=hook)


def _state_hash(rl: Any, prev_tau: np.ndarray) -> str:
    """A deterministic content hash over the physically-relevant post-reset state (q, qdot, coin, contacts, step, τ)."""
    import hashlib
    import json
    d = rl.inner.data
    con = primary_fingertip_contacts(rl)
    payload = {
        "qpos": [round(float(x), 8) for x in d.qpos],
        "qvel": [round(float(x), 8) for x in d.qvel],
        "prev_tau": [round(float(x), 8) for x in prev_tau],
        "contacts": int(con["left"] is not None) + int(con["right"] is not None),
        "time": round(float(d.time), 6),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
