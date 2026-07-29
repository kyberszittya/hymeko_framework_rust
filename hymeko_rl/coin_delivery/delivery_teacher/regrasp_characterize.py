"""Capture->delivery re-grasp characterization: does each attempt reach the certified delivery-start manifold?

R11.3 localized failures to "delivery/settle", but the min_dtz taxonomy conflated two mechanisms. This measures, per
attempt, whether the frozen capture->delivery transition actually re-establishes a delivery-ready bilateral grasp (reaches
KINETIC, dual contact, valid delivery start) — separating the delivery-teacher-*addressable* cases from
``CAPTURE_TO_DELIVERY_REGRASP`` interface failures.

This is a coverage measurement of the *current* frozen re-grasp controller over the RRT-straddle set — G_RRT ⊆ B_regrasp.
A collision-free precontact is not the same as a delivery-ready bilateral grasp. Physical feasibility is not in question
(prior trajectories delivered); this measures where the *current* controller's basin of attraction covers the RRT inputs.
A coin nudge produced while still in APPROACH is NOT delivery progress.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.moving_precapture import R2_ALPHA
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout

DELIVERY_BAND_MM = 40.0    # min_dtz > this = "fell short" (delivery failure); <= this = "in the vicinity" (settle failure)


class RegraspClass(Enum):
    """Primary capture->delivery interface class (exactly one per attempt)."""

    DELIVERY_ADDRESSABLE = "delivery_addressable"                               # reaches KINETIC + valid bilateral start
    CAPTURE_TO_DELIVERY_REGRASP_STUCK_APPROACH = "regrasp_stuck_approach"       # never KINETIC, never dual contact
    CAPTURE_TO_DELIVERY_REGRASP_PARTIAL_CONTACT = "regrasp_partial_contact"     # never KINETIC, some dual contact
    CAPTURE_TO_DELIVERY_REGRASP_ESTABLISHED_THEN_LOST = "regrasp_established_then_lost"  # KINETIC but grasp lost early


@dataclass(frozen=True)
class RegraspMetrics:
    reaches_kinetic: bool
    step_to_kinetic: int
    handoff_reset_fired: bool
    max_bilateral_dwell: int
    touched: bool
    grasp_at_delivery_start: bool
    coin_disp_before_kinetic_mm: float
    max_progress_in_approach_mm: float
    min_dtz_mm: float
    k6: bool
    regrasp_class: str
    addressable_outcome: str    # "" unless DELIVERY_ADDRESSABLE: SUCCESS / DELIVERY_FAILURE_.. / SETTLE_FAILURE_..


def _controller(snap: Any, down: Any) -> Any:
    return HandoffResetTemporalController(snap, CloneActor(down.model, down.norm), down.r2_fn,
                                          ResidualBounds(alpha=R2_ALPHA))


def _trace_delivery(snap: Any, down: Any) -> "tuple[list[dict[str, Any]], Any, dict[str, Any]]":
    """Run the frozen delivery with a phase/contact hook; return (per-step trace, controller, primitive metrics)."""
    ctrl = _controller(snap, down)
    coin0 = np.asarray(snap.branch().inner._planar_metrics.disk_pos, np.float64)[:2]
    trace: list[dict[str, Any]] = []

    def hook(rl: Any, t: int) -> None:
        con = primary_fingertip_contacts(rl)
        nc = int(con["left"] is not None) + int(con["right"] is not None)
        coin = np.asarray(rl.inner._planar_metrics.disk_pos, np.float64)[:2]
        trace.append({"t": t, "phase": int(getattr(ctrl, "phase", 0)), "contacts": nc,
                      "reset": len(getattr(ctrl, "handoff_trace", [])) > 0,
                      "dtz_mm": float(rl.inner.direction_to_zone()[1]) * 1000.0,
                      "coin_disp_mm": float(np.linalg.norm(coin - coin0)) * 1000.0})

    m = velocity_rollout(snap, ctrl, down.cfg, frame_hook=hook)
    return trace, ctrl, m


def _max_bilateral_dwell(trace: "list[dict[str, Any]]") -> int:
    dwell, run = 0, 0
    for r in trace:
        run = run + 1 if r["contacts"] >= 2 else 0
        dwell = max(dwell, run)
    return dwell


@dataclass(frozen=True)
class _TraceStats:
    reaches: bool
    step_to_kin: int
    dwell: int
    grasp_start: bool
    disp_before: float
    prog_approach: float


def _approach_window(trace: "list[dict[str, Any]]", reaches: bool, step_to_kin: int) -> "tuple[float, float]":
    """(max coin displacement, max dtz progress) during APPROACH only — a nudge before KINETIC is NOT delivery progress."""
    before = trace if not reaches else [r for r in trace if r["t"] < step_to_kin]
    disp = max((r["coin_disp_mm"] for r in before), default=0.0)
    dtzs = [r["dtz_mm"] for r in before]
    prog = (dtzs[0] - min(dtzs)) if dtzs else 0.0
    return round(disp, 2), round(max(prog, 0.0), 2)


def _trace_stats(trace: "list[dict[str, Any]]") -> _TraceStats:
    kin = [r for r in trace if r["phase"] == KINETIC]
    reaches = len(kin) > 0
    step_to_kin = kin[0]["t"] if reaches else -1
    disp_before, prog = _approach_window(trace, reaches, step_to_kin)
    return _TraceStats(reaches=reaches, step_to_kin=int(step_to_kin), dwell=_max_bilateral_dwell(trace),
                       grasp_start=bool(reaches and kin[0]["contacts"] >= 2), disp_before=disp_before,
                       prog_approach=prog)


def _classify(trace: "list[dict[str, Any]]", m: dict[str, Any], down: Any) -> RegraspMetrics:
    ts = _trace_stats(trace)
    min_dtz = round(m["dtz_end"] * 1000, 2)
    k6 = bool(delivery_success(m, down.cfg))
    cls, outcome = _primary_class(ts.reaches, ts.dwell, ts.grasp_start, k6, min_dtz)
    return RegraspMetrics(reaches_kinetic=ts.reaches, step_to_kinetic=ts.step_to_kin,
                          handoff_reset_fired=any(r["reset"] for r in trace), max_bilateral_dwell=ts.dwell,
                          touched=ts.dwell > 0, grasp_at_delivery_start=ts.grasp_start,
                          coin_disp_before_kinetic_mm=ts.disp_before, max_progress_in_approach_mm=ts.prog_approach,
                          min_dtz_mm=min_dtz, k6=k6, regrasp_class=cls.value, addressable_outcome=outcome)


def _primary_class(reaches: bool, dwell: int, grasp_start: bool, k6: bool, min_dtz: float) -> "tuple[RegraspClass, str]":
    if not reaches:
        return ((RegraspClass.CAPTURE_TO_DELIVERY_REGRASP_STUCK_APPROACH if dwell == 0
                 else RegraspClass.CAPTURE_TO_DELIVERY_REGRASP_PARTIAL_CONTACT), "")
    if not grasp_start:
        return RegraspClass.CAPTURE_TO_DELIVERY_REGRASP_ESTABLISHED_THEN_LOST, ""
    if k6:
        return RegraspClass.DELIVERY_ADDRESSABLE, "SUCCESS"
    outcome = ("SETTLE_FAILURE_AFTER_VALID_REGRASP" if min_dtz <= DELIVERY_BAND_MM
               else "DELIVERY_FAILURE_AFTER_VALID_REGRASP")
    return RegraspClass.DELIVERY_ADDRESSABLE, outcome


def characterize_delivery(snap: Any, down: Any) -> RegraspMetrics:
    """Run + classify one capture->delivery transition. Postcondition: exactly one primary class; addressable outcome set
    iff DELIVERY_ADDRESSABLE."""
    trace, _ctrl, m = _trace_delivery(snap, down)
    return _classify(trace, m, down)
