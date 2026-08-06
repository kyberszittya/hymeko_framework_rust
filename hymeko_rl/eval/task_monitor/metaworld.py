"""MetaWorld-style runtime monitors — reward-independent, generated to a task template (coffee-push first).

This is the generalisation proof that the monitor framework is not coin-delivery-only: the same style
(:class:`~hymeko_rl.eval.task_monitor.submonitors.SubVerdict`, a Strategy submonitor per aspect, a Composite
root producing a verdict with ``monitor_pass`` / ``monitor_score`` / ``sub_verdicts``) works on a task whose
geometry is an object pushed toward a target, with no coin / fingertip / zone assumptions.

`coffee-push` scenario story: ``approach → contact/manipulation → object moves toward target → target reached``
(or stagnation / timeout). The monitor observes a position-only trajectory and judges that story without touching
the reward. It reuses :class:`~hymeko_rl.eval.task_monitor.submonitors.StagnationMonitor` unchanged (it accepts
any :class:`~hymeko_rl.eval.task_monitor.context.SupportsDistanceSeries`), and its verdict is readable by
:func:`~hymeko_rl.eval.task_monitor.cip_export.export_cip_variables`.

MetaWorld is NOT imported or run here — the trajectory is a plain list of per-step dicts, so the template is
testable with synthetic data. ``dial-turn`` and a real env wrapper are deliberately out of scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from .context import contiguous_runs
from .submonitors import StagnationMonitor, SubVerdict

# Gating aspects of the coffee-push success formula (stagnation, if composed, is diagnostic — not a gate).
_COFFEE_GATING = ("coffee_progress", "coffee_success")


def _tanh(x: float) -> float:
    return float(np.tanh(x))


def _wrap(x: Any) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi] — the shortest signed angular difference (handles the +/-pi and 0/2pi seams)."""
    return (np.asarray(x, np.float64) + np.pi) % (2.0 * np.pi) - np.pi


_Ctx = TypeVar("_Ctx")


class MetaWorldSubmonitor(ABC, Generic[_Ctx]):
    """A Strategy verifying one aspect of a MetaWorld task contract over its task context ``_Ctx``.

    One generic base for both distance-based (coffee-push) and angular (dial-turn) submonitors: they differ only
    in the context type and the aspect they check, not in the pattern — so there is one submonitor contract, not
    one per task family.
    """

    name: str = "submonitor"

    @abstractmethod
    def evaluate(self, ctx: _Ctx) -> SubVerdict:
        ...


@dataclass
class CoffeePushContext:
    """Precomputed distance-to-target view of a coffee-push trajectory (satisfies ``SupportsDistanceSeries``).

    Expected per-step trajectory fields (a plain ``list[dict]``, no MetaWorld dependency):

    * ``object_xy`` — 2-vector, object position (required),
    * ``target_xy`` — 2-vector, target position (required; usually constant across steps),
    * ``gripper_xy`` — 2-vector, end-effector position (optional),
    * ``contact`` — bool, object contact/proximity signal (optional).
    """

    n: int
    dist: np.ndarray            # [N] object-to-target distance (the SupportsDistanceSeries signal)
    object_xy: np.ndarray       # [N, 2]
    target_xy: np.ndarray       # [N, 2]
    toward: np.ndarray          # [N] max(0, dist[t-1] - dist[t]) per-step toward-target displacement
    d0: float
    dfin: float
    delta: float                # d0 - dfin (net toward-target improvement; >0 = closer)
    gripper_xy: np.ndarray | None = None    # [N, 2] if provided
    contact: np.ndarray | None = None       # [N] bool if provided

    @classmethod
    def build(cls, traj: list[dict[str, Any]]) -> "CoffeePushContext":
        n = len(traj)
        obj = np.array([s["object_xy"] for s in traj], np.float64)
        tgt = np.array([s["target_xy"] for s in traj], np.float64)
        dist = np.linalg.norm(obj - tgt, axis=1)
        toward = np.zeros(n)
        toward[1:] = np.maximum(0.0, dist[:-1] - dist[1:])
        grip = np.array([s["gripper_xy"] for s in traj], np.float64) if "gripper_xy" in traj[0] else None
        contact = np.array([bool(s["contact"]) for s in traj], bool) if "contact" in traj[0] else None
        return cls(n=n, dist=dist, object_xy=obj, target_xy=tgt, toward=toward,
                   d0=float(dist[0]), dfin=float(dist[-1]), delta=float(dist[0] - dist[-1]),
                   gripper_xy=grip, contact=contact)


class CoffeePushSubmonitor(MetaWorldSubmonitor[CoffeePushContext]):
    """A Strategy verifying one aspect of the coffee-push contract over a :class:`CoffeePushContext`."""

    name = "coffee_submonitor"


class CoffeePushProgressMonitor(CoffeePushSubmonitor):
    """Did the object move meaningfully toward the target (net improvement, not pushed away)?"""

    name = "coffee_progress"

    def __init__(self, min_progress: float = 0.01) -> None:
        if min_progress < 0.0:
            raise ValueError(f"min_progress must be >= 0, got {min_progress}")
        self.min_progress = min_progress

    def evaluate(self, ctx: CoffeePushContext) -> SubVerdict:
        moved_toward = ctx.delta > self.min_progress
        score = _tanh(ctx.delta / max(ctx.d0, 1e-3))
        viol: list[str] = []
        if not moved_toward:
            viol.append("object_moved_away_from_target" if ctx.delta < -self.min_progress
                        else "object_did_not_move_toward_target")
        return SubVerdict(self.name, bool(moved_toward), score, viol,
                          time_indices=list(np.where(ctx.toward > self.min_progress)[0]),
                          slices=contiguous_runs(ctx.toward > self.min_progress))


class CoffeePushSuccessMonitor(CoffeePushSubmonitor):
    """Is the object within the success radius of the target at the end of the rollout?"""

    name = "coffee_success"

    def __init__(self, success_radius: float = 0.05) -> None:
        if success_radius <= 0.0:
            raise ValueError(f"success_radius must be > 0, got {success_radius}")
        self.success_radius = success_radius

    def evaluate(self, ctx: CoffeePushContext) -> SubVerdict:
        reached = ctx.dfin <= self.success_radius
        score = _tanh((self.success_radius - ctx.dfin) / self.success_radius)
        viol = [] if reached else ["target_not_reached"]
        in_region = ctx.dist <= self.success_radius
        return SubVerdict(self.name, bool(reached), score, viol,
                          time_indices=list(np.where(in_region)[0]), slices=contiguous_runs(in_region))


@dataclass
class CoffeePushVerdict:
    """The coffee-push task verdict — mirrors :class:`TaskVerdict` style, with coffee-push scalar outputs.

    ``monitor_pass`` and ``progress_score`` are exposed under those names so
    :func:`~hymeko_rl.eval.task_monitor.cip_export.export_cip_variables` reads the verdict unchanged.
    """

    monitor_pass: bool
    monitor_score: float
    progress_score: float
    object_target_distance_initial: float
    object_target_distance_final: float
    object_target_distance_delta: float
    object_moved_toward_target: bool
    target_reached: bool
    violation_reason: str
    sub_verdicts: dict[str, SubVerdict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "monitor_pass": self.monitor_pass, "monitor_score": round(self.monitor_score, 4),
            "progress_score": round(self.progress_score, 4),
            "object_target_distance_initial": round(self.object_target_distance_initial, 5),
            "object_target_distance_final": round(self.object_target_distance_final, 5),
            "object_target_distance_delta": round(self.object_target_distance_delta, 5),
            "object_moved_toward_target": self.object_moved_toward_target,
            "target_reached": self.target_reached, "violation_reason": self.violation_reason,
            "sub_verdicts": {k: v.as_dict() for k, v in self.sub_verdicts.items()},
        }


class CoffeePushMonitor:
    """Composite reward-independent monitor for the MetaWorld coffee-push task template.

    Gates on progress AND success; an optional :class:`StagnationMonitor` is composed as a *diagnostic*
    (reported under ``sub_verdicts['stagnation']``, never a gate — mirroring the coin monitor's treatment).

    # Preconditions
    * ``success_radius > 0``, ``min_progress >= 0``.
    * each trajectory step carries ``object_xy`` + ``target_xy`` (``gripper_xy`` / ``contact`` optional).

    # Postconditions
    * Returns a :class:`CoffeePushVerdict`; ``monitor_pass == (object_moved_toward_target and target_reached)``.
    * Reward-independent: no reward or policy internals are read.
    """

    def __init__(self, success_radius: float = 0.05, min_progress: float = 0.01,
                 stagnation: StagnationMonitor | None = None) -> None:
        self._progress = CoffeePushProgressMonitor(min_progress)
        self._success = CoffeePushSuccessMonitor(success_radius)
        self._stagnation = stagnation

    def evaluate(self, traj: list[dict[str, Any]]) -> CoffeePushVerdict:
        if not traj:
            empty = SubVerdict("empty", False, -1.0, ["empty_trajectory"])
            return CoffeePushVerdict(False, -1.0, -1.0, 0.0, 0.0, 0.0, False, False,
                                     "empty_trajectory", {"empty": empty})
        ctx = CoffeePushContext.build(traj)
        prog = self._progress.evaluate(ctx)
        succ = self._success.evaluate(ctx)
        subs: dict[str, SubVerdict] = {prog.name: prog, succ.name: succ}
        if self._stagnation is not None:
            subs["stagnation"] = self._stagnation.evaluate(ctx)   # ctx satisfies SupportsDistanceSeries
        gate = [subs[k].passed for k in _COFFEE_GATING]
        monitor_pass = bool(all(gate))
        monitor_score = float(np.mean([subs[k].score for k in _COFFEE_GATING]))
        reason = "none"
        for k in _COFFEE_GATING:
            if not subs[k].passed:
                reason = subs[k].violations[0] if subs[k].violations else f"failed_{k}"
                break
        return CoffeePushVerdict(
            monitor_pass=monitor_pass, monitor_score=monitor_score, progress_score=prog.score,
            object_target_distance_initial=ctx.d0, object_target_distance_final=ctx.dfin,
            object_target_distance_delta=ctx.delta, object_moved_toward_target=prog.passed,
            target_reached=succ.passed, violation_reason=reason, sub_verdicts=subs)


# --- dial-turn (angular progress) -----------------------------------------------------------------------------
_DIAL_GATING = ("dial_progress", "dial_success", "dial_overshoot")


@dataclass
class DialTurnContext:
    """Precomputed angular-error view of a dial-turn trajectory (satisfies ``SupportsDistanceSeries``).

    ``dist`` is the absolute angular target error ``|wrap(angle - target)|``, so the generic
    :class:`StagnationMonitor` composes on angular progress — no angular-specific stagnation class is needed.

    Expected per-step fields (plain ``list[dict]``, no MetaWorld dependency):

    * ``dial_angle`` — scalar dial angle in radians (required),
    * ``target_angle`` — scalar target angle in radians (required; usually constant),
    * ``gripper_xy`` / ``dial_xy`` / ``contact`` / ``time`` — optional (only ``contact`` is retained here).
    """

    n: int
    dist: np.ndarray            # [N] |wrap(angle - target)| angular target error (SupportsDistanceSeries signal)
    angle: np.ndarray           # [N] raw dial angle (radians)
    signed_err: np.ndarray      # [N] wrap(angle - target) in (-pi, pi]
    target_angle: float
    angle_initial: float
    angle_final: float
    angle_delta: float          # wrap(angle[-1] - angle[0]) net signed shortest-path rotation
    err0: float
    errfin: float
    contact: np.ndarray | None = None

    @classmethod
    def build(cls, traj: list[dict[str, Any]]) -> "DialTurnContext":
        n = len(traj)
        angle = np.array([s["dial_angle"] for s in traj], np.float64)
        target = np.array([s["target_angle"] for s in traj], np.float64)
        signed = _wrap(angle - target)
        abs_err = np.abs(signed)
        contact = np.array([bool(s["contact"]) for s in traj], bool) if "contact" in traj[0] else None
        return cls(n=n, dist=abs_err, angle=angle, signed_err=signed, target_angle=float(target[-1]),
                   angle_initial=float(angle[0]), angle_final=float(angle[-1]),
                   angle_delta=float(_wrap(angle[-1] - angle[0])),
                   err0=float(abs_err[0]), errfin=float(abs_err[-1]), contact=contact)


class DialTurnSubmonitor(MetaWorldSubmonitor[DialTurnContext]):
    """A Strategy verifying one aspect of the dial-turn contract over a :class:`DialTurnContext`."""

    name = "dial_submonitor"


class DialTurnProgressMonitor(DialTurnSubmonitor):
    """Did the dial rotate meaningfully toward the target angle (angular target error decreased)?"""

    name = "dial_progress"

    def __init__(self, min_rotation: float = 0.05) -> None:
        if min_rotation < 0.0:
            raise ValueError(f"min_rotation must be >= 0, got {min_rotation}")
        self.min_rotation = min_rotation

    def evaluate(self, ctx: DialTurnContext) -> SubVerdict:
        improvement = ctx.err0 - ctx.errfin           # >0 = closer to the target angle
        rotated_toward = improvement > self.min_rotation
        score = _tanh(improvement / max(ctx.err0, 1e-3))
        viol: list[str] = []
        if not rotated_toward:
            viol.append("dial_rotated_away_from_target" if improvement < -self.min_rotation
                        else "dial_did_not_rotate_toward_target")
        step_toward = np.zeros(ctx.n, bool)
        step_toward[1:] = (ctx.dist[:-1] - ctx.dist[1:]) > 0.0     # per-step angular-error decrease
        return SubVerdict(self.name, bool(rotated_toward), score, viol,
                          time_indices=list(np.where(step_toward)[0]), slices=contiguous_runs(step_toward))


class DialTurnSuccessMonitor(DialTurnSubmonitor):
    """Is the dial within the success tolerance of the target angle at the end of the rollout?"""

    name = "dial_success"

    def __init__(self, success_tolerance: float = 0.05) -> None:
        if success_tolerance <= 0.0:
            raise ValueError(f"success_tolerance must be > 0, got {success_tolerance}")
        self.success_tolerance = success_tolerance

    def evaluate(self, ctx: DialTurnContext) -> SubVerdict:
        reached = ctx.errfin <= self.success_tolerance
        score = _tanh((self.success_tolerance - ctx.errfin) / self.success_tolerance)
        viol = [] if reached else ["target_angle_not_reached"]
        in_tol = ctx.dist <= self.success_tolerance
        return SubVerdict(self.name, bool(reached), score, viol,
                          time_indices=list(np.where(in_tol)[0]), slices=contiguous_runs(in_tol))


class DialTurnOvershootMonitor(DialTurnSubmonitor):
    """Did the dial pass beyond the target by more than the overshoot tolerance (given / inferred direction)?"""

    name = "dial_overshoot"

    def __init__(self, overshoot_tolerance: float = 0.10,
                 direction: Literal["positive", "negative", "auto"] = "auto") -> None:
        if overshoot_tolerance <= 0.0:
            raise ValueError(f"overshoot_tolerance must be > 0, got {overshoot_tolerance}")
        if direction not in ("positive", "negative", "auto"):
            raise ValueError(f"direction must be positive|negative|auto, got {direction!r}")
        self.overshoot_tolerance = overshoot_tolerance
        self.direction = direction

    def _far_mask(self, ctx: DialTurnContext) -> np.ndarray:
        """Steps where the dial sits on the far side of the target beyond the overshoot tolerance."""
        s, tol = ctx.signed_err, self.overshoot_tolerance
        if self.direction == "positive":
            return s > tol
        if self.direction == "negative":
            return s < -tol
        s0 = float(ctx.signed_err[0])                 # auto: infer the intended direction from the initial side
        if abs(s0) < 1e-9:
            return np.zeros(ctx.n, bool)
        return (s > tol) if s0 < 0 else (s < -tol)

    def evaluate(self, ctx: DialTurnContext) -> SubVerdict:
        far = self._far_mask(ctx)
        overshot = bool(np.any(far))
        worst = float(np.max(np.where(far, np.abs(ctx.signed_err) - self.overshoot_tolerance, 0.0)))
        score = -_tanh(worst / max(self.overshoot_tolerance, 1e-3)) if overshot else _tanh(1.0)
        viol = ["dial_overshot_target"] if overshot else []
        return SubVerdict(self.name, not overshot, score, viol,
                          time_indices=list(np.where(far)[0]), slices=contiguous_runs(far))


@dataclass
class DialTurnVerdict:
    """The dial-turn task verdict — mirrors the :class:`CoffeePushVerdict` style with angular outputs.

    ``monitor_pass`` and ``progress_score`` are exposed under those names so ``export_cip_variables`` reads it.
    """

    monitor_pass: bool
    monitor_score: float
    progress_score: float
    angle_initial: float
    angle_final: float
    target_angle: float
    angle_delta: float
    target_error_initial: float
    target_error_final: float
    rotated_toward_target: bool
    rotated_away_from_target: bool
    target_reached: bool
    overshot: bool
    violation_reason: str
    sub_verdicts: dict[str, SubVerdict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "monitor_pass": self.monitor_pass, "monitor_score": round(self.monitor_score, 4),
            "progress_score": round(self.progress_score, 4),
            "angle_initial": round(self.angle_initial, 5), "angle_final": round(self.angle_final, 5),
            "target_angle": round(self.target_angle, 5), "angle_delta": round(self.angle_delta, 5),
            "target_error_initial": round(self.target_error_initial, 5),
            "target_error_final": round(self.target_error_final, 5),
            "rotated_toward_target": self.rotated_toward_target,
            "rotated_away_from_target": self.rotated_away_from_target,
            "target_reached": self.target_reached, "overshot": self.overshot,
            "violation_reason": self.violation_reason,
            "sub_verdicts": {k: v.as_dict() for k, v in self.sub_verdicts.items()},
        }


class DialTurnMonitor:
    """Composite reward-independent monitor for the MetaWorld dial-turn task template (angular progress).

    Gates on rotate-toward-target AND target-angle-reached AND not-overshot; an optional
    :class:`StagnationMonitor` composes on the angular error series as a diagnostic (reported, not gating).

    # Preconditions
    * ``success_tolerance > 0``, ``min_rotation >= 0``, ``overshoot_tolerance > 0``,
      ``direction in {positive, negative, auto}``.
    * each step carries ``dial_angle`` + ``target_angle`` (radians).

    # Postconditions
    * Returns a :class:`DialTurnVerdict`; ``monitor_pass == rotated_toward_target and target_reached and
      not overshot``. Angle wrapping is handled — no false failure across the +/-pi seam.
    * Reward-independent: no reward or policy internals are read.
    """

    def __init__(self, success_tolerance: float = 0.05, min_rotation: float = 0.05,
                 overshoot_tolerance: float = 0.10,
                 direction: Literal["positive", "negative", "auto"] = "auto",
                 stagnation: StagnationMonitor | None = None) -> None:
        self._progress = DialTurnProgressMonitor(min_rotation)
        self._success = DialTurnSuccessMonitor(success_tolerance)
        self._overshoot = DialTurnOvershootMonitor(overshoot_tolerance, direction)
        self._stagnation = stagnation

    def evaluate(self, traj: list[dict[str, Any]]) -> DialTurnVerdict:
        if not traj:
            empty = SubVerdict("empty", False, -1.0, ["empty_trajectory"])
            return DialTurnVerdict(False, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   False, False, False, False, "empty_trajectory", {"empty": empty})
        ctx = DialTurnContext.build(traj)
        prog = self._progress.evaluate(ctx)
        succ = self._success.evaluate(ctx)
        over = self._overshoot.evaluate(ctx)
        subs: dict[str, SubVerdict] = {prog.name: prog, succ.name: succ, over.name: over}
        if self._stagnation is not None:
            subs["stagnation"] = self._stagnation.evaluate(ctx)   # ctx satisfies SupportsDistanceSeries (dist=err)
        gate = [subs[k].passed for k in _DIAL_GATING]
        monitor_pass = bool(all(gate))
        monitor_score = float(np.mean([subs[k].score for k in _DIAL_GATING]))
        reason = "none"
        for k in _DIAL_GATING:
            if not subs[k].passed:
                reason = subs[k].violations[0] if subs[k].violations else f"failed_{k}"
                break
        rotated_away = (ctx.errfin - ctx.err0) > self._progress.min_rotation
        return DialTurnVerdict(
            monitor_pass=monitor_pass, monitor_score=monitor_score, progress_score=prog.score,
            angle_initial=ctx.angle_initial, angle_final=ctx.angle_final, target_angle=ctx.target_angle,
            angle_delta=ctx.angle_delta, target_error_initial=ctx.err0, target_error_final=ctx.errfin,
            rotated_toward_target=prog.passed, rotated_away_from_target=bool(rotated_away),
            target_reached=succ.passed, overshot=not over.passed, violation_reason=reason, sub_verdicts=subs)
