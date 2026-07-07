"""The hierarchical trajectory submonitors — one Strategy per aspect of the manipulation contract.

Each :class:`TrajectoryMonitor` reads the shared :class:`~hymeko_rl.eval.task_monitor.context.MonitorContext`
and returns a :class:`SubVerdict` (pass/fail + robustness score + violation reasons + relevant time indices +
trajectory slices). The root :class:`~hymeko_rl.eval.task_monitor.root.TaskMonitor` composes them.

Hierarchy: Geometry (base facts) → Approach → Contact → Progress → Delivery → AntiExploit. The score convention
is ``>0`` good / ``<0`` bad (``tanh``-squashed), so scores are comparable across aspects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .context import MonitorContext, SupportsDistanceSeries, contiguous_runs


@dataclass
class SubVerdict:
    """One submonitor's output: boolean verdict, robustness score, and explanation with time evidence."""

    name: str
    passed: bool
    score: float
    violations: list[str] = field(default_factory=list)
    time_indices: list[int] = field(default_factory=list)   # steps most relevant to the verdict
    slices: list[tuple[int, int]] = field(default_factory=list)  # contiguous [start, end) evidence runs

    def as_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "score": round(self.score, 4),
                "violations": self.violations, "time_indices": self.time_indices[:32], "slices": self.slices}


@dataclass
class StagnationVerdict(SubVerdict):
    """A stagnation sub-verdict: extends :class:`SubVerdict` with the no-progress-window duration + flag.

    ``stagnated == not passed`` (a stagnating rollout does not pass this aspect). Both are surfaced by name
    because a downstream consumer (a future CIP export) reads ``stagnation_duration`` / ``stagnated`` directly;
    :meth:`as_dict` serialises them so they survive into :meth:`TaskVerdict.as_dict`.
    """

    stagnation_duration: int = 0
    stagnated: bool = False

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["stagnation_duration"] = self.stagnation_duration
        d["stagnated"] = self.stagnated
        return d


def _tanh(x: float) -> float:
    return float(np.tanh(x))


class TrajectoryMonitor(ABC):
    """A submonitor verifying one aspect of the manipulation contract over a recorded trajectory."""

    name: str = "monitor"

    @abstractmethod
    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        ...


class GeometryMonitor(TrajectoryMonitor):
    """Base geometric facts: coin-target distance, fingertip-coin distances, arm-body-coin, zone membership."""

    name = "geometry"

    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        c = ctx.contract
        min_d = float(ctx.dist.min())
        reached_zone = min_d <= c.zone_half
        score = _tanh((ctx.d0 - min_d) / max(ctx.d0, 1e-3))
        viol = [] if reached_zone else ["coin_never_reached_zone"]
        return SubVerdict(self.name, reached_zone, score, viol,
                          time_indices=list(np.where(ctx.in_zone)[0]), slices=contiguous_runs(ctx.in_zone))


class ApproachMonitor(TrajectoryMonitor):
    """Fingertips approach the coin, the coin is not pushed away during approach, approach precedes displacement."""

    name = "approach"

    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        c = ctx.contract
        approached = ctx.first_approach > 0
        # coin displacement magnitude from its start position
        disp = np.linalg.norm(ctx.coin_xy - ctx.coin_xy[0], axis=1)
        disp_mask = disp > c.near_coin
        first_disp = int(np.argmax(disp_mask) + 1) if disp_mask.any() else 0
        approach_before_disp = approached and (first_disp == 0 or ctx.first_approach <= first_disp)
        # not pushed away before engagement: coin didn't recede a lot before both-fingertip contact
        pre_end = ctx.first_two_finger if ctx.first_two_finger else ctx.n
        pushed = float(ctx.dist[:pre_end].max() - ctx.d0) > c.near_coin if pre_end > 0 else False
        passed = bool(approached and approach_before_disp and not pushed)
        score = _tanh((c.near_coin - float(ctx.min_tip.min())) / c.near_coin)
        viol = []
        if not approached:
            viol.append("fingertips_never_approached")
        if not approach_before_disp:
            viol.append("coin_displaced_before_approach")
        if pushed:
            viol.append("coin_pushed_away_during_approach")
        appr_steps = list(np.where(ctx.min_tip <= c.near_coin)[0])
        return SubVerdict(self.name, passed, score, viol, time_indices=appr_steps,
                          slices=contiguous_runs(ctx.min_tip <= c.near_coin))


class ContactMonitor(TrajectoryMonitor):
    """Left/right fingertip contact, both-fingertip engagement, contact duration + timing."""

    name = "contact"

    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        c = ctx.contract
        engaged = ctx.two_finger_steps >= c.min_two_finger_steps
        score = _tanh(ctx.two_finger_steps / 10.0) - (0.0 if engaged else 1.0)
        viol = []
        if not ctx.left_contact.any():
            viol.append("no_left_fingertip_contact")
        if not ctx.right_contact.any():
            viol.append("no_right_fingertip_contact")
        if not engaged:
            viol.append("no_both_fingertip_engagement")
        return SubVerdict(self.name, bool(engaged), score, viol,
                          time_indices=list(np.where(ctx.both_contact)[0]),
                          slices=contiguous_runs(ctx.both_contact))


class ProgressMonitor(TrajectoryMonitor):
    """Distance-to-target decreases, positive target-directed progress, fingertip- vs body-attributed progress."""

    name = "progress"

    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        c = ctx.contract
        net = ctx.d0 - ctx.dfin
        moved_toward = net > c.progress_eps
        useful_ft = ctx.ft_prog > c.progress_eps
        passed = bool(moved_toward and useful_ft)
        score = _tanh(net / max(ctx.d0, 1e-3))
        viol = []
        if not moved_toward:
            viol.append("coin_not_closer_to_target")
        if not useful_ft:
            viol.append("no_useful_fingertip_progress")
        return SubVerdict(self.name, passed, score, viol,
                          time_indices=list(np.where(ctx.ft_prog_step > c.progress_eps)[0]),
                          slices=contiguous_runs(ctx.toward > c.progress_eps))


class DeliveryMonitor(TrajectoryMonitor):
    """Coin enters the target zone, remains for k steps, final distance, stable delivery."""

    name = "delivery"

    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        c = ctx.contract
        delivered = ctx.max_dwell >= c.success_steps
        score = min(ctx.max_dwell / max(c.success_steps, 1), 1.0) if delivered else -0.5
        viol = [] if delivered else (["entered_but_unstable"] if ctx.in_zone.any() else ["never_delivered"])
        return SubVerdict(self.name, bool(delivered), float(score), viol,
                          time_indices=list(np.where(ctx.dwell >= c.success_steps)[0]),
                          slices=contiguous_runs(ctx.in_zone))


class AntiExploitMonitor(TrajectoryMonitor):
    """Body-driven / body-assisted delivery, arm-body shove, large body-only progress, no-engagement delivery."""

    name = "anti_exploit"

    def evaluate(self, ctx: MonitorContext) -> SubVerdict:
        c = ctx.contract
        delivered = ctx.max_dwell >= c.success_steps
        body_low = ctx.body_prog <= c.body_eps
        ft_dominant = ctx.body_prog <= max(c.body_eps, c.ft_dominant_frac * ctx.total_prog)
        body_driven = delivered and ctx.body_prog > ctx.ft_prog
        no_engagement_delivery = delivered and ctx.two_finger_steps == 0
        passed = bool(body_low and ft_dominant and not body_driven and not no_engagement_delivery)
        score = _tanh((ctx.ft_prog - ctx.body_prog) / max(ctx.total_prog, 1e-3))
        viol = []
        if not body_low:
            viol.append("body_only_progress_too_high")
        if not ft_dominant:
            viol.append("progress_not_fingertip_dominant")
        if body_driven:
            viol.append("body_driven_delivery")
        if no_engagement_delivery:
            viol.append("delivery_without_fingertip_engagement")
        body_steps = list(np.where(ctx.body_prog_step > 0.0)[0])
        return SubVerdict(self.name, passed, score, viol, time_indices=body_steps,
                          slices=contiguous_runs(ctx.body_prog_step > 0.0))


class StagnationMonitor(TrajectoryMonitor):
    """Reward-independent no-progress detector over the distance-to-target series.

    Flags a *no-progress window*: a trailing window of ``window`` steps whose NET progress toward the target
    (``dist[t-window] - dist[t]``) is below ``eps``. It reads only the generic distance-to-target series
    (:attr:`MonitorContext.dist`) — no coin / fingertip / contact fields — so it ports to any task whose context
    exposes a distance-to-target, which is why it is the first generalisation step off the coin-delivery task.

    NET window displacement is used deliberately, NOT a one-sided per-step ``toward`` sum: an oscillatory
    trajectory accumulates large one-sided ``toward`` motion yet makes zero net improvement, and must read as
    stagnation. ``dist`` is derived from positions only, so this stays reward-independent.

    It accepts any :class:`~hymeko_rl.eval.task_monitor.context.SupportsDistanceSeries` (``n`` + ``dist``), not
    just the coin :class:`MonitorContext`, so the same monitor composes onto MetaWorld tasks (coffee-push, …).

    # Preconditions
    * ``window >= 1``, ``eps >= 0.0``, ``min_steps >= 0`` (validated at construction).
    * ``ctx.dist`` is the per-step distance-to-target; ``ctx.n`` its length (both on any distance-series context).

    # Postconditions
    * Returns a :class:`StagnationVerdict` with ``passed == (not stagnated)`` and ``stagnation_duration`` =
      the longest contiguous run of no-progress steps.
    * A rollout shorter than ``max(window, min_steps)`` never stagnates (insufficient evidence, not a pass claim).
    * Reward-independent: no reward or policy internals are read.
    """

    name = "stagnation"

    def __init__(self, window: int = 20, eps: float = 0.005, min_steps: int = 0) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")
        self.window = window
        self.eps = eps
        self.min_steps = min_steps

    def evaluate(self, ctx: SupportsDistanceSeries) -> StagnationVerdict:
        w, n = self.window, ctx.n
        start = max(w, self.min_steps)   # earliest step with a full window of history past the min-step floor
        if n <= start:                   # insufficient evidence to judge → never flag (not a positive claim)
            return StagnationVerdict(self.name, True, 0.0, [], [], [],
                                     stagnation_duration=0, stagnated=False)
        net = np.full(n, np.inf)
        net[w:] = ctx.dist[:-w] - ctx.dist[w:]        # net[t] = dist[t-w] - dist[t] (>0 = got closer)
        stag_mask = np.zeros(n, bool)
        stag_mask[start:] = net[start:] < self.eps
        runs = contiguous_runs(stag_mask)
        stagnation_duration = int(max((e - s for s, e in runs), default=0))
        stagnated = stagnation_duration > 0
        if stagnated:
            score = -_tanh(stagnation_duration / w)
            viol = [f"no_progress_window (stagnation_duration={stagnation_duration}, "
                    f"window={w}, eps={self.eps})"]
        else:
            score = _tanh(float(net[start:].max()) / max(self.eps, 1e-3))   # best window improvement, normalised
            viol = []
        return StagnationVerdict(self.name, not stagnated, score, viol,
                                 time_indices=list(np.where(stag_mask)[0]), slices=runs,
                                 stagnation_duration=stagnation_duration, stagnated=stagnated)


def default_submonitors() -> list[TrajectoryMonitor]:
    """The canonical gating hierarchy, in evaluation order.

    :class:`StagnationMonitor` is intentionally NOT included here: it is a reward-independent diagnostic, not a
    gating aspect of the coin-delivery success formula, so the root task verdict is unchanged. Compose it
    explicitly (``default_submonitors() + [StagnationMonitor()]``) to report it alongside the gating aspects.
    """
    return [GeometryMonitor(), ApproachMonitor(), ContactMonitor(),
            ProgressMonitor(), DeliveryMonitor(), AntiExploitMonitor()]
