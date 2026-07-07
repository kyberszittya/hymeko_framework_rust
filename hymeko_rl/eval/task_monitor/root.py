"""The root :class:`TaskMonitor` — composes the trajectory submonitors (Composite pattern) into one task verdict.

``monitor_pass`` is the conjunctive task formula: fingertips approach, both fingertips engage, the coin moves
toward the target under fingertip-dominant progress with low body-only progress, the coin enters and holds the
zone for k steps, and no body-driven exploit occurs. Geometry is a base-fact layer (reported, not gated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .context import MonitorContext, record_trajectory
from .contract import MonitorContract
from .submonitors import SubVerdict, TrajectoryMonitor, default_submonitors

# The submonitors whose pass is required for the task-level PASS (geometry is a base-fact layer, not a gate).
_GATING = ("approach", "contact", "progress", "delivery", "anti_exploit")


@dataclass
class TaskVerdict:
    monitor_pass: bool
    monitor_score: float
    approach_score: float
    contact_score: float
    progress_score: float
    delivery_score: float
    anti_exploit_score: float
    violation_reason: str
    sub_verdicts: dict[str, SubVerdict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"monitor_pass": self.monitor_pass, "monitor_score": round(self.monitor_score, 4),
                "approach_score": round(self.approach_score, 4), "contact_score": round(self.contact_score, 4),
                "progress_score": round(self.progress_score, 4), "delivery_score": round(self.delivery_score, 4),
                "anti_exploit_score": round(self.anti_exploit_score, 4),
                "violation_reason": self.violation_reason,
                "sub_verdicts": {k: v.as_dict() for k, v in self.sub_verdicts.items()}}


class TaskMonitor:
    """External deterministic verifier over trajectories, generated from the HyMeKo task contract on the env."""

    def __init__(self, contract: MonitorContract,
                 submonitors: list[TrajectoryMonitor] | None = None) -> None:
        self.contract = contract
        self.submonitors = submonitors or default_submonitors()

    @classmethod
    def from_env(cls, env: Any) -> "TaskMonitor":
        return cls(MonitorContract.from_env(env))

    def evaluate(self, traj: list[dict[str, Any]]) -> TaskVerdict:
        """Run every submonitor over one recorded trajectory and aggregate into the task verdict."""
        if not traj:
            empty = SubVerdict("empty", False, -1.0, ["empty_trajectory"])
            return TaskVerdict(False, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, "empty_trajectory",
                               {"empty": empty})
        ctx = MonitorContext.build(traj, self.contract)
        subs = {m.name: m.evaluate(ctx) for m in self.submonitors}
        gate = [subs[k].passed for k in _GATING if k in subs]
        monitor_pass = bool(all(gate)) if gate else False
        scored = [subs[k].score for k in _GATING if k in subs]
        monitor_score = float(np.mean(scored)) if scored else -1.0
        # first failing gating submonitor's first violation → the headline reason
        reason = "none"
        for k in _GATING:
            if k in subs and not subs[k].passed:
                reason = subs[k].violations[0] if subs[k].violations else f"failed_{k}"
                break
        return TaskVerdict(
            monitor_pass=monitor_pass, monitor_score=monitor_score,
            approach_score=subs["approach"].score, contact_score=subs["contact"].score,
            progress_score=subs["progress"].score, delivery_score=subs["delivery"].score,
            anti_exploit_score=subs["anti_exploit"].score, violation_reason=reason, sub_verdicts=subs)

    def evaluate_policy(self, make_env: Any, make_action_fn: Any, n: int, seed0: int = 9000) -> dict[str, Any]:
        """Aggregate the task verdict over ``n`` episodes of a policy. ``make_action_fn(env) -> action_fn``."""
        env = make_env()
        mon = TaskMonitor.from_env(env)
        verdicts = []
        for ep in range(n):
            fn = make_action_fn(env)
            verdicts.append(mon.evaluate(record_trajectory(env, fn, seed0 + ep)))
        env.close()
        viol: dict[str, int] = {}
        for v in verdicts:
            if not v.monitor_pass:
                viol[v.violation_reason] = viol.get(v.violation_reason, 0) + 1
        top = max(viol, key=lambda kk: viol[kk]) if viol else "none"

        def mean(attr: str) -> float:
            return round(float(np.mean([getattr(v, attr) for v in verdicts])), 4)

        return {
            "monitor_pass_rate": round(sum(v.monitor_pass for v in verdicts) / n, 3),
            "monitor_score_mean": mean("monitor_score"),
            "monitor_score_min": round(float(np.min([v.monitor_score for v in verdicts])), 4),
            "approach_score": mean("approach_score"), "contact_score": mean("contact_score"),
            "progress_score": mean("progress_score"), "delivery_score": mean("delivery_score"),
            "anti_exploit_score": mean("anti_exploit_score"),
            "violations": dict(sorted(viol.items(), key=lambda kv: -kv[1])), "top_violation": top, "n": n,
        }
