"""DemoRunner — run ONE scenario × actor × seed and surface the full per-episode state.

The demo is the human-facing lens on the framework: it reports the active scenario, actor, contact-policy
certification, the frozen mechanism classification, the α attribution vector, target progress + dwell progress,
and the delivery success / failure reason — all from the ONE shared monitor. A run is reconstructable from
(scenario config, actor, env config, seed, commit). Batch-only concerns stay in :mod:`.experiment`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hymeko_rl.coin_delivery.monitoring.attribution import (
    alpha_vector,
    decompose_alpha_free,
)
from hymeko_rl.coin_delivery.monitoring.contact_certification import contact_trace
from hymeko_rl.coin_delivery.monitoring.delivery import DeliveryMonitor
from hymeko_rl.coin_delivery.provenance import git_commit
from hymeko_rl.coin_delivery.scenarios.registry import ScenarioRegistry


@dataclass(frozen=True)
class DemoResult:
    """Everything the demo observed for one (scenario, actor, seed) run."""

    scenario_id: str
    scenario_config: dict
    actor: str
    delivery_actor: str
    seed: int
    max_steps: int
    commit: str
    strict_delivery: bool
    mechanism: str
    attribution: list          # α = [L, R, body, support, free]
    fingertip_fraction: float
    target_progress: float     # start_dtz - min_dtz
    min_dtz: float
    dwell: int
    settle_vel: float
    contact_certified: bool
    contact_reason: str
    contact_phases: int
    free_decomposition: dict
    outcome_reason: str
    trajectory: list = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("trajectory", None)
        return d


def _outcome_reason(strict: bool, initial: bool, dwell: int, mech: str) -> str:
    if strict:
        return f"delivered ({mech}, dwell {dwell})"
    if initial:
        return "rejected: initial-state success"
    if dwell == 0:
        return f"not delivered: never held in zone ({mech})"
    return f"not delivered: {mech}"


class DemoRunner:
    """Run and narrate a single scenario × actor × seed."""

    def __init__(self, monitor: "DeliveryMonitor | None" = None) -> None:
        self._monitor = monitor or DeliveryMonitor()

    @property
    def monitor(self) -> DeliveryMonitor:
        return self._monitor

    def run(self, scenario_id: str, actor_name: str, seed: int, *, max_steps: int = 60,
            deterministic: bool = True, save_trajectory: bool = False) -> DemoResult:
        """Evaluate one run through the shared monitor + contact policy. ``deterministic`` is a no-op flag for
        parity with the batch API (the frozen rollout is already seed-exact) and is recorded for provenance."""
        cfg = ScenarioRegistry.get(scenario_id)
        actor = ScenarioRegistry.list_actors()[actor_name]
        env = cfg.kinematic_variant.build_env()
        result = self._monitor.evaluate(env, seed, actor.delivery_actor, max_steps=max_steps)
        trace = contact_trace(env, seed, actor.delivery_actor, max_steps=max_steps)
        cert = cfg.contact_policy.certify(trace)
        free = decompose_alpha_free(trace)
        traj = [{"t": s.t, "state": s.state.value, "left": s.left, "right": s.right,
                 "body": s.body, "progress": round(s.progress, 5)} for s in trace] if save_trajectory else []
        return DemoResult(
            scenario_id=cfg.scenario_id, scenario_config=cfg.to_dict(), actor=actor_name,
            delivery_actor=actor.delivery_actor.value, seed=int(seed), max_steps=max_steps,
            commit=git_commit(), strict_delivery=result.strict_delivery, mechanism=result.mechanism,
            attribution=[round(float(x), 3) for x in alpha_vector(result)],
            fingertip_fraction=round(float(result.fingertip_fraction), 3), target_progress=result.progress,
            min_dtz=result.min_dtz, dwell=result.dwell, settle_vel=result.settle_vel,
            contact_certified=cert.certified, contact_reason=cert.reason, contact_phases=cert.n_contact_phases,
            free_decomposition=free.as_dict(),
            outcome_reason=_outcome_reason(result.strict_delivery, result.initial_success, result.dwell,
                                           result.mechanism),
            trajectory=traj)

    @staticmethod
    def render(result: DemoResult, *, record_video: bool = False) -> str:
        """Rendering hook. Headless-safe: with no display this is a NO-OP that logs and returns a message rather
        than attempting an off-screen GL context. State honestly that no frames were produced."""
        import os
        if not (os.environ.get("DISPLAY") or os.environ.get("MUJOCO_GL")):
            msg = ("render: no DISPLAY / MUJOCO_GL — headless no-op (set MUJOCO_GL=egl to enable). "
                   f"scenario={result.scenario_id} actor={result.actor} seed={result.seed}")
            print(msg, flush=True)
            return msg
        msg = f"render: video={record_video} available for {result.scenario_id}/{result.actor}/{result.seed}"
        print(msg, flush=True)
        return msg

    @staticmethod
    def print_report(result: DemoResult) -> None:
        """Human-readable one-run report to flushed stdout."""
        print(f"=== scenario {result.scenario_id} | actor {result.actor} "
              f"({result.delivery_actor}) | seed {result.seed} | commit {result.commit} ===", flush=True)
        print(f"  contact-policy : certified={result.contact_certified} "
              f"phases={result.contact_phases} — {result.contact_reason}", flush=True)
        print(f"  mechanism      : {result.mechanism}", flush=True)
        print(f"  attribution α  : {result.attribution} (fingertip {result.fingertip_fraction})", flush=True)
        print(f"  free decomp    : {result.free_decomposition}", flush=True)
        print(f"  target/dwell   : progress={result.target_progress} min_dtz={result.min_dtz} "
              f"dwell={result.dwell} settle_vel={result.settle_vel}", flush=True)
        print(f"  DELIVERY       : {result.strict_delivery} — {result.outcome_reason}", flush=True)

    @staticmethod
    def save(result: DemoResult, metrics_path: "str | None" = None,
             trajectory_path: "str | None" = None) -> None:
        """Persist the run: ``metrics_path`` = summary JSON, ``trajectory_path`` = per-step trace JSON."""
        if metrics_path is not None:
            p = Path(metrics_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result.summary(), indent=2, default=float))
        if trajectory_path is not None:
            p = Path(trajectory_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"scenario_id": result.scenario_id, "actor": result.actor,
                                     "seed": result.seed, "trajectory": result.trajectory}, indent=2, default=float))
