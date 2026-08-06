"""ExperimentRunner — batch factorial evaluation over scenarios × actors × paired seeds.

For each (scenario, actor) cell the runner evaluates the shared delivery monitor over a paired-seed set, records
the contact-policy certification and the frozen mechanism, and (optionally) a correct-vs-scramble control on the
SAME seeds. Each scenario also gets a fresh same-harness baseline (the neutral actor). The runner is batch-only:
it returns structured tables (per-cell / per-scenario / cross-scenario) and prints periodic progress — no
demo-convenience narration (that lives in :mod:`.demo`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from hymeko_rl.coin_delivery.actors.base import cooperative_scramble
from hymeko_rl.coin_delivery.monitoring.contact_certification import certify
from hymeko_rl.coin_delivery.monitoring.delivery import DeliveryMonitor
from hymeko_rl.coin_delivery.monitoring.mechanism import is_clean
from hymeko_rl.coin_delivery.provenance import git_commit
from hymeko_rl.coin_delivery.scenarios.config import ScenarioConfig
from hymeko_rl.coin_delivery.scenarios.registry import ScenarioRegistry
from hymeko_rl.train.coin_delivery_actor import Mechanism

_BASELINE_ACTOR = "neutral"


@dataclass(frozen=True)
class SeedOutcome:
    seed: int
    strict_delivery: bool
    scramble_delivery: "bool | None"
    mechanism: str
    contact_certified: bool
    fingertip_fraction: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CellResult:
    """Aggregated per-(scenario, actor) statistics over the paired seeds."""

    scenario_id: str
    actor: str
    n_seeds: int
    delivery_rate: float
    scramble_delivery_rate: "float | None"
    correct_minus_scramble: "float | None"
    contact_cert_rate: float
    clean_mechanism_rate: float
    per_seed: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["per_seed"] = [o.as_dict() for o in self.per_seed]
        return d


@dataclass(frozen=True)
class ExperimentReport:
    scenarios: list
    actors: list
    seeds: list
    max_steps: int
    commit: str
    scramble: bool
    cells: list
    per_scenario: dict
    cross_scenario: dict

    def to_dict(self) -> dict:
        return {"scenarios": self.scenarios, "actors": self.actors, "seeds": self.seeds,
                "max_steps": self.max_steps, "commit": self.commit, "scramble": self.scramble,
                "cells": [c.as_dict() for c in self.cells], "per_scenario": self.per_scenario,
                "cross_scenario": self.cross_scenario}


def _mean(xs: list) -> float:
    return round(float(np.mean(xs)), 4) if xs else 0.0


class ExperimentRunner:
    """Factorial evaluator. Holds ONE :class:`DeliveryMonitor` reused across every scenario/actor/seed."""

    def __init__(self, monitor: "DeliveryMonitor | None" = None) -> None:
        self._monitor = monitor or DeliveryMonitor()

    @property
    def monitor(self) -> DeliveryMonitor:
        return self._monitor

    def _eval_cell(self, cfg: ScenarioConfig, actor_name: str, seeds: list, max_steps: int,
                   scramble_fn: "Callable[[np.ndarray, int], np.ndarray] | None", env: Any) -> CellResult:
        actor = ScenarioRegistry.list_actors()[actor_name]
        outcomes: list[SeedOutcome] = []
        for sd in seeds:
            res = self._monitor.evaluate(env, sd, actor.delivery_actor, max_steps=max_steps)
            cert = certify(cfg.contact_policy, env, sd, actor.delivery_actor, max_steps=max_steps)
            scr = None
            if scramble_fn is not None:
                scr = self._monitor.evaluate(env, sd, actor.delivery_actor, max_steps=max_steps,
                                             scramble=scramble_fn).strict_delivery
            outcomes.append(SeedOutcome(int(sd), res.strict_delivery, scr, res.mechanism,
                                        cert.certified, round(float(res.fingertip_fraction), 3)))
        return _aggregate_cell(cfg.scenario_id, actor_name, outcomes)

    def run(self, scenario_ids: list, actor_names: list, seeds: list, *, scramble: bool = True,
            max_steps: int = 60) -> ExperimentReport:
        """Evaluate the full factorial. # Preconditions ids are registered; ``seeds`` is a paired-seed list."""
        full_ids = [ScenarioRegistry.get(s).scenario_id for s in scenario_ids]
        scramble_fn = cooperative_scramble if scramble else None
        cells: list[CellResult] = []
        per_scenario: dict = {}
        for sid, full in zip(scenario_ids, full_ids):
            cfg = ScenarioRegistry.get(sid)
            env = cfg.kinematic_variant.build_env()
            scen_cells = [self._eval_cell(cfg, an, seeds, max_steps, scramble_fn, env) for an in actor_names]
            baseline = self._eval_cell(cfg, _BASELINE_ACTOR, seeds, max_steps, None, env)
            cells.extend(scen_cells)
            per_scenario[full] = _scenario_summary(scen_cells, baseline)
            print(f"  [factorial] {full}: delivery_by_actor="
                  f"{per_scenario[full]['delivery_by_actor']} baseline={baseline.delivery_rate}", flush=True)
        return ExperimentReport(full_ids, list(actor_names), list(seeds), max_steps, git_commit(),
                                scramble, cells, per_scenario, _cross_scenario(per_scenario))


def _aggregate_cell(scenario_id: str, actor_name: str, outcomes: list) -> CellResult:
    n = len(outcomes)
    scr_vals = [o.scramble_delivery for o in outcomes if o.scramble_delivery is not None]
    scr_rate = _mean(scr_vals) if scr_vals else None
    deliv = _mean([o.strict_delivery for o in outcomes])
    cvs = round(deliv - scr_rate, 4) if scr_rate is not None else None
    return CellResult(scenario_id, actor_name, n, deliv, scr_rate, cvs,
                      _mean([o.contact_certified for o in outcomes]),
                      _mean([is_clean(Mechanism(o.mechanism)) for o in outcomes]), outcomes)


def _scenario_summary(scen_cells: list, baseline: CellResult) -> dict:
    return {"delivery_by_actor": {c.actor: c.delivery_rate for c in scen_cells},
            "contact_cert_by_actor": {c.actor: c.contact_cert_rate for c in scen_cells},
            "correct_minus_scramble_by_actor": {c.actor: c.correct_minus_scramble for c in scen_cells},
            "baseline_actor": _BASELINE_ACTOR, "baseline_delivery": baseline.delivery_rate,
            "best_actor": max(scen_cells, key=lambda c: c.delivery_rate).actor if scen_cells else None,
            "best_delivery": max((c.delivery_rate for c in scen_cells), default=0.0)}


def _cross_scenario(per_scenario: dict) -> dict:
    best = {s: v["best_delivery"] for s, v in per_scenario.items()}
    return {"best_delivery_by_scenario": best,
            "mean_best_delivery": round(float(np.mean(list(best.values()))), 4) if best else 0.0,
            "baseline_by_scenario": {s: v["baseline_delivery"] for s, v in per_scenario.items()}}
