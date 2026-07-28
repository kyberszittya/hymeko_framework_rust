"""R11.3 coin/target demonstration bank — certificate-gated demonstration *generation* (no BC/RL/refinement).

For each coin/target scenario, the pipeline runs EXACT_ZERO_HOME -> IC certificate -> admissibility -> deployed RRT reach
-> precontact handoff -> per-instance CEM teacher -> strict K6 / classified failure, and records the full provenance +
measured energy ledger. The CEM capture is a training teacher only (labelled in provenance); RRT is the deployed planner.
"""
from __future__ import annotations

from hymeko_rl.coin_delivery.demo_bank.failure_class import (
    ClassifyThresholds,
    FailureClass,
    RolloutSignals,
    classify,
)
from hymeko_rl.coin_delivery.demo_bank.pipeline import PipelineConfig, run_scenario
from hymeko_rl.coin_delivery.demo_bank.record import SUCCESS_LABEL, DemonstrationRecord, replay_matches
from hymeko_rl.coin_delivery.demo_bank.scenario import (
    CANONICAL_COIN,
    CoinTargetScenario,
    ScenarioSplit,
    build_bank_scenarios,
    build_pilot_scenarios,
    curriculum_stage,
    split_ids,
)
from hymeko_rl.coin_delivery.demo_bank.store import BankSummary, DemonstrationBank

__all__ = [
    "FailureClass",
    "ClassifyThresholds",
    "RolloutSignals",
    "classify",
    "CoinTargetScenario",
    "ScenarioSplit",
    "CANONICAL_COIN",
    "build_pilot_scenarios",
    "build_bank_scenarios",
    "curriculum_stage",
    "split_ids",
    "DemonstrationRecord",
    "SUCCESS_LABEL",
    "replay_matches",
    "DemonstrationBank",
    "BankSummary",
    "PipelineConfig",
    "run_scenario",
]
