"""COIN-DELIVERY-DERIVED-SCENARIOS-1 — a reusable scenario framework for HyMeKo coin-delivery demonstrations.

Composed abstractions layered around the FROZEN ``hymeko_rl.train.coin_delivery_actor`` module (never modified):

* ``scenarios`` — contact policies (C0 continuous / C1 intermittent), kinematic embodiments (K0 canonical /
  K1 canonical + 1 distal pad-orientation DOF), the composed :class:`ScenarioConfig`, and the central
  :class:`ScenarioRegistry` (the single source of scenario ids).
* ``actors`` — the embodiment-independent :class:`SemanticCommand` + :class:`KinematicAdapter`, wrapping the
  frozen delivery actors so they run in the framework unchanged.
* ``monitoring`` — thin adapters over the frozen strict delivery monitor / attribution / mechanism, plus
  contact-continuity certification (kept SEPARATE from delivery success).
* ``runners`` — :class:`ExperimentRunner` (batch factorial) and :class:`DemoRunner` (one scenario × actor × seed).
"""
from __future__ import annotations

from hymeko_rl.coin_delivery.scenarios.config import ScenarioConfig
from hymeko_rl.coin_delivery.scenarios.registry import ScenarioRegistry

__all__ = ["ScenarioConfig", "ScenarioRegistry"]
