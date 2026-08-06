"""Central registry of coin-delivery scenarios and actors — the single source of scenario ids.

Scenarios are the 2×2 factorial of contact policy {C0 continuous, C1 intermittent} × embodiment {K0 canonical,
K1 canonical+1 distal DOF}. Scripts resolve scenarios by id (full or short alias) through :class:`ScenarioRegistry`
rather than hardcoding the construction, so a new scenario is added in ONE place.
"""
from __future__ import annotations

from hymeko_rl.coin_delivery.actors.base import DeliveryActorSemantic, build_actor_registry
from hymeko_rl.coin_delivery.scenarios.config import ScenarioConfig
from hymeko_rl.coin_delivery.scenarios.contact_policy import C0Continuous, C1Intermittent
from hymeko_rl.coin_delivery.scenarios.kinematic_variant import K0Canonical, K1DistalOrientation

_C0, _C1 = C0Continuous(), C1Intermittent()
_K0, _K1 = K0Canonical(), K1DistalOrientation()

_SCENARIOS: dict[str, ScenarioConfig] = {
    s.scenario_id: s for s in (
        ScenarioConfig("coin_delivery.c0k0_continuous_canonical", _C0, _K0,
                       "continuous grip on the canonical arm"),
        ScenarioConfig("coin_delivery.c1k0_intermittent_canonical", _C1, _K0,
                       "intermittent contact (impulse→free→recontact) on the canonical arm"),
        ScenarioConfig("coin_delivery.c0k1_continuous_plus1dof", _C0, _K1,
                       "continuous grip on the canonical arm + 1 distal pad-orientation DOF per fingertip"),
        ScenarioConfig("coin_delivery.c1k1_intermittent_plus1dof", _C1, _K1,
                       "intermittent contact on the canonical arm + 1 distal pad-orientation DOF per fingertip"),
    )
}

_ALIASES: dict[str, str] = {
    "c0k0": "coin_delivery.c0k0_continuous_canonical",
    "c1k0": "coin_delivery.c1k0_intermittent_canonical",
    "c0k1": "coin_delivery.c0k1_continuous_plus1dof",
    "c1k1": "coin_delivery.c1k1_intermittent_plus1dof",
}


class ScenarioRegistry:
    """Immutable lookup over the registered scenarios + actors."""

    @staticmethod
    def list_scenarios() -> list[str]:
        """The full scenario ids, in registration order."""
        return list(_SCENARIOS)

    @staticmethod
    def aliases() -> dict[str, str]:
        """Short alias → full scenario id."""
        return dict(_ALIASES)

    @staticmethod
    def get(scenario_id: str) -> ScenarioConfig:
        """Resolve a full id or a short alias to its :class:`ScenarioConfig`.

        # Preconditions ``scenario_id`` is a registered id or alias. # Raises ``KeyError`` otherwise."""
        full = _ALIASES.get(scenario_id, scenario_id)
        if full not in _SCENARIOS:
            raise KeyError(f"unknown scenario {scenario_id!r}; known: {ScenarioRegistry.list_scenarios()}")
        return _SCENARIOS[full]

    @staticmethod
    def list_actors() -> dict[str, DeliveryActorSemantic]:
        """The named semantic actors (wrapping the frozen delivery actors)."""
        return build_actor_registry()
