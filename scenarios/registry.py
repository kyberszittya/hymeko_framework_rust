"""Discoverable multi-embodiment registry over the CIP-0 scenario adapters.

Until now the multi-embodiment surface was convention-only: each ``scenarios/<x>/`` package standalone-loaded its own
``cip_*.hymeko.yaml`` and implemented :class:`hymeko_control.cip.protocol.CIP0Adapter`, but nothing enumerated the set.
This registry is that enumeration — the single place that lists every embodiment speaking the CIP-0 contract, its
declarative :class:`~hymeko_control.language.ir.ControlModel` loader, its CIP qualification tag, and its measured status.

Design contract:

* **Torch-free / lazy.** Entries hold a *loader* (:data:`EmbodimentEntry.load_model`), never a constructed adapter or
  MuJoCo env, so ``import scenarios.registry`` and discovery cost nothing heavy. Only :meth:`EmbodimentEntry.model`
  actually parses+validates the YAML into a ``ControlModel`` (still torch-free); adapter/env construction stays in the
  scenario packages.
* **Dependency direction preserved.** This module lives in the *consumer* layer (``scenarios``) and depends on the
  scenario packages + the ``hymeko_control`` core — never the reverse. The core (``hymeko_control``) does not import it.
* **Honest status.** ``PENDING`` embodiments (e.g. coin, whose runtime lives in a separate tree and has no CIP-0 adapter
  yet) are listed with no loadable model, so the roadmap is discoverable without faking an integration.

Preconditions: the scenario packages import cleanly (torch-free YAML path). Postconditions: an immutable lookup keyed by
both the short embodiment name (``"aibo"``) and the scenario id (``"CIP-AIBO-01"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from hymeko_control.language.ir import ControlModel

from . import aibo, humanoid, pick_place


class EmbodimentStatus(Enum):
    """Qualification state of an embodiment against the CIP-0 contract.

    ``CERTIFIED`` — passes its certificate suite and carries a release tag.
    ``PRESENT_UNTAGGED`` — a working adapter + conformance-green, but the headline qualification is blocked
    (e.g. needs a floating base / robust bearing) so no tag is claimed.
    ``PENDING`` — declared as an embodiment of the program but not yet a CIP-0 consumer (no adapter/model here).
    """

    CERTIFIED = "certified"
    PRESENT_UNTAGGED = "present_untagged"
    PENDING = "pending"


@dataclass(frozen=True)
class EmbodimentEntry:
    """One embodiment's registration. ``load_model is None`` iff ``status is PENDING``."""

    embodiment: str
    scenario_id: str
    title: str
    status: EmbodimentStatus
    measured: str
    load_model: Optional[Callable[[], ControlModel]] = None
    spec_path: Optional[Path] = None
    cip_tag: Optional[str] = None

    def __post_init__(self) -> None:
        # Invariant: a loadable model exists iff the embodiment is not PENDING.
        has_loader = self.load_model is not None
        is_pending = self.status is EmbodimentStatus.PENDING
        assert has_loader != is_pending, (
            f"{self.scenario_id}: PENDING must have no loader, non-PENDING must have one"
        )
        assert (self.cip_tag is not None) == (self.status is EmbodimentStatus.CERTIFIED), (
            f"{self.scenario_id}: a cip_tag is present iff the status is CERTIFIED"
        )

    def model(self) -> ControlModel:
        """Parse+validate this embodiment's declarative CIP-0 contract into a ``ControlModel``.

        # Raises ``LookupError`` if the embodiment is PENDING (no adapter/model registered yet)."""
        if self.load_model is None:
            raise LookupError(
                f"{self.scenario_id} is PENDING — no CIP-0 model registered "
                f"(measured: {self.measured})"
            )
        return self.load_model()


def _coin_load_model() -> ControlModel:
    """CIP-COIN-00's ControlModel, lazily built from the coin repo's adapter (``hymeko_rl.coin_delivery.cip``) — its
    runtime lives in a separate tree (hymeko/rl), so it is imported on demand (put that repo on PYTHONPATH), mirroring
    how the aibo adapter lazily imports its env. Torch-free: only the schema-v0 ControlModel dict + this framework's
    validator are touched. # Raises ``ImportError`` if the coin runtime tree is not importable."""
    from hymeko_rl.coin_delivery.cip.coin_adapter import load_model as _coin
    return _coin()


_ENTRIES: tuple[EmbodimentEntry, ...] = (
    EmbodimentEntry(
        embodiment="pick_place",
        scenario_id="CIP-PNP-01",
        title="Two-finger pick-and-place (PNP-4)",
        status=EmbodimentStatus.CERTIFIED,
        measured="PNP-4 externally certified (simulation)",
        load_model=pick_place.load_model,
        spec_path=pick_place.SPEC_PATH,
        cip_tag="cip-pick-place-v0",
    ),
    EmbodimentEntry(
        embodiment="aibo",
        scenario_id="CIP-AIBO-01",
        title="ERS-1000 quadruped (22-DOF, AIBO-2)",
        status=EmbodimentStatus.PRESENT_UNTAGGED,
        measured="wide-bearing reach 0.50->0.786 (stab scaffold); robust ~0.89 held-out scripted; AIBO-3/4 blocked",
        load_model=aibo.load_model,
        spec_path=aibo.SPEC_PATH,
    ),
    EmbodimentEntry(
        embodiment="humanoid",
        scenario_id="CIP-HUM-01",
        title="Humanoid (18-DOF, HUM-1)",
        status=EmbodimentStatus.PRESENT_UNTAGGED,
        measured="peak 0.83 m/s gait but indefinite walk not achieved (speed<->stability dial); HUM-2/3/4 blocked",
        load_model=humanoid.load_model,
        spec_path=humanoid.SPEC_PATH,
    ),
    EmbodimentEntry(
        embodiment="coin",
        scenario_id="CIP-COIN-00",
        title="Two-arm coin delivery (Reference Scenario 0)",
        status=EmbodimentStatus.PRESENT_UNTAGGED,
        measured="HOME->K6 46/55 (adaptive brake, supersedes 44/55). CIP-COIN-00 adapter is a consumer: CoinCIPAdapter "
                 "delegates to the deployed coin runtime (hymeko_rl_standalone, separate tree); parity-green.",
        load_model=_coin_load_model,
    ),
)


class EmbodimentRegistry:
    """Immutable lookup over the registered CIP-0 embodiments (by short name or scenario id)."""

    _BY_KEY: dict[str, EmbodimentEntry] = {}
    for _e in _ENTRIES:
        _BY_KEY[_e.embodiment] = _e
        _BY_KEY[_e.scenario_id] = _e
    del _e

    @staticmethod
    def list_embodiments() -> tuple[str, ...]:
        """The short embodiment names, in registration order."""
        return tuple(e.embodiment for e in _ENTRIES)

    @staticmethod
    def entries() -> tuple[EmbodimentEntry, ...]:
        """All entries, in registration order."""
        return _ENTRIES

    @staticmethod
    def get(key: str) -> EmbodimentEntry:
        """Resolve a short embodiment name or a scenario id to its entry.

        # Preconditions ``key`` is a registered embodiment or scenario id. # Raises ``KeyError`` otherwise."""
        try:
            return EmbodimentRegistry._BY_KEY[key]
        except KeyError:
            raise KeyError(
                f"unknown embodiment {key!r}; known: {EmbodimentRegistry.list_embodiments()}"
            ) from None

    @staticmethod
    def integrated() -> tuple[EmbodimentEntry, ...]:
        """Embodiments that are actual CIP-0 consumers (a loadable model exists — not PENDING)."""
        return tuple(e for e in _ENTRIES if e.status is not EmbodimentStatus.PENDING)

    @staticmethod
    def certified() -> tuple[EmbodimentEntry, ...]:
        """Embodiments that pass their certificate suite and carry a release tag."""
        return tuple(e for e in _ENTRIES if e.status is EmbodimentStatus.CERTIFIED)


__all__ = ["EmbodimentStatus", "EmbodimentEntry", "EmbodimentRegistry"]
