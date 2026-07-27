"""CIP-0 --- measured authority.

Authority answers "how much control power does the embodiment actually have,
right now, in this mode?" -- e.g. available grasp force, support margin,
reachable yaw rate. Each channel carries a *provenance*: whether it was
observed, modelled, or assumed, and a human-readable note. The runtime refuses
an authority map with any un-sourced channel, so a decoder can never silently
rely on a fabricated capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .._frozen import freeze_mapping


class AuthoritySource(str, Enum):
    """Where an authority value came from (most to least trustworthy)."""

    OBSERVED = "observed"
    MODELED = "modeled"
    ASSUMED = "assumed"


class AuthorityProvenanceError(ValueError):
    """Raised when an authority channel lacks a source or provenance note."""


@dataclass(frozen=True)
class AuthorityChannel:
    """One measured authority: a value plus where it came from."""

    name: str
    value: float
    source: AuthoritySource
    provenance: str

    def __post_init__(self) -> None:
        if not self.provenance:
            raise AuthorityProvenanceError(
                f"authority {self.name!r}: empty provenance note"
            )


@dataclass(frozen=True)
class AuthorityMap:
    """A named collection of authority channels.

    # Invariants
    Frozen; ``require_provenance`` holds for any instance whose channels each
    carry a non-empty provenance note (enforced at :class:`AuthorityChannel`
    construction, re-checked here for maps built from raw dicts).
    """

    channels: Mapping[str, AuthorityChannel] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", freeze_mapping(self.channels))

    def value(self, name: str) -> float:
        return self.channels[name].value

    def provenance(self, name: str) -> str:
        return self.channels[name].provenance

    def require_provenance(self) -> None:
        """Assert every channel is sourced and annotated.

        # Postconditions
        Returns ``None`` if valid; raises :class:`AuthorityProvenanceError`
        naming the first offending channel otherwise.
        """
        for name, ch in self.channels.items():
            if not ch.provenance:
                raise AuthorityProvenanceError(f"authority {name!r}: missing provenance")
            if not isinstance(ch.source, AuthoritySource):
                raise AuthorityProvenanceError(f"authority {name!r}: invalid source")
