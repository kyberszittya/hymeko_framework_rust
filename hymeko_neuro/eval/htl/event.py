"""HypergraphEvent: the unit observed by an HTL monitor.

An event carries:
- a logical timestamp (``t``) — typically epoch index for training-time monitors
- ``scalar_signals`` — a dict of named scalars (``val_auc``, ``alpha[c5]``, ...)
- ``hypergraph`` — an opaque reference to a hypergraph snapshot at time ``t``;
  predicates that need hypergraph state (per-cycle balance, shell dominance)
  pull from this field.  The monitor itself does not interpret it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class HypergraphEvent:
    t: float
    scalar_signals: Mapping[str, float] = field(default_factory=dict)
    hypergraph: Any = None


__all__ = ["HypergraphEvent"]
