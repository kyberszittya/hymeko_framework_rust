"""Small shared leaf utilities for building genuinely-immutable value types.

Kept dependency-free (stdlib only) so every ``hymeko_control`` module can import
it without creating a cycle or pulling a heavy dependency.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


def freeze_mapping(items: Mapping[_K, _V] | None) -> Mapping[_K, _V]:
    """Return a read-only view over a *copy* of ``items``.

    # Preconditions
    ``items`` is a mapping or ``None``.

    # Postconditions
    The result is a ``MappingProxyType`` over an independent ``dict`` copy;
    mutating the original argument afterwards does not affect the returned view,
    and the view itself raises ``TypeError`` on item assignment. This is how the
    CIP-0 value types guarantee "no hidden state modification".
    """
    return MappingProxyType(dict(items) if items is not None else {})
