"""Named-predicate registry for HTL.

A ``ScalarPred`` carries a *name* that is resolved against this registry at
evaluation time.  Default predicates pull from ``event.scalar_signals``;
users can register custom callables that read from ``event.hypergraph``
(per-cycle balance, shell-dominance maps, alpha_k slots, ...).

Usage::

    from htl.predicates import register, signal_value

    @register("val_auc")
    def _val_auc(event):
        return event.scalar_signals["val_auc"]
"""

from __future__ import annotations

from typing import Callable, Dict

from .event import HypergraphEvent

PredicateFn = Callable[[HypergraphEvent], float]

_REGISTRY: Dict[str, PredicateFn] = {}


def register(name: str) -> Callable[[PredicateFn], PredicateFn]:
    """Decorator: register ``fn`` under ``name`` in the global registry."""

    def _decorate(fn: PredicateFn) -> PredicateFn:
        _REGISTRY[name] = fn
        return fn

    return _decorate


def signal_value(event: HypergraphEvent, name: str) -> float:
    """Resolve ``name`` to a scalar via the registry, falling back to
    ``event.scalar_signals[name]``.

    Preconditions
    -------------
    - ``name`` is registered OR present in ``event.scalar_signals``.

    Raises
    ------
    KeyError
        If ``name`` is unknown and missing from ``event.scalar_signals``.
    """

    fn = _REGISTRY.get(name)
    if fn is not None:
        return float(fn(event))
    if name in event.scalar_signals:
        return float(event.scalar_signals[name])
    raise KeyError(
        f"unknown HTL signal {name!r}: not in registry, not in event.scalar_signals"
    )


def registered_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Test-only: wipe the registry (re-registers are responsibility of caller)."""
    _REGISTRY.clear()


__all__ = ["register", "signal_value", "registered_names", "clear_registry", "PredicateFn"]
