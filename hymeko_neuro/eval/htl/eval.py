"""Robust-STL evaluator over a bounded event history.

The robustness function ``rho`` returns a real number whose sign is the
boolean truth value of the formula at ``t_now`` and whose magnitude is the
*slack* — how far the signal is from violating the bound.

Semantics (plan §3):

- rho(ScalarPred(n, op, v), s, t):
    op = >  or >=  :  s.signal(n, t) - v
    op = <  or <=  :  v - s.signal(n, t)
    op = ==        :  -|s.signal(n, t) - v|
- rho(Not(psi))           = -rho(psi)
- rho(And(psi1, psi2))    = min(rho(psi1), rho(psi2))
- rho(Or(psi1, psi2))     = max(rho(psi1), rho(psi2))
- rho(G_{[a,b]}(psi))     = inf  over t' in [t_now+a, t_now+b] (clamped to history)
- rho(F_{[a,b]}(psi))     = sup  over t' in [t_now+a, t_now+b] (clamped to history)

For training-time use, the interval is interpreted as *past* events: the
monitor maintains a history with ``t`` increasing forward in time, and
``G[0, inf]`` reduces to inf over the entire observed past.  This matches
how robust-STL is typically used for online runtime monitoring (rather than
look-ahead prediction).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence

from .ast import And, CmpOp, Eventually, Globally, HtlNode, Not, Or, ScalarPred
from .event import HypergraphEvent
from .parser import parse as parse_formula
from .predicates import signal_value


def _scalar_robustness(node: ScalarPred, event: HypergraphEvent) -> float:
    x = signal_value(event, node.name)
    v = node.value
    if node.op in (CmpOp.GT, CmpOp.GE):
        return x - v
    if node.op in (CmpOp.LT, CmpOp.LE):
        return v - x
    if node.op is CmpOp.EQ:
        return -abs(x - v)
    raise ValueError(f"unknown CmpOp: {node.op}")


def robustness_at(node: HtlNode, event: HypergraphEvent) -> float:
    """Robustness of ``node`` evaluated at a *single* event (no temporal
    quantifiers around the call site).

    Used by both the bounded-interval ``Globally`` / ``Eventually`` reduction
    and as the leaf evaluator for non-temporal sub-formulas.
    """
    if isinstance(node, ScalarPred):
        return _scalar_robustness(node, event)
    if isinstance(node, Not):
        return -robustness_at(node.inner, event)
    if isinstance(node, And):
        return min(robustness_at(node.left, event), robustness_at(node.right, event))
    if isinstance(node, Or):
        return max(robustness_at(node.left, event), robustness_at(node.right, event))
    # Temporal at a single event: collapse to inner.  This matches the
    # convention that a singleton history evaluates G/F as the leaf value.
    if isinstance(node, (Globally, Eventually)):
        return robustness_at(node.inner, event)
    raise TypeError(f"unsupported HTL node type: {type(node).__name__}")


def _events_in_interval(
    history: Sequence[HypergraphEvent], t_now: float, t1: float, t2: float
) -> List[HypergraphEvent]:
    """Past-interval semantics: select events with t in [t_now - t2, t_now - t1].

    The plan writes G_{[t1, t2]} relative to ``t``; for online monitoring
    we interpret the interval as a *backward* look from ``t_now``: ``t1``
    is the most recent offset, ``t2`` is the oldest.  ``[0, inf)`` therefore
    means "the whole observed past, including now".
    """
    lo = t_now - t2 if math.isfinite(t2) else -math.inf
    hi = t_now - t1
    return [e for e in history if lo <= e.t <= hi]


def robustness(
    node: HtlNode,
    history: Sequence[HypergraphEvent],
    t_now: Optional[float] = None,
) -> float:
    """Robustness of ``node`` against ``history`` at time ``t_now``.

    Preconditions
    -------------
    - ``history`` is non-empty.
    - ``history`` is sorted by ``t`` ascending (the monitor maintains this).

    If ``t_now`` is None, the timestamp of the last event in history is used.

    Returns
    -------
    float
        Robustness in (-inf, +inf).  Satisfied iff > 0.
    """

    if not history:
        raise ValueError("history must be non-empty")
    if t_now is None:
        t_now = history[-1].t

    if isinstance(node, ScalarPred):
        return _scalar_robustness(node, _event_at_or_before(history, t_now))
    if isinstance(node, Not):
        return -robustness(node.inner, history, t_now)
    if isinstance(node, And):
        return min(
            robustness(node.left, history, t_now),
            robustness(node.right, history, t_now),
        )
    if isinstance(node, Or):
        return max(
            robustness(node.left, history, t_now),
            robustness(node.right, history, t_now),
        )
    if isinstance(node, Globally):
        evts = _events_in_interval(history, t_now, node.t1, node.t2)
        if not evts:
            # No events in the interval — by convention, return +inf (vacuously true).
            return math.inf
        return min(robustness(node.inner, history, e.t) for e in evts)
    if isinstance(node, Eventually):
        evts = _events_in_interval(history, t_now, node.t1, node.t2)
        if not evts:
            return -math.inf
        return max(robustness(node.inner, history, e.t) for e in evts)
    raise TypeError(f"unsupported HTL node type: {type(node).__name__}")


def _event_at_or_before(
    history: Sequence[HypergraphEvent], t_now: float
) -> HypergraphEvent:
    candidates = [e for e in history if e.t <= t_now]
    if not candidates:
        # Fall back to the earliest event; we promised history is non-empty.
        return history[0]
    return candidates[-1]


def satisfied(rho: float) -> bool:
    """Boolean satisfaction: ``rho > 0``."""
    return rho > 0.0


class HtlMonitor:
    """Online HTL monitor with a bounded ring-buffer history.

    Usage::

        mon = HtlMonitor("G(val_auc > 0.85)", horizon=256)
        for epoch in range(n_epochs):
            ...
            mon.observe(HypergraphEvent(t=float(epoch),
                                        scalar_signals={"val_auc": auc}))
            print(epoch, mon.robustness(), mon.satisfied())

    Parameters
    ----------
    formula : str | HtlNode
        Either a formula string (parsed once at construction time) or a
        pre-built AST.
    horizon : int
        Maximum number of events retained.  Older events are evicted.
    """

    def __init__(self, formula: "str | HtlNode", horizon: int = 1024) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        self._node: HtlNode = parse_formula(formula) if isinstance(formula, str) else formula
        self._history: Deque[HypergraphEvent] = deque(maxlen=horizon)

    @property
    def history(self) -> List[HypergraphEvent]:
        return list(self._history)

    @property
    def node(self) -> HtlNode:
        return self._node

    def observe(self, event: HypergraphEvent) -> float:
        """Append ``event`` and return the current robustness."""
        if self._history and event.t < self._history[-1].t:
            raise ValueError(
                f"events must arrive in non-decreasing time order: "
                f"got t={event.t} after t={self._history[-1].t}"
            )
        self._history.append(event)
        return self.robustness()

    def observe_many(self, events: Iterable[HypergraphEvent]) -> List[float]:
        return [self.observe(e) for e in events]

    def robustness(self) -> float:
        return robustness(self._node, list(self._history))

    def satisfied(self) -> bool:
        return satisfied(self.robustness())

    def reset(self) -> None:
        self._history.clear()


__all__ = [
    "robustness",
    "robustness_at",
    "satisfied",
    "HtlMonitor",
]
