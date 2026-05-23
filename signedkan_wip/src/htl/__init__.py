"""Hypergraph Temporal Logic (HTL).

Pure-Python implementation of a robust-STL monitor specialised for
training-time observation of hypergraph neural networks.  See
``docs/plans/2026-05-21-htl-python-impl/plan.tex`` for the design.

Quick start
-----------

>>> from htl import HtlMonitor, HypergraphEvent
>>> mon = HtlMonitor("G(val_auc > 0.85)", horizon=256)
>>> mon.observe(HypergraphEvent(t=0.0, scalar_signals={"val_auc": 0.88}))
0.030000000000000027
>>> mon.satisfied()
True
"""

from .ast import (
    And,
    CmpOp,
    Eventually,
    Globally,
    HtlNode,
    Not,
    Or,
    ScalarPred,
)
from .eval import HtlMonitor, robustness, robustness_at, satisfied
from .event import HypergraphEvent
from .parser import ParseError, parse
from .predicates import (
    PredicateFn,
    clear_registry,
    register,
    registered_names,
    signal_value,
)

__all__ = [
    # AST
    "CmpOp",
    "ScalarPred",
    "Not",
    "And",
    "Or",
    "Globally",
    "Eventually",
    "HtlNode",
    # parser
    "parse",
    "ParseError",
    # event
    "HypergraphEvent",
    # eval
    "robustness",
    "robustness_at",
    "satisfied",
    "HtlMonitor",
    # predicates
    "register",
    "signal_value",
    "registered_names",
    "clear_registry",
    "PredicateFn",
]
