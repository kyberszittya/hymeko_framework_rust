"""HTL AST node dataclasses.

The AST is the parser's output and the evaluator's input.  Every node type
that the parser can emit is enumerated here; the evaluator dispatches on
``type(node)`` via :func:`htl.eval.robustness`.

Robust-STL semantics (see ``docs/plans/2026-05-21-htl-python-impl/plan.tex``):

- ``ScalarPred(name, op, value)`` — atomic predicate over a named signal
- ``Not(inner)`` / ``And(left, right)`` / ``Or(left, right)`` — boolean
- ``Globally(t1, t2, inner)`` — \\square_{[t1, t2]} (G operator)
- ``Eventually(t1, t2, inner)`` — \\diamond_{[t1, t2]} (F operator)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union


class CmpOp(str, Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="

    @staticmethod
    def parse(token: str) -> "CmpOp":
        for member in CmpOp:
            if member.value == token:
                return member
        raise ValueError(f"unknown comparison operator: {token!r}")


@dataclass(frozen=True)
class ScalarPred:
    """Atomic predicate: ``<signal_name> <op> <value>``.

    ``name`` is looked up in the predicate registry at evaluation time.
    """

    name: str
    op: CmpOp
    value: float


@dataclass(frozen=True)
class Not:
    inner: "HtlNode"


@dataclass(frozen=True)
class And:
    left: "HtlNode"
    right: "HtlNode"


@dataclass(frozen=True)
class Or:
    left: "HtlNode"
    right: "HtlNode"


@dataclass(frozen=True)
class Globally:
    """G_{[t1, t2]}(inner) — robustness = inf over the interval."""

    t1: float
    t2: float
    inner: "HtlNode"


@dataclass(frozen=True)
class Eventually:
    """F_{[t1, t2]}(inner) — robustness = sup over the interval."""

    t1: float
    t2: float
    inner: "HtlNode"


HtlNode = Union[ScalarPred, Not, And, Or, Globally, Eventually]


__all__ = [
    "CmpOp",
    "ScalarPred",
    "Not",
    "And",
    "Or",
    "Globally",
    "Eventually",
    "HtlNode",
]
