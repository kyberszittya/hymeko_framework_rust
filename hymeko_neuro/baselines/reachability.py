"""Reachability rules for the label-shuffle audit (and, by analogy, the
P-graph producibility closure — see
``docs/plans/2026-06-14-reachability-rules-audit-pgraph/argument.md``).

A *reachability rule* decides which edges and labels seed the message-passing
closure a signed-link model reaches at inference. The three rules form a monotone
lattice ``STRICT ⊆ TRANSDUCTIVE_TOPOLOGY ⊆ TRANSDUCTIVE_FULL``:

* **STRICT** — only training edges/signs are reachable; test-edge signs are, by
  construction, unreachable. Shuffling the train labels destroys the only signal
  → held-out AUROC collapses to chance.
* **TRANSDUCTIVE_TOPOLOGY** — training edges plus test-edge *topology*; the test
  edge is a routable connection but its sign is masked to ``NEUTRAL_SIGN``. A
  method that still leaks here is exploiting *structural* leakage.
* **TRANSDUCTIVE_FULL** — training edges plus test edges *with their signs* (how
  these baselines are evaluated in their own papers). The held-out sign is in the
  closure → a leak here may be the direct-label channel.

This module owns only the rule semantics (which edges/signs seed the closure);
the per-model adjacency construction consumes its output via the strategy's
``build_context``.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

# Sentinel sign for a topology-only test edge: reachable as a connection, sign
# withheld. Distinct from ±1 so a masked edge can never be mistaken for a label.
NEUTRAL_SIGN = 0


class ReachabilityRule(Enum):
    """Which edges/labels seed the inference-time message-passing closure."""

    STRICT = "strict"
    TRANSDUCTIVE_TOPOLOGY = "topo"
    TRANSDUCTIVE_FULL = "full"

    @classmethod
    def from_str(cls, s: str) -> "ReachabilityRule":
        """Parse a CLI token; raise ``ValueError`` naming the valid set."""
        try:
            return cls(s)
        except ValueError:
            raise ValueError(
                f"unknown reachability rule {s!r}; valid: {[r.value for r in cls]}"
            ) from None


def reachable_edges(
    rule: ReachabilityRule,
    e_tr: np.ndarray,
    s_tr: np.ndarray,
    e_te: np.ndarray,
    s_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Edges and signs that seed the message-passing closure under ``rule``.

    # Preconditions
    - ``e_tr``/``e_te`` are ``(E, 2)`` int; ``s_tr``/``s_te`` are ``(E,)`` in
      ``{+1, -1}``; per-array lengths match.

    # Postconditions
    - STRICT → exactly ``(e_tr, s_tr)`` (test edges unreachable).
    - TRANSDUCTIVE_TOPOLOGY → ``train ∪ test`` edges; train signs kept, test
      signs set to ``NEUTRAL_SIGN``.
    - TRANSDUCTIVE_FULL → ``train ∪ test`` edges; all true signs.

    # Invariants
    - No value of ``s_te`` appears in the result unless ``rule`` is
      TRANSDUCTIVE_FULL (the leakage guarantee the audit rests on).
    """
    if rule is ReachabilityRule.STRICT:
        return e_tr.copy(), s_tr.copy()

    edges = np.concatenate([e_tr, e_te], axis=0)
    if rule is ReachabilityRule.TRANSDUCTIVE_TOPOLOGY:
        test_signs = np.full(len(e_te), NEUTRAL_SIGN, dtype=s_tr.dtype)
    elif rule is ReachabilityRule.TRANSDUCTIVE_FULL:
        test_signs = s_te.astype(s_tr.dtype, copy=True)
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"unhandled rule {rule!r}")
    signs = np.concatenate([s_tr, test_signs])
    return edges, signs


def reachable_nodes(
    rule: ReachabilityRule,
    e_tr: np.ndarray,
    e_te: np.ndarray,
) -> set[int]:
    """Node set touched by the reachable edges under ``rule`` (lattice probe)."""
    edges, _ = reachable_edges(
        rule, e_tr, np.ones(len(e_tr), dtype=np.int64),
        e_te, np.ones(len(e_te), dtype=np.int64),
    )
    return set(int(v) for v in edges.reshape(-1))
