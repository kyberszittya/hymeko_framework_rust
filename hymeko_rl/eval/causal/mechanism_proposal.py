"""Deterministic mechanism *proposal* — LiNGAM-SH step 2 (spec §6, discovery-lite; no factorization).

Turns existing pairwise causal evidence + known task/reward/monitor structure into candidate
:class:`~hymeko_rl.eval.causal.hymeko_emit.Mechanism` hyperedges, **deterministically** (no search, no matrix
factorization). Three proposal sources:

* **common-child** — several variables pointing to the same child in a pairwise DAG become one multi-input
  mechanism ``{parents} → {child}`` (the coffee-push ``{near_fraction, progress_score} → {total_reward}`` case);
* **reward-terms** — an explicit set of reward-driving variables becomes ``{terms} → {total_reward}``;
* **monitor-contract** — an explicit set of monitor sub-variables becomes ``{monitor_terms} → {monitor_pass}``.

A proposal is a *hypothesis* (framework doctrine: DirectLiNGAM proposes, HyMeKo groups, ablation decides). The
scoring is deterministic and intentionally simple (median/mean magnitudes, summed-sign, coverage-ratio confidence);
the ``B ≈ A_out Σ A_inᵀ`` factorization, stochastic search, and intervention logic are out of scope for this step.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .hymeko_emit import CausalHypergraph, Mechanism


@dataclass(frozen=True)
class MechanismProposal:
    """A proposed directed mechanism hyperedge with deterministic scoring + provenance.

    ``confidence`` is a coverage ratio in ``[0, 1]`` (edge count / possible-edge count) for evidence-derived
    proposals, and ``1.0`` for explicit reward/monitor-contract proposals. ``source`` is one of
    ``"common_child" | "reward_terms" | "monitor_contract"``.
    """

    name: str
    tail: tuple[str, ...]
    head: tuple[str, ...]
    strength: float
    sign: int
    confidence: float
    source: str
    evidence: Mapping[str, Any]

    def to_mechanism(self) -> Mechanism:
        """Materialize this proposal as a :class:`Mechanism` (evidence copied into the mechanism)."""
        return Mechanism(name=self.name, tail=self.tail, head=self.head, strength=self.strength,
                         sign=self.sign, evidence=dict(self.evidence))


def _aggregate_weights(weights: Sequence[float]) -> tuple[float, int]:
    """Deterministic (strength, sign) from supporting weights: strength = mean ``|w|``, sign = ``sign(Σ w)``."""
    arr = np.asarray(list(weights), dtype=np.float64)
    if arr.size == 0:
        return 1.0, 1
    strength = float(np.mean(np.abs(arr)))
    sign = int(np.sign(float(np.sum(arr)))) or 1        # tie / all-zero → +1
    return strength, sign


def propose_common_child(edges: "Sequence[tuple[str, str, float]]", *, min_parents: int = 2,
                         name_prefix: str = "mech_to") -> list[MechanismProposal]:
    """Group pairwise ``(cause, effect, weight)`` edges by child → one ``{parents} → {child}`` mechanism per child.

    A child with fewer than ``min_parents`` parents is **not** grouped (a single parent does not become a
    multi-input mechanism unless ``min_parents <= 1`` is explicitly passed). Deterministic: children and parents
    are sorted by name. # Preconditions ``min_parents >= 1``.
    """
    if min_parents < 1:
        raise ValueError(f"min_parents must be >= 1, got {min_parents}")
    by_child: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for cause, effect, weight in edges:
        by_child[effect].append((cause, float(weight)))
    all_vars = {v for c, e, _w in edges for v in (c, e)}
    denom = max(1, len(all_vars) - 1)
    proposals: list[MechanismProposal] = []
    for child in sorted(by_child):
        parents = sorted(by_child[child])               # (parent, weight) sorted by parent then weight
        if len(parents) < min_parents:
            continue
        tail = tuple(p for p, _w in parents)
        strength, sign = _aggregate_weights([w for _p, w in parents])
        proposals.append(MechanismProposal(
            name=f"{name_prefix}_{child}", tail=tail, head=(child,), strength=round(strength, 6), sign=sign,
            confidence=round(min(1.0, len(tail) / denom), 6), source="common_child",
            evidence={"child": child, "supporting_edges": [[p, child, round(w, 6)] for p, w in parents]}))
    return proposals


def _explicit_proposal(tail_vars: "Sequence[str]", output: str, *, weights: "Mapping[str, float] | None",
                       source: str, name: str) -> MechanismProposal:
    """Shared builder for the explicit (contract-given) reward/monitor proposals — confidence is 1.0."""
    tail = tuple(dict.fromkeys(tail_vars))              # de-dup, preserve order
    if not tail:
        raise ValueError(f"{source} proposal needs at least one tail variable")
    if weights is not None:
        strength, sign = _aggregate_weights([float(weights[t]) for t in tail])
    else:
        strength, sign = 1.0, 1
    return MechanismProposal(name=name, tail=tail, head=(output,), strength=round(strength, 6), sign=sign,
                             confidence=1.0, source=source,
                             evidence={"tail": list(tail), "output": output, "explicit": True})


def propose_reward_terms(terms: "Sequence[str]", output: str = "total_reward", *,
                         weights: "Mapping[str, float] | None" = None,
                         name: str = "reward_mechanism") -> MechanismProposal:
    """An explicit reward mechanism ``{terms} → {output}`` (confidence 1.0; the reward IS this term set)."""
    return _explicit_proposal(terms, output, weights=weights, source="reward_terms", name=name)


def propose_monitor_contract(monitor_terms: "Sequence[str]", output: str = "monitor_pass", *,
                             weights: "Mapping[str, float] | None" = None,
                             name: str = "monitor_mechanism") -> MechanismProposal:
    """An explicit monitor-contract mechanism ``{monitor_terms} → {output}`` (confidence 1.0)."""
    return _explicit_proposal(monitor_terms, output, weights=weights, source="monitor_contract", name=name)


def proposals_to_causal_hypergraph(variables: "Sequence[str]", proposals: "Sequence[MechanismProposal]", *,
                                   name: str) -> CausalHypergraph:
    """Assemble proposals into a mechanism-form :class:`CausalHypergraph`.

    The variable set is ``variables`` unioned with every tail/head referenced by a proposal (so a caller cannot
    accidentally omit a referenced variable). # Postconditions the returned graph is mechanism-form; validate it
    with :meth:`CausalHypergraph.check_acyclicity` / :func:`cross_view_verify` before trusting it.
    """
    referenced = [v for p in proposals for v in (*p.tail, *p.head)]
    all_vars = list(dict.fromkeys([*variables, *referenced]))
    return CausalHypergraph(name=name, variables=all_vars, edges=[],
                            mechanisms=[p.to_mechanism() for p in proposals])
