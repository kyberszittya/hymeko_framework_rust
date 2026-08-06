"""Test cases for the reachability-rule semantics (audit side).

These encode the claims of
``docs/plans/2026-06-14-reachability-rules-audit-pgraph/argument.md``:
reduction (STRICT = train-only), the monotone lattice
(STRICT ⊆ TOPOLOGY ⊆ FULL), and the leakage invariant (test signs are reachable
*only* under FULL). They are the audit-side half of the proposed article's
formal core.

Run: ``pytest -p no:randomly hymeko_neuro/tests/test_reachability.py``
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_neuro.baselines.reachability import (
    NEUTRAL_SIGN,
    ReachabilityRule,
    reachable_edges,
    reachable_nodes,
)

R = ReachabilityRule


def _fixture():
    e_tr = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    s_tr = np.array([1, -1, 1], dtype=np.int64)
    e_te = np.array([[3, 4], [4, 5]], dtype=np.int64)
    s_te = np.array([-1, -1], dtype=np.int64)
    return e_tr, s_tr, e_te, s_te


# ── reduction: STRICT is exactly the training graph ──────────────────────

def test_strict_reduces_to_train_only() -> None:
    e_tr, s_tr, e_te, s_te = _fixture()
    edges, signs = reachable_edges(R.STRICT, e_tr, s_tr, e_te, s_te)
    assert np.array_equal(edges, e_tr)
    assert np.array_equal(signs, s_tr)


def test_strict_returns_copies_not_aliases() -> None:
    e_tr, s_tr, e_te, s_te = _fixture()
    edges, signs = reachable_edges(R.STRICT, e_tr, s_tr, e_te, s_te)
    signs[0] = 99
    assert s_tr[0] == 1  # mutating the result must not touch the input


# ── monotone lattice ─────────────────────────────────────────────────────

def test_edge_count_lattice() -> None:
    e_tr, s_tr, e_te, s_te = _fixture()
    n = {r: len(reachable_edges(r, e_tr, s_tr, e_te, s_te)[0]) for r in R}
    assert n[R.STRICT] <= n[R.TRANSDUCTIVE_TOPOLOGY] == n[R.TRANSDUCTIVE_FULL]
    assert n[R.STRICT] == len(e_tr)
    assert n[R.TRANSDUCTIVE_FULL] == len(e_tr) + len(e_te)


def test_node_reachability_lattice() -> None:
    e_tr, _, e_te, _ = _fixture()
    ns = {r: reachable_nodes(r, e_tr, e_te) for r in R}
    assert ns[R.STRICT] <= ns[R.TRANSDUCTIVE_TOPOLOGY]
    assert ns[R.TRANSDUCTIVE_TOPOLOGY] == ns[R.TRANSDUCTIVE_FULL]
    assert {4, 5} <= ns[R.TRANSDUCTIVE_FULL]  # test-only nodes reachable
    assert {4, 5}.isdisjoint(ns[R.STRICT])    # ...but not under strict


# ── the leakage invariant (the audit's reason to exist) ──────────────────

def test_topology_masks_test_signs() -> None:
    e_tr, s_tr, e_te, s_te = _fixture()
    _, signs = reachable_edges(R.TRANSDUCTIVE_TOPOLOGY, e_tr, s_tr, e_te, s_te)
    test_part = signs[len(e_tr):]
    assert np.all(test_part == NEUTRAL_SIGN)        # signs withheld
    assert np.array_equal(signs[: len(e_tr)], s_tr)  # train signs intact


def test_test_signs_reachable_only_under_full() -> None:
    e_tr, s_tr, e_te, s_te = _fixture()
    # Make test signs a value absent from the train set so we can detect leakage.
    s_te = np.array([7, 7], dtype=np.int64)
    for rule in (R.STRICT, R.TRANSDUCTIVE_TOPOLOGY):
        _, signs = reachable_edges(rule, e_tr, s_tr, e_te, s_te)
        assert 7 not in set(signs.tolist()), f"{rule} leaked a test sign"
    _, full = reachable_edges(R.TRANSDUCTIVE_FULL, e_tr, s_tr, e_te, s_te)
    assert (full[len(e_tr):] == 7).all()  # only FULL makes them reachable


# ── parsing / failure case / determinism ─────────────────────────────────

def test_from_str_roundtrip_and_unknown() -> None:
    for r in R:
        assert ReachabilityRule.from_str(r.value) is r
    with pytest.raises(ValueError, match="unknown reachability rule"):
        ReachabilityRule.from_str("transductive")


def test_determinism() -> None:
    e_tr, s_tr, e_te, s_te = _fixture()
    a = reachable_edges(R.TRANSDUCTIVE_FULL, e_tr, s_tr, e_te, s_te)
    b = reachable_edges(R.TRANSDUCTIVE_FULL, e_tr, s_tr, e_te, s_te)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ── per-model wiring: a neutral (sign-0) test edge enters message passing ────

def test_neutral_sign_enters_signed_adjacency() -> None:
    """SGCN/SGCL/DADSGNN builder: a sign-0 (topology) edge joins BOTH channels,
    so a test-only node becomes reachable under TOPOLOGY but not STRICT."""
    import torch

    from hymeko_neuro.baselines.sgcn_model import build_signed_adj

    e_tr, s_tr, e_te, s_te = _fixture()  # test node 5 appears only in test edges
    dev = torch.device("cpu")
    for rule, reachable in ((R.STRICT, False), (R.TRANSDUCTIVE_TOPOLOGY, True)):
        edges, signs = reachable_edges(rule, e_tr, s_tr, e_te, s_te)
        a_pos, a_neg = build_signed_adj(edges, signs, 6, dev)
        deg5 = (torch.sparse.sum(a_pos, 1).to_dense()[5]
                + torch.sparse.sum(a_neg, 1).to_dense()[5]).item()
        assert (deg5 > 0) is reachable


def test_neutral_sign_enters_neighbour_buckets() -> None:
    """SiGAT/SiGformer builder: a sign-0 edge lands in BOTH pos and neg buckets."""
    from hymeko_neuro.baselines.sigat_model import build_neighbour_lists

    e_tr, s_tr, e_te, s_te = _fixture()
    edges, signs = reachable_edges(R.TRANSDUCTIVE_TOPOLOGY, e_tr, s_tr, e_te, s_te)
    pos, neg = build_neighbour_lists(edges, signs, 6)
    # node 4–5 test edge (neutral) appears in both channels for node 5.
    assert 4 in pos[5] and 4 in neg[5]
