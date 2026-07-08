"""Tests for the causal-DAG → ``.hymeko`` emitter + cross-view verifier (``hymeko_emit``).

The discriminating test: a declared signed DAG must survive the HyMeKo IR round-trip *identically* — same
edges, same signs, consistent star/clique counts. If the engine is built (native ``hymeko``) we also assert the
canonical hash is present and stable; otherwise the literal fallback checks the count invariants. Contract guards
(unknown var, self-loop, non-identifier) must raise, not silently corrupt the declaration.
"""
from __future__ import annotations

import importlib.util

import pytest

from hymeko_rl.eval.causal import (
    CausalHypergraph,
    DirectLiNGAM,
    cross_view_verify,
    sample_linear_sem,
    to_hymeko_source,
)
from hymeko_rl.eval.causal.hymeko_emit import _engine_signed_edges, _literal_signed_edges

_HAS_ENGINE = importlib.util.find_spec("hymeko") is not None


def _chain() -> CausalHypergraph:
    # approach -> contact (+), contact -> delivery (-): a signed 2-edge chain.
    return CausalHypergraph(name="Probe", variables=["approach", "contact", "delivery"],
                            edges=[("approach", "contact", 0.8), ("contact", "delivery", -0.6)])


def test_to_hymeko_source_grammar() -> None:
    src = to_hymeko_source(_chain())
    assert src.startswith("Probe{}\nprobe\n{")
    assert "    approach {}" in src and "    delivery {}" in src
    assert "(+approach, +contact)" in src        # positive weight -> effect arc +
    assert "(+contact, -delivery)" in src        # negative weight -> effect arc -


def test_declared_signed_edges() -> None:
    assert _chain().declared_signed_edges() == {("approach", "contact", 1), ("contact", "delivery", -1)}


def test_literal_reparse_roundtrip() -> None:
    """The literal fallback reparses the emitted grammar to the same signed edges (engine-independent)."""
    cg = _chain()
    edges, n, sum_arities = _literal_signed_edges(to_hymeko_source(cg))
    assert edges == cg.declared_signed_edges()
    assert n == 2 and sum_arities == 4


def test_cross_view_agrees(tmp_path) -> None:
    """Declared DAG re-derives identically through whichever backend is available; counts and edges match."""
    cg = _chain()
    report = cross_view_verify(cg, tmp_path / "probe.hymeko")
    assert (tmp_path / "probe.hymeko").exists()
    assert report.edges_match, report.notes
    assert report.counts_match, report.notes
    assert report.agree
    assert report.sum_arities == 4 and report.n_edges_declared == 2


@pytest.mark.skipif(not _HAS_ENGINE, reason="native hymeko engine not built")
def test_engine_hash_present_and_stable(tmp_path) -> None:
    """With the engine, the canonical Blake3 hash is present and deterministic for the same declaration."""
    cg = _chain()
    r1 = cross_view_verify(cg, tmp_path / "a.hymeko")
    r2 = cross_view_verify(cg, tmp_path / "b.hymeko")
    assert r1.backend == "engine"
    assert r1.canonical_hash.startswith("blake3:")
    assert r1.canonical_hash == r2.canonical_hash        # same DAG -> same fingerprint


def test_from_lingam_recovers_declared_chain(tmp_path) -> None:
    """A LiNGAM fit on a known chain declares a DAG whose cross-view agrees (end-to-end, discovery → IR)."""
    x, _b = sample_linear_sem([(0, 1, 0.9), (1, 2, -0.7)], 3, 400, seed=0, noise="uniform")
    result = DirectLiNGAM().fit(x, ["a", "b", "c"])
    cg = CausalHypergraph.from_lingam(result, "Recovered")
    assert cg.edges, "expected at least one recovered edge"
    report = cross_view_verify(cg, tmp_path / "recovered.hymeko")
    assert report.agree, report.notes


def test_unknown_variable_raises() -> None:
    with pytest.raises(ValueError, match="unknown variable"):
        CausalHypergraph(name="G", variables=["a", "b"], edges=[("a", "z", 1.0)])


def test_self_loop_raises() -> None:
    with pytest.raises(ValueError, match="self-loop"):
        CausalHypergraph(name="G", variables=["a"], edges=[("a", "a", 1.0)])


def test_non_identifier_variable_raises() -> None:
    with pytest.raises(ValueError, match="not a valid"):
        CausalHypergraph(name="G", variables=["1bad"], edges=[])


def test_engine_reparse_rejects_non_binary_edge() -> None:
    """A ternary hyperedge is not a causal edge — the engine reparser must reject it, not average it."""
    snap = {"edges": [{"name": "c0", "arcs": [{"target_name": "a", "sign": 1},
                                              {"target_name": "b", "sign": 1},
                                              {"target_name": "c", "sign": 1}]}]}
    with pytest.raises(ValueError, match="binary"):
        _engine_signed_edges(snap)
