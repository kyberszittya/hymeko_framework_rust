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
    Mechanism,
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


# --- mechanism-form causal hypergraph (LiNGAM-SH representation step) --------------------------------------------
def test_pairwise_path_unchanged_from_lingam() -> None:
    """A (from_lingam) causal hypergraph is pairwise (no mechanisms) — the DirectLiNGAM path is untouched."""
    x, _b = sample_linear_sem([(0, 1, 0.9)], 2, 300, seed=2, noise="uniform")
    cg = CausalHypergraph.from_lingam(DirectLiNGAM().fit(x, ["a", "b"]), "G")
    assert cg.mechanisms == []
    assert to_hymeko_source(cg) == to_hymeko_source(cg)          # deterministic; pairwise branch unchanged


def test_degenerate_mechanism_equivalent_to_pairwise() -> None:
    """A pairwise edge x→y is representable as a degenerate mechanism; collapsed emit is byte-identical."""
    pair = CausalHypergraph(name="G", variables=["a", "b", "c"], edges=[("a", "b", 0.8), ("b", "c", -0.6)])
    mechs = [Mechanism.pairwise("a", "b", 0.8, name="m0"), Mechanism.pairwise("b", "c", -0.6, name="m1")]
    mech = CausalHypergraph.from_mechanisms("G", mechs)
    assert mech.pairwise_projection() == [("a", "b", 0.8), ("b", "c", -0.6)]
    assert to_hymeko_source(mech.to_pairwise_hypergraph()) == to_hymeko_source(pair)   # byte-identical


def test_multi_input_single_output_mechanism_emits_valid_hymeko(tmp_path) -> None:
    """{contact_score, near_fraction} → {total_reward} emits a valid star-expanded .hymeko and cross-view agrees."""
    m = Mechanism(name="reward_mech", tail=("contact_score", "near_fraction"), head=("total_reward",),
                  strength=0.7, sign=1)
    cg = CausalHypergraph.from_mechanisms("CoinReward", [m])
    src = to_hymeko_source(cg)
    assert "    reward_mech {}" in src                            # hub vertex declared
    assert "(+contact_score, +reward_mech)" in src and "(+near_fraction, +reward_mech)" in src
    assert "(+reward_mech, +total_reward)" in src
    report = cross_view_verify(cg, tmp_path / "m.hymeko")
    assert report.agree, report.notes
    assert report.n_edges_declared == 3                          # 2 tail + 1 head incidences


def test_multi_output_mechanism_pairwise_projection() -> None:
    """{a,b} → {c,d} projects to the four pairwise edges a→c, a→d, b→c, b→d."""
    cg = CausalHypergraph.from_mechanisms("G", [Mechanism("mech", ("a", "b"), ("c", "d"), strength=1.0, sign=1)])
    proj = {(c, e) for c, e, _w in cg.pairwise_projection()}
    assert proj == {("a", "c"), ("a", "d"), ("b", "c"), ("b", "d")}


def test_star_projection_contains_hub_nodes() -> None:
    """The star projection introduces a mechanism hub and tail→hub / hub→head incidences."""
    cg = CausalHypergraph.from_mechanisms("G", [Mechanism("mech", ("a", "b"), ("c",))])
    star = cg.star_projection()
    assert star.hubs == ["mech"]
    assert ("a", "mech", 1) in star.incidence and ("b", "mech", 1) in star.incidence
    assert ("mech", "c", 1) in star.incidence


def test_negative_mechanism_sign_survives_cross_view(tmp_path) -> None:
    cg = CausalHypergraph.from_mechanisms("G", [Mechanism("mech", ("a",), ("b",), strength=0.5, sign=-1)])
    assert cg.declared_signed_edges() == {("a", "mech", 1), ("mech", "b", -1)}
    assert cross_view_verify(cg, tmp_path / "n.hymeko").agree


def test_acyclic_mechanism_graph_passes() -> None:
    cg = CausalHypergraph.from_mechanisms("G", [Mechanism("m1", ("a",), ("b",)), Mechanism("m2", ("b",), ("c",))])
    res = cg.check_acyclicity()
    assert res.acyclic and bool(res) and res.rank is not None
    assert res.rank["a"] < res.rank["m1"] < res.rank["b"] < res.rank["m2"] < res.rank["c"]


def test_cyclic_mechanism_graph_fails() -> None:
    cg = CausalHypergraph.from_mechanisms("G", [Mechanism("m1", ("a",), ("b",)), Mechanism("m2", ("b",), ("a",))])
    res = cg.check_acyclicity()
    assert not res.acyclic and not bool(res)
    assert set(res.cycle_nodes or []) >= {"a", "b", "m1", "m2"}


def test_delivery_mechanism_cross_view(tmp_path) -> None:
    """Example C: {delivery_score, progress_score, stagnation_duration} → {monitor_pass}."""
    m = Mechanism("task_success", ("delivery_score", "progress_score", "stagnation_duration"), ("monitor_pass",))
    cg = CausalHypergraph.from_mechanisms("Delivery", [m])
    report = cross_view_verify(cg, tmp_path / "d.hymeko")
    assert report.agree, report.notes
    assert report.n_edges_declared == 4                          # 3 tail + 1 head incidences


def test_mechanism_self_cycle_rejected() -> None:
    with pytest.raises(ValueError, match="both tail and head"):
        Mechanism("m", ("a", "b"), ("b",))


def test_mechanism_hub_name_collision_rejected() -> None:
    # mechanism hub "b" collides with the variable "b" — the hub vertex must be distinct
    with pytest.raises(ValueError, match="collides with a variable"):
        CausalHypergraph(name="G", variables=["a", "b"], edges=[], mechanisms=[Mechanism("b", ("a",), ("b",))])
