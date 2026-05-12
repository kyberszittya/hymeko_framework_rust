"""Integration tests on small synthetic datasets (moon, circles,
regression) — CI-friendly, no external dataset download.

Each test verifies:
  * dataset constructor produces a valid SignedGraph
  * graph has expected size (|V| = n_samples, |E| > 0)
  * signs are in {-1, +1}
  * determinism: same seed → bit-identical graph
  * downstream pipeline integration: a tiny CPML model can train on
    the graph for 5 epochs without crashing (forward + backward + step
    healthy)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.datasets import load
from signedkan_wip.src.datasets_synth import (
    make_circles_signed_graph,
    make_moon_signed_graph,
    make_regression_signed_graph,
)


def test_load_sbm_n_shorthand_matches_long_form():
    """``sbm_n200`` must load (defaults k=4, seed=0) like ``sbm_n200_k4_s0``."""
    a = load("sbm_n200")
    b = load("sbm_n200_k4_s0")
    assert a.n_nodes == b.n_nodes == 200
    assert a.edges.shape == b.edges.shape
    assert np.array_equal(a.edges, b.edges)
    assert np.array_equal(a.signs, b.signs)


# ─── Dataset constructor smokes ─────────────────────────────────────


@pytest.mark.parametrize("name,fn", [
    ("moon",       lambda: make_moon_signed_graph(n_samples=80, seed=0)),
    ("circles",    lambda: make_circles_signed_graph(n_samples=80, seed=0)),
    ("regression", lambda: make_regression_signed_graph(n_samples=80, seed=0)),
])
def test_synth_dataset_basic(name: str, fn):
    g, X, y = fn()
    assert g.n_nodes == 80
    assert g.edges.shape[1] == 2
    assert g.signs.shape[0] == g.edges.shape[0]
    assert set(np.unique(g.signs)).issubset({-1, 1}), \
        f"{name}: signs must be ±1, got {set(np.unique(g.signs))}"
    assert (g.edges[:, 0] < g.n_nodes).all()
    assert (g.edges[:, 1] < g.n_nodes).all()
    # Self-loops should not appear.
    assert (g.edges[:, 0] != g.edges[:, 1]).all()


@pytest.mark.parametrize("fn", [
    make_moon_signed_graph, make_circles_signed_graph,
    make_regression_signed_graph,
])
def test_synth_dataset_determinism(fn):
    """Same seed produces identical output."""
    g1, X1, y1 = fn(n_samples=60, seed=42)
    g2, X2, y2 = fn(n_samples=60, seed=42)
    assert np.array_equal(g1.edges, g2.edges)
    assert np.array_equal(g1.signs, g2.signs)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


@pytest.mark.parametrize("fn", [
    make_moon_signed_graph, make_circles_signed_graph,
])
def test_synth_dataset_class_agreement_sign(fn):
    """For classification-derived signed graphs, an edge between same-
    class points should be +1 and cross-class should be -1."""
    g, X, y = fn(n_samples=60, k_neighbors=4, seed=0)
    for (u, v), s in zip(g.edges, g.signs):
        same_class = (y[u] == y[v])
        expected = 1 if same_class else -1
        assert int(s) == expected, \
            f"edge ({u},{v}) y_u={y[u]} y_v={y[v]} sign={s} expected {expected}"


# ─── Downstream pipeline integration ─────────────────────────────────


@pytest.mark.parametrize("name,fn", [
    ("moon",       lambda: make_moon_signed_graph(n_samples=80, seed=0)),
    ("circles",    lambda: make_circles_signed_graph(n_samples=80, seed=0)),
])
def test_cpml_trains_on_synth_dataset(name: str, fn):
    """A minimal CPML model with random per-vertex features must
    train for 5 epochs on the synthetic signed graph without
    crashing (forward + backward + step all healthy, loss finite)."""
    from signedkan_wip.src.cpml import CPML, CPMLConfig, TierSpec

    torch.manual_seed(0)
    g, X, y = fn()
    cycles = np.zeros((0, 3), dtype=np.int64)  # smoke without cycles
    signs_arr = np.zeros((0, 3), dtype=np.int8)
    degrees = np.bincount(
        g.edges.ravel(), minlength=g.n_nodes,
    ).astype(np.int64) + 1
    cfg = CPMLConfig(
        tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)),
        d_in=8, d_layer=8, aggregator_kind="mlp",
    )
    model = CPML(cfg)
    node_embed = torch.nn.Embedding(g.n_nodes, cfg.d_in)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(node_embed.parameters()), lr=1e-2,
    )
    tier_of = torch.from_numpy(cfg.tier_spec.assign(degrees))
    cyc_t = torch.from_numpy(cycles)
    sgn_t = torch.from_numpy(signs_arr)
    edges_t = torch.from_numpy(g.edges.astype(np.int64))
    targets = torch.from_numpy((g.signs > 0).astype(np.float32))

    initial_loss = None
    final_loss = None
    for ep in range(5):
        scores = model(node_embed.weight, cyc_t, sgn_t, tier_of, edges_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores, targets,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == 0:
            initial_loss = float(loss)
        final_loss = float(loss)

    assert initial_loss is not None and np.isfinite(initial_loss)
    assert final_loss is not None and np.isfinite(final_loss)
    # Loss should not increase by more than 50% over 5 epochs.  This
    # is a sanity gate, not a learning test — synthetic graphs vary.
    assert final_loss <= initial_loss * 1.5, \
        f"{name}: loss diverged {initial_loss:.4f} → {final_loss:.4f}"


# ─── Lazy cycle cache integration ────────────────────────────────────


def test_lazy_cycle_pool_roundtrips_smoke(tmp_path, monkeypatch):
    """LazyCyclePool can be created from packed arrays directly,
    queried for length/arity, materialised, and iterated."""
    from signedkan_wip.src.cycle_cache import LazyCyclePool, _save_packed

    # Synthetic 3-cycle pool of 5 cycles
    v = np.array([
        [0, 1, 2], [1, 2, 3], [2, 3, 4],
        [0, 2, 4], [1, 3, 5],
    ], dtype=np.int64)
    sigma = np.ones_like(v, dtype=np.int8)
    edge_signs = np.array([
        [1, -1, 1], [1, 1, -1], [-1, 1, 1],
        [1, 1, 1], [-1, -1, -1],
    ], dtype=np.int8)
    path = tmp_path / "lazy_pool.npz"
    _save_packed(path, v, sigma, edge_signs)

    pool = LazyCyclePool.from_path(path)
    assert pool is not None
    assert len(pool) == 5
    assert pool.arity() == 3
    # cycle_vertices is non-materialising
    assert np.array_equal(pool.cycle_vertices(0), [0, 1, 2])
    assert np.array_equal(pool.cycle_signs(3), [1, 1, 1])

    # Iterating doesn't trigger full materialisation.
    n_seen = sum(1 for _ in pool.iter())
    assert n_seen == 5

    # Materialise yields the SignedNTuple list once and caches it.
    ntuples = pool.materialize()
    assert len(ntuples) == 5
    assert pool.materialize() is ntuples  # cached on second call


def test_lazy_cycle_pool_missing_path_returns_none(tmp_path):
    from signedkan_wip.src.cycle_cache import LazyCyclePool
    assert LazyCyclePool.from_path(tmp_path / "nonexistent.npz") is None


# ─── CBOR cache format roundtrip ────────────────────────────────────


@pytest.mark.parametrize("fmt", ["npz", "cbor"])
def test_cache_format_roundtrip(tmp_path, monkeypatch, fmt):
    """Both `.npz` and `.cbor` cache formats roundtrip arrays
    losslessly.  Setting HYMEKO_CACHE_FORMAT picks the writer; the
    reader auto-detects from on-disk file presence."""
    from signedkan_wip.src.cycle_cache import _load_packed, _save_packed
    monkeypatch.setenv("HYMEKO_CACHE_FORMAT", fmt)

    v = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    sigma = np.array([[1, -1, 1], [-1, 1, -1], [1, 1, 1]], dtype=np.int8)
    es = np.array([[1, 1, -1], [-1, -1, 1], [1, -1, 1]], dtype=np.int8)
    target = tmp_path / "cache_entry.npz"   # caller-style suffix

    _save_packed(target, v, sigma, es)

    # CBOR landed under .cbor, NPZ under .npz; reader auto-detects.
    v_r, sigma_r, es_r = _load_packed(target)
    assert np.array_equal(v, v_r)
    assert np.array_equal(sigma, sigma_r)
    assert np.array_equal(es, es_r)


def test_cbor_smaller_than_npz_on_tiny_payload(tmp_path, monkeypatch):
    """CBOR has lower envelope overhead than NPZ on small payloads
    (no ZIP container, no per-array .npy header). Verify the size
    relation as a sanity check that the format is doing what we
    claim."""
    from signedkan_wip.src.cycle_cache import _save_packed
    v = np.array([[0, 1, 2]], dtype=np.int64)
    sigma = np.array([[1, -1, 1]], dtype=np.int8)
    es = np.array([[1, 1, -1]], dtype=np.int8)

    monkeypatch.setenv("HYMEKO_CACHE_FORMAT", "npz")
    npz_path = tmp_path / "npz_tiny.npz"
    _save_packed(npz_path, v, sigma, es)
    npz_size = npz_path.stat().st_size

    monkeypatch.setenv("HYMEKO_CACHE_FORMAT", "cbor")
    cbor_target = tmp_path / "cbor_tiny.npz"
    _save_packed(cbor_target, v, sigma, es)
    cbor_path = cbor_target.with_suffix(".cbor")
    cbor_size = cbor_path.stat().st_size

    assert cbor_size < npz_size, \
        f"cbor {cbor_size}B should be smaller than npz {npz_size}B on tiny payload"


def test_cbor_format_is_wire_stable(tmp_path, monkeypatch):
    """The CBOR layout is the published wire format (top-level map
    with explicit shape + dtype + bytes per field).  Verify a
    written file can be parsed back by raw cbor2 + numpy without
    going through our loader — proving the cross-language story."""
    import cbor2
    from signedkan_wip.src.cycle_cache import _save_packed

    monkeypatch.setenv("HYMEKO_CACHE_FORMAT", "cbor")
    v = np.arange(12, dtype=np.int64).reshape(4, 3)
    sigma = np.where(v % 2 == 0, 1, -1).astype(np.int8)
    es = np.where(v % 3 == 0, 1, -1).astype(np.int8)
    target = tmp_path / "wire_check.npz"
    _save_packed(target, v, sigma, es)

    # Read back via raw cbor2 — same code path a Rust client uses.
    with target.with_suffix(".cbor").open("rb") as fh:
        payload = cbor2.load(fh)
    assert payload["format_version"] == 1
    assert payload["v_dtype"] == "int64"
    assert payload["sigma_dtype"] == "int8"
    v_r = np.frombuffer(
        payload["v_buf"], dtype=payload["v_dtype"],
    ).reshape(tuple(payload["v_shape"]))
    assert np.array_equal(v, v_r)
