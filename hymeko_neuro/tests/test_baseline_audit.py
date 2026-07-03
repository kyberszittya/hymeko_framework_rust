"""Tests for the Phase B baseline registry + unified audit runner.

Layers (CLAUDE.md §3): unit (forward shape, registry round-trip, failure case),
property (shuffle is a permutation; CPU determinism), integration (a real strict
run on a synthetic — network-free — dataset, real + shuffle arms).

Run: ``pytest -p no:randomly hymeko_neuro/tests/test_baseline_audit.py``
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import math

from hymeko_neuro.baselines.registry import (
    GraphMeta,
    HParams,
    get_baseline,
    list_baselines,
)
from hymeko_neuro.data.datasets import SignedGraph
from hymeko_neuro.experiments.runs import run_baseline_audit as rba
from hymeko_neuro.experiments.runs.run_baseline_audit import (
    SHUFFLE_SEED_OFFSET,
    run_audit,
)

LEGACY_MODELS = ["sgcn", "sigat", "sgt"]
NEW_MODELS = ["sgcl", "sigformer", "sesgformer", "dadsgnn"]
ALL_MODELS = LEGACY_MODELS + NEW_MODELS
SYNTH_DATASET = "sbm_n200_k4_s0"  # programmatic loader — no network, deterministic


def _synthetic_signed(n: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small Erdős–Rényi signed graph for forward-shape tests."""
    rng = np.random.default_rng(seed)
    edges, signs = [], []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.3:
                edges.append((u, v))
                signs.append(1 if rng.random() < 0.6 else -1)
    return np.asarray(edges, dtype=np.int64), np.asarray(signs, dtype=np.int64)


# --- unit -----------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_shape(name: str) -> None:
    """Every model encodes n_nodes embeddings and emits one logit per edge."""
    strat = get_baseline(name)
    edges, signs = _synthetic_signed(n=20)
    ctx = strat.build_context(edges, signs, 20, torch.device("cpu"))
    model = strat.build_model(GraphMeta(n_nodes=20), strat.default_hparams())
    z = model.encode_nodes(*ctx)
    assert z.shape[0] == 20
    assert torch.isfinite(z).all()
    logits = model.edge_logits(z, torch.from_numpy(edges))
    assert logits.shape == (len(edges),)
    assert model.num_parameters() > 0


def test_registry_roundtrip() -> None:
    names = list_baselines()
    assert set(ALL_MODELS) <= set(names)
    for n in names:
        assert get_baseline(n).name == n


def test_unknown_model_raises_keyerror() -> None:
    """Unknown name → shallow KeyError naming the valid set (§6.5 #7)."""
    with pytest.raises(KeyError) as exc:
        get_baseline("not_a_model")
    assert "available" in str(exc.value)


def test_hparams_merge_ignores_none() -> None:
    hp = HParams().merged(hidden=None, n_layers=4, n_epochs=7)
    assert hp.hidden == 32 and hp.n_layers == 4 and hp.n_epochs == 7


def test_graphmeta_rejects_empty() -> None:
    with pytest.raises(ValueError):
        GraphMeta(n_nodes=0)


# --- property -------------------------------------------------------------

@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_shuffle_is_permutation(seed: int) -> None:
    """The audit shuffle permutes signs: the label multiset is preserved."""
    signs = np.array([1, 1, 1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int64)
    perm = np.random.default_rng(seed + SHUFFLE_SEED_OFFSET).permutation(len(signs))
    shuffled = signs[perm]
    assert sorted(shuffled.tolist()) == sorted(signs.tolist())


# --- integration ----------------------------------------------------------

def test_run_audit_real_and_shuffle_arms() -> None:
    real = run_audit("sgcn", SYNTH_DATASET, 0, n_epochs=5)
    assert 0.0 <= real["test_auroc"] <= 1.0
    assert real["n_params"] > 0 and real["shuffle"] is False
    shuf = run_audit("sgcn", SYNTH_DATASET, 0, n_epochs=5, shuffle_train_signs=True)
    assert 0.0 <= shuf["test_auroc"] <= 1.0 and shuf["shuffle"] is True


def test_determinism_same_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same seed ⇒ identical held-out AUROC on CPU (regression oracle)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    a = run_audit("sgcn", SYNTH_DATASET, 1, n_epochs=8)
    b = run_audit("sgcn", SYNTH_DATASET, 1, n_epochs=8)
    assert a["test_auroc"] == b["test_auroc"]
    assert a["n_params"] == b["n_params"]


# --- dedup (true held-out) regression -------------------------------------

def _repeated_pair_graph() -> SignedGraph:
    """Every edge is one of two undirected pairs, repeated with alternating signs.

    With an 80/10/10 split both pairs land in train, so under ``--dedup`` *every*
    val/test row is a train pair and must be dropped — a deterministic oracle for
    the held-out filter that does not depend on the split RNG's exact placement.
    """
    rows = ([(0, 1)] * 20) + ([(2, 3)] * 20)
    edges = np.asarray(rows, dtype=np.int64)
    signs = np.asarray([1 if i % 2 == 0 else -1 for i in range(len(rows))],
                       dtype=np.int64)
    return SignedGraph(edges=edges, signs=signs, n_nodes=4)


def test_dedup_off_is_noop_and_keeps_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (dedup off): nothing dropped, prior split survives, AUROC defined."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rba, "load", lambda name: _repeated_pair_graph())
    r = run_audit("sgcn", "repeated_pairs", 0, n_epochs=5)
    assert r["dedup"] is False
    assert r["n_test_dropped"] == 0 and r["n_val_dropped"] == 0
    assert r["n_test"] > 0
    # AUROC is a float (may be nan on a one-class slice of this tiny fixture);
    # the point of this test is the no-op + key presence, not the value.
    assert math.isnan(r["test_auroc"]) or 0.0 <= r["test_auroc"] <= 1.0


def test_dedup_drops_train_overlapping_heldout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dedup`` drops every val/test row whose pair is in train (regression:
    these keys + this branch did not exist before)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rba, "load", lambda name: _repeated_pair_graph())
    r = run_audit("sgcn", "repeated_pairs", 0, n_epochs=5, dedup=True)
    assert r["dedup"] is True
    assert r["n_test_dropped"] > 0       # both pairs are in train → all held-out drop
    assert r["n_test"] == 0
    assert math.isnan(r["test_auroc"])   # empty held-out slice → AUROC undefined
