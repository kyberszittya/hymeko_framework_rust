"""ARCHITECTURAL_ASSIMILATION_V1 — mandatory runtime-core closing tests (structured state + LSTM temporal + fusion).

Task-independent (no coin/CIP/pick-place). Covers the contracts required before wiring the first task adapter:
batch≡stream LSTM, no cross-episode hidden leakage, checkpoint round-trip, K=1 reproduces FixedBudgetSearch, StructuredState
↔ FlatStateView, fusion, and the domain-purity of `option_rl`.
"""
import importlib
import pkgutil

import numpy as np
import torch

import hymeko_rl.option_rl as orl
from hymeko_rl.option_rl import (
    FixedBudgetSearch,
    FlatStateView,
    LSTMTemporalEncoder,
    MultimodalBudgetSearch,
    ProposalMode,
    StructuredState,
    fuse_state,
)


# ── structured state ──
class _HG:
    def node_features(self):
        return np.arange(12, dtype=np.float32).reshape(4, 3)
    edges = [(0, 1), (1, 2, 3)]


def test_structured_state_from_hypergraph_and_flat_view():
    s = StructuredState.from_hypergraph(_HG(), phase=2, geometry=np.array([0.02, 0.04]))
    assert s.n_nodes == 4 and s.edges == [(0, 1), (1, 2, 3)]
    flat = FlatStateView().view(s)
    assert flat.shape[0] == 12 + 2 + 1 and flat[-1] == 2.0        # nodes ⊕ geometry ⊕ phase
    assert np.allclose(flat[:12], np.arange(12))


def test_structured_state_rejects_bad_edges():
    import pytest
    with pytest.raises(ValueError):
        StructuredState(np.zeros((2, 3)), edges=[(0, 5)])          # index outside [0,2)


# ── LSTM temporal: batch ≡ stream, no leakage, checkpoint round-trip ──
def _enc():
    torch.manual_seed(0)
    return LSTMTemporalEncoder(in_dim=5, hidden=8, out_dim=4).eval()


def test_lstm_batch_equals_stream():
    enc = _enc()
    X = torch.randn(1, 6, 5)
    with torch.no_grad():
        batch_emb, _ = enc(X)
        h, stream = enc.initial_hidden(1), []
        for t in range(6):
            e, h = enc.update(X[:, t, :], h)
            stream.append(e)
        stream = torch.stack(stream, 1)
    assert torch.allclose(batch_emb, stream, atol=1e-5)           # streaming ≡ batch (causal)


def test_lstm_reset_no_cross_episode_leakage():
    enc = _enc()
    with torch.no_grad():
        _e, h = enc.update(torch.randn(1, 5), enc.initial_hidden(1))   # episode 1 leaves a hidden state
        x = torch.randn(1, 5)
        e_fresh, _ = enc.update(x, enc.initial_hidden(1))         # episode 2 with a RESET hidden
        e_leaked, _ = enc.update(x, h)                            # same input but carrying ep-1 hidden
    assert not torch.allclose(e_fresh, e_leaked)                  # reset genuinely clears; leakage would be a bug
    e_fresh2, _ = enc.update(x, enc.initial_hidden(1))
    assert torch.allclose(e_fresh, e_fresh2)                      # reset is deterministic


def test_lstm_checkpoint_restore_identical_next(tmp_path):
    enc = _enc()
    with torch.no_grad():
        _e, h = enc.update(torch.randn(1, 5), enc.initial_hidden(1))
        x = torch.randn(1, 5)
        expected, _ = enc.update(x, h)
    ckpt = tmp_path / "enc.pt"
    torch.save({"weights": enc.state_dict(), "hidden": (h[0], h[1])}, ckpt)     # mid-episode checkpoint
    enc2 = LSTMTemporalEncoder(5, 8, 4)
    blob = torch.load(ckpt, weights_only=False)
    enc2.load_state_dict(blob["weights"])
    enc2.eval()
    with torch.no_grad():
        got, _ = enc2.update(x, blob["hidden"])
    assert torch.allclose(got, expected, atol=1e-6)              # restore → identical next embedding


def test_fuse_state_concat_contract():
    f = fuse_state(np.ones(3), np.full(4, 2.0), np.full(2, 3.0))
    assert f.shape[0] == 9 and f[0] == 1.0 and f[3] == 2.0 and f[7] == 3.0


# ── multimodal search: K=1 reproduces FixedBudgetSearch ──
class _Gen:
    def sample(self, center, n, rng):
        return np.asarray(center, np.float64) + rng.normal(0, 0.05, (n, len(center)))


class _Scorer:
    def score(self, cand, rng):
        return -float(abs(float(np.asarray(cand)[0]) - 0.3)), {"k6": 0}    # peak at 0.3


class _K1:
    def modes(self, obs):
        return [ProposalMode(1.0, np.array([0.0]), None, 0)]


def test_k1_multimodal_reproduces_fixed_budget_search():
    gen, sc = _Gen(), _Scorer()
    a = FixedBudgetSearch(gen, sc, budget=8).select(np.array([0.0]), np.random.default_rng(7))
    b = MultimodalBudgetSearch(gen, sc, budget=8).select(_K1(), np.zeros(1), np.random.default_rng(7))
    assert abs(float(a.selected[0]) - float(b.selected[0])) < 1e-9 and abs(a.score - b.score) < 1e-9


# ── domain purity: option_rl imports no task module ──
def test_option_rl_imports_no_domain_module():
    banned = ("coin", "galambos", "pick_place", "cip", "lingam", "disk", "delivery", "grasp")
    for m in pkgutil.iter_modules(orl.__path__):
        mod = importlib.import_module(f"hymeko_rl.option_rl.{m.name}")
        for name in dir(mod):
            obj = getattr(mod, name)
            src = getattr(obj, "__module__", "") or ""
            assert not any(b in src.lower() for b in banned), f"option_rl.{m.name}.{name} pulls in {src}"
