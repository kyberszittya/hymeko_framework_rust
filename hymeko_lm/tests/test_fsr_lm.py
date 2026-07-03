"""Phase-0 tests for hymeko_lm: contracts of every public component + a learning integration test.

Run: ``uv run pytest hymeko_lm/tests -p no:randomly``
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from hymeko_lm import (
    Activation,
    FSRConfig,
    FSRLanguageModel,
    FiberSpikeRotorMixer,
    l2_normalize,
    spherical_residual,
)
from hymeko_lm.checkpoint import load_checkpoint, save_checkpoint
from hymeko_lm.data import make_associative_recall_batch, make_lag_copy_batch


def _cfg(*, n_blocks: int = 4, n_layers: int = 2, max_seq_len: int = 16, gate_rank: int = 8) -> FSRConfig:
    return FSRConfig(vocab_size=16, n_blocks=n_blocks, n_layers=n_layers,
                     max_seq_len=max_seq_len, gate_rank=gate_rank)


# ----- Gömb (sphere) -----

def test_l2_normalize_unit_norm() -> None:
    x = torch.randn(3, 5, 12)
    out = l2_normalize(x)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3, 5), atol=1e-5)


def test_l2_normalize_zero_is_finite() -> None:
    out = l2_normalize(torch.zeros(2, 4))
    assert torch.isfinite(out).all()


def test_spherical_residual_returns_to_sphere() -> None:
    h = l2_normalize(torch.randn(2, 3, 12))
    out = spherical_residual(h, torch.randn(2, 3, 12))
    assert torch.allclose(out.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


# ----- Fiber-Spike-Rotor mixer -----

def test_mixer_shape() -> None:
    mix = FiberSpikeRotorMixer(n_blocks=4, max_seq_len=16, gate_rank=8)
    h = l2_normalize(torch.randn(2, 10, 12))
    assert mix(h).shape == (2, 10, 12)


def test_mixer_rotor_identity_at_init() -> None:
    """Zero-init bivector => identity rotor [1,0,0,0]; the mixer starts as a pure signed gather."""
    mix = FiberSpikeRotorMixer(n_blocks=4, max_seq_len=16, gate_rank=8)
    from signedkan_wip.src.embeddings.cayley_rotor import cayley_to_unit_quat

    q = cayley_to_unit_quat(mix.offset_bivec)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    assert torch.allclose(q, ident, atol=1e-6)


def test_mixer_seq_len_guard() -> None:
    mix = FiberSpikeRotorMixer(n_blocks=4, max_seq_len=8, gate_rank=8)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        mix(torch.randn(1, 9, 12))


def test_mixer_sparse_topk_shape() -> None:
    mix = FiberSpikeRotorMixer(n_blocks=4, max_seq_len=16, gate_rank=8, spike_k=3)
    h = l2_normalize(torch.randn(2, 10, 12))
    assert mix(h).shape == (2, 10, 12)


def test_model_is_causal_with_spike_k() -> None:
    """Hard top-k spike gate must stay causal: logits at <=k unaffected by future tokens."""
    torch.manual_seed(0)
    cfg = FSRConfig(vocab_size=16, n_blocks=4, n_layers=2, max_seq_len=16, gate_rank=8, spike_k=2)
    model = FSRLanguageModel(cfg).eval()
    k = 5
    a = torch.randint(0, cfg.vocab_size, (2, 12))
    b = a.clone()
    b[:, k + 1:] = torch.randint(0, cfg.vocab_size, b[:, k + 1:].shape)
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[:, : k + 1], lb[:, : k + 1], atol=1e-5)


def test_config_rejects_bad_spike_k() -> None:
    with pytest.raises(ValueError, match="spike_k"):
        FSRConfig(vocab_size=16, spike_k=0)


# ----- generation + checkpoint -----

def test_generate_extends_and_is_deterministic() -> None:
    cfg = _cfg()
    model = FSRLanguageModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 4))
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    o1 = model.generate(ids, 5, generator=g1)
    o2 = model.generate(ids, 5, generator=g2)
    assert o1.shape == (2, 9)
    assert torch.equal(o1, o2)            # same seed -> same samples
    assert torch.equal(o1[:, :4], ids)    # prefix preserved


def test_generate_rejects_bad_temperature() -> None:
    model = FSRLanguageModel(_cfg())
    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.zeros(1, 2, dtype=torch.long), 1, temperature=0.0)


def test_checkpoint_roundtrip(tmp_path: object) -> None:
    cfg = FSRConfig(vocab_size=16, n_blocks=8, n_layers=2, max_seq_len=16, gate_rank=8, spike_k=4)
    model = FSRLanguageModel(cfg).eval()
    ids = torch.randint(0, 16, (2, 10))
    with torch.no_grad():
        y0 = model(ids)
    path = str(tmp_path) + "/ckpt.pt"  # type: ignore[operator]
    save_checkpoint(path, model, step=42, extra={"val_bpb": 1.25})
    model2, step, extra = load_checkpoint(path)
    model2.eval()
    with torch.no_grad():
        y1 = model2(ids)
    assert step == 42 and extra["val_bpb"] == 1.25
    assert torch.allclose(y0, y1, atol=1e-6)


# ----- model: causality, shape, grads -----

def test_model_output_shape() -> None:
    cfg = _cfg()
    model = FSRLanguageModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 12))
    assert model(ids).shape == (3, 12, cfg.vocab_size)


def test_model_is_causal() -> None:
    """Logits at positions <= k must not change when tokens after k change."""
    torch.manual_seed(0)
    cfg = _cfg()
    model = FSRLanguageModel(cfg).eval()
    k = 5
    a = torch.randint(0, cfg.vocab_size, (2, 12))
    b = a.clone()
    b[:, k + 1:] = torch.randint(0, cfg.vocab_size, b[:, k + 1:].shape)
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[:, : k + 1], lb[:, : k + 1], atol=1e-5)


def test_model_backward_populates_finite_grads() -> None:
    cfg = _cfg()
    model = FSRLanguageModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    tgt = torch.randint(0, cfg.vocab_size, (2, 10))
    model.loss(ids, tgt).backward()   # type: ignore[no-untyped-call]  # torch stub: Tensor.backward untyped
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_cr_cheby_is_the_default_channel_activation() -> None:
    assert _cfg().activation is Activation.CR_CHEBY


# ----- config + data contracts -----

def test_config_rejects_small_cr_cheby_grid() -> None:
    with pytest.raises(ValueError, match="grid >= 8"):
        FSRConfig(vocab_size=16, grid=4)


def test_config_d_model_is_three_times_blocks() -> None:
    assert FSRConfig(vocab_size=16, n_blocks=7).d_model == 21


def test_lag_copy_property_and_shape() -> None:
    gen = torch.Generator().manual_seed(1)
    ids, tgt = make_lag_copy_batch(4, 12, 16, lag=3, generator=gen)
    assert ids.shape == (4, 12) and tgt.shape == (4, 12)
    # x[t] == x[t-lag] for t >= lag
    assert torch.equal(ids[:, 3:], ids[:, : 12 - 3][:, : 9])
    # targets are ids shifted left by one within the generated stream
    assert torch.equal(tgt[:, :-1], ids[:, 1:])


def test_lag_copy_rejects_bad_lag() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="lag"):
        make_lag_copy_batch(2, 8, 16, lag=8, generator=gen)


def test_associative_recall_shape_and_target() -> None:
    gen = torch.Generator().manual_seed(1)
    ids, tgt, qpos = make_associative_recall_batch(4, n_pairs=3, key_vocab=8, val_vocab=8, generator=gen)
    assert ids.shape == (4, 7) and tgt.shape == (4, 7) and qpos == 6
    # the query key (last id) recalls a value that appeared earlier in the same row
    for row in range(4):
        kq = ids[row, qpos]
        seen_keys = ids[row, 0:qpos:2]
        assert (seen_keys == kq).any()


def test_associative_recall_rejects_too_many_pairs() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="key_vocab"):
        make_associative_recall_batch(2, n_pairs=9, key_vocab=8, val_vocab=8, generator=gen)


# ----- integration: the mixer actually learns to route -----

def test_fsr_lm_learns_lag_copy() -> None:
    """The discriminator: if FSR routing works, loss drops below the uniform (no-routing) bound."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    vocab, seq, lag = 16, 16, 3
    cfg = FSRConfig(vocab_size=vocab, n_blocks=8, n_layers=2, max_seq_len=seq, gate_rank=16)
    model = FSRLanguageModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    final = math.inf
    for _ in range(300):
        ids, tgt = make_lag_copy_batch(32, seq, vocab, lag, gen)
        loss = model.loss(ids, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub: Tensor.backward untyped
        opt.step()
        final = float(loss.detach())
    # Floor is ~ (lag-1)/seq · ln(vocab) (the genuinely-random positions); beating half of uniform
    # already proves the mixer routes by offset. (Diagnostic at this capacity converges to ~0.37.)
    assert final < math.log(vocab) * 0.5, f"did not learn to route: final={final:.3f} vs uniform={math.log(vocab):.3f}"


def test_fsr_lm_learns_associative_recall() -> None:
    """Memory discriminator: in-context content-addressed recall (per-sequence-random key->value map,
    so it cannot be memorised in the weights). Requires the value fiber + a >=2-layer induction circuit."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    n_pairs, kv, vv = 3, 8, 8
    seq = 2 * n_pairs + 1
    cfg = FSRConfig(vocab_size=kv + vv, n_blocks=8, n_layers=3, max_seq_len=seq, gate_rank=16)
    model = FSRLanguageModel(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    final = math.inf
    for _ in range(700):
        ids, tgt, qpos = make_associative_recall_batch(64, n_pairs, kv, vv, gen)
        loss = F.cross_entropy(model(ids)[:, qpos], tgt[:, qpos])
        opt.zero_grad(set_to_none=True)
        loss.backward()   # type: ignore[no-untyped-call]  # torch stub: Tensor.backward untyped
        opt.step()
        final = float(loss.detach())
    assert final < math.log(vv) * 0.4, f"no recall: final={final:.3f} vs uniform={math.log(vv):.3f}"
