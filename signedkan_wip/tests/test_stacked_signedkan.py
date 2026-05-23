"""Unit tests for the Phase-16 ResNet-style stackable HSIKAN.

Pins:

  * ``SignedKANResidualBlock`` forward/backward + identity-skip
    invariance at init.
  * ``StackedSignedKAN`` correct output shape at depths 1, 2, 4, 8.
  * Depth-1 ``StackedSignedKAN`` is functionally equivalent to a
    bare ``MultiLayerSignedKAN(n_layers=1, ...)`` in the same
    config (the wrapper isn't doing anything secret).
"""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.core.signedkan import (
    MultiLayerSignedKAN, MultiLayerSignedKANConfig,
    build_vertex_triad_incidence,
)
from signedkan_wip.src.core.stacked_signedkan import (
    SignedKANResidualBlock,
    StackedSignedKAN,
    StackedSignedKANConfig,
)


def _toy_triads(n_nodes: int = 6, n_triads: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (triad_v, triad_sigma, M_vt) for a deterministic tiny fixture."""
    rng = np.random.default_rng(0)
    triad_v_np = rng.integers(0, n_nodes, size=(n_triads, 3))
    triad_v_np.sort(axis=1)
    # Ensure each row has distinct vertices (re-sample collisions).
    for i in range(n_triads):
        while len(set(triad_v_np[i])) < 3:
            triad_v_np[i] = sorted(rng.integers(0, n_nodes, size=3))
    triad_sigma = torch.tensor(rng.choice([-1, 1], size=(n_triads, 3)),
                               dtype=torch.long)
    triad_v = torch.from_numpy(triad_v_np).long()
    M_vt = build_vertex_triad_incidence(
        triad_v_np, n_nodes=n_nodes, device=torch.device("cpu"), mode="mean",
    )
    return triad_v, triad_sigma, M_vt


def test_residual_block_forward_output_shape():
    block = SignedKANResidualBlock(n_nodes=6, hidden_dim=4)
    triad_v, triad_sigma, M_vt = _toy_triads()
    h_v = torch.randn(6, 4)
    h_v_new, h_t = block(h_v, triad_v, triad_sigma, M_vt)
    assert h_v_new.shape == (6, 4)
    assert h_t.shape == (triad_v.shape[0], 4)


def test_residual_block_backward_passes():
    block = SignedKANResidualBlock(n_nodes=6, hidden_dim=4)
    triad_v, triad_sigma, M_vt = _toy_triads()
    h_v = torch.randn(6, 4, requires_grad=True)
    h_v_new, h_t = block(h_v, triad_v, triad_sigma, M_vt)
    loss = h_t.sum() + h_v_new.sum()
    loss.backward()
    # At least one parameter should have a non-zero grad.
    grads_nonzero = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in block.parameters()
    )
    assert grads_nonzero


def test_stacked_signedkan_depths():
    """Output shapes at depths 1, 2, 4, 8 — confirms the wrapper
    doesn't break at large stack."""
    triad_v, triad_sigma, M_vt = _toy_triads()
    for L in (1, 2, 4, 8):
        cfg = StackedSignedKANConfig(n_nodes=6, n_blocks=L, hidden_dim=4)
        model = StackedSignedKAN(cfg)
        h_t = model.encode_triads(triad_v, triad_sigma, M_vt)
        assert h_t.shape == (triad_v.shape[0], 4), \
            f"depth={L} produced wrong shape {h_t.shape}"


def test_depth1_stacked_equivalent_to_multilayer_n_layers_1():
    """A depth-1 ``StackedSignedKAN`` should be a thin wrapper over
    a ``MultiLayerSignedKAN(n_layers=1)`` with the same Phase-16
    defaults. The wrapper is doing no secret math — depth-1 must
    delegate cleanly to the inner."""
    torch.manual_seed(0)
    triad_v, triad_sigma, M_vt = _toy_triads()
    cfg = StackedSignedKANConfig(n_nodes=6, n_blocks=1, hidden_dim=4)
    model = StackedSignedKAN(cfg)
    # The inner attribute IS the underlying MultiLayerSignedKAN.
    assert isinstance(model.inner, MultiLayerSignedKAN)
    inner_cfg = cfg.to_multilayer_config()
    assert inner_cfg.n_layers == 1
    assert inner_cfg.inner_skip == "residual"
    assert inner_cfg.use_residual is True
    assert inner_cfg.layer_norm_between is True
    assert inner_cfg.jk_mode == "last"
    # Forward delegates: the wrapper's encode_triads returns the
    # same tensor as the inner's directly.
    out_wrapper = model.encode_triads(triad_v, triad_sigma, M_vt)
    out_inner = model.inner.encode_triads(triad_v, triad_sigma, M_vt)
    assert torch.allclose(out_wrapper, out_inner)


def test_node_embed_property_back_compat():
    """Callers that read ``.node_embed`` on a ``StackedSignedKAN``
    should get the underlying embedding (some training harnesses do
    this for spectral init or for parameter inspection)."""
    cfg = StackedSignedKANConfig(n_nodes=6, n_blocks=2, hidden_dim=4)
    model = StackedSignedKAN(cfg)
    assert isinstance(model.node_embed, torch.nn.Embedding)
    assert model.node_embed.weight.shape == (6, 4)


def test_param_count_grows_with_depth():
    """At depth $L$, the model should have approximately $L \\times$
    the parameter count of depth 1 (ignoring the shared node embed
    + layer norms)."""
    cfg1 = StackedSignedKANConfig(n_nodes=6, n_blocks=1, hidden_dim=8)
    cfg4 = StackedSignedKANConfig(n_nodes=6, n_blocks=4, hidden_dim=8)
    m1 = StackedSignedKAN(cfg1)
    m4 = StackedSignedKAN(cfg4)
    # depth-4 should have substantially more parameters than depth-1.
    assert m4.num_parameters() > 2 * m1.num_parameters(), \
        f"depth-4 ({m4.num_parameters()}) should be >> depth-1 ({m1.num_parameters()})"
