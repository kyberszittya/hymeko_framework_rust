"""Regression guard: the signed-link pipeline must NOT collapse multidimensional
representations to scalars during propagation or pooling.

Root-cause finding (2026-06-17): the rotor line is bottlenecked, and a standing
concern (Dr. Hajdu) is premature scalarisation — the belief-propagation lesson
that the full vector *message* must be propagated, and only the final readout may
marginalise to a scalar logit. These tests pin that invariant so a future refactor
cannot silently scalarise the message passing (`encode_triads`) or the attention
*value* path. The final logit IS a scalar by design and is therefore not guarded.

Run: pytest -p no:randomly signedkan_wip/tests/test_multidim_preservation.py
"""
from __future__ import annotations

import torch

from signedkan_wip.src.core.geometric_triad_attention import (
    GeometricTriadAttentionPool,
    build_vertex_triad_pairs,
)
from signedkan_wip.src.core.signedkan import (
    MultiLayerSignedKAN,
    MultiLayerSignedKANConfig,
    build_vertex_triad_incidence,
)

HIDDEN = 8
N_NODES = 5
TRIAD_V = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
TRIAD_SIGMA = torch.tensor([[1, -1, -1], [1, -1, -1]], dtype=torch.long)


def _model():
    torch.manual_seed(0)
    cfg = MultiLayerSignedKANConfig(n_nodes=N_NODES, n_layers=2, hidden_dim=HIDDEN,
                                    grid=5, k=3, use_bilinear=True)
    return MultiLayerSignedKAN(cfg)


def test_encode_triads_propagates_multidim_not_scalar():
    """encode_triads must return per-triad AND per-vertex embeddings of width
    == hidden (> 1). A regression that scalarised either (feature dim → 1) is the
    premature-collapse bug this guards against."""
    model = _model()
    M_vt = build_vertex_triad_incidence(TRIAD_V.numpy(), N_NODES, torch.device("cpu"))
    out, h_v = model.encode_triads(TRIAD_V, TRIAD_SIGMA, M_vt, return_h_v=True)
    assert out.shape == (TRIAD_V.shape[0], HIDDEN)      # (T, d) — d>1, not (T,) or (T,1)
    assert h_v.shape == (N_NODES, HIDDEN)               # (V, d) — propagation kept the vector
    assert out.shape[-1] > 1 and h_v.shape[-1] > 1


def test_geometric_pool_value_path_stays_multidim():
    """The geometric attention pool collapses only the *weight* to a scalar; the
    pooled *value* must remain (V, d_out) with d_out > 1."""
    torch.manual_seed(1)
    pool = GeometricTriadAttentionPool(d_node=HIDDEN, d_triad=HIDDEN, hidden=HIDDEN,
                                       d_out=HIDDEN)
    h_node = torch.randn(N_NODES, HIDDEN)
    h_triad = torch.randn(TRIAD_V.shape[0], HIDDEN)
    inc_v, inc_t = build_vertex_triad_pairs(TRIAD_V)
    pooled = pool(h_node, h_triad, inc_v, inc_t)
    assert pooled.shape == (N_NODES, HIDDEN)
    assert pooled.shape[-1] > 1                         # value not scalarised


def test_rotor_relative_keeps_full_multivector_before_projection():
    """The rotor-relative head must form the full 4-component relative rotor per
    block (conj(q_u)⊗q_v) before its single logit projection — never a scalar-part
    truncation like the geometric score's Re()/⟨·⟩₀."""
    from signedkan_wip.src.core.rotor_relative_head import RotorRelativeHead
    from signedkan_wip.src.embeddings.cayley_rotor import (
        cayley_to_unit_quat,
        quat_conjugate,
        quat_mul,
    )
    n_blocks = 3
    head = RotorRelativeHead(n_blocks=n_blocks)
    assert head.proj.in_features == 4 * n_blocks        # full multivector, not n_blocks
    q_u = cayley_to_unit_quat(torch.randn(4, n_blocks, 3))
    q_v = cayley_to_unit_quat(torch.randn(4, n_blocks, 3))
    rel = quat_mul(quat_conjugate(q_u), q_v)
    assert rel.shape == (4, n_blocks, 4)                # 4-dim per block retained
