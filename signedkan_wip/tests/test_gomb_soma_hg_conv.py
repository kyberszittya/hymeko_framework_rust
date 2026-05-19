"""Tests for the HypergraphConv ABC (GömbSoma phase 1).

Pins the contract of the abstract message-passing primitive:

  * the ABC cannot be instantiated directly;
  * concrete subclasses with valid configs run a forward pass and
    return the documented shape;
  * preconditions reject malformed inputs with informative errors;
  * the forward map is permutation-equivariant over the vertex set
    (the GömbSoma headline invariant);
  * the default sum-pool aggregator does not materialise a dense
    |V|×|P| tensor.

A minimal dummy subclass ``MeanConv`` is used to exercise the ABC;
real WalkConv / PolygonConv / TriangleConv land in phases 2-4.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma import (
    HypergraphConv,
    HypergraphConvConfig,
)


# ---------------------------------------------------------------------
# Test fixture: a minimal concrete subclass
# ---------------------------------------------------------------------


class MeanConv(HypergraphConv):
    """Trivial concrete HypergraphConv that averages vertex features
    inside each primitive and projects to out_features.

    Used to exercise the ABC machinery. Not a GömbSoma layer.
    """

    def __init__(self, config: HypergraphConvConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(
            self.in_features, self.out_features, bias=self.config.bias,
        )

    def _forward_messages(
        self,
        x: torch.Tensor,
        primitives: torch.Tensor,
        primitive_signs: torch.Tensor,
    ) -> torch.Tensor:
        # Mean over vertices in each primitive.
        # x[primitives] is (n_prim, k_arity, in_features).
        gathered = x[primitives]
        mean_in = gathered.mean(dim=1)  # (n_prim, in_features)
        # Sign branching: route through a learnable scalar if asked.
        if self.use_sign_branching:
            mean_in = mean_in * primitive_signs.float().unsqueeze(-1)
        return self.proj(mean_in)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_inputs(n_nodes: int, k: int, n_prim: int, d_in: int, seed: int = 0):
    """Build a synthetic (x, primitives, signs, M_v) tuple."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, d_in, generator=rng)
    primitives = torch.randint(
        0, n_nodes, (n_prim, k), generator=rng, dtype=torch.long,
    )
    signs = torch.where(
        torch.rand(n_prim, generator=rng) > 0.5,
        torch.ones(n_prim, dtype=torch.int64),
        -torch.ones(n_prim, dtype=torch.int64),
    )
    # M_v sparse: one entry per (vertex-in-primitive, primitive) pair,
    # with weight 1/k_arity so the row sums to 1 over primitives that
    # contain v.
    rows = primitives.reshape(-1)
    cols = torch.arange(n_prim).repeat_interleave(k)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.full((rows.shape[0],), 1.0 / k)
    M_v = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_prim),
    ).coalesce()
    return x, primitives, signs, M_v


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_abc_cannot_be_instantiated():
    """HypergraphConv is abstract; direct instantiation must fail."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    with pytest.raises(TypeError):
        HypergraphConv(cfg)  # type: ignore[abstract]


def test_config_validates_dimensions():
    """Bad config dimensions are caught at validation time."""
    for bad in [
        dict(in_features=0, out_features=4, k_arity=3),
        dict(in_features=4, out_features=-1, k_arity=3),
        dict(in_features=4, out_features=4, k_arity=1),
    ]:
        cfg = HypergraphConvConfig(**bad)
        with pytest.raises(ValueError):
            cfg.validate()


def test_concrete_subclass_runs():
    """The fixture MeanConv runs end-to-end and returns the documented shape."""
    cfg = HypergraphConvConfig(in_features=8, out_features=16, k_arity=3)
    layer = MeanConv(cfg)
    x, prims, signs, M_v = _build_inputs(
        n_nodes=10, k=3, n_prim=20, d_in=8,
    )
    y = layer(x, prims, signs, M_v)
    assert y.shape == (10, 16), (
        f"expected (10, 16), got {tuple(y.shape)}"
    )


def test_precondition_rejects_wrong_x_shape():
    cfg = HypergraphConvConfig(in_features=8, out_features=4, k_arity=3)
    layer = MeanConv(cfg)
    x, prims, signs, M_v = _build_inputs(
        n_nodes=5, k=3, n_prim=7, d_in=4,  # WRONG d_in
    )
    with pytest.raises(ValueError, match="x has shape"):
        layer(x, prims, signs, M_v)


def test_precondition_rejects_wrong_primitive_arity():
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    layer = MeanConv(cfg)
    x, prims, signs, M_v = _build_inputs(
        n_nodes=5, k=4, n_prim=6, d_in=4,  # k=4, but layer expects 3
    )
    with pytest.raises(ValueError, match="primitives has shape"):
        layer(x, prims, signs, M_v)


def test_precondition_rejects_wrong_sign_values():
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    layer = MeanConv(cfg)
    x, prims, _signs, M_v = _build_inputs(
        n_nodes=5, k=3, n_prim=6, d_in=4,
    )
    bad_signs = torch.tensor([1, -1, 0, 1, -1, 1], dtype=torch.int64)  # 0 ∉ {-1,+1}
    with pytest.raises(ValueError, match="primitive_signs must be in"):
        layer(x, prims, bad_signs, M_v)


def test_precondition_rejects_mismatched_M_v_shape():
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    layer = MeanConv(cfg)
    x, prims, signs, _M_v = _build_inputs(
        n_nodes=5, k=3, n_prim=6, d_in=4,
    )
    # Build M_v of the WRONG shape.
    indices = torch.zeros((2, 0), dtype=torch.long)
    values = torch.zeros((0,))
    bad_M_v = torch.sparse_coo_tensor(indices, values, (5, 99)).coalesce()
    with pytest.raises(ValueError, match="M_v has shape"):
        layer(x, prims, signs, bad_M_v)


def test_permutation_equivariance():
    """The GömbSoma headline invariant: a vertex permutation of the
    input produces the same permutation of the output (up to numerical
    tolerance). This is what makes the layer a function of the
    isomorphism class of the input hypergraph."""
    cfg = HypergraphConvConfig(in_features=6, out_features=8, k_arity=3,
                                use_sign_branching=False)
    layer = MeanConv(cfg)
    layer.eval()
    torch.manual_seed(0)
    n_nodes, k, n_prim, d_in = 12, 3, 25, 6
    x, prims, signs, M_v = _build_inputs(
        n_nodes=n_nodes, k=k, n_prim=n_prim, d_in=d_in, seed=42,
    )

    # Random vertex permutation.
    perm = torch.randperm(n_nodes)
    inv = torch.argsort(perm)
    x_p = x[inv]                 # permuted features: vertex `perm[i]` of original is at row i
    prims_p = perm[prims]        # remap primitive vertex IDs
    # M_v: relabel rows.
    M_dense = M_v.to_dense()
    M_p_dense = M_dense[inv]
    M_v_p = M_p_dense.to_sparse().coalesce()

    y = layer(x, prims, signs, M_v)
    y_p = layer(x_p, prims_p, signs, M_v_p)

    # y_p[i] should equal y[perm[i]] = y[inv-inverse], i.e., the
    # permuted output should match the output of the permuted input.
    expected = y[inv]
    assert torch.allclose(y_p, expected, atol=1e-5), (
        f"permutation equivariance failed; max diff = "
        f"{(y_p - expected).abs().max().item():.2e}"
    )


def test_sparse_aggregator_does_not_materialise_dense():
    """The default _aggregate uses torch.sparse.mm, not a dense product.
    We verify by passing a sparse M_v and confirming the layer accepts
    it without densifying. A direct memory test is platform-fragile;
    we settle for: the sparse path is selected and produces the same
    answer as the manually-densified version."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    layer = MeanConv(cfg)
    layer.eval()
    x, prims, signs, M_v = _build_inputs(
        n_nodes=8, k=3, n_prim=10, d_in=4, seed=7,
    )
    assert M_v.is_sparse, "fixture must produce a sparse M_v"
    y_sparse = layer(x, prims, signs, M_v)
    # Manually densify and aggregate, compare.
    M_dense = M_v.to_dense()
    msgs = layer._forward_messages(x, prims, signs)
    y_dense = M_dense @ msgs
    assert torch.allclose(y_sparse, y_dense, atol=1e-6), (
        f"sparse vs dense aggregate disagree; max diff = "
        f"{(y_sparse - y_dense).abs().max().item():.2e}"
    )


def test_gradient_flow():
    """Backward pass must populate gradients on every learnable parameter.
    This guards against silent dead-branch parameters."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    layer = MeanConv(cfg)
    x, prims, signs, M_v = _build_inputs(
        n_nodes=6, k=3, n_prim=8, d_in=4,
    )
    y = layer(x, prims, signs, M_v)
    loss = y.pow(2).sum()
    loss.backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, (
            f"parameter {name!r} has zero / missing gradient — "
            f"dead branch?"
        )


def test_extra_repr_documents_config():
    cfg = HypergraphConvConfig(in_features=4, out_features=8, k_arity=5)
    layer = MeanConv(cfg)
    rep = repr(layer)
    for token in ("in_features=4", "out_features=8", "k_arity=5"):
        assert token in rep, f"{token!r} missing from repr: {rep}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
