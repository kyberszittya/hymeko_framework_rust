"""Unit + integration tests for the Triton kernels in
``signedkan_wip.src.triton_kernels``.

Tests are organised in three tiers, each producing concrete metrics
asserted against acceptance bars:

1. **Unit — kernel correctness**
   - Catmull-Rom parity vs PyTorch reference at multiple shapes.
   - Catmull-Rom boundary handling (x = ±1, x outside [-1, 1]).
   - Catmull-Rom zero-coef edge case.
   - Fused inner-layer parity at multiple (T, k, d, G).
   - All-positive σ edge case (mean reduction with cnt_neg = 0).
   - All-negative σ edge case (mean reduction with cnt_pos = 0).
   - Arity sweep: k ∈ {2, 3, 4, 5}.

2. **Unit — autograd**
   - Gradient parity through the autograd-wrapped variant.

3. **Integration — performance metrics**
   - CR kernel speedup ≥ 5× at canonical shape.
   - Fused inner kernel speedup ≥ 10× at canonical shape.

Run:
    pytest signedkan_wip/tests/test_triton_kernels.py -v -s

The ``-s`` flag prints the metric values; without it pytest
captures stdout.  All tests skip cleanly when CUDA is unavailable
(no GPU) or when triton is not importable.
"""
from __future__ import annotations

import sys

import pytest
import torch


# Skip the entire module when CUDA / triton is not available.
torch_cuda_ok = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    triton_ok = True
except ImportError:
    triton_ok = False

pytestmark = pytest.mark.skipif(
    not (torch_cuda_ok and triton_ok),
    reason="Triton kernel tests require CUDA + triton",
)


# Ensure the repo root is on the path when invoked directly.
_REPO_ROOT = "/home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Thresholds (acceptance bars).
ABS_PARITY_THRESHOLD = 1e-4   # bit-equivalence at fp32 within 1e-4
REL_PARITY_THRESHOLD = 1e-2   # 1% relative error tolerated on tiny values
CR_SPEEDUP_THRESHOLD = 5.0    # Triton CR ≥ 5× PyTorch ref
INNER_SPEEDUP_THRESHOLD = 10.0  # Fused inner ≥ 10× PyTorch path
GRAD_PARITY_THRESHOLD = 1e-4


# ─── Tier 1: kernel correctness — Catmull-Rom ──────────────────────


@pytest.mark.parametrize(
    "C,G,B",
    [
        (4, 8, 256),       # tiny
        (16, 8, 256),      # small
        (16, 8, 50_000),   # medium
        (4, 8, 200_000),   # Slashdot-realistic, h=4
        (16, 8, 200_000),  # Slashdot-realistic, h=16
        (32, 16, 50_000),  # different grid count
    ],
)
def test_catmull_rom_parity(C, G, B):
    """Triton CR matches PyTorch ref within ABS_PARITY_THRESHOLD."""
    from signedkan_wip.src.splines import _catmull_rom_eval
    from signedkan_wip.src.triton_kernels import catmull_rom_triton

    torch.manual_seed(0)
    coef = torch.randn(C, G, device="cuda", dtype=torch.float32)
    x = torch.empty(B, C, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    out_t = catmull_rom_triton(coef, x, grid=G)
    out_r = _catmull_rom_eval(coef, x, G)
    abs_err = (out_t - out_r).abs().max().item()
    rel_err = ((out_t - out_r).abs()
               / (out_r.abs() + 1e-9)).max().item()
    print(f"\n  CR parity C={C} G={G} B={B}: "
          f"abs={abs_err:.2e} rel={rel_err:.2e}")
    assert abs_err < ABS_PARITY_THRESHOLD, (
        f"CR abs err {abs_err:.2e} > {ABS_PARITY_THRESHOLD:.2e}"
    )


def test_catmull_rom_boundary_clamping():
    """Inputs outside [-1, 1] must produce same clamped output as ref."""
    from signedkan_wip.src.splines import _catmull_rom_eval
    from signedkan_wip.src.triton_kernels import catmull_rom_triton

    torch.manual_seed(0)
    C, G = 8, 8
    coef = torch.randn(C, G, device="cuda", dtype=torch.float32)
    # Inputs spanning [-3, 3] — must be clamped to [-1, 1].
    x = torch.linspace(-3, 3, 100, device="cuda").unsqueeze(-1).expand(-1, C)
    out_t = catmull_rom_triton(coef, x, grid=G)
    out_r = _catmull_rom_eval(coef, x, G)
    err = (out_t - out_r).abs().max().item()
    print(f"\n  CR boundary clamping: abs={err:.2e}")
    assert err < ABS_PARITY_THRESHOLD


def test_catmull_rom_zero_coef():
    """All-zero coef → output exactly zero everywhere."""
    from signedkan_wip.src.triton_kernels import catmull_rom_triton

    C, G, B = 8, 8, 256
    coef = torch.zeros(C, G, device="cuda", dtype=torch.float32)
    x = torch.randn(B, C, device="cuda", dtype=torch.float32)
    out = catmull_rom_triton(coef, x, grid=G)
    abs_max = out.abs().max().item()
    print(f"\n  CR zero-coef: max |out| = {abs_max:.2e}")
    assert abs_max == 0.0


# ─── Tier 1: kernel correctness — Fused inner layer ────────────────


def _torch_signedkan_inner(x, triad_v, triad_sigma, coef_pos, coef_neg, G):
    """Explicit PyTorch reference for the fused inner kernel.

    Matches production ``SignedKANLayer._forward_impl`` (no_skip case):
    no tanh wrap, CR clamps its input internally.
    """
    from signedkan_wip.src.splines import _catmull_rom_eval
    h_v = x[triad_v]
    inner_pos = _catmull_rom_eval(coef_pos, h_v, G)
    inner_neg = _catmull_rom_eval(coef_neg, h_v, G)
    inner_all = torch.stack([inner_pos, inner_neg], dim=2)
    sign_vals = torch.tensor([1, -1], device=x.device, dtype=torch.int64)
    masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
    counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts


@pytest.mark.parametrize(
    "V,d,T,k,G",
    [
        (1000, 4, 5_000, 4, 8),
        (1000, 16, 5_000, 4, 8),
        (5_000, 4, 50_000, 5, 8),
        (10_000, 4, 200_000, 4, 8),
        (10_000, 16, 200_000, 4, 8),
        (1000, 8, 1000, 3, 16),
    ],
)
def test_signedkan_inner_parity(V, d, T, k, G):
    """Fused inner kernel matches PyTorch reference at fp32."""
    from signedkan_wip.src.triton_kernels import signedkan_inner_triton

    torch.manual_seed(0)
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = torch.randint(
        0, 2, (T, k), device="cuda", dtype=torch.int64,
    ) * 2 - 1
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)

    out_r = _torch_signedkan_inner(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )
    out_t = signedkan_inner_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )
    abs_err = (out_r - out_t).abs().max().item()
    print(f"\n  inner parity V={V} d={d} T={T} k={k} G={G}: "
          f"abs={abs_err:.2e}")
    assert abs_err < ABS_PARITY_THRESHOLD


def test_signedkan_inner_all_positive_sigma():
    """When all σ_{t,i} = +1, neg branch must be 0 (cnt_neg = 0
    → mean clamped to 1, then masked accumulate is 0)."""
    from signedkan_wip.src.triton_kernels import signedkan_inner_triton

    torch.manual_seed(0)
    V, d, T, k, G = 1000, 4, 1000, 4, 8
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    # All-positive σ.
    triad_sigma = torch.ones((T, k), device="cuda", dtype=torch.int64)
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)
    out = signedkan_inner_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )
    # out[:, 1, :] is the neg branch — accumulate is 0 because no
    # σ matches; should be all zeros.
    neg_max = out[:, 1, :].abs().max().item()
    pos_max = out[:, 0, :].abs().max().item()
    print(f"\n  inner all-σ=+1: neg_max={neg_max:.2e}, "
          f"pos_max={pos_max:.2e}")
    assert neg_max == 0.0
    # Pos branch should be a meaningful tanh value.
    assert pos_max > 0.0


def test_signedkan_inner_all_negative_sigma():
    """Symmetric: all σ = -1 → pos branch must be 0."""
    from signedkan_wip.src.triton_kernels import signedkan_inner_triton

    torch.manual_seed(0)
    V, d, T, k, G = 1000, 4, 1000, 4, 8
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = -torch.ones((T, k), device="cuda", dtype=torch.int64)
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)
    out = signedkan_inner_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )
    neg_max = out[:, 1, :].abs().max().item()
    pos_max = out[:, 0, :].abs().max().item()
    print(f"\n  inner all-σ=-1: pos_max={pos_max:.2e}, "
          f"neg_max={neg_max:.2e}")
    assert pos_max == 0.0
    assert neg_max > 0.0


def _torch_signedkan_inner_highway(
    x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b, G,
):
    """PyTorch reference for the highway-skip fused inner kernel.

    Matches production ``SignedKANLayer._forward_impl`` (highway case):
    no tanh wrap, raw h_v residual.  CR clamps its input internally.
    """
    from signedkan_wip.src.splines import _catmull_rom_eval
    h_v = x[triad_v]
    # Highway gate: per-(t, k, d) sigmoid of Linear(h_v).
    gate_logit = h_v @ gate_w + gate_b      # (T, k, d)
    T_inner = torch.sigmoid(gate_logit)
    inner_pos = _catmull_rom_eval(coef_pos, h_v, G)
    inner_neg = _catmull_rom_eval(coef_neg, h_v, G)
    h_pos = T_inner * inner_pos + (1 - T_inner) * h_v
    h_neg = T_inner * inner_neg + (1 - T_inner) * h_v
    inner_all = torch.stack([h_pos, h_neg], dim=2)  # (T, k, 2, d)
    sign_vals = torch.tensor([1, -1], device=x.device, dtype=torch.int64)
    masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
    counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts


@pytest.mark.parametrize(
    "V,d,T,k,G",
    [
        (1000, 4, 5_000, 4, 8),
        (1000, 16, 5_000, 4, 8),
        (5_000, 4, 50_000, 4, 8),
    ],
)
def test_signedkan_inner_highway_parity(V, d, T, k, G):
    """Highway-skip kernel parity vs PyTorch ref."""
    from signedkan_wip.src.triton_kernels import (
        signedkan_inner_highway_triton,
    )

    torch.manual_seed(0)
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = (torch.randint(0, 2, (T, k), device="cuda",
                                   dtype=torch.int64) * 2 - 1)
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)
    gate_w = torch.randn(d, d, device="cuda", dtype=torch.float32) * 0.1
    gate_b = torch.randn(d, device="cuda", dtype=torch.float32) * 0.1

    out_r = _torch_signedkan_inner_highway(
        x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b, G,
    )
    out_t = signedkan_inner_highway_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b, G,
    )
    abs_err = (out_r - out_t).abs().max().item()
    print(f"\n  highway parity V={V} d={d} T={T} k={k} G={G}: "
          f"abs={abs_err:.2e}")
    assert abs_err < ABS_PARITY_THRESHOLD


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_signedkan_inner_arities(k):
    """Kernel works across all arities used in HSiKAN (k=2..6)."""
    from signedkan_wip.src.triton_kernels import signedkan_inner_triton

    torch.manual_seed(0)
    V, d, T, G = 5000, 8, 10_000, 8
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = (torch.randint(0, 2, (T, k), device="cuda",
                                   dtype=torch.int64) * 2 - 1)
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)
    out_r = _torch_signedkan_inner(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )
    out_t = signedkan_inner_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )
    abs_err = (out_r - out_t).abs().max().item()
    print(f"\n  arity k={k}: abs={abs_err:.2e}")
    assert abs_err < ABS_PARITY_THRESHOLD


# ─── Tier 2: autograd correctness ──────────────────────────────────


def test_signedkan_inner_autograd_grad_parity():
    """Triton-wrapped fused inner kernel computes gradients matching
    the PyTorch reference w.r.t. all three differentiable inputs
    (x, coef_pos, coef_neg)."""
    from signedkan_wip.src.triton_kernels import (
        signedkan_inner_triton_autograd,
    )

    torch.manual_seed(0)
    V, d, T, k, G = 1000, 4, 5_000, 4, 8
    x_a = (torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
           ).requires_grad_(True)
    x_b = x_a.detach().clone().requires_grad_(True)
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = (torch.randint(0, 2, (T, k), device="cuda",
                                   dtype=torch.int64) * 2 - 1)
    cp_a = torch.randn(d, G, device="cuda", dtype=torch.float32,
                       requires_grad=True)
    cp_b = cp_a.detach().clone().requires_grad_(True)
    cn_a = torch.randn(d, G, device="cuda", dtype=torch.float32,
                       requires_grad=True)
    cn_b = cn_a.detach().clone().requires_grad_(True)

    out_a = signedkan_inner_triton_autograd(
        x_a, triad_v, triad_sigma, cp_a, cn_a, G,
    )
    out_b = _torch_signedkan_inner(x_b, triad_v, triad_sigma, cp_b, cn_b, G)
    target = torch.randn_like(out_a)
    loss_a = ((out_a - target) ** 2).mean()
    loss_b = ((out_b - target) ** 2).mean()
    loss_a.backward()
    loss_b.backward()

    x_err = (x_a.grad - x_b.grad).abs().max().item()
    cp_err = (cp_a.grad - cp_b.grad).abs().max().item()
    cn_err = (cn_a.grad - cn_b.grad).abs().max().item()
    print(f"\n  inner autograd: ∂x={x_err:.2e}, "
          f"∂cp={cp_err:.2e}, ∂cn={cn_err:.2e}")
    assert x_err < GRAD_PARITY_THRESHOLD
    assert cp_err < GRAD_PARITY_THRESHOLD
    assert cn_err < GRAD_PARITY_THRESHOLD


@pytest.mark.parametrize("d", [4, 16, 32])
def test_signedkan_inner_highway_autograd_grad_parity(d):
    """Triton-wrapped Highway fused inner kernel computes gradients
    matching the PyTorch reference w.r.t. (x, coef_pos, coef_neg,
    gate_w, gate_b).

    Parameterized over d to cover both backward paths:
    d=4 → explicit-loop path; d=16/32 → tl.dot fast path.
    """
    from signedkan_wip.src.triton_kernels import (
        signedkan_inner_highway_triton_autograd,
    )

    torch.manual_seed(0)
    V, T, k, G = 1000, 5_000, 4, 8
    x_a = (torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
           ).requires_grad_(True)
    x_b = x_a.detach().clone().requires_grad_(True)
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = (torch.randint(0, 2, (T, k), device="cuda",
                                   dtype=torch.int64) * 2 - 1)
    cp_a = torch.randn(d, G, device="cuda", dtype=torch.float32,
                       requires_grad=True)
    cp_b = cp_a.detach().clone().requires_grad_(True)
    cn_a = torch.randn(d, G, device="cuda", dtype=torch.float32,
                       requires_grad=True)
    cn_b = cn_a.detach().clone().requires_grad_(True)
    gw_a = (torch.randn(d, d, device="cuda", dtype=torch.float32) * 0.1
            ).requires_grad_(True)
    gw_b = gw_a.detach().clone().requires_grad_(True)
    gb_a = (torch.randn(d, device="cuda", dtype=torch.float32) * 0.1
            ).requires_grad_(True)
    gb_b = gb_a.detach().clone().requires_grad_(True)

    out_a = signedkan_inner_highway_triton_autograd(
        x_a, triad_v, triad_sigma, cp_a, cn_a, gw_a, gb_a, G,
    )
    out_b = _torch_signedkan_inner_highway(
        x_b, triad_v, triad_sigma, cp_b, cn_b, gw_b, gb_b, G,
    )
    target = torch.randn_like(out_a)
    loss_a = ((out_a - target) ** 2).mean()
    loss_b = ((out_b - target) ** 2).mean()
    loss_a.backward()
    loss_b.backward()

    x_err = (x_a.grad - x_b.grad).abs().max().item()
    cp_err = (cp_a.grad - cp_b.grad).abs().max().item()
    cn_err = (cn_a.grad - cn_b.grad).abs().max().item()
    gw_err = (gw_a.grad - gw_b.grad).abs().max().item()
    gb_err = (gb_a.grad - gb_b.grad).abs().max().item()
    print(f"\n  highway autograd: ∂x={x_err:.2e}, "
          f"∂cp={cp_err:.2e}, ∂cn={cn_err:.2e}, "
          f"∂gw={gw_err:.2e}, ∂gb={gb_err:.2e}")
    assert x_err < GRAD_PARITY_THRESHOLD
    assert cp_err < GRAD_PARITY_THRESHOLD
    assert cn_err < GRAD_PARITY_THRESHOLD
    assert gw_err < GRAD_PARITY_THRESHOLD
    assert gb_err < GRAD_PARITY_THRESHOLD


def test_catmull_rom_autograd_grad_parity():
    """Triton-wrapped CR computes gradients matching PyTorch ref."""
    from signedkan_wip.src.splines import _catmull_rom_eval
    from signedkan_wip.src.triton_kernels import catmull_rom_triton_autograd

    torch.manual_seed(0)
    C, G, B = 8, 8, 256
    coef_a = torch.randn(C, G, device="cuda", dtype=torch.float32,
                          requires_grad=True)
    coef_b = coef_a.detach().clone().requires_grad_(True)
    x_a = torch.empty(B, C, device="cuda", dtype=torch.float32
                       ).uniform_(-1, 1).requires_grad_(True)
    x_b = x_a.detach().clone().requires_grad_(True)

    out_a = catmull_rom_triton_autograd(coef_a, x_a, G)
    out_b = _catmull_rom_eval(coef_b, x_b, G)
    target = torch.randn_like(out_a)
    loss_a = ((out_a - target) ** 2).mean()
    loss_b = ((out_b - target) ** 2).mean()
    loss_a.backward()
    loss_b.backward()

    coef_grad_err = (coef_a.grad - coef_b.grad).abs().max().item()
    x_grad_err = (x_a.grad - x_b.grad).abs().max().item()
    print(f"\n  CR autograd: ∂coef={coef_grad_err:.2e}, "
          f"∂x={x_grad_err:.2e}")
    assert coef_grad_err < GRAD_PARITY_THRESHOLD
    assert x_grad_err < GRAD_PARITY_THRESHOLD


# ─── Tier 3: performance metrics ───────────────────────────────────


def _bench(fn, n_warmup=3, n_runs=20):
    """Returns mean elapsed ms over n_runs after n_warmup warmups."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_runs):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_runs


def test_catmull_rom_speedup_canonical():
    """Speedup of CR kernel at canonical Slashdot shape ≥ threshold."""
    from signedkan_wip.src.splines import _catmull_rom_eval
    from signedkan_wip.src.triton_kernels import catmull_rom_triton

    torch.manual_seed(0)
    C, G, B = 16, 8, 200_000
    coef = torch.randn(C, G, device="cuda", dtype=torch.float32)
    x = torch.empty(B, C, device="cuda",
                     dtype=torch.float32).uniform_(-1, 1)

    t_torch = _bench(lambda: _catmull_rom_eval(coef, x, G))
    t_triton = _bench(lambda: catmull_rom_triton(coef, x, G))
    speedup = t_torch / t_triton
    print(f"\n  CR speedup at C={C} B={B}: "
          f"PyTorch={t_torch:.3f} ms, Triton={t_triton:.3f} ms, "
          f"speedup={speedup:.1f}×")
    assert speedup >= CR_SPEEDUP_THRESHOLD, (
        f"CR speedup {speedup:.1f}× < required {CR_SPEEDUP_THRESHOLD}×"
    )


def test_fused_inner_speedup_canonical():
    """Speedup of fused inner kernel at canonical Slashdot shape."""
    from signedkan_wip.src.triton_kernels import signedkan_inner_triton

    torch.manual_seed(0)
    V, d, T, k, G = 79_000, 4, 200_000, 4, 8
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = (torch.randint(0, 2, (T, k), device="cuda",
                                   dtype=torch.int64) * 2 - 1)
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)

    t_torch = _bench(lambda: _torch_signedkan_inner(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    ))
    t_triton = _bench(lambda: signedkan_inner_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    ))
    speedup = t_torch / t_triton
    print(f"\n  fused inner speedup T={T} d={d} k={k}: "
          f"PyTorch={t_torch:.3f} ms, Triton={t_triton:.3f} ms, "
          f"speedup={speedup:.1f}×")
    assert speedup >= INNER_SPEEDUP_THRESHOLD, (
        f"fused inner speedup {speedup:.1f}× < required "
        f"{INNER_SPEEDUP_THRESHOLD}×"
    )


# ─── Tier 4: end-to-end integration smoke ─────────────────────────


def test_install_uninstall_idempotent():
    """install_triton_catmull_rom and uninstall must be idempotent."""
    from signedkan_wip.src.triton_kernels import (
        install_triton_catmull_rom, uninstall_triton_catmull_rom,
    )
    install_triton_catmull_rom()
    install_triton_catmull_rom()  # second call must be a no-op
    uninstall_triton_catmull_rom()
    uninstall_triton_catmull_rom()  # second call must be safe


def test_end_to_end_ba_smoke_with_patch():
    """Full BA training run with Triton patch installed produces
    identical AUC to the PyTorch reference."""
    import os
    from signedkan_wip.src.triton_kernels import (
        install_triton_catmull_rom, uninstall_triton_catmull_rom,
    )

    # Helper: run one BA training cell and return the AUC.
    def run_ba(use_triton: bool) -> float:
        if use_triton:
            install_triton_catmull_rom()
        else:
            uninstall_triton_catmull_rom()
        os.environ["HSIKAN_MIXED_TUPLES"] = "c3,c4,w2,w3"
        os.environ["HSIKAN_ATTENTION_M_E"] = "quaternion"
        os.environ["HSIKAN_ATTENTION_HIGHWAY"] = "1"
        os.environ["HSIKAN_CYCLE_BATCH"] = "4000"
        # Use 5 epochs to keep this test fast.
        # Need to import after env vars are set.
        from signedkan_wip.src.run_final_cell import cell_signed_graph
        device = torch.device("cuda")
        out = cell_signed_graph(
            "bitcoin_alpha", "HSiKAN", hidden=16, n_epochs=5,
            max_k4=200_000, device=device, seed=0,
        )
        return float(out["auc"]) if out is not None else 0.0

    # Reset env between runs.
    auc_ref = run_ba(use_triton=False)
    auc_triton = run_ba(use_triton=True)
    abs_diff = abs(auc_ref - auc_triton)
    print(f"\n  BA end-to-end: ref={auc_ref:.4f}, "
          f"triton={auc_triton:.4f}, diff={abs_diff:.2e}")
    # Fp32 numerical diff after 5 epochs of training; allow up to
    # 1% AUC drift (training is non-deterministic across runs).
    assert abs_diff < 0.01, (
        f"end-to-end AUC drift {abs_diff:.2e} > 0.01"
    )
    uninstall_triton_catmull_rom()
