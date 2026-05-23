"""Auto-split from signedkan_wip/src/triton_kernels.py 2026-05-11.
CLAUDE.md §6.5 #4 (no monstrosity > 300 LOC).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

from .dispatch import _triton_backward_enabled
from .inner import signedkan_inner_triton
from .inner_highway import signedkan_inner_highway_triton
from .inner_backward import signedkan_inner_backward_triton
from .inner_highway_backward import signedkan_inner_highway_backward_triton

class _SignedKANInnerTritonFn(torch.autograd.Function):
    """Autograd-aware fused inner-kernel.

    Forward: Triton kernel.
    Backward: fused Triton kernel (closed-form gradients per
    docs/triton_kernel_integration_tutorial_2026_05_09.md §10.2).
    Set HSIKAN_TRITON_BACKWARD=0 to fall back to a PyTorch reference
    re-computation (slower; useful for debugging).
    """

    @staticmethod
    def forward(ctx, x, triad_v, triad_sigma, coef_pos, coef_neg, grid):
        ctx.save_for_backward(x, triad_v, triad_sigma, coef_pos, coef_neg)
        ctx.grid = grid
        return signedkan_inner_triton(
            x, triad_v, triad_sigma, coef_pos, coef_neg, grid,
        )

    @staticmethod
    def backward(ctx, grad_out):
        import os
        x, triad_v, triad_sigma, coef_pos, coef_neg = ctx.saved_tensors
        grid = ctx.grid
        if _triton_backward_enabled():
            grad_x, grad_cp, grad_cn = signedkan_inner_backward_triton(
                x, triad_v, triad_sigma, coef_pos, coef_neg,
                grad_out.contiguous(), grid,
            )
            return grad_x, None, None, grad_cp, grad_cn, None
        # PyTorch fallback (correctness reference).
        from ..core.splines import _catmull_rom_eval as _ref
        x_g = x.detach().requires_grad_(True)
        coef_pos_g = coef_pos.detach().requires_grad_(True)
        coef_neg_g = coef_neg.detach().requires_grad_(True)
        with torch.enable_grad():
            h_v = x_g[triad_v]
            inner_pos = _ref(coef_pos_g, h_v, grid)
            inner_neg = _ref(coef_neg_g, h_v, grid)
            inner_all = torch.stack([inner_pos, inner_neg], dim=2)
            sign_vals = torch.tensor(
                [1, -1], device=x.device, dtype=torch.int64,
            )
            masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
            counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
            out = (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts
            grads = torch.autograd.grad(
                out, (x_g, coef_pos_g, coef_neg_g), grad_out,
                retain_graph=False, create_graph=False,
            )
        return grads[0], None, None, grads[1], grads[2], None


def signedkan_inner_triton_autograd(
    x, triad_v, triad_sigma, coef_pos, coef_neg, grid,
):
    """Autograd-wrapped fused inner kernel.  Drop-in replacement for
    the gather + per-sign CR + σ-mask + mean reduce that appears in
    `SignedKANLayer._forward_impl` when ``inner_skip == "none"``."""
    return _SignedKANInnerTritonFn.apply(
        x, triad_v, triad_sigma, coef_pos, coef_neg, grid,
    )


class _SignedKANInnerHighwayTritonFn(torch.autograd.Function):
    """Autograd-aware fused inner-kernel with Highway skip.

    Forward: Triton kernel.
    Backward: PyTorch reference re-runs forward under autograd.
    """

    @staticmethod
    def forward(ctx, x, triad_v, triad_sigma,
                coef_pos, coef_neg, gate_w, gate_b, grid):
        ctx.save_for_backward(
            x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b,
        )
        ctx.grid = grid
        return signedkan_inner_highway_triton(
            x, triad_v, triad_sigma, coef_pos, coef_neg,
            gate_w, gate_b, grid,
        )

    @staticmethod
    def backward(ctx, grad_out):
        import os
        (x, triad_v, triad_sigma, coef_pos, coef_neg,
         gate_w, gate_b) = ctx.saved_tensors
        grid = ctx.grid
        if _triton_backward_enabled():
            (grad_x, grad_cp, grad_cn,
             grad_gw, grad_gb) = signedkan_inner_highway_backward_triton(
                x, triad_v, triad_sigma, coef_pos, coef_neg,
                gate_w, gate_b, grad_out.contiguous(), grid,
            )
            return (grad_x, None, None,
                    grad_cp, grad_cn, grad_gw, grad_gb, None)
        # PyTorch fallback (correctness reference).
        from ..core.splines import _catmull_rom_eval as _ref
        x_g = x.detach().requires_grad_(True)
        coef_pos_g = coef_pos.detach().requires_grad_(True)
        coef_neg_g = coef_neg.detach().requires_grad_(True)
        gate_w_g = gate_w.detach().requires_grad_(True)
        gate_b_g = gate_b.detach().requires_grad_(True)
        with torch.enable_grad():
            h_v = x_g[triad_v]
            T_inner = torch.sigmoid(h_v @ gate_w_g + gate_b_g)
            inner_pos = _ref(coef_pos_g, h_v, grid)
            inner_neg = _ref(coef_neg_g, h_v, grid)
            h_pos = T_inner * inner_pos + (1 - T_inner) * h_v
            h_neg = T_inner * inner_neg + (1 - T_inner) * h_v
            inner_all = torch.stack([h_pos, h_neg], dim=2)
            sign_vals = torch.tensor(
                [1, -1], device=x.device, dtype=torch.int64,
            )
            masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
            counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
            out = (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts
            grads = torch.autograd.grad(
                out,
                (x_g, coef_pos_g, coef_neg_g, gate_w_g, gate_b_g),
                grad_out,
                retain_graph=False, create_graph=False,
            )
        return (grads[0], None, None,
                grads[1], grads[2], grads[3], grads[4], None)


def signedkan_inner_highway_triton_autograd(
    x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b, grid,
):
    """Autograd-wrapped Highway fused inner kernel.  Drop-in for the
    inner-highway path of ``SignedKANLayer._forward_impl``."""
    return _SignedKANInnerHighwayTritonFn.apply(
        x, triad_v, triad_sigma, coef_pos, coef_neg,
        gate_w, gate_b, grid,
    )

