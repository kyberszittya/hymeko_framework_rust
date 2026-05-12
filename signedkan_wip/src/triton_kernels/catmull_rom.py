"""Auto-split from signedkan_wip/src/triton_kernels.py 2026-05-11.
CLAUDE.md §6.5 #4 (no monstrosity > 300 LOC).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _catmull_rom_kernel(
    coef_ptr,           # (C, G) per-channel control points (flattened)
    x_ptr,              # (N, C) inputs in [-1, 1] (flattened)
    out_ptr,            # (N, C) output (flattened)
    N, C, G,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Tile (BLOCK_N, BLOCK_C) of inputs → outputs.

    For each (n, c) tile element:
      1. Clamp x_{n,c} to [-1, 1]
      2. Map to grid coord u = (x + 1) · (G - 1) / 2
      3. i = floor(u),  t = u − i
      4. Evaluate 4 CR basis weights from t, t^2, t^3
      5. Gather 4 control points coef[c, i±k] for k ∈ {-1, 0, 1, 2}
      6. Output = w_m1·P_m1 + w_0·P_0 + w_p1·P_p1 + w_p2·P_p2

    Boundary handling: clamp gather indices into [0, G-1] (matches
    PyTorch reference's `_catmull_rom_eval`).
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C
    mask = mask_n[:, None] & mask_c[None, :]

    # Load x_{n, c}.
    x_offs = offs_n[:, None] * C + offs_c[None, :]
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

    # Clamp to [-1, 1] and map to grid.
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    u = (x + 1.0) * 0.5 * (G - 1)
    # Integer segment index, fractional offset.
    i = u.to(tl.int32)
    # tl.minimum on integers + clamping to [0, G-2].
    i = tl.minimum(i, G - 2)
    t = u - i.to(tl.float32)
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis weights.
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)

    # Gather indices, clamped to [0, G-1].
    idx_m1 = tl.minimum(tl.maximum(i - 1, 0), G - 1)
    idx_0  = i
    idx_p1 = tl.minimum(i + 1, G - 1)
    idx_p2 = tl.minimum(i + 2, G - 1)

    # Compute base offsets into coef: coef[c, idx_*] flattened →
    # offs_c * G + idx_*.  Need (BLOCK_N, BLOCK_C) shape via
    # broadcasting of offs_c.
    cG = offs_c[None, :] * G                         # (1, BLOCK_C)
    P_m1 = tl.load(coef_ptr + cG + idx_m1, mask=mask, other=0.0)
    P_0  = tl.load(coef_ptr + cG + idx_0,  mask=mask, other=0.0)
    P_p1 = tl.load(coef_ptr + cG + idx_p1, mask=mask, other=0.0)
    P_p2 = tl.load(coef_ptr + cG + idx_p2, mask=mask, other=0.0)

    out = w_m1 * P_m1 + w_0 * P_0 + w_p1 * P_p1 + w_p2 * P_p2
    tl.store(out_ptr + x_offs, out, mask=mask)


def catmull_rom_triton(
    coef: torch.Tensor,
    x: torch.Tensor,
    grid: int,
) -> torch.Tensor:
    """Triton-fused Catmull-Rom evaluation.  Matches the signature of
    `splines._catmull_rom_eval` exactly.

    Parameters
    ----------
    coef : (..., C, G) per-channel control points
    x    : (..., C)    inputs in [-1, 1]
    grid : G

    Returns
    -------
    out : (..., C) spline values
    """
    if not x.is_cuda:
        raise RuntimeError("catmull_rom_triton requires CUDA inputs")
    # Only the shared-coef case is supported by the kernel: coef is
    # (C, G), not per-batch.  Detect by checking dimensionality;
    # genuinely per-batch coef (BatchedCatmullRomActivation, etc.)
    # falls back to the PyTorch reference because each row needs a
    # different (C, G) load and the kernel is not yet vectorised
    # over per-row coef.
    if coef.dim() != 2:
        from ..splines import _catmull_rom_eval as _ref
        return _ref(coef, x, grid)
    C = coef.shape[0]
    if grid != coef.shape[1]:
        raise RuntimeError(
            f"grid ({grid}) must match coef.shape[1] ({coef.shape[1]})"
        )
    if x.shape[-1] != C:
        raise RuntimeError(
            f"last dim of x ({x.shape[-1]}) must match coef.shape[0] "
            f"({C})"
        )
    *batch_shape, _ = x.shape
    x_flat = x.reshape(-1, C).contiguous()
    coef_flat = coef.contiguous()
    out = torch.empty_like(x_flat)
    N = x_flat.shape[0]
    BLOCK_N = 64
    BLOCK_C = min(64, triton.next_power_of_2(C))
    grid_lambda = (
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(C, BLOCK_C),
    )
    _catmull_rom_kernel[grid_lambda](
        coef_flat, x_flat, out,
        N, C, grid,
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
    )
    return out.reshape(*batch_shape, C)


# ─── Parity test against PyTorch reference ──────────────────────────


class _CatmullRomTritonFn(torch.autograd.Function):
    """Autograd-aware Triton-fused Catmull-Rom evaluation.

    Forward: Triton kernel (fast).
    Backward: PyTorch reference computes gradients (slower, but
    correct).  Future work can replace the backward with a Triton
    kernel of its own; for now correctness > speed on the backward.
    """

    @staticmethod
    def forward(ctx, coef, x, grid):
        ctx.save_for_backward(coef, x)
        ctx.grid = grid
        return catmull_rom_triton(coef, x, grid)

    @staticmethod
    def backward(ctx, grad_out):
        coef, x = ctx.saved_tensors
        grid = ctx.grid
        # Use the PyTorch reference to compute the backward.  Re-run
        # forward with grad-enabled inputs to get the autograd graph.
        from ..splines import _catmull_rom_eval as _ref
        coef_g = coef.detach().requires_grad_(True)
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _ref(coef_g, x_g, grid)
            grads = torch.autograd.grad(
                out, (coef_g, x_g), grad_out,
                retain_graph=False, create_graph=False,
            )
        return grads[0], grads[1], None


def catmull_rom_triton_autograd(coef: torch.Tensor, x: torch.Tensor,
                                  grid: int) -> torch.Tensor:
    """Autograd-wrapped Triton-fused Catmull-Rom evaluation."""
    return _CatmullRomTritonFn.apply(coef, x, grid)


def install_triton_catmull_rom():
    """Monkey-patch `splines._catmull_rom_eval` to dispatch to the
    Triton kernel when inputs are on CUDA, falling back to the
    PyTorch reference on CPU.  Idempotent: safe to call multiple
    times.  Returns the original function for restoration.
    """
    from .. import splines
    if hasattr(splines, "_orig_catmull_rom_eval"):
        return splines._orig_catmull_rom_eval  # already installed
    splines._orig_catmull_rom_eval = splines._catmull_rom_eval

    def _dispatch(coef, x, grid):
        if x.is_cuda and coef.is_cuda and coef.dim() == 2:
            try:
                # Use autograd-wrapped variant so gradients flow back.
                return catmull_rom_triton_autograd(coef, x, grid)
            except Exception:
                return splines._orig_catmull_rom_eval(coef, x, grid)
        return splines._orig_catmull_rom_eval(coef, x, grid)

    splines._catmull_rom_eval = _dispatch
    return splines._orig_catmull_rom_eval


def uninstall_triton_catmull_rom():
    """Restore the PyTorch-only `_catmull_rom_eval`."""
    from .. import splines
    if hasattr(splines, "_orig_catmull_rom_eval"):
        splines._catmull_rom_eval = splines._orig_catmull_rom_eval
        del splines._orig_catmull_rom_eval


if __name__ == "__main__":
    import json
    print("--- parity ---")
    print(json.dumps(_parity_check_catmull_rom(), indent=2))
    print("--- benchmark (HSiKAN-realistic shapes) ---")
    for B in (10_000, 50_000, 200_000):
        for C in (4, 16):
            print(json.dumps(
                _benchmark_catmull_rom(C=C, G=8, B=B), indent=0,
            ))

