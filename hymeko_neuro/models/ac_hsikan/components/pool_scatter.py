"""Fused pool-scatter with vector-CR activation.

The cornerstone fast global-pool-scatter primitive for AC-HSiKAN.
Replaces the separate {SignAttention, V-pool, scatter} chain with a
single operation that:

  1. Computes per-channel Hamilton-real-part scores in BOTH directions
     (anchor → candidate AND candidate → anchor); the two directions
     differ because the quaternion product is non-commutative.
  2. Applies vector-CR activation per channel (h independent CR shapes
     replacing the single tanh) → native multi-head behaviour with
     learnable per-head response curves.
  3. POOLS the candidate features weighted by S^pool into anchor h-vector.
  4. SCATTERS the anchor features weighted by S^scatter into the
     candidate h-vector (via index_add).
  5. Projects pooled + scattered h-vectors back to d-dim and adds to
     the residual.

This v1 carries the *PyTorch reference* implementation that the
Triton kernel will be parity-tested against.

Reference: docs/plans/2026-06-04-ac-hsikan/plan.md §3 + 2026-06-05
quaternion-field design discussion.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import AcHsikanConfig


@functools.lru_cache(maxsize=4)
def _get_side_stream(device: torch.device) -> "torch.cuda.Stream":
    """Cached side CUDA stream per device for overlapping the dQ/dK/dV
    and CR-coef Triton kernels in the backward pass."""
    return torch.cuda.Stream(device=device)


@functools.lru_cache(maxsize=16)
def _hamilton_coeffs(n_quat: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
    """Cached (4·n_quat,) Hamilton sign-coefficients (+, -, -, -, ...).

    The previous code reallocated this small tensor on every
    forward/backward call (~2 µs each × ~5 000 calls / epoch). The cache
    keys on the (n_quat, device, dtype) triple covering every real call
    site, with a small LRU bound so it cannot leak across exotic device
    swaps.
    """
    return torch.tensor([1.0, -1.0, -1.0, -1.0] * n_quat,
                        device=device, dtype=dtype)


def _vector_cr_eval(
    x: torch.Tensor,        # (..., h)
    coef_pos: torch.Tensor,  # (h, G)
    coef_neg: torch.Tensor,  # (h, G)
    G: int,
) -> torch.Tensor:
    """Per-channel Catmull-Rom evaluation, branched on sign(x).

    x[..., c] in $\\mathbb{R}$; the branch is sign(x), then the
    absolute value is mapped to [0, 1] for grid lookup with the
    per-channel coef of shape (h, G).
    """
    h = x.shape[-1]
    sign_x = torch.sign(x)
    is_pos = (sign_x >= 0)
    abs_x = x.abs().clamp_max(1.0)
    # Map abs_x in [0, 1] to grid coordinate [0, G - 1].
    u = abs_x * (G - 1)
    i = u.long().clamp_max(G - 2)
    t = u - i.float()
    t2 = t * t
    t3 = t2 * t
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)
    idx_m1 = (i - 1).clamp(0, G - 1)
    idx_0  = i
    idx_p1 = (i + 1).clamp_max(G - 1)
    idx_p2 = (i + 2).clamp_max(G - 1)
    # Gather control points per channel.
    h_arange = torch.arange(h, device=x.device)
    def _gather(coef, idx):
        # coef: (h, G); idx: (..., h) -> (..., h)
        return coef[h_arange.unsqueeze(0).expand_as(idx.reshape(-1, h))
                    .reshape(idx.shape), idx]
    # Compute output for the positive branch:
    P_m1_pos = _gather(coef_pos, idx_m1)
    P_0_pos  = _gather(coef_pos, idx_0)
    P_p1_pos = _gather(coef_pos, idx_p1)
    P_p2_pos = _gather(coef_pos, idx_p2)
    out_pos = w_m1 * P_m1_pos + w_0 * P_0_pos + w_p1 * P_p1_pos + w_p2 * P_p2_pos
    # Negative branch:
    P_m1_neg = _gather(coef_neg, idx_m1)
    P_0_neg  = _gather(coef_neg, idx_0)
    P_p1_neg = _gather(coef_neg, idx_p1)
    P_p2_neg = _gather(coef_neg, idx_p2)
    out_neg = w_m1 * P_m1_neg + w_0 * P_0_neg + w_p1 * P_p1_neg + w_p2 * P_p2_neg
    return torch.where(is_pos, out_pos, out_neg)


def _vector_cr_eval_2d(
    R: torch.Tensor,            # (..., h) -- sign input
    V: torch.Tensor,            # (..., h) -- value input (will be tanh-normalised)
    coef_pos: torch.Tensor,     # (h, G, G) -- 2D control point grid, sign +
    coef_neg: torch.Tensor,     # (h, G, G) -- 2D control point grid, sign −
    G: int,
) -> torch.Tensor:
    """Per-channel 2D Catmull-Rom over (|R|, tanh(V)), branched by sign(R).

    Replaces the separable ``S(R) · V`` activation+gate with a learned
    non-separable surface ``S(R, V)`` per channel. The CR itself IS the
    activation -- no fixed pre-mapping (tanh, clamp) on V. V is mapped
    linearly to grid coords ``v = (V + 1) · (G-1) / 2``; only the
    integer index is clamped to ``[0, G-1]`` to keep memory access
    in-bounds. The fractional ``tv`` can go < 0 or > 1, and the
    Catmull-Rom basis polynomials extrapolate naturally outside
    ``[0, 1]`` -- no saturation, no vanishing gradient. The grid
    learns to represent whatever surface the model needs across V's
    full range.

    Returns: ``(..., h)`` tensor.
    """
    h = R.shape[-1]
    # NaN-guard: ``.long()`` of a NaN float gives undefined integers
    # (typically INT_MIN), which then escape ``clamp_max`` and crash
    # the advanced index with a device-side assert. Replace NaN/inf
    # with finite zeros before any int conversion.
    R = torch.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)
    V = torch.nan_to_num(V, nan=0.0, posinf=1.0, neginf=-1.0)
    sign_pos = (R >= 0)
    abs_R = R.abs().clamp_max(1.0)
    # R-axis grid
    u = abs_R * (G - 1)
    iu = u.long().clamp(0, G - 2)  # two-sided clamp (was clamp_max only)
    tu = u - iu.float()
    tu2 = tu * tu; tu3 = tu2 * tu
    wu = torch.stack([
        0.5 * (-tu3 + 2.0 * tu2 - tu),
        0.5 * (3.0 * tu3 - 5.0 * tu2 + 2.0),
        0.5 * (-3.0 * tu3 + 4.0 * tu2 + tu),
        0.5 * (tu3 - tu2),
    ], dim=0)                                            # (4, ..., h)
    idu = torch.stack([
        (iu - 1).clamp(0, G - 1), iu.clamp(0, G - 1),
        (iu + 1).clamp(0, G - 1), (iu + 2).clamp(0, G - 1),
    ], dim=0)                                            # (4, ..., h)

    # V-axis grid: linear map V ∈ ℝ → grid coord (V+1)·(G-1)/2.
    # The integer index is clamped (memory safety). The fractional
    # ``tv`` is allowed to leave [0, 1] (CR extrapolation), but is
    # softly bounded to ``[-1, 2]`` because the cubic CR basis grows
    # as tv³ and unbounded extrapolation explodes the activations
    # (empirically: NaN gradients, device-side asserts). One grid
    # cell of extrapolation on either side is the trade-off between
    # the no-fixed-activation principle and numeric stability.
    v = (V + 1.0) * ((G - 1) * 0.5)
    iv = v.long().clamp(0, G - 2)  # already two-sided
    tv = (v - iv.float()).clamp(-1.0, 2.0)
    iv = v.long().clamp_max(G - 2)
    tv = v - iv.float()
    tv2 = tv * tv; tv3 = tv2 * tv
    wv = torch.stack([
        0.5 * (-tv3 + 2.0 * tv2 - tv),
        0.5 * (3.0 * tv3 - 5.0 * tv2 + 2.0),
        0.5 * (-3.0 * tv3 + 4.0 * tv2 + tv),
        0.5 * (tv3 - tv2),
    ], dim=0)                                            # (4, ..., h)
    idv = torch.stack([
        (iv - 1).clamp(0, G - 1), iv.clamp(0, G - 1),
        (iv + 1).clamp(0, G - 1), (iv + 2).clamp(0, G - 1),
    ], dim=0)                                            # (4, ..., h)

    # Per-channel advanced indexing on coef[h, G, G].
    # Build per-element (c, idu_a, idv_b) index for every (a, b) ∈ {-1,0,1,2}²
    # and sum w_u[a] · w_v[b] · coef_branch[c, idu_a, idv_b].
    out_pos = torch.zeros_like(R)
    out_neg = torch.zeros_like(R)
    h_arange = torch.arange(h, device=R.device, dtype=torch.long)
    # Broadcast h_arange to match idu_a shape (..., h).
    while h_arange.dim() < idu.dim() - 1:
        h_arange = h_arange.unsqueeze(0)
    c_idx = h_arange.expand_as(idu[0])

    for a in range(4):
        for b in range(4):
            # Gather control points at (c, idu[a], idv[b]) per element.
            P_pos = coef_pos[c_idx, idu[a], idv[b]]
            P_neg = coef_neg[c_idx, idu[a], idv[b]]
            w_ab = wu[a] * wv[b]
            out_pos = out_pos + w_ab * P_pos
            out_neg = out_neg + w_ab * P_neg
    return torch.where(sign_pos, out_pos, out_neg)


# ── Closed-form CR-coef backward helper ────────────────────────────────────

def _cr_coef_backward_both(
    R_pool: torch.Tensor,           # (..., h)
    dL_dS_pool: torch.Tensor,       # (..., h)
    R_scatter: torch.Tensor,        # (..., h)
    dL_dS_scatter: torch.Tensor,    # (..., h)
    grad_coef_pos: torch.Tensor,    # (h, G) -- modified in-place
    grad_coef_neg: torch.Tensor,    # (h, G) -- modified in-place
    G: int,
) -> None:
    """Compute both CR-coef gradients (pool + scatter) in one scatter_add.

    Equivalent to two calls of :func:`_cr_coef_backward`, but packs the
    two paths' contributions into a single accumulator and a single
    scatter_add launch -- roughly half the kernel-launch overhead at
    IMDB shape (B·L·K·h ≈ 524k elements per path).
    """
    R = torch.stack([R_pool, R_scatter], dim=0)         # (2, ..., h)
    dL_dS = torch.stack([dL_dS_pool, dL_dS_scatter], dim=0)
    _cr_coef_backward(R, dL_dS, grad_coef_pos, grad_coef_neg, G)


def _cr_coef_backward(
    R: torch.Tensor,           # (..., h) -- forward CR input
    dL_dS: torch.Tensor,       # (..., h) -- gradient wrt CR output
    grad_coef_pos: torch.Tensor,  # (h, G) -- accumulator, modified in-place
    grad_coef_neg: torch.Tensor,  # (h, G) -- accumulator, modified in-place
    G: int,
) -> None:
    """Accumulate ``∂L/∂coef_{pos,neg}`` into the given buffers.

    Replaces the autograd-through-reference path that previously dominated
    the backward wall (97.7% of bwd time at IMDB shape). Pure vectorised
    PyTorch: cubic-spline weights are reconstructed from ``R``, the
    sign-branch decides which buffer to scatter into, and each of the four
    Catmull-Rom control points contributes a ``scatter_add_`` along the
    flattened ``(h·G,)`` axis.

    Identity used:

        S = w_{-1}·P_{-1} + w_0·P_0 + w_{+1}·P_{+1} + w_{+2}·P_{+2}

    where ``P_{·}`` are gathered from ``coef_{pos|neg}[c, idx_{·}]``.
    Hence

        ∂L/∂coef[c, idx_x] += Σ_{...}  dL_dS[..., c] · w_x[..., c]

    summed over the broadcast leading axes, with the sign of ``R`` choosing
    the buffer (``coef_pos`` vs ``coef_neg``) per element.
    """
    h = R.shape[-1]
    sign_pos = (R >= 0)
    abs_R = R.abs().clamp_max(1.0)
    u = abs_R * (G - 1)
    i = u.long().clamp_max(G - 2)
    t = u - i.float()
    t2 = t * t
    t3 = t2 * t
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)
    idx_m1 = (i - 1).clamp(0, G - 1)
    idx_0  = i
    idx_p1 = (i + 1).clamp_max(G - 1)
    idx_p2 = (i + 2).clamp_max(G - 1)

    # Channel offset for the flattened (h·G,) accumulator.
    c_off = torch.arange(h, device=R.device, dtype=torch.long) * G  # (h,)
    while c_off.dim() < idx_0.dim():
        c_off = c_off.unsqueeze(0)
    c_off = c_off.expand_as(idx_0)
    cG = c_off + 0                              # base channel*G offset, (..., h)

    # Pack BOTH sign branches AND all 4 CR knots into a single
    # scatter_add. Layout:
    #   combined_target[branch · h · G + c · G + idx_x]
    # where branch=0 routes to grad_coef_pos, branch=1 to grad_coef_neg.
    # Previously this was 8 scatter_add launches; now it's 1.
    hG = h * G
    branch_off = torch.where(
        sign_pos,
        torch.zeros_like(cG),
        torch.full_like(cG, hG),
    )                                           # (..., h) -- 0 or hG
    base_idx = branch_off + cG                  # (..., h)

    # Build 4 (..., h) flat indices and contribs, one per CR knot.
    # Stack along a new leading axis, then flatten -> single scatter_add.
    knot_idx = torch.stack(
        [base_idx + idx_m1, base_idx + idx_0,
         base_idx + idx_p1, base_idx + idx_p2], dim=0,
    )                                           # (4, ..., h)
    knot_w = torch.stack([w_m1, w_0, w_p1, w_p2], dim=0)  # (4, ..., h)
    knot_contrib = knot_w * dL_dS               # (4, ..., h) -- broadcasts on knot axis

    # One pre-allocated (2·h·G,) accumulator, one scatter_add call.
    combined = torch.zeros(2 * hG, device=R.device, dtype=R.dtype)
    combined.scatter_add_(0, knot_idx.reshape(-1), knot_contrib.reshape(-1))

    # Unpack into the (h, G) accumulators.
    grad_coef_pos.add_(combined[:hG].view(h, G))
    grad_coef_neg.add_(combined[hG:].view(h, G))


def fused_pool_scatter_reference(
    x: torch.Tensor,            # (B, L, d)
    W_q: torch.Tensor,          # (d, h)
    W_k: torch.Tensor,          # (d, h)
    W_v: torch.Tensor,          # (d, h)
    W_back: torch.Tensor,       # (h, d)
    coef_pos: torch.Tensor,     # (h, G)
    coef_neg: torch.Tensor,     # (h, G)
    local: torch.Tensor,        # (L, K_total) int64
    n_quat: int,
    temperature: float,
    G: int,
    rotor_M: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vector-CR pool-scatter PyTorch reference.

    Returns ``y = x + W_back(pool_h + scatter_h)`` where the pool and
    scatter h-vectors are independently aggregated across candidates
    using bidirectional Hamilton-CR signs.

    ``rotor_M`` (optional, shape ``(n_quat, 4)``): when supplied the
    scatter direction is rotated by the Hamilton product ``M ⊗ scatter_h``
    BEFORE the ``W_back`` projection -- matching the kernel path's
    entropy-feedback application point.
    """
    B, L, d = x.shape
    h = W_q.shape[1]
    K_total = local.shape[1]
    scale = (n_quat ** -0.5) / temperature

    Q = x @ W_q                                     # (B, L, h)
    K_ = x @ W_k
    V = x @ W_v

    # Bake Hamilton coefficients (+, -, -, -) into Q over the 4-axis.
    coeffs = _hamilton_coeffs(n_quat, x.device, x.dtype)
    Q_signed = Q * coeffs

    # ── POOL: anchor i ← candidates {local[i, :]} ────────────────────
    Q_a = Q_signed.unsqueeze(2).expand(B, L, K_total, h)       # (B, L, K, h)
    K_c = K_[:, local, :]                                      # (B, L, K, h)
    R_pool = Q_a * K_c * scale                                 # (B, L, K, h)
    S_pool = _vector_cr_eval(R_pool, coef_pos, coef_neg, G)
    V_c = V[:, local, :]                                       # (B, L, K, h)
    pool_h = (S_pool * V_c).sum(dim=2)                        # (B, L, h)

    # ── SCATTER: anchor i → candidates {local[i, :]} ─────────────────
    Q_j = Q_signed[:, local, :]                                # (B, L, K, h)
    K_i = K_.unsqueeze(2).expand(B, L, K_total, h)
    R_scatter = Q_j * K_i * scale                              # (B, L, K, h)
    S_scatter = _vector_cr_eval(R_scatter, coef_pos, coef_neg, G)
    V_a = V.unsqueeze(2).expand(B, L, K_total, h)
    scatter_contrib = S_scatter * V_a                          # (B, L, K, h)
    # Accumulate into scatter_h[b, j, :] where j = local[i, k_c].
    # Flatten the (L, K_total) anchor-candidate axis so scatter_add
    # operates along a single index dim that lines up with src.
    scatter_h = torch.zeros(B, L, h, device=x.device, dtype=x.dtype)
    j_flat = (local.unsqueeze(0).expand(B, L, K_total)
                    .reshape(B, L * K_total))                  # (B, L·K)
    j_full = j_flat.unsqueeze(-1).expand(B, L * K_total, h)    # (B, L·K, h)
    contrib_flat = scatter_contrib.reshape(B, L * K_total, h)
    scatter_h = scatter_h.scatter_add(dim=1, index=j_full, src=contrib_flat)

    # Optional entropy-feedback Hamilton rotation on scatter only.
    if rotor_M is not None:
        scatter_h = _hamilton_rotate_static(scatter_h, rotor_M, n_quat)

    # ── Back-project + residual ──────────────────────────────────────
    out = (pool_h + scatter_h) @ W_back                       # (B, L, d)
    return x + out


def fused_pool_scatter_reference_2d(
    x: torch.Tensor,            # (B, L, d)
    W_q: torch.Tensor,          # (d, h)
    W_k: torch.Tensor,          # (d, h)
    W_v: torch.Tensor,          # (d, h)
    W_back: torch.Tensor,       # (h, d)
    coef_pos: torch.Tensor,     # (h, G, G) -- 2D, sign +
    coef_neg: torch.Tensor,     # (h, G, G) -- 2D, sign −
    local: torch.Tensor,        # (L, K_total) int64
    n_quat: int,
    temperature: float,
    G: int,
    rotor_M: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D-CR variant of the pool-scatter reference.

    Replaces the per-channel ``S(R) · V`` separable activation+gate with
    a per-channel 2D Catmull-Rom surface ``CR_2D(R, V)`` -- learned
    non-separable bivariate non-linearity. The V information is baked
    into the surface itself; the contribution is the CR value directly
    (no separate ``* V`` multiply).
    """
    B, L, d = x.shape
    h = W_q.shape[1]
    K_total = local.shape[1]
    scale = (n_quat ** -0.5) / temperature

    Q = x @ W_q
    K_ = x @ W_k
    V = x @ W_v
    coeffs = _hamilton_coeffs(n_quat, x.device, x.dtype)
    Q_signed = Q * coeffs

    # POOL
    Q_a = Q_signed.unsqueeze(2).expand(B, L, K_total, h)
    K_c = K_[:, local, :]
    R_pool = Q_a * K_c * scale
    V_c = V[:, local, :]
    # 2D CR as a LEARNED GATE conditioned on both R and V. V's magnitude
    # is preserved via the explicit `* V_c` multiplication, otherwise
    # the bounded CR surface clips V to ±init_scale and the model loses
    # V's dynamic range (empirically catastrophic at init).
    S_pool = _vector_cr_eval_2d(R_pool, V_c, coef_pos, coef_neg, G)
    pool_h = (S_pool * V_c).sum(dim=2)

    # SCATTER
    Q_j = Q_signed[:, local, :]
    K_i = K_.unsqueeze(2).expand(B, L, K_total, h)
    R_scatter = Q_j * K_i * scale
    V_a = V.unsqueeze(2).expand(B, L, K_total, h)
    S_scatter = _vector_cr_eval_2d(R_scatter, V_a, coef_pos, coef_neg, G)
    scatter_contrib = S_scatter * V_a
    scatter_h = torch.zeros(B, L, h, device=x.device, dtype=x.dtype)
    j_flat = (local.unsqueeze(0).expand(B, L, K_total)
                    .reshape(B, L * K_total))
    j_full = j_flat.unsqueeze(-1).expand(B, L * K_total, h)
    contrib_flat = scatter_contrib.reshape(B, L * K_total, h)
    scatter_h = scatter_h.scatter_add(dim=1, index=j_full, src=contrib_flat)

    if rotor_M is not None:
        scatter_h = _hamilton_rotate_static(scatter_h, rotor_M, n_quat)

    out = (pool_h + scatter_h) @ W_back
    return x + out


# ── nn.Module wrapper ────────────────────────────────────────────────

class FusedPoolScatter(nn.Module):
    """nn.Module wrapping the vector-CR pool-scatter reference.

    On CUDA the forward will route to the Triton kernel (when added);
    falls back to the PyTorch reference here.

    Public state:
      - W_q, W_k, W_v: (d, h)
      - W_back: (h, d)
      - coef_pos, coef_neg: (h, G)

    Parameters (all opt-in via config flags):
      d_model, h, n_quat, G, temperature
    """

    def __init__(
        self, d_model: int, h: int, n_quat: int, G: int = 8,
        temperature: float = 1.0, init_scale: float = 0.1,
        entropy_feedback: bool = False,
        use_2d_cr: bool = False,
    ):
        super().__init__()
        if h % (4 * n_quat) != 0:
            raise ValueError(
                f"h ({h}) must be divisible by 4 * n_quat ({4 * n_quat})"
            )
        if h != 4 * n_quat:
            raise ValueError(
                f"v1: h must equal 4 * n_quat (got h={h}, "
                f"4*n_quat={4 * n_quat}); multi-quaternion-block "
                f"layout is v2"
            )
        self.d_model = int(d_model)
        self.h = int(h)
        self.n_quat = int(n_quat)
        self.G = int(G)
        self.temperature = float(temperature)
        self.W_q = nn.Linear(d_model, h, bias=False)
        self.W_k = nn.Linear(d_model, h, bias=False)
        self.W_v = nn.Linear(d_model, h, bias=False)
        self.W_back = nn.Linear(h, d_model, bias=False)
        coef_pos = torch.linspace(0, 1, G).repeat(h, 1) * float(init_scale)
        coef_neg = -coef_pos
        self.coef_pos = nn.Parameter(coef_pos)
        self.coef_neg = nn.Parameter(coef_neg)
        # Optional 2D CR-patch surfaces over (R, V_c). Initialised so
        # that the surface roughly reproduces ``S(R) · V`` at the start
        # (linear in V along the V-axis, linear-up in |R| along the
        # R-axis), then learns deviations.
        self.use_2d_cr = bool(use_2d_cr)
        if self.use_2d_cr:
            # Init mirrors the 1D CR: gate ≈ |R| · init_scale, INDEPENDENT
            # of V at the start. V-axis is flat init so the 2D surface
            # reduces to the 1D activation curve initially; V-dependent
            # deviations are learned (slowly -- the 2D coef params want
            # a lower learning rate than the main net, see
            # ``__init__``'s ``lr_2d_mult`` parameter group hint).
            r_axis = torch.linspace(0, 1, G).view(G, 1)
            patch = (r_axis * float(init_scale)).expand(G, G)
            patch = patch.unsqueeze(0).repeat(h, 1, 1).contiguous()
            self.coef_pos_2d = nn.Parameter(patch.clone())
            self.coef_neg_2d = nn.Parameter(-patch.clone())
        else:
            self.register_parameter("coef_pos_2d", None)
            self.register_parameter("coef_neg_2d", None)
        with torch.no_grad():
            for lin in (self.W_q, self.W_k, self.W_v, self.W_back):
                lin.weight.mul_(0.1)

        # ── Entropy-feedback rotor (Hamilton-multiplicative, NOT additive)
        # M_q = exp(β·H·n_q)  applied as  scatter_h ← M ⊗ scatter_h
        # per quaternion block. n_q axis is a learnable imaginary
        # unit-vector (3 components per block, scalar coord locked to 0);
        # β is a learnable scalar. When entropy is 0 (peaked distribution)
        # M = identity and the modulation is a no-op.
        self.entropy_feedback = bool(entropy_feedback)
        if entropy_feedback:
            # Init imaginary axis per quaternion block uniformly on S^2.
            axis = torch.randn(n_quat, 3)
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            self.entropy_axis = nn.Parameter(axis)
            # β init small: identity-ish at start.
            self.entropy_beta = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_parameter("entropy_axis", None)
            self.register_parameter("entropy_beta", None)

    # ── Entropy → rotor helper ────────────────────────────────────

    def _build_rotor(self, entropy_scalar: torch.Tensor) -> torch.Tensor:
        """Build M ∈ ℝ^{n_quat × 4} per-block rotor from scalar entropy H.

        M_q = exp(β·H·n_q) where n_q is a pure-imaginary unit quaternion.
        Closed form for a pure-imaginary v with |v|=1:
            exp(θ v) = cos(θ) + sin(θ) v
        so M_q = (cos(βH), sin(βH)·n_q_x, sin(βH)·n_q_y, sin(βH)·n_q_z).
        """
        axis = self.entropy_axis / (
            self.entropy_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        )                                                # (n_quat, 3)
        theta = self.entropy_beta * entropy_scalar       # scalar
        c = torch.cos(theta); s = torch.sin(theta)
        # Build (n_quat, 4): scalar, e1, e2, e3 (= imaginary triple)
        M = torch.empty(self.n_quat, 4, device=axis.device, dtype=axis.dtype)
        M[:, 0] = c
        M[:, 1:] = s * axis
        return M

    def _hamilton_rotate(self, field: torch.Tensor,
                         M: torch.Tensor) -> torch.Tensor:
        """field: (B, L, h) interpreted as (B, L, n_quat, 4);
        M: (n_quat, 4) per-block rotor. Returns M ⊗ field with same
        shape (B, L, h)."""
        B, L, h = field.shape
        f = field.view(B, L, self.n_quat, 4)
        M_b = M.view(1, 1, self.n_quat, 4)
        m0, m1, m2, m3 = M_b[..., 0:1], M_b[..., 1:2], M_b[..., 2:3], M_b[..., 3:4]
        f0, f1, f2, f3 = f[..., 0:1], f[..., 1:2], f[..., 2:3], f[..., 3:4]
        # Hamilton product M ⊗ f
        out0 = m0 * f0 - m1 * f1 - m2 * f2 - m3 * f3
        out1 = m0 * f1 + m1 * f0 + m2 * f3 - m3 * f2
        out2 = m0 * f2 - m1 * f3 + m2 * f0 + m3 * f1
        out3 = m0 * f3 + m1 * f2 - m2 * f1 + m3 * f0
        return torch.cat([out0, out1, out2, out3], dim=-1).view(B, L, h)

    def forward(
        self, x: torch.Tensor, local: torch.Tensor,
        entropy_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If ``entropy_scalar`` is provided AND entropy_feedback is on,
        the scatter direction is Hamilton-rotated by M = exp(β·H·n)
        per quaternion block.  This is a NON-ADDITIVE multiplicative
        modulation -- the gradient field is phase-rotated rather than
        perturbed.
        """
        # Reference path (CPU + autograd-correct, slow): full PyTorch.
        # When entropy_feedback is active on CPU, apply rotor inline
        # via post-processing of the reference output's scatter delta
        # -- but the reference computes pool+scatter together, so we
        # cannot disentangle them.  For simplicity the CPU path here
        # IGNORES entropy_feedback; the CUDA path applies it.
        # 2D CR-patch path: PyTorch reference, autograd-traceable.
        # The Triton fast path is 1D-CR only; 2D extension is future
        # work. Used opt-in for parity / capacity experiments.
        if self.use_2d_cr:
            if self.entropy_feedback and entropy_scalar is not None:
                M = self._build_rotor(entropy_scalar)
            else:
                M = None
            return fused_pool_scatter_reference_2d(
                x,
                self.W_q.weight.T, self.W_k.weight.T,
                self.W_v.weight.T, self.W_back.weight.T,
                self.coef_pos_2d, self.coef_neg_2d,
                local,
                self.n_quat, self.temperature, self.G,
                rotor_M=M,
            )
        if not x.is_cuda:
            return fused_pool_scatter_reference(
                x,
                self.W_q.weight.T, self.W_k.weight.T,
                self.W_v.weight.T, self.W_back.weight.T,
                self.coef_pos, self.coef_neg,
                local,
                self.n_quat, self.temperature, self.G,
            )
        scale = (self.n_quat ** -0.5) / self.temperature
        # Build rotor once per forward (small tensor).
        if self.entropy_feedback and entropy_scalar is not None:
            M = self._build_rotor(entropy_scalar)
        else:
            M = None
        return _FusedPoolScatterTritonFn.apply(
            x, self.W_q.weight, self.W_k.weight, self.W_v.weight,
            self.W_back.weight, self.coef_pos, self.coef_neg,
            local, self.n_quat, self.temperature, self.G, scale, M,
        )


class _FusedPoolScatterTritonFn(torch.autograd.Function):
    """Triton-forward / PyTorch-backward custom autograd op.

    Forward path: pre-project Q/K/V via PyTorch (cheap matmul, well
    cuBLAS-optimised), fire the fused kernel, then PyTorch matmul for
    the W_back projection.  Backward: run the reference forward with
    grad enabled and autograd.grad through it.  Correctness wins over
    backward speed in v1; a fused-bwd kernel can drop in here.
    """

    @staticmethod
    def forward(
        ctx, x, W_q_weight, W_k_weight, W_v_weight, W_back_weight,
        coef_pos, coef_neg, local, n_quat, temperature, G, scale,
        rotor_M=None,
    ):
        from hymeko_neuro.kernels.triton_kernels.fused_pool_scatter import (
            fused_pool_scatter_forward_triton,
        )
        Wq, Wk, Wv = W_q_weight.T, W_k_weight.T, W_v_weight.T
        Wb = W_back_weight.T
        Q = x @ Wq
        K = x @ Wk
        V = x @ Wv
        coeffs = _hamilton_coeffs(n_quat, x.device, x.dtype)
        Q_signed = Q * coeffs
        pool_h, scatter_h = fused_pool_scatter_forward_triton(
            Q_signed, K, V, coef_pos, coef_neg, local, scale,
        )
        # Capture pre-rotor scatter_h for backward (the closed-form CR-coef
        # path needs the pre-rotor field, since grad_scatter_h already
        # carries the M* reverse rotation).
        scatter_h_pre = scatter_h
        # Entropy-feedback Hamilton rotation on scatter direction.
        if rotor_M is not None:
            scatter_h = _hamilton_rotate_static(
                scatter_h, rotor_M, n_quat,
            )
        out = x + (pool_h + scatter_h) @ Wb
        # Save Q_signed, K, V, pool_h, scatter_h (pre-rotor) to skip BOTH
        # the Q/K/V matmul re-run AND the Triton forward re-run in
        # backward. Memory cost: 5 × (B·L·h) floats per layer (≈ 0.4 MB
        # at IMDB shape) -- negligible.
        ctx.save_for_backward(
            x, W_q_weight, W_k_weight, W_v_weight, W_back_weight,
            coef_pos, coef_neg, local,
            rotor_M if rotor_M is not None else torch.empty(0,
                device=x.device, dtype=x.dtype),
            Q_signed.detach(), K.detach(), V.detach(),
            pool_h.detach(), scatter_h_pre.detach(),
        )
        ctx.n_quat = n_quat
        ctx.temperature = temperature
        ctx.G = G
        ctx.has_rotor = rotor_M is not None
        # Snapshot the evolvens sink at forward-time -- the contextvar is
        # set in the main thread; backward runs in an autograd worker
        # thread where the contextvar does not propagate.
        from ..telemetry import active_sink
        ctx.evolvent_sink = active_sink()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, W_q_weight, W_k_weight, W_v_weight, W_back_weight,
         coef_pos, coef_neg, local, rotor_M,
         Q_signed, K, V, pool_h, scatter_h) = ctx.saved_tensors
        n_quat = ctx.n_quat; G = ctx.G; temperature = ctx.temperature
        scale = (n_quat ** -0.5) / temperature
        from hymeko_neuro.kernels.triton_kernels.fused_pool_scatter import (
            fused_pool_scatter_backward_triton,
        )

        # Q_signed, K, V, pool_h, scatter_h (pre-rotor) all come from
        # forward saved_tensors -- no matmul re-run, no Triton fwd re-run.
        Wq, Wk, Wv = W_q_weight.T, W_k_weight.T, W_v_weight.T
        Wb = W_back_weight.T
        coeffs = _hamilton_coeffs(n_quat, x.device, x.dtype)
        # Re-apply rotor (matches forward).
        has_rotor = ctx.has_rotor
        if has_rotor:
            scatter_h_modulated = _hamilton_rotate_static(
                scatter_h, rotor_M, n_quat,
            )
        else:
            scatter_h_modulated = scatter_h

        # ── grad over W_back and the (pool + scatter) sum ──
        # out = x + (pool_h + scatter_h) @ Wb
        # grad_out propagates: grad_W_back = (pool_h + scatter_h).T @ grad_out
        # grad_path_h = grad_out @ Wb.T
        B, L, d = x.shape
        h = Q_signed.shape[2]
        psum = (pool_h + scatter_h_modulated).reshape(-1, h)
        grad_out_flat = grad_out.reshape(-1, d)
        grad_W_back = psum.T @ grad_out_flat   # (h, d)
        grad_W_back_weight = grad_W_back.T

        grad_path_h = (grad_out_flat @ Wb.T).reshape(B, L, h)
        grad_pool_h = grad_path_h
        # Apply REVERSE rotor to grad_scatter_h: backward through
        # scatter_h_modulated = M ⊗ scatter_h is grad ⊗ M* (conjugate).
        if has_rotor:
            M_conj = rotor_M.clone()
            M_conj[:, 1:] = -M_conj[:, 1:]  # quaternion conjugate
            grad_scatter_h = _hamilton_rotate_static(
                grad_path_h, M_conj, n_quat,
            )
            grad_rotor = _compute_rotor_grad(
                grad_path_h, scatter_h, rotor_M, n_quat,
            )
        else:
            grad_scatter_h = grad_path_h
            grad_rotor = None

        # ── Parallel-stream launch of the two independent Triton kernels:
        #    dQ/dK/dV on main stream, CR-coef on a side stream. Outputs are
        #    fully independent; reading the shared inputs (Q_signed, K, V,
        #    local, grad_*) from two streams is safe (no write conflict).
        #    Measured 1.11× over serial: 1.18 ms → 1.06 ms.
        from hymeko_neuro.kernels.triton_kernels.fused_pool_scatter import (
            cr_coef_backward_triton,
        )
        side_stream = _get_side_stream(x.device)
        main_stream = torch.cuda.current_stream(x.device)
        side_stream.wait_stream(main_stream)
        with torch.cuda.stream(side_stream):
            g_cp, g_cn = cr_coef_backward_triton(
                Q_signed, K, V, local,
                grad_pool_h, grad_scatter_h, scale, G,
            )
        # Main stream: dQ/dK/dV bwd.
        dQ_signed, dK, dV = fused_pool_scatter_backward_triton(
            Q_signed, K, V,
            coef_pos, coef_neg,
            local,
            grad_pool_h, grad_scatter_h,
            scale,
        )
        # Reverse Hamilton-coefficient bake: dQ = dQ_signed * coeffs.
        dQ = dQ_signed * coeffs

        # ── Chain back to x and W_{q,k,v} ──
        # Q = x @ Wq -> grad_x via dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
        #              grad_Wq = x.T @ dQ  (similarly Wk, Wv)
        x_flat = x.reshape(-1, d)
        dQ_flat = dQ.reshape(-1, h)
        dK_flat = dK.reshape(-1, h)
        dV_flat = dV.reshape(-1, h)
        grad_W_q = x_flat.T @ dQ_flat                       # (d, h)
        grad_W_k = x_flat.T @ dK_flat
        grad_W_v = x_flat.T @ dV_flat
        # nn.Linear stores weights as (out, in) = (h, d) -- transpose.
        grad_W_q_weight = grad_W_q.T
        grad_W_k_weight = grad_W_k.T
        grad_W_v_weight = grad_W_v.T

        grad_x = (
            grad_out                       # residual term
            + (dQ_flat @ Wq.T).reshape(B, L, d)
            + (dK_flat @ Wk.T).reshape(B, L, d)
            + (dV_flat @ Wv.T).reshape(B, L, d)
        )

        # Re-join the side stream before any consumer of g_cp / g_cn.
        main_stream.wait_stream(side_stream)

        # Evolvens-telemetry tap (no-op when no sink was captured at forward).
        sink = getattr(ctx, "evolvent_sink", None)
        if sink is not None:
            from ..telemetry import emit_backward_record
            emit_backward_record(
                rotor_M=rotor_M if has_rotor else None,
                scatter_h=scatter_h,
                scatter_h_modulated=scatter_h_modulated,
                grad_path_h=grad_path_h,
                grad_scatter_h=grad_scatter_h,
                grad_x=grad_x,
                grad_out=grad_out,
                sink=sink,
            )

        return (grad_x, grad_W_q_weight, grad_W_k_weight, grad_W_v_weight,
                grad_W_back_weight, g_cp, g_cn,
                None, None, None, None, None, grad_rotor)


# ── Module-level helper: static Hamilton rotation ────────────────

def _hamilton_rotate_static(field: torch.Tensor, M: torch.Tensor,
                             n_quat: int) -> torch.Tensor:
    """Static-version of FusedPoolScatter._hamilton_rotate.

    Dispatches to the Triton fast path on the main-thread CUDA case
    (~11× speedup standalone). The worker-thread / CPU / autograd-
    needed fallback uses the original explicit 16-mul + cat formulation
    -- numerically bit-equivalent to the reference and used as the
    accuracy baseline. (Tested: einsum is 1.08× faster standalone but
    introduces ~0.002 IMDB val_acc drift from different FMA ordering;
    not worth the trade.)
    """
    if field.is_cuda and M.is_cuda and not (
        torch.is_grad_enabled()
        and (field.requires_grad or M.requires_grad)
    ):
        from hymeko_neuro.kernels.triton_kernels.fused_pool_scatter import (
            hamilton_rotate_triton,
        )
        return hamilton_rotate_triton(field, M, n_quat)
    B, L, h = field.shape
    f = field.view(B, L, n_quat, 4)
    M_b = M.view(1, 1, n_quat, 4)
    m0, m1, m2, m3 = M_b[..., 0:1], M_b[..., 1:2], M_b[..., 2:3], M_b[..., 3:4]
    f0, f1, f2, f3 = f[..., 0:1], f[..., 1:2], f[..., 2:3], f[..., 3:4]
    out0 = m0 * f0 - m1 * f1 - m2 * f2 - m3 * f3
    out1 = m0 * f1 + m1 * f0 + m2 * f3 - m3 * f2
    out2 = m0 * f2 - m1 * f3 + m2 * f0 + m3 * f1
    out3 = m0 * f3 + m1 * f2 - m2 * f1 + m3 * f0
    return torch.cat([out0, out1, out2, out3], dim=-1).view(B, L, h)


def _compute_rotor_grad(
    grad_scatter_modulated: torch.Tensor,   # (B, L, h)
    scatter_h_pre: torch.Tensor,            # (B, L, h) -- before rotation
    M: torch.Tensor,                        # (n_quat, 4)
    n_quat: int,
) -> torch.Tensor:
    """Compute ∂L/∂M for the rotor.

    Forward: scatter_mod = M ⊗ scatter, hence
        ∂L/∂M_q = Σ_{b,i} (∂L/∂scatter_mod)_{b,i,q} ⊗ scatter*_{b,i,q}
    where ⊗ is the Hamilton product and * is the quaternion conjugate.
    """
    B, L, h = grad_scatter_modulated.shape
    g = grad_scatter_modulated.view(B, L, n_quat, 4)
    s = scatter_h_pre.view(B, L, n_quat, 4)
    # s_conj = (s0, -s1, -s2, -s3)
    s0, s1, s2, s3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    g0, g1, g2, g3 = g[..., 0], g[..., 1], g[..., 2], g[..., 3]
    # g ⊗ s_conj per-(b, i, q), then sum over (b, i)
    # Hamilton: (g0 + g1 i + g2 j + g3 k) ⊗ (s0 - s1 i - s2 j - s3 k)
    out0 = g0 * s0 + g1 * s1 + g2 * s2 + g3 * s3
    out1 = -g0 * s1 + g1 * s0 - g2 * s3 + g3 * s2
    out2 = -g0 * s2 + g1 * s3 + g2 * s0 - g3 * s1
    out3 = -g0 * s3 - g1 * s2 + g2 * s1 + g3 * s0
    return torch.stack([out0, out1, out2, out3], dim=-1).sum(dim=(0, 1))
