"""AcHsikanLayer — composes the component building blocks.

After 2026-06-04's components refactor the layer is a thin assembler:

    context  : ContextEncoder      pre-attention multivector mix
    attn     : SignAttention       per-pair sign computation
    walk_op  : WalkOp              per-arity cycle sign-product
    ffn      : FFNBlock            post-attention feed-forward

The forward pass is:

    x_ctx = context(x)
    candidates = _hybrid_indices(L, K_local, n_jumps)        # (L, K_total)
    anchor_to_cand = attn.compute_anchor_signs(x_ctx, cand)  # (B, L, K_total)
    for k in arities:
        score = walk_op.compute(anchor_to_cand, attn, x_ctx, cand, k)
        z_k = per_arity_proj[k](pooled_features) * score
    y = Σ α_k z_k
    y = layer_norm(x + dropout(y))
    return ffn(y)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .components import (
    build_context_encoder, build_ffn_block, build_sign_attention,
    build_walk_op,
)
from .config import AcHsikanConfig


def _local_indices(L: int, K: int, device) -> torch.Tensor:
    """Per-anchor positions {i±1, ..., i±K/2} clipped to [0, L)."""
    half = K // 2
    offsets = torch.tensor(
        list(range(-half, 0)) + list(range(1, K - half + 1)),
        device=device, dtype=torch.long,
    )[:K]
    anchors = torch.arange(L, device=device, dtype=torch.long).unsqueeze(1)
    return (anchors + offsets.unsqueeze(0)).clamp(0, L - 1)  # (L, K)


def _hybrid_indices(L: int, K_local: int, n_jumps: int,
                    jumps_seed: int, device) -> torch.Tensor:
    """Local neighbours + deterministic remote jumps. (L, K_local + n_jumps)."""
    near = _local_indices(L, K_local, device)
    if n_jumps <= 0:
        return near
    g = torch.Generator(device="cpu").manual_seed(int(jumps_seed))
    jumps = torch.randint(0, L - 1, (L, n_jumps), generator=g)
    anchors_cpu = torch.arange(L, dtype=torch.long).unsqueeze(1)
    jumps = jumps + (jumps >= anchors_cpu).long()
    return torch.cat([near, jumps.to(device)], dim=1)


class AcHsikanLayer(nn.Module):
    """One AC-HSiKAN block, composed of swappable components."""

    def __init__(self, cfg: AcHsikanConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Components -- each is independently testable / swappable.
        self.context = build_context_encoder(cfg)
        self.attn = build_sign_attention(cfg)
        self.walk_op = build_walk_op(cfg)
        self.ffn = build_ffn_block(cfg)
        # Per-arity output projection -- N separate ``nn.Linear`` is
        # faster than a fused batched bmm at this scale (the bmm
        # backward serialises worse for (d, d) = (16, 16) tiles than
        # 4 independent Linear backwards run concurrently on different
        # SMs). Measured 2026-06-06: fwd −0.12 ms but bwd +0.47 ms net.
        self.per_arity_proj = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.d_model) for _ in cfg.arities]
        )
        # α_k logits.
        alpha_logits = torch.log(torch.tensor(cfg.alpha_init, dtype=torch.float))
        self.alpha_logits = nn.Parameter(alpha_logits)
        # Attention block norm / dropout.
        self.layer_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        # Optional value projection (kept here, not a separate component
        # because it lives inside the attention block).
        self.value_proj = (
            nn.Linear(cfg.d_model, cfg.d_model)
            if cfg.use_value_projection else None
        )
        # Optional magnitude weight projections (orthogonal to sign).
        if cfg.use_magnitude_weight:
            self.mag_q = nn.Linear(cfg.d_model, cfg.sign_head_hidden, bias=False)
            self.mag_k = nn.Linear(cfg.d_model, cfg.sign_head_hidden, bias=False)
        else:
            self.mag_q = None
            self.mag_k = None

        # Pre-compute hybrid candidate indices once at __init__: the seed,
        # L, K, and n_jumps are all fixed, so the (L, K_total) integer
        # tensor never changes across training steps. Register as buffer
        # so .to(device) moves it with the module.
        self.register_buffer(
            "_local_indices_cache",
            _hybrid_indices(
                cfg.n_positions, cfg.top_k_per_position,
                cfg.n_jumps_per_anchor, cfg.jumps_seed,
                torch.device("cpu"),
            ),
            persistent=False,
        )

        # v1.6: optional dual-path pool-scatter (synaptic primitive)
        if cfg.use_pool_scatter:
            from .components.pool_scatter import FusedPoolScatter
            self.pool_scatter = FusedPoolScatter(
                d_model=cfg.d_model,
                h=cfg.pool_scatter_h,
                n_quat=cfg.pool_scatter_n_quat,
                G=cfg.pool_scatter_G,
                temperature=cfg.pool_scatter_temperature,
                entropy_feedback=cfg.use_pool_scatter_rotor,
                use_2d_cr=cfg.use_2d_cr,
            )
            # Learnable sigmoid gate -- 0.5 at init (both paths equal).
            self.path_gate = nn.Parameter(
                torch.tensor(cfg.pool_scatter_gate_init, dtype=torch.float)
            )
        else:
            self.pool_scatter = None
            self.path_gate = None

        # Depth-stability options (Touvron 2021 LayerScale + pre-norm).
        # Per-channel learnable scale on the residual branch -- init
        # to a tiny constant so the layer starts as near-identity and
        # the network's effective depth at init is ~1, growing with
        # training. Critical for n_layers >= 4.
        if cfg.use_layer_scale:
            self.layer_scale = nn.Parameter(
                torch.full((cfg.d_model,), cfg.layer_scale_init,
                            dtype=torch.float)
            )
        else:
            self.layer_scale = None

    # ── Public helpers ────────────────────────────────────────────

    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.alpha_logits / self.cfg.alpha_temperature,
                              dim=0)

    # ── Forward ───────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"AcHsikanLayer expects (B, L, d); got {x.shape}")
        B, L, d = x.shape
        if d != self.cfg.d_model:
            raise ValueError(
                f"d_model mismatch: got {d}, cfg {self.cfg.d_model}"
            )

        # Pre-norm: apply LayerNorm at the ENTRY of the block so the
        # residual identity carries un-normalised x. Post-norm (default)
        # applies the LayerNorm AFTER the residual sum at the exit.
        if self.cfg.use_pre_norm:
            x_normed = self.layer_norm(x)
            x_ctx = self.context(x_normed)
        else:
            x_ctx = self.context(x)

        K = self.cfg.top_k_per_position
        # Candidate index selection: static positional (default) or
        # dynamic top-K by sign-head magnitude (the AC-HSiKAN analogue
        # of ABB top-K-per-vertex on signed graphs).
        if self.cfg.use_dynamic_topk and hasattr(self.attn, "_dense"):
            # Compute full (B, L, L) sign matrix once via the dense head.
            full_signs = self.attn._dense(x_ctx)
            K_static = self.cfg.dynamic_topk_static_k
            # Mixed selection: K_static positional + K_dyn dynamic.
            # When K_static = 0, pure dynamic top-K (no positional guarantee).
            if 0 < K_static < K:
                # Static positional: take first K_static nearest from
                # _local_indices (centred symmetric ±K_static/2).
                if L == self.cfg.n_positions and K_static == K:
                    static_local = self._local_indices_cache
                else:
                    static_local = _local_indices(L, K_static, x.device)
            else:
                static_local = None
            with torch.no_grad():
                mag = full_signs.detach().abs().mean(dim=0)   # (L, L)
                mag.fill_diagonal_(-1.0)
                # Long-cycle inductive prior: scale the score by
                # (positional-distance)^α. α > 0 biases the top-K
                # toward distant candidates, which form longer walks
                # through the walk-op composition.
                alpha_lc = self.cfg.long_cycle_prior_alpha
                if alpha_lc > 0:
                    idx = torch.arange(L, device=x.device, dtype=torch.float)
                    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
                    dist_w = (dist / max(L - 1, 1)).pow(alpha_lc)
                    mag = mag * dist_w
                if static_local is not None:
                    # Suppress positions already chosen as static so dynamic
                    # picks DISJOINT new ones (no duplication).
                    mag = mag.scatter(
                        1, static_local.long(),
                        torch.full_like(mag, -1.0),
                    )
                    K_dyn = K - K_static
                    dyn_local = mag.topk(K_dyn, dim=-1, sorted=False).indices
                    local = torch.cat([static_local.long(), dyn_local], dim=1)
                else:
                    local = mag.topk(K, dim=-1, sorted=False).indices
            # Gather the corresponding sign values without re-running
            # the dense head.
            K_total = local.shape[1]
            anchor_to_cand = torch.gather(
                full_signs, dim=2,
                index=local.unsqueeze(0).expand(B, L, K_total),
            )
        else:
            # Fast path: use the cached static indices when L matches
            # the configured n_positions; fall back to recompute for
            # ragged-L variants (e.g. short-sequence inference).
            if L == self.cfg.n_positions:
                local = self._local_indices_cache
            else:
                local = _hybrid_indices(
                    L, K, self.cfg.n_jumps_per_anchor,
                    self.cfg.jumps_seed, x.device,
                )
            K_total = local.shape[1]
            anchor_to_cand = self.attn.compute_anchor_signs(x_ctx, local)
        # Pool values.
        v = self.value_proj(x_ctx) if self.value_proj is not None else x_ctx
        neighbors = v[:, local, :]                       # (B, L, K_total, d)

        mag_weight = self._compute_mag_weight(x_ctx, local, B, L, K_total)

        per_arity: list[torch.Tensor] = []
        for proj, arity in zip(self.per_arity_proj, self.cfg.arities):
            k = min(arity, K_total)
            feats_k = neighbors[:, :, :k, :]
            score = self.walk_op.compute(
                anchor_to_cand, self.attn, x_ctx, local, k
            )                                            # (B, L, 1)
            if mag_weight is not None:
                w = mag_weight[:, :, :k]
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                pooled = (w.unsqueeze(-1) * feats_k).sum(dim=2)
            else:
                pooled = feats_k.mean(dim=2)
            per_arity.append(proj(pooled) * score)

        alpha = self.alpha()
        z_stack = torch.stack(per_arity, dim=0)          # (n_arities, B, L, d)
        y_walk = (alpha.view(-1, 1, 1, 1) * z_stack).sum(dim=0)  # walk-op path

        # Dual-path mix: pool-scatter (synaptic) ⊕ walk-op (rhythm).
        if self.pool_scatter is not None:
            g = torch.sigmoid(self.path_gate)
            # Entropy-feedback rotor input: per-layer scalar H built from
            # the (alpha, gate) distributions -- the two channels that
            # carry the layer's regime + path-mixing state. When the
            # network is "confident" (peaked α or saturated gate)
            # H → 0 and the rotor reduces to identity; high uncertainty
            # produces a non-trivial multiplicative twist on scatter_h.
            if self.cfg.use_pool_scatter_rotor:
                H_alpha = -(alpha * alpha.clamp_min(1e-12).log()).sum()
                # Bernoulli entropy of the (g, 1-g) gate distribution.
                g_c = g.clamp(1e-12, 1.0 - 1e-12)
                H_gate = -(g_c * g_c.log() + (1.0 - g_c) * (1.0 - g_c).log())
                H_scalar = H_alpha + H_gate
            else:
                H_scalar = None
            y_ps = self.pool_scatter(x_ctx, local,
                                      entropy_scalar=H_scalar) - x_ctx
            # y_ps - x_ctx because FusedPoolScatter already adds the
            # residual internally; we want only the delta to mix with
            # walk-op (the outer residual happens below).
            y = g * y_ps + (1.0 - g) * y_walk
        else:
            y = y_walk

        # Optional LayerScale on the residual branch (γ·y).
        if self.layer_scale is not None:
            y = self.layer_scale * y

        if self.cfg.use_pre_norm:
            # Pre-norm: residual identity carries un-normalised x; the
            # block body already saw `LN(x)` at entry. No second norm
            # here -- the FFN does its own pre-norm internally if used.
            h = x + self.dropout(y)
        else:
            # Post-norm: classic transformer convention (default).
            h = self.layer_norm(x + self.dropout(y))
        return self.ffn(h)

    # ── Internal: magnitude weight ────────────────────────────────

    def _compute_mag_weight(
        self, x: torch.Tensor, local: torch.Tensor,
        B: int, L: int, K_total: int,
    ) -> torch.Tensor | None:
        if self.mag_q is None:
            return None
        mq = self.mag_q(x); mk = self.mag_k(x)
        # Sparse: gather mk at candidate positions only.
        mk_g = mk[:, local, :]                             # (B, L, K_total, h)
        s = (mq.unsqueeze(2) * mk_g).sum(-1)              # (B, L, K_total)
        s = s / (self.cfg.d_model ** 0.5)
        return torch.softmax(s.abs(), dim=-1)
