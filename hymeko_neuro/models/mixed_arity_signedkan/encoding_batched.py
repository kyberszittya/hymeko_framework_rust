"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import MixedAritySignedKANConfig
from .scatter import _attn_softmax_dispatch

def encode_edges_batched(
    self,
    per_arity_inputs: list[tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> torch.Tensor:
    """Mini-batch over cycles within each layer.

    Bounds peak forward-activation memory at O(batch_size · k · S · d)
    regardless of total cycle count. Uses gradient checkpointing per
    batch so the backward pass also stays bounded.

    Decomposition relies on three facts:

    1. ``M_vt @ h_t`` (vertex pool, mode="sum") = scatter-add of
       ``h_t[t]`` to every vertex appearing in cycle ``t``. Equivalent
       to ``index_add_(0, triad_v.flatten(), h_t.repeat_interleave(k))``.
    2. ``M_e @ h_final`` (edge pool) = sparse mat-mul along the
       cycle dimension; decomposes additively over cycle batches.
    3. JK-{last, sum, concat} all commute with summation over
       cycles — for "concat" we accumulate per-layer edge pools
       and concat at the end (instead of materialising per-cycle
       layer-stacked embeddings).

    Numerics differ from ``_encode_edges_full`` only by FP
    non-associativity in the order of summation (atol ≈ 1e-4
    on hidden=16, T=30k).
    """
    from torch.utils.checkpoint import checkpoint

    cfg = self.cfg.base
    n_layers = cfg.n_layers
    layer = self.base.shared_layer
    h_v = self.node_embed.weight                      # (V, d)
    d = cfg.hidden_dim
    alpha = self.alpha()
    n_arities = len(self.cfg.arities)
    jk = cfg.jk_mode
    use_attention = self.cfg.attention_m_e

    # Pre-extract M_e COO data per arity (one-time per call).
    # NOTE: M_e was built with .coalesce() so indices are sorted by
    # (row, col) — column-filter masks are O(nnz) but trivial.
    M_e_meta = []
    for ai in range(n_arities):
        M_e = per_arity_inputs[ai][3]
        idx = M_e._indices()
        val = M_e._values()
        n_edges = M_e.shape[0]
        M_e_meta.append((idx[0], idx[1], val, n_edges))

    # When attention is on, we need the full per-cycle embeddings
    # post-encoder before computing softmax over each edge's
    # incident cycles (denominator is global per row). Collect
    # batch slices in a list and torch.cat once after the loop —
    # linear memory in T (vs the quadratic O(n_batches · T)
    # cost of repeated index_copy on a full tensor).  When
    # attention is off, accumulate edge pools directly per batch.
    if use_attention:
        h_t_slices_per_arity_per_layer: list[list[list[torch.Tensor]]] = [
            [[] for _ in range(n_layers)] for _ in range(n_arities)
        ]
        edge_pool_per_arity_per_layer = None
    else:
        h_t_slices_per_arity_per_layer = None
        edge_pool_per_arity_per_layer = [
            [None] * n_layers for _ in range(n_arities)
        ]

    for li in range(n_layers):
        # Per-arity vertex-update accumulators (V, d) for THIS layer.
        h_v_step_per_arity = [
            torch.zeros_like(h_v) for _ in range(n_arities)
        ]
        # Per-arity edge-pool accumulators for THIS layer (only
        # used when attention is off — attention defers pooling).
        if not use_attention:
            edge_pool_this_layer = [
                torch.zeros(M_e_meta[ai][3], d,
                             device=h_v.device, dtype=h_v.dtype)
                for ai in range(n_arities)
            ]

        for ai in range(n_arities):
            triad_v, triad_sigma, _M_vt, _M_e = per_arity_inputs[ai]
            T = triad_v.shape[0]
            k_arity = triad_v.shape[1]
            e_rows_all, e_cols_all, e_vals_all, _ = M_e_meta[ai]

            for bs in range(0, T, batch_size):
                be = min(bs + batch_size, T)
                v_b = triad_v[bs:be]                  # (B, k)
                sig_b = triad_sigma[bs:be]            # (B, k)

                # Checkpointed forward — frees inner activations.
                h_t_b = checkpoint(
                    layer, h_v, v_b, sig_b,
                    use_reentrant=False,
                )                                     # (B, d)

                # Vertex update: scatter-add h_t_b[b] into h_v_step
                # at each vertex of cycle b. Matches mode="sum".
                if li < n_layers - 1:
                    flat_v = v_b.reshape(-1)          # (B*k,)
                    flat_h = (h_t_b.unsqueeze(1)
                                   .expand(-1, k_arity, -1)
                                   .reshape(-1, d))
                    h_v_step_per_arity[ai] = (
                        h_v_step_per_arity[ai].index_add(
                            0, flat_v, flat_h,
                        )
                    )

                if use_attention:
                    # Stash batch slice; cat at end of layer loop.
                    h_t_slices_per_arity_per_layer[ai][li].append(h_t_b)
                else:
                    # Edge pool: gather M_e nnz where col ∈ [bs, be).
                    mask = (e_cols_all >= bs) & (e_cols_all < be)
                    if mask.any():
                        e_rows_b = e_rows_all[mask]
                        e_cols_b = e_cols_all[mask] - bs
                        e_vals_b = e_vals_all[mask]
                        contrib = (e_vals_b.unsqueeze(-1)
                                    * h_t_b[e_cols_b])
                        edge_pool_this_layer[ai] = (
                            edge_pool_this_layer[ai].index_add(
                                0, e_rows_b, contrib,
                            )
                        )

            if not use_attention:
                edge_pool_per_arity_per_layer[ai][li] = (
                    edge_pool_this_layer[ai]
                )

        # Combine per-arity vertex updates, advance h_v.
        if li < n_layers - 1:
            h_v_new = torch.zeros_like(h_v)
            for ai in range(n_arities):
                h_v_new = h_v_new + alpha[ai] * h_v_step_per_arity[ai]
            h_v = (h_v + h_v_new) if cfg.use_residual else h_v_new
            if self.base.layer_norms is not None:
                h_v = self.base.layer_norms[li](h_v)

    # JK fold per arity. Without attention, fold the (E, d) edge
    # pools directly. With attention, fold the (T, d) cycle
    # embeddings, then apply per-arity attention to derive the
    # final (E, d_jk) edge pool.
    per_arity_pools: list[torch.Tensor] = []
    if use_attention:
        # Build query embedding for attention.
        query_edges = getattr(self, "_pending_query_edges", None)
        if query_edges is None:
            raise RuntimeError(
                "attention_m_e requires query_edges; ensure they "
                "are passed via encode_edges()"
            )
        h_query = h_v[query_edges[:, 0]] + h_v[query_edges[:, 1]]
        for ai in range(n_arities):
            # Materialise (T_ai, d) per layer by concatenating
            # the per-batch slices once.
            stack = [
                torch.cat(h_t_slices_per_arity_per_layer[ai][li], dim=0)
                for li in range(n_layers)
            ]
            if jk == "last":
                h_final = stack[-1]
            elif jk == "sum":
                h_final = stack[0]
                for li in range(1, n_layers):
                    h_final = h_final + stack[li]
            elif jk == "concat":
                h_final = torch.cat(stack, dim=-1)   # (T_ai, d_jk)
            else:
                raise ValueError(f"unknown jk_mode: {jk}")
            M_e = per_arity_inputs[ai][3]
            idx = M_e._indices()
            rows = idx[0]
            cols = idx[1]
            attn_vals = self.attention_m_e(h_query, h_final, idx)
            # A2 hook: accumulate per-edge attention entropy.
            if self._attn_entropy_terms is not None:
                eps = 1e-12
                H_per_pair = -(
                    attn_vals * attn_vals.clamp_min(eps).log()
                )
                H_per_edge = torch.zeros(
                    M_e.shape[0],
                    device=attn_vals.device, dtype=attn_vals.dtype,
                ).index_add(0, rows, H_per_pair)
                self._attn_entropy_terms.append(H_per_edge.mean())
            # Direct scatter (avoids torch.sparse.mm sparse-autograd
            # densifying backward at 14+ GiB on Slashdot).
            weighted = attn_vals.unsqueeze(-1) * h_final[cols]
            E = M_e.shape[0]
            d_jk = h_final.shape[1]
            attn_pool = torch.zeros(
                E, d_jk, device=h_final.device, dtype=h_final.dtype,
            ).index_add(0, rows, weighted)
            if (self.attn_gate_logits is not None
                    or self.gate_projs is not None):
                uniform_pool = torch.sparse.mm(M_e, h_final)
                g = self._highway_gate(ai, h_query=h_query)
                if g.dim() == 0:
                    # Scalar gate.
                    arity_pool = ((1.0 - g) * uniform_pool
                                  + g * attn_pool)
                else:
                    # Per-edge gate of shape (E,).
                    g_b = g.unsqueeze(-1)
                    arity_pool = ((1.0 - g_b) * uniform_pool
                                  + g_b * attn_pool)
            else:
                arity_pool = attn_pool
            per_arity_pools.append(arity_pool)
    else:
        for ai in range(n_arities):
            stack = edge_pool_per_arity_per_layer[ai]
            if jk == "last":
                arity_pool = stack[-1]
            elif jk == "sum":
                arity_pool = stack[0]
                for li in range(1, n_layers):
                    arity_pool = arity_pool + stack[li]
            elif jk == "concat":
                arity_pool = torch.cat(stack, dim=-1)
            else:
                raise ValueError(f"unknown jk_mode: {jk}")
            per_arity_pools.append(arity_pool)

    if self.cfg.per_edge_gate:
        query_edges = getattr(self, "_pending_query_edges", None)
        if query_edges is None:
            raise RuntimeError(
                "per_edge_gate requires query_edges; ensure they "
                "are passed via encode_edges()"
            )
        z_u = h_v[query_edges[:, 0]]
        z_v = h_v[query_edges[:, 1]]
        gate_in = torch.cat([z_u, z_v, (z_u - z_v).abs()], dim=-1)
        gate_logits = self.gate_mlp(gate_in)
        if self.cfg.gumbel_hard:
            gate = F.gumbel_softmax(
                gate_logits, tau=self.cfg.gumbel_tau,
                hard=True, dim=-1,
            )
        else:
            gate = F.softmax(gate_logits, dim=-1)
        stacked = torch.stack(per_arity_pools, dim=1)  # (E, A, d_jk)
        edge_emb = (gate.unsqueeze(-1) * stacked).sum(dim=1)
    else:
        edge_emb = None
        for ai in range(n_arities):
            edge_emb = (alpha[ai] * per_arity_pools[ai]
                        if edge_emb is None
                        else edge_emb + alpha[ai] * per_arity_pools[ai])
    return edge_emb


