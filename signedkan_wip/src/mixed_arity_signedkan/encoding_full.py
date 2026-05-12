"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import MixedAritySignedKANConfig
from .scatter import _attn_softmax_dispatch

def encode_edges_full(
    self,
    per_arity_inputs: list[tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]],
    query_edges: torch.Tensor | None = None,
) -> torch.Tensor:
    cfg = self.cfg.base
    n_layers = cfg.n_layers
    layer = self.base.shared_layer
    h_v = self.node_embed.weight                      # (V, d)
    # Apply per-vertex continuous feature injection (if configured).
    vf = getattr(self, "_pending_vertex_features", None)
    if vf is not None and self.vertex_feat_proj is not None:
        h_v = h_v + self.vertex_feat_proj(vf)
    # Apply per-edge continuous feature injection (if configured).
    # We project per-edge features to d_hidden, pool to vertices via
    # the edge_to_vertex sparse incidence (V × E), and add to h_v.
    ef = getattr(self, "_pending_edge_features", None)
    e2v = getattr(self, "_pending_edge_to_vertex", None)
    if (ef is not None and self.edge_feat_proj is not None
            and e2v is not None):
        h_v = h_v + torch.sparse.mm(e2v, self.edge_feat_proj(ef))
    d = cfg.hidden_dim
    alpha = self.alpha()

    per_arity_per_layer_t: list[list[torch.Tensor]] = [
        [] for _ in self.cfg.arities
    ]
    for li in range(n_layers):
        # Per-arity triad/tuple embeddings via the SHARED layer.
        arity_h_t = []
        for triad_v, triad_sigma, _M_vt, _M_e in per_arity_inputs:
            h_t = layer(h_v, triad_v, triad_sigma)    # (T_k, d)
            arity_h_t.append(h_t)
        for ai, h_t in enumerate(arity_h_t):
            per_arity_per_layer_t[ai].append(h_t)

        # Vertex update by mixing per-arity scatter-pools.
        if li < n_layers - 1:
            h_v_step = torch.zeros_like(h_v)
            for ai, h_t in enumerate(arity_h_t):
                M_vt = per_arity_inputs[ai][2]
                h_v_step = h_v_step + alpha[ai] * torch.sparse.mm(
                    M_vt, h_t,
                )
            # Optional SGCN-style direct sign-conditional path.
            if (self.cfg.direct_messaging
                    and getattr(self, "_A_pos", None) is not None):
                h_pos = torch.sparse.mm(self._A_pos, self.W_pos(h_v))
                h_neg = torch.sparse.mm(self._A_neg, self.W_neg(h_v))
                # 2-element softmax (sigmoid-free, KAN-aligned).
                g_pair = torch.stack(
                    [torch.zeros_like(self.direct_gate),
                     self.direct_gate]
                )
                g = F.softmax(g_pair, dim=0)[1]
                h_v_step = (1.0 - g) * h_v_step + g * (h_pos + h_neg)
            h_v = (h_v + h_v_step) if cfg.use_residual else h_v_step
            if self.base.layer_norms is not None:
                h_v = self.base.layer_norms[li](h_v)

    # JK-aggregate per arity (uses base config).
    jk = cfg.jk_mode
    per_arity_final = []
    for ai in range(len(self.cfg.arities)):
        stack = per_arity_per_layer_t[ai]
        if jk == "last":
            per_arity_final.append(stack[-1])
        elif jk == "sum":
            per_arity_final.append(torch.stack(stack, dim=0).sum(dim=0))
        elif jk == "concat":
            per_arity_final.append(torch.cat(stack, dim=-1))
        else:
            raise ValueError(f"unknown jk_mode: {jk}")

    # Pool per-arity to edges and mix.
    # Build query embedding once if attention OR per-edge gate is on.
    if self.cfg.attention_m_e or self.cfg.per_edge_gate:
        # query_edges shape (E, 2) of (u, v); use h_v[u] + h_v[v] as
        # the additive permutation-invariant query for attention,
        # and (z_u, z_v, |z_u - z_v|) as the gate input.
        h_query = h_v[query_edges[:, 0]] + h_v[query_edges[:, 1]]

    # First, gather all per-arity edge pools (shape (E, d_jk)).
    per_arity_edge_pools = []
    for ai, h_final in enumerate(per_arity_final):
        M_e = per_arity_inputs[ai][3]
        if self.cfg.attention_m_e:
            idx = M_e._indices()
            rows = idx[0]
            cols = idx[1]
            attn_vals = self.attention_m_e(
                h_query, h_final, idx,
            )
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
            weighted = attn_vals.unsqueeze(-1) * h_final[cols]
            E = M_e.shape[0]
            d_jk = h_final.shape[1]
            attn_pool = torch.zeros(
                E, d_jk,
                device=h_final.device, dtype=h_final.dtype,
            ).index_add(0, rows, weighted)
            if self.attn_gate_logits is not None:
                uniform_pool = torch.sparse.mm(M_e, h_final)
                g = self.cfg.attention_highway_max * torch.sigmoid(
                    self.attn_gate_logits[ai]
                )
                edge_pool = (1.0 - g) * uniform_pool + g * attn_pool
            else:
                edge_pool = attn_pool
        else:
            edge_pool = torch.sparse.mm(M_e, h_final)
        per_arity_edge_pools.append(edge_pool)

    if self.cfg.per_edge_gate:
        # Per-edge gating: weights of shape (E, n_arities).
        z_u = h_v[query_edges[:, 0]]
        z_v = h_v[query_edges[:, 1]]
        gate_in = torch.cat([z_u, z_v, (z_u - z_v).abs()], dim=-1)
        gate_logits = self.gate_mlp(gate_in)        # (E, n_arities)
        if self.cfg.gumbel_hard:
            # Hard one-hot in forward, soft Gumbel in backward.
            gate = F.gumbel_softmax(
                gate_logits,
                tau=self.cfg.gumbel_tau,
                hard=True,
                dim=-1,
            )
        else:
            gate = F.softmax(gate_logits, dim=-1)
        stacked = torch.stack(per_arity_edge_pools, dim=1)  # (E, A, d)
        edge_emb = (gate.unsqueeze(-1) * stacked).sum(dim=1)
    else:
        edge_emb = None
        for ai, edge_pool in enumerate(per_arity_edge_pools):
            edge_emb = (alpha[ai] * edge_pool if edge_emb is None
                        else edge_emb + alpha[ai] * edge_pool)
    # Stash post-encoder vertex embeddings for downstream node-level
    # heads (used by tabular node classification, mesh
    # correspondence, etc.).  This is the h_v after L−1 vertex
    # updates from the cycle-pool aggregation.
    self._final_h_v = h_v
    return edge_emb

