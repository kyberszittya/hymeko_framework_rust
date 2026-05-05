"""Mixed-arity HSiKAN — k=3 + k=4 (or any mix of k≥3) sharing the
same SignedKANLayer parameters, with learnable αₖ mixing weights at
the edge-pool stage.

Paper-narrative motivation: SGCN is structurally k=2 (binary edges).
HSiKAN's signed-incidence machinery generalises to any k via Davis
weak balance (`n_tuples.py`). Mixed-arity is the only experimental
direction in this codebase that SGCN cannot follow.

Architecture:
  - One shared `SignedKANLayer` (cfg.share_weights=True), L stacked
    applications.
  - Two arity-specific forward passes per layer (k=3 triads, k=4
    tuples, both sub-sampled to a manageable count).
  - Vertex-update fuses arity contributions:
      h_v ← h_v + α_3·(M_vt_k3 @ h_t_k3) + α_4·(M_vt_k4 @ h_t_k4)
  - Edge-pool fuses again at the prediction head:
      edge_emb = α_3·(M_e_k3 @ h_t_k3) + α_4·(M_e_k4 @ h_t_k4)
  - α = softmax(arity_logits) — two learnable scalars on the
    2-simplex.

Subsampling: k=4 tuples are random-sampled to ``max_k4`` (default
30k), seeded for reproducibility. The full enumeration on Bitcoin is
600k–1M tuples, intractable to forward at h=32 every step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .signedkan import (MultiLayerSignedKAN, MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)


@dataclass
class MixedAritySignedKANConfig:
    """Inherits the architecture from MultiLayerSignedKANConfig and
    adds the arity-mixing piece."""
    base: MultiLayerSignedKANConfig
    arities: tuple[int, ...] = (3, 4)        # which k's to mix
    init_arity_logits: tuple[float, ...] = (0.0, 0.0)
    # Per-edge learned mixture gate.  When True, the αₖ at the final
    # edge-pool mixing step is replaced by a per-edge softmax over a
    # gate MLP applied to the query edge's endpoint embeddings.  This
    # lets the model dynamically choose which arity (cycle vs walk)
    # carries the signal for each individual query — useful when
    # different graph regions have different geometric / topological
    # character (e.g. cube faces vs path-rich social graphs).  Within-
    # layer vertex aggregation still uses the global αₖ.
    per_edge_gate: bool = False
    # Gumbel--softmax hard gate.  When True (and `per_edge_gate=True`),
    # the gate uses `F.gumbel_softmax(.., tau=gumbel_tau, hard=True)`
    # instead of plain `softmax`.  In hard mode the forward pass
    # selects exactly ONE channel per query edge (one-hot), but the
    # backward pass uses the soft-Gumbel surrogate so gradients still
    # flow.  This breaks the soft-mixing ceiling we observed on the
    # mesh-cube experiment: when one channel needs to be totally
    # suppressed for the optimum, hard gating can do that, soft
    # gating cannot.
    gumbel_hard: bool = False
    # Gumbel--softmax temperature.  $\tau \to 0$ approaches a hard
    # one-hot; $\tau \to \infty$ approaches uniform.  Default $1.0$
    # matches the PyTorch convention; cooler values are crisper.
    gumbel_tau: float = 1.0
    # When set, encode_edges processes cycles in mini-batches of this
    # size per layer and uses gradient checkpointing, bounding the
    # peak (T, k, S, d) activation memory at O(cycle_batch_size).
    # When None, the full-batch encode_edges path is used (the
    # original implementation; identical numerics).
    cycle_batch_size: int | None = None
    # When True, replace uniform 1/|N(query)| pooling in the edge-
    # incidence stage with learned softmax attention over the cycles
    # incident to each query edge. The precomputed M_e gives the
    # sparse structure (which (query, cycle) pairs are non-zero); at
    # forward time, the weights are recomputed via dot-product
    # attention between the query-edge embedding and each cycle
    # embedding, then row-softmaxed.
    attention_m_e: bool = False
    # Choice of attention head when `attention_m_e=True`:
    #   "dot"        — standard scalar dot-product (default)
    #   "quaternion" — Hamilton-product real part: treats the
    #                  d_attn projection as d_attn/4 quaternions and
    #                  uses real(q ⊗ k) = qa·ka − qb·kb − qc·kc − qd·kd
    #                  as the score. The (i, j, k) components contribute
    #                  with negative sign — natural for signed-graph
    #                  data where i, j, k can encode sign-rotation
    #                  phases. Requires d_attn % 4 == 0.
    attention_m_e_kind: str = "dot"
    # When True, add a parallel SGCN-style sign-conditional direct
    # message-passing path between layers. h_v is updated by both the
    # cycle-pool aggregation and a per-sign-channel propagation:
    #   h_v ← h_v + α_cycle·(cycle update) + α_direct·(SGCN update)
    # The α_direct weight is learnable (sigmoid-gated). Combines
    # HSiKAN's higher-arity structural signal with SGCN's transitive
    # multi-hop signed-aggregation strength on dense graphs.
    direct_messaging: bool = False
    # Optional per-vertex continuous features. When > 0, encode_edges
    # accepts a (V, vertex_feat_dim) tensor that's projected to
    # d_hidden and added to the node embedding at layer 0. Use cases:
    #   - kinematic graphs: per-link continuous attributes (mass,
    #     inertia, joint angles aggregated from incident joints)
    #   - scene graphs: per-object features (bbox xyxy, area,
    #     category embedding)
    #   - context graphs: per-entity attribute vectors
    vertex_feat_dim: int = 0
    # Optional per-edge continuous features. When > 0, encode_edges
    # accepts a (E_edges_global, edge_feat_dim) tensor + a per-cycle
    # ``cycle_edge_idx`` tensor that maps each cycle's k edges to
    # global edge indices. The features get pooled to vertices via
    # the cycle's adjacency. Use cases:
    #   - kinematic graphs: joint angle / velocity / torque per joint
    #   - scene graphs: spatial-overlap (IoU), confidence, distance
    #   - context graphs: temporal stamp, source confidence
    edge_feat_dim: int = 0


def _scatter_softmax(scores: torch.Tensor, index: torch.Tensor,
                       n_rows: int) -> torch.Tensor:
    """Row-wise softmax over a sparse representation.

    scores : (nnz,) raw attention scores
    index  : (nnz,) row index for each score
    n_rows : int total number of rows

    Returns: (nnz,) softmax values, where for each row, the values at
    positions in that row sum to 1. Handles empty rows by leaving them
    untouched (no entries).
    """
    # Per-row max for numerical stability.
    max_per_row = torch.full(
        (n_rows,), float("-inf"), device=scores.device, dtype=scores.dtype,
    )
    max_per_row.scatter_reduce_(
        0, index, scores, reduce="amax", include_self=True,
    )
    # Replace -inf (empty rows) with 0 so subtraction doesn't propagate NaN.
    max_per_row = max_per_row.masked_fill(
        max_per_row == float("-inf"), 0.0,
    )
    shifted = scores - max_per_row[index]
    exp_scores = shifted.exp()
    sum_per_row = torch.zeros(
        n_rows, device=scores.device, dtype=scores.dtype,
    )
    sum_per_row.scatter_add_(0, index, exp_scores)
    return exp_scores / (sum_per_row[index] + 1e-12)


class _AttentionM_e(nn.Module):
    """Replaces uniform 1/|N(query)| pooling in M_e with learned softmax
    attention over cycles. Uses dot-product attention between query-edge
    embedding and per-cycle embedding.

    Parameters: 2 Linear layers (W_q, W_k) projecting to ``d_attn`` dim.

    Init strategy: W_q, W_k initialised to very small values so softmax
    starts approximately uniform (same as 1/|N(query)| baseline). Random
    Kaiming init at training start gives extreme attention concentration
    that wrecks early training; near-zero init lets attention learn
    deviations from uniform incrementally.
    """
    def __init__(self, d_query: int, d_cycle: int, d_attn: int = 32):
        super().__init__()
        self.W_q = nn.Linear(d_query, d_attn, bias=False)
        self.W_k = nn.Linear(d_cycle, d_attn, bias=False)
        # Near-uniform init: scale weights by 1e-2 so initial scores ≈ 0
        # → softmax over rows ≈ 1/|N(row)|.
        with torch.no_grad():
            self.W_q.weight.mul_(0.01)
            self.W_k.weight.mul_(0.01)
        self.scale = d_attn ** -0.5

    def forward(self, h_query: torch.Tensor, h_cycle: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        """
        h_query: (E, d_query) query-edge embeddings
        h_cycle: (T, d_cycle) per-cycle embeddings
        indices: (2, nnz) — indices[0]=query row, indices[1]=cycle col

        Returns: (nnz,) softmax-normalised attention weights to use as
        the values of the sparse M_e tensor.
        """
        rows = indices[0]
        cols = indices[1]
        q_proj = self.W_q(h_query)              # (E, d_attn)
        k_proj = self.W_k(h_cycle)              # (T, d_attn)
        scores = (q_proj[rows] * k_proj[cols]).sum(dim=-1) * self.scale
        return _scatter_softmax(scores, rows, h_query.shape[0])


class _QuaternionAttentionM_e(nn.Module):
    """Quaternion-valued attention head over (query_edge, cycle) pairs.

    Same I/O contract as :class:`_AttentionM_e` (returns softmax
    weights over the sparse `M_e` non-zeros), but the score function
    treats the per-pair `d_attn` projection as `d_attn / 4`
    independent quaternions and uses

        score = Σ_q  real(q_i ⊗ k_i)
              = Σ_q  (q_a·k_a − q_b·k_b − q_c·k_c − q_d·k_d)

    The negative sign on the (i, j, k) components is the
    distinguishing feature: in standard scalar attention, every
    embedding dimension contributes positively to the score, so
    "agreement" and "anti-agreement" both pull attention up. With
    Hamilton-product real-part scoring, (i, j, k) components
    *subtract* — geometrically, anti-aligned imaginary parts reduce
    the score even when the magnitudes are large. For signed graphs
    where the (i, j, k) axes can carry sign / phase information, this
    asymmetry is what we want.

    Implementation note: the layout is (E, n_quaternions, 4) where
    the last axis is the (real, i, j, k) ordering. Init scale 0.01
    keeps initial scores near zero so softmax starts ≈ uniform —
    same warm-start as :class:`_AttentionM_e`.
    """
    def __init__(self, d_query: int, d_cycle: int, d_attn: int = 32):
        super().__init__()
        if d_attn % 4 != 0:
            raise ValueError(
                f"_QuaternionAttentionM_e requires d_attn % 4 == 0, "
                f"got d_attn={d_attn}",
            )
        self.d_attn = d_attn
        self.n_quat = d_attn // 4
        self.W_q = nn.Linear(d_query, d_attn, bias=False)
        self.W_k = nn.Linear(d_cycle, d_attn, bias=False)
        with torch.no_grad():
            self.W_q.weight.mul_(0.01)
            self.W_k.weight.mul_(0.01)
        self.scale = self.n_quat ** -0.5

    def forward(self, h_query: torch.Tensor, h_cycle: torch.Tensor,
                indices: torch.Tensor) -> torch.Tensor:
        rows = indices[0]
        cols = indices[1]
        # Project to (E, n_quat, 4) and (T, n_quat, 4).
        q = self.W_q(h_query).view(-1, self.n_quat, 4)
        k = self.W_k(h_cycle).view(-1, self.n_quat, 4)
        qg = q[rows]                    # (nnz, n_quat, 4)
        kg = k[cols]                    # (nnz, n_quat, 4)
        # Hamilton-product real component, summed over quaternion
        # blocks: q ⊗ k → real = qa·ka − qb·kb − qc·kc − qd·kd.
        scores = (
            qg[..., 0] * kg[..., 0]
            - qg[..., 1] * kg[..., 1]
            - qg[..., 2] * kg[..., 2]
            - qg[..., 3] * kg[..., 3]
        ).sum(dim=-1) * self.scale
        return _scatter_softmax(scores, rows, h_query.shape[0])


class MixedAritySignedKAN(nn.Module):
    """Wraps the shared SignedKANLayer of a MultiLayerSignedKAN and
    runs it once per arity per layer, fusing via learnable αₖ."""

    def __init__(self, cfg: MixedAritySignedKANConfig):
        super().__init__()
        # Force share_weights=True; mixing arities through different
        # parameters defeats the architectural point.
        if not cfg.base.share_weights:
            raise ValueError(
                "Mixed-arity requires share_weights=True so the same "
                "SignedKANLayer applies to every arity."
            )
        self.cfg = cfg
        self.base = MultiLayerSignedKAN(cfg.base)
        self.arity_logits = nn.Parameter(
            torch.tensor(list(cfg.init_arity_logits), dtype=torch.float32)
        )

        # Per-edge learned mixture gate (alternative to global αₖ at
        # the final edge-pool stage).  Input: concatenated endpoint
        # embeddings + their absolute difference.  Output: per-edge
        # logits over arity slots, softmax-normalised at forward time.
        if cfg.per_edge_gate:
            d = cfg.base.hidden_dim
            n_arities = len(cfg.arities)
            self.gate_mlp = nn.Sequential(
                nn.Linear(3 * d, max(8, d // 2)),
                nn.GELU(),
                nn.Linear(max(8, d // 2), n_arities),
            )
            # Initialise the final layer near zero so the gate starts
            # uniform; it converges away from uniform only as the
            # edge-classification gradient drives it.
            with torch.no_grad():
                self.gate_mlp[-1].weight.mul_(0.05)
                self.gate_mlp[-1].bias.zero_()
        else:
            self.gate_mlp = None

        # Attention M_e head (one shared per arity).
        # JK-concat output dim = d * n_layers; we project from that.
        if cfg.attention_m_e:
            d = cfg.base.hidden_dim
            d_jk = d * cfg.base.n_layers if cfg.base.jk_mode == "concat" else d
            d_attn = max(16, d // 2)
            kind = getattr(cfg, "attention_m_e_kind", "dot")
            if kind == "quaternion":
                # Round d_attn up to a multiple of 4.
                if d_attn % 4 != 0:
                    d_attn = ((d_attn + 3) // 4) * 4
                self.attention_m_e = _QuaternionAttentionM_e(
                    d_query=d, d_cycle=d_jk, d_attn=d_attn,
                )
            elif kind == "dot":
                self.attention_m_e = _AttentionM_e(
                    d_query=d, d_cycle=d_jk, d_attn=d_attn,
                )
            else:
                raise ValueError(
                    f"unknown attention_m_e_kind: {kind!r}; "
                    f"valid: 'dot', 'quaternion'"
                )
        else:
            self.attention_m_e = None

        # Optional per-edge continuous-feature projection.
        if cfg.edge_feat_dim > 0:
            self.edge_feat_proj = nn.Linear(
                cfg.edge_feat_dim, cfg.base.hidden_dim, bias=True,
            )
            with torch.no_grad():
                self.edge_feat_proj.weight.mul_(0.01)
                self.edge_feat_proj.bias.zero_()
        else:
            self.edge_feat_proj = None

        # Optional per-vertex continuous-feature projection.
        if cfg.vertex_feat_dim > 0:
            self.vertex_feat_proj = nn.Linear(
                cfg.vertex_feat_dim, cfg.base.hidden_dim, bias=True,
            )
            # Initialise small so the cycle-pool features dominate at
            # the start of training; the model can lift the projection
            # weight if the continuous features actually help.
            with torch.no_grad():
                self.vertex_feat_proj.weight.mul_(0.01)
                self.vertex_feat_proj.bias.zero_()
        else:
            self.vertex_feat_proj = None

        # SGCN-style direct sign-conditional message passing components.
        if cfg.direct_messaging:
            d = cfg.base.hidden_dim
            self.W_pos = nn.Linear(d, d, bias=False)
            self.W_neg = nn.Linear(d, d, bias=False)
            # Initialise as identity so the direct path starts as a
            # passive copy of h_v; the gate starts neutral (sigmoid(0)=0.5)
            # so cycle and direct contribute equally at init.
            with torch.no_grad():
                nn.init.eye_(self.W_pos.weight)
                nn.init.eye_(self.W_neg.weight)
            self.direct_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.W_pos = None
            self.W_neg = None
            self.direct_gate = None

    @property
    def node_embed(self):
        return self.base.node_embed

    @property
    def classifier(self):
        return self.base.classifier

    @property
    def bilinear(self):
        return self.base.bilinear

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def alpha(self) -> torch.Tensor:
        a = F.softmax(self.arity_logits, dim=0)
        # Optional B&B-style arity mask. When set, zeros out alpha for
        # excluded arities and re-normalises the survivors. Persistent
        # across forward passes; clear by passing None.
        m = getattr(self, "_arity_mask", None)
        if m is not None:
            a = a * m.to(a.device).to(a.dtype)
            a = a / (a.sum() + 1e-12)
        return a

    def encode_graph(
        self,
        per_arity_inputs: list[tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]],
        query_edges: torch.Tensor | None = None,
        pool: str = "mean",
        vertex_features: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_to_vertex: torch.sparse.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce a graph-level fixed-dim embedding by pooling
        per-edge embeddings.

        ``pool``: "mean" (default), "max", or "sum".

        Use this for:
          - graph-level classification (mechanism family, pose class)
          - graph-level regression (DOF count, scalar properties)
          - graph retrieval (nearest-neighbour in embedding space)

        Returns a tensor of shape (d_jk,) — single vector per call.
        For batched graph processing, call once per graph and stack.
        """
        edge_emb = self.encode_edges(
            per_arity_inputs, query_edges,
            vertex_features=vertex_features,
            edge_features=edge_features,
            edge_to_vertex=edge_to_vertex,
        )
        if pool == "mean":
            return edge_emb.mean(dim=0)
        elif pool == "max":
            return edge_emb.max(dim=0).values
        elif pool == "sum":
            return edge_emb.sum(dim=0)
        else:
            raise ValueError(f"unknown pool: {pool!r}")

    def set_signed_adjacency(self, A_pos: torch.Tensor,
                              A_neg: torch.Tensor) -> None:
        """Set the per-sign sparse adjacency matrices used by the
        direct sign-conditional message-passing path.

        ``A_pos``: (V, V) sparse, A_pos[u, v] = 1 iff (u, v) is a
                    *positive* edge in the train graph (symmetric for
                    undirected — set both directions).
        ``A_neg``: (V, V) sparse, same for negative edges.

        Both are usually pre-normalised by row-degree (D^-1 A) at the
        caller site to give mean-aggregation semantics.
        """
        self._A_pos = A_pos
        self._A_neg = A_neg

    def set_arity_mask(self, mask) -> None:
        """Mask α over a subset of arities (B&B bound oracle).

        ``mask``: 0/1 tensor of shape ``(n_arities,)``, or None to clear.
        With mask set, ``alpha()`` returns the trained α restricted and
        re-normalised to the masked subset — letting a single trained
        all-arities model produce honest subset-AUC estimates without
        retraining. The masked-AUC is the *bound* in the αₖ-mask B&B.
        """
        if mask is None:
            self._arity_mask = None
            return
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        if mask.shape != (len(self.cfg.arities),):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != "
                f"({len(self.cfg.arities)},)"
            )
        self._arity_mask = mask.float()

    def encode_edges(
        self,
        per_arity_inputs: list[tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]],
        query_edges: torch.Tensor | None = None,
        vertex_features: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        edge_to_vertex: torch.sparse.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the shared layer on each arity, pool via per-arity
        incidence, mix.

        per_arity_inputs : list of length len(self.cfg.arities) giving
            (triad_v, triad_sigma, M_vt, M_edge) tuples per arity:
                triad_v     : (T_k, k) long
                triad_sigma : (T_k, k) long, ±1
                M_vt        : sparse (V, T_k)
                M_edge      : sparse (E, T_k) — when attention_m_e=True
                              the values are placeholders; recomputed
                              dynamically via the AttentionM_e head and
                              ``query_edges`` is required.
        query_edges       : (E, 2) long tensor of (u, v) per query edge.
                            Required when ``cfg.attention_m_e=True``.
        Returns
        -------
        edge_emb : (E, d_jk)  — to be fed straight into self.classifier
        """
        if self.cfg.attention_m_e and query_edges is None:
            raise ValueError(
                "attention_m_e=True requires query_edges to be passed "
                "into encode_edges()"
            )
        if self.cfg.per_edge_gate and query_edges is None:
            raise ValueError(
                "per_edge_gate=True requires query_edges to be passed "
                "into encode_edges()"
            )
        # Stash vertex/edge features for the inner forward path to read.
        # Could thread through args but module state keeps signature stable.
        self._pending_vertex_features = vertex_features
        self._pending_edge_features = edge_features
        self._pending_edge_to_vertex = edge_to_vertex
        # Stash query_edges so the batched path can access it for the
        # per-edge gate without breaking the signature.
        self._pending_query_edges = query_edges
        if self.cfg.cycle_batch_size is None:
            return self._encode_edges_full(per_arity_inputs, query_edges)
        if self.cfg.attention_m_e:
            raise NotImplementedError(
                "attention_m_e + cycle_batch_size not yet implemented; "
                "use either feature alone."
            )
        return self._encode_edges_batched(per_arity_inputs,
                                           self.cfg.cycle_batch_size)

    def _encode_edges_full(
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
                    g = torch.sigmoid(self.direct_gate)
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
                attn_vals = self.attention_m_e(
                    h_query, h_final, idx,
                )
                M_e_attn = torch.sparse_coo_tensor(
                    idx, attn_vals, M_e.shape,
                ).coalesce()
                edge_pool = torch.sparse.mm(M_e_attn, h_final)
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
        return edge_emb

    def _encode_edges_batched(
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

        # Per-arity, per-layer edge-pool accumulators (E, d).
        # We always store all layers because jk="concat" / "sum" need
        # them; "last" just discards the others.
        edge_pool_per_arity_per_layer: list[list[torch.Tensor]] = [
            [None] * n_layers for _ in range(n_arities)
        ]

        for li in range(n_layers):
            # Per-arity vertex-update accumulators (V, d) for THIS layer.
            h_v_step_per_arity = [
                torch.zeros_like(h_v) for _ in range(n_arities)
            ]
            # Per-arity edge-pool accumulators for THIS layer.
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

                edge_pool_per_arity_per_layer[ai][li] = edge_pool_this_layer[ai]

            # Combine per-arity vertex updates, advance h_v.
            if li < n_layers - 1:
                h_v_new = torch.zeros_like(h_v)
                for ai in range(n_arities):
                    h_v_new = h_v_new + alpha[ai] * h_v_step_per_arity[ai]
                h_v = (h_v + h_v_new) if cfg.use_residual else h_v_new
                if self.base.layer_norms is not None:
                    h_v = self.base.layer_norms[li](h_v)

        # JK fold per arity (over the stored per-layer edge pools).
        per_arity_pools: list[torch.Tensor] = []
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


def subsample_tuples(tuples, max_count: int, seed: int):
    """Deterministic random subsample. Returns the same list when
    max_count >= len(tuples)."""
    if len(tuples) <= max_count:
        return list(tuples)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(tuples), size=max_count, replace=False)
    return [tuples[int(i)] for i in idx]


def build_edge_to_tuples(tuples,
                          directed: bool = False) -> dict[tuple[int, int], list[int]]:
    """For each edge appearing as a cycle edge of some n-tuple, list
    the tuple indices it belongs to.

    ``directed=False`` (default): keys are unordered ``(min, max)``
    pairs. Each cycle edge contributes one key per cycle position.
    ``directed=True``: keys are directional ``(u, v)`` in cycle order
    — query edge ``(src, dst)`` only matches a cycle if that exact
    direction appears as one of the cycle's directed edges.
    """
    out: dict[tuple[int, int], list[int]] = {}
    for ti, t in enumerate(tuples):
        v = t.v
        k = len(v)
        if k == 2:
            # k=2 hyperedge IS a single edge — record it once.
            u, w = int(v[0]), int(v[1])
            key = (u, w) if directed else (min(u, w), max(u, w))
            out.setdefault(key, []).append(ti)
            continue
        for i in range(k):
            u, w = int(v[i]), int(v[(i + 1) % k])
            key = (u, w) if directed else (min(u, w), max(u, w))
            out.setdefault(key, []).append(ti)
    return out


def build_vertex_to_tuples(tuples) -> dict[int, list[int]]:
    """For each vertex, list of tuple indices having that vertex as
    one of its endpoints. Used for k=2 line-graph-style incidence."""
    out: dict[int, list[int]] = {}
    for ti, t in enumerate(tuples):
        for vid in t.v:
            out.setdefault(int(vid), []).append(ti)
    return out
