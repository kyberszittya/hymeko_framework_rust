"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass

from ..core.signedkan import MultiLayerSignedKANConfig


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
    # Number of attention heads. With n_heads>1, the d_attn dimension
    # is split across heads, each computing its own per-row softmax;
    # final attention weights are the mean across heads. Reduces
    # softmax sharpness on dense graphs (Slashdot regime).
    attention_m_e_n_heads: int = 1
    # Highway-gated attention. When True, the final per-arity edge
    # pool is a learned mix of the uniform 1/|N(query)| baseline and
    # the attention-weighted pool:
    #     edge_pool = (1 - g_k) · uniform_k  +  g_k · attention_k
    # where g_k = sigmoid(logit_k), per-arity. Initialised at a low
    # logit (~ -3) so g_k ≈ 0.05 at start — i.e., the model begins as
    # the uniform-pool baseline and gradient pushes attention in only
    # where it helps. Resolves the "softmax-too-sharp" failure mode
    # where dense attention over hundreds of cycles concentrates
    # signal that uniform pooling preserves (Slashdot regime).
    attention_highway: bool = False
    attention_highway_init_logit: float = -3.0
    # Maximum attention contribution: g_effective = max · sigmoid(logit).
    # 1.0 = unbounded (sigmoid range). Lower values force uniform pool
    # to dominate even when training pushes the gate logit high —
    # useful when softmax attention overfits the edge-in-cycle leak
    # while uniform pool generalises (Slashdot regime).
    attention_highway_max: float = 1.0
    # Highway gate parameterisation:
    #   "scalar"  — per-arity scalar gate (sigmoid-free 2-softmax form)
    #   "edge_cr" — per-edge KAN-aligned gate: a learnable Catmull-Rom
    #               spline maps tanh(W_κ · h_query) → 2 logits which
    #               are softmax-normalised to (uniform, attention)
    #               weights.  Per-edge variation (no fixed nonlinearity
    #               at the gate level).  Adds ~ d·2 + 2·n_grid params
    #               per arity slot; init biased toward uniform pool.
    attention_highway_kind: str = "scalar"
    attention_highway_n_grid: int = 8
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


