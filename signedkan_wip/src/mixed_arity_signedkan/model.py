"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import MixedAritySignedKANConfig
from .scatter import _attn_softmax_dispatch
from .attention import _AttentionM_e, _QuaternionAttentionM_e
from .utils import subsample_tuples, build_edge_to_tuples, build_vertex_to_tuples
from ..splines import _catmull_rom_eval
from ..signedkan import (MultiLayerSignedKAN, MultiLayerSignedKANConfig,
                          build_vertex_triad_incidence)

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
            n_heads = getattr(cfg, "attention_m_e_n_heads", 1)
            if kind == "quaternion":
                # d_attn must satisfy: 4 | d_attn AND n_heads | (d_attn / 4).
                # Round d_attn up to a multiple of (4 * n_heads).
                step = 4 * n_heads
                if d_attn % step != 0:
                    d_attn = ((d_attn + step - 1) // step) * step
                self.attention_m_e = _QuaternionAttentionM_e(
                    d_query=d, d_cycle=d_jk, d_attn=d_attn, n_heads=n_heads,
                )
            elif kind == "dot":
                if d_attn % n_heads != 0:
                    d_attn = ((d_attn + n_heads - 1) // n_heads) * n_heads
                self.attention_m_e = _AttentionM_e(
                    d_query=d, d_cycle=d_jk, d_attn=d_attn, n_heads=n_heads,
                )
            else:
                raise ValueError(
                    f"unknown attention_m_e_kind: {kind!r}; "
                    f"valid: 'dot', 'quaternion'"
                )
        else:
            self.attention_m_e = None

        # Highway-gated attention parameters.  Two parameterisations:
        #
        # "scalar"  — one learnable logit per arity slot, sigmoid-free
        #             2-element softmax → g_κ ∈ (0, 1).  Init at low
        #             logit so g_κ ≈ 0.05 at start (uniform-leaning).
        # "edge_cr" — per-arity Linear projection from h_query to a
        #             2-d input, tanh-bounded, fed to a learnable
        #             Catmull-Rom spline whose 2 output channels are
        #             softmax-normalised to (uniform_weight,
        #             attention_weight).  Per-edge gate, no fixed
        #             nonlinearities — KAN-aligned.
        self.attn_gate_kind = (cfg.attention_highway_kind
                                if cfg.attention_m_e and cfg.attention_highway
                                else "scalar")
        if cfg.attention_m_e and cfg.attention_highway:
            if self.attn_gate_kind == "scalar":
                self.attn_gate_logits = nn.Parameter(
                    torch.full(
                        (len(cfg.arities),),
                        cfg.attention_highway_init_logit,
                        dtype=torch.float32,
                    )
                )
                self.gate_projs = None
                self.gate_coefs = None
            elif self.attn_gate_kind == "edge_cr":
                self.attn_gate_logits = None
                d_query = cfg.base.hidden_dim
                n_grid = cfg.attention_highway_n_grid
                self.gate_n_grid = n_grid
                self.gate_projs = nn.ModuleList([
                    nn.Linear(d_query, 2)
                    for _ in cfg.arities
                ])
                # Init: small weights, bias [3, 0] so initial output
                # softmax([3, 0]) ≈ (0.95, 0.05) → uniform-leaning.
                with torch.no_grad():
                    for p in self.gate_projs:
                        p.weight.mul_(0.01)
                        p.bias.copy_(
                            torch.tensor([3.0, 0.0], dtype=torch.float32)
                        )
                # CR control points: linspace(-3, 3, n_grid) per channel
                # (identity-like curve at init, gives σ-like mapping).
                init_coef = (torch.linspace(-3.0, 3.0, n_grid)
                             .unsqueeze(0).expand(2, -1).contiguous())
                self.gate_coefs = nn.Parameter(
                    init_coef.unsqueeze(0).expand(
                        len(cfg.arities), -1, -1,
                    ).contiguous()
                )
            else:
                raise ValueError(
                    f"unknown attention_highway_kind: "
                    f"{self.attn_gate_kind!r}",
                )
        else:
            self.attn_gate_logits = None
            self.gate_projs = None
            self.gate_coefs = None

        # Per-edge attention entropy accumulator. Filled during forward
        # (one (mean) entropy scalar per arity slot in the batched and
        # full encode paths). Reset at the start of each encode_edges
        # call. The training loop reads this list and adds
        #     -λ · mean(H_attn)
        # to the BCE loss when HSIKAN_ATTN_ENTROPY_LAMBDA > 0.  Plain
        # tensor accumulation keeps backward through the attention
        # softmax.
        self._attn_entropy_terms: list[torch.Tensor] | None = None

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

    def _highway_gate(self, ai: int,
                       h_query: torch.Tensor | None) -> torch.Tensor:
        """Compute the per-arity Highway gate for slot ``ai``.

        Returns a 0-d (scalar) or 1-d (per-edge) tensor depending on
        ``attention_highway_kind``.

        - "scalar"  : per-arity learnable scalar; sigmoid-free
                      2-element softmax (math-equivalent to
                      sigmoid(η_κ)).
        - "edge_cr" : per-edge KAN-aligned gate.  ``h_query`` (E, d)
                      is required.  Returns a (E,) attention-weight
                      tensor; the uniform-pool weight is (1 − return).
        """
        if self.attn_gate_kind == "scalar":
            eta = self.attn_gate_logits[ai]
            pair = torch.stack([torch.zeros_like(eta), eta])
            return self.cfg.attention_highway_max * F.softmax(pair, dim=0)[1]
        if self.attn_gate_kind == "edge_cr":
            if h_query is None:
                raise RuntimeError(
                    "edge_cr Highway gate requires h_query to be passed",
                )
            x = torch.tanh(self.gate_projs[ai](h_query))   # (E, 2) ∈ [-1, 1]
            cr_out = _catmull_rom_eval(
                self.gate_coefs[ai], x, self.gate_n_grid,
            )                                                # (E, 2)
            weights = F.softmax(cr_out, dim=-1)              # (E, 2)
            return self.cfg.attention_highway_max * weights[:, 1]
        raise ValueError(
            f"unknown attention_highway_kind: {self.attn_gate_kind!r}"
        )

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
        # Reset the per-edge attention entropy buffer for this forward.
        self._attn_entropy_terms = []
        if self.cfg.cycle_batch_size is None:
            return self._encode_edges_full(per_arity_inputs, query_edges)
        return self._encode_edges_batched(per_arity_inputs,
                                           self.cfg.cycle_batch_size)


    # Delegated encoding implementations (extracted to module fns to keep
    # this file under 300 LOC per CLAUDE.md §6.5 #4).
    def _encode_edges_full(self, *args, **kwargs):
        from .encoding_full import encode_edges_full
        return encode_edges_full(self, *args, **kwargs)

    def _encode_edges_batched(self, *args, **kwargs):
        from .encoding_batched import encode_edges_batched
        return encode_edges_batched(self, *args, **kwargs)
