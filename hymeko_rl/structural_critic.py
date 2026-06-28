"""Structural critic — value decomposed over the signed cycles and walks of the (fixed) kinematic hypergraph.

Standard critics emit a scalar V(s) from pooled features. This one keeps the structure: it enumerates the
hypergraph's top-K signed **cycles** and **walks** ONCE at construction (the graph is fixed per env), and at
forward time decomposes V(s) over those motifs — per-motif a sign-weighted gather of the backbone's per-vertex
features → a per-motif value v_c → an aggregate V = pool_c(α_c v_c). With `task_graph` the grasp/goal hyperedges
form an arm-coin-zone cycle, so value is routed along the task's structural path (structural credit assignment;
`project-actor-critic-shared-reasoning`). The per-motif v_c are the learned FuzzySignature object (interpretable).

Enumeration reuses the Rust `hymeko` binding (`enumerate_top_k_cycles_rs` / `enumerate_top_k_walks_rs`) — the
algorithm lives in `hymeko_graph`, not here (§6.5 #2). Aggregation and pooling are **config fields** (one class,
internal dispatch — §6.5 #1), so the aggregation×pooling ablation is a config sweep, not per-cell classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

from hymeko_rl.hypergraph_state import HypergraphState

AGGREGATIONS = ("sign_mean", "mlp", "fir")   # per-motif feature aggregation (the ablation axis)
POOLINGS = ("attention", "mean", "sum")       # motif -> V pooling (the ablation axis)


class NodeFeatureBackbone(Protocol):
    """The contract the structural critic needs from a backbone: per-vertex features before pooling.

    ``SignedKANBackbone`` (HSiKAN) satisfies this via ``node_activations``."""

    def node_activations(self, obs: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class StructuralCriticConfig:
    """Which motifs to enumerate and how to aggregate/pool them.

    # Invariants ``aggregation in AGGREGATIONS``; ``pooling in POOLINGS``; at least one of cycles/walks on."""
    aggregation: str = "sign_mean"
    pooling: str = "attention"
    use_cycles: bool = True
    use_walks: bool = True
    cycle_len: int = 3
    cycle_keep: int = 16
    walk_len: int = 3
    walk_keep: int = 16
    score_kind: str = "balance"


def _signed_adjacency_map(hg: HypergraphState) -> dict[tuple[int, int], int]:
    """``(u, v) -> sign`` for both arc directions (so a cycle's consecutive pairs resolve regardless of order)."""
    m: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(hg.edges.tolist(), hg.signs.tolist()):
        m[(int(u), int(v))] = int(s)
        m.setdefault((int(v), int(u)), int(s))
    return m


def enumerate_motifs(hg: HypergraphState, cfg: StructuralCriticConfig,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate cycles+walks once → padded ``(vertex_idx (M,Lmax), sign (M,Lmax), length (M,))``.

    Cycle signs are recomputed from the signed adjacency (the cycle binding returns vertices only); walk signs
    come from the binding. Padded entries carry sign 0 (zero contribution); ``length`` is the true motif size.
    # Errors raises ``ImportError`` if the ``hymeko`` binding is not built (run ``maturin develop`` in hymeko_py).
    """
    try:
        import hymeko   # the built PyO3 module (module-name = "hymeko")
    except ImportError as err:                          # pragma: no cover - environment guard
        raise ImportError("the `hymeko` cycle binding is not built; run `maturin develop` in hymeko_py") from err

    eu = hg.edges[:, 0].astype(np.uint32).tolist()
    ev = hg.edges[:, 1].astype(np.uint32).tolist()
    es = hg.signs.astype(np.int8).tolist()
    n = int(hg.n_vertices)
    rows: list[tuple[list[int], list[int]]] = []        # (vertices, signs) per motif

    if cfg.use_cycles:
        cyc, _scores = hymeko.enumerate_top_k_cycles_rs(eu, ev, es, n, cfg.cycle_len, cfg.cycle_keep,
                                                        score_kind=cfg.score_kind)
        smap = _signed_adjacency_map(hg)
        for c in np.asarray(cyc).tolist():
            k = len(c)
            signs = [smap.get((c[i], c[(i + 1) % k]), 0) for i in range(k)]   # ring edges
            rows.append((list(c), signs))
    if cfg.use_walks:
        wlk, wsg, _ws = hymeko.enumerate_top_k_walks_rs(eu, ev, es, n, cfg.walk_len, cfg.walk_keep,
                                                        score_kind=cfg.score_kind)
        wlk_l, wsg_l = np.asarray(wlk).tolist(), np.asarray(wsg).tolist()
        for verts, edge_signs in zip(wlk_l, wsg_l):       # verts: L+1, signs: L (one per edge → pad to verts len)
            rows.append((list(verts), list(edge_signs) + [0]))

    if not rows:
        return (np.zeros((0, 1), np.int64), np.zeros((0, 1), np.float32), np.zeros((0,), np.float32))
    lmax = max(len(v) for v, _ in rows)
    vid = np.zeros((len(rows), lmax), np.int64)
    sgn = np.zeros((len(rows), lmax), np.float32)
    length = np.zeros((len(rows),), np.float32)
    for i, (verts, signs) in enumerate(rows):
        vid[i, : len(verts)] = verts
        sgn[i, : len(signs)] = signs
        length[i] = float(len(verts))
    return vid, sgn, length


class StructuralCritic(nn.Module):
    """A value head: per-vertex backbone features → per-motif sign-weighted aggregate → per-motif value → pool.

    # Preconditions ``backbone`` exposes ``node_activations(obs) -> (B, N, feat_dim)``; ``feat_dim >= 1``.
    # Postconditions ``forward(obs) -> (B,)``. Falls back to a pooled-linear scalar if no motifs are found.
    """

    motif_vid: torch.Tensor   # (M, Lmax) vertex indices per motif (registered buffer)
    motif_sgn: torch.Tensor   # (M, Lmax) per-position signs (pad = 0)
    motif_len: torch.Tensor   # (M,) true motif lengths
    # Declared at class level so mypy uses these (not nn.Module.__getattr__'s Tensor|Module union).
    value_head: nn.Module
    agg_mlp: nn.Module
    att: nn.Module
    fir_coef: torch.Tensor
    backbone: NodeFeatureBackbone

    def __init__(self, backbone: NodeFeatureBackbone, feat_dim: int, hg_state: HypergraphState,
                 cfg: StructuralCriticConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or StructuralCriticConfig()
        if cfg.aggregation not in AGGREGATIONS or cfg.pooling not in POOLINGS:
            raise ValueError(f"aggregation in {AGGREGATIONS}, pooling in {POOLINGS}; got {cfg}")
        if not (cfg.use_cycles or cfg.use_walks):
            raise ValueError("at least one of use_cycles/use_walks must be True")
        self.backbone = backbone
        self.cfg = cfg
        self.feat_dim = feat_dim
        vid, sgn, length = enumerate_motifs(hg_state, cfg)
        self.n_motifs = int(vid.shape[0])
        self.register_buffer("motif_vid", torch.as_tensor(vid, dtype=torch.long))
        self.register_buffer("motif_sgn", torch.as_tensor(sgn, dtype=torch.float32))
        self.register_buffer("motif_len", torch.as_tensor(length, dtype=torch.float32).clamp_min(1.0))
        lmax = int(self.motif_vid.shape[1]) if self.n_motifs > 0 else 1
        # All sub-heads constructed up front (cheap); forward uses only the one the config selects (config
        # dispatch, §6.5 #1) — keeps the module statically typed and the ablation's param counts comparable.
        self.value_head = nn.Linear(feat_dim, 1)
        self.agg_mlp = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.ReLU())
        self.fir_coef = nn.Parameter(torch.ones(lmax))    # learned per-position coefficient bank (signed FIR)
        self.att = nn.Linear(feat_dim, 1)

    def _node_feats(self, obs: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.backbone.node_activations(obs)   # (B, N, feat_dim)
        return h

    def _aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """Per-motif sign-weighted aggregate ``e_c`` ``(B, M, feat_dim)`` from per-vertex feats ``h (B,N,d)``."""
        b, _n, d = h.shape
        gathered = h[:, self.motif_vid.reshape(-1), :].reshape(b, self.n_motifs, -1, d)   # (B,M,Lmax,d)
        weighted = gathered * self.motif_sgn.unsqueeze(0).unsqueeze(-1)                    # sign-weight, pad=0
        if self.cfg.aggregation == "fir":
            weighted = weighted * self.fir_coef.view(1, 1, -1, 1)
        e: torch.Tensor = weighted.sum(dim=2) / self.motif_len.view(1, -1, 1)             # (B,M,d)
        if self.cfg.aggregation == "mlp":
            e = self.agg_mlp(e)
        return e

    def _pool(self, e: torch.Tensor) -> torch.Tensor:
        """Aggregate per-motif values ``v_c = value_head(e_c)`` into ``V (B,)`` by the configured pooling."""
        v: torch.Tensor = self.value_head(e).squeeze(-1)               # (B, M)
        if self.cfg.pooling == "attention":
            alpha: torch.Tensor = torch.softmax(self.att(e).squeeze(-1), dim=1)   # (B, M)
            attended: torch.Tensor = (alpha * v).sum(dim=1)
            return attended
        pooled: torch.Tensor = v.mean(dim=1) if self.cfg.pooling == "mean" else v.sum(dim=1)
        return pooled

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self._node_feats(obs)
        if self.n_motifs == 0:                           # fallback: scalar over the mean-pooled features
            fallback: torch.Tensor = self.value_head(h.mean(dim=1)).squeeze(-1)
            return fallback
        return self._pool(self._aggregate(h))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """``ActorCritic``-compatible alias (the critic seam in policy.py / ppo.py)."""
        return self.forward(obs)

    def per_motif_values(self, obs: torch.Tensor) -> torch.Tensor:
        """The decomposed per-motif values ``(B, M)`` — the structural signature (interpretability)."""
        if self.n_motifs == 0:
            return torch.zeros(obs.shape[0], 0)
        pmv: torch.Tensor = self.value_head(self._aggregate(self._node_feats(obs))).squeeze(-1)
        return pmv
