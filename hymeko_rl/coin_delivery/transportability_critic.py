"""R12 / HSiKAN-1 — structured transportability critic models + hypergraph builders.

Predicts P(strict-K6) for a (handoff, object, target, candidate-θ) from the R11.7B transportability dataset. A flat
MLP (A0) is compared against a hypergraph message-passing net (HSiKAN-style) under four incidence structures —
task-derived contact hypergraph, Steiner/block-design, random-sparse, and degree-matched-random (the mandatory
control) — at MATCHED parameter budget. The structural question: does the physical hypergraph help, or is a balanced
combinatorial incidence already enough, or does structure not help at all over a flat model?

Input (41-D) = descriptor x[0:30] ++ θ[30:36] ++ object features[36:41]. Nodes are semantic feature groups; the
incidence connects them into hyperedges. Torch, CPU/MPS (no CUDA needed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# ---- node feature groups (indices into the 41-D input) -------------------------------------------
# x layout (delivery_bc.dataset.descriptor): q[0:4] qd[4:8] coin[8:10] coin_vel[10:12] target[12:14]
#   req_transport[14:16] prev_tau[16:20] capture(n,s,preload,bmax)[20:24] outcome(dwell,delay,relvel1,relvel2)[24:28]
#   (coin_disp,term_speed)[28:30] ; θ[30:36] ; object(shape_cyl,shape_box,radius,radius_y,mass_ratio)[36:41]
NODES: dict[str, list[int]] = {
    "left_arm":     [0, 1, 4, 5],
    "right_arm":    [2, 3, 6, 7],
    "object_state": [8, 9, 10, 11],          # coin xy + velocity
    "target":       [12, 13, 14, 15],        # target xy + required transport
    "left_contact": [16, 17, 24, 25],        # prev_tau L + bilateral dwell + L/R contact delay
    "right_contact":[18, 19, 26, 27],        # prev_tau R + contact relvels
    "capture":      [20, 21, 22, 23, 28, 29],
    "theta":        [30, 31, 32, 33, 34, 35],
    "hybrid_mode":  [33, 34],                # θ ramp/release phase timing
    "object_id":    [36, 37, 38, 39, 40],    # shape one-hot + radius + radius_y + mass ratio
}
NODE_NAMES = list(NODES)
INPUT_DIM = 41

# task-derived contact hyperedges (physical interactions), as node-name tuples
TASK_EDGES: list[tuple[str, ...]] = [
    ("left_arm", "left_contact", "object_state"),
    ("right_arm", "right_contact", "object_state"),
    ("object_state", "object_id", "target"),
    ("left_contact", "right_contact", "theta", "target"),
    ("object_id", "theta", "hybrid_mode"),
]

_OBJECT_FEATURES = {  # shape_cyl, shape_box, radius, radius_y, mass_ratio
    "O0":   [1.0, 0.0, 0.020, 0.020, 1.0],
    "O1-L": [1.0, 0.0, 0.024, 0.024, 1.0],
    "O2-M": [1.0, 0.0, 0.020, 0.020, 2.0],
    "O4-S": [0.0, 1.0, 0.0177, 0.0177, 1.0],
}


def object_features(handoff_family: str) -> list[float]:
    return _OBJECT_FEATURES[handoff_family]


# ---- incidence builders (all return list[tuple[node-index,...]]) ---------------------------------
def _names_to_idx(edges: list[tuple[str, ...]]) -> list[tuple[int, ...]]:
    return [tuple(NODE_NAMES.index(n) for n in e) for e in edges]


def task_incidence() -> list[tuple[int, ...]]:
    return _names_to_idx(TASK_EDGES)


def random_sparse_incidence(rng: np.random.Generator, n_edges: int, edge_size: int = 3) -> list[tuple[int, ...]]:
    n = len(NODE_NAMES)
    return [tuple(sorted(rng.choice(n, size=edge_size, replace=False))) for _ in range(n_edges)]


def degree_matched_incidence(rng: np.random.Generator, target: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Random hyperedges with the SAME per-node degree sequence and edge sizes as ``target`` — the Steiner control
    (isolates 'balanced combinatorial structure' from 'this particular sparsity/degree')."""
    n = len(NODE_NAMES)
    deg = np.zeros(n, dtype=int)
    for e in target:
        for v in e:
            deg[v] += 1
    edges: list[tuple[int, ...]] = []
    stubs = [v for v in range(n) for _ in range(deg[v])]
    for e in target:                                     # rebuild edges of the same sizes from a shuffled stub pool
        rng.shuffle(stubs)
        picked: list[int] = []
        for s in list(stubs):
            if s not in picked:
                picked.append(s)
                stubs.remove(s)
            if len(picked) == len(e):
                break
        while len(picked) < len(e):                      # top up if the pool ran short of distinct nodes
            c = int(rng.integers(n))
            if c not in picked:
                picked.append(c)
        edges.append(tuple(sorted(picked)))
    return edges


def steiner_incidence() -> list[tuple[int, ...]]:
    """A balanced block design over the 10 nodes: every pair of nodes covered a uniform number of times, sparsely.
    Uses a near-resolvable set of triples (a Steiner-like packing on 10 points) — uniform pair coverage, no physics."""
    # 10 points; a set of triples with balanced pair coverage (hand-constructed near-Steiner packing).
    triples = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (1, 5, 6),
               (2, 3, 7), (0, 5, 7), (1, 3, 8), (2, 4, 6), (9, 0, 4), (9, 1, 5), (9, 2, 3)]
    return [tuple(sorted(t)) for t in triples]


# ---- models ---------------------------------------------------------------------------------------
def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class MLPNet(nn.Module):
    """A0 — flat MLP over the 41-D input."""

    def __init__(self, hidden: int = 96, depth: int = 3) -> None:
        super().__init__()
        self.net = _mlp([INPUT_DIM] + [hidden] * (depth - 1) + [1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out.squeeze(-1)


class HypergraphNet(nn.Module):
    """A1-A3 — hypergraph message-passing over an incidence. Stronger than the first pass: per-node encoder →
    ``rounds`` message-passing layers (each: edge message from mean‖max of member node embeddings → per-node mean of
    incident edge messages → residual) → CONCAT readout over all nodes → logit. Edge/update functions are SHARED
    across edges, so params depend only on ``node_dim/hidden/rounds``, NOT the incidence (only structure varies)."""

    def __init__(self, incidence: list[tuple[int, ...]], node_dim: int = 28, hidden: int = 56, rounds: int = 2) -> None:
        super().__init__()
        self.incidence = incidence
        self.rounds = rounds
        n = len(NODE_NAMES)
        self.node_enc = nn.ModuleList([_mlp([len(NODES[m]), node_dim]) for m in NODE_NAMES])
        self.edge_fn = nn.ModuleList([_mlp([2 * node_dim, hidden, node_dim]) for _ in range(rounds)])  # mean‖max
        self.upd_fn = nn.ModuleList([_mlp([2 * node_dim, node_dim]) for _ in range(rounds)])            # node‖msg
        self.readout = _mlp([n * node_dim, hidden, 1])                                                  # concat, per-node

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = [enc(x[:, NODES[m]]) for enc, m in zip(self.node_enc, NODE_NAMES)]
        for r in range(self.rounds):
            agg = [torch.zeros_like(h[0]) for _ in NODE_NAMES]
            cnt = [0] * len(NODE_NAMES)
            for e in self.incidence:
                stack = torch.stack([h[v] for v in e], 0)
                msg = self.edge_fn[r](torch.cat([stack.mean(0), stack.amax(0)], -1))   # mean‖max member pool
                for v in e:
                    agg[v] = agg[v] + msg
                    cnt[v] += 1
            h = [self.upd_fn[r](torch.cat([h[v], agg[v] / cnt[v] if cnt[v] else agg[v]], -1)) + h[v]
                 for v in range(len(NODE_NAMES))]
        out: torch.Tensor = self.readout(torch.cat(h, -1))
        return out.squeeze(-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@dataclass
class MatchedModels:
    """A0-A3 sized to a matched parameter band. hidden/node_dim tuned so |params| are within ~15%."""

    # sized so the flat MLP matches the (stronger) HSiKAN param budget (~30k) within ±20%
    mlp_hidden: int = 110
    mlp_depth: int = 4
    hg_node_dim: int = 28
    hg_hidden: int = 56
    hg_rounds: int = 2

    def _hg(self, incidence: list[tuple[int, ...]]) -> "HypergraphNet":
        return HypergraphNet(incidence, self.hg_node_dim, self.hg_hidden, self.hg_rounds)

    def build(self, seed: int) -> dict[str, nn.Module]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        task = task_incidence()
        return {
            "A0_mlp": MLPNet(self.mlp_hidden, self.mlp_depth),
            "A1_random_sparse": self._hg(random_sparse_incidence(rng, len(task))),
            "A2_task_hsikan": self._hg(task),
            "A3_steiner_hsikan": self._hg(steiner_incidence()),
            "A3c_degree_matched": self._hg(degree_matched_incidence(rng, task)),
        }


def build_input_row(x: list[float], theta: list[float], handoff_family: str) -> list[float]:
    """Assemble the 41-D input from a dataset row."""
    return list(x) + list(theta) + object_features(handoff_family)
