"""TreeChannel — a dynamic cross-channel between parallel actor and critic graphs.

Not a shared trunk (which *merges* the two graphs) and not a readout inside the robot graph. The actor graph and
critic graph stay **parallel and distinct**; a learned channel routes information *between* them. Two constraints
combine (Hajdu, 2026-07-01):

  * **fixed channel** — the candidate edges are exactly the **kinematic hypergraph** incidence (the channel can
    only route along real robot couplings, plus self-loops). The hypergraph's sparsity *is* the tree structure.
  * **state-adaptive, attention-selected** — since this is RL, a learned attention over those candidate edges
    weights which couplings carry actor<->critic information *at the current state* (the softmax concentrates on
    the load-bearing edges — the "selected tree" — and rewires as the state changes).

So the critic's value-structure flows to the actor and vice versa, only along kinematic couplings, weighted by
state-dependent learned attention. Dense N×N cross-attention masked by the fixed adjacency — cheap at robot scale
(N~6-14). Opt-in; the default off-policy path leaves the two graphs fully separate.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def adjacency_from_hg(hg: Any) -> torch.Tensor:
    """The FIXED channel: an ``(N, N)`` boolean adjacency (True = coupled) from a ``HypergraphState`` — two
    vertices are coupled where the signed incidence has any arc. Self-loops are added so every row has at least
    one candidate (no all-masked softmax). # Preconditions ``hg`` exposes ``dense_signed_adj`` + ``n_vertices``."""
    a_pos, a_neg = hg.dense_signed_adj(torch.device("cpu"))
    coupled = (a_pos.abs() + a_neg.abs()) > 0.0
    n = int(hg.n_vertices)
    adj: torch.Tensor = coupled.bool() | torch.eye(n, dtype=torch.bool)
    return adj


def _masked_route(q_recv: torch.Tensor, k_send: torch.Tensor, v_send: torch.Tensor,
                  neg_mask: torch.Tensor, scale: float) -> torch.Tensor:
    """One directed masked-attention channel: the receiver queries, the sender keys/values; attention is zeroed
    off the fixed hypergraph edges (``neg_mask`` is 0 on edges, ``-inf`` off). The shared kernel of both channels."""
    scores = (q_recv @ k_send.transpose(-1, -2)) * scale + neg_mask
    out: torch.Tensor = torch.softmax(scores, dim=-1) @ v_send
    return out


class TreeChannel(nn.Module):
    """State-adaptive, attention-selected cross-channel between parallel actor/critic per-node graphs, routed on
    the fixed kinematic-hypergraph adjacency. ``forward(H_actor, H_critic) -> (H_actor', H_critic')`` augments
    each graph with information routed from the other along the (learned, masked) channel.

    # Preconditions ``adj`` is a square ``(N, N)`` boolean adjacency; ``H_*`` are ``(B, N, hidden)``.
    # Invariants attention is zero off the hypergraph edges; self-loops keep every softmax row well-defined.
    """

    neg_mask: torch.Tensor

    def __init__(self, hidden: int, adj: torch.Tensor) -> None:
        super().__init__()
        if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square (N, N); got {tuple(adj.shape)}")
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        n = int(adj.shape[0])
        adj = adj.bool() | torch.eye(n, dtype=torch.bool)                  # ensure self-loops
        self.register_buffer("neg_mask", torch.where(adj, 0.0, float("-inf")))   # 0 on edges, -inf off
        self.scale = float(hidden) ** -0.5
        self.wq_a, self.wk_c, self.wv_c = (nn.Linear(hidden, hidden, bias=False) for _ in range(3))  # critic->actor
        self.wq_c, self.wk_a, self.wv_a = (nn.Linear(hidden, hidden, bias=False) for _ in range(3))  # actor->critic

    def _route(self, q_recv: torch.Tensor, k_send: torch.Tensor, v_send: torch.Tensor) -> torch.Tensor:
        return _masked_route(q_recv, k_send, v_send, self.neg_mask, self.scale)

    def forward(self, h_actor: torch.Tensor, h_critic: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        m_actor = self._route(self.wq_a(h_actor), self.wk_c(h_critic), self.wv_c(h_critic))   # value info -> actor
        m_critic = self._route(self.wq_c(h_critic), self.wk_a(h_actor), self.wv_a(h_actor))   # policy info -> critic
        return h_actor + m_actor, h_critic + m_critic


class MultiTreeChannel(nn.Module):
    """Cross-channel among ``n_agents`` parallel agent graphs sharing one kinematic hypergraph — the collaborative
    generalisation of :class:`TreeChannel`. Each agent receives state-adaptive, attention-routed messages from
    every *other* agent, along the fixed hypergraph edges. The **3-agent case (2 actors + 1 centralized critic)**
    is the coin-toss collaboration: the two arms coordinate **through the shared coin/zone couplings** on the
    joint hypergraph, and the critic informs both. Turns the fixed-backbone sharing (unreliable coordination)
    into a *learned, state-adaptive* coordination channel.

    # Preconditions ``n_agents >= 2``; ``adj`` square ``(N, N)``; each ``feats[i]`` is ``(B, N, hidden)``.
    # Postconditions returns ``n_agents`` augmented tensors, each ``feats[i]`` $+$ the mean incoming message.
    """

    neg_mask: torch.Tensor

    def __init__(self, hidden: int, adj: torch.Tensor, n_agents: int) -> None:
        super().__init__()
        if n_agents < 2:
            raise ValueError(f"n_agents must be >= 2; got {n_agents}")
        if adj.dim() != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be square (N, N); got {tuple(adj.shape)}")
        if hidden < 1:
            raise ValueError("hidden must be >= 1")
        n = int(adj.shape[0])
        adj = adj.bool() | torch.eye(n, dtype=torch.bool)
        self.register_buffer("neg_mask", torch.where(adj, 0.0, float("-inf")))
        self.scale = float(hidden) ** -0.5
        self.n_agents = int(n_agents)
        self.q = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(n_agents))  # receiver queries
        self.k = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(n_agents))  # sender keys
        self.v = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(n_agents))  # sender values

    def forward(self, feats: "list[torch.Tensor]") -> "list[torch.Tensor]":
        if len(feats) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} agent tensors; got {len(feats)}")
        outs: list[torch.Tensor] = []
        for i in range(self.n_agents):
            senders = [j for j in range(self.n_agents) if j != i]
            q_i = self.q[i](feats[i])
            msg = torch.zeros_like(feats[i])
            for j in senders:
                msg = msg + _masked_route(q_i, self.k[j](feats[j]), self.v[j](feats[j]), self.neg_mask, self.scale)
            outs.append(feats[i] + msg / float(len(senders)))
        return outs
