"""R2 graph adapter — the verified R1 canonical extractor → explicit HyMeKo typed nodes/edges/bimanual-hyperedge.

Per the frozen R2 contract (`reports/2026-07-27-coin-r2-relational-contract.md`): R2 receives EXACTLY the R1 v3 canonical
physical information, only reorganised as relations. The graph is built from the ALREADY-canonicalised R1 grouped features
(so the L/R node ordering is the canonical one — mirror equivalence + permutation invariance hold by construction), and
every R1 v3 group has exactly one home so information parity is testable both ways. The two sides share a node TYPE (and,
downstream, tied encoder weights), so the arbitrary L/R label cannot be re-learned.

NOT built from the env's raw `node_features()` (which may carry world coords / raw joint angles / the original L/R label /
non-canonical order) — the invariance comes from the R1 canonical extractor, audited, not recovered post-hoc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.theta_option.canonical_frame import canonicalise, r1_grouped_features

# Node type → the ordered R1 groups whose CANONICAL-side value fills that node. Singletons (COIN/TARGET) take shared
# groups; per-side types (TIP/CONTACT/PORT) take side index s ∈ {0,1}. Tied weights ⇒ TIP_A and TIP_B use the same MLP.
COIN_GROUPS = ("dtz", "coin_vel_along", "coin_vel_perp")                       # + phase (const 0 at t=0)
TIP_GROUPS = ("tip_coin_along", "tip_coin_perp")                              # per side
CONTACT_GROUPS = ("normal_along", "normal_perp", "vrel_normal", "vrel_tangent", "fn", "friction_util")   # per side
PORT_GROUPS = ("slew_head_up", "slew_head_dn", "prev_tau_arm", "btau_side_auth")                         # per side
# Edge/hyperedge attribute groups
GOAL_GROUPS = ("dtz", "coin_vel_along", "coin_vel_perp")                       # coin→target
AUTH_SHARED_GROUPS = ("btau_svals", "btau_summary", "forward_push_reach", "forward_reverse_reach",
                      "brake_opposed_reach", "bcoin_min_attenuation")           # port_i→contact_i→coin (+ side lateral)
BIMANUAL_GROUPS = ("straddle", "normal_force_reach_pair", "balance_reach_signed",
                   "forward_push_reach", "brake_opposed_reach", "lateral_reach_pair")   # {contact_L,contact_R,coin,target}


def _cat(g: dict[str, np.ndarray], names: "tuple[str, ...]") -> np.ndarray:
    return np.concatenate([np.asarray(g[n], np.float64).ravel() for n in names]).astype(np.float64)


def _side(g: dict[str, np.ndarray], names: "tuple[str, ...]", s: int) -> np.ndarray:
    return np.array([float(np.asarray(g[n], np.float64).ravel()[s]) for n in names], np.float64)


@dataclass
class CoinGraph:
    """Typed HyMeKo graph for one cradle at the handoff. `nodes[type] = list of feature vectors` (two entries for the
    per-side types, canonical order A,B); `edges[relation] = list of (src_idx, dst_idx, attr)`; `bimanual` is the
    hyperedge attribute over {contact_A, contact_B, coin, target}. `was_swapped` decodes a predicted canonical θ."""

    nodes: dict[str, np.ndarray]                  # type -> (n_type, d_type) canonical-ordered node features
    goal: np.ndarray                              # coin→target edge attributes
    authority: np.ndarray                         # (2, d) per-side port→contact→coin authority (canonical order)
    bimanual: np.ndarray                          # the {L,R,coin,target} hyperedge attributes
    was_swapped: bool

    def node(self, t: str, i: int = 0) -> np.ndarray:
        return self.nodes[t][i]


def build_graph_from_canonical(g: dict[str, np.ndarray], was_swapped: bool) -> CoinGraph:
    """Assemble the typed graph from ALREADY-canonicalised R1 grouped features. PURE (no physics) ⇒ directly testable for
    mirror equivalence + permutation invariance + information parity. # Postconditions: TIP/CONTACT/PORT have 2 canonical-
    ordered rows; every attribute traces to an R1 group."""
    coin = np.concatenate([_cat(g, COIN_GROUPS), np.array([0.0])])            # +phase placeholder (const at t=0)
    target = np.array([CENTER_TOL, SETTLE_VEL, float(HELD_DWELL)], np.float64)
    tip = np.stack([_side(g, TIP_GROUPS, 0), _side(g, TIP_GROUPS, 1)])
    contact = np.stack([_side(g, CONTACT_GROUPS, 0), _side(g, CONTACT_GROUPS, 1)])
    port = np.stack([_side(g, PORT_GROUPS, 0), _side(g, PORT_GROUPS, 1)])
    shared_auth = _cat(g, AUTH_SHARED_GROUPS)
    lateral = np.asarray(g["lateral_reach_pair"], np.float64).ravel()
    authority = np.stack([np.concatenate([shared_auth, lateral[[0]]]),        # side A gets lateral[0]
                          np.concatenate([shared_auth, lateral[[1]]])])        # side B gets lateral[1]
    return CoinGraph(nodes={"COIN": coin[None, :], "TARGET": target[None, :], "TIP": tip, "CONTACT": contact, "PORT": port},
                     goal=_cat(g, GOAL_GROUPS), authority=authority, bimanual=_cat(g, BIMANUAL_GROUPS),
                     was_swapped=bool(was_swapped))


def build_graph(snap: CradleSnapshot) -> CoinGraph:
    """Physics entry: canonicalise the R1 features, then assemble the graph. Returns the graph (carrying `was_swapped` for
    the θ decode)."""
    g, sw = canonicalise(r1_grouped_features(snap))
    return build_graph_from_canonical(g, sw)


def recover_r1_groups(graph: CoinGraph) -> dict[str, np.ndarray]:
    """INFORMATION-PARITY inverse: reconstruct every R1 v3 group value from the graph node/edge attributes (proves nothing
    was added or lost). # Postconditions: keys ⊇ the informative R1 groups; values match the canonical grouped features."""
    coin = graph.node("COIN")
    tip, contact, port = graph.nodes["TIP"], graph.nodes["CONTACT"], graph.nodes["PORT"]
    auth = graph.authority
    bim = graph.bimanual
    out: dict[str, np.ndarray] = {}
    # coin (drop the trailing phase placeholder)
    for k, name in enumerate(COIN_GROUPS):
        out[name] = np.array([coin[k]])
    # per-side node groups
    for gi, name in enumerate(TIP_GROUPS):
        out[name] = np.array([tip[0][gi], tip[1][gi]])
    for gi, name in enumerate(CONTACT_GROUPS):
        out[name] = np.array([contact[0][gi], contact[1][gi]])
    for gi, name in enumerate(PORT_GROUPS):
        out[name] = np.array([port[0][gi], port[1][gi]])
    # shared authority (from side-A authority edge; identical shared prefix on both sides)
    off = 0
    from hymeko_rl.coin_delivery.theta_option.canonical_frame import group_len
    for name in AUTH_SHARED_GROUPS:
        n = group_len(name)
        out[name] = auth[0][off:off + n].copy()
        off += n
    out["lateral_reach_pair"] = np.array([auth[0][-1], auth[1][-1]])          # each side's lateral in its authority edge
    # bimanual-only groups (straddle, normal_force_reach_pair, balance_reach_signed live only here)
    b = 0
    for name in ("straddle", "normal_force_reach_pair", "balance_reach_signed"):
        n = group_len(name)
        out[name] = bim[b:b + n].copy()
        b += n
    return out
