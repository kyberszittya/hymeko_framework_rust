"""R2 — capacity-matched HyMeKo relational theta encoder (the isolated relational-organisation axis over flat R1 v3).

The flat R1 K-head (`multimodal_proposal.KHeadProposalNet`, 25 240 params) sees the SAME 43-D canonical information as one
vector and fails held-out (dev 2/2, held 0/2 -> FLAT_R1_LEARNED_AMORTISATION_FAILS). R2 receives EXACTLY that information,
reorganised as the frozen HyMeKo typed graph (`relational_graph.CoinGraph`), and asks whether relational *organisation*
alone extracts the held-out control rule at the same parameter budget.

Architecture (contract `reports/2026-07-27-coin-r2-relational-contract.md`):

    typed node encoders  ->  EXACTLY 2 rounds of typed message passing  ->  coin+target+bimanual pooling  ->  K-head

Frozen invariants that make the comparison honest:
  * TIED weights for the two TIP / CONTACT / PORT sides (one MLP per node TYPE, applied to both rows) — the net can never
    re-learn the arbitrary L/R label; combined with the canonical graph this gives mirror + side-permutation invariance.
  * relation-type-specific linear message transforms; SUM/MEAN aggregation only; NO attention; NO recurrence (the two
    rounds carry SEPARATE weights); NO residual policy outside the K-head; NO access to physical side identity.
  * the bimanual hyperedge {contact_L, contact_R, coin, target} is an EXPLICIT message (straddle / combined push-reverse-
    brake / lateral-spin / squeeze-internal-force / L/R-balance / slew-admissible context) — the cross-term flat R1 had to
    implicitly multiply. The whole graph is NOT flattened before message passing.
  * output = Tanh -> `ThetaBox.denorm` (legal canonical theta), the SAME K-head acceptable-set semantics + `set_loss` as R1.

Deploy is a drop-in for the R1 K-head: `RelationalKHeadProposal.modes(graph)` returns K uniform-prob canonical theta
centres, decoded (inverse T_theta via `graph.was_swapped`) and searched by the UNCHANGED budget-8 centre-inclusive search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.theta_option.relational_graph import CoinGraph
from hymeko_rl.coin_delivery.theta_option.semantics import DIM, ThetaBox
from hymeko_rl.option_rl.proposal import ProposalMode

# Frozen node/relation attribute dimensions (verified against relational_graph.build_graph_from_canonical + the contract).
D_COIN, D_TARGET, D_TIP, D_CONTACT, D_PORT = 4, 3, 2, 6, 4
D_GOAL, D_AUTH, D_BIMAN = 3, 11, 8
N_ROUNDS = 2                                                    # EXACTLY two typed message-passing rounds (frozen)


def _lin(in_dim: int, out_dim: int) -> nn.Linear:
    return nn.Linear(int(in_dim), int(out_dim))


class _MessageRound(nn.Module):
    """One typed message-passing round with SEPARATE weights (not shared across rounds -> not recurrent). Every per-side
    module is TIED (the same Linear runs on both rows of a per-side type); cross-side pooling is MEAN; distinct typed
    messages into a node are combined by SUM. Nodes updated: contact, coin, target (tip/port are source-only leaves, so a
    tip's signal reaches the coin over the two rounds tip->contact->coin). Returns updated states + the bimanual readout."""

    def __init__(self, h: int):
        super().__init__()
        self.geom = _lin(2 * h, h)                             # tip_i -> contact_i
        self.auth = _lin(2 * h + D_AUTH, h)                    # port_i -> contact_i (authority edge attrs)
        self.biman = _lin(3 * h + D_BIMAN, h)                  # {mean(contact), coin, target, biman-attr} hyperedge
        self.prop = _lin(2 * h, h)                             # contact_i -> coin (mean over sides)
        self.goal = _lin(2 * h + D_GOAL, h)                    # coin -> target (goal edge attrs)
        self.u_contact = _lin(2 * h, h)
        self.u_coin = _lin(2 * h, h)
        self.u_target = _lin(2 * h, h)

    def forward(self, h_coin: torch.Tensor, h_target: torch.Tensor, h_tip: torch.Tensor, h_contact: torch.Tensor,
                h_port: torch.Tensor, auth_attr: torch.Tensor, biman_attr: torch.Tensor,
                goal_attr: torch.Tensor) -> "tuple[torch.Tensor, ...]":
        relu = torch.relu
        mean_contact = h_contact.mean(dim=0)                   # (H,) order-invariant side pool
        # messages (computed from round-start states)
        m_geom = relu(self.geom(torch.cat([h_tip, h_contact], dim=-1)))                       # (2,H)
        m_auth = relu(self.auth(torch.cat([h_port, h_contact, auth_attr], dim=-1)))           # (2,H)
        m_biman = relu(self.biman(torch.cat([mean_contact, h_coin, h_target, biman_attr])))   # (H,)
        m_prop = relu(self.prop(torch.cat([mean_contact, h_coin])))                           # (H,)
        m_goal = relu(self.goal(torch.cat([h_coin, h_target, goal_attr])))                    # (H,)
        # synchronous updates
        agg_contact = m_geom + m_auth + m_biman.unsqueeze(0)                                  # (2,H) SUM of typed msgs
        h_contact = relu(self.u_contact(torch.cat([h_contact, agg_contact], dim=-1)))
        h_coin = relu(self.u_coin(torch.cat([h_coin, m_prop + m_biman])))
        h_target = relu(self.u_target(torch.cat([h_target, m_goal + m_biman])))
        return h_coin, h_target, h_tip, h_contact, h_port, m_biman


class RelationalKHeadNet(nn.Module):
    """Typed relational encoder -> shared pooled embedding -> K bounded 6-D heads. forward(graph tensors) -> (K, 6)
    normalised theta in [-1,1] (Tanh). The two rounds carry separate weights; the K heads are distinct linear read-outs of
    the same pooled encoding (the SAME K-head contract as R1's `heads`)."""

    def __init__(self, k: int, h: int = 25):
        super().__init__()
        self.k, self.h = int(k), int(h)
        self.enc_coin = _lin(D_COIN, h)
        self.enc_target = _lin(D_TARGET, h)
        self.enc_tip = _lin(D_TIP, h)                          # TIED across the two tip rows
        self.enc_contact = _lin(D_CONTACT, h)                  # TIED across the two contact rows
        self.enc_port = _lin(D_PORT, h)                        # TIED across the two port rows
        self.rounds = nn.ModuleList([_MessageRound(h) for _ in range(N_ROUNDS)])
        self.heads = _lin(4 * h, self.k * DIM)                 # pool = [coin, target, mean(contact), bimanual] -> K*6

    def forward(self, t: "dict[str, torch.Tensor]") -> torch.Tensor:
        h_coin = torch.relu(self.enc_coin(t["coin"]))
        h_target = torch.relu(self.enc_target(t["target"]))
        h_tip = torch.relu(self.enc_tip(t["tip"]))             # (2,H)
        h_contact = torch.relu(self.enc_contact(t["contact"]))
        h_port = torch.relu(self.enc_port(t["port"]))
        m_biman = h_contact.mean(dim=0)                        # placeholder until first round overwrites it
        for rnd in self.rounds:
            h_coin, h_target, h_tip, h_contact, h_port, m_biman = rnd(
                h_coin, h_target, h_tip, h_contact, h_port, t["auth"], t["biman"], t["goal"])
        pool = torch.cat([h_coin, h_target, h_contact.mean(dim=0), m_biman])   # (4H,) order-invariant
        z = torch.tanh(self.heads(pool))
        return z.reshape(self.k, DIM)


def graph_tensors(graph: CoinGraph, dtype: torch.dtype = torch.float32) -> "dict[str, torch.Tensor]":
    """CoinGraph -> the fixed-shape tensors the net consumes. PURE (no learned state) -> mirror/permutation tests can drive
    it directly. # Postconditions: keys {coin(4), target(3), tip(2,2), contact(2,6), port(2,4), auth(2,11), biman(8),
    goal(3)}; canonical node order preserved (the net is invariant to it by tied weights + mean pooling)."""
    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a, np.float32), dtype=dtype)
    return {"coin": _t(graph.node("COIN")), "target": _t(graph.node("TARGET")), "tip": _t(graph.nodes["TIP"]),
            "contact": _t(graph.nodes["CONTACT"]), "port": _t(graph.nodes["PORT"]), "auth": _t(graph.authority),
            "biman": _t(graph.bimanual), "goal": _t(graph.goal)}


@dataclass
class RelationalKHeadProposal:
    """A trained relational K-head conforming to `option_rl.MultimodalProposalPolicy` — a drop-in for the flat R1
    `KHeadProposal`. `modes(graph)` returns K uniform-prob `ProposalMode`s with LEGAL canonical theta centres (Tanh ->
    `ThetaBox.denorm`). The physical decode (inverse T_theta via graph.was_swapped) + budget-8 search stay in the deploy."""

    k: int
    net: RelationalKHeadNet
    box: ThetaBox

    def _heads(self, graph: CoinGraph) -> np.ndarray:
        with torch.no_grad():
            z = self.net(graph_tensors(graph)).numpy()        # (K,6) normalised canonical theta
        return np.asarray([self.box.denorm(zk) for zk in z], np.float64)

    def modes(self, graph: CoinGraph) -> "list[ProposalMode]":
        centers = self._heads(graph)
        p = 1.0 / self.k
        return [ProposalMode(prob=p, center=np.asarray(centers[j], np.float32), std=None, mode_id=j) for j in range(self.k)]


@dataclass
class RelKHeadTrainState:
    """One dev state's R2 training example: the typed graph + its acceptable set (canonical normalised theta) + the
    multimodal gate. `targets_norm` is IDENTICAL to the flat R1 example for the same cradle — only the input changes."""

    tag: str
    graph: CoinGraph
    targets_norm: np.ndarray
    multimodal: bool


def count_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def fit_relational_khead(states: "list[RelKHeadTrainState]", k: int, *, epochs: int = 1500, lr: float = 1e-3, seed: int = 0,
                         diversity_weight: float = 0.5, h: int = 25) -> "tuple[RelationalKHeadProposal, dict[str, Any]]":
    """Fit the relational K-head on the per-state acceptable sets with the FROZEN permutation-invariant `set_loss`. The
    training contract is byte-for-byte the R1 `fit_khead` procedure (full-batch: one set-loss term per dev state per epoch,
    mean over states, one Adam step; deterministic given ``seed``) — ONLY the model (graph encoder vs flat MLP) differs.
    # Preconditions: every state has >=1 target. # Postconditions: returns (proposal, {final_loss, per_state_recall, ...})."""
    from hymeko_rl.coin_delivery.theta_option.multimodal_proposal import set_loss
    torch.manual_seed(seed)
    net = RelationalKHeadNet(k, h=h)
    opt = torch.optim.Adam(net.parameters(), lr)
    tens = [graph_tensors(s.graph) for s in states]
    tgts = [torch.as_tensor(np.asarray(s.targets_norm, np.float32)) for s in states]
    last = 0.0
    for _ in range(int(epochs)):
        opt.zero_grad()
        total = torch.zeros(())
        for tn, tg, s in zip(tens, tgts, states):
            heads = net(tn)
            total = total + set_loss(heads, tg, diversity_weight=diversity_weight, multimodal=s.multimodal)
        (total / max(1, len(states))).backward()
        opt.step()
        last = float(total.item()) / max(1, len(states))
    prop = RelationalKHeadProposal(k=k, net=net, box=ThetaBox())
    with torch.no_grad():
        rec = {}
        for tn, tg, s in zip(tens, tgts, states):
            heads = net(tn)
            d2 = ((heads[:, None, :] - tg[None, :, :]) ** 2).sum(-1)
            rec[s.tag] = round(float(torch.sqrt(d2.min(dim=0).values + 1e-9).mean()), 4)
    return prop, {"final_loss": round(last, 6), "per_state_recall": rec, "k": k, "h": h, "epochs": epochs,
                  "trainable_params": count_params(net)}


def relational_deploy_one(snap: Any, prop: RelationalKHeadProposal, graph: CoinGraph, rng: Any, budget: int,
                          box: ThetaBox, *, cfg: Any = None, scorer: Any = None) -> "dict[str, Any]":
    """R2 deploy on ONE cradle — the SAME procedure as the flat R1 `_r1_deploy_one`, graph input only: predict K CANONICAL
    theta heads -> decode each to physical theta (inverse T_theta via ``graph.was_swapped``) -> fair budget-8 split
    (`allocate_budget`) -> coin CENTRE-INCLUSIVE `fixed_search_select` per mode -> global argmax. ``scorer`` is injectable
    for tests (physics otherwise). Records the full provenance. # Postconditions: sum(per_mode_budget)==budget; exactly
    ``budget`` candidate rollouts; every centre decoded, none aliased."""
    from hymeko_rl.coin_delivery.theta_option.canonical_frame import from_canonical_theta
    from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, fixed_search_select
    from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
    from hymeko_rl.option_rl.proposal import allocate_budget
    cfg = cfg if cfg is not None else DELIVERY_CFG
    modes = prop.modes(graph)
    canon = [np.asarray(m.center, np.float64) for m in modes]
    phys = [from_canonical_theta(c, graph.was_swapped) for c in canon]           # inverse T_theta -> physical
    per = allocate_budget([1.0 / len(modes)] * len(modes), int(budget))
    children = list(rng.spawn(len(modes))) if len(modes) > 1 else [rng]
    best = None
    for k, (c, b_k) in enumerate(zip(phys, per)):
        if b_k <= 0:
            continue
        prov = fixed_search_select(snap, np.asarray(c, np.float64), children[k], budget=int(b_k), cfg=cfg,
                                   std=SEARCH_STD, scorer=scorer)
        if best is None or prov.score > best[1].score:
            best = (k, prov)
    sel, prov = best
    disp = float(np.linalg.norm(box.norm(prov.selected) - box.norm(phys[sel])))
    o = prov.outcome
    ok = bool(o.get("delivery_success"))
    fail = None
    if not ok:
        from hymeko_rl.coin_delivery.theta_option.cradle_expansion import classify_failure_mode
        fail = classify_failure_mode({"dtz_end_mm": o.get("dtz_end", 0.0) * 1000, "k6_max_dwell": o.get("k6_max_dwell", 0),
                                      "peak_qdot": o.get("peak_qdot", 0.0), "peak_coin_speed": o.get("peak_coin_speed", 0.0)})
    return {"selected_head": int(sel), "n_modes": len(modes), "per_mode_budget": per, "was_swapped": bool(graph.was_swapped),
            "canonical_heads": [[round(float(x), 4) for x in c] for c in canon],
            "decoded_physical_centres": [[round(float(x), 4) for x in c] for c in phys],
            "theta_exec": [round(float(x), 4) for x in prov.selected], "search_displacement_norm": round(disp, 4),
            "budget_total": int(sum(per)), "k6_delivered": bool(o.get("k6_delivered")), "delivery_success": ok,
            "k6_max_dwell": int(o.get("k6_max_dwell", 0)), "dtz_start_mm": round(o.get("dtz_start", 0.0) * 1000, 2),
            "dtz_end_mm": round(o.get("dtz_end", 0.0) * 1000, 2), "zone_entry": bool(o.get("dtz_end", 1.0) <= 0.02),
            "terminal_coin_speed": round(o.get("terminal_coin_speed", 0.0), 4), "peak_qdot": round(o.get("peak_qdot", 0.0), 4),
            "peak_coin_speed": round(o.get("peak_coin_speed", 0.0), 4), "failure_phase": fail}


def save_relational_khead(prop: RelationalKHeadProposal, path: str) -> None:
    torch.save({"k": prop.k, "h": prop.net.h, "state_dict": prop.net.state_dict()}, path)


def load_relational_khead(path: str) -> RelationalKHeadProposal:
    blob = torch.load(path, weights_only=False)
    net = RelationalKHeadNet(int(blob["k"]), h=int(blob["h"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return RelationalKHeadProposal(k=int(blob["k"]), net=net, box=ThetaBox())
