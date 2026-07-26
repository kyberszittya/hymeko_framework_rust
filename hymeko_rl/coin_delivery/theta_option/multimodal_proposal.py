"""M1 — K-head acceptable-set (multimodal) proposal: the fix M0 justified.

M0 established the coin held-out failure is MODALITY, not representation: a single MSE centre averages the distant valid
modes into a non-delivering θ (4/6 dev acceptable-set centroids miss), and the single-θ regressor collapses both held-out
proposals into one wrong region. The fix is a proposal that emits K DISTINCT legal-θ modes instead of one interpolated
centre, so the fixed budget-8 search can reach a delivering mode.

    obs (frozen B0 features, 42-D) → shared trunk → K bounded 6-D heads (Tanh → legal θ), uniform mode prob.

This implements `option_rl.MultimodalProposalPolicy` (K=1 recovers the single-θ B0). DEPLOY reuses `option_rl.allocate_
budget` for the FAIR fixed-total-budget split (K × 8/K) and the coin's CENTRE-INCLUSIVE `fixed_search_select` per mode —
NOT the generic `MultimodalBudgetSearch`, which (like `FixedBudgetSearch`) never evaluates a mode centre when that mode
draws ≥2 candidates; centre-inclusion is the coin's established correctness fix and must hold per mode so the learned
mode-centres are actually evaluated. Total physical budget is 8 for every K — the multimodal model gets no extra tries.

LOSS = permutation-invariant bidirectional Chamfer over the state's acceptable set (RECALL: every acceptable mode is
covered by some head; PRECISION: every head lands on a real acceptable θ, not an average between modes) + a head-collapse
penalty GATED on the acceptable set actually being multimodal (else it would force spurious spread on a unimodal state).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.contact_velocity import CradleSnapshot
from hymeko_rl.coin_delivery.forward_displacement import ForwardConfig
from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, fixed_search_select
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG, DIM, ThetaBox, ThetaProvenance
from hymeko_rl.option_rl.proposal import ProposalMode, allocate_budget

FEATURE_DIM = 42
MULTIMODAL_TARGET_SPREAD = 0.8       # a target set whose max pairwise (normalised) distance exceeds this is multimodal
DIVERSITY_MARGIN = 0.8               # heads should be ≥ this apart (normalised) on a multimodal state


class KHeadProposalNet(nn.Module):
    """Shared B0-feature trunk → K bounded 6-D heads. forward(feats) → (B, K, 6) normalised θ ∈ [-1,1] (Tanh). The heads
    share the trunk so the modes are conditioned on the same state encoding; each head is a distinct linear read-out."""

    def __init__(self, k: int, h: int = 128):
        super().__init__()
        self.k = int(k)
        self.trunk = nn.Sequential(nn.Linear(FEATURE_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.heads = nn.Linear(h, self.k * DIM)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.heads(self.trunk(feats)))
        return z.reshape(z.shape[0], self.k, DIM)


@dataclass
class KHeadProposal:
    """A trained K-head proposal conforming to `option_rl.MultimodalProposalPolicy`. `modes(obs)` returns K uniform-prob
    `ProposalMode`s with LEGAL θ centres (Tanh output → `ThetaBox.denorm`, always in-box). K=1 ≡ the single-θ proposal."""

    k: int
    net: KHeadProposalNet
    box: ThetaBox

    def _heads(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.net(torch.as_tensor(np.asarray(obs, np.float32))[None])[0].numpy()   # (K,6) normalised
        return np.asarray([self.box.denorm(zk) for zk in z], np.float64)

    def modes(self, obs: np.ndarray) -> "list[ProposalMode]":
        centers = self._heads(obs)
        p = 1.0 / self.k
        return [ProposalMode(prob=p, center=np.asarray(centers[k], np.float32), std=None, mode_id=k) for k in range(self.k)]


def is_multimodal_target_set(targets_norm: np.ndarray, spread: float = MULTIMODAL_TARGET_SPREAD) -> bool:
    """True if the acceptable set spans multiple modes (max pairwise normalised distance > ``spread``). Gates the
    head-collapse penalty: forcing head diversity on a UNIMODAL state would push heads off the single valid mode."""
    X = np.asarray(targets_norm, np.float64)
    if len(X) < 2:
        return False
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return bool(D.max() > spread)


def set_loss(heads_norm: torch.Tensor, targets_norm: torch.Tensor, *, diversity_weight: float = 0.5,
             multimodal: bool = False, margin: float = DIVERSITY_MARGIN) -> torch.Tensor:
    """Permutation-invariant acceptable-set loss for ONE state. ``heads_norm`` (K,6), ``targets_norm`` (M,6), both in
    normalised θ. RECALL = mean over targets of the nearest-head squared distance (cover every acceptable mode); PRECISION
    = mean over heads of the nearest-target squared distance (every head on a real acceptable θ, no between-mode average).
    On a MULTIMODAL state, a hinge head-collapse penalty pushes the closest head pair ≥ ``margin`` apart. Invariant to head
    and target ordering. # Postconditions: scalar ≥ 0."""
    H, T = heads_norm, targets_norm
    d2 = ((H[:, None, :] - T[None, :, :]) ** 2).sum(-1)          # (K, M) squared distances
    recall = d2.min(dim=0).values.mean()                        # each target → nearest head
    precision = d2.min(dim=1).values.mean()                     # each head → nearest target
    loss = recall + precision
    if multimodal and H.shape[0] >= 2:
        hh = ((H[:, None, :] - H[None, :, :]) ** 2).sum(-1)     # (K,K) head-head sq dist
        eye = torch.eye(H.shape[0], dtype=torch.bool)
        min_pair = torch.sqrt(hh[~eye].reshape(H.shape[0], -1).min(dim=1).values + 1e-9).min()
        loss = loss + diversity_weight * torch.relu(torch.tensor(margin) - min_pair)
    return loss


@dataclass
class KHeadTrainState:
    """One dev state's training example: the frozen B0 features + its acceptable set (normalised θ) + the multimodal gate."""

    tag: str
    features: np.ndarray
    targets_norm: np.ndarray
    multimodal: bool


def fit_khead(states: "list[KHeadTrainState]", k: int, *, epochs: int = 1500, lr: float = 1e-3, seed: int = 0,
              diversity_weight: float = 0.5, h: int = 128) -> "tuple[KHeadProposal, dict[str, Any]]":
    """Fit the K-head proposal on the per-state acceptable sets with the permutation-invariant set loss. Deterministic
    given ``seed``. Each state contributes one set-loss term (variable-size target set). # Preconditions: every state has
    ≥1 target. # Postconditions: returns (proposal, {final_loss, per_state_recall})."""
    torch.manual_seed(seed)
    net = KHeadProposalNet(k, h=h)
    opt = torch.optim.Adam(net.parameters(), lr)
    feats = [torch.as_tensor(np.asarray(s.features, np.float32))[None] for s in states]
    tgts = [torch.as_tensor(np.asarray(s.targets_norm, np.float32)) for s in states]
    last = 0.0
    for _ in range(int(epochs)):
        opt.zero_grad()
        total = torch.zeros(())
        for f, t, s in zip(feats, tgts, states):
            heads = net(f)[0]
            total = total + set_loss(heads, t, diversity_weight=diversity_weight, multimodal=s.multimodal)
        (total / max(1, len(states))).backward()
        opt.step()
        last = float(total.item()) / max(1, len(states))
    prop = KHeadProposal(k=k, net=net, box=ThetaBox())
    # per-state recall (mean nearest-head distance) as an offline coverage diagnostic (NOT the deploy metric)
    with torch.no_grad():
        rec = {}
        for f, t, s in zip(feats, tgts, states):
            heads = net(f)[0]
            d2 = ((heads[:, None, :] - t[None, :, :]) ** 2).sum(-1)
            rec[s.tag] = round(float(torch.sqrt(d2.min(dim=0).values + 1e-9).mean()), 4)
    return prop, {"final_loss": round(last, 6), "per_state_recall": rec, "k": k, "epochs": epochs}


@dataclass
class MultimodalDeploy:
    """The multimodal deploy result on one cradle: which mode won, its θ_exec provenance, the per-mode budget split, and
    the frozen K6 outcome. The Bellman action for RL would be the SELECTED MODE's centre (kept separate from θ_exec)."""

    selected_mode: int
    n_modes: int
    per_mode_budget: "list[int]"
    mode_centers: "list[np.ndarray]"
    provenance: ThetaProvenance

    def as_dict(self) -> "dict[str, Any]":
        pr = self.provenance
        return {"selected_mode": int(self.selected_mode), "n_modes": int(self.n_modes),
                "per_mode_budget": list(self.per_mode_budget),
                "theta0_selected_mode": [round(float(x), 5) for x in self.mode_centers[self.selected_mode]],
                "theta_exec": [round(float(x), 5) for x in pr.selected],
                "k6_delivered": bool(pr.outcome.get("k6_delivered")),
                "delivery_success": bool(pr.outcome.get("delivery_success")),
                "k6_max_dwell": int(pr.outcome.get("k6_max_dwell", 0)),
                "dtz_end_mm": round(pr.outcome.get("dtz_end", 0.0) * 1000, 2),
                "terminal_coin_speed": round(pr.outcome.get("terminal_coin_speed", 0.0), 4),
                "peak_qdot": round(pr.outcome.get("peak_qdot", 0.0), 4),
                "peak_coin_speed": round(pr.outcome.get("peak_coin_speed", 0.0), 4)}


def multimodal_search_select(snap: CradleSnapshot, proposal: KHeadProposal, obs: np.ndarray, rng: np.random.Generator, *,
                             budget: int = 8, cfg: ForwardConfig = DELIVERY_CFG, std: float = SEARCH_STD) -> MultimodalDeploy:
    """Deploy the K-head proposal on one cradle at a FIXED total ``budget``. Split the budget across the K modes with
    `option_rl.allocate_budget` (≥1 each, remainder ∝ prob — for uniform prob this is the even K×(budget/K) split), then
    run the coin's CENTRE-INCLUSIVE `fixed_search_select` around EACH mode centre with its allocated budget, and keep the
    global argmax of the frozen delivery score. Each mode gets an independent child rng keyed by canonical mode identity
    (order-invariant). # Preconditions: budget ≥ 1. # Postconditions: sum(per_mode_budget) == budget; exactly ``budget``
    candidate rollouts; the winning mode/θ are order-invariant."""
    modes = proposal.modes(np.asarray(obs, np.float32))
    centers = [np.asarray(m.center, np.float64) for m in modes]
    probs = [float(max(0.0, m.prob)) for m in modes]
    per_mode = allocate_budget(probs, int(budget))
    children = list(rng.spawn(len(modes))) if len(modes) > 1 else [rng]
    rank = {i: r for r, i in enumerate(sorted(range(len(modes)),
            key=lambda i: (int(modes[i].mode_id), centers[i].astype(np.float64).tobytes())))}
    best: "tuple[int, ThetaProvenance] | None" = None
    for mi, (c, b_k) in enumerate(zip(centers, per_mode)):
        if b_k <= 0:
            continue
        prov = fixed_search_select(snap, c, children[rank[mi]], budget=int(b_k), cfg=cfg, std=std)
        if best is None or prov.score > best[1].score:
            best = (mi, prov)
    assert best is not None, "multimodal_search_select: no mode received budget"
    return MultimodalDeploy(selected_mode=best[0], n_modes=len(modes), per_mode_budget=per_mode,
                            mode_centers=centers, provenance=best[1])


def save_khead(prop: KHeadProposal, path: str) -> None:
    torch.save({"k": prop.k, "state_dict": prop.net.state_dict()}, path)


def load_khead(path: str) -> KHeadProposal:
    blob = torch.load(path, weights_only=False)
    net = KHeadProposalNet(int(blob["k"]))
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return KHeadProposal(k=int(blob["k"]), net=net, box=ThetaBox())
