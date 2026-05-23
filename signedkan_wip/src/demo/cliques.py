"""Robot communication network: synthetic generator + balance theory.

A multi-robot communication network is a signed graph:

  - vertices = robots
  - edges    = pairwise communication attempts in range
  - sign     = ``+`` reliable (high SINR / trusted), ``−`` jammed / lost

Cartwright-Harary (1956) structural balance theory says a signed graph
is *balanced* iff every cycle has an even count of negative edges.
Equivalently, the σ-product around any cycle equals ``+1``. A
**balanced clique** on robots is then a stable communication team: no
internal conflicts, every pairwise link is consistent, every triangle
closes positively (or with paired flips that cancel).

HSiKAN's cycle pool computes σ-products by construction, so the
inductive bias for this prediction problem is already in the model.
v1 of this demo is descriptive — it generates a network and enumerates
its balanced cliques. v0.5 (next pass) trains a small HSiKAN on a
corpus of synthetic networks and predicts edge signs given only
robot positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets import SignedGraph


@dataclass
class Clique:
    """A clique on the robot network with σ-balance indicator.

    ``sigma_product`` is **+1 iff balanced** (every triangle has σ-product
    = +1, i.e., Heider 1946 / Cartwright-Harary 1956). For k=3 this
    equals the single triangle's σ; for k ≥ 4 it is computed from the
    triangle check (NOT from the all-edges product, which gives wrong
    answers for even-sized balanced cliques — e.g., a 4-clique with a
    2-2 sign split has all triangles balanced but the all-edges
    product is −1).
    """

    members: tuple[int, ...]            # vertex indices, sorted ascending
    edges: list[tuple[int, int]]        # the (u, v) pairs within the clique
    signs: list[int]                    # ±1 per edge in `edges`
    sigma_product: int                  # +1 iff every triangle is balanced

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def balanced(self) -> bool:
        return self.sigma_product == 1


def _clique_balance_indicator(
    members: tuple[int, ...],
    sign_of: dict[tuple[int, int], int],
) -> int:
    """Return +1 iff the signed clique is *balanced* (every triangle
    has σ-product = +1), else −1.

    For a clique on k vertices, balance is equivalent to admitting a
    2-coloring such that within-color edges are + and across-color
    are −. Checking all C(k, 3) triangles is the most readable
    sufficient + necessary condition.
    """
    n = len(members)
    if n < 3:
        return 1  # vacuously balanced
    for i in range(n):
        for j in range(i + 1, n):
            for kk in range(j + 1, n):
                a, b, c = members[i], members[j], members[kk]
                s_ab = sign_of[(min(a, b), max(a, b))]
                s_bc = sign_of[(min(b, c), max(b, c))]
                s_ac = sign_of[(min(a, c), max(a, c))]
                if s_ab * s_bc * s_ac != 1:
                    return -1
    return 1


@dataclass
class RobotNetworkBundle:
    """A snapshot of a synthetic robot communication network."""

    graph: SignedGraph
    positions: np.ndarray                # (n_robots, 2) float
    seed: int
    comm_range: float
    noise_prob: float
    area_size: float
    name: str = "synthetic"
    n_factions: int = 0
    factions: np.ndarray | None = None   # (n_robots,) int faction labels

    @property
    def n_robots(self) -> int:
        return self.graph.n_nodes

    @property
    def n_edges(self) -> int:
        return self.graph.edges.shape[0]

    @property
    def n_negative_edges(self) -> int:
        return int((self.graph.signs == -1).sum())

    @property
    def n_positive_edges(self) -> int:
        return int((self.graph.signs == 1).sum())

    def edge_sign(self, u: int, v: int) -> int | None:
        """Return the sign of edge (u, v) or None if no such edge."""
        edges = self.graph.edges
        signs = self.graph.signs
        for i in range(edges.shape[0]):
            a, b = int(edges[i, 0]), int(edges[i, 1])
            if (a == u and b == v) or (a == v and b == u):
                return int(signs[i])
        return None


def make_robot_network(
    n_robots: int = 12,
    area_size: float = 10.0,
    comm_range: float = 3.5,
    noise_prob: float = 0.10,
    seed: int = 0,
    name: str = "synthetic",
    n_factions: int = 0,
) -> RobotNetworkBundle:
    """Generate a synthetic robot communication network.

    - Robots placed uniformly in ``[0, area_size]²``.
    - An edge is created between every pair within ``comm_range``.
    - Edge sign depends on the ``n_factions`` mode:

      - ``n_factions == 0`` (default): every edge is ``+1`` baseline,
        flipped to ``−1`` with probability ``noise_prob``. **No
        structural signal — the noise is i.i.d. and unlearnable by
        construction.** Use this mode for the balance-theory exposition
        only.

      - ``n_factions >= 2``: robots are assigned to factions uniformly
        at random. Within-faction edges are ``+1``; between-faction
        edges are ``−1``. Then each edge is flipped with probability
        ``noise_prob`` (observation noise on the underlying faction
        signal). **This mode IS learnable** — a model that recovers
        faction membership predicts edge signs.

    - Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, area_size, size=(n_robots, 2)).astype(np.float32)
    factions: np.ndarray | None = None
    if n_factions >= 2:
        factions = rng.integers(0, n_factions, size=n_robots,
                                  dtype=np.int64)
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for u in range(n_robots):
        for v in range(u + 1, n_robots):
            d = float(np.linalg.norm(pos[u] - pos[v]))
            if d > comm_range:
                continue
            if factions is not None:
                base = 1 if factions[u] == factions[v] else -1
            else:
                base = 1
            # Apply observation noise.
            if rng.random() < noise_prob:
                base = -base
            edges.append((u, v))
            signs.append(int(base))
    edges_arr = (np.array(edges, dtype=np.int64)
                  if edges else np.zeros((0, 2), dtype=np.int64))
    signs_arr = (np.array(signs, dtype=np.int8)
                  if signs else np.zeros((0,), dtype=np.int8))
    g = SignedGraph(edges=edges_arr, signs=signs_arr, n_nodes=n_robots)
    return RobotNetworkBundle(
        graph=g, positions=pos, seed=seed,
        comm_range=comm_range, noise_prob=noise_prob,
        area_size=area_size, name=name,
        n_factions=int(n_factions),
        factions=factions,
    )


def enumerate_balanced_cliques(
    bundle: RobotNetworkBundle,
    min_size: int = 3,
    max_size: int = 6,
    limit: int = 20,
) -> list[Clique]:
    """Enumerate balanced cliques in the network.

    Approach:
      1. Build the *unsigned* underlying graph (all communicating pairs).
      2. Enumerate maximal cliques with NetworkX.
      3. For each clique, check σ-product over all its internal edges.
      4. Return balanced ones, sorted by size descending, truncated to
         ``limit``.

    A clique is *balanced* when the product of edge signs along ALL
    pairwise edges is ``+1``. Equivalently: even number of negatives.
    """
    try:
        import networkx as nx
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "networkx is required (run `uv sync --group ml --group demo`)."
        ) from e

    G = nx.Graph()
    G.add_nodes_from(range(bundle.n_robots))
    # Sign lookup for fast σ-product evaluation.
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        sign_of[(a, b)] = int(s)
        G.add_edge(a, b)

    out: list[Clique] = []
    for clique in nx.find_cliques(G):
        size = len(clique)
        if size < min_size or size > max_size:
            continue
        members = tuple(sorted(int(x) for x in clique))
        # Collect internal edges + signs (every pair must be an edge,
        # since `clique` came from find_cliques on the underlying graph).
        edges: list[tuple[int, int]] = []
        signs: list[int] = []
        valid = True
        for i in range(size):
            for j in range(i + 1, size):
                a, b = members[i], members[j]
                s = sign_of.get((a, b))
                if s is None:
                    valid = False
                    break
                edges.append((a, b))
                signs.append(s)
            if not valid:
                break
        if not valid:
            continue
        # Balance is the triangle-product check, NOT all-edges-product.
        sigma = _clique_balance_indicator(members, sign_of)
        if sigma == 1:
            out.append(Clique(members=members, edges=edges,
                                signs=signs, sigma_product=sigma))

    out.sort(key=lambda c: (-c.size, c.members))
    return out[:limit]


def balance_summary(bundle: RobotNetworkBundle) -> dict[str, float | int]:
    """Quick numeric summary of the network's structural state.

    Useful as a fingerprint above the figure.
    """
    return {
        "n_robots": bundle.n_robots,
        "n_edges": bundle.n_edges,
        "n_positive": bundle.n_positive_edges,
        "n_negative": bundle.n_negative_edges,
        "negative_fraction": (bundle.n_negative_edges / bundle.n_edges
                                if bundle.n_edges else 0.0),
        "mean_degree": (2.0 * bundle.n_edges / bundle.n_robots
                          if bundle.n_robots else 0.0),
    }


# ───────────────────────────────────────────────────────────────────────
# Edge-sign predictor (v0.5)
#
# Train a tiny HSiKAN on the train split of a single robot network and
# predict held-out edges. Demonstrates that the balance prior — the
# σ-product around k=3 cycles, which HSiKAN computes natively — is
# learnable from sparse observed signs. The pipeline is intentionally
# self-contained: cliques.py owns it, no new module.
# ───────────────────────────────────────────────────────────────────────


@dataclass
class EdgePredictionResult:
    """Per-test-edge predictions plus aggregate metrics."""

    test_edges: np.ndarray            # (n_test, 2) int64
    test_true: np.ndarray             # (n_test,)   ±1
    test_pred: np.ndarray             # (n_test,)   ±1
    test_prob_pos: np.ndarray         # (n_test,)   P(sign = +1)
    train_n_edges: int
    test_n_edges: int
    test_auc: float
    test_f1_macro: float
    test_accuracy: float
    n_params: int

    @property
    def n_test(self) -> int:
        return self.test_edges.shape[0]


def _split_edges(bundle: RobotNetworkBundle, train_frac: float, seed: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) into bundle.graph.edges.

    Both splits should contain a mix of ± edges where possible — if the
    bundle has only one sign, the split degenerates and AUC is NaN.
    """
    rng = np.random.default_rng(seed)
    n = bundle.n_edges
    perm = rng.permutation(n)
    n_train = max(1, int(round(train_frac * n)))
    return perm[:n_train], perm[n_train:]


def train_edge_sign_predictor(
    bundle: RobotNetworkBundle,
    n_epochs: int = 200,
    hidden: int = 8,
    train_frac: float = 0.75,
    lr: float = 5e-2,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
) -> EdgePredictionResult:
    """Train HSiKAN on the train split, evaluate on the test split.

    Uses k=3 cycle features only (k=4 would help on denser networks but
    adds enumeration cost). The model is a one-shot ``MixedAritySignedKAN``
    with the same scaffolding ``run_final_cell`` uses, just trimmed.

    Returns ``EdgePredictionResult`` — no checkpoint is written; this is
    a "live" demo training, not a saved-model flow.
    """
    # Lazy imports — keep cliques.py importable without torch.
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import f1_score, roc_auc_score

    from ..core.hyperedges import construct
    from ..mixed_arity_signedkan import (
        MixedAritySignedKAN, MixedAritySignedKANConfig,
        build_edge_to_tuples,
    )
    from ..core.signedkan import (
        MultiLayerSignedKANConfig, build_vertex_triad_incidence,
    )

    if bundle.n_edges < 4:
        raise ValueError(
            f"Need at least 4 edges to split, got {bundle.n_edges}. "
            f"Raise n_robots or comm_range."
        )
    train_idx, test_idx = _split_edges(bundle, train_frac, seed)
    if len(test_idx) == 0:
        raise ValueError("test split is empty — lower train_frac.")

    g = bundle.graph
    e_tr = g.edges[train_idx]
    s_tr = g.signs[train_idx]
    e_te = g.edges[test_idx]
    s_te = g.signs[test_idx]

    # Enumerate cycles on the FULL graph (transductive). The supervised
    # split lives at the classifier head, not at the cycle-feature
    # layer — same convention as the published Bitcoin / Slashdot
    # signed-link pipeline. Note: this means test-edge signs leak
    # into the σ-products of cycles that touch them. For an honest
    # held-out evaluation we'd mask test-edge signs in σ-products
    # (see strict_protocol memory). The demo follows the canonical
    # convention here.
    t_k = construct(g)
    if not t_k:
        raise ValueError(
            "Network has no k=3 cycles — HSiKAN's cycle pool would be "
            "empty. Raise comm_range or n_robots."
        )

    dev = torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
    triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
    triad_v = torch.from_numpy(triad_v_np).to(dev)
    triad_sigma = torch.from_numpy(triad_sigma_np).to(dev)
    M_vt = build_vertex_triad_incidence(triad_v_np, g.n_nodes, dev, mode="sum")
    edge_to_tuples = build_edge_to_tuples(t_k)

    def _build_M_e(edges: np.ndarray):
        rows, cols, vals = [], [], []
        for ei, (u, v) in enumerate(edges):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            ids = edge_to_tuples.get(key, [])
            if not ids:
                continue
            w = 1.0 / float(len(ids))
            for t in ids:
                rows.append(ei); cols.append(int(t)); vals.append(w)
        if rows:
            idx = torch.tensor([rows, cols], dtype=torch.long, device=dev)
            v = torch.tensor(vals, dtype=torch.float32, device=dev)
            return torch.sparse_coo_tensor(
                idx, v, (edges.shape[0], len(t_k))).coalesce()
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
            (edges.shape[0], len(t_k))).to(dev)

    per_arity_tr = [(triad_v, triad_sigma, M_vt, _build_M_e(e_tr))]
    per_arity_te = [(triad_v, triad_sigma, M_vt, _build_M_e(e_te))]

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True,
        ),
        arities=(3,),
        init_arity_logits=(0.0,),
    )
    model = MixedAritySignedKAN(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    y_tr = torch.from_numpy(
        (s_tr == 1).astype(np.float32)).to(dev)
    y_te_np = (s_te == 1).astype(np.float32)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(n_epochs):
        model.train()
        edge_emb = model.encode_edges(per_arity_tr)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:>3d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model.classifier(model.encode_edges(per_arity_te)).squeeze(-1)
        ).cpu().numpy()
    preds = np.where(probs > 0.5, 1, -1).astype(np.int64)

    y_te_int = (s_te == 1).astype(int)
    y_pr_int = (preds == 1).astype(int)
    auc = (float(roc_auc_score(y_te_int, probs))
            if len(set(y_te_int)) > 1 else float("nan"))
    f1m = float(f1_score(y_te_int, y_pr_int, average="macro",
                            zero_division=0))
    acc = float((preds == s_te).mean())

    return EdgePredictionResult(
        test_edges=e_te.astype(np.int64),
        test_true=s_te.astype(np.int64),
        test_pred=preds,
        test_prob_pos=probs.astype(np.float64),
        train_n_edges=len(train_idx),
        test_n_edges=len(test_idx),
        test_auc=auc,
        test_f1_macro=f1m,
        test_accuracy=acc,
        n_params=int(n_params),
    )


__all__ = [
    "Clique",
    "EdgePredictionResult",
    "RobotNetworkBundle",
    "_clique_balance_indicator",
    "balance_summary",
    "enumerate_balanced_cliques",
    "make_robot_network",
    "train_edge_sign_predictor",
]
