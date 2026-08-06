"""The structural discriminating probe — the §0 decider, isolated from RL.

Does HSiKAN's signed message-passing beat a params-matched MLP *when the target is a graph property*? The
robot-task wiring audit (``reports/2026-06-26-hsikan-wiring-audit.md``) cleared every env, leaving two
options for "HSiKAN ties/loses to MLP": a silent backbone-forward bug, or *structure-not-load-bearing* (the
robot objectives aren't graph properties). The user's refinement: the latter means the **reward** should
carry structural content — a robot scenario *could* be won if its objective considered structure.

This probe settles it on a fixed signed graph with **two supervised targets**:

* ``structural`` — ``y = Σ_v tanh(α·(B² x)_v)`` with ``B = A⁺ − A⁻`` (the signed adjacency the HSiKAN
  backbone is built on): a signed two-hop nonlinear aggregate, the exact computation HSiKAN's conv performs.
* ``bag`` — ``y = Σ_v tanh(α·x_v)``: a per-node nonlinear sum, structure-independent (the control).

Params-matched HSiKAN vs MLP, K seeds, held-out test MSE. The decisive prediction:
``MSE_struct(HSiKAN) ≪ MSE_struct(MLP)`` while ``MSE_bag(HSiKAN) ≈ MSE_bag(MLP)``. Then: HSiKAN wins on
struct ⇒ the backbone forward is correct (bug hypothesis falsified) *and* structure helps when load-bearing;
the bag tie ⇒ the gap is structural, not capacity. If HSiKAN cannot beat the MLP even on ``structural``, the
backbone forward is buggy — localise in ``hymeko_neuro/core/backbone.py``.

Reuses the production backbones (no rebuild, §6.1): :class:`hymeko_neuro.core.SignedKANBackbone` and
:func:`hymeko_rl.agents.policy.mlp_backbone`.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.agents.policy import mlp_backbone
from hymeko_neuro.core import SignedKANBackbone

Target = Literal["structural", "bag", "local", "pernode"]
Backbone = Literal["hsikan", "mlp"]
Readout = Literal["pool", "mean", "sum", "max", "attention", "concat"]
# the readout bake-off set (over per-vertex activations) — "pool" is the backbone-native mean (the baseline);
# "attention" learns per-node weights (generalises mean-pool); "concat" keeps full node identity.
READOUT_MODES: tuple[str, ...] = ("mean", "sum", "max", "attention", "concat")
_ALPHA = 1.5            # target nonlinearity scale (tanh saturates gently over x ~ N(0,1) and B²x)
_LOCAL_NODE = 6         # the node the 'local' target reads (the signed-tail endpoint) — pooling cannot isolate it


def build_toy_graph() -> HypergraphState:
    """A fixed 7-vertex signed graph with two cycles and both signs present — so ``B = A⁺ − A⁻`` is a
    genuine signed operator (a frustrated triangle makes the sign load-bearing, not cosmetic).

    # Postconditions ``n_vertices == 7``; ``signs`` contains both ``+1`` and ``-1``; the graph has ``≥1``
      cycle (the triangle 0-1-2 plus the 1-4-3-2 loop)."""
    # undirected signed edges (a, b, sign); expanded to both directions for a symmetric signed adjacency.
    undirected = [(0, 1, +1), (1, 2, +1), (2, 0, -1),     # frustrated triangle (product of signs = −)
                  (2, 3, +1), (3, 4, -1), (1, 4, +1),     # second loop 1-4-3-2
                  (4, 5, +1), (5, 6, -1)]                  # a signed tail
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for a, b, s in undirected:
        edges += [(a, b), (b, a)]
        signs += [s, s]
    labels = tuple(f"n{i}" for i in range(7))
    return HypergraphState(vertex_labels=labels, edges=np.asarray(edges, np.int64),
                           signs=np.asarray(signs, np.int64), topo_hash="toy:structural-probe")


def build_chain_graph(n_nodes: int, *, seed: int = 0) -> HypergraphState:
    """A signed **chain** of ``n_nodes`` (a line ``0—1—…—(N−1)``) — the maximally sparse topology and the
    'worst case' for structure (cart-pole/arm are chains). Each undirected edge gets a random ±1 sign,
    expanded to both directions. Demonstrates that even a chain's kinematic structure is exploitable by
    sparse signed reasoning (vs a flat MLP with no locality prior).

    # Preconditions ``n_nodes >= 2``. # Postconditions ``2*(n_nodes-1)`` directed arcs; both signs likely present."""
    if n_nodes < 2:
        raise ValueError("a chain needs >= 2 nodes")
    rng = np.random.default_rng(seed)
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for i in range(n_nodes - 1):
        s = int(rng.choice([-1, 1]))
        edges += [(i, i + 1), (i + 1, i)]
        signs += [s, s]
    return HypergraphState(vertex_labels=tuple(f"n{i}" for i in range(n_nodes)),
                           edges=np.asarray(edges, np.int64), signs=np.asarray(signs, np.int64),
                           topo_hash=f"chain{n_nodes}:{seed}")


def run_chain_probe(*, lengths: "list[int] | None" = None, hsikan_hidden: int = 32, n_layers: int = 2,
                    n_train: int = 256, n_test: int = 1024, seeds: int = 3, epochs: int = 300,
                    ) -> dict[str, object]:
    """HSiKAN (sparse chain adjacency) vs params-matched MLP on the structural target, swept over chain length.

    The structural target reads the signed 2-hop neighbourhood (``B²x``); HSiKAN computes it with sparse local
    messages, the MLP must learn the ``N→target`` map flat. # Postconditions one row per length with both
    median MSEs + ratio; the HSiKAN advantage is expected to grow with chain length (no MLP locality prior)."""
    if lengths is None:
        lengths = [4, 6, 8, 12, 16]
    rows: list[dict[str, float]] = []
    for n in lengths:
        hg = build_chain_graph(n)
        mlp_hidden, hk_p, mlp_p = match_mlp_hidden(hg, hsikan_hidden)
        hidden_of: dict[Backbone, int] = {"hsikan": hsikan_hidden, "mlp": mlp_hidden}

        def med(kind: Backbone) -> float:
            out = []
            for s in range(seeds):
                split = _standardised_split(hg, "structural", n_train=n_train, n_test=n_test, seed=1000 + s)
                torch.manual_seed(s)
                model = build_model(kind, hg, hidden_of[kind], n_layers=n_layers)
                out.append(train_eval(model, split, epochs=epochs, seed=s))
            return round(statistics.median(out), 5)

        hk, mlp = med("hsikan"), med("mlp")
        rows.append(dict(n_nodes=n, hsikan_mse=hk, mlp_mse=mlp, ratio=round(mlp / max(hk, 1e-9), 3),
                         hk_params=hk_p, mlp_params=mlp_p))
    return dict(hsikan_hidden=hsikan_hidden, n_train=n_train, seeds=seeds, epochs=epochs, rows=rows)


def plot_chain_probe(report: dict[str, object], out_path: str | Path) -> Path:
    """Structural-target test MSE vs chain length — HSiKAN (sparse) vs MLP — and the advantage ratio (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    ns = [r["n_nodes"] for r in rows]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax.plot(ns, [r["hsikan_mse"] for r in rows], "-o", color="#2ca02c", label="HSiKAN (sparse chain)")
    ax.plot(ns, [r["mlp_mse"] for r in rows], "-o", color="#7f7f7f", label="MLP (flat)")
    ax.set_xlabel("chain length (nodes)")
    ax.set_ylabel("structural-target test MSE")
    ax.set_title("Even a chain is exploitable")
    ax.legend()
    ax2.plot(ns, [r["ratio"] for r in rows], "-o", color="#1f77b4")
    ax2.axhline(1.0, color="#bbbbbb", ls=":")
    ax2.set_xlabel("chain length (nodes)")
    ax2.set_ylabel("MLP / HSiKAN  MSE ratio  (>1 = HSiKAN better)")
    ax2.set_title("sparse-structure advantage vs length")
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def make_dataset(hg: HypergraphState, n: int, kind: Target, *, seed: int,
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``(X (n,N,1), y (n,))``. ``structural`` reads the signed adjacency (``B²x``); ``bag`` does
    not. Deterministic in ``seed``; ``y`` is left raw here (standardised per-split by the caller).

    # Preconditions ``n >= 1``; ``kind in {'structural','bag'}``.
    # Postconditions shapes ``(n,N,1)`` / ``(n,)``; ``structural`` ``y`` changes if a sign flips."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    nodes = hg.n_vertices
    x = rng.standard_normal((n, nodes, 1)).astype(np.float64)
    if kind == "bag":
        y = np.tanh(_ALPHA * x).sum(axis=(1, 2))                  # per-node separable sum (no structure)
    elif kind in ("structural", "local", "pernode"):
        a_pos, a_neg = hg.dense_signed_adj()
        b = (a_pos - a_neg).numpy().astype(np.float64)            # the signed operator B = A⁺ − A⁻
        m2 = np.einsum("ij,njk->nik", b, np.einsum("ij,njk->nik", b, x))   # 2-hop signed message B²x
        if kind == "structural":
            y = np.tanh(_ALPHA * m2).sum(axis=(1, 2))            # (n,)   pooled over nodes (mean-pool friendly)
        elif kind == "pernode":
            y = np.tanh(_ALPHA * m2[:, :, 0])                    # (n,N)  EVERY node's value — a vector target
        else:                                                    # 'local': one node's value — pooling can't isolate it
            if not 0 <= _LOCAL_NODE < nodes:
                raise ValueError(f"_LOCAL_NODE {_LOCAL_NODE} out of range for {nodes} vertices")
            y = np.tanh(_ALPHA * m2[:, _LOCAL_NODE, 0])          # (n,)
    else:
        raise ValueError(f"unknown target kind {kind!r}")
    return (torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32))


class ProbeModel(nn.Module):
    """A backbone (HSiKAN or MLP) + a linear readout to a scalar — the regressor under test.

    # Invariants ``forward(X (B,N,1)) -> (B,)``."""

    def __init__(self, backbone: nn.Module, hidden: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.readout(self.backbone(x)).squeeze(-1)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ReadoutBackbone(nn.Module):
    """Apply a chosen READOUT to a signed-conv backbone's per-vertex activations ``(B,N,H)``. The backbone is
    identical across modes; only the readout differs, so a gap isolates the readout effect.

    Modes: ``mean``/``sum``/``max`` (permutation-invariant collapses — ``max`` keeps the extreme, not identity);
    ``attention`` (learned per-node softmax weights → ``Σ_v α_v h_v``; **generalises mean-pool** — uniform
    ``α`` recovers it, focused ``α`` recovers a specific node); ``concat`` (flatten ``→ (B,N·H)``, full identity).
    The attention scorer mirrors :meth:`hymeko_rl.agents.structural_critic.StructuralCritic._pool` (§6.1 — same pattern)."""

    def __init__(self, backbone: SignedKANBackbone, mode: str, hidden: int, n_vertices: int) -> None:
        super().__init__()
        if mode not in READOUT_MODES:
            raise ValueError(f"readout mode must be in {READOUT_MODES}; got {mode!r}")
        self.backbone = backbone
        self.mode = mode
        self.out_dim = n_vertices * hidden if mode == "concat" else hidden
        self.att = nn.Linear(hidden, 1) if mode == "attention" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.node_activations(x)                    # (B, N, hidden) — before any pooling
        if self.mode == "mean":
            return h.mean(dim=1)
        if self.mode == "sum":
            return h.sum(dim=1)
        if self.mode == "max":
            return h.max(dim=1).values
        if self.mode == "concat":
            return h.reshape(h.shape[0], -1)
        assert self.att is not None                              # mode == "attention" (the only remaining)
        alpha = torch.softmax(self.att(h).squeeze(-1), dim=1)    # (B, N) learned per-node weights
        return (alpha.unsqueeze(-1) * h).sum(dim=1)              # Σ_v α_v h_v → (B, hidden)


def build_model(kind: Backbone, hg: HypergraphState, hidden: int, *, n_layers: int = 2,
                readout: Readout = "pool") -> ProbeModel:
    """Construct a :class:`ProbeModel` of ``kind`` on the toy graph (in_feat=1). HSiKAN gets the signed
    adjacency as its fixed prior; the MLP flattens the per-node features (the same input, structure-blind).

    ``readout`` (HSiKAN only): ``"pool"`` = the backbone-native mean-pool (the baseline); or one of
    :data:`READOUT_MODES` (``mean``/``sum``/``max``/``attention``/``concat``) — the bake-off axis. ``"pool"``
    and ``"mean"`` are numerically identical (both average the per-vertex activations)."""
    if kind == "hsikan":
        a_pos, a_neg = hg.dense_signed_adj()
        bb = SignedKANBackbone(1, hidden, n_layers, a_pos, a_neg)
        if readout == "pool":
            return ProbeModel(bb, hidden)                        # backbone-native mean-pool (the baseline)
        wrap = _ReadoutBackbone(bb, readout, hidden, hg.n_vertices)
        return ProbeModel(wrap, wrap.out_dim)
    if kind == "mlp":
        backbone, hidden = mlp_backbone(hg.n_vertices * 1, hidden=hidden, depth=n_layers)
        return ProbeModel(backbone, hidden)                      # the MLP is flat; readout is N/A
    raise ValueError(f"unknown backbone {kind!r}")


# ── multidimensional (per-node) readouts — the "we have multiple joints anyway" line ────────────────────
# The robot action is per-joint and the graph is per-joint, so a vector output y ∈ R^N is the natural shape.
# These four heads predict it; the contrast is whether the readout COLLAPSES the per-node structure or keeps it.
class _PoolExpand(nn.Module):
    """Collapse-then-expand: mean-pool ``(B,N,H)→(B,H)`` then a Linear ``(B,H)→(B,N)``. Must reconstruct every
    node's value from one collapsed vector — the shape mismatch the per-node heads avoid (the bad baseline)."""

    def __init__(self, backbone: SignedKANBackbone, hidden: int, n_out: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.backbone(x))          # backbone mean-pools → (B,H) → (B,N)
        return out


class _PerNode(nn.Module):
    """Per-node multidimensional output: a *shared* head maps each node's (message-passed) activation to that
    node's value — ``(B,N,H) → (B,N)``. Keeps the node↔output correspondence; no collapse."""

    def __init__(self, backbone: SignedKANBackbone, hidden: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.node_activations(x)                    # (B,N,H)
        return self.head(h).squeeze(-1)                          # (B,N)


class _PerNodeGlobal(nn.Module):
    """Per-node output WITH global pooling as broadcast context: each node reads ``[h_node ; mean-pool(h)]``
    → its value. This is "global pooling AND multidimensional output" — local identity + shared context."""

    def __init__(self, backbone: SignedKANBackbone, hidden: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone.node_activations(x)                    # (B,N,H)
        g = h.mean(dim=1, keepdim=True).expand(-1, h.shape[1], -1)   # global context, broadcast to nodes
        return self.head(torch.cat([h, g], dim=-1)).squeeze(-1)  # (B,N)


class _MlpMultiDim(nn.Module):
    """Flat MLP baseline for the vector target: ``(B,N,1)`` flattened → hidden → ``Linear(hidden, N)``."""

    def __init__(self, n_in: int, hidden: int, n_out: int, depth: int) -> None:
        super().__init__()
        backbone, h = mlp_backbone(n_in, hidden=hidden, depth=depth)
        self.backbone = backbone
        self.head = nn.Linear(h, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.backbone(x))
        return out


_MULTIDIM_CONFIGS = ("pool_expand", "per_node", "per_node_global", "mlp")


def build_multidim_model(config: str, hg: HypergraphState, hidden: int, *, n_layers: int = 2,
                         mlp_hidden: int = 64) -> nn.Module:
    """One of the four vector-output heads (:data:`_MULTIDIM_CONFIGS`) on the toy graph (in_feat=1)."""
    n = hg.n_vertices
    if config == "mlp":
        return _MlpMultiDim(n, mlp_hidden, n, n_layers)
    a_pos, a_neg = hg.dense_signed_adj()
    bb = SignedKANBackbone(1, hidden, n_layers, a_pos, a_neg)
    if config == "pool_expand":
        return _PoolExpand(bb, hidden, n)
    if config == "per_node":
        return _PerNode(bb, hidden)
    if config == "per_node_global":
        return _PerNodeGlobal(bb, hidden)
    raise ValueError(f"config must be in {_MULTIDIM_CONFIGS}; got {config!r}")


def run_multidim_probe(*, hsikan_hidden: int = 32, n_layers: int = 2, n_train: int = 512, n_test: int = 1024,
                       seeds: int = 5, epochs: int = 300, mlp_hidden: int = 64) -> dict[str, object]:
    """Compare the four vector-output heads on the per-node target ``y ∈ R^N`` — does keeping the multidimensional
    per-node structure beat collapse-then-expand? Reports test MSE (median/IQR) + params per config.

    # Postconditions one row per config; ``ratio_poolexpand_over_pernode`` quantifies the collapse cost."""
    hg = build_toy_graph()
    rows: list[dict[str, object]] = []
    med: dict[str, float] = {}
    for config in _MULTIDIM_CONFIGS:
        mses: list[float] = []
        params = 0
        for s in range(seeds):
            split = _standardised_split(hg, "pernode", n_train=n_train, n_test=n_test, seed=1000 + s)
            torch.manual_seed(s)
            model = build_multidim_model(config, hg, hsikan_hidden, n_layers=n_layers, mlp_hidden=mlp_hidden)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mses.append(train_eval(model, split, epochs=epochs, seed=s))  # type: ignore[arg-type]
        ms = sorted(mses)
        q1, q3 = np.percentile(ms, [25, 75])
        med[config] = round(statistics.median(mses), 5)
        rows.append(dict(config=config, n_params=params, mse_median=med[config],
                         mse_iqr=round(float(q3 - q1), 5), per_seed=tuple(round(m, 5) for m in ms)))
    ratio = round(med["pool_expand"] / max(med["per_node"], 1e-9), 3)
    return dict(hsikan_hidden=hsikan_hidden, mlp_hidden=mlp_hidden, n_train=n_train, seeds=seeds, epochs=epochs,
                ratio_poolexpand_over_pernode=ratio,
                winner=min(med, key=lambda k: med[k]), rows=rows)


def plot_multidim(report: dict[str, object], out_path: str | Path) -> Path:
    """Bar chart of vector-target test MSE per readout config (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    cfgs = [r["config"] for r in rows]
    meds = [r["mse_median"] for r in rows]
    colors = ["#d62728" if c == "pool_expand" else "#2ca02c" if c.startswith("per_node") else "#7f7f7f"
              for c in cfgs]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(range(len(cfgs)), meds, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=12)
    ax.set_ylabel("vector-target test MSE (lower = better)")
    ax.set_title(f"Per-node (multidim) output vs collapse-then-expand "
                 f"(pool_expand/per_node = {report['ratio_poolexpand_over_pernode']}×)")
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def match_mlp_hidden(hg: HypergraphState, hsikan_hidden: int, *, search: range = range(8, 257),
                     ) -> tuple[int, int, int]:
    """Pick the MLP width whose param count is closest to HSiKAN@``hsikan_hidden``. Returns
    ``(mlp_hidden, hsikan_params, mlp_params)``."""
    hk = build_model("hsikan", hg, hsikan_hidden).n_params()
    best = min(search, key=lambda h: abs(build_model("mlp", hg, h).n_params() - hk))
    return best, hk, build_model("mlp", hg, best).n_params()


@dataclass(frozen=True)
class _Split:
    x_tr: torch.Tensor
    y_tr: torch.Tensor
    x_te: torch.Tensor
    y_te: torch.Tensor


def _standardised_split(hg: HypergraphState, target: Target, *, n_train: int, n_test: int,
                        seed: int) -> _Split:
    """Train/test tensors with ``y`` standardised by the **train** mean/std (so test MSE is a clean,
    comparable generalisation number across targets)."""
    x_tr, y_tr = make_dataset(hg, n_train, target, seed=seed)
    x_te, y_te = make_dataset(hg, n_test, target, seed=seed + 9973)     # disjoint stream
    mu, sd = float(y_tr.mean()), float(y_tr.std()) + 1e-8
    return _Split(x_tr, (y_tr - mu) / sd, x_te, (y_te - mu) / sd)


def train_eval(model: ProbeModel, split: _Split, *, epochs: int = 300, batch: int = 64, lr: float = 3e-3,
               seed: int = 0) -> float:
    """Train ``model`` on ``split`` (Adam, MSE, minibatch) and return the held-out **test MSE**.

    # Postconditions returns a finite, non-negative float; deterministic in ``seed``."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = split.x_tr.shape[0]
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(split.x_tr[idx]), split.y_tr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        mse = float(nn.functional.mse_loss(model(split.x_te), split.y_te))
    return mse


@dataclass(frozen=True)
class ProbeResult:
    target: Target
    backbone: Backbone
    n_params: int
    mse_median: float
    mse_iqr: float
    per_seed: tuple[float, ...]


def run_probe(*, hsikan_hidden: int = 32, n_layers: int = 2, n_train: int = 256, n_test: int = 1024,
              seeds: int = 5, epochs: int = 300) -> dict[str, object]:
    """The full params-matched K-seed A/B over the two targets. Returns a report dict (numbers + the
    structural/bag advantage ratios MSE_MLP / MSE_HSiKAN)."""
    hg = build_toy_graph()
    mlp_hidden, hk_params, mlp_params = match_mlp_hidden(hg, hsikan_hidden)
    results: list[ProbeResult] = []
    hidden_of: dict[Backbone, int] = {"hsikan": hsikan_hidden, "mlp": mlp_hidden}
    for target in ("structural", "bag"):
        for kind in ("hsikan", "mlp"):
            mses: list[float] = []
            for s in range(seeds):
                split = _standardised_split(hg, target, n_train=n_train, n_test=n_test, seed=1000 + s)
                torch.manual_seed(s)        # seed BEFORE build so weight init is reproducible regardless of
                model = build_model(kind, hg, hidden_of[kind], n_layers=n_layers)   # prior call history
                mses.append(train_eval(model, split, epochs=epochs, seed=s))
            mses_sorted = sorted(mses)
            q1, q3 = np.percentile(mses_sorted, [25, 75])
            results.append(ProbeResult(target, kind, model.n_params(), round(statistics.median(mses), 5),
                                       round(float(q3 - q1), 5), tuple(round(m, 5) for m in mses_sorted)))

    def med(t: Target, k: Backbone) -> float:
        return next(r.mse_median for r in results if r.target == t and r.backbone == k)

    ratios = {t: round(med(t, "mlp") / max(med(t, "hsikan"), 1e-9), 3) for t in ("structural", "bag")}
    return dict(
        hsikan_hidden=hsikan_hidden, mlp_hidden=mlp_hidden, hk_params=hk_params, mlp_params=mlp_params,
        n_train=n_train, n_test=n_test, seeds=seeds, epochs=epochs,
        advantage_ratio_mlp_over_hsikan=ratios,
        verdict=("backbone OK + structure load-bearing" if ratios["structural"] > 1.5
                 and 0.5 < ratios["bag"] < 2.0 else "INVESTIGATE (see report)"),
        results=[r.__dict__ for r in results])


def run_readout_ablation(*, hsikan_hidden: int = 32, n_layers: int = 2, n_train: int = 512,
                         n_test: int = 1024, seeds: int = 5, epochs: int = 300) -> dict[str, object]:
    """The readout ablation: HSiKAN·mean-pool vs HSiKAN·concat vs MLP, across the three targets, on the
    fixed graph. Tests whether the mean-pool readout (not the backbone) is the liability — concat keeps node
    identity, mean-pool collapses it, so the ``local`` (node-specific) target separates them.

    # Postconditions one row per (target, config) with median/IQR test MSE + params; the ``local`` row
      carries the verdict (concat ≫ mean-pool ⇒ pooling is the control bottleneck)."""
    hg = build_toy_graph()
    mlp_hidden, _hk, _mk = match_mlp_hidden(hg, hsikan_hidden)
    configs: list[tuple[Backbone, Readout]] = [("hsikan", "pool"), ("hsikan", "concat"), ("mlp", "pool")]
    hidden_of: dict[Backbone, int] = {"hsikan": hsikan_hidden, "mlp": mlp_hidden}
    rows: list[dict[str, object]] = []
    for target in ("structural", "bag", "local"):
        for kind, readout in configs:
            mses: list[float] = []
            params = 0
            for s in range(seeds):
                split = _standardised_split(hg, target, n_train=n_train, n_test=n_test, seed=1000 + s)
                torch.manual_seed(s)                              # reproducible init (cf run_probe)
                model = build_model(kind, hg, hidden_of[kind], n_layers=n_layers, readout=readout)
                params = model.n_params()
                mses.append(train_eval(model, split, epochs=epochs, seed=s))
            mses_sorted = sorted(mses)
            q1, q3 = np.percentile(mses_sorted, [25, 75])
            rows.append(dict(target=target, config=(f"hsikan_{readout}" if kind == "hsikan" else "mlp"),
                             n_params=params, mse_median=round(statistics.median(mses), 5),
                             mse_iqr=round(float(q3 - q1), 5), per_seed=tuple(round(m, 5) for m in mses_sorted)))

    def med(t: Target, c: str) -> float:
        return next(r["mse_median"] for r in rows if r["target"] == t and r["config"] == c)  # type: ignore[return-value]

    local_pool_vs_concat = round(med("local", "hsikan_pool") / max(med("local", "hsikan_concat"), 1e-9), 3)
    return dict(hsikan_hidden=hsikan_hidden, mlp_hidden=mlp_hidden, n_train=n_train, seeds=seeds,
                epochs=epochs, local_pool_over_concat=local_pool_vs_concat,
                verdict=("mean-pool IS the liability (concat ≫ pool on local)" if local_pool_vs_concat > 1.5
                         else "pooling NOT the liability on local — look elsewhere"),
                rows=rows)


def run_readout_bakeoff(*, hsikan_hidden: int = 32, n_layers: int = 2, n_train: int = 512,
                        n_test: int = 1024, seeds: int = 5, epochs: int = 300) -> dict[str, object]:
    """Rank candidate HSiKAN readouts (mean/sum/max/attention/concat) + an MLP reference across the three
    targets. "Better than mean-pool" = a readout good on ALL targets — especially not failing the
    node-specific ``local`` one. Ranked by **worst-case** (max) test MSE over targets (the min-max readout).

    # Postconditions ``rows`` has one entry per (readout|mlp, target); ``ranking`` orders readouts by
      worst-case MSE ascending (the winner first)."""
    hg = build_toy_graph()
    mlp_hidden, _hk, _mk = match_mlp_hidden(hg, hsikan_hidden)
    targets: tuple[Target, ...] = ("structural", "bag", "local")

    def cell(kind: Backbone, readout: Readout, target: Target) -> tuple[float, int]:
        mses: list[float] = []
        params = 0
        for s in range(seeds):
            split = _standardised_split(hg, target, n_train=n_train, n_test=n_test, seed=1000 + s)
            torch.manual_seed(s)
            hidden = hsikan_hidden if kind == "hsikan" else mlp_hidden
            model = build_model(kind, hg, hidden, n_layers=n_layers, readout=readout)
            params = model.n_params()
            mses.append(train_eval(model, split, epochs=epochs, seed=s))
        return round(statistics.median(mses), 5), params

    rows: list[dict[str, object]] = []
    worst: dict[str, float] = {}
    for ro in READOUT_MODES:
        per_target = {}
        for t in targets:
            mse, params = cell("hsikan", ro, t)  # type: ignore[arg-type]
            rows.append(dict(readout=ro, target=t, mse_median=mse, n_params=params))
            per_target[t] = mse
        worst[ro] = round(max(per_target.values()), 5)
    for t in targets:                                          # MLP reference (flat; readout N/A)
        mse, params = cell("mlp", "pool", t)
        rows.append(dict(readout="mlp", target=t, mse_median=mse, n_params=params))
    worst["mlp"] = round(max(r["mse_median"] for r in rows if r["readout"] == "mlp"), 5)  # type: ignore[type-var]
    ranking = sorted(worst, key=lambda k: worst[k])
    return dict(hsikan_hidden=hsikan_hidden, mlp_hidden=mlp_hidden, n_train=n_train, seeds=seeds,
                epochs=epochs, worst_case_mse=worst, ranking=ranking,
                winner=ranking[0], mean_pool_worst=worst.get("mean"), rows=rows)


def plot_bakeoff(report: dict[str, object], out_path: str | Path) -> Path:
    """Grouped-bar test-MSE (readout × target) — the readout bake-off figure (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    readouts = list(report["ranking"])  # type: ignore[arg-type]
    targets = ["structural", "bag", "local"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.8 / len(readouts)
    for i, ro in enumerate(readouts):
        meds = [next((r["mse_median"] for r in rows if r["readout"] == ro and r["target"] == t), 0.0)
                for t in targets]
        ax.bar(np.arange(len(targets)) + (i - (len(readouts) - 1) / 2) * width, meds, width,
               label=ro, edgecolor="black", alpha=0.85)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_ylabel("held-out test MSE (lower = better)")
    ax.set_title(f"Readout bake-off — winner (min worst-case): {report['winner']}")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_ablation(report: dict[str, object], out_path: str | Path) -> Path:
    """Grouped-bar test-MSE (target × config) — the readout-ablation figure (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    targets = ["structural", "bag", "local"]
    configs = ["hsikan_pool", "hsikan_concat", "mlp"]
    colors = {"hsikan_pool": "#d62728", "hsikan_concat": "#2ca02c", "mlp": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(8, 4.4))
    width = 0.27
    for i, cfg in enumerate(configs):
        meds = [next(r["mse_median"] for r in rows if r["target"] == t and r["config"] == cfg)
                for t in targets]
        ax.bar(np.arange(len(targets)) + (i - 1) * width, meds, width, label=cfg, color=colors[cfg],
               edgecolor="black", alpha=0.85)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_ylabel("held-out test MSE (lower = better)")
    ax.set_title(f"Readout ablation — local pool/concat = {report['local_pool_over_concat']}×")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def sweep_n_train(sizes: "list[int]", *, hsikan_hidden: int = 32, seeds: int = 5, epochs: int = 300,
                  n_test: int = 1024) -> dict[str, object]:
    """Run :func:`run_probe` across training-set sizes — the data-scaling curve that distinguishes a
    sample-efficiency advantage (gap shrinks with data) from a representational one (gap grows with data).

    # Postconditions one row per size with the four (target, backbone) median MSEs + both ratios."""
    rows: list[dict[str, object]] = []
    for n in sizes:
        rep = run_probe(hsikan_hidden=hsikan_hidden, n_train=n, n_test=n_test, seeds=seeds, epochs=epochs)
        med = {(r["target"], r["backbone"]): r["mse_median"] for r in rep["results"]}  # type: ignore[index]
        rows.append(dict(
            n_train=n,
            struct_hsikan=med[("structural", "hsikan")], struct_mlp=med[("structural", "mlp")],
            bag_hsikan=med[("bag", "hsikan")], bag_mlp=med[("bag", "mlp")],
            ratios=rep["advantage_ratio_mlp_over_hsikan"]))
    return dict(sizes=list(sizes), hsikan_hidden=hsikan_hidden, seeds=seeds, epochs=epochs, rows=rows)


def plot_sweep(sweep: dict[str, object], out_path: str | Path) -> Path:
    """Log-log test-MSE vs n_train for the four (target, backbone) curves — the §9 data-scaling figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sweep["rows"]  # type: ignore[index]
    ns = [r["n_train"] for r in rows]
    series = {"struct_hsikan": ("HSiKAN · structural", "-o", "#1f77b4"),
              "struct_mlp": ("MLP · structural", "--o", "#1f77b4"),
              "bag_hsikan": ("HSiKAN · bag", "-s", "#d62728"),
              "bag_mlp": ("MLP · bag", "--s", "#d62728")}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, (label, style, color) in series.items():
        ax.plot(ns, [r[key] for r in rows], style, label=label, color=color,
                alpha=0.6 if "mlp" in key else 1.0, mfc="none" if "mlp" in key else color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training samples")
    ax.set_ylabel("held-out test MSE (lower = better)")
    ax.set_title("Structural probe — data scaling (HSiKAN solid, MLP dashed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_probe(report: dict[str, object], out_path: str | Path) -> Path:
    """Grouped-bar test-MSE (target × backbone) — the §9 plotted output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["results"]
    targets = ["structural", "bag"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.35
    for i, kind in enumerate(("hsikan", "mlp")):
        meds = [next(r["mse_median"] for r in rows if r["target"] == t and r["backbone"] == kind)
                for t in targets]
        ax.bar(np.arange(len(targets)) + (i - 0.5) * width, meds, width, label=kind,
               edgecolor="black", alpha=0.85)
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([f"{t}\n(MLP/HSiKAN={report['advantage_ratio_mlp_over_hsikan'][t]}×)"  # type: ignore[index]
                        for t in targets])
    ax.set_ylabel("held-out test MSE (lower = better)")
    ax.set_title("Structural probe — HSiKAN vs params-matched MLP")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hidden", type=int, default=32, help="HSiKAN width (MLP params-matched)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out-dir", default="reports/structural_probe")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated n_train sizes → the data-scaling curve (e.g. 16,32,64,128,256,512)")
    ap.add_argument("--ablation", action="store_true",
                    help="run the readout ablation (HSiKAN pool vs concat vs MLP × 3 targets incl. 'local')")
    ap.add_argument("--bakeoff", action="store_true",
                    help="rank candidate readouts (mean/sum/max/attention/concat) + MLP by worst-case across targets")
    ap.add_argument("--multidim", action="store_true",
                    help="vector (per-node) target: collapse-then-expand vs per-node vs per-node+global vs MLP")
    ap.add_argument("--chain", action="store_true",
                    help="sparse-reasoning on a CHAIN: HSiKAN vs MLP on the structural target, swept over length")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.chain:
        report = run_chain_probe(hsikan_hidden=a.hidden, seeds=a.seeds, epochs=a.epochs)
        (out / "chain_probe.json").write_text(json.dumps(report, indent=2))
        plot_chain_probe(report, out / "chain_probe")
        print(json.dumps(report, indent=2))
        return 0
    if a.multidim:
        report = run_multidim_probe(hsikan_hidden=a.hidden, n_train=a.n_train, seeds=a.seeds, epochs=a.epochs)
        (out / "multidim_probe.json").write_text(json.dumps(report, indent=2))
        plot_multidim(report, out / "multidim_probe")
        print(json.dumps(report, indent=2))
        return 0
    if a.bakeoff:
        report = run_readout_bakeoff(hsikan_hidden=a.hidden, n_train=a.n_train, seeds=a.seeds, epochs=a.epochs)
        (out / "readout_bakeoff.json").write_text(json.dumps(report, indent=2))
        plot_bakeoff(report, out / "readout_bakeoff")
        print(json.dumps(report, indent=2))
        return 0
    if a.ablation:
        report = run_readout_ablation(hsikan_hidden=a.hidden, n_train=a.n_train, seeds=a.seeds, epochs=a.epochs)
        (out / "readout_ablation.json").write_text(json.dumps(report, indent=2))
        plot_ablation(report, out / "readout_ablation")
        print(json.dumps(report, indent=2))
        return 0
    if a.sweep:
        sizes = [int(s) for s in a.sweep.split(",")]
        sweep = sweep_n_train(sizes, hsikan_hidden=a.hidden, seeds=a.seeds, epochs=a.epochs)
        (out / "structural_probe_sweep.json").write_text(json.dumps(sweep, indent=2))
        plot_sweep(sweep, out / "structural_probe_sweep")
        print(json.dumps(sweep, indent=2))
        return 0
    report = run_probe(hsikan_hidden=a.hidden, n_train=a.n_train, seeds=a.seeds, epochs=a.epochs)
    (out / "structural_probe.json").write_text(json.dumps(report, indent=2))
    plot_probe(report, out / "structural_probe")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
