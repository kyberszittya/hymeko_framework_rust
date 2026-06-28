"""StructuralActor toy — walks/cycles straight to the output, no message-passing (Hajdu's idea).

HSiKAN propagates signals layer-by-layer (launch-bound: a Catmull-Rom spline per edge × L layers, ~2 ms). This
gathers raw node features along the topology's **precomputed walks** in one static index op, aggregates per walk,
and reads out — a handful of large ops instead of dozens of tiny ones. Same structural prior (a walk is a signed
transport path), a fraction of the cost. Design: ``docs/theory/structural_actor_design.md``; dual of
``structural_critic`` (which gathers *backbone* features for the value); pattern proven by ``spike_probe``.

The toy compares StructuralActor vs HSiKAN vs MLP on a fixed signed graph + a structural target, on **test MSE**
and **forward latency**. A self-contained pure-Python walk enumerator is used so the toy needs no Rust binding;
the production actor reuses ``structural_critic.enumerate_motifs`` (the ``enumerate_top_k_walks_rs`` binding).
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from signed_kan import CatmullRomActivation                     # the HSiKAN cell's learnable CR spline

from hymeko_rl.structural_probe import _standardised_split, build_chain_graph, build_model, train_eval


def enumerate_walks_py(hg, *, walk_len: int, keep: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """All signed walks of exactly ``walk_len`` edges (DFS over the fixed graph), padded to
    ``(vid (M,Lmax), sgn (M,Lmax), length (M,))`` — the same layout as ``enumerate_motifs``. Capped at ``keep``."""
    adj: dict[int, list[tuple[int, int]]] = {}
    for (u, v), s in zip(hg.edges.tolist(), hg.signs.tolist()):
        adj.setdefault(int(u), []).append((int(v), int(s)))
    walks: list[tuple[list[int], list[int]]] = []

    def dfs(verts: list[int], signs: list[int]) -> None:
        if len(verts) - 1 == walk_len:
            walks.append((verts, signs))
            return
        for nxt, sg in adj.get(verts[-1], []):
            dfs([*verts, nxt], [*signs, sg])

    for start in range(hg.n_vertices):
        dfs([start], [])
    walks = walks[:keep]
    lmax = walk_len + 1
    vid = np.zeros((len(walks), lmax), np.int64)
    sgn = np.zeros((len(walks), lmax), np.float32)
    length = np.full((len(walks),), float(lmax), np.float32)
    for i, (verts, signs) in enumerate(walks):
        vid[i, : len(verts)] = verts
        sgn[i, : len(signs)] = signs          # sign per edge; last position (the start) stays 0 → no self term
        sgn[i, len(signs)] = 1.0              # the terminal vertex carries +1 (its own feature enters)
    return vid, sgn, length


class StructuralActor(nn.Module):
    """Gather raw per-vertex features along the precomputed walks → per-node aggregate → readout. No
    message-passing.

    ``agg="holonomy"`` (the principled v2): each length-L walk ``v→…→e`` carries the **sign-product** (the Z₂
    holonomy) times the *endpoint* feature, scattered back to its **start node** ``v`` — reproducing the signed
    L-hop operator ``Bᴸx`` per node, which a signed *sum* along the walk cannot. ``agg="sum"`` is the (wrong) v1
    ablation. ``node_mlp`` adds a deeper per-node nonlinearity (the "more complex parameters" axis).

    # Preconditions ``in_feat >= 1``; walks are exactly ``walk_len`` edges (fixed at construction).
    # Postconditions ``forward(x: (B,N,in_feat)) -> (B,)`` (per-node value, then summed — matches the target)."""

    vid: torch.Tensor
    sgn: torch.Tensor
    length: torch.Tensor
    incidence: torch.Tensor    # (N, M) start-node one-hot (scatter walks back to their start vertex)
    bl: torch.Tensor           # (N, N) precomputed signed L-hop operator = Σ_walks sign_prod at [start, end]
    node_head: nn.Module
    node_out: nn.Module

    def __init__(self, hg, in_feat: int, hidden: int, *, walk_len: int = 2, keep: int = 64,
                 agg: str = "holonomy", node_mlp: bool = False, readout: str = "kan") -> None:
        super().__init__()
        if agg not in ("holonomy", "sum"):
            raise ValueError(f"agg must be 'holonomy' or 'sum'; got {agg!r}")
        if readout not in ("kan", "mlp"):
            raise ValueError(f"readout must be 'kan' (HSiKAN CR cell) or 'mlp' (tanh); got {readout!r}")
        self.agg_mode, self.walk_len = agg, walk_len
        vid, sgn, length = enumerate_walks_py(hg, walk_len=walk_len, keep=keep)
        n, m = int(hg.n_vertices), int(vid.shape[0])
        inc = np.zeros((n, m), np.float32)
        inc[vid[:, 0], np.arange(m)] = 1.0                           # start node of each walk
        # The holonomy path is LINEAR in x, so collapse gather+sign+scatter into one precomputed (N,N) operator:
        # Bᴸ[start, end] = Σ_{walk start→…→end} sign_product (the signed L-hop adjacency power the walks enumerate).
        bl = np.zeros((n, n), np.float32)
        np.add.at(bl, (vid[:, 0], vid[:, walk_len]), np.prod(sgn[:, :walk_len], axis=1))
        self.register_buffer("vid", torch.as_tensor(vid, dtype=torch.long))
        self.register_buffer("sgn", torch.as_tensor(sgn, dtype=torch.float32))
        self.register_buffer("length", torch.as_tensor(length, dtype=torch.float32).clamp_min(1.0))
        self.register_buffer("incidence", torch.as_tensor(inc))
        self.register_buffer("bl", torch.as_tensor(bl))
        # The per-node readout IS an HSiKAN cell: a signed-KAN nonlinearity (the learnable Catmull-Rom spline)
        # over the walk-holonomy features — so the StructuralActor is literally signed-gather → HSiKAN cell.
        def _act(c: int) -> nn.Module:
            return CatmullRomActivation(c) if readout == "kan" else nn.Tanh()
        layers: list[nn.Module] = [nn.Linear(in_feat, hidden), _act(hidden)]
        if node_mlp:
            layers += [nn.Linear(hidden, hidden), _act(hidden)]
        self.node_head = nn.Sequential(*layers)
        self.node_out = nn.Linear(hidden, 1)

    def per_node(self, x: torch.Tensor) -> torch.Tensor:
        """Per-vertex value ``(B,N)`` — walk-holonomy gather (``Bᴸx``) then the per-node readout. This is the
        **per-actuator control** in a closed loop (one value per joint/node)."""
        if self.agg_mode == "holonomy":
            node_agg = torch.einsum("ne,bef->bnf", self.bl, x)       # (B,N,F) = Bᴸx in ONE precomputed matmul
        else:                                                        # v1: signed sum along the walk (loses BᴸX)
            walk_feat = (x[:, self.vid, :] * self.sgn.unsqueeze(0).unsqueeze(-1)).sum(dim=2) \
                / self.length.view(1, -1, 1)
            node_agg = torch.einsum("nm,bmf->bnf", self.incidence, walk_feat)      # (B,N,F) scatter to start node
        out: torch.Tensor = self.node_out(self.node_head(node_agg)).squeeze(-1)
        return out

    def control(self, x: torch.Tensor) -> torch.Tensor:
        """Closed-loop control: per-node state ``x (B,N)`` → per-node action ``u (B,N)``."""
        return self.per_node(x.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.per_node(x).sum(dim=-1)             # pool (matches the summed structural target)
        return out


def _forward_latency_us(model: nn.Module, x: torch.Tensor, *, warmup: int = 30, iters: int = 300) -> float:
    """Median forward latency in microseconds (the launch-bound cost is the point)."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t) * 1e6)
    return statistics.median(ts)


def run_structural_actor_probe(*, n_nodes: int = 9, hidden: int = 24, seeds: int = 3, epochs: int = 250,
                               walk_len: int = 2, hg=None, keep: int = 8192) -> dict[str, object]:
    """The variants vs HSiKAN vs MLP on a fixed signed graph + the structural target — test MSE + forward
    latency. Isolates the holonomy fix (sign-product vs sign-sum) and the richer-params axis (per-node MLP).
    ``hg`` defaults to a chain; pass a ``hypergraph_designs`` Steiner/k-uniform/sunflower to run the actor over a
    real combinatorial design (its blocks become the actor's walk basis).
    # Postconditions one entry per model with median MSE and median forward µs."""
    if hg is None:
        hg = build_chain_graph(n_nodes, seed=0)
    n_nodes = int(hg.n_vertices)

    def make(name: str) -> nn.Module:
        if name == "struct_holonomy":
            return StructuralActor(hg, 1, hidden, walk_len=walk_len, keep=keep, agg="holonomy")
        if name == "struct_holo_mlp":
            return StructuralActor(hg, 1, hidden, walk_len=walk_len, keep=keep, agg="holonomy", node_mlp=True)
        if name == "struct_sum_v1":
            return StructuralActor(hg, 1, hidden, walk_len=walk_len, keep=keep, agg="sum")
        return build_model(name, hg, hidden, n_layers=2)

    names = ["struct_holonomy", "struct_holo_mlp", "struct_sum_v1", "hsikan", "mlp"]
    rows: dict[str, dict[str, float]] = {}
    for name in names:
        mses: list[float] = []
        for s in range(seeds):
            split = _standardised_split(hg, "structural", n_train=256, n_test=1024, seed=1000 + s)
            torch.manual_seed(s)
            mses.append(train_eval(make(name), split, epochs=epochs, seed=s))
        x = torch.randn(1, n_nodes, 1)
        ref = make(name)
        rows[name] = dict(mse_median=round(statistics.median(mses), 5),
                          fwd_us_median=round(_forward_latency_us(ref, x), 1),
                          n_params=int(sum(p.numel() for p in ref.parameters())))
    return dict(n_nodes=n_nodes, seeds=seeds, walk_len=walk_len, models=rows)


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--out-dir", default="reports/structural_actor")
    a = ap.parse_args(argv)
    report = run_structural_actor_probe(seeds=a.seeds, epochs=a.epochs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structural_actor.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
