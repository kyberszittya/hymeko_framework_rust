"""The spiral-highway toy — is the highway a structure-FREE spiral, and does the spiral carry the holonomy?

User insight (2026-06-26/27): the HSiKAN highway gate ``out = T·H + (1−T)·carry`` is the structure-free
skeleton of the gauge **spiral** ``ŷ = ⊕_W α_W·Hol(W)·h``. The highway's *carry* transports the feature with
**identity** (no rotation, no holonomy); the spiral replaces that carry with **rotor parallel-transport along
walks** and the gate ``T`` with the walk-collection weights ``α_W``. So a plain highway adds capacity but no
structural signal — which is why ``skip="highway"`` was a null on galambos.

This toy tests the conjecture on a **walk-holonomy collection** target. A θ-graph has ``K`` parallel walks
source→target; per sample the connection ``θ ∈ R^E`` (edge angles) varies and a source vector ``x ∈ R²`` is
**collected over the walks**: ``y = mean_k R(Σ_{e∈W_k} θ_e)·x`` (each walk transports ``x`` by its holonomy,
then we average — the spiral). Three models predict ``y`` from ``(θ, x)``:

* **spiral** — knows the walk structure (which edges form which walk), composes per-edge rotors along each walk
  and α-collects: ``ŷ = Σ_k softmax(α)_k R((g⊙θ)·W_k)·x``. Learnable: a per-edge gain ``g`` and the weights
  ``α`` (a handful of params). The right inductive bias.
* **highway_mlp** — a gated flat net (``T·H(in) + (1−T)·carry(in)``): the *plain highway*, no walk transport.
* **mlp** — a flat net. The structure-blind baseline.

Prediction (the spiral signature): the spiral fits the walk-holonomy with ~constant error as ``K`` grows and a
handful of params; the flat/highway nets must learn the ``(E+2)→2`` walk-rotation map and degrade with ``K``.
Reuses :func:`hymeko_rl.rotor_probe.rot_matrix` (SO(2)) — no rebuild. See
``docs/plans/2026-06-26-rotor-spikes-ablation/`` (this is its designed spikes/spiral next-toy).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn

_SpiralConfig = str  # "spiral" | "highway_mlp" | "mlp"


def theta_graph_walks(k_paths: int) -> torch.Tensor:
    """A θ-graph's walk-incidence ``(K, E=2K)``: ``K`` parallel source→target walks, each a 2-edge path
    (``s→mid_k→t``). Row ``k`` marks the two edges of walk ``k``. # Preconditions ``k_paths >= 1``."""
    if k_paths < 1:
        raise ValueError("need >= 1 walk")
    inc = torch.zeros(k_paths, 2 * k_paths)
    for k in range(k_paths):
        inc[k, 2 * k] = 1.0
        inc[k, 2 * k + 1] = 1.0
    return inc


def make_spiral_data(walk_inc: torch.Tensor, n: int, *, seed: int,
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(theta (n,E), x (n,2), y (n,2))`` with ``y = mean_k R((walk_inc·theta)_k)·x`` — the source vector
    collected over the walks' holonomies. The connection ``theta`` varies per sample (so the structure must
    be *used*, not memorised). # Postconditions ``y`` is the mean of ``x`` rotated by each walk's holonomy."""
    k, e = walk_inc.shape
    g = torch.Generator().manual_seed(seed)
    theta = (torch.rand(n, e, generator=g) * 2.0 - 1.0) * math.pi      # U(-π, π)^E
    x = torch.randn(n, 2, generator=g)
    phi = theta @ walk_inc.T                                           # (n, K) per-walk holonomy
    c, s = torch.cos(phi), torch.sin(phi)                             # (n, K)
    rx = torch.stack([c * x[:, 0:1] - s * x[:, 1:2], s * x[:, 0:1] + c * x[:, 1:2]], dim=-1)  # (n,K,2)
    return theta, x, rx.mean(dim=1)                                   # collect (mean) over walks → (n,2)


class SpiralModel(nn.Module):
    """The spiral: compose per-edge rotors along the KNOWN walks, α-collect the transported fibers. The
    learnable part is a per-edge gain ``g`` and the walk weights ``α`` — a handful of params with the right
    structure."""

    walk_inc: torch.Tensor

    def __init__(self, walk_inc: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("walk_inc", walk_inc)
        self.g = nn.Parameter(torch.ones(walk_inc.shape[1]))
        self.alpha = nn.Parameter(torch.zeros(walk_inc.shape[0]))

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        phi = (theta * self.g) @ self.walk_inc.T                      # (B, K)
        a = torch.softmax(self.alpha, dim=0)                          # (K,)
        c, s = torch.cos(phi), torch.sin(phi)
        rx = torch.stack([c * x[:, 0:1] - s * x[:, 1:2], s * x[:, 0:1] + c * x[:, 1:2]], dim=-1)  # (B,K,2)
        out: torch.Tensor = (a.view(1, -1, 1) * rx).sum(dim=1)        # α-collected transport → (B,2)
        return out


class FlatMlp(nn.Module):
    """Structure-blind baseline: ``concat(θ, x) → MLP → ŷ``."""

    def __init__(self, e: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(e + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(torch.cat([theta, x], dim=-1))
        return out


class HighwayMlp(nn.Module):
    """The *plain highway*: ``T·H(in) + (1−T)·carry(in)`` — a gated flat net whose carry is an identity-class
    projection of the input (no walk transport, no holonomy). The structure-free spiral skeleton."""

    def __init__(self, e: int, hidden: int = 64) -> None:
        super().__init__()
        self.h = nn.Sequential(nn.Linear(e + 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.gate = nn.Linear(e + 2, 2)
        nn.init.constant_(self.gate.bias, -2.0)                       # carry-dominant init (the highway)
        self.carry = nn.Linear(e + 2, 2)

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([theta, x], dim=-1)
        t = torch.sigmoid(self.gate(inp))
        out: torch.Tensor = t * self.h(inp) + (1.0 - t) * self.carry(inp)
        return out


def _build(config: _SpiralConfig, walk_inc: torch.Tensor) -> nn.Module:
    e = int(walk_inc.shape[1])
    if config == "spiral":
        return SpiralModel(walk_inc)
    if config == "highway_mlp":
        return HighwayMlp(e)
    if config == "mlp":
        return FlatMlp(e)
    raise ValueError(f"unknown config {config!r}")


def train_eval(model: nn.Module, walk_inc: torch.Tensor, *, n_train: int, n_test: int, epochs: int,
               lr: float, seed: int) -> float:
    """Fit ``model`` to the walk-holonomy target; return held-out test MSE (deterministic)."""
    torch.manual_seed(seed)
    th_tr, x_tr, y_tr = make_spiral_data(walk_inc, n_train, seed=seed)
    th_te, x_te, y_te = make_spiral_data(walk_inc, n_test, seed=seed + 4231)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(th_tr, x_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(th_te, x_te), y_te))


def run_spiral_probe(*, k_paths: "list[int] | None" = None, n_train: int = 256, n_test: int = 1024,
                     seeds: int = 3, epochs: int = 400, lr: float = 0.02) -> dict[str, object]:
    """Sweep the number of walks ``K``; fit spiral vs highway_mlp vs mlp on the walk-holonomy target.

    # Postconditions one row per ``K`` with each config's median test MSE + params; the spiral is expected to
      stay ~flat (few params, right structure) while the flat/highway nets degrade with ``K``."""
    if k_paths is None:
        k_paths = [1, 2, 4, 8]
    configs: tuple[_SpiralConfig, ...] = ("spiral", "highway_mlp", "mlp")
    rows: list[dict[str, object]] = []
    for k in k_paths:
        walk_inc = theta_graph_walks(k)
        row: dict[str, object] = {"k_paths": k}
        params = 0
        for cfg in configs:
            mses = []
            for s in range(seeds):
                model = _build(cfg, walk_inc)
                params = sum(p.numel() for p in model.parameters())
                mses.append(train_eval(model, walk_inc, n_train=n_train, n_test=n_test, epochs=epochs,
                                       lr=lr, seed=s))
            row[f"{cfg}_mse"] = round(statistics.median(mses), 5)
            row[f"{cfg}_params"] = params
        row["mlp_over_spiral"] = round(float(row["mlp_mse"]) / max(float(row["spiral_mse"]), 1e-9), 2)  # type: ignore[arg-type]
        rows.append(row)
    return dict(n_train=n_train, seeds=seeds, epochs=epochs, rows=rows)


def plot_spiral_probe(report: dict[str, object], out_path: str | Path) -> Path:
    """Walk-holonomy test MSE vs #walks for spiral / highway_mlp / mlp (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    ks = [r["k_paths"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for cfg, style, color in (("spiral", "-o", "#2ca02c"), ("highway_mlp", "-s", "#d62728"),
                              ("mlp", "-^", "#7f7f7f")):
        ax.plot(ks, [r[f"{cfg}_mse"] for r in rows], style, color=color, label=cfg)
    ax.set_xlabel("number of walks K (θ-graph)")
    ax.set_ylabel("walk-holonomy test MSE")
    ax.set_title("Spiral carries the holonomy; the plain highway carries nothing structural")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: "list[str] | None" = None) -> int:
    torch.set_num_threads(1)            # supervised A/B on fixed data → strict-deterministic (CLAUDE.md §3)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out-dir", default="reports/spiral_probe")
    a = ap.parse_args(argv)
    report = run_spiral_probe(seeds=a.seeds, epochs=a.epochs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spiral_probe.json").write_text(json.dumps(report, indent=2))
    plot_spiral_probe(report, out / "spiral_probe")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
