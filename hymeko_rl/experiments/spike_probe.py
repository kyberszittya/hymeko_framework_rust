"""The spikes toy — when the connection is non-abelian, timing (spikes) must select the walk order.

Completes the rotor/spikes ablation (``docs/plans/2026-06-26-rotor-spikes-ablation/``). SO(2) (the rotor
toy) is abelian, so a walk's holonomy is order-independent and spikes do nothing. Spikes earn their keep on a
**non-abelian** connection (SO(3)): the holonomy of a walk depends on the **order** edges are traversed, and
**spike timing selects the time-ordered walk**.

Setup: two non-commuting SO(3) edge rotations ``R_a = R_x(θ)``, ``R_b = R_y(θ)`` (``R_aR_b ≠ R_bR_a`` for
θ≠0). Per sample a source ``v ∈ R³`` and a **spike** ``s ∈ {0,1}`` choosing the traversal order; the target is
the order-selected transport ``y = (s ? R_aR_b : R_bR_a)·v``. Three models:

* **spike_gated** — learns ``θ_a, θ_b`` and uses the spike ``s`` to select the ordered product. The right
  structure (timing + order) → fits.
* **order_blind** — a full learnable linear map of ``v`` that **ignores the spike** (no timing). Best it can do
  is the *average* of the two orders → stuck at the order-dependent residual.
* **mlp** — flat ``(v, s) → MLP``: the structure-blind reference.

Prediction (the spike signature): sweep θ (non-commutativity). At θ=0 the rotations commute, order is
irrelevant, and order_blind fits (MSE 0). As θ grows the order_blind model's error rises ∝ the commutator while
spike_gated stays ~0 — *you need the spike to select order once the connection is non-abelian*.

Reuses nothing from the SO(2) probes (different group); see also ``hymeko_rl/rotor_probe.py`` (SO(2) rotor).
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn


def rot_x(t: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) rotation about the x-axis by angle ``t`` (0-dim tensor)."""
    c, s = torch.cos(t), torch.sin(t)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack([torch.stack([one, zero, zero]),
                        torch.stack([zero, c, -s]),
                        torch.stack([zero, s, c])])


def rot_y(t: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) rotation about the y-axis by angle ``t`` (0-dim tensor)."""
    c, s = torch.cos(t), torch.sin(t)
    one, zero = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack([torch.stack([c, zero, s]),
                        torch.stack([zero, one, zero]),
                        torch.stack([-s, zero, c])])


def make_spike_data(theta: float, n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(v (n,3), s (n,), y (n,3))`` with ``y = (s ? R_aR_b : R_bR_a)·v`` for ``R_a=R_x(θ)``, ``R_b=R_y(θ)``.

    The spike ``s ∈ {0,1}`` selects the traversal order; for θ≠0 the two orders differ (non-abelian).
    # Postconditions ``y[i] = M_{s_i} v_i`` with ``M_1 = R_aR_b``, ``M_0 = R_bR_a``."""
    g = torch.Generator().manual_seed(seed)
    th = torch.tensor(float(theta))
    m1 = (rot_x(th) @ rot_y(th)).detach()                 # order: R_b first, then R_a  (s = 1)
    m0 = (rot_y(th) @ rot_x(th)).detach()                 # order: R_a first, then R_b  (s = 0)
    v = torch.randn(n, 3, generator=g)
    s = (torch.rand(n, generator=g) < 0.5).float()
    y1, y0 = v @ m1.T, v @ m0.T
    y = torch.where(s.unsqueeze(1) > 0.5, y1, y0)
    return v, s, y


class SpikeGated(nn.Module):
    """Learns ``θ_a, θ_b`` and uses the spike to select the ordered SO(3) product — the timing+order structure."""

    def __init__(self) -> None:
        super().__init__()
        self.theta_a = nn.Parameter(torch.zeros(()))
        self.theta_b = nn.Parameter(torch.zeros(()))

    def forward(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        m1 = rot_x(self.theta_a) @ rot_y(self.theta_b)    # s = 1 order
        m0 = rot_y(self.theta_b) @ rot_x(self.theta_a)    # s = 0 order
        out: torch.Tensor = torch.where((s > 0.5).unsqueeze(1), v @ m1.T, v @ m0.T)
        return out


class OrderBlind(nn.Module):
    """A full learnable linear map of ``v`` that IGNORES the spike (no timing). Can only reach the average of
    the two orders → stuck at the order-dependent residual."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Linear(3, 3, bias=False)

    def forward(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.w(v)                     # no s → cannot select the order
        return out


class SpikeMlp(nn.Module):
    """Flat reference: ``concat(v, s) → MLP → ŷ``."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, v: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(torch.cat([v, s.unsqueeze(1)], dim=1))
        return out


_CONFIGS = ("spike_gated", "order_blind", "mlp")


def _build(config: str) -> nn.Module:
    if config == "spike_gated":
        return SpikeGated()
    if config == "order_blind":
        return OrderBlind()
    if config == "mlp":
        return SpikeMlp()
    raise ValueError(f"unknown config {config!r}")


def train_eval(model: nn.Module, theta: float, *, n_train: int, n_test: int, epochs: int, lr: float,
               seed: int) -> float:
    """Fit ``model`` to the order-selected transport at ``theta``; return held-out test MSE (deterministic)."""
    torch.manual_seed(seed)
    v_tr, s_tr, y_tr = make_spike_data(theta, n_train, seed=seed)
    v_te, s_te, y_te = make_spike_data(theta, n_test, seed=seed + 8101)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(v_tr, s_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(v_te, s_te), y_te))


def run_spike_probe(*, thetas: "list[float] | None" = None, n_train: int = 512, n_test: int = 2048,
                    seeds: int = 3, epochs: int = 500, lr: float = 0.03) -> dict[str, object]:
    """Sweep the rotation angle θ (= non-commutativity); fit the three models, record median test MSE.

    # Postconditions one row per θ with each config's MSE; order_blind rises with θ while spike_gated stays
      ~0 (the spike is needed once the connection is non-abelian)."""
    if thetas is None:
        thetas = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    rows: list[dict[str, float]] = []
    for theta in thetas:
        row: dict[str, float] = {"theta": round(theta, 3)}
        for cfg in _CONFIGS:
            mses = [train_eval(_build(cfg), theta, n_train=n_train, n_test=n_test, epochs=epochs, lr=lr,
                               seed=s) for s in range(seeds)]
            row[f"{cfg}_mse"] = round(statistics.median(mses), 5)
        row["blind_over_gated"] = round(row["order_blind_mse"] / max(row["spike_gated_mse"], 1e-9), 2)
        rows.append(row)
    return dict(n_train=n_train, seeds=seeds, epochs=epochs, rows=rows)


def plot_spike_probe(report: dict[str, object], out_path: str | Path) -> Path:
    """Test MSE vs rotation angle θ (non-commutativity) for the three configs (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    th = [r["theta"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for cfg, style, color in (("spike_gated", "-o", "#2ca02c"), ("order_blind", "-s", "#d62728"),
                              ("mlp", "-^", "#7f7f7f")):
        ax.plot(th, [r[f"{cfg}_mse"] for r in rows], style, color=color, label=cfg)
    ax.set_xlabel("rotation angle θ  (non-commutativity ‖[R_a,R_b]‖ grows with θ)")
    ax.set_ylabel("order-selected-transport test MSE")
    ax.set_title("Non-abelian connection: you need the spike to select walk order")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ── stronger version: varying connection + multi-edge walks → a GENERALIZATION win, not just representability ──
# Each of K=2 walks is a length-L chain of SO(3) edge rotations about cycling axes (x,y,z — consecutive edges
# don't commute). The connection (edge angles θ) VARIES per sample, a spike selects the walk, and the target is
# the spike-selected walk's holonomy applied to a source v. The spike-gated model composes the rotations along
# the selected walk (the right structure, E learnable gains); the flat MLP must learn the whole conditional
# non-commuting product. As the walk grows the MLP degrades while the spike-gated stays flat — a generalization
# advantage (cf. the chain probe), not the earlier mere representability.
_AXES = (0, 1, 2)   # x, y, z — cycled per edge so consecutive edges are non-commuting


def _batch_rot(axis: int, ang: torch.Tensor) -> torch.Tensor:
    """Per-sample SO(3) rotation ``(B,3,3)`` about ``axis`` ∈ {0:x,1:y,2:z} by angles ``ang (B,)``."""
    c, s = torch.cos(ang), torch.sin(ang)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    if axis == 0:
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == 1:
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def _walk_holonomy(axes: "list[int]", angles: torch.Tensor) -> torch.Tensor:
    """Composed rotation ``R_L···R_1`` ``(B,3,3)`` of a walk: edges (axes in order) with per-sample ``angles
    (B,L)``. Edge 0 is applied first (rightmost in the product)."""
    b = angles.shape[0]
    r = torch.eye(3).expand(b, 3, 3)
    for i, ax in enumerate(axes):
        r = torch.bmm(_batch_rot(ax, angles[:, i]), r)
    return r


def _walk_axes(walk: int, walk_len: int) -> "list[int]":
    """The cycling axes (x,y,z,…) of ``walk``'s ``walk_len`` edges — distinct consecutive axes → non-commuting."""
    return [_AXES[(walk * walk_len + i) % 3] for i in range(walk_len)]


def make_walk_data(walk_len: int, n: int, *, seed: int, n_walks: int = 2, gain: float = 1.0,
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(theta (n,E), s (n,), v (n,3), y (n,3))``, ``E = n_walks·walk_len``. Per sample the connection
    ``theta`` (edge angles) varies; spike ``s`` selects a walk; ``y = gain·Hol(walk_s; theta)·v``."""
    g = torch.Generator().manual_seed(seed)
    e = n_walks * walk_len
    theta = (torch.rand(n, e, generator=g) * 2.0 - 1.0)              # U(-1,1)^E — the varying connection
    s = torch.randint(0, n_walks, (n,), generator=g)
    v = torch.randn(n, 3, generator=g)
    per_walk = []
    for w in range(n_walks):
        cols = list(range(w * walk_len, (w + 1) * walk_len))
        r = _walk_holonomy(_walk_axes(w, walk_len), gain * theta[:, cols])
        per_walk.append(torch.bmm(r, v.unsqueeze(-1)).squeeze(-1))   # (n,3)
    stacked = torch.stack(per_walk, dim=0)                           # (n_walks, n, 3)
    y = stacked[s, torch.arange(n)]                                  # select the spiked walk → (n,3)
    return theta, s, v, y


class WalkSpikeGated(nn.Module):
    """Composes the SO(3) edge rotations along the SPIKE-SELECTED walk (the right structure: walk topology +
    rotation form + order). Learnable = a per-edge gain ``g`` (E params); the spike picks the walk."""

    def __init__(self, walk_len: int, n_walks: int = 2) -> None:
        super().__init__()
        self.walk_len, self.n_walks = walk_len, n_walks
        self.g = nn.Parameter(torch.ones(n_walks * walk_len))

    def forward(self, theta: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        per_walk = []
        for w in range(self.n_walks):
            cols = list(range(w * self.walk_len, (w + 1) * self.walk_len))
            r = _walk_holonomy(_walk_axes(w, self.walk_len), self.g[cols] * theta[:, cols])
            per_walk.append(torch.bmm(r, v.unsqueeze(-1)).squeeze(-1))
        stacked = torch.stack(per_walk, dim=0)                       # (n_walks, B, 3)
        out: torch.Tensor = stacked[s, torch.arange(theta.shape[0])]
        return out


class WalkMlp(nn.Module):
    """Flat reference: ``concat(theta, onehot(s), v) → MLP → ŷ`` — must learn the conditional product."""

    def __init__(self, walk_len: int, n_walks: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.n_walks = n_walks
        in_dim = n_walks * walk_len + n_walks + 3
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, theta: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(s, self.n_walks).float()
        out: torch.Tensor = self.net(torch.cat([theta, onehot, v], dim=1))
        return out


def _walk_train_eval(model: nn.Module, walk_len: int, *, n_train: int, n_test: int, epochs: int, lr: float,
                     seed: int) -> float:
    torch.manual_seed(seed)
    th_tr, s_tr, v_tr, y_tr = make_walk_data(walk_len, n_train, seed=seed)
    th_te, s_te, v_te, y_te = make_walk_data(walk_len, n_test, seed=seed + 6271)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(th_tr, s_tr, v_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(th_te, s_te, v_te), y_te))


def run_spike_walks(*, walk_lens: "list[int] | None" = None, n_train: int = 512, n_test: int = 2048,
                    seeds: int = 3, epochs: int = 500, lr: float = 0.02) -> dict[str, object]:
    """Sweep walk length L; fit spike_gated (structured) vs flat MLP on the spike-selected walk-holonomy.

    # Postconditions one row per L with both median MSEs + ratio; the spike-gated stays ~flat while the MLP
      degrades with L — a generalization win (not the earlier representability)."""
    if walk_lens is None:
        walk_lens = [1, 2, 3, 4]
    rows: list[dict[str, float]] = []
    for length in walk_lens:
        def med(make: "Callable[[], nn.Module]") -> float:
            return statistics.median(
                _walk_train_eval(make(), length, n_train=n_train, n_test=n_test, epochs=epochs, lr=lr, seed=s)
                for s in range(seeds))
        gated = med(lambda: WalkSpikeGated(length))
        mlp = med(lambda: WalkMlp(length))
        rows.append(dict(walk_len=length, spike_gated_mse=round(gated, 5), mlp_mse=round(mlp, 5),
                         mlp_over_gated=round(mlp / max(gated, 1e-9), 2)))
    return dict(n_train=n_train, seeds=seeds, epochs=epochs, rows=rows)


def plot_spike_walks(report: dict[str, object], out_path: str | Path) -> Path:
    """Test MSE vs walk length — spike_gated (structured) vs flat MLP (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    ls = [r["walk_len"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(ls, [r["spike_gated_mse"] for r in rows], "-o", color="#2ca02c", label="spike_gated (structured)")
    ax.plot(ls, [r["mlp_mse"] for r in rows], "-^", color="#7f7f7f", label="flat MLP")
    ax.set_xlabel("walk length L (composition depth)")
    ax.set_ylabel("spike-selected walk-holonomy test MSE")
    ax.set_title("Spike-gated composition generalizes; the flat net degrades with walk length")
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
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--walks", action="store_true",
                    help="the stronger version: varying connection + multi-edge walks (generalization vs length)")
    ap.add_argument("--out-dir", default="reports/spike_probe")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.walks:
        report = run_spike_walks(seeds=a.seeds, epochs=a.epochs)
        (out / "spike_walks.json").write_text(json.dumps(report, indent=2))
        plot_spike_walks(report, out / "spike_walks")
        print(json.dumps(report, indent=2))
        return 0
    report = run_spike_probe(seeds=a.seeds, epochs=a.epochs)
    (out / "spike_probe.json").write_text(json.dumps(report, indent=2))
    plot_spike_probe(report, out / "spike_probe")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
