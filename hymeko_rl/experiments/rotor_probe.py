"""The rotor / holonomy toy — does a rotor *connection* capture graph holonomy a signed Z2 link cannot?

The gauge picture (memory ``project-gauge-holonomy-signed-hsikan``): a signed edge ``s_ij ∈ {±1}`` is a
**Z2 connection**, and structural balance is its trivial **holonomy** (sign-product around a cycle = +1). A
**rotor** replaces the sign with a continuous rotation ``R_ij ∈ SO(2)`` — a genuine connection that parallel-
transports a node's feature vector along an edge; its holonomy ``∏_cycle R_ij ∈ SO(2)`` is *continuous*, and
balance is only its Z2 quotient. So the signed model sees a **shadow** of what the rotor sees.

This toy isolates that in the simplest setting. A cycle of ``E`` edges has a hidden per-edge rotation
``θ_e``; the holonomy is ``Φ = Σ θ_e``. A source vector ``x ∈ R²`` is transported around the loop:
``y = R(Φ) x``. Two models learn per-edge transforms and compose them:

* **rotor** — per-edge angle ``α_e``; predicts ``R(Σ α_e) x`` → can match Φ → MSE → 0.
* **signed** — per-edge real **scalar** ``c_e`` (the Z2 ``±1`` case generalised); predicts ``(∏ c_e) x = c·x``
  — a scalar can *scale* x but never *rotate* it.

Analytic prediction (verified by the run): the signed model's best scalar is ``c* = cos Φ``, so
``MSE_signed(Φ) = sin²Φ`` while ``MSE_rotor(Φ) ≈ 0``. Sweeping Φ, the signed curve is ``sin²Φ`` (zero only at
Φ ∈ {0, π}, the Z2 points) and the rotor curve is flat at ~0 — one plot showing *the rotor sees the holonomy
the sign cannot*.

The **production** rotor is the quaternion/SO(3) ``RotorInjector`` / ``CayleyRotorEmbedding`` /
``SignedRotorPropagation`` in ``hymeko_neuro/.../run_hsikan_rotor.py``; this is a deliberately minimal SO(2)
isolation of the holonomy *principle* for a clean ablation figure, not a reimplementation of that head.
See ``docs/plans/2026-06-26-rotor-spikes-ablation/``.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn as nn


def rot_matrix(angle: torch.Tensor | float) -> torch.Tensor:
    """The 2×2 rotation ``R(angle)`` (orthogonal, ``det = +1``)."""
    a = torch.as_tensor(angle, dtype=torch.float32)
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def make_holonomy_data(phi: float, n: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(X (n,2), Y (n,2))`` with ``Y[i] = R(phi) · X[i]`` — the source vectors transported around the loop.

    # Preconditions ``n >= 1``. # Postconditions ``Y`` equals ``X`` rotated by ``phi`` (a hand check passes)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    y = x @ rot_matrix(phi).T                       # row i = R(phi) x_i
    return x, y


class RotorTransport(nn.Module):
    """The rotor connection's **holonomy**: a single learnable angle ``α`` (the composed cycle rotation
    ``R(Σ_e α_e)`` collapses to one SO(2) element). ``transport(x) = R(α) x`` — any continuous holonomy."""

    def __init__(self) -> None:
        super().__init__()
        self.angle = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x @ rot_matrix(self.angle).T
        return out


class SignedTransport(nn.Module):
    """The signed (Z2) connection's **holonomy**: a single learnable real **scalar** ``c`` (the product
    ``∏_e c_e`` over the cycle collapses to one gain; Z2 is ``c = ±1``). ``transport(x) = c·x`` — a scalar
    can *scale* ``x`` but never *rotate* it, so it only reaches the Z2 holonomy points."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x * self.scale
        return out


def train_eval(model: nn.Module, phi: float, *, n_train: int, n_test: int, epochs: int, lr: float,
               seed: int) -> float:
    """Fit ``model`` to the holonomy-transport target at ``phi``; return held-out test MSE (deterministic)."""
    torch.manual_seed(seed)
    x_tr, y_tr = make_holonomy_data(phi, n_train, seed=seed)
    x_te, y_te = make_holonomy_data(phi, n_test, seed=seed + 7919)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(x_tr), y_tr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(x_te), y_te))


def run_rotor_probe(*, phis: "list[float] | None" = None, n_train: int = 512, n_test: int = 2048,
                    seeds: int = 3, epochs: int = 400, lr: float = 0.05) -> dict[str, object]:
    """Sweep the holonomy Φ; for each, fit rotor vs signed (K seeds) and record median test MSE.

    # Postconditions one row per Φ with ``rotor_mse``/``signed_mse``/the analytic ``sin2``; the signed/rotor
      ratio at Φ≈π/2 is large (the rotor advantage). The signed curve should track ``sin²Φ``."""
    if phis is None:
        phis = [round(k * math.pi / 8, 4) for k in range(17)]      # 0 .. 2π in π/8 steps
    rows: list[dict[str, float]] = []
    for phi in phis:
        def med(make: "type[nn.Module]") -> float:
            return statistics.median(
                train_eval(make(), phi, n_train=n_train, n_test=n_test, epochs=epochs, lr=lr, seed=s)
                for s in range(seeds))
        rows.append(dict(phi=round(phi, 4), rotor_mse=round(med(RotorTransport), 5),
                         signed_mse=round(med(SignedTransport), 5), sin2=round(math.sin(phi) ** 2, 5)))
    half_pi = min(rows, key=lambda r: abs(r["phi"] - math.pi / 2))
    ratio = round(half_pi["signed_mse"] / max(half_pi["rotor_mse"], 1e-9), 1)
    return dict(n_train=n_train, seeds=seeds, epochs=epochs,
                signed_over_rotor_at_halfpi=ratio, rows=rows)


def plot_rotor_probe(report: dict[str, object], out_path: str | Path) -> Path:
    """Test MSE vs holonomy Φ — rotor (flat ~0) vs signed (sin²Φ), with the analytic curve overlaid (§9)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["rows"]  # type: ignore[index]
    phis = [r["phi"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(phis, [r["signed_mse"] for r in rows], "-o", color="#d62728", label="signed (Z₂ scalar)")
    ax.plot(phis, [r["rotor_mse"] for r in rows], "-o", color="#2ca02c", label="rotor (SO(2))")
    ax.plot(phis, [r["sin2"] for r in rows], "--", color="#888888", label="analytic  sin²Φ")
    ax.set_xlabel("holonomy Φ  (rotation accumulated around the cycle)")
    ax.set_ylabel("held-out test MSE")
    ax.set_title("Rotor connection captures the holonomy the signed Z₂ link cannot")
    ax.axvline(math.pi, color="#bbbbbb", lw=0.8, ls=":")
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
    ap.add_argument("--out-dir", default="reports/rotor_probe")
    a = ap.parse_args(argv)
    report = run_rotor_probe(seeds=a.seeds, epochs=a.epochs)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotor_probe.json").write_text(json.dumps(report, indent=2))
    plot_rotor_probe(report, out / "rotor_probe")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
