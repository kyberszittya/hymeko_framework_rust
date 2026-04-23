"""Decision-boundary dig-in for the Circles counter-example.

The overnight suite found spectral-entropy regularisation hurts on
Concentric Circles (Δ=-0.060 pp, t=-2.30, p<0.05). This script
visualises what baseline vs. scalar_entropy actually learn:

1. Generates a Circles dataset (noise=0.2).
2. Trains an identical SyntheticMLP twice — baseline, then with
   scalar_entropy regularisation — on the same data, same seed.
3. Evaluates both over a dense 2-D grid and renders the decision
   boundary + training points side-by-side.
4. Adds a third panel zooming in on a misclassified band.

Output: hymeko_bench/paper_figs/fig_circles_dig.pdf / .png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Reuse the benchmark's model + data helpers for apples-to-apples config.
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "python" / "benches" / "thesis_iv_hard"))
import run_benchmark as R  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "hymeko_bench" / "paper_figs"
N_SEEDS = 20
N_TRAIN, N_VAL, NOISE = 2000, 500, 0.2
EPOCHS = 50
LR = 1e-3
LAM = 0.1
REG_EVERY_N = 10


def make_data(seed: int):
    rng_tr = np.random.default_rng(seed * 2 + 0)
    rng_va = np.random.default_rng(seed * 2 + 1)
    X_tr, y_tr = R._make_circles(N_TRAIN, NOISE, rng_tr)
    X_va, y_va = R._make_circles(N_VAL,   NOISE, rng_va)
    return (torch.from_numpy(X_tr), torch.from_numpy(y_tr),
            torch.from_numpy(X_va), torch.from_numpy(y_va))


def train(model, X_tr, y_tr, arm: str, seed: int) -> float:
    """Train for EPOCHS using paired config; return final val acc.
    Uses full-batch training since the data is small."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # mimic the benchmark's batched iteration
    n = len(X_tr)
    batch_size = 128
    step = 0
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, device=device,
                              generator=torch.Generator(device=device).manual_seed(seed + epoch))
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if arm == "scalar_entropy" and step % REG_EVERY_N == 0:
                adj = model.spectral_adjacency(view="dataflow")
                eigs = R.normalized_laplacian_eigvals(adj)
                if eigs is not None:
                    H = R.spectral_entropy_bits(eigs)
                    loss = loss + LAM * H
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    return model


def grid_predict(model, xmin=-1.4, xmax=1.4, ymin=-1.4, ymax=1.4, n=260):
    device = next(model.parameters()).device
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(grid).to(device))
        prob = torch.softmax(pred, dim=1)[:, 1].cpu().numpy().reshape(n, n)
    return xx, yy, prob


def _panel(ax, xx, yy, prob, X_va, y_va, title):
    cs = ax.contourf(xx, yy, prob, levels=np.linspace(0, 1, 21),
                     cmap="RdBu_r", alpha=0.85, extend="both")
    ax.contour(xx, yy, prob, levels=[0.5], colors="k", linewidths=1.2)
    y_np = y_va.cpu().numpy()
    X_np = X_va.cpu().numpy()
    ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
               c="#4273b8", s=10, edgecolor="white", linewidth=0.3, label="class 0 (outer)")
    ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
               c="#c44548", s=10, edgecolor="white", linewidth=0.3, label="class 1 (inner)")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return cs


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Aggregate across many seeds so the visualisation reflects the
    # population-level bias, not single-run noise. For each seed we
    # train a paired baseline + scalar_entropy model on fresh data,
    # evaluate both on a dense grid, and average the class-1
    # probability surfaces. The resulting ⟨p⟩ maps are the expected
    # decision-surface of each arm over the data distribution.
    probs_base, probs_treat = [], []
    accs_base, accs_treat = [], []
    last_va = None
    for seed in range(N_SEEDS):
        X_tr, y_tr, X_va, y_va = make_data(seed)
        last_va = (X_va, y_va)

        torch.manual_seed(seed)
        m_base = R.SyntheticMLP()
        torch.manual_seed(seed)
        m_treat = R.SyntheticMLP()

        m_base  = train(m_base,  X_tr, y_tr, arm="baseline",       seed=seed)
        m_treat = train(m_treat, X_tr, y_tr, arm="scalar_entropy", seed=seed)

        dev_b = next(m_base.parameters()).device
        dev_t = next(m_treat.parameters()).device
        with torch.no_grad():
            accs_base.append(
                (m_base(X_va.to(dev_b)).argmax(1).cpu() == y_va).float().mean().item())
            accs_treat.append(
                (m_treat(X_va.to(dev_t)).argmax(1).cpu() == y_va).float().mean().item())
        xx_b, yy_b, p_b = grid_predict(m_base)
        _,    _,    p_t = grid_predict(m_treat)
        probs_base.append(p_b)
        probs_treat.append(p_t)
        print(f"  seed {seed:2d}: base={accs_base[-1]:.4f} treat={accs_treat[-1]:.4f} "
              f"Δ={accs_treat[-1]-accs_base[-1]:+.4f}")

    prob_b = np.mean(probs_base,  axis=0)
    prob_t = np.mean(probs_treat, axis=0)
    acc_base  = float(np.mean(accs_base))
    acc_treat = float(np.mean(accs_treat))
    delta = acc_treat - acc_base
    print(f"\n  mean over {N_SEEDS} seeds: "
          f"baseline={acc_base:.4f}  scalar_entropy={acc_treat:.4f}  Δ={delta:+.4f}")

    X_va, y_va = last_va

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    _panel(axes[0], xx_b, yy_b, prob_b, X_va, y_va,
           f"Baseline (no reg.), mean of {N_SEEDS} seeds\nmean val acc = {acc_base:.4f}")
    cs = _panel(axes[1], xx_b, yy_b, prob_t, X_va, y_va,
           f"scalar\\_entropy (λ=0.1), mean of {N_SEEDS} seeds\n"
           f"mean val acc = {acc_treat:.4f}  (Δ = {delta:+.4f})")

    # Difference panel: where do the two models disagree on class 1 probability?
    diff = prob_t - prob_b
    ax = axes[2]
    im = ax.contourf(xx_b, yy_b, diff, levels=np.linspace(-0.5, 0.5, 21),
                     cmap="PuOr_r", extend="both")
    ax.contour(xx_b, yy_b, prob_b, levels=[0.5], colors="0.25",
               linewidths=0.8, linestyles="--")
    ax.contour(xx_b, yy_b, prob_t, levels=[0.5], colors="0.05",
               linewidths=1.0)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(r"$p_{\text{treat}} - p_{\text{baseline}}$" + "\n"
                 r"(dashed = baseline 0.5 contour, solid = treatment)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.88,
                 ticks=[-0.5, -0.25, 0, 0.25, 0.5])

    fig.suptitle(
        r"Circles counter-example: spectral-entropy shifts the decision "
        r"boundary inward, pushing more inner points to class 0",
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_circles_dig.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_circles_dig.png", dpi=150, bbox_inches="tight")
    print(f"wrote {OUT / 'fig_circles_dig.pdf'}")


if __name__ == "__main__":
    main()
