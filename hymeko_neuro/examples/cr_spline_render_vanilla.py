"""Render learned Catmull-Rom splines from a trained vanilla HSiKAN
pose model — **without** the fuzzy / Atanassov-pair overlay.

This is the companion figure to `cr_spline_render.py`. Where the
fuzzy version interprets the two CR branches as an Atanassov
$(\\mu^+, \\mu^-)$ pair and adds the Zadeh hedge $g(x)$ + mix
$\\mu(x)$ + hesitancy $\\pi$, this script shows the **same model
family** but renders only the raw, unbounded Catmull-Rom curves
themselves — the vanilla neural-network reading.

This contrast directly answers the reviewer question: *"is the
Atanassov / Zadeh / hesitancy interpretation real, or is it just
a relabelling of generic neural primitives?"* The same architecture
admits both readings; the choice of how to interpret the two
signed branches (raw activation vs IFS membership pair) is what
distinguishes the framework's positions.

Differences from `cr_spline_render.py`:

- Model: `HSiKANPoseModel` (vanilla HSiKAN, signed branches above/below
  patch mean) instead of `FuzzySignaturePoseModel`.
- CR activation: the `n_branches=2` `CRActivation` inside each layer's
  `SignedBranchConv`, **without** `clamp_to_01` post-σ. Output is
  unbounded.
- x-axis: the CR's native [-3, 3] grid domain (no fuzzy [0, 1]
  rescaling).
- Plotted curves: CR(branch=0)(x) "above-mean polarity" + CR(branch=1)(x)
  "below-mean polarity". No gate, no mix, no hesitancy.
- Optional: the 8 control points marked as dots on each curve so the
  reader can see the spline scaffolding.

Run:
    cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
    PYTHONPATH=. python hymeko_neuro/examples/cr_spline_render_vanilla.py

Output:
    /tmp/cr_spline_render_vanilla/
      cr_vanilla_per_layer_grid.png      # rows=layers, cols=channels
      cr_vanilla_init_vs_trained.png     # before/after for ch 0
      cr_vanilla_data.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from hymeko_neuro.experiments.vision.fuzzy_pose import (
    HSiKANPoseModel,
    SyntheticPoseDataset,
)


H = W = 32
N_KP = 8
OUT_DIR = Path("/tmp/cr_spline_render_vanilla")
MODEL_CACHE = Path("/tmp/cr_spline_render_vanilla/hsikan_vanilla_pose.pt")


# ─── Train (or load) a small vanilla HSiKAN pose model ─────────────


def get_model(
    device: torch.device,
    *, n_train: int = 500, n_epochs: int = 15,
    force_retrain: bool = False,
) -> HSiKANPoseModel:
    model = HSiKANPoseModel(
        H=H, W=W, n_keypoints=N_KP, d=16, n_layers=8,
        arities=[(3, 1), (5, 2)],
    ).to(device)
    if MODEL_CACHE.exists() and not force_retrain:
        try:
            model.load_state_dict(torch.load(
                MODEL_CACHE, map_location=device,
            ))
            print(f"[cr-vanilla] loaded cached model from {MODEL_CACHE}",
                  flush=True)
            return model
        except (RuntimeError, KeyError) as e:
            print(f"[cr-vanilla] cache mismatch ({e}); re-training",
                  flush=True)
    print(f"[cr-vanilla] training tiny model "
          f"(n_train={n_train}, n_epochs={n_epochs})", flush=True)
    ds = SyntheticPoseDataset(
        n_samples=n_train, H=H, W=W, n_keypoints=N_KP, seed=0,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    t0 = time.time()
    for ep in range(n_epochs):
        for imgs, coords in loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            pred = model(imgs)
            loss = torch.nn.functional.mse_loss(pred, coords)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep in (0, n_epochs - 1):
            print(f"[cr-vanilla]   ep {ep:2d} loss={loss.item():.3f}",
                  flush=True)
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_CACHE)
    print(f"[cr-vanilla]   trained in {time.time() - t0:.1f}s, "
          f"cached to {MODEL_CACHE}", flush=True)
    return model


# ─── Extract & evaluate one signed-branch CR pair (vanilla) ────────


def evaluate_signed_cr(
    cr_module, *, n_eval: int = 200,
) -> dict[str, np.ndarray]:
    """Evaluate the vanilla two-branch CR activation on a dense x-grid
    over its native [-3, 3] domain. Returns NumPy arrays for both
    branches plus the control-point coordinates so the figure can
    overlay them.

    ``cr_module`` is the `CRActivation` inside a `SignedBranchConv`
    (in `HSiKANVisionLayer.convs[arity_idx].activation`).
    """
    device = cr_module.cpts.device
    x = torch.linspace(-3.0, 3.0, n_eval, device=device)
    d = cr_module.cpts.shape[1]  # channels
    x_broadcast = x.view(1, n_eval, 1).expand(1, n_eval, d)
    with torch.no_grad():
        y0 = cr_module(x_broadcast, branch_idx=0).squeeze(0)  # above-mean
        y1 = cr_module(x_broadcast, branch_idx=1).squeeze(0)  # below-mean
    # Control points live on the same [-3, 3] grid (see CRActivation).
    cp_x = cr_module.x_grid.detach().cpu().numpy()         # (m,)
    cp_y0 = cr_module.cpts[0].detach().cpu().numpy()       # (d, m)
    cp_y1 = cr_module.cpts[1].detach().cpu().numpy()       # (d, m)
    return {
        "x": x.detach().cpu().numpy(),
        "y_above": y0.detach().cpu().numpy(),              # (n_eval, d)
        "y_below": y1.detach().cpu().numpy(),              # (n_eval, d)
        "cp_x":    cp_x,                                    # (m,)
        "cp_y_above": cp_y0,                                # (d, m)
        "cp_y_below": cp_y1,                                # (d, m)
    }


def get_first_arity_cr(hsikan_layer) -> "CRActivation":
    """The signed-branch CR is at ``layer.convs[0].activation``.
    For multi-arity layers we take the first arity for the figure."""
    return hsikan_layer.convs[0].activation


# ─── Plot per-layer grid ───────────────────────────────────────────


def plot_layer_grid(
    layers_data: list[dict],
    *, channels: list[int], out_path: Path,
    show_control_points: bool = True,
) -> None:
    """Multi-panel: rows=layer, cols=channel. Each panel plots the
    two unbounded signed-branch CR curves on x∈[-3, 3] with the
    control points overlaid (if requested)."""
    n_rows = len(layers_data)
    n_cols = len(channels)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.0 * n_rows),
        squeeze=False, sharex=True,
    )
    # Per-panel y-axis is per-panel data range (the unbounded
    # activation can be very different scale across layers).
    for r, layer_data in enumerate(layers_data):
        x = layer_data["x"]
        for c_idx, ch in enumerate(channels):
            ax = axes[r][c_idx]
            y_above = layer_data["y_above"][:, ch]
            y_below = layer_data["y_below"][:, ch]
            ax.plot(x, y_above, color="#d23838", linewidth=1.8,
                      label=r"CR$^+(x)$  (above-mean branch)"
                              if (r == 0 and c_idx == 0) else None)
            ax.plot(x, y_below, color="#2f6ec9", linewidth=1.8,
                      label=r"CR$^-(x)$  (below-mean branch)"
                              if (r == 0 and c_idx == 0) else None)
            if show_control_points:
                cp_x = layer_data["cp_x"]
                cp_y_a = layer_data["cp_y_above"][ch]
                cp_y_b = layer_data["cp_y_below"][ch]
                ax.scatter(cp_x, cp_y_a, color="#d23838",
                              s=14, zorder=5,
                              label="control pts (CR$^+$)"
                                      if (r == 0 and c_idx == 0) else None)
                ax.scatter(cp_x, cp_y_b, color="#2f6ec9",
                              s=14, zorder=5, marker="s",
                              label="control pts (CR$^-$)"
                                      if (r == 0 and c_idx == 0) else None)
            ax.axhline(0, color="gray", linewidth=0.6,
                          linestyle=":", alpha=0.7)
            ax.set_xlim(-3, 3)
            ax.grid(True, linestyle=":", alpha=0.4)
            if r == 0:
                ax.set_title(f"channel {ch}", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel(f"layer {r}", fontsize=9)
            ax.tick_params(axis="both", which="major", labelsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.005),
        )
    fig.suptitle(
        "Learned vanilla Catmull-Rom splines per (layer, channel) — "
        "two signed branches, no fuzzy overlay",
        fontsize=11, fontweight="bold", y=1.04,
    )
    fig.text(0.5, -0.01, "CR input $x$ (native [-3, 3] grid)",
              ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_init_vs_trained(
    init_data: dict, trained_data: dict,
    *, channel: int, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, (data, title) in zip(axes, [
        (init_data,    "Before training"),
        (trained_data, "After training"),
    ]):
        x = data["x"]
        y_a = data["y_above"][:, channel]
        y_b = data["y_below"][:, channel]
        ax.plot(x, y_a, color="#d23838", linewidth=1.8, label=r"CR$^+(x)$")
        ax.plot(x, y_b, color="#2f6ec9", linewidth=1.8, label=r"CR$^-(x)$")
        cp_x = data["cp_x"]
        ax.scatter(cp_x, data["cp_y_above"][channel],
                       color="#d23838", s=18, zorder=5)
        ax.scatter(cp_x, data["cp_y_below"][channel],
                       color="#2f6ec9", s=18, zorder=5, marker="s")
        ax.axhline(0, color="gray", linewidth=0.6,
                       linestyle=":", alpha=0.7)
        ax.set_xlim(-3, 3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x$")
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel("CR output (unbounded)")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"Vanilla HSiKAN — layer 0, channel {channel}: "
        "what training learned",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--n-epochs", type=int, default=15)
    ap.add_argument("--channels", type=int, nargs="+",
                    default=[0, 1, 2, 3])
    ap.add_argument("--n-layers-show", type=int, default=4)
    ap.add_argument("--no-control-points", action="store_true")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cr-vanilla] device={device}", flush=True)

    # Init model for before-training panel.
    init_model = HSiKANPoseModel(
        H=H, W=W, n_keypoints=N_KP, d=16, n_layers=8,
        arities=[(3, 1), (5, 2)],
    ).to(device)
    init_cr0 = get_first_arity_cr(init_model.layers[0])
    init_data = evaluate_signed_cr(init_cr0)

    # Train (or load) the trained model.
    trained_model = get_model(
        device, n_epochs=args.n_epochs,
        force_retrain=args.force_retrain,
    )

    n_layers_show = min(args.n_layers_show, trained_model.n_layers)
    layers_data = []
    for li in range(n_layers_show):
        cr_mod = get_first_arity_cr(trained_model.layers[li])
        layers_data.append(evaluate_signed_cr(cr_mod))

    # Per-layer grid figure.
    grid_path = OUT_DIR / "cr_vanilla_per_layer_grid.png"
    plot_layer_grid(
        layers_data, channels=args.channels,
        out_path=grid_path,
        show_control_points=not args.no_control_points,
    )
    print(f"[cr-vanilla] per-layer grid → {grid_path}", flush=True)

    # Init vs trained for channel 0 layer 0.
    init_vs_trained_path = OUT_DIR / "cr_vanilla_init_vs_trained.png"
    plot_init_vs_trained(
        init_data, layers_data[0],
        channel=0, out_path=init_vs_trained_path,
    )
    print(f"[cr-vanilla] init-vs-trained → {init_vs_trained_path}",
          flush=True)

    # Numerical dump.
    json_path = OUT_DIR / "cr_vanilla_data.json"
    serializable = {
        "init_layer0_channel0": {
            "x": init_data["x"].tolist(),
            "y_above": init_data["y_above"][:, 0].tolist(),
            "y_below": init_data["y_below"][:, 0].tolist(),
            "cp_x": init_data["cp_x"].tolist(),
            "cp_y_above": init_data["cp_y_above"][0].tolist(),
            "cp_y_below": init_data["cp_y_below"][0].tolist(),
        },
        "trained_layers": [
            {
                "layer_idx": li,
                "x": d["x"].tolist(),
                "cp_x": d["cp_x"].tolist(),
                "channels": [
                    {
                        "channel_idx": ch,
                        "y_above": d["y_above"][:, ch].tolist(),
                        "y_below": d["y_below"][:, ch].tolist(),
                        "cp_y_above": d["cp_y_above"][ch].tolist(),
                        "cp_y_below": d["cp_y_below"][ch].tolist(),
                    }
                    for ch in args.channels
                ],
            }
            for li, d in enumerate(layers_data)
        ],
    }
    with json_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[cr-vanilla] numerical data → {json_path}", flush=True)
    print(f"[cr-vanilla] done. All outputs in {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
