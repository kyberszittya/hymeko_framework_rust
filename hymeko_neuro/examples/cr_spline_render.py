"""Render learned Catmull-Rom splines from a trained FuzzySignature
pose model.

For the Kóczy / Niitsuma audience this is the most direct evidence
that the framework's "named fuzzy primitive" claim is real: every
learnable Catmull-Rom curve is a *learnable membership function*
on the unit interval, and we can plot what the model actually
learned for each (layer, channel).

The figure shows, for the first ``n_layers_show`` layers and the
first ``n_channels_show`` channels of a trained FuzzySignaturePose
model:

- **μ⁺(x)** (Atanassov membership) — red, after σ ∘ CR.
- **μ⁻(x)** (Atanassov non-membership) — blue, after σ ∘ CR.
- **g(x) = σ(τ·(x − ½))** (learnable Zadeh hedge) — green dashed.
- **μ(x) = g(x)·μ⁺(x) + (1−g(x))·μ⁻(x)** (the mix) — solid purple.
- **π(x) = 1 − μ(x)** (hesitancy proxy: how undecided the IFS pair
  is at input x) — light gray fill.

The x-axis is the fuzzy input domain [0, 1] (the model's
``cr_input_scale="unit_to_grid"`` rescales x → 6x−3 internally so
the CR's [-3, 3] grid is fully used; the plot un-rescales for
readability).

The figure communicates exactly what Kóczy's classical fuzzy
signature framework prescribes — a learnable, smooth, bounded
membership function per concept — except that here the curves
are *learned by gradient descent*, not designed by hand.

Run:
    cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
    PYTHONPATH=. python hymeko_neuro/examples/cr_spline_render.py

Output:
    /tmp/cr_spline_render/
      cr_per_layer_grid.png        # main figure: rows=layers, cols=channels
      cr_init_vs_trained.png       # before/after comparison for channel 0
      cr_data.json                 # numerical (x, μ⁺, μ⁻, μ) per (layer, ch)
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
    FuzzySignaturePoseModel,
    SyntheticPoseDataset,
)


H = W = 32
N_KP = 8
OUT_DIR = Path("/tmp/cr_spline_render")
POSE_MODEL_CACHE = Path("/tmp/pose_demo_outputs/fuzzy_pose_model.pt")


# ─── Train (or load) a small model ─────────────────────────────────


def get_model(device: torch.device,
              *, n_train: int = 500, n_epochs: int = 20,
              force_retrain: bool = False) -> FuzzySignaturePoseModel:
    """Reuse the pose_demo cached model if present, else train a tiny
    one quickly. We use d=16 to match the demo cache; if the cache is
    a different d, we re-train."""
    model = FuzzySignaturePoseModel(
        H=H, W=W, n_keypoints=N_KP, d=16, n_layers=8,
        arities=[(3, 1), (5, 2)],
    ).to(device)
    if POSE_MODEL_CACHE.exists() and not force_retrain:
        try:
            model.load_state_dict(torch.load(
                POSE_MODEL_CACHE, map_location=device,
            ))
            print(f"[cr] loaded cached pose model from {POSE_MODEL_CACHE}",
                  flush=True)
            return model
        except (RuntimeError, KeyError) as e:
            print(f"[cr] cache mismatch ({e}); training fresh model",
                  flush=True)
    print(f"[cr] training tiny model "
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
            print(f"[cr]   ep {ep:2d} loss={loss.item():.3f}",
                  flush=True)
    print(f"[cr]   trained in {time.time() - t0:.1f}s", flush=True)
    return model


# ─── Extract & evaluate one FuzzySignatureLayer's CR pair ─────────


def evaluate_cr_pair(
    fuzzy_layer, *, n_eval: int = 200,
) -> dict[str, np.ndarray]:
    """Evaluate the learnable CR pair (μ⁺, μ⁻, gate, mix, hesitancy)
    on a dense x-grid in fuzzy input space [0, 1]. Returns numpy
    arrays for plotting.

    ``fuzzy_layer`` is a ``FuzzySignatureLayer`` (single-arity) or
    one of the sub-layers inside a ``MultiArityFuzzySignatureLayer``.
    """
    device = fuzzy_layer.mu_plus.cpts.device
    # x in fuzzy input space [0, 1]. The layer's forward rescales to
    # [-3, 3] when cr_input_scale="unit_to_grid".
    x_fuzzy = torch.linspace(0.0, 1.0, n_eval, device=device)
    if fuzzy_layer.cr_input_scale == "unit_to_grid":
        x_hat = 6.0 * x_fuzzy - 3.0
    else:
        x_hat = x_fuzzy
    # Reshape to (1, n_eval, d) so CRActivation's per-channel forward
    # broadcasts across the eval dim. Use channel d=0 and replicate.
    d = fuzzy_layer.d
    x_hat_broadcast = x_hat.view(1, n_eval, 1).expand(1, n_eval, d)
    with torch.no_grad():
        mu_p = fuzzy_layer.mu_plus(x_hat_broadcast, branch_idx=0)
        mu_n = fuzzy_layer.mu_minus(x_hat_broadcast, branch_idx=0)
        tau = fuzzy_layer.tau                              # (d,)
        g = torch.sigmoid(tau * (x_fuzzy.view(-1, 1) - fuzzy_layer.c_gate))
        # g: (n_eval, d). mu_p/mu_n: (1, n_eval, d). Squeeze and align.
        mu_p_e = mu_p.squeeze(0)                           # (n_eval, d)
        mu_n_e = mu_n.squeeze(0)                           # (n_eval, d)
        mu_mix = g * mu_p_e + (1.0 - g) * mu_n_e           # (n_eval, d)
    return {
        "x_fuzzy": x_fuzzy.detach().cpu().numpy(),
        "mu_plus":   mu_p_e.detach().cpu().numpy(),        # (n_eval, d)
        "mu_minus":  mu_n_e.detach().cpu().numpy(),        # (n_eval, d)
        "gate":      g.detach().cpu().numpy(),             # (n_eval, d)
        "mu_mix":    mu_mix.detach().cpu().numpy(),        # (n_eval, d)
        "tau":       tau.detach().cpu().numpy(),           # (d,)
        "c_gate":    fuzzy_layer.c_gate.detach().cpu().numpy(),  # (d,)
    }


def get_first_fuzzy_layer(layer):
    """A FuzzySignatureClassifier's `layers` list contains either
    plain ``FuzzySignatureLayer`` (single-arity) or
    ``MultiArityFuzzySignatureLayer`` (multi-arity). For multi-arity
    the per-arity ``FuzzySignatureLayer`` instances live in
    ``layer.layers``. We pick the first one to extract the CR pair."""
    if hasattr(layer, "mu_plus"):
        return layer  # single-arity
    return layer.layers[0]


# ─── Plot per-layer grid of CR curves ──────────────────────────────


def plot_layer_grid(
    layers_data: list[dict],
    *, channels: list[int],
    out_path: Path,
) -> None:
    """Multi-panel: rows=layer index, cols=selected channel index.
    Each panel shows μ⁺, μ⁻, gate g, mix μ on x ∈ [0,1] + light
    π = 1-μ hesitancy fill."""
    n_rows = len(layers_data)
    n_cols = len(channels)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.0 * n_rows),
        squeeze=False, sharex=True, sharey=True,
    )
    for r, layer_data in enumerate(layers_data):
        x = layer_data["x_fuzzy"]
        for c_idx, ch in enumerate(channels):
            ax = axes[r][c_idx]
            mu_p = layer_data["mu_plus"][:, ch]
            mu_n = layer_data["mu_minus"][:, ch]
            g_   = layer_data["gate"][:, ch]
            mu   = layer_data["mu_mix"][:, ch]
            pi   = 1.0 - mu  # mix-based hesitancy proxy
            # Hesitancy as filled background (gray, low alpha).
            ax.fill_between(x, 0, pi, color="gray", alpha=0.15,
                              label=r"$\pi=1-\mu$" if (r == 0 and c_idx == 0) else None)
            # μ⁺, μ⁻, gate, mix curves.
            ax.plot(x, mu_p, color="#d23838", linewidth=1.8,
                      label=r"$\mu^+$" if (r == 0 and c_idx == 0) else None)
            ax.plot(x, mu_n, color="#2f6ec9", linewidth=1.8,
                      label=r"$\mu^-$" if (r == 0 and c_idx == 0) else None)
            ax.plot(x, g_, color="#2aa14a", linewidth=1.2,
                      linestyle="--",
                      label=r"$g(x)$" if (r == 0 and c_idx == 0) else None)
            ax.plot(x, mu, color="#7e3f99", linewidth=2.2,
                      label=r"$\mu(x)$" if (r == 0 and c_idx == 0) else None)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle=":", alpha=0.4)
            if r == 0:
                ax.set_title(f"channel {ch}", fontsize=9)
            if c_idx == 0:
                tau_val = float(layer_data["tau"][ch])
                c_val = float(layer_data["c_gate"][ch])
                ax.set_ylabel(f"layer {r}\nτ_ch0={tau_val:.2f}",
                                fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)
    # Shared legend in the upper-right corner of fig.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center",
            ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.005),
        )
    fig.suptitle(
        "Learned Catmull-Rom splines per (layer, channel) — "
        "Atanassov μ⁺/μ⁻ pair + Zadeh hedge gate + IFS mix",
        fontsize=11, fontweight="bold", y=1.02,
    )
    fig.text(0.5, -0.01, "fuzzy input $x \\in [0, 1]$",
              ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_init_vs_trained(
    init_layer_data: dict, trained_layer_data: dict,
    *, channel: int, out_path: Path,
) -> None:
    """Before/after-training comparison for one channel of layer 0."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, (data, title) in zip(axes, [
        (init_layer_data,    "Before training (ramp init)"),
        (trained_layer_data, "After training"),
    ]):
        x = data["x_fuzzy"]
        mu_p = data["mu_plus"][:, channel]
        mu_n = data["mu_minus"][:, channel]
        g_ = data["gate"][:, channel]
        mu = data["mu_mix"][:, channel]
        pi = 1.0 - mu
        ax.fill_between(x, 0, pi, color="gray", alpha=0.15,
                          label=r"$\pi$")
        ax.plot(x, mu_p, color="#d23838", linewidth=1.8, label=r"$\mu^+$")
        ax.plot(x, mu_n, color="#2f6ec9", linewidth=1.8, label=r"$\mu^-$")
        ax.plot(x, g_, color="#2aa14a", linewidth=1.2,
                  linestyle="--", label=r"$g(x)$")
        ax.plot(x, mu, color="#7e3f99", linewidth=2.2, label=r"$\mu(x)$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$x \in [0, 1]$")
        ax.grid(True, linestyle=":", alpha=0.4)
    axes[0].set_ylabel("membership")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Layer 0, channel {channel} — what training learned",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--n-epochs", type=int, default=20)
    ap.add_argument("--channels", type=int, nargs="+",
                    default=[0, 1, 2, 3])
    ap.add_argument("--n-layers-show", type=int, default=4,
                    help="How many layers of the L=8 stack to plot.")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cr] device={device}", flush=True)

    # 1. Build init model (for before-training comparison).
    print("[cr] building fresh init model for comparison", flush=True)
    init_model = FuzzySignaturePoseModel(
        H=H, W=W, n_keypoints=N_KP, d=16, n_layers=8,
        arities=[(3, 1), (5, 2)],
    ).to(device)
    init_layer0_data = evaluate_cr_pair(
        get_first_fuzzy_layer(init_model.layers[0]),
    )

    # 2. Train (or load) the model and extract per-layer CR pairs.
    trained_model = get_model(
        device, n_epochs=args.n_epochs,
        force_retrain=args.force_retrain,
    )
    n_layers_show = min(args.n_layers_show, trained_model.n_layers)
    layers_data = []
    for li in range(n_layers_show):
        layer = trained_model.layers[li]
        fuzzy_sub = get_first_fuzzy_layer(layer)
        data = evaluate_cr_pair(fuzzy_sub)
        layers_data.append(data)

    # 3. Plot per-layer grid.
    grid_path = OUT_DIR / "cr_per_layer_grid.png"
    plot_layer_grid(layers_data, channels=args.channels,
                       out_path=grid_path)
    print(f"[cr] per-layer grid → {grid_path}", flush=True)

    # 4. Before vs after for channel 0.
    init_vs_trained_path = OUT_DIR / "cr_init_vs_trained.png"
    plot_init_vs_trained(
        init_layer0_data, layers_data[0],
        channel=0, out_path=init_vs_trained_path,
    )
    print(f"[cr] init-vs-trained → {init_vs_trained_path}", flush=True)

    # 5. Dump numerical data for downstream analysis.
    json_path = OUT_DIR / "cr_data.json"
    serializable = {
        "init_layer0": {
            "x_fuzzy": init_layer0_data["x_fuzzy"].tolist(),
            "channel_0": {
                "mu_plus":  init_layer0_data["mu_plus"][:, 0].tolist(),
                "mu_minus": init_layer0_data["mu_minus"][:, 0].tolist(),
                "gate":     init_layer0_data["gate"][:, 0].tolist(),
                "mu_mix":   init_layer0_data["mu_mix"][:, 0].tolist(),
                "tau":      float(init_layer0_data["tau"][0]),
                "c_gate":   float(init_layer0_data["c_gate"][0]),
            },
        },
        "trained_layers": [
            {
                "layer_idx": li,
                "x_fuzzy": d["x_fuzzy"].tolist(),
                "channels": [
                    {
                        "channel_idx": ch,
                        "mu_plus":  d["mu_plus"][:, ch].tolist(),
                        "mu_minus": d["mu_minus"][:, ch].tolist(),
                        "gate":     d["gate"][:, ch].tolist(),
                        "mu_mix":   d["mu_mix"][:, ch].tolist(),
                        "tau":      float(d["tau"][ch]),
                        "c_gate":   float(d["c_gate"][ch]),
                    }
                    for ch in args.channels
                ],
            }
            for li, d in enumerate(layers_data)
        ],
    }
    with json_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[cr] numerical data → {json_path}", flush=True)
    print(f"[cr] done. All outputs in {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
