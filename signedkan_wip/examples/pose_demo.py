"""Pose-detection example application using FuzzySignaturePoseModel.

End-to-end demonstration of the framework on a robotic-pose-like task:
  1. Train a small (9.6k params) FuzzySignaturePoseModel on the
     synthetic kinematic-chain dataset.
  2. Run inference on held-out test samples.
  3. Visualize the result with FOUR diagnostic panels per sample:
       - Input image (Gaussian-blob "skeleton").
       - Predicted keypoints overlaid on the image (filled circles)
         + ground-truth keypoints (hollow circles for direct comparison).
       - 8-panel per-keypoint heatmap grid (one per keypoint).
       - Per-keypoint spatial entropy / uncertainty score (bar plot).
  4. Save the composite figure as `pose_demo_output_<sample>.png`.

The figure communicates exactly what the framework provides over a
pure-coordinate output: spatial uncertainty *per keypoint*, derived
from the heatmap softmax entropy. This is the kind of confidence
signal a robot collaborator needs (e.g., "I'm sure about the left
shoulder, uncertain about the right hand — don't act on the right
hand prediction").

The model is trained once and cached to `/tmp/fuzzy_pose_demo_*.pt`
for subsequent runs (~3 min first time, ~3 s afterward).

Run:
    cd /home/kyberszittya/hakiko-ws/hymeko/hymeko_framework_rust
    PYTHONPATH=. python signedkan_wip/examples/pose_demo.py

Output:
    /tmp/pose_demo_outputs/sample_{0..7}.png        # per-sample 4-panel
    /tmp/pose_demo_outputs/summary.png              # composite of all 8
    /tmp/pose_demo_outputs/uncertainty_table.txt    # text-mode summary
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from signedkan_wip.src.vision.fuzzy_pose import (
    FuzzySignaturePoseModel,
    SyntheticPoseDataset,
    soft_argmax_2d,
)


# ─── Config ────────────────────────────────────────────────────────


H = W = 32
N_KP = 8
KEYPOINT_NAMES = [
    "head", "L-shoulder", "R-shoulder",
    "L-elbow", "R-elbow",
    "L-hand", "R-hand",
    "mid-hip",
]
KEYPOINT_COLORS = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#95e1d3",
    "#f38181", "#aa96da", "#fcbad3", "#a8d8ea",
]
N_DEMO_SAMPLES = 8
CACHE_DIR = Path("/tmp/pose_demo_outputs")
MODEL_PATH = CACHE_DIR / "fuzzy_pose_model.pt"


# ─── Train (or load cached) FuzzySignaturePoseModel ────────────────


def get_or_train_model(
    device: torch.device,
    *, n_train: int = 1000, n_epochs: int = 30,
    lr: float = 5e-3, batch_size: int = 32,
    force_retrain: bool = False,
) -> FuzzySignaturePoseModel:
    """Load cached or train a small FuzzySignaturePoseModel.

    Uses a smaller config (n_train=1000, n_epochs=30) than the full
    smoke (10k×60ep) so the demo trains in ~3 min on GPU or ~10 min
    on CPU. The model is the same architecture, just less converged
    — final test_mse is around 8-10 instead of 5.5, still meaningful
    for the demo.
    """
    model = FuzzySignaturePoseModel(
        H=H, W=W, n_keypoints=N_KP, d=16, n_layers=8,
        arities=[(3, 1), (5, 2)],
    ).to(device)
    if MODEL_PATH.exists() and not force_retrain:
        print(f"[demo] loading cached model from {MODEL_PATH}", flush=True)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        return model

    print(f"[demo] training FuzzySignaturePoseModel "
          f"(n_train={n_train}, n_epochs={n_epochs})", flush=True)
    train_ds = SyntheticPoseDataset(
        n_samples=n_train, H=H, W=W, n_keypoints=N_KP, seed=0,
    )
    loader = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(n_epochs):
        running = 0.0
        n_batches = 0
        for imgs, coords in loader:
            imgs = imgs.to(device)
            coords = coords.to(device)
            pred = model(imgs)
            loss = torch.nn.functional.mse_loss(pred, coords)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        if ep in (0, 9, 19, 29):
            print(f"[demo]   ep {ep:2d} loss={running / n_batches:.3f}",
                  flush=True)
    wall = time.time() - t0
    print(f"[demo] training done in {wall:.1f}s; "
          f"caching to {MODEL_PATH}", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model


# ─── Uncertainty per keypoint (heatmap-softmax entropy) ────────────


def heatmap_entropy(heatmap: torch.Tensor, beta: float = 1.0) -> float:
    """Entropy of the softmax-normalized heatmap. High entropy = the
    model spreads probability mass across many pixels = uncertain
    about the keypoint location. Low entropy = mass concentrated at
    one pixel = confident."""
    flat = heatmap.flatten() / beta
    p = torch.softmax(flat, dim=-1)
    # Add epsilon for numerical stability.
    eps = 1e-12
    return float(-(p * (p + eps).log()).sum().item())


# ─── Plotting ──────────────────────────────────────────────────────


def plot_one_sample(
    img: torch.Tensor, gt_coords: torch.Tensor,
    pred_coords: torch.Tensor, heatmaps: torch.Tensor,
    entropies: list[float], out_path: Path,
) -> None:
    """4-panel diagnostic figure for one sample.

    Panel layout:
      ┌──────────┬─────────────────┐
      │ INPUT    │ PRED vs GT      │
      ├──────────┴─────────────────┤
      │  HEATMAPS (8 small panels)  │
      ├────────────────────────────┤
      │  PER-KEYPOINT UNCERTAINTY  │
      └────────────────────────────┘
    """
    img_np = img.cpu().squeeze().numpy()
    gt_np = gt_coords.cpu().numpy()
    pred_np = pred_coords.cpu().numpy()
    heatmaps_np = heatmaps.cpu().numpy()  # (n_kp, H, W)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 8, height_ratios=[1.4, 1.0, 0.6],
                            hspace=0.4, wspace=0.3)

    # Panel 1: input image
    ax_in = fig.add_subplot(gs[0, :4])
    ax_in.imshow(img_np, cmap="gray", origin="upper")
    ax_in.set_title("Input image (sensor frame)", fontsize=11)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

    # Panel 2: predicted vs ground-truth
    ax_pred = fig.add_subplot(gs[0, 4:])
    ax_pred.imshow(img_np, cmap="gray", alpha=0.6, origin="upper")
    for k in range(N_KP):
        ax_pred.scatter(
            gt_np[k, 0], gt_np[k, 1],
            facecolors="none", edgecolors=KEYPOINT_COLORS[k],
            s=80, linewidths=2, marker="o",
        )
        ax_pred.scatter(
            pred_np[k, 0], pred_np[k, 1],
            color=KEYPOINT_COLORS[k], s=40, marker="o",
        )
        ax_pred.plot(
            [gt_np[k, 0], pred_np[k, 0]],
            [gt_np[k, 1], pred_np[k, 1]],
            color=KEYPOINT_COLORS[k], linewidth=1, alpha=0.5,
        )
    ax_pred.set_title("Predicted (filled) vs Ground truth (hollow)", fontsize=11)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    # Panel 3: per-keypoint heatmaps (8 sub-panels)
    for k in range(N_KP):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(heatmaps_np[k], cmap="hot", origin="upper")
        ax.set_title(f"{KEYPOINT_NAMES[k]}\nH={entropies[k]:.2f}",
                       fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Panel 4: uncertainty bar plot (entropy per keypoint)
    ax_unc = fig.add_subplot(gs[2, :])
    bars = ax_unc.bar(
        range(N_KP), entropies,
        color=KEYPOINT_COLORS, edgecolor="black", linewidth=0.5,
    )
    ax_unc.set_xticks(range(N_KP))
    ax_unc.set_xticklabels(KEYPOINT_NAMES, rotation=30, ha="right", fontsize=9)
    ax_unc.set_ylabel("Heatmap entropy\n(higher = more uncertain)", fontsize=9)
    ax_unc.set_title("Per-keypoint uncertainty (softmax entropy)", fontsize=10)
    ax_unc.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "FuzzySignaturePose — pose detection with per-keypoint uncertainty",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_summary(samples_data: list[dict], out_path: Path) -> None:
    """Composite: one row per sample, three columns: input, prediction
    overlay, uncertainty bar. Tight version of plot_one_sample, 8
    rows tall."""
    fig, axes = plt.subplots(
        N_DEMO_SAMPLES, 3, figsize=(11, 2.4 * N_DEMO_SAMPLES),
        gridspec_kw={"width_ratios": [1, 1, 1.4]},
    )
    for s_idx, s in enumerate(samples_data):
        img_np = s["img"].cpu().squeeze().numpy()
        gt_np = s["gt"].cpu().numpy()
        pred_np = s["pred"].cpu().numpy()
        ent = s["entropies"]

        ax0 = axes[s_idx, 0]
        ax0.imshow(img_np, cmap="gray", origin="upper")
        ax0.set_xticks([])
        ax0.set_yticks([])
        if s_idx == 0:
            ax0.set_title("Input", fontsize=10)
        ax0.set_ylabel(f"sample {s_idx}", fontsize=9)

        ax1 = axes[s_idx, 1]
        ax1.imshow(img_np, cmap="gray", alpha=0.6, origin="upper")
        for k in range(N_KP):
            ax1.scatter(gt_np[k, 0], gt_np[k, 1],
                          facecolors="none", edgecolors=KEYPOINT_COLORS[k],
                          s=40, linewidths=1.4, marker="o")
            ax1.scatter(pred_np[k, 0], pred_np[k, 1],
                          color=KEYPOINT_COLORS[k], s=18, marker="o")
        ax1.set_xticks([])
        ax1.set_yticks([])
        if s_idx == 0:
            ax1.set_title("Pred (filled) vs GT (hollow)", fontsize=10)

        ax2 = axes[s_idx, 2]
        ax2.bar(range(N_KP), ent,
                  color=KEYPOINT_COLORS, edgecolor="black", linewidth=0.4)
        ax2.set_xticks(range(N_KP))
        ax2.set_xticklabels(KEYPOINT_NAMES,
                              rotation=30, ha="right", fontsize=7)
        ax2.set_ylim(0, max(7.0, max(ent) * 1.1))
        if s_idx == 0:
            ax2.set_title("Per-keypoint uncertainty", fontsize=10)
        ax2.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle(
        "FuzzySignaturePose — 8-sample pose demo with uncertainty",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-retrain", action="store_true",
                    help="Ignore cached model and re-train.")
    ap.add_argument("--n-epochs", type=int, default=30,
                    help="Training epochs (default 30; raise to 60 "
                         "for the gate-passing config).")
    args = ap.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] device={device}", flush=True)

    model = get_or_train_model(
        device, n_epochs=args.n_epochs,
        force_retrain=args.force_retrain,
    )
    model.eval()

    # Test set: 8 held-out samples with a different seed.
    test_ds = SyntheticPoseDataset(
        n_samples=100, H=H, W=W, n_keypoints=N_KP, seed=1,
    )
    samples_data = []
    text_lines = []
    text_lines.append(
        "FuzzySignaturePose — pose-detection example output\n"
        + "=" * 72 + "\n"
    )
    print(f"[demo] running inference on {N_DEMO_SAMPLES} samples", flush=True)
    with torch.no_grad():
        for s_idx in range(N_DEMO_SAMPLES):
            img, gt_coords = test_ds[s_idx]
            img_b = img.unsqueeze(0).to(device)            # (1, 1, H, W)
            heatmaps = model.heatmaps(img_b)               # (1, n_kp, H, W)
            pred_coords = soft_argmax_2d(heatmaps).squeeze(0)
            heatmaps_one = heatmaps.squeeze(0)             # (n_kp, H, W)
            entropies = [
                heatmap_entropy(heatmaps_one[k]) for k in range(N_KP)
            ]
            per_sample_mse = float(torch.nn.functional.mse_loss(
                pred_coords.cpu(), gt_coords,
            ).item())

            samples_data.append({
                "img": img, "gt": gt_coords,
                "pred": pred_coords, "heatmaps": heatmaps_one,
                "entropies": entropies,
                "mse": per_sample_mse,
            })

            # Per-sample figure.
            out_path = CACHE_DIR / f"sample_{s_idx}.png"
            plot_one_sample(
                img, gt_coords, pred_coords, heatmaps_one,
                entropies, out_path,
            )
            print(f"[demo]   sample {s_idx}: MSE={per_sample_mse:.3f}"
                  f"  most-uncertain={KEYPOINT_NAMES[int(np.argmax(entropies))]}"
                  f"  (H={max(entropies):.2f}); written {out_path.name}",
                  flush=True)

            # Text-mode summary line.
            text_lines.append(f"\n--- sample {s_idx} ---")
            text_lines.append(
                f"  total MSE: {per_sample_mse:.3f}"
            )
            text_lines.append("  per-keypoint uncertainty (entropy):")
            order = sorted(
                range(N_KP), key=lambda k: -entropies[k]
            )
            for k in order:
                bar = "█" * int(min(20, entropies[k] * 3))
                text_lines.append(
                    f"    {KEYPOINT_NAMES[k]:>12s}: "
                    f"{entropies[k]:5.2f}  {bar}"
                )

    summary_path = CACHE_DIR / "summary.png"
    plot_summary(samples_data, summary_path)
    print(f"[demo] summary figure → {summary_path}", flush=True)

    text_path = CACHE_DIR / "uncertainty_table.txt"
    with text_path.open("w") as f:
        f.write("\n".join(text_lines) + "\n")
    print(f"[demo] text summary → {text_path}", flush=True)

    print(f"[demo] done. All outputs in {CACHE_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
