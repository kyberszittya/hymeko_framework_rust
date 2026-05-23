"""Tkinter + matplotlib GUI for HyMeYOLO detection demo.

Loads a trained HyMeYOLO checkpoint (or trains a quick model on the
fly) and shows live detection on Cluttered MNIST images.

Run:
    python -m signedkan_wip.src.vision.demo_hymeyolo_tk
        [--checkpoint <path>]
        [--n-train-quick <int>]   # if no checkpoint, how many images
                                    # to train on quickly (default 1000)

What you see:
    * Left panel: the input cluttered-MNIST image.
    * Right panel: same image with the model's predicted boxes /
      circles overlaid, labelled with the top-1 class + confidence.
    * Bottom controls:
        - "New random image"  generates a fresh stimulus.
        - Score threshold slider (0..1) filters low-confidence
          predictions.
        - Stage label shows which checkpoint is loaded.

Architecture notes:
    * No streamlit/gradio dependency — pure stdlib Tk + matplotlib
      embedded via FigureCanvasTkAgg.
    * Inference is CPU by default; if the checkpoint was trained on
      CUDA and the GPU is free, pass `--device cuda` to inference
      there too (default `cpu` so the demo is always responsive).
"""
from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import torch
import torch.nn.functional as F

# Matplotlib uses the TkAgg backend; pin it before importing pyplot.
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from signedkan_wip.src.vision.cluttered_mnist import (
    make_cluttered_mnist_hungarian_format,
)
from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
    RicciHyMeYOLOMulti,
)


# ─── Model load / quick-train ─────────────────────────────────────────


def load_or_train(
    checkpoint: str | None,
    n_train_quick: int,
    device: str,
) -> tuple[torch.nn.Module, dict]:
    """Return (model, meta_dict).

    If `checkpoint` is a valid path, load it and return. Otherwise
    train a fresh small `RicciHyMeYOLOMulti` for a short time on a
    small Cluttered MNIST split — enough to produce visible
    detections.
    """
    if checkpoint is not None and os.path.isfile(checkpoint):
        print(f"[load] {checkpoint}", file=sys.stderr)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cls_name = ckpt.get("model_class", "RicciHyMeYOLOMulti")
        if cls_name != "RicciHyMeYOLOMulti":
            raise NotImplementedError(
                f"this demo only supports RicciHyMeYOLOMulti checkpoints; "
                f"got {cls_name}"
            )
        backbone = ckpt.get("backbone", "tiny")
        fpn = ckpt.get("fpn", "none")
        model = RicciHyMeYOLOMulti(
            n_box_queries=4, n_circle_queries=2, circle_k=8,
            n_classes=10, d_hidden=32,
            ricci_modulation=True,
            ricci_scale=float(ckpt.get("ricci_scale", 1.0)),
            use_layernorm=bool(ckpt.get("use_layernorm", False)),
            backbone=backbone,
            fpn=fpn,
        )
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()
        return model, dict(
            source="checkpoint",
            path=checkpoint,
            label=ckpt.get("label", "?"),
            epochs=ckpt.get("epochs", "?"),
            ricci_scale=ckpt.get("ricci_scale", 1.0),
            schedule=ckpt.get("schedule", "?"),
            warm_start=ckpt.get("warm_start", "?"),
            backbone=backbone,
            fpn=fpn,
        )

    # No checkpoint — quick train.
    print(f"[quick-train] training fresh model on {n_train_quick} images "
          f"for 30 epochs (device={device})", file=sys.stderr)
    from signedkan_wip.src.vision.train_circles_ricci import (
        combined_set_loss,
    )
    Xn, boxes_n, classes_n, counts_n = make_cluttered_mnist_hungarian_format(
        n=n_train_quick, canvas=64, max_objects=3, seed=0, rgb=True,
    )
    X = torch.from_numpy(Xn).to(device)
    boxes = torch.from_numpy(boxes_n).to(device)
    classes = torch.from_numpy(classes_n).to(device)
    counts = torch.from_numpy(counts_n).to(device)

    torch.manual_seed(0)
    model = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
    ).to(device)

    # Saliency-driven warm-start (the Stage A-1 lever).
    try:
        from signedkan_wip.src.vision.hymeyolo_warmstart import (
            warmstart_query_corners,
        )
        warmstart_query_corners(
            model, X[:min(128, n_train_quick)], seed=0,
        )
        print("[quick-train] warm-start applied", file=sys.stderr)
    except Exception as e:
        print(f"[quick-train] warm-start skipped: {e}", file=sys.stderr)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    bs = 32
    n = X.shape[0]
    for ep in range(30):
        perm = torch.randperm(n, device=device)
        ep_losses = []
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb, bb, cb, kb = X[idx], boxes[idx], classes[idx], counts[idx]
            pred = model(xb)
            loss, _ = combined_set_loss(pred, bb, cb, kb, n_classes=10)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_losses.append(float(loss.detach()))
        if (ep + 1) % 5 == 0:
            mean_loss = sum(ep_losses) / max(1, len(ep_losses))
            print(f"  ep {ep+1:2d}/30  loss={mean_loss:.4f}",
                  file=sys.stderr)

    model.eval()
    return model, dict(
        source="quick-train",
        path="(none)",
        label="+ricci-mod (quick)",
        epochs=30,
        ricci_scale=1.0,
        schedule="constant",
        warm_start=True,
    )


# ─── Inference + visualisation ─────────────────────────────────────────


def _aabb_from_corners(corners: torch.Tensor) -> torch.Tensor:
    """(M, k, 2) → (M, 4) AABB (x0, y0, x1, y1)."""
    return torch.stack([
        corners[..., 0].min(dim=-1).values,
        corners[..., 1].min(dim=-1).values,
        corners[..., 0].max(dim=-1).values,
        corners[..., 1].max(dim=-1).values,
    ], dim=-1)


@torch.no_grad()
def predict(model, img: torch.Tensor, device: str) -> dict:
    """One-image inference. Returns a dict with per-query AABB + class
    + score for box queries and circle queries separately, plus the
    raw corner positions for visualisation."""
    model.eval()
    x = img.unsqueeze(0).to(device)
    out = model(x)
    box_corners = out["box_corners"][0]      # (Nb, 4, 2)
    box_logits = out["box_cls"][0]           # (Nb, C+1)
    circ_corners = out["circle_corners"][0]  # (Nc, k, 2)
    circ_logits = out["circle_cls"][0]       # (Nc, C+1)
    n_classes = box_logits.shape[-1] - 1     # last slot = no-object

    def _decode(corners, logits):
        if corners.numel() == 0:
            return dict(aabb=torch.zeros(0, 4), score=torch.zeros(0),
                        cls=torch.zeros(0, dtype=torch.long))
        probs = F.softmax(logits, dim=-1)
        obj = probs[:, :n_classes]
        score, cls = obj.max(dim=-1)
        aabb = _aabb_from_corners(corners).clamp(0.0, 1.0)
        return dict(
            aabb=aabb.cpu(), score=score.cpu(),
            cls=cls.cpu(), corners=corners.cpu(),
        )

    return dict(
        box=_decode(box_corners, box_logits),
        circle=_decode(circ_corners, circ_logits),
    )


def render_axes(
    ax_input, ax_pred,
    img_np: np.ndarray,
    gt_boxes: np.ndarray, gt_classes: np.ndarray, gt_count: int,
    pred: dict,
    score_threshold: float,
    canvas_px: int = 64,
) -> None:
    """Render the input image + the prediction overlay on a (1, 2) figure."""
    for ax in (ax_input, ax_pred):
        ax.clear()
        ax.set_xlim(0, canvas_px)
        ax.set_ylim(canvas_px, 0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    # Input image (HWC, [0, 1]).
    img_show = np.transpose(img_np, (1, 2, 0))
    img_show = (img_show - img_show.min()) / max(
        1e-9, img_show.max() - img_show.min(),
    )
    ax_input.imshow(img_show)
    ax_input.set_title("Input — Cluttered MNIST")

    # GT boxes on input panel (faint cyan).
    for gi in range(gt_count):
        x0, y0, x1, y1 = (gt_boxes[gi] * canvas_px).astype(int)
        ax_input.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.2, edgecolor="cyan", facecolor="none", alpha=0.6,
        ))
        ax_input.text(
            x0, y0 - 1, f"GT {int(gt_classes[gi])}",
            fontsize=7, color="cyan",
        )

    # Predictions on right panel.
    ax_pred.imshow(img_show)
    ax_pred.set_title(
        f"Predictions  (threshold = {score_threshold:.2f})"
    )

    # Box queries (red); circle queries (orange).
    for kind, color in (("box", "red"), ("circle", "orange")):
        d = pred[kind]
        if len(d["score"]) == 0:
            continue
        for i in range(len(d["score"])):
            s = float(d["score"][i])
            if s < score_threshold:
                continue
            x0, y0, x1, y1 = (d["aabb"][i].numpy() * canvas_px)
            ax_pred.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.4, edgecolor=color, facecolor="none",
            ))
            cls_id = int(d["cls"][i])
            ax_pred.text(
                x0, y0 - 1.5,
                f"{kind[0]}{i}: {cls_id} ({s:.2f})",
                fontsize=7, color=color,
                bbox=dict(facecolor="white", alpha=0.4, pad=0.5,
                          edgecolor="none"),
            )


# ─── Tk app ───────────────────────────────────────────────────────────


class HyMeYOLODemoApp(tk.Tk):
    def __init__(self, model, meta: dict, device: str):
        super().__init__()
        self.title("HyMeYOLO detection demo — Cluttered MNIST")
        self.geometry("1000x700")
        self.model = model
        self.device = device
        self.meta = meta
        self.canvas_px = 64
        self.score_thr = tk.DoubleVar(value=0.30)
        self.current_seed = 0

        self._build_ui()
        self._refresh()

    def _build_ui(self):
        # Header — model metadata.
        header_frame = ttk.Frame(self)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        meta_text = (
            f"model: {self.meta.get('label', '?')}  |  "
            f"source: {self.meta.get('source', '?')}  |  "
            f"epochs: {self.meta.get('epochs', '?')}  |  "
            f"schedule: {self.meta.get('schedule', '?')}  |  "
            f"warm-start: {self.meta.get('warm_start', '?')}  |  "
            f"backbone: {self.meta.get('backbone', 'tiny')}  |  "
            f"fpn: {self.meta.get('fpn', 'none')}  |  "
            f"device: {self.device}"
        )
        ttk.Label(header_frame, text=meta_text,
                   font=("Helvetica", 9)).pack(anchor=tk.W)

        # Matplotlib figure.
        self.fig = Figure(figsize=(9, 4.5), dpi=100)
        self.ax_input = self.fig.add_subplot(1, 2, 1)
        self.ax_pred = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH,
                                          expand=True, padx=8, pady=4)

        # Controls.
        ctrl_frame = ttk.Frame(self)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        ttk.Button(ctrl_frame, text="New random image",
                    command=self._next_image).pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl_frame, text="Score threshold:").pack(
            side=tk.LEFT, padx=(16, 4))
        ttk.Scale(ctrl_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.score_thr, length=200,
                  command=lambda _e: self._redraw_only()
                 ).pack(side=tk.LEFT, padx=4)
        self.thr_label = ttk.Label(ctrl_frame, text=f"{self.score_thr.get():.2f}")
        self.thr_label.pack(side=tk.LEFT)

        # Seed control.
        ttk.Label(ctrl_frame, text="  seed:").pack(side=tk.LEFT, padx=(16, 2))
        self.seed_var = tk.IntVar(value=self.current_seed)
        ttk.Spinbox(ctrl_frame, from_=0, to=99999, width=6,
                     textvariable=self.seed_var,
                     command=self._refresh_from_seed
                    ).pack(side=tk.LEFT, padx=2)

        # Status bar.
        self.status_text = tk.StringVar(value="ready")
        ttk.Label(self, textvariable=self.status_text,
                   relief=tk.SUNKEN, anchor=tk.W,
                   font=("Helvetica", 9)
                  ).pack(side=tk.BOTTOM, fill=tk.X)

    def _next_image(self):
        self.current_seed = int(self.seed_var.get()) + 1
        self.seed_var.set(self.current_seed)
        self._refresh()

    def _refresh_from_seed(self):
        self.current_seed = int(self.seed_var.get())
        self._refresh()

    def _refresh(self):
        """Generate one new cluttered-MNIST image at the current seed,
        run inference, render."""
        self.status_text.set(f"generating image at seed={self.current_seed}…")
        self.update_idletasks()
        Xn, boxes_n, classes_n, counts_n = (
            make_cluttered_mnist_hungarian_format(
                n=1, canvas=self.canvas_px, max_objects=3,
                seed=self.current_seed, rgb=True,
            )
        )
        img = torch.from_numpy(Xn[0])
        self._gt_boxes = boxes_n[0]
        self._gt_classes = classes_n[0]
        self._gt_count = int(counts_n[0])
        self._img_np = Xn[0]

        self.status_text.set("running inference…")
        self.update_idletasks()
        self._pred = predict(self.model, img, device=self.device)

        self._redraw_only()
        self.status_text.set(
            f"seed={self.current_seed}  gt_count={self._gt_count}  "
            f"n_box_preds_above={int((self._pred['box']['score'] >= self.score_thr.get()).sum())}  "
            f"n_circle_preds_above={int((self._pred['circle']['score'] >= self.score_thr.get()).sum())}"
        )

    def _redraw_only(self):
        """Re-render with the current threshold but cached image +
        prediction. Used when the threshold slider moves."""
        if not hasattr(self, "_pred"):
            return
        thr = float(self.score_thr.get())
        self.thr_label.config(text=f"{thr:.2f}")
        render_axes(
            self.ax_input, self.ax_pred,
            self._img_np, self._gt_boxes, self._gt_classes, self._gt_count,
            self._pred, thr, canvas_px=self.canvas_px,
        )
        self.fig.tight_layout()
        self.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                     help="Path to a .pt file from train_circles_ricci "
                          "--save-checkpoint. If omitted, the GUI trains "
                          "a fresh quick model on launch (~30 s on CPU).")
    ap.add_argument("--n-train-quick", type=int, default=1000,
                     help="If no checkpoint, train on this many images "
                          "for 30 epochs. Default 1000.")
    ap.add_argument("--device", default="cpu",
                     help="Inference device (cpu / cuda). Default cpu so "
                          "the demo doesn't contend with concurrent "
                          "training runs.")
    args = ap.parse_args()

    model, meta = load_or_train(
        args.checkpoint, args.n_train_quick, args.device,
    )
    app = HyMeYOLODemoApp(model, meta, device=args.device)
    app.mainloop()


if __name__ == "__main__":
    main()
