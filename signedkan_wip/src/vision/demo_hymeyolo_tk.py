"""Tkinter + matplotlib GUI for HyMeYOLO detection demo.

Loads a trained HyMeYOLO checkpoint (or trains a quick model on the
fly) and shows live detection on Cluttered MNIST images.

The inference + rendering core (``load_or_train``, ``predict``,
``render_axes``) lives in :mod:`signedkan_wip.src.vision.demo_hymeyolo_core`
and is backend-agnostic. This module is the **interactive** entry point;
for non-interactive seminar slide capture (static panels + an honest
mAP/latency line) use::

    python -m signedkan_wip.src.vision.demo_hymeyolo_core \
        --checkpoint <ckpt> --headless 6

Run the GUI:
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
import tkinter as tk
from tkinter import ttk

import torch

# This module owns the interactive GUI; pin the Tk backend here. The
# core module deliberately forces no backend so it stays headless-safe.
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from signedkan_wip.src.vision.cluttered_mnist import (
    make_cluttered_mnist_hungarian_format,
)
from signedkan_wip.src.vision.demo_hymeyolo_core import (
    load_or_train,
    predict,
    render_axes,
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
