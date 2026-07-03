"""Backend-agnostic core for the HyMeYOLO detection demo.

This module holds the inference + rendering logic shared by the live
tkinter GUI (``demo_hymeyolo_tk``) and the non-interactive
slide-capture path (``render_headless`` / ``main`` here). It imports
**no** tkinter and forces **no** matplotlib GUI backend, so it runs on
a headless machine (CI, slide capture) without a display.

Two entry points, one responsibility each (CLAUDE.md §6.5 #8 —
structural difference → separate path):

* ``demo_hymeyolo_tk``   — interactive GUI (event loop).
* ``demo_hymeyolo_core`` — batch capture: render N ground-truth /
  prediction panels to ``demo_out/yolo/`` and print one honest
  ``mAP_50 + latency + RSS`` line for the seminar deck.

    python -m hymeko_neuro.experiments.vision.demo_hymeyolo_core \
        --checkpoint checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt \
        --headless 6 --n-eval 200

All metrics come from the corrected COCO matcher
``train_circles_ricci.compute_detection_metrics`` (consumed-GT
greedy IoU matching); this module does not reimplement mAP.

Plan: docs/plans/2026-06-10-hymeyolo-headless-capture/.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Backend-agnostic matplotlib only: Figure + Agg canvas, never pyplot,
# never matplotlib.use(). The GUI module pins TkAgg for itself.
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

from hymeko_neuro.experiments.vision.cluttered_mnist import (
    make_cluttered_mnist_hungarian_format,
)
from hymeko_neuro.experiments.vision.hymeyolo_circles_ricci import (
    RicciHyMeYOLOMulti,
)
from hymeko_neuro.experiments.vision.train_circles_ricci import (
    compute_detection_metrics,
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

    Preconditions: if `checkpoint` is given it must be a
    `RicciHyMeYOLOMulti` checkpoint; otherwise `n_train_quick > 0`.
    Postconditions: returned model is in `.eval()` mode on `device`.
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
    from hymeko_neuro.experiments.vision.train_circles_ricci import (
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
        from hymeko_neuro.experiments.vision.hymeyolo_warmstart import (
            warmstart_query_corners,
        )
        warmstart_query_corners(
            model, X[:min(128, n_train_quick)], seed=0,
        )
        print("[quick-train] warm-start applied", file=sys.stderr)
    except Exception as e:  # noqa: BLE001 - warm-start is best-effort
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
        ax.set_xticks([])
        ax.set_yticks([])

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


# ─── Headless slide capture ───────────────────────────────────────────


@dataclass(frozen=True)
class HeadlessResult:
    """Outcome of one headless capture run (see `render_headless`)."""
    mAP_50: float
    mAP_50_95: float
    mean_iou_matched: float
    fwd_ms_median: float
    fwd_ms_iqr: float
    fwd_ms_worst: float
    n_panels: int
    n_eval: int
    peak_rss_mb: float
    wall_s: float
    label: str
    panel_paths: list[Path] = field(default_factory=list)


def _peak_rss_mb() -> float:
    """Peak resident-set (working-set) size of this process, in MB.

    Windows: `GetProcessMemoryInfo.PeakWorkingSetSize` via stdlib
    `ctypes` (no psutil dependency). POSIX: `resource.getrusage`
    (`ru_maxrss`, KB on Linux / bytes on macOS). Returns -1.0 if the
    platform query fails. No external dependency is introduced.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # argtypes/restype are mandatory: without them the 64-bit
        # process handle is truncated to int and the call fails.
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_mem = psapi.GetProcessMemoryInfo
        get_mem.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        get_mem.restype = wintypes.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        ok = get_mem(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb,
        )
        if not ok:
            return -1.0
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)

    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return ru / (1024.0 * 1024.0)   # bytes → MB
        return ru / 1024.0                  # KB → MB (Linux)
    except Exception:  # noqa: BLE001 - RSS is diagnostic, never fatal
        return -1.0


@torch.no_grad()
def _measure_fwd_ms(
    model, img: torch.Tensor, device: str,
    reps: int = 7, warmup: int = 2,
) -> dict:
    """Median / IQR / worst single-image forward latency in ms.

    Preconditions: `reps >= 5` after warmup (CLAUDE.md §3 benchmark
    stability). Postconditions: all three returned values are > 0.
    """
    assert reps >= 5, "benchmark requires >= 5 post-warmup reps"
    model.eval()
    x = img.unsqueeze(0).to(device)
    for _ in range(warmup):
        model(x)
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        model(x)
        samples.append((time.perf_counter() - t) * 1e3)
    samples.sort()
    n = len(samples)
    median = samples[n // 2]
    q1 = samples[n // 4]
    q3 = samples[(3 * n) // 4]
    return dict(median_ms=median, iqr_ms=q3 - q1, worst_ms=samples[-1])


def _save_panel(
    img_np: np.ndarray,
    gt_boxes: np.ndarray, gt_classes: np.ndarray, gt_count: int,
    pred: dict, score_threshold: float,
    path: Path, canvas_px: int = 64,
) -> Path:
    """Render one GT/pred panel to `path` via an Agg figure (no GUI)."""
    fig = Figure(figsize=(9, 4.5), dpi=100)
    FigureCanvasAgg(fig)  # attach Agg backend so savefig works headless
    ax_in = fig.add_subplot(1, 2, 1)
    ax_pred = fig.add_subplot(1, 2, 2)
    render_axes(
        ax_in, ax_pred, img_np, gt_boxes, gt_classes, gt_count,
        pred, score_threshold, canvas_px=canvas_px,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def render_headless(
    checkpoint: str | None,
    n_panels: int,
    n_eval: int,
    out_dir: Path,
    device: str = "cpu",
    seed: int = 0,
    score_threshold: float = 0.30,
    canvas_px: int = 64,
) -> HeadlessResult:
    """Batch slide-capture: load a checkpoint, render `n_panels` GT/pred
    panels to `out_dir`, and report an honest mAP + latency + RSS line.

    Preconditions:
        * `checkpoint` is a valid path to a `RicciHyMeYOLOMulti` `.pt`
          (headless never quick-trains — a demo number must come from a
          named checkpoint).
        * `n_panels >= 1`, `n_eval >= 1`.
    Postconditions:
        * exactly `n_panels` non-empty PNGs exist under `out_dir`;
        * peak RSS < 16 GB (raises `RuntimeError` otherwise);
        * returned `HeadlessResult` mirrors the printed line.
    """
    if checkpoint is None or not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"headless capture requires a valid --checkpoint; got "
            f"{checkpoint!r}. The demo number must name a checkpoint."
        )
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1; got {n_panels}")
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, meta = load_or_train(checkpoint, 0, device)

    n_eval = max(int(n_eval), n_panels)
    Xn, boxes_n, classes_n, counts_n = make_cluttered_mnist_hungarian_format(
        n=n_eval, canvas=canvas_px, max_objects=3, seed=seed, rgb=True,
    )
    metrics = compute_detection_metrics(
        model,
        torch.from_numpy(Xn), torch.from_numpy(boxes_n),
        torch.from_numpy(classes_n), torch.from_numpy(counts_n),
        n_classes=10, batch_size=32,
    )

    panel_paths: list[Path] = []
    for i in range(n_panels):
        img = torch.from_numpy(Xn[i])
        pred = predict(model, img, device=device)
        path = out_dir / f"panel_{i:02d}_seed{seed}.png"
        _save_panel(
            Xn[i], boxes_n[i], classes_n[i], int(counts_n[i]),
            pred, score_threshold, path, canvas_px=canvas_px,
        )
        panel_paths.append(path)

    lat = _measure_fwd_ms(model, torch.from_numpy(Xn[0]), device)
    peak_rss = _peak_rss_mb()
    wall = time.perf_counter() - t0

    result = HeadlessResult(
        mAP_50=float(metrics["mAP_50"]),
        mAP_50_95=float(metrics["mAP_50_95"]),
        mean_iou_matched=float(metrics["mean_iou_matched"]),
        fwd_ms_median=lat["median_ms"],
        fwd_ms_iqr=lat["iqr_ms"],
        fwd_ms_worst=lat["worst_ms"],
        n_panels=n_panels,
        n_eval=n_eval,
        peak_rss_mb=peak_rss,
        wall_s=wall,
        label=str(meta.get("label", "?")),
        panel_paths=panel_paths,
    )

    print(
        f"[headless] mAP_50={result.mAP_50:.3f} "
        f"mAP_50_95={result.mAP_50_95:.3f} "
        f"mean_iou={result.mean_iou_matched:.3f} | "
        f"fwd_ms median={result.fwd_ms_median:.1f} "
        f"iqr={result.fwd_ms_iqr:.1f} worst={result.fwd_ms_worst:.1f} | "
        f"peak_rss={result.peak_rss_mb:.0f}MB wall={result.wall_s:.1f}s | "
        f"panels={n_panels} eval={n_eval} "
        f"ckpt={result.label} (epochs={meta.get('epochs')})"
    )
    for p in panel_paths:
        print(f"[headless] wrote {p}")

    if peak_rss >= 0 and peak_rss >= 16_000:
        raise RuntimeError(
            f"peak RSS {peak_rss:.0f} MB exceeds the 16 GB cap "
            f"(CLAUDE.md §4)"
        )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HyMeYOLO headless slide-capture: render N GT/pred "
                    "panels + one honest mAP/latency/RSS line.",
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to a RicciHyMeYOLOMulti .pt checkpoint. "
                         "Headless never quick-trains.")
    ap.add_argument("--headless", type=int, default=6,
                    help="Number of GT/pred panels to render (default 6).")
    ap.add_argument("--n-eval", type=int, default=200,
                    help="Images used for the mAP estimate (default 200).")
    ap.add_argument("--out-dir", default="demo_out/yolo",
                    help="Output directory for panels (default demo_out/yolo).")
    ap.add_argument("--device", default="cpu",
                    help="Inference device (cpu / cuda). Default cpu.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Deterministic stimulus seed (default 0).")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Score threshold for drawn predictions (default 0.30).")
    args = ap.parse_args()

    render_headless(
        checkpoint=args.checkpoint,
        n_panels=args.headless,
        n_eval=args.n_eval,
        out_dir=Path(args.out_dir),
        device=args.device,
        seed=args.seed,
        score_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
