"""Held-out evaluation + slide-panel capture for a HyMeYOLO VOC checkpoint.

`train_voc_stagec.py` trains *and* evaluates on a single split — it has
no held-out path and saves no number you can publish. This module closes
that gap: load a saved checkpoint, evaluate it on **any** VOC split
(typically `test`), and render real-photo detection panels for the deck.

Reuse (no new algorithm / decoding code):
  * `VocPersonDetector` — reconstructs `RicciHyMeYOLOMulti` from the
    checkpoint metadata and decodes the nodelet gate head correctly
    (the demo's `predict` assumes the hungarian +1 slot, so it is not
    used here);
  * `compute_detection_metrics` — the corrected consumed-GT COCO matcher;
  * `load_voc_hungarian` / `VOC_CLASSES` — the dataset loader;
  * `_peak_rss_mb` — the dependency-free peak-RSS probe.

    python -m signedkan_wip.src.vision.eval_voc \
        --checkpoint <stage_d1_voc_seed0.pt> \
        --image-set test --mode both --panels 6

Plan: docs/plans/2026-06-10-voc-test-baseline/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

from signedkan_wip.src.vision.demo_hymeyolo_core import _peak_rss_mb
from signedkan_wip.src.vision.train_circles_ricci import (
    compute_detection_metrics,
)
from signedkan_wip.src.vision.voc_dataset import (
    VOC_CLASSES, load_voc_hungarian,
)
from signedkan_wip.src.rapport_ros2.voc_detector import VocPersonDetector


def _load_split(
    image_set: str, year: str, input_size: int,
    n_images: int | None, data_root: str,
):
    """Load a VOC split in Hungarian format (X on CPU, padding-safe classes).

    Returns (X, boxes, classes, counts, Xn) where the first four are CPU
    tensors ready for `compute_detection_metrics` and `Xn` is the raw
    numpy image stack for rendering.
    """
    Xn, boxes_n, classes_n, counts_n, _names = load_voc_hungarian(
        year=year, image_set=image_set, input_size=input_size,
        max_objects=12, root=data_root, subset_n=n_images, download=True,
    )
    # Padding sentinel -1 must never index the class output; the counts
    # mask excludes padding rows from the metric, but map to 0 for safety.
    classes_safe = np.where(classes_n < 0, 0, classes_n)
    return (
        torch.from_numpy(Xn),
        torch.from_numpy(boxes_n),
        torch.from_numpy(classes_safe),
        torch.from_numpy(counts_n),
        Xn,
    )


def eval_checkpoint_on_split(
    checkpoint: str,
    image_set: str = "test",
    year: str = "2007",
    input_size: int | None = None,
    n_images: int | None = None,
    batch_size: int = 8,
    device: str = "cuda",
    data_root: str = "data/torchvision",
) -> dict:
    """Evaluate a saved VOC checkpoint on a held-out split.

    Preconditions: `checkpoint` is a valid `RicciHyMeYOLOMulti` `.pt`;
    `image_set` is a real VOC split with data available (downloads on
    demand). Postconditions: returns the `compute_detection_metrics`
    dict (`mAP_50`, `mAP_50_95`, `mean_iou_matched`, `n_*`) augmented
    with `image_set`, `n_images`, `wall_s`, `peak_rss_mb`.
    """
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint!r}")
    t0 = time.perf_counter()
    detector = VocPersonDetector(checkpoint, device=device)
    isize = int(input_size or detector.input_size)
    X, boxes, classes, counts, _Xn = _load_split(
        image_set, year, isize, n_images, data_root,
    )
    # `compute_detection_metrics` lazily streams X batch-by-batch to the
    # model's device but expects the GT tensors already there (the
    # trainer moved them up front). X stays on CPU; move the small GT
    # tensors to the model device or the IoU step mixes cuda/cpu.
    gt_device = next(detector.model.parameters()).device
    boxes = boxes.to(gt_device)
    classes = classes.to(gt_device)
    counts = counts.to(gt_device)
    metrics = compute_detection_metrics(
        detector.model, X, boxes, classes, counts,
        n_classes=detector.n_classes, batch_size=batch_size,
    )
    metrics = dict(metrics)
    metrics.update(
        image_set=image_set, n_images=int(X.shape[0]),
        input_size=isize, wall_s=time.perf_counter() - t0,
        peak_rss_mb=_peak_rss_mb(),
    )
    return metrics


def _draw_voc_panel(
    fig: Figure, img_np: np.ndarray,
    gt_boxes: np.ndarray, gt_classes: np.ndarray, gt_count: int,
    detections, class_names, canvas_px: int,
) -> None:
    """Two-panel render: GT (cyan, named) vs detections (red, named+score)."""
    img_show = np.transpose(img_np, (1, 2, 0))
    img_show = (img_show - img_show.min()) / max(
        1e-9, float(img_show.max() - img_show.min()),
    )
    ax_in, ax_pred = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    for ax, title in ((ax_in, "Input — VOC2007 test"),
                      (ax_pred, "HyMeYOLO detections")):
        ax.imshow(img_show)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for gi in range(gt_count):
        x0, y0, x1, y1 = (gt_boxes[gi] * canvas_px)
        ax_in.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.2, edgecolor="cyan", facecolor="none", alpha=0.7,
        ))
        name = class_names[int(gt_classes[gi])]
        ax_in.text(x0, y0 - 1, name, fontsize=7, color="cyan")

    for d in detections:
        ax_pred.add_patch(Rectangle(
            (d.x0, d.y0), d.x1 - d.x0, d.y1 - d.y0,
            linewidth=1.4, edgecolor="red", facecolor="none",
        ))
        ax_pred.text(
            d.x0, d.y0 - 1.5, f"{d.agent_kind} ({d.score:.2f})",
            fontsize=7, color="red",
            bbox=dict(facecolor="white", alpha=0.4, pad=0.5,
                      edgecolor="none"),
        )


def render_voc_panels(
    checkpoint: str,
    n_panels: int,
    out_dir: Path,
    image_set: str = "test",
    year: str = "2007",
    input_size: int | None = None,
    score_threshold: float = 0.30,
    device: str = "cuda",
    data_root: str = "data/torchvision",
) -> list[Path]:
    """Render the first `n_panels` deterministic split images as GT/pred
    panels for slide capture (no cherry-picking).

    Preconditions: valid checkpoint; `n_panels >= 1`. Postconditions:
    exactly `n_panels` non-empty PNGs under `out_dir`.
    """
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint!r}")
    if n_panels < 1:
        raise ValueError(f"n_panels must be >= 1; got {n_panels}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = VocPersonDetector(
        checkpoint, score_threshold=score_threshold, device=device,
    )
    isize = int(input_size or detector.input_size)
    _X, _b, _c, _k, Xn = _load_split(
        image_set, year, isize, n_panels, data_root,
    )
    # GT (unmodified, for display) — reload boxes/classes/counts as numpy.
    Xn2, boxes_n, classes_n, counts_n, _names = load_voc_hungarian(
        year=year, image_set=image_set, input_size=isize,
        max_objects=12, root=data_root, subset_n=n_panels, download=True,
    )
    class_names = (VOC_CLASSES if detector.n_classes == len(VOC_CLASSES)
                   else tuple(f"class_{i}" for i in range(detector.n_classes)))

    paths: list[Path] = []
    for i in range(n_panels):
        rgb_u8 = (np.transpose(Xn2[i], (1, 2, 0)) * 255.0
                  ).clip(0, 255).astype(np.uint8)
        dets = detector.detect(rgb_u8)
        fig = Figure(figsize=(9, 4.5), dpi=100)
        FigureCanvasAgg(fig)
        _draw_voc_panel(
            fig, Xn2[i], boxes_n[i], classes_n[i], int(counts_n[i]),
            dets, class_names, canvas_px=isize,
        )
        fig.tight_layout()
        path = out_dir / f"voc_panel_{i:02d}.png"
        fig.savefig(path, dpi=120)
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Held-out VOC eval + slide panels for a HyMeYOLO "
                    "checkpoint.",
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["metrics", "panels", "both"],
                    default="both")
    ap.add_argument("--image-set", default="test",
                    choices=["train", "val", "trainval", "test"])
    ap.add_argument("--year", default="2007", choices=["2007", "2012"])
    ap.add_argument("--input-size", type=int, default=None,
                    help="Override; default = checkpoint's input_size.")
    ap.add_argument("--n-images", type=int, default=None,
                    help="Eval subset (metrics mode); None = full split.")
    ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--out-dir", default="demo_out/voc")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-root", default="data/torchvision")
    args = ap.parse_args()

    if args.mode in ("metrics", "both"):
        m = eval_checkpoint_on_split(
            args.checkpoint, image_set=args.image_set, year=args.year,
            input_size=args.input_size, n_images=args.n_images,
            batch_size=args.batch_size, device=args.device,
            data_root=args.data_root,
        )
        print(
            f"[eval] {args.image_set}: mAP_50={m['mAP_50']:.4f} "
            f"mAP_50_95={m['mAP_50_95']:.4f} "
            f"mean_iou={m['mean_iou_matched']:.4f} | "
            f"n_images={m['n_images']} n_preds={m.get('n_preds_used')} "
            f"n_gts={m.get('n_gts_total')} | "
            f"peak_rss={m['peak_rss_mb']:.0f}MB wall={m['wall_s']:.1f}s"
        )

    if args.mode in ("panels", "both"):
        paths = render_voc_panels(
            args.checkpoint, n_panels=args.panels, out_dir=Path(args.out_dir),
            image_set=args.image_set, year=args.year,
            input_size=args.input_size, score_threshold=args.threshold,
            device=args.device, data_root=args.data_root,
        )
        for p in paths:
            print(f"[eval] wrote {p}")


if __name__ == "__main__":
    main()
