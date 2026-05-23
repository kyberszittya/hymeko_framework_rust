"""VocPersonDetector — load a Stage H / Stage D-3 checkpoint and run inference.

Drop-in replacement for the HSV blob detector in
``signedkan_wip.src.rapport_ros2.vision_sidecar_node``. The
checkpoint format is exactly what ``train_voc_person.py`` and
``train_voc_stagec.py`` emit (model_class=RicciHyMeYOLOMulti +
the kinematic config + state_dict).

Both head variants are supported:

* **legacy Hungarian** (``query_head_kind="hungarian"`` in the
  checkpoint, or absent): predictions are filtered by the
  class-confidence margin against the no-object slot.
* **nodelet** (``query_head_kind="nodelet"`` in the checkpoint):
  predictions are filtered by the per-query gate score.

The detect API returns the same tuple-list contract that the HSV
blob detector exposes (so the sidecar dispatch is a one-line
change).

Plan: docs/plans/2026-05-19-stage-h-voc-eyes-for-rapport/ +
docs/plans/2026-05-19-stage-d3-nodelet-head/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Class-name table per checkpoint dataset. The Stage H checkpoint
# uses single-class person; Stage D-3 checkpoints will use the full
# 20-class VOC list.
_VOC20_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)


@dataclass
class Detection:
    """One detection in image coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    agent_kind: str   # the class name ("person", "chair", ...)


class VocPersonDetector:
    """Wraps a HyMeYOLO RicciHyMeYOLOMulti checkpoint for inference.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to a .pt file emitted by ``train_voc_person.py`` or
        ``train_voc_stagec.py``.
    score_threshold : float, default 0.3
        Minimum score (class probability for Hungarian; gate value for
        nodelet) to include a detection.
    device : str, default "cpu"
        torch device. CPU is the default because Mihoko's lab CPUs
        are the expected runtime.
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        score_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.score_threshold = float(score_threshold)
        self.device = torch.device(device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device,
                          weights_only=False)
        self._ckpt = ckpt

        # Reconstruct the kinematic config.
        self.n_box_queries = int(ckpt.get("n_box_queries", 12))
        self.n_classes = int(ckpt.get("n_classes", 20))
        self.input_size = int(ckpt.get("input_size", 224))
        self.backbone = ckpt.get("backbone", "resnet")
        self.fpn = ckpt.get("fpn", "2level")
        self.query_head_kind = ckpt.get("query_head_kind", "hungarian")
        ricci_scale = float(ckpt.get("ricci_scale", 1.0))

        # Class-name table.
        if self.n_classes == 1:
            self.class_names = ("person",)
        elif self.n_classes == len(_VOC20_CLASSES):
            self.class_names = _VOC20_CLASSES
        else:
            # Generic fallback — class indices as their own names.
            self.class_names = tuple(f"class_{i}"
                                       for i in range(self.n_classes))

        # Build the model.
        from ..vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
        self.model = RicciHyMeYOLOMulti(
            n_box_queries=self.n_box_queries,
            n_circle_queries=0,
            n_classes=self.n_classes,
            d_hidden=32,
            ricci_modulation=True,
            ricci_scale=ricci_scale,
            use_layernorm=bool(ckpt.get("use_layernorm", False)),
            backbone=self.backbone,
            fpn=self.fpn,
            query_head_kind=self.query_head_kind,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Run inference on a single RGB image.

        Args:
            rgb: (H, W, 3) uint8 RGB image.

        Returns:
            list of :class:`Detection`. Filtered by ``score_threshold``.
        """
        if rgb.dtype != np.uint8:
            raise ValueError(f"expected uint8, got {rgb.dtype}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3), got {rgb.shape}")
        H_orig, W_orig = rgb.shape[:2]
        # Resize to model's input_size (square, bilinear).
        t = torch.from_numpy(rgb).to(self.device).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        t = F.interpolate(t, size=(self.input_size, self.input_size),
                            mode="bilinear", align_corners=False)
        pred = self.model(t)

        box_corners = pred["box_corners"][0]    # (N, 4, 2) in [0, 1]
        box_cls = pred["box_cls"][0]            # (N, n_classes [+1])
        out: list[Detection] = []

        # Two dispatch paths: nodelet (gate-filtered) or legacy.
        if "box_gates" in pred:
            gates = pred["box_gates"][0]        # (N,) in [0, 1]
            cls_probs = F.softmax(box_cls, dim=-1)
            class_ids = cls_probs.argmax(dim=-1)
            class_scores = cls_probs.gather(
                -1, class_ids.unsqueeze(-1)
            ).squeeze(-1)
            # Score = gate × class_prob (joint confidence).
            scores = gates * class_scores
            keep = scores > self.score_threshold
        else:
            # Legacy Hungarian: cls head emits n_classes + 1 logits;
            # slot n_classes is no-object. A real-object prediction is
            # one where the softmax mass on a real class slot exceeds
            # the threshold.
            cls_probs = F.softmax(box_cls, dim=-1)
            real_class_probs = cls_probs[:, :self.n_classes]
            class_ids = real_class_probs.argmax(dim=-1)
            scores = real_class_probs.gather(
                -1, class_ids.unsqueeze(-1)
            ).squeeze(-1)
            keep = scores > self.score_threshold

        for q in keep.nonzero(as_tuple=True)[0].tolist():
            corners = box_corners[q]                 # (4, 2) in [0, 1]
            x_min = float(corners[:, 0].min().item())
            y_min = float(corners[:, 1].min().item())
            x_max = float(corners[:, 0].max().item())
            y_max = float(corners[:, 1].max().item())
            # Map back to original image coordinates.
            cls_idx = int(class_ids[q].item())
            cls_name = self.class_names[cls_idx] \
                if 0 <= cls_idx < len(self.class_names) \
                else f"class_{cls_idx}"
            out.append(Detection(
                x0=int(round(x_min * W_orig)),
                y0=int(round(y_min * H_orig)),
                x1=int(round(x_max * W_orig)),
                y1=int(round(y_max * H_orig)),
                score=float(scores[q].item()),
                agent_kind=cls_name,
            ))
        return out

    @property
    def info(self) -> dict:
        """Compact metadata for logging."""
        return {
            "ckpt": str(self.ckpt_path),
            "n_classes": self.n_classes,
            "n_box_queries": self.n_box_queries,
            "backbone": self.backbone,
            "fpn": self.fpn,
            "query_head_kind": self.query_head_kind,
            "input_size": self.input_size,
            "label": self._ckpt.get("label", "?"),
            "dataset": self._ckpt.get("dataset", "?"),
        }
