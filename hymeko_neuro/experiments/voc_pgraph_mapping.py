"""Map HyMeYOLO Stage-D VOC P-graph unit names →
``train_voc_stagec`` kwargs.

The sweep fixture ``data/hsikan/sweep_msg_voc.hymeko`` describes
the VOC2007 detector architecture-search as a P-graph with three
orthogonal axes:

  * Backbone        backbone_{resnet, resnet18_imagenet, hsikan}
                    → ``--backbone``
  * Query count     q4 / q8 / q12
                    → ``--n-box-queries``
  * lam_no_obj      lam_low (0.5) / lam_high (2.0)
                    → ``--lam-no-obj``

Phase 15 (2026-05-20 overnight) shipped this module alongside the
parallel HSIKAN / Gömb / cortical mappings so the VOC sweep driver
consumes the ABB-selected unit set in the same way.
"""
from __future__ import annotations

from typing import Any


VOC_UNIT_TO_KNOBS: dict[str, dict[str, Any]] = {
    # Backbone axis — three Stage-D variants.
    "backbone_resnet":            {"backbone": "resnet"},
    "backbone_resnet18_imagenet": {"backbone": "resnet18_imagenet"},
    "backbone_hsikan":            {"backbone": "hsikan"},
    # Query-count axis.
    "q4":  {"n_box_queries":  4},
    "q8":  {"n_box_queries":  8},
    "q12": {"n_box_queries": 12},
    # No-object-loss-weight axis.
    "lam_low":  {"lam_no_obj": 0.5},
    "lam_high": {"lam_no_obj": 2.0},
}


def merge_structure_knobs(unit_names: list[str]) -> dict[str, Any]:
    """Merge knob dicts for all units in one ABB / SSG solution.

    Later keys win on collision (prefer the last unit in
    ``unit_names``). Raises ``KeyError`` on any unrecognised unit
    name — same contract as the HSIKAN / Gömb / cortical mappings.
    """
    merged: dict[str, Any] = {}
    for raw in unit_names:
        u = raw.strip()
        patch = VOC_UNIT_TO_KNOBS.get(u)
        if patch is None:
            raise KeyError(
                f"unknown VOC P-graph unit {u!r}; add it to "
                f"VOC_UNIT_TO_KNOBS in voc_pgraph_mapping.py"
            )
        merged.update(patch)
    return merged


def train_voc_kwargs(
    *,
    seed: int,
    structure: dict[str, Any],
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``train_voc_stagec.train_voc_stagec_one(...)``
    or for the CLI argparse equivalent.

    ``base`` provides defaults for axes not in the P-graph
    (e.g.\\ ``n_images``, ``epochs``, ``input_size``,
    ``query_head_kind``). The Phase-15 smoke uses small defaults so
    the wiring runs in $\\sim 60\\,$s.
    """
    base = dict(base or {})
    backbone = str(structure.get("backbone",
                                 base.get("backbone", "resnet")))
    n_box_queries = int(structure.get("n_box_queries",
                                       base.get("n_box_queries", 12)))
    lam_no_obj = float(structure.get("lam_no_obj",
                                      base.get("lam_no_obj", 0.5)))
    return {
        "seed": seed,
        "year": str(base.get("year", "2007")),
        "image_set": str(base.get("image_set", "trainval")),
        "n_images": base.get("n_images"),  # None = full split
        "input_size": int(base.get("input_size", 224)),
        "max_objects": int(base.get("max_objects", 12)),
        "n_box_queries": n_box_queries,
        "epochs": int(base.get("epochs", 30)),
        "lr": float(base.get("lr", 3e-3)),
        "batch_size": int(base.get("batch_size", 8)),
        "device": base.get("device"),
        "backbone": backbone,
        "lam_no_obj": lam_no_obj,
        "query_head_kind": str(base.get("query_head_kind", "hungarian")),
        "schedule": str(base.get("schedule", "cosine")),
        "warmup_epochs": int(base.get("warmup_epochs", 2)),
        "ricci_scale": float(base.get("ricci_scale", 1.0)),
    }


__all__ = [
    "VOC_UNIT_TO_KNOBS",
    "merge_structure_knobs",
    "train_voc_kwargs",
]
