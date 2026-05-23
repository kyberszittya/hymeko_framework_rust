"""Visual Genome scene-graph adapter for HSiKAN.

Visual Genome: 108k images, ~38 objects per image on average, ~22
relations per image. Object types: ~80k (long-tail). Relation types:
~40k (also long-tail). The 150-class object / 50-class relation
filtered subset (Xu et al. 2017, "Scene Graph Generation by Iterative
Message Passing") is the standard benchmark.

Adapter responsibilities:
  - Parse VG annotations (`scene_graphs.json` from VG-150)
  - Build SignedGraph per image: vertices=objects, edges=relations
  - Sign assignment: spatial-positive (above/in/on/etc.) vs
    spatial-negative (below/under) vs other (drop, or use 0-branch)
  - Per-vertex features: bbox xyxy + area + class index
  - Per-edge features: relative-position vector + IoU + relation-class
    embedding

Sign assignment (spatial relations only)
----------------------------------------
  +1: above, on, sitting on, standing on, walking on, in, inside,
       containing, behind (from camera POV), next to (above plane)
  -1: below, under, beneath, in front of, hanging from
   0: holding, wearing, eating, looking at, near, with, ... (semantic
       relations; need their own pathway)

For binary HSiKAN: drop type-0 relations or batch them as a separate
sign-class via use_zero_branch.

Tasks
-----
  - **Predicate prediction (PredCls)**: given objects + bboxes, predict
    relation type. Edge-classification with class-balanced loss.
  - **Scene Graph Generation (SGGen)**: predict objects + relations.
    More complex; requires object detector head.

Status: scaffold only. We ship a synthetic substitute (50 small
"kitchen-style" scenes) so the architectural pathway can be tested
without the VG download.

Data download
-------------
  https://visualgenome.org/
  Filtered VG-150 annotations:
  https://github.com/rowanz/neural-motifs/tree/master/data
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..datasets import SignedGraph
from ..core.scene_graph import SceneGraph, SceneObject, SceneRelation


# Standard VG-150 sign convention.
SIGN_BY_RELATION = {
    "above": +1, "on": +1, "sitting on": +1, "standing on": +1,
    "walking on": +1, "in": +1, "inside": +1, "containing": +1,
    "behind": +1, "next to": +1,
    "below": -1, "under": -1, "beneath": -1, "in front of": -1,
    "hanging from": -1,
    # Semantic relations: drop or use zero-branch.
    "holding": 0, "wearing": 0, "eating": 0, "looking at": 0,
    "near": 0, "with": 0,
}


# A small "kitchen scene" template used for the synthetic substitute.
KITCHEN_OBJECTS = [
    "table", "chair", "plate", "cup", "apple", "banana", "knife",
    "fork", "bowl", "bread", "jar", "shelf",
]
KITCHEN_RELATIONS = [
    ("plate", "table", "on"),
    ("apple", "plate", "on"),
    ("cup", "table", "on"),
    ("chair", "table", "next to"),
    ("knife", "plate", "next to"),
    ("fork", "plate", "next to"),
    ("bread", "plate", "on"),
    ("jar", "shelf", "on"),
    ("banana", "bowl", "in"),
    ("bowl", "table", "on"),
]


def synth_scene_graph(seed: int = 0,
                       n_objects: int = 8) -> tuple[SceneGraph, np.ndarray]:
    """Generate a synthetic kitchen-style scene with random bboxes and
    a subset of canonical relations from KITCHEN_RELATIONS.

    Returns: (SceneGraph, per-object bbox array of shape (N, 4))
    """
    rng = random.Random(seed)
    sg = SceneGraph()
    chosen = rng.sample(KITCHEN_OBJECTS, k=min(n_objects, len(KITCHEN_OBJECTS)))
    bboxes = []
    for obj in chosen:
        x = rng.random() * 0.8
        y = rng.random() * 0.8
        w = 0.05 + rng.random() * 0.15
        h = 0.05 + rng.random() * 0.15
        sg.add_object(obj, category="kitchen",
                       bbox=(x, y, x + w, y + h))
        bboxes.append([x, y, x + w, y + h])
    available_rels = [(u, v, r) for u, v, r in KITCHEN_RELATIONS
                       if u in chosen and v in chosen]
    n_rels = min(len(available_rels), max(2, n_objects // 2))
    for u, v, rel in rng.sample(available_rels, k=n_rels):
        sg.add_relation((u, v), rel)
    return sg, np.array(bboxes, dtype=np.float32)


def synth_dataset(n_scenes: int = 50, seed: int = 0) -> list[tuple]:
    """Build n_scenes synthetic scene graphs."""
    out = []
    for i in range(n_scenes):
        sg, bboxes = synth_scene_graph(seed=seed + i)
        g, names = sg.to_signed_graph(
            relation_to_sign=SIGN_BY_RELATION,
            binary_only=True,
            unknown_sign=None,
        )
        # Per-vertex features: bbox xyxy.
        # Pad to ensure (n_nodes, 4).
        vf = np.zeros((g.n_nodes, 4), dtype=np.float32)
        vf[:bboxes.shape[0]] = bboxes
        out.append((g, vf, sg))
    return out


def edge_features_from_bboxes(g: SignedGraph, bboxes: np.ndarray) -> np.ndarray:
    """Per-edge spatial features: relative position + IoU + bbox sizes.
    Returns: (E, 6) array."""
    feats = np.zeros((g.edges.shape[0], 6), dtype=np.float32)
    for ei, (u, v) in enumerate(g.edges):
        bu = bboxes[u]; bv = bboxes[v]
        # Relative center position.
        cu = ((bu[0] + bu[2]) / 2, (bu[1] + bu[3]) / 2)
        cv = ((bv[0] + bv[2]) / 2, (bv[1] + bv[3]) / 2)
        feats[ei, 0] = cu[0] - cv[0]    # dx
        feats[ei, 1] = cu[1] - cv[1]    # dy
        # IoU.
        ix1 = max(bu[0], bv[0]); iy1 = max(bu[1], bv[1])
        ix2 = min(bu[2], bv[2]); iy2 = min(bu[3], bv[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = ((bu[2]-bu[0])*(bu[3]-bu[1]) + (bv[2]-bv[0])*(bv[3]-bv[1])
                 - inter)
        feats[ei, 2] = inter / max(1e-6, union)
        # Bbox sizes.
        feats[ei, 3] = (bu[2]-bu[0])*(bu[3]-bu[1])
        feats[ei, 4] = (bv[2]-bv[0])*(bv[3]-bv[1])
        # Vertical offset (y distance).
        feats[ei, 5] = abs(cu[1] - cv[1])
    return feats


if __name__ == "__main__":
    sg, bboxes = synth_scene_graph(seed=0, n_objects=8)
    print(f"Sample scene: {sg.stats()}")
    g, names = sg.to_signed_graph(
        relation_to_sign=SIGN_BY_RELATION,
        binary_only=True, unknown_sign=None,
    )
    print(f"  SignedGraph: {g.stats()}")
    print(f"  Objects: {names}")
    ef = edge_features_from_bboxes(g, bboxes)
    print(f"  Per-edge features (E x 6): {ef.shape}")
    print(f"  Sample edge features (relative xy, IoU, bbox sizes, y-offset):")
    for ei in range(min(3, ef.shape[0])):
        print(f"    edge {tuple(g.edges[ei])} "
              f"(sign={int(g.signs[ei])}): {ef[ei]}")
    print()
    ds = synth_dataset(n_scenes=50, seed=0)
    print(f"Synthetic dataset: {len(ds)} scenes")
    sizes = [g.n_nodes for g, _, _ in ds]
    edges = [g.edges.shape[0] for g, _, _ in ds]
    print(f"  vertex count range: {min(sizes)}–{max(sizes)}")
    print(f"  edge count range: {min(edges)}–{max(edges)}")
