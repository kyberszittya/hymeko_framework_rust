"""NTU RGB+D action-recognition adapter for HSiKAN.

NTU RGB+D 60: 60 action classes, ~56k samples, 25 body joints per
skeleton, 3D coordinates per joint per frame. Provides skeleton graphs
(fixed bone topology) + per-frame continuous features.

Adapter responsibilities:
  - Parse NTU's `.skeleton` files (or pre-extracted .npz)
  - Build the 25-joint skeleton SignedGraph (vertices=joints,
    edges=bones, sign assignment by joint kinematic role)
  - Extract per-vertex continuous features (3D position per frame)
  - Extract per-edge continuous features (bone vector + length)
  - Provide action-class label for each sample

Sign assignment for skeleton edges
----------------------------------
Three plausible binary dichotomies:
  - **anatomical**: spine/torso (+1) vs limb (−1)  — fixed per dataset
  - **active/passive** (per-pose): joint-extended (+1) vs flexed (−1) —
    derived from joint angles per frame; gives a *temporal* sign
  - **upper/lower body**: above-spine-midpoint (+1) vs below (−1) —
    fixed per skeleton

We use **anatomical** for the static graph (consistent across frames)
and bake **per-frame joint angles** into per-vertex/edge continuous
features.

Data download
-------------
NTU RGB+D 60 is gated; request access at:
  https://rose1.ntu.edu.sg/dataset/actionRecognition/

After download, point `NTU_DATA_DIR` at the extracted folder. We use
the pre-processed `.npz` files (joints already normalised).

Tasks
-----
  - **Action recognition** (60-class): graph-level classification on
    the skeleton + per-frame features
  - **Pose classification** (smaller subset): same head, fewer classes
  - **Anomaly detection**: out-of-distribution score

Status: scaffold only. Ship a synthetic substitute (8 actions x 25
joints x 30 frames) so the architectural pathway can be tested without
the real dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from ..datasets import SignedGraph


# Standard NTU RGB+D 25-joint skeleton topology.
# Joint indices follow Microsoft Kinect v2 convention.
NTU_BONES_25 = [
    (0, 1),    # spine_base → spine_mid
    (1, 20),   # spine_mid → spine_shoulder
    (20, 2),   # spine_shoulder → neck
    (2, 3),    # neck → head
    # Right arm
    (20, 4),   # spine_shoulder → r_shoulder
    (4, 5),    # r_shoulder → r_elbow
    (5, 6),    # r_elbow → r_wrist
    (6, 7),    # r_wrist → r_hand
    (7, 21),   # r_hand → r_handtip
    (7, 22),   # r_hand → r_thumb
    # Left arm
    (20, 8),   # spine_shoulder → l_shoulder
    (8, 9),    # l_shoulder → l_elbow
    (9, 10),   # l_elbow → l_wrist
    (10, 11),  # l_wrist → l_hand
    (11, 23),  # l_hand → l_handtip
    (11, 24),  # l_hand → l_thumb
    # Right leg
    (0, 12),   # spine_base → r_hip
    (12, 13),  # r_hip → r_knee
    (13, 14),  # r_knee → r_ankle
    (14, 15),  # r_ankle → r_foot
    # Left leg
    (0, 16),   # spine_base → l_hip
    (16, 17),  # l_hip → l_knee
    (17, 18),  # l_knee → l_ankle
    (18, 19),  # l_ankle → l_foot
]

# Anatomical sign convention (spine/torso=+1, limb=−1).
SPINE_TORSO_NODES = {0, 1, 20, 2, 3, 4, 8, 12, 16}
def _bone_sign(u: int, v: int) -> int:
    spine_u = u in SPINE_TORSO_NODES
    spine_v = v in SPINE_TORSO_NODES
    return +1 if (spine_u and spine_v) else -1


@dataclass
class NTUSample:
    """One skeleton-action sample."""
    graph: SignedGraph                 # 25-joint skeleton (static)
    joint_positions: np.ndarray        # (T_frames, 25, 3)
    joint_velocities: np.ndarray       # (T_frames-1, 25, 3)
    bone_vectors: np.ndarray           # (T_frames, 24, 3)
    bone_lengths: np.ndarray           # (T_frames, 24)
    action_label: int                  # 0..C-1
    sample_id: str = ""


def build_ntu_signed_graph() -> SignedGraph:
    """Static 25-joint NTU skeleton SignedGraph (anatomical sign)."""
    edges = np.array(NTU_BONES_25, dtype=np.int64)
    signs = np.array([_bone_sign(u, v) for u, v in NTU_BONES_25],
                       dtype=np.int8)
    return SignedGraph(edges=edges, signs=signs, n_nodes=25)


def synth_ntu_dataset(n_classes: int = 8, n_per_class: int = 50,
                        n_frames: int = 30, seed: int = 0) -> list[NTUSample]:
    """Synthetic NTU-style dataset for testing the adapter pipeline
    without the real download. Each class has its own deterministic
    motion pattern (sine on different joint subsets at different
    frequencies)."""
    rng = np.random.RandomState(seed)
    g = build_ntu_signed_graph()
    samples = []
    for cls in range(n_classes):
        # Each class: a unique sine-pattern on a unique joint subset.
        active_joints = rng.choice(25, size=8, replace=False).tolist()
        freq = 0.5 + cls * 0.3
        for inst in range(n_per_class):
            base_pose = rng.randn(25, 3) * 0.3
            phases = rng.rand(25) * 2 * np.pi
            positions = np.zeros((n_frames, 25, 3), dtype=np.float32)
            for t in range(n_frames):
                positions[t] = base_pose
                for j in active_joints:
                    positions[t, j] += 0.2 * np.sin(2 * np.pi * freq * t / n_frames + phases[j])
            velocities = np.diff(positions, axis=0)
            bone_vectors = np.array([
                positions[:, v] - positions[:, u] for u, v in NTU_BONES_25
            ]).transpose(1, 0, 2)
            bone_lengths = np.linalg.norm(bone_vectors, axis=-1)
            samples.append(NTUSample(
                graph=g,
                joint_positions=positions.astype(np.float32),
                joint_velocities=velocities.astype(np.float32),
                bone_vectors=bone_vectors.astype(np.float32),
                bone_lengths=bone_lengths.astype(np.float32),
                action_label=cls,
                sample_id=f"synth_cls{cls}_inst{inst}",
            ))
    rng.shuffle(samples)
    return samples


def aggregate_per_vertex_features(sample: NTUSample,
                                     pool: str = "last") -> np.ndarray:
    """Reduce per-frame joint positions/velocities to a per-vertex
    feature vector for HSiKAN's vertex_features pathway.

    Returns: (25, F) array. F depends on pool.
    """
    if pool == "last":
        # Position at last frame + velocity at last frame.
        return np.concatenate([
            sample.joint_positions[-1],     # (25, 3)
            sample.joint_velocities[-1],    # (25, 3)
        ], axis=-1)                          # (25, 6)
    elif pool == "mean":
        return np.concatenate([
            sample.joint_positions.mean(axis=0),
            sample.joint_velocities.mean(axis=0),
        ], axis=-1)
    elif pool == "stats":
        # Mean + std of position over frames.
        return np.concatenate([
            sample.joint_positions.mean(axis=0),
            sample.joint_positions.std(axis=0),
            sample.joint_velocities.mean(axis=0),
        ], axis=-1)
    else:
        raise ValueError(f"unknown pool: {pool!r}")


def aggregate_per_edge_features(sample: NTUSample,
                                  pool: str = "stats") -> np.ndarray:
    """Reduce per-frame bone vectors to a per-edge feature vector.

    Returns: (24, F) array. F depends on pool.
    """
    if pool == "last":
        return np.concatenate([
            sample.bone_vectors[-1],                           # (24, 3)
            sample.bone_lengths[-1:].T,                        # (24, 1)
        ], axis=-1)
    elif pool == "stats":
        return np.concatenate([
            sample.bone_vectors.mean(axis=0),                  # (24, 3)
            sample.bone_vectors.std(axis=0),                   # (24, 3)
            sample.bone_lengths.mean(axis=0)[:, None],         # (24, 1)
            sample.bone_lengths.std(axis=0)[:, None],          # (24, 1)
        ], axis=-1)
    else:
        raise ValueError(f"unknown pool: {pool!r}")


if __name__ == "__main__":
    g = build_ntu_signed_graph()
    print(f"NTU skeleton graph: {g.stats()}")
    print(f"  edges: {len(NTU_BONES_25)}")
    print(f"  pos signs (spine-spine): {int((g.signs == 1).sum())}")
    print(f"  neg signs (spine-limb):  {int((g.signs == -1).sum())}")
    samples = synth_ntu_dataset(n_classes=8, n_per_class=20,
                                  n_frames=30, seed=0)
    print(f"\nSynthetic dataset: {len(samples)} samples, "
          f"{len(set(s.action_label for s in samples))} classes")
    s = samples[0]
    print(f"  sample shapes: "
          f"positions={s.joint_positions.shape}  "
          f"bones={s.bone_vectors.shape}")
    vf = aggregate_per_vertex_features(s, pool="stats")
    ef = aggregate_per_edge_features(s, pool="stats")
    print(f"  per-vertex features: {vf.shape}")
    print(f"  per-edge features:   {ef.shape}")
