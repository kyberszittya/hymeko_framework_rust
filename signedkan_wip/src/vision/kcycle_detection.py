"""k-Cycle Vision Detection (kCVD) — first scaffold.

Replaces YOLO-style fixed-rectangle anchor grids with detection
primitives derived from the image's own structure. The flow:

    image
      ↓  (keypoint placement: regular-grid + perturbation, or detector)
    keypoints
      ↓  (Delaunay triangulation)
    planar graph (vertices = keypoints, edges = Delaunay edges)
      ↓  (k-cycle enumeration; k=3 = Delaunay triangles)
    candidate detection FACES
      ↓  (per-vertex image features + HSiKAN-style per-face aggregation)
    per-face class probabilities

This module ships the first runnable scaffold: synthetic polygon
detection on 256×256 images. The smoke test below exercises the full
pipeline on 32 random images and asserts loss decreases over a few
optimiser steps.

Caveats / scope:
- v1 only uses k=3 (Delaunay triangles) as detection candidates. Adding
  k=4, 5, 6 cycles via DFS over the Delaunay edge set is the natural
  next step (the αₖ mixer plugs into the existing HSiKAN machinery).
- Keypoint placement is a perturbed regular grid — content-agnostic.
  Real keypoint detectors (SIFT / ORB / SuperPoint) are easy to swap in.
- Per-vertex image features are simple (mean RGB + gradient magnitude
  in a small window). A real version would use a frozen ResNet
  backbone — same way Mask R-CNN does.

See `docs/plans_kcycle_vision_2026_05_07.md` for the design rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay


# ─── Step 1: image → graph ──────────────────────────────────────────


@dataclass
class ImageGraph:
    """A planar graph derived from an image.

    - `keypoints`: (n_v, 2) float array of (y, x) pixel coords
    - `edges`: (n_e, 2) int array of vertex-pair indices (undirected)
    - `triangles`: (n_t, 3) int array of vertex triples (Delaunay simplices)
    - `H`, `W`: image dims (kept for feature extraction at known scale)
    """
    keypoints: np.ndarray
    edges: np.ndarray
    triangles: np.ndarray
    H: int
    W: int


def make_image_graph(
    H: int, W: int, n_keypoints: int = 200, jitter: float = 0.4, seed: int = 0,
) -> ImageGraph:
    """Build a planar graph by placing roughly-regular keypoints + Delaunay.

    `n_keypoints` is approximate — actual count depends on the grid
    factorisation. `jitter` ∈ [0, 0.5] perturbs each keypoint within
    its grid cell; 0 = strict grid, 0.5 = neighbouring cell boundaries.
    """
    rng = np.random.default_rng(seed)

    # Choose grid dimensions so rows*cols ≈ n_keypoints with W:H ratio.
    aspect = W / H
    rows = max(2, int(round(math.sqrt(n_keypoints / aspect))))
    cols = max(2, int(round(rows * aspect)))
    cell_h, cell_w = H / rows, W / cols

    pts = []
    for r in range(rows):
        for c in range(cols):
            cy = (r + 0.5) * cell_h + rng.uniform(-jitter, jitter) * cell_h
            cx = (c + 0.5) * cell_w + rng.uniform(-jitter, jitter) * cell_w
            pts.append((cy, cx))
    keypoints = np.array(pts, dtype=np.float32)

    # Delaunay over (x, y) — scipy uses (col, row) convention.
    tri = Delaunay(keypoints[:, [1, 0]])
    triangles = np.array(tri.simplices, dtype=np.int64)

    # Edge set from triangle simplices (undirected, dedup).
    edge_set: set[tuple[int, int]] = set()
    for t in triangles:
        for a, b in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]:
            edge_set.add((min(a, b), max(a, b)))
    edges = np.array(sorted(edge_set), dtype=np.int64)

    return ImageGraph(keypoints=keypoints, edges=edges, triangles=triangles, H=H, W=W)


# ─── Step 2: per-vertex image features ──────────────────────────────


def vertex_features(img: np.ndarray, graph: ImageGraph,
                    window: int = 8) -> np.ndarray:
    """Sample per-vertex image features in a `window × window` patch
    around each keypoint.

    Returns (n_v, F) float32 with F = 5 channels:
       (mean R, mean G, mean B, mean gradient magnitude, gradient angle).
    For grayscale input: R=G=B = the single channel.
    """
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.float32) / 255.0
    H, W, _ = img.shape
    half = window // 2

    # Sobel-like gradient magnitude.
    gy = np.zeros((H, W), dtype=np.float32)
    gy[1:-1, :] = img.mean(axis=-1)[2:, :] - img.mean(axis=-1)[:-2, :]
    gx = np.zeros((H, W), dtype=np.float32)
    gx[:, 1:-1] = img.mean(axis=-1)[:, 2:] - img.mean(axis=-1)[:, :-2]
    gmag = np.sqrt(gx * gx + gy * gy)
    gang = np.arctan2(gy, gx)

    feats = []
    for (y, x) in graph.keypoints:
        y0 = int(max(0, y - half)); y1 = int(min(H, y + half))
        x0 = int(max(0, x - half)); x1 = int(min(W, x + half))
        patch = img[y0:y1, x0:x1]
        gp = gmag[y0:y1, x0:x1]
        ap = gang[y0:y1, x0:x1]
        feats.append([
            patch[..., 0].mean(),
            patch[..., 1].mean(),
            patch[..., 2].mean(),
            gp.mean(),
            ap.mean(),
        ])
    return np.array(feats, dtype=np.float32)


# ─── Step 3: per-triangle (k=3 face) HSiKAN aggregation ─────────────


class KCycleDetector(nn.Module):
    """HSiKAN-style per-face aggregation for triangle (k=3) detection.

    For each triangle t = (v_1, v_2, v_3):
        h_t = φ_e(Σ_i φ_v(h_v_i))
    where φ_v, φ_e are linear+activation pairs (Catmull-Rom-like splines
    in the full version; tanh+linear in this scaffold for speed). Output
    is a per-triangle class logit.

    Fields:
      - d_in:    per-vertex feature dim (5 for the simple sampler)
      - d_hidden: aggregation dim
      - n_classes: number of object classes (incl. background)
    """
    def __init__(self, d_in: int, d_hidden: int = 32, n_classes: int = 2):
        super().__init__()
        self.d_hidden = d_hidden
        self.embed = nn.Linear(d_in, d_hidden)
        self.phi_v = nn.Linear(d_hidden, d_hidden)
        self.phi_e = nn.Linear(d_hidden, d_hidden)
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, vertex_feats: torch.Tensor,
                triangles: torch.Tensor) -> torch.Tensor:
        """
        vertex_feats: (B, n_v, d_in)  per-vertex features (B-batched OK)
        triangles:    (n_t, 3)         vertex indices per triangle (shared)
        Returns:      (B, n_t, n_classes) per-face class logits
        """
        if vertex_feats.dim() == 2:
            vertex_feats = vertex_feats.unsqueeze(0)
        h_v = self.embed(vertex_feats)           # (B, n_v, d_hidden)

        # Per-triangle: gather, project, sum, project, classify.
        # h_v[..., triangles, :] has shape (B, n_t, 3, d_hidden)
        h_t_v = h_v[:, triangles, :]              # (B, n_t, 3, d_hidden)
        h_t_v = torch.tanh(self.phi_v(h_t_v))
        h_t = h_t_v.sum(dim=-2)                    # (B, n_t, d_hidden)
        h_t = self.phi_e(h_t)
        h_t = torch.tanh(h_t)
        return self.head(h_t)                      # (B, n_t, n_classes)


# ─── Step 4: synthetic polygon dataset ──────────────────────────────


def make_synthetic_polygons(
    n_images: int = 32, H: int = 256, W: int = 256,
    n_polygons_range=(1, 4), seed: int = 0,
):
    """Generate `n_images` synthetic 256×256 RGB images with random
    convex polygons (triangle / square / pentagon / hexagon) on a
    plain noisy background. Returns:
       images: (N, H, W, 3) uint8
       polygons: list[N] of list[Polygon] with .vertices: (k, 2) and .label: int
    """
    rng = np.random.default_rng(seed)
    images = np.zeros((n_images, H, W, 3), dtype=np.uint8)
    all_polys = []

    for n in range(n_images):
        # Background: noisy grey.
        bg_color = rng.integers(80, 180, size=3, dtype=np.uint8)
        img = np.tile(bg_color, (H, W, 1)).astype(np.float32)
        img += rng.normal(0, 8, size=img.shape)

        polys = []
        n_p = rng.integers(*n_polygons_range, endpoint=True)
        for _ in range(n_p):
            cy, cx = rng.uniform(40, H - 40), rng.uniform(40, W - 40)
            r = rng.uniform(20, 50)
            n_sides = rng.choice([3, 4, 5, 6])
            theta0 = rng.uniform(0, 2 * np.pi)
            color = rng.integers(0, 255, size=3, dtype=np.uint8).astype(np.float32)
            verts = []
            for k in range(n_sides):
                a = theta0 + 2 * np.pi * k / n_sides
                verts.append([cy + r * np.sin(a), cx + r * np.cos(a)])
            verts = np.array(verts, dtype=np.float32)
            _fill_polygon(img, verts, color)
            polys.append({"vertices": verts, "label": int(n_sides - 3)})  # 0..3
        images[n] = np.clip(img, 0, 255).astype(np.uint8)
        all_polys.append(polys)
    return images, all_polys


def _fill_polygon(img: np.ndarray, verts: np.ndarray, color: np.ndarray):
    """Cheap point-in-polygon rasteriser. img is (H, W, 3) float32."""
    H, W, _ = img.shape
    y0, y1 = max(0, int(verts[:, 0].min())), min(H, int(verts[:, 0].max()) + 1)
    x0, x1 = max(0, int(verts[:, 1].min())), min(W, int(verts[:, 1].max()) + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            if _point_in_poly(y + 0.5, x + 0.5, verts):
                img[y, x] = color


def _point_in_poly(y: float, x: float, verts: np.ndarray) -> bool:
    """Crossing-number test."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        yi, xi = verts[i]
        yj, xj = verts[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def label_triangles_by_polygons(
    graph: ImageGraph, polygons: list, n_classes: int = 5,
) -> np.ndarray:
    """For each triangle, assign a class label. Class 0 = background;
    1..4 = triangles whose centroid falls inside a polygon with k sides
    in {3, 4, 5, 6}. Returns (n_t,) int64.
    """
    centroids = graph.keypoints[graph.triangles].mean(axis=1)  # (n_t, 2)
    labels = np.zeros(len(graph.triangles), dtype=np.int64)
    for ti, (cy, cx) in enumerate(centroids):
        for poly in polygons:
            if _point_in_poly(cy, cx, poly["vertices"]):
                labels[ti] = poly["label"] + 1   # 0=bg, 1..4=k=3..6 polygons
                break
    return labels


# ─── Step 5: focal loss ─────────────────────────────────────────────


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-class focal loss: ``-α_t (1 - p_t)^γ log p_t`` averaged over inputs.

    `logits` is (N, C); `targets` is (N,) with class indices in [0, C).
    `alpha` is an optional (C,) per-class weight tensor (background usually
    smaller than foreground). With γ=0 and uniform α this reduces to CE.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal_term = (1.0 - pt).clamp(min=1e-12).pow(gamma)
    if alpha is not None:
        at = alpha.to(logits.device)[targets]
        return -(at * focal_term * log_pt).mean()
    return -(focal_term * log_pt).mean()


# ─── Step 6: per-image triangle resampling ──────────────────────────


def resample_triangles(
    Y: torch.Tensor, neg_pos_ratio: float = 3.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-image: keep all positives + ``neg_pos_ratio × n_pos`` random negatives.

    Y: (N, n_t) class labels with 0 = background.
    Returns a (N, n_t) bool mask of triangles to include in the loss.
    Images with zero positives fall back to a tiny neg sample (still trains on
    them but with negligible weight).
    """
    N, _ = Y.shape
    mask = torch.zeros_like(Y, dtype=torch.bool)
    for n in range(N):
        pos = (Y[n] > 0).nonzero(as_tuple=True)[0]
        neg = (Y[n] == 0).nonzero(as_tuple=True)[0]
        n_neg = min(len(neg), int(neg_pos_ratio * max(len(pos), 1)))
        if n_neg > 0:
            perm = torch.randperm(len(neg), generator=generator)[:n_neg]
            mask[n, neg[perm]] = True
        mask[n, pos] = True
    return mask


# ─── Step 7: smoke test ─────────────────────────────────────────────


def smoke(n_images: int = 32, n_epochs: int = 5, seed: int = 0,
          n_keypoints: int = 600, d_hidden: int = 32) -> dict:
    """Train the scaffold for a few epochs on synthetic polygons and
    assert loss decreases. Returns metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    images, polys = make_synthetic_polygons(n_images=n_images, seed=seed)
    H, W = images[0].shape[:2]

    # Shared graph (keypoints don't depend on image content in v1).
    graph = make_image_graph(H, W, n_keypoints=n_keypoints, seed=seed)

    # Pre-compute vertex features + labels for every image.
    vfeats = []
    labels = []
    for img, ps in zip(images, polys):
        vfeats.append(vertex_features(img, graph))
        labels.append(label_triangles_by_polygons(graph, ps))
    X = torch.from_numpy(np.stack(vfeats))                # (N, n_v, 5)
    Y = torch.from_numpy(np.stack(labels))                # (N, n_t)
    triangles = torch.from_numpy(graph.triangles)         # (n_t, 3)

    print(f"[smoke] N={n_images} n_v={graph.keypoints.shape[0]} "
          f"n_e={graph.edges.shape[0]} n_t={graph.triangles.shape[0]}")
    print(f"[smoke] class distribution: {np.bincount(Y.numpy().flatten(), minlength=5)}")

    model = KCycleDetector(d_in=5, d_hidden=d_hidden, n_classes=5)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Focal loss + per-image resampling. Focal-alone left the model stuck
    # at the always-bg minimum because the sheer count of bg triangles
    # outweighs the per-triangle (1-pt)^γ down-weight. Resampling at a
    # fixed neg:pos ratio per image makes the loss dominated by hard
    # examples regardless of dataset imbalance.
    # 4 fg classes vs 1 bg ⇒ fg is split, so even 3:1 neg:pos leaves
    # bg outweighing each individual class ~10:1. Use 1:1 + a small bg
    # alpha to keep the per-class loss in the same order of magnitude.
    alpha = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0])
    gen = torch.Generator().manual_seed(seed)

    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, triangles)                       # (N, n_t, 5)
        loss_mask = resample_triangles(Y, neg_pos_ratio=1.0, generator=gen)
        flat_logits = logits[loss_mask]                     # (M, 5)
        flat_Y = Y[loss_mask]                               # (M,)
        loss = focal_loss(flat_logits, flat_Y, gamma=2.0, alpha=alpha)
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
        with torch.no_grad():
            preds = logits.argmax(-1)
            acc = (preds == Y).float().mean().item()
            fg_mask = Y > 0
            fg_acc = (preds[fg_mask] == Y[fg_mask]).float().mean().item() \
                if fg_mask.any() else float("nan")
            n_kept = int(loss_mask.sum().item())
        print(f"[smoke] epoch {epoch+1:3d}  loss={loss.item():.4f}  "
              f"acc={acc:.3f}  fg_acc={fg_acc:.3f}  kept={n_kept}")

    assert loss_history[-1] < loss_history[0], \
        f"loss should decrease: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}"

    return {
        "n_images": n_images,
        "n_epochs": n_epochs,
        "loss_first": loss_history[0],
        "loss_last": loss_history[-1],
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_triangles": int(graph.triangles.shape[0]),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(smoke(n_images=32, n_epochs=10), indent=2))
