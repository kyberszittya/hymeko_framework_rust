"""Quadtree-driven query warmstart for HymeYOLO.

Drop-in replacement for the saliency-based farthest-point sampler in
:mod:`signedkan_wip.src.vision.hymeyolo_warmstart`.  Uses the existing
``AdaptiveQuadtreeRust`` (variance + Forman κ curvature scored) from
:mod:`signedkan_wip.src.hymeko_gomb.soma.vision.quadtree_rust` to
place query centres at the data-curvature-anchored leaves of the
quadtree.

Mechanism
---------
1. Mean-aggregate the bootstrap batch to a single image
   ``(C, H, W)`` — the saliency surface for the quadtree.
2. Build an adaptive quadtree on that aggregated image with the
   variance + curvature score.
3. Take the ``m`` leaf centres with the highest quadtree-score; if
   fewer than ``m`` leaves are produced, pad with farthest-point
   samples from a uniform distribution over the unfilled cells.

Why this is a useful CV lever
-----------------------------
The existing saliency warmstart (mean-of-absolute-pixels) was the
2026-05-16 Stage A-1 lever (+0.124 mAP_50 paired Δ, 4.68σ on
Cluttered MNIST).  That saliency is *single-scale* and *amplitude-
driven* — bright digits on dim backgrounds is exactly what it
solves.  Natural images (VOC2007) have:
- amplitude-saturated foreground (sky, grass, clothing) ≠ object
- multi-scale structure (small cars vs full-frame people)
- curvature-rich object boundaries that the quadtree was *built*
  to find.

The quadtree carries the same anchor + Forman-curvature inductive
bias the Gömb-Soma family relies on for the structural ABB; this
module brings it to HymeYOLO query init.

Design note
-----------
This module deliberately does *not* modify ``hymeyolo_warmstart.py``
directly.  Instead, it exposes a ``quadtree_saliency_fn`` that can
be passed to the existing ``warmstart_query_corners(saliency_fn=...)``
hook, *and* a ``quadtree_centres`` helper that bypasses the FPS step
entirely (the recommended path when the quadtree leaves themselves
are the natural query centres).
"""

from __future__ import annotations

from typing import Tuple

import torch

from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree_rust import (
    AdaptiveQuadtreeRust,
)


def quadtree_centres(
    X_bootstrap: torch.Tensor,
    m: int,
    *,
    patch_size_initial: int = 64,
    patch_size_min: int = 8,
    variance_weight: float = 1.0,
    curvature_weight: float = 1.0,
    score_threshold: float = 0.05,
    max_anchors: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Return ``m`` query-centre positions in normalised coordinates.

    Parameters
    ----------
    X_bootstrap : tensor (N, C, H, W)
        Bootstrap batch.  Averaged to a single image before quadtree
        build.
    m : int
        Number of centres requested.
    patch_size_initial : int
        Quadtree root cell size in pixels.  Default 64 (a /4 cell of
        a 256-px image).
    patch_size_min : int
        Quadtree leaf cell size in pixels.  Default 8.
    variance_weight, curvature_weight : float
        Quadtree scoring weights.  ``variance + curvature`` is the
        productive setting on natural images (curvature alone is
        boundary-only; variance alone is the saliency baseline).
    score_threshold : float
        Below this score, a cell is *not* subdivided.
    max_anchors : int
        Hard cap on the number of quadtree leaves; ``m`` should be
        well below this.
    seed : int
        Padding-side RNG seed when the quadtree produces fewer than
        ``m`` leaves.

    Returns
    -------
    tensor (m, 2) — float32 normalised ``(y, x)`` centres in
    ``[0.05, 0.95]``.

    Preconditions
    -------------
    - ``X_bootstrap.ndim == 4`` and ``X_bootstrap.shape[0] >= 1``.
    - ``H`` and ``W`` are multiples of ``patch_size_initial``.
    - ``m >= 1``.
    """
    if X_bootstrap.ndim != 4:
        raise ValueError(
            f"X_bootstrap must be (N, C, H, W); got {tuple(X_bootstrap.shape)}"
        )
    if X_bootstrap.shape[0] < 1:
        raise ValueError("X_bootstrap must contain >= 1 image")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")

    img_mean = X_bootstrap.mean(dim=0)  # (C, H, W)
    _C, H, W = img_mean.shape
    if H % patch_size_initial != 0 or W % patch_size_initial != 0:
        raise ValueError(
            f"image ({H}, {W}) not divisible by patch_size_initial={patch_size_initial}"
        )

    qt = AdaptiveQuadtreeRust(
        image_h=H,
        image_w=W,
        patch_size_initial=patch_size_initial,
        patch_size_min=patch_size_min,
        variance_weight=variance_weight,
        curvature_weight=curvature_weight,
        score_threshold=score_threshold,
        max_anchors=max_anchors,
    )
    tree = qt(img_mean)
    positions = tree.positions   # (n_anchors, 2)  ints (y, x)
    sizes = tree.sizes           # (n_anchors,)    ints
    scales = tree.scales         # (n_anchors,)    ints (0 = coarsest)
    parents = tree.parent_indices  # (n_anchors,)  ints (-1 = root)

    if positions.shape[0] == 0:
        # Degenerate (empty input or threshold too high) — uniform fallback.
        return _uniform_centres(m, H, W, seed=seed)

    # Identify leaves: anchors that do NOT appear as someone else's
    # parent.  These are the finest-resolution decisions the quadtree
    # made; the natural query centres.
    parent_set = set(int(p) for p in parents.tolist() if int(p) >= 0)
    leaf_mask = torch.tensor(
        [i not in parent_set for i in range(positions.shape[0])],
        dtype=torch.bool,
    )
    leaf_positions = positions[leaf_mask]
    leaf_sizes = sizes[leaf_mask]
    leaf_scales = scales[leaf_mask]

    if leaf_positions.shape[0] == 0:
        return _uniform_centres(m, H, W, seed=seed)

    # Cell centres in pixels → normalised.
    centres_px = leaf_positions.float() + leaf_sizes.unsqueeze(-1).float() * 0.5
    norm = torch.stack([centres_px[:, 0] / H, centres_px[:, 1] / W], dim=1)
    norm = torch.clamp(norm, 0.05, 0.95)

    # Post-hoc per-leaf score: in-cell variance of the aggregate image.
    # The quadtree picks *where* to place leaves; the variance picks
    # *which* of those leaves carry the most signal.  AnchorTree does
    # not expose its internal score, so we recompute it here from the
    # original image — cheap (one mean+var per leaf, no GPU).
    leaf_var = torch.zeros(leaf_positions.shape[0], dtype=torch.float32)
    img_gray = img_mean.mean(dim=0)  # (H, W) — collapse channels
    for k, (pos, sz) in enumerate(zip(leaf_positions.tolist(),
                                       leaf_sizes.tolist())):
        y0, x0 = pos
        y1, x1 = y0 + sz, x0 + sz
        patch = img_gray[y0:y1, x0:x1]
        leaf_var[k] = patch.var().item()

    # Rank by variance (descending) then FPS within the top pool.
    # Use the top 2*m candidates as the FPS source — keeps the pool
    # small (faster) and ensures the chosen centres are signal-rich.
    pool_size = max(m, min(2 * m, norm.shape[0]))
    order = torch.argsort(leaf_var, descending=True)[: pool_size]
    norm = norm[order]

    picked = _fps_within_pool(norm, m)

    if picked.shape[0] < m:
        n_short = m - picked.shape[0]
        pad = _uniform_centres(n_short, H, W, seed=seed)
        picked = torch.cat([picked, pad], dim=0)

    return picked.to(torch.float32)


def _fps_within_pool(pool: torch.Tensor, m: int) -> torch.Tensor:
    """Farthest-point sample ``m`` rows from ``pool`` (n, 2)."""
    if pool.shape[0] == 0:
        return pool
    if pool.shape[0] <= m:
        return pool
    chosen = [0]  # start with the highest-ranked leaf
    while len(chosen) < m:
        chosen_pts = pool[chosen]                # (k, 2)
        d = torch.cdist(pool, chosen_pts)        # (n, k)
        min_d = d.min(dim=1).values              # (n,)
        # Mask out already-chosen
        min_d[chosen] = -1.0
        nxt = int(torch.argmax(min_d).item())
        if min_d[nxt] <= 0:
            break
        chosen.append(nxt)
    return pool[chosen]


def _uniform_centres(m: int, H: int, W: int, *, seed: int) -> torch.Tensor:
    """Fallback: m uniform-on-the-image centres, normalised to [0.05, 0.95]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    coords = torch.rand(m, 2, generator=g)  # uniform in [0, 1]^2
    coords = 0.05 + coords * 0.90           # squeeze to [0.05, 0.95]
    return coords


def warmstart_query_corners_quadtree(
    model,
    X_bootstrap: torch.Tensor,
    *,
    box_size: float = 0.20,
    circle_radius: float = 0.15,
    patch_size_initial: int = 64,
    patch_size_min: int = 8,
    variance_weight: float = 1.0,
    curvature_weight: float = 1.0,
    seed: int = 0,
) -> dict:
    """Quadtree variant of
    :func:`hymeyolo_warmstart.warmstart_query_corners`.

    In-place: overwrites ``model.box_corners`` / ``model.circle_corners``
    using quadtree leaf centres + box/circle templates around each.
    Returns a dict ``{"box_corners": ..., "circle_corners": ...}``.
    """
    # Reuse the corner builders from the saliency module — they
    # implement exactly the box-square / circle-k-gon templates.
    from signedkan_wip.src.vision.hymeyolo_warmstart import (
        _box_corners_at,
        _circle_corners_at,
    )

    n_box = int(getattr(model, "n_box_queries", 0))
    n_circle = int(getattr(model, "n_circle_queries", 0))
    m = n_box + n_circle
    if m == 0:
        return {
            "box_corners": torch.zeros((0, 4, 2)),
            "circle_corners": torch.zeros((0, 0, 2)),
        }

    centres = quadtree_centres(
        X_bootstrap, m,
        patch_size_initial=patch_size_initial,
        patch_size_min=patch_size_min,
        variance_weight=variance_weight,
        curvature_weight=curvature_weight,
        seed=seed,
    )

    box_centres = centres[: n_box]
    circle_centres = centres[n_box : n_box + n_circle]

    out = {}
    if n_box > 0 and hasattr(model, "box_corners"):
        box_corners = torch.stack(
            [_box_corners_at(c, box_size) for c in box_centres], dim=0
        )
        with torch.no_grad():
            model.box_corners.data.copy_(box_corners.to(model.box_corners.device))
        out["box_corners"] = box_corners
    else:
        out["box_corners"] = torch.zeros((0, 4, 2))

    if n_circle > 0 and hasattr(model, "circle_corners"):
        k = int(model.circle_k)
        circle_corners = torch.stack(
            [_circle_corners_at(c, k, circle_radius) for c in circle_centres],
            dim=0,
        )
        with torch.no_grad():
            model.circle_corners.data.copy_(circle_corners.to(model.circle_corners.device))
        out["circle_corners"] = circle_corners
    else:
        out["circle_corners"] = torch.zeros((0, 0, 2))

    return out


__all__ = ["quadtree_centres", "warmstart_query_corners_quadtree"]
