"""Saliency-driven warm-start for HymeYOLO query corners.

Replaces the seed-dependent fixed-base + Gaussian-noise init of
``RicciHyMeYOLOMulti.box_corners`` / ``circle_corners`` with a
deterministic saliency-coverage init derived from a small
bootstrap batch of training images.

The premise (from the 2026-05-13 5-seed backfill, open issue #3,
and the 2026-05-16 night-shift YOLO-parity ladder, lever #1):
the per-seed variance σ ≈ 0.18 on ``+ricci-mod`` mAP_50 is
dominated by *initialisation* brittleness — random query positions
land different seeds in different local minima of the offset head.
Replacing the random init with a spatially-distributed,
data-aware one removes that source of variance.

The approach:

  1. Compute a model-free saliency over the bootstrap batch
     (mean absolute pixel magnitude per spatial location).
  2. Gaussian-smooth + normalise → a probability map ``p(y, x)``.
  3. Farthest-point sample ``N_box + N_circle`` centres from the
     image plane, weighted by ``p``.
  4. Emit corner patterns (axis-aligned square for box queries;
     regular k-gon for circle queries) around each centre.
  5. Clamp to ``[0.05, 0.95]`` (same bound as the original init).
  6. Overwrite ``model.box_corners`` and ``model.circle_corners``
     in place.

Determinism: given the same bootstrap batch and seed, the output
is bit-identical. The function does not mutate any global RNG.

Plan: ``docs/plans/2026-05-16-hymeyolo-warmstart-query-init/``.
"""
from __future__ import annotations

import math
from typing import Callable

import torch


# ---------------------------------------------------------------------
# Saliency
# ---------------------------------------------------------------------


def _default_saliency(X: torch.Tensor) -> torch.Tensor:
    """Mean-of-absolute-pixels saliency: ``|X|.mean(dim=channel)``.

    Args
    ----
    X : tensor (N, C, H, W) — bootstrap batch.

    Returns
    -------
    tensor (H, W) — bootstrap-mean saliency, with no model
    dependency. Good on Cluttered MNIST (bright digits on dim
    background); for inverted-polarity datasets, pass a different
    ``saliency_fn`` to ``warmstart_query_corners``.
    """
    per_image = X.abs().mean(dim=1)            # (N, H, W)
    return per_image.mean(dim=0)               # (H, W)


def _gaussian_smooth_2d(
    S: torch.Tensor, sigma_px: float = 2.0,
) -> torch.Tensor:
    """Separable Gaussian smoothing without an extra dependency.

    Kernel width = 2 * ceil(2σ) + 1 (truncated at 2σ each side).
    Reflect padding at the borders so corner saliency does not
    decay to zero artificially.
    """
    if sigma_px <= 0:
        return S
    half = int(math.ceil(2.0 * sigma_px))
    xs = torch.arange(-half, half + 1, dtype=S.dtype, device=S.device)
    kernel_1d = torch.exp(-(xs ** 2) / (2.0 * sigma_px ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    # (H, W) → (1, 1, H, W) for conv2d.
    x = S.unsqueeze(0).unsqueeze(0)
    k_h = kernel_1d.view(1, 1, -1, 1)
    k_w = kernel_1d.view(1, 1, 1, -1)
    pad_h = (half, half, 0, 0)
    pad_w = (0, 0, half, half)
    # Horizontal pass.
    x_pad = torch.nn.functional.pad(x, pad_h, mode="reflect")
    x = torch.nn.functional.conv2d(x_pad, k_h)
    # Vertical pass.
    x_pad = torch.nn.functional.pad(x, pad_w, mode="reflect")
    x = torch.nn.functional.conv2d(x_pad, k_w)
    return x.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------


def _farthest_point_sample(
    p: torch.Tensor,
    m: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Saliency-weighted farthest-point sampling on the (H, W) grid.

    The first centre is sampled from ``p`` directly (so a strongly
    peaked saliency biases towards that peak). Each subsequent
    centre maximises ``min_{c ∈ chosen} d(grid, c) × sqrt(p(grid))``
    — i.e., farthest from already-chosen centres, biased toward
    higher-saliency cells.

    Args
    ----
    p : tensor (H, W) — non-negative; normalised so ``p.sum() == 1``.
    m : int — number of centres to return.
    seed : int — RNG seed for the first centre and ties.

    Returns
    -------
    tensor (m, 2) — (y, x) integer grid coordinates of the chosen
    centres, in [0, H) × [0, W).
    """
    if m <= 0:
        return torch.zeros((0, 2), dtype=torch.long, device=p.device)
    H, W = p.shape
    p_flat = p.flatten()
    # Guard against an all-zero saliency: fall back to uniform.
    if p_flat.sum().item() <= 0:
        p_flat = torch.ones_like(p_flat) / p_flat.numel()

    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Draw the first centre by inverse-CDF sampling from p.
    cdf = torch.cumsum(p_flat.cpu().double(), dim=0)
    cdf = cdf / cdf[-1]  # robust normalise
    u0 = torch.rand((), generator=gen).item()
    first_idx = int(torch.searchsorted(cdf, torch.tensor(u0)).item())
    first_idx = min(first_idx, p_flat.numel() - 1)

    ys = torch.arange(H, device=p.device).unsqueeze(1).expand(H, W).flatten().float()
    xs = torch.arange(W, device=p.device).unsqueeze(0).expand(H, W).flatten().float()

    chosen = [first_idx]
    # Squared distance from each cell to the nearest chosen centre.
    cy = float(first_idx // W)
    cx = float(first_idx % W)
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2

    sqrt_p = torch.sqrt(p_flat.clamp_min(1e-12))
    for _ in range(m - 1):
        score = d2 * sqrt_p
        # Forbid re-picking the same cell.
        for c in chosen:
            score[c] = -1.0
        idx = int(torch.argmax(score).item())
        chosen.append(idx)
        cy = float(idx // W)
        cx = float(idx % W)
        d2_new = (ys - cy) ** 2 + (xs - cx) ** 2
        d2 = torch.minimum(d2, d2_new)

    out = torch.zeros((m, 2), dtype=torch.long, device=p.device)
    for i, c in enumerate(chosen):
        out[i, 0] = c // W
        out[i, 1] = c % W
    return out


# ---------------------------------------------------------------------
# Corner pattern emission
# ---------------------------------------------------------------------


def _box_corners_at(centre_norm: torch.Tensor, size: float) -> torch.Tensor:
    """4 axis-aligned corners around ``centre_norm`` in normalised
    image coords. ``centre_norm`` is (2,): (y_norm, x_norm)."""
    cy, cx = centre_norm[0], centre_norm[1]
    half = size * 0.5
    return torch.stack([
        torch.stack([cx - half, cy - half]),  # top-left
        torch.stack([cx + half, cy - half]),  # top-right
        torch.stack([cx + half, cy + half]),  # bottom-right
        torch.stack([cx - half, cy + half]),  # bottom-left
    ], dim=0)


def _circle_corners_at(
    centre_norm: torch.Tensor, k: int, radius: float,
) -> torch.Tensor:
    """k corners on a regular k-gon of radius ``radius`` around
    ``centre_norm``. Matches the orientation of ``_circle_init``."""
    cy, cx = centre_norm[0], centre_norm[1]
    angles = torch.arange(k, dtype=centre_norm.dtype, device=centre_norm.device) \
                  * (2.0 * math.pi / k)
    out = torch.zeros((k, 2), dtype=centre_norm.dtype, device=centre_norm.device)
    out[:, 0] = cx + radius * torch.cos(angles)
    out[:, 1] = cy + radius * torch.sin(angles)
    return out


# ---------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------


def warmstart_query_corners(
    model,
    X_bootstrap: torch.Tensor,
    *,
    box_size: float = 0.20,
    circle_radius: float = 0.15,
    saliency_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    seed: int = 0,
    smooth_sigma_px: float = 2.0,
) -> dict[str, torch.Tensor]:
    """In-place warm-start of ``model.box_corners`` and
    ``model.circle_corners`` from a saliency map.

    Preconditions
    -------------
    * ``model`` has integer attributes ``n_box_queries``,
      ``n_circle_queries``, ``circle_k``.
    * ``model.box_corners`` is an ``nn.Parameter`` of shape
      ``(n_box_queries, 4, 2)`` when ``n_box_queries > 0``; ditto
      ``model.circle_corners`` of shape
      ``(n_circle_queries, circle_k, 2)`` when
      ``n_circle_queries > 0``. Both may be absent if their query
      count is zero — the function tolerates that.
    * ``X_bootstrap`` has shape ``(N, C, H, W)``, finite.

    Postconditions
    --------------
    * ``model.box_corners`` and ``model.circle_corners`` are
      overwritten in place. Parameter count of ``model`` is
      unchanged.
    * All emitted corner positions are in ``[0.05, 0.95]``.
    * No other parameters of ``model`` are touched.
    * No global RNG is consumed (deterministic given inputs).

    Returns
    -------
    A dict with keys ``box_corners`` and ``circle_corners`` (the
    new tensor values), for logging.
    """
    if X_bootstrap.ndim != 4:
        raise ValueError(
            f"X_bootstrap must have shape (N, C, H, W); got "
            f"{tuple(X_bootstrap.shape)}"
        )
    if X_bootstrap.shape[0] == 0:
        raise ValueError("X_bootstrap must contain at least one image")
    if not torch.isfinite(X_bootstrap).all().item():
        raise ValueError("X_bootstrap contains non-finite values")

    sal_fn = saliency_fn or _default_saliency
    S = sal_fn(X_bootstrap)
    if S.ndim != 2:
        raise ValueError(
            f"saliency_fn must return (H, W); got {tuple(S.shape)}"
        )
    H, W = S.shape
    S = _gaussian_smooth_2d(S, sigma_px=smooth_sigma_px)
    S = torch.clamp(S, min=0.0)
    total = S.sum()
    if total.item() <= 0:
        # Degenerate bootstrap (all-zero images): fall back to uniform.
        S = torch.ones_like(S)
        total = S.sum()
    p = S / total

    n_box = int(getattr(model, "n_box_queries", 0))
    n_circle = int(getattr(model, "n_circle_queries", 0))
    m = n_box + n_circle
    if m == 0:
        return {
            "box_corners": torch.zeros((0, 4, 2)),
            "circle_corners": torch.zeros((0, 0, 2)),
        }

    grid_centres = _farthest_point_sample(p, m, seed=seed)  # (m, 2) int

    # Convert grid coords (y, x) → normalised (y_norm, x_norm).
    # Use cell centres: (i + 0.5) / H, etc.
    norm = torch.stack([
        (grid_centres[:, 0].float() + 0.5) / H,
        (grid_centres[:, 1].float() + 0.5) / W,
    ], dim=1)  # (m, 2): (y_norm, x_norm)

    out: dict[str, torch.Tensor] = {}

    if n_box > 0 and hasattr(model, "box_corners"):
        box_list = []
        for i in range(n_box):
            c = norm[i]
            box_list.append(_box_corners_at(c, box_size))
        new_box = torch.stack(box_list, dim=0).clamp(0.05, 0.95)
        new_box = new_box.to(
            dtype=model.box_corners.dtype,
            device=model.box_corners.device,
        )
        if new_box.shape != model.box_corners.shape:
            raise ValueError(
                f"warm-start box-corner shape {tuple(new_box.shape)} "
                f"does not match model.box_corners "
                f"{tuple(model.box_corners.shape)}"
            )
        with torch.no_grad():
            model.box_corners.copy_(new_box)
        out["box_corners"] = new_box.detach().clone()

    if n_circle > 0 and hasattr(model, "circle_corners"):
        circle_k = int(getattr(model, "circle_k", 0))
        if circle_k <= 0:
            raise ValueError(
                "model has n_circle_queries > 0 but no usable circle_k"
            )
        circ_list = []
        for i in range(n_circle):
            c = norm[n_box + i]
            circ_list.append(
                _circle_corners_at(c, circle_k, circle_radius)
            )
        new_circ = torch.stack(circ_list, dim=0).clamp(0.05, 0.95)
        new_circ = new_circ.to(
            dtype=model.circle_corners.dtype,
            device=model.circle_corners.device,
        )
        if new_circ.shape != model.circle_corners.shape:
            raise ValueError(
                f"warm-start circle-corner shape {tuple(new_circ.shape)} "
                f"does not match model.circle_corners "
                f"{tuple(model.circle_corners.shape)}"
            )
        with torch.no_grad():
            model.circle_corners.copy_(new_circ)
        out["circle_corners"] = new_circ.detach().clone()

    return out
