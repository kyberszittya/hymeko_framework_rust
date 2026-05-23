"""Brain-Score-style scoring for the GömbSoma cortical benchmark.

The Brain-Score 2018 protocol (Schrimpf et al.) is, simplified:

1. Per (model, ROI) pair, reduce model features to a small
   number of PLS components.
2. K-fold cross-validate Ridge regression: hold out the
   k-th image, predict its voxel responses from features,
   compute voxel-wise Pearson r between predicted and actual.
3. Mean r across voxels and folds → raw score.
4. Noise ceiling: split-half Pearson r across subjects,
   Spearman-Brown-corrected, → upper bound on what's predictable.
5. Brain-Score = raw / noise_ceiling.

We ship our own implementation rather than depend on the
upstream ``brainscore`` Python package because (a) it's not
installed in the active env and (b) for tonight's slice the
synthetic-data smoke is sufficient — real data + Brain-Score
API integration is a future slice (per the plan).

Object-oriented commitment: :class:`BrainScore` is a frozen
dataclass; :class:`BrainScorer` holds CV / PLS / Ridge config
as instance state. Comparison of two models on the same ROI
uses *identical* CV folds (deterministic seed) so the
paired-bootstrap delta is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class BrainScore:
    """One ROI's score for one (model, dataset) pair.

    Attributes
    ----------
    roi : str
        ROI name (e.g.\\ ``"V1"``, ``"V2"``, ``"V4"``).
    r_squared : float
        Raw mean voxel-wise Pearson r across CV folds.
    noise_ceiling : float
        Upper-bound predictability inferred from inter-subject
        consistency. ``raw / noise_ceiling`` would be 1.0 if the
        model is at the cortical-data ceiling.
    noise_ceiling_corrected : float
        ``r_squared / noise_ceiling`` (clamped to ``[0, 1]``).
    n_voxels : int
        ROI voxel count.
    n_pls_components : int
        PLS components used.
    n_cv_folds : int
        K-fold CV depth.
    """

    roi: str
    r_squared: float
    noise_ceiling: float
    noise_ceiling_corrected: float
    n_voxels: int
    n_pls_components: int
    n_cv_folds: int


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays. NaN/constant-input safe."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a_centred = a - a.mean()
    b_centred = b - b.mean()
    denom = float(np.sqrt((a_centred ** 2).sum() * (b_centred ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((a_centred * b_centred).sum() / denom)


def _voxelwise_r(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Voxel-wise Pearson r. Both arrays ``[n_images, n_voxels]``."""
    if pred.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs actual {actual.shape}"
        )
    n_voxels = pred.shape[1]
    rs = np.zeros(n_voxels, dtype=np.float64)
    for v in range(n_voxels):
        rs[v] = _pearson_r(pred[:, v], actual[:, v])
    return rs


class BrainScorer:
    """Brain-Score-style scorer.

    Parameters
    ----------
    n_pls_components
        Number of PLS components in the feature-side reduction.
        Brain-Score 2018 default is 25.
    ridge_alpha
        L2 regularisation for the per-voxel Ridge.
    n_cv_folds
        K-fold cross-validation depth on images.
    seed
        RNG seed (controls CV-fold splits; the scorer is otherwise
        deterministic).
    """

    def __init__(
        self,
        n_pls_components: int = 25,
        ridge_alpha: float = 1.0,
        n_cv_folds: int = 5,
        seed: int = 0,
    ) -> None:
        if n_pls_components < 1:
            raise ValueError("n_pls_components must be >= 1")
        if n_cv_folds < 2:
            raise ValueError("n_cv_folds must be >= 2")
        self.n_pls_components = int(n_pls_components)
        self.ridge_alpha = float(ridge_alpha)
        self.n_cv_folds = int(n_cv_folds)
        self.seed = int(seed)

    def score(
        self,
        features: torch.Tensor | np.ndarray,
        roi_signal: torch.Tensor | np.ndarray,
        roi: str,
        noise_ceiling: float | None = None,
    ) -> BrainScore:
        """Score one ROI for one model.

        Parameters
        ----------
        features
            ``[n_images, d_feat]`` per-image flattened model features.
        roi_signal
            ``[n_images, n_voxels]`` — subject-mean (or single-subject)
            ROI signal per image. For full per-subject noise-ceiling,
            pass :meth:`noise_ceiling` separately and supply the
            ``noise_ceiling`` argument here.
        roi
            ROI name (used in the result, not in any computation).
        noise_ceiling
            Pre-computed noise ceiling for this ROI. If ``None``,
            the score's ``noise_ceiling`` is set to 1.0 (no
            correction).

        Returns
        -------
        BrainScore
            Frozen dataclass with raw + corrected scores.
        """
        X = _to_numpy(features)
        Y = _to_numpy(roi_signal)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"image-axis mismatch: features {X.shape[0]} vs roi {Y.shape[0]}"
            )

        n_images, d_feat = X.shape
        n_voxels = Y.shape[1]
        if n_images <= self.n_cv_folds:
            raise ValueError(
                f"need n_images > n_cv_folds; got {n_images} <= {self.n_cv_folds}"
            )
        # `n_components` upper bound in sklearn 1.8+ is
        # `min(n_train_samples, n_features) - 1` (the deflation
        # step needs one degree of freedom left over). Inside the
        # K-fold CV loop, the *training* sample count is
        # n_images * (k-1) / k. Clip conservatively so small
        # fixtures (Phase 12's cortical sweep at deep binning ×
        # d_hidden=16 / n_images=30 / k=4 gives n_train ≈ 22, so
        # the cap is 21) do not trip the sklearn upper-bound
        # check.
        n_train_min = (n_images * (self.n_cv_folds - 1)) // self.n_cv_folds
        n_components = max(
            1,
            min(self.n_pls_components, d_feat - 1, n_train_min - 1),
        )

        kf = KFold(
            n_splits=self.n_cv_folds, shuffle=True, random_state=self.seed
        )
        fold_rs: list[np.ndarray] = []
        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            # PLS reduce on training set only.
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_tr, Y_tr)
            X_tr_red = pls.transform(X_tr)
            X_te_red = pls.transform(X_te)
            # Ridge per voxel — sklearn does it as a multivariate
            # output in a single call.
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
            ridge.fit(X_tr_red, Y_tr)
            Y_te_pred = ridge.predict(X_te_red)
            fold_rs.append(_voxelwise_r(Y_te_pred, Y_te))
        mean_r = float(np.mean(np.concatenate(fold_rs)))

        nc = float(noise_ceiling) if noise_ceiling is not None else 1.0
        if nc < 1e-9:
            corrected = 0.0
        else:
            corrected = max(0.0, min(1.0, mean_r / nc))

        return BrainScore(
            roi=roi,
            r_squared=mean_r,
            noise_ceiling=nc,
            noise_ceiling_corrected=corrected,
            n_voxels=int(n_voxels),
            n_pls_components=int(n_components),
            n_cv_folds=int(self.n_cv_folds),
        )

    def noise_ceiling(
        self,
        roi_per_subject: torch.Tensor | np.ndarray,
        roi: str = "",
    ) -> float:
        """Split-half noise ceiling for one ROI.

        Parameters
        ----------
        roi_per_subject
            ``[n_subjects, n_images, n_voxels]``.
        roi
            ROI name (unused in the computation, present for
            symmetry).

        Returns
        -------
        float
            Spearman-Brown-corrected split-half voxel-mean Pearson r.

        Notes
        -----
        For each random split of subjects into two halves:
          1. Compute each half's voxel-image response (mean across
             subjects in the half).
          2. Compute voxel-wise Pearson r between the two halves
             across images.
          3. Take the mean across voxels.
        Repeat for several random splits; average; correct via
        ``r_corrected = 2 r / (1 + r)`` (Spearman-Brown for half
        → full reliability).
        """
        Y = _to_numpy(roi_per_subject)
        if Y.ndim != 3:
            raise ValueError(
                f"need [S, N, V]; got {Y.shape}"
            )
        n_subjects, n_images, n_voxels = Y.shape
        if n_subjects < 2:
            # Single subject: no split possible; conservative upper
            # bound is 1.0 (no correction).
            return 1.0

        rng = np.random.default_rng(self.seed)
        n_iter = min(10, max(2, n_subjects))
        rs = []
        for _ in range(n_iter):
            perm = rng.permutation(n_subjects)
            half = n_subjects // 2
            a_idx = perm[:half]
            b_idx = perm[half: 2 * half]
            a_mean = Y[a_idx].mean(axis=0)  # (N, V)
            b_mean = Y[b_idx].mean(axis=0)
            voxel_rs = _voxelwise_r(a_mean, b_mean)
            rs.append(float(np.mean(voxel_rs)))
        r = float(np.mean(rs))
        # Spearman-Brown half → full.
        denom = 1.0 + r
        if abs(denom) < 1e-9:
            return 0.0
        r_corr = 2.0 * r / denom
        # Clamp to [0, 1] for downstream divide-by-ceiling safety.
        return max(0.0, min(1.0, r_corr))


# ─── Convenience: score a model on every ROI ───────────────────────


def score_all_rois(
    scorer: BrainScorer,
    features: torch.Tensor | np.ndarray,
    roi_signals: dict[str, torch.Tensor],
) -> dict[str, BrainScore]:
    """Score one model against every ROI in ``roi_signals``.

    ``roi_signals`` is the same dict shape :class:`SyntheticCorticalDataset`
    exposes: ``ROI → Tensor[n_subjects, n_images, n_voxels]``. The
    function computes:

      1. Per-ROI noise ceiling (subject-aware).
      2. Subject-mean ROI signal as the regression target.
      3. Per-ROI BrainScore.
    """
    out: dict[str, BrainScore] = {}
    for roi_name, sig in roi_signals.items():
        nc = scorer.noise_ceiling(sig, roi=roi_name)
        mean_sig = _to_numpy(sig).mean(axis=0)  # (N, V) subject-mean
        out[roi_name] = scorer.score(features, mean_sig, roi=roi_name, noise_ceiling=nc)
    return out


__all__ = [
    "BrainScore",
    "BrainScorer",
    "score_all_rois",
]
