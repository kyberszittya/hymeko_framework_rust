"""Drop-in extended metrics helper for signed-link prediction.

Used by the Nature Comm submission track to fill the per-class
precision/recall/accuracy fields that the existing eval helpers
in ``run_final_cell.py`` and ``run_ntuples_mixed.py`` don't emit.

Plan: docs/plans/2026-05-17-nature-comm-leakage-audit/.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score,
)


def full_binary_metrics(
    logits_or_probs,
    signs,
    threshold: float = 0.5,
    is_logits: bool = True,
) -> dict:
    """Compute the full SE-SGformer/DADSGNN-comparable metric set
    from binary link-prediction outputs.

    Parameters
    ----------
    logits_or_probs : array-like
        Per-edge model output. If ``is_logits=True`` (default), interpreted
        as logits and sigmoid-transformed; otherwise as probabilities.
    signs : array-like
        Per-edge true sign in {-1, +1} (or {0, 1}).
    threshold : float, default 0.5
        Decision threshold on the probability for the hard prediction.
    is_logits : bool, default True
        Whether ``logits_or_probs`` is raw logits or pre-sigmoid'd probs.

    Returns
    -------
    dict with keys: auc, accuracy, f1_macro, f1_pos, f1_neg,
        precision_pos, recall_pos, precision_neg, recall_neg, n,
        n_pos, n_neg. All scalars; suitable for JSONL emission.
    """
    arr = np.asarray(logits_or_probs)
    if is_logits:
        probs = 1.0 / (1.0 + np.exp(-arr))
    else:
        probs = arr
    preds = (probs > threshold).astype(int)
    y_arr = np.asarray(signs)
    # Normalise {-1, +1} → {0, 1}; treats {0, 1} input as-is.
    y01 = (y_arr == 1).astype(int) if y_arr.min() < 0 else y_arr.astype(int)
    n_classes = len(np.unique(y01))
    return {
        "auc": (float(roc_auc_score(y01, probs)) if n_classes > 1
                  else float("nan")),
        "accuracy":      float(accuracy_score(y01, preds)),
        "f1_macro":      float(f1_score(y01, preds, average="macro", zero_division=0)),
        "f1_pos":        float(f1_score(y01, preds, pos_label=1, zero_division=0)),
        "f1_neg":        float(f1_score(y01, preds, pos_label=0, zero_division=0)),
        "precision_pos": float(precision_score(y01, preds, pos_label=1, zero_division=0)),
        "recall_pos":    float(recall_score(y01, preds, pos_label=1, zero_division=0)),
        "precision_neg": float(precision_score(y01, preds, pos_label=0, zero_division=0)),
        "recall_neg":    float(recall_score(y01, preds, pos_label=0, zero_division=0)),
        "n":             int(y01.shape[0]),
        "n_pos":         int(y01.sum()),
        "n_neg":         int(y01.shape[0] - y01.sum()),
    }
