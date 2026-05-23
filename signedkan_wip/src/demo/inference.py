"""Load a checkpoint, run inference, compute metrics.

Designed for ``MixedAritySignedKAN`` checkpoints saved by
``run_final_cell.py --save-checkpoint``. Other models work in principle
provided they expose ``classifier`` + ``encode_edges`` and the
checkpoint includes a matching ``inference_bundle``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from ..datasets import SignedGraph, load as load_signed_graph
from .checkpoint import (
    CheckpointMeta, InferenceBundle, load_checkpoint,
)


@dataclass
class ModelBundle:
    """A loaded model + cfg + dataset + (optional) precomputed test-set
    inference inputs + (optional) external classifier module."""

    model: torch.nn.Module
    cfg: Any
    meta: CheckpointMeta
    graph: SignedGraph
    inference_bundle: InferenceBundle | None
    device: torch.device
    classifier: torch.nn.Module | None = None

    @property
    def dataset(self) -> str:
        return self.meta.dataset

    @property
    def n_nodes(self) -> int:
        return self.graph.n_nodes

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def alpha_vector(self) -> np.ndarray | None:
        """Learned αₖ over arities, or None if the model lacks
        ``arity_logits``."""
        logits = getattr(self.model, "arity_logits", None)
        if logits is None:
            return None
        with torch.no_grad():
            a = torch.softmax(logits, dim=0)
        return a.detach().cpu().numpy()

    def tuple_labels(self) -> list[str]:
        """Human labels matching the αₖ entries (one per tuple spec)."""
        labels: list[str] = []
        for spec in self.meta.tuple_specs:
            kind = spec[0]
            if kind == "walk":
                walk_len = spec[2]
                labels.append(f"w{walk_len}")
            else:
                k = spec[1]
                labels.append(f"c{k}")
        return labels


@dataclass
class PredictionResult:
    """Test-set predictions and metrics."""

    edges: np.ndarray            # (n_test, 2) int64
    true_signs: np.ndarray       # (n_test,)   {-1, +1}
    predicted_prob: np.ndarray   # (n_test,)   p(sign = +1)
    predicted_sign: np.ndarray   # (n_test,)   {-1, +1}
    auc: float
    f1_macro: float
    accuracy: float
    roc_curve_xy: tuple[np.ndarray, np.ndarray]   # (fpr, tpr)

    def confusion(self) -> dict[str, int]:
        y_true = (self.true_signs == 1).astype(int)
        y_pred = (self.predicted_sign == 1).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def load_bundle(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> ModelBundle:
    """Load a checkpoint and the matching dataset; returns a bundle."""
    dev = torch.device(device)
    model, cfg, meta, inf_bundle, classifier = load_checkpoint(
        checkpoint_path, map_location=str(dev),
    )
    g = load_signed_graph(meta.dataset)
    if g.n_nodes != meta.n_nodes:
        print(
            f"[demo] WARNING: meta.n_nodes={meta.n_nodes} but current "
            f"load('{meta.dataset}').n_nodes={g.n_nodes} — checkpoint may "
            f"be stale.",
        )
    return ModelBundle(
        model=model, cfg=cfg, meta=meta, graph=g,
        inference_bundle=inf_bundle, device=dev,
        classifier=classifier,
    )


def predict_test_edges(bundle: ModelBundle) -> PredictionResult:
    """Compute predictions + metrics using the bundled inference inputs.

    Raises if the checkpoint did not include an ``inference_bundle``.
    To re-enumerate tuples from scratch, train + save again with
    ``--save-checkpoint`` (run_final_cell builds the bundle).
    """
    if bundle.inference_bundle is None:
        raise RuntimeError(
            "checkpoint has no inference_bundle. Re-save with "
            "`run_final_cell.py --save-checkpoint <path>` to bundle the "
            "precomputed test-set tuples alongside the state_dict."
        )
    ib = bundle.inference_bundle
    model = bundle.model
    dev = bundle.device

    # Move query edges to device.
    q_te = torch.as_tensor(ib.query_edges, dtype=torch.long, device=dev)
    true_signs = np.asarray(ib.true_signs).astype(np.int64).reshape(-1)
    # `per_arity_te` is the same opaque structure used by run_final_cell.
    # Two dispatch paths:
    #   - external classifier (Bitcoin / OTC etc.): clf(encode_edges(...))
    #   - internal classifier (Slashdot etc.):       model.classifier(encode_edges(...))
    classifier = bundle.classifier
    with torch.no_grad():
        edge_emb = model.encode_edges(ib.per_arity_te, query_edges=q_te)
        if classifier is not None:
            logits = classifier(edge_emb)
        else:
            logits = model.classifier(edge_emb)
    logits_np = logits.detach().cpu().numpy().astype(np.float64).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = np.where(probs > 0.5, 1, -1).astype(np.int64)

    y_true = (true_signs == 1).astype(int)
    y_pred = (preds == 1).astype(int)
    auc = (float(roc_auc_score(y_true, probs))
           if len(np.unique(y_true)) > 1 else float("nan"))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float((preds == true_signs).mean())
    fpr, tpr, _ = roc_curve(y_true, probs)

    return PredictionResult(
        edges=np.asarray(ib.query_edges).astype(np.int64),
        true_signs=true_signs,
        predicted_prob=probs, predicted_sign=preds,
        auc=auc, f1_macro=f1m, accuracy=acc,
        roc_curve_xy=(fpr, tpr),
    )


__all__ = [
    "ModelBundle",
    "PredictionResult",
    "load_bundle",
    "predict_test_edges",
]
