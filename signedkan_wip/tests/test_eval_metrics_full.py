"""Tests for the full-binary-metrics helper used by the Nature Comm
audit track."""
from __future__ import annotations

import numpy as np
import pytest

from signedkan_wip.experiments.eval.eval_metrics_full import full_binary_metrics


def test_full_metrics_perfect_predictions():
    logits = np.array([10.0, -10.0, 10.0, -10.0])
    signs = np.array([1, -1, 1, -1])
    m = full_binary_metrics(logits, signs)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["precision_pos"] == 1.0
    assert m["recall_pos"] == 1.0
    assert m["precision_neg"] == 1.0
    assert m["recall_neg"] == 1.0
    assert m["n"] == 4
    assert m["n_pos"] == 2
    assert m["n_neg"] == 2


def test_full_metrics_random_predictions_yield_chance_auc():
    rng = np.random.default_rng(0)
    n = 500
    logits = rng.standard_normal(n)
    signs = rng.choice([-1, 1], n)
    m = full_binary_metrics(logits, signs)
    assert 0.40 <= m["auc"] <= 0.60


def test_full_metrics_accepts_01_signs():
    """The helper accepts {0, 1} as well as {-1, +1}."""
    logits = np.array([5.0, -5.0])
    signs_01 = np.array([1, 0])
    signs_pm1 = np.array([1, -1])
    m_01 = full_binary_metrics(logits, signs_01)
    m_pm1 = full_binary_metrics(logits, signs_pm1)
    assert m_01["accuracy"] == m_pm1["accuracy"]
    assert m_01["f1_macro"] == m_pm1["f1_macro"]


def test_full_metrics_accepts_probs_directly():
    probs = np.array([0.9, 0.1, 0.7, 0.3])
    signs = np.array([1, -1, 1, -1])
    m = full_binary_metrics(probs, signs, is_logits=False)
    assert m["accuracy"] == 1.0


def test_full_metrics_emits_all_keys_for_jsonl():
    """Sanity: every field is a JSON-serialisable scalar."""
    import json
    logits = np.array([1.0, -1.0, 2.0])
    signs = np.array([1, -1, 1])
    m = full_binary_metrics(logits, signs)
    # Each value must be json-serialisable.
    json.dumps(m)
    # The Nature-comm comparison table needs all of these.
    for key in ("auc", "accuracy", "f1_macro", "f1_pos", "f1_neg",
                  "precision_pos", "recall_pos",
                  "precision_neg", "recall_neg", "n", "n_pos", "n_neg"):
        assert key in m, f"missing key: {key}"
