"""Tests for the Tier-1 synthetic Sequential-HSiKAN benchmark.

Verifies:
  - shape contracts
  - class balance
  - oracle-baseline separability (the dataset is solvable in principle)

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/ §6.
"""
from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score

from signedkan_wip.src.sequence.synthetic_seq import (
    SynthConfig, make_dataset,
    fourier_class0_oracle, sigma_product_class1_oracle,
)


def test_dataset_shapes():
    cfg = SynthConfig(n_samples=64, L=128, seed=0)
    raw, sigma, labels = make_dataset(cfg)
    assert raw.shape == (64, 128, 2)
    assert sigma.shape == (64, 128)
    assert labels.shape == (64,)
    assert labels.dtype == torch.long


def test_dataset_class_balance():
    cfg = SynthConfig(n_samples=200, L=128, seed=0)
    _, _, labels = make_dataset(cfg)
    n0 = (labels == 0).sum().item()
    n1 = (labels == 1).sum().item()
    assert n0 == 100 and n1 == 100


def test_sigma_in_pm_one():
    cfg = SynthConfig(n_samples=32, L=128, seed=0)
    _, sigma, _ = make_dataset(cfg)
    # All sigma values must be exactly +1 or -1.
    assert ((sigma == 1.0) | (sigma == -1.0)).all()


def test_dataset_seed_reproducible():
    cfg = SynthConfig(n_samples=16, L=64, seed=42)
    raw_a, sigma_a, lab_a = make_dataset(cfg)
    raw_b, sigma_b, lab_b = make_dataset(cfg)
    assert torch.equal(raw_a, raw_b)
    assert torch.equal(sigma_a, sigma_b)
    assert torch.equal(lab_a, lab_b)


def test_dataset_different_seeds_produce_different_samples():
    cfg_a = SynthConfig(n_samples=16, L=64, seed=0)
    cfg_b = SynthConfig(n_samples=16, L=64, seed=1)
    raw_a, _, _ = make_dataset(cfg_a)
    raw_b, _, _ = make_dataset(cfg_b)
    assert not torch.equal(raw_a, raw_b)


# ─── Oracle-baseline separability sanity checks ──────────────────────


def test_fourier_oracle_separates_class0_from_class1():
    """A Fourier-energy detector on raw[:, 0] should achieve AUC ≥ 0.95
    for "is class 0 vs class 1" — i.e. the class-0 tone is meaningfully
    embedded above the noise floor."""
    cfg = SynthConfig(n_samples=200, L=256, seed=0,
                       freq_cycles=24, signal_amplitude=1.0,
                       noise_std=1.0)
    raw, _, labels = make_dataset(cfg)
    scores = fourier_class0_oracle(raw, cfg).numpy()
    # labels=0 → class 0 (high tone energy)
    is_class0 = (labels == 0).numpy()
    auc = roc_auc_score(is_class0, scores)
    assert auc >= 0.95, f"Fourier oracle AUC {auc:.3f} < 0.95"


def test_sigma_product_oracle_separates_class1_from_class0():
    """The σ-product variance oracle should rank class-0 sequences
    (random σ, high variance) above class-1 sequences (periodic σ,
    low variance) at AUC ≥ 0.95."""
    cfg = SynthConfig(n_samples=200, L=256, seed=0, info_period=7)
    _, sigma, labels = make_dataset(cfg)
    scores = sigma_product_class1_oracle(sigma, K=4).numpy()
    is_class0 = (labels == 0).numpy()
    auc = roc_auc_score(is_class0, scores)
    assert auc >= 0.95, f"σ-product variance oracle AUC {auc:.3f} < 0.95"


def test_no_nan_or_inf_in_samples():
    cfg = SynthConfig(n_samples=64, L=128, seed=0)
    raw, sigma, _ = make_dataset(cfg)
    assert torch.isfinite(raw).all()
    assert torch.isfinite(sigma).all()
