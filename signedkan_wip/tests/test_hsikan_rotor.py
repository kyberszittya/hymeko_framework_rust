"""Tests for the leakage-free HSiKAN rotor driver and its tuning knobs.

Covers the new `TrainConfig` levers (weight_decay, grad_clip, class-weighted
BCE, early-stop) added on 2026-06-17 to close the SiGAT gap. The load-bearing
leakage gate (shuffle -> chance) is exercised at experiment scale, not here:
toy epochs do not converge, so a unit-scale ~0.5 assertion would be flaky. These
tests pin the *pure* helpers deterministically and smoke the full `run` path on
the small bitcoin_otc fixture at 3 epochs.

Run: pytest -p no:randomly signedkan_wip/tests/test_hsikan_rotor.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.experiments.runs.run_hsikan_rotor import (
    TrainConfig,
    _drop_train_pairs,
    _optimise,
    _pos_weight,
    run,
)

CPU = torch.device("cpu")


# --------------------------------------------------------------------------- #
# Unit — pure helpers                                                         #
# --------------------------------------------------------------------------- #
def test_train_config_defaults_reproduce_verify_recipe():
    """Defaults must equal the 2026-06-17 verify recipe so prior numbers hold."""
    c = TrainConfig()
    assert (c.lr, c.weight_decay, c.grad_clip) == (1e-3, 1e-5, 0.0)
    assert c.class_weight is False and c.early_stop is False
    assert c.n_epochs == 120


def test_pos_weight_balances_majority_positive():
    # 4 positive, 1 negative -> pos_weight = n_neg/n_pos = 1/4.
    pw = _pos_weight(np.array([1, 1, 1, 1, -1]), CPU)
    assert pw is not None
    assert abs(float(pw) - 0.25) < 1e-6


def test_pos_weight_degenerate_returns_none():
    assert _pos_weight(np.array([1, 1, 1]), CPU) is None
    assert _pos_weight(np.array([-1, -1]), CPU) is None


def test_drop_train_pairs_removes_undirected_overlap():
    edges = np.array([[0, 1], [2, 3], [1, 0]])
    signs = np.array([1, -1, 1])
    e, s = _drop_train_pairs(edges, signs, {(0, 1)})
    # (0,1) and its reverse (1,0) both drop; only (2,3) survives.
    assert e.tolist() == [[2, 3]]
    assert s.tolist() == [-1]


def test_drop_train_pairs_empty_is_noop():
    e = np.zeros((0, 2), dtype=np.int64)
    s = np.zeros((0,), dtype=np.int64)
    e2, s2 = _drop_train_pairs(e, s, {(0, 1)})
    assert len(e2) == 0 and len(s2) == 0


def test_optimise_rejects_nonpositive_epochs():
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    with pytest.raises(AssertionError):
        _optimise(opt, lambda: m(torch.ones(1, 1)).sum(), lambda: 0.5, [m],
                  TrainConfig(n_epochs=0))


def test_optimise_early_stop_restores_best_val_state():
    """The params after _optimise must equal the snapshot at the best-val epoch,
    and patience must trigger an early break."""
    torch.manual_seed(0)
    m = torch.nn.Linear(2, 1, bias=False)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    x, target = torch.ones(1, 2), torch.zeros(1)

    def train_loss():
        return ((m(x) - target) ** 2).mean()

    # best val at the 2nd eval; then two non-improving steps trip patience=2.
    schedule = iter([0.50, 0.90, 0.40, 0.40, 0.40])
    seen: list[torch.Tensor] = []

    def val_auroc():
        seen.append(m.weight.detach().clone())
        return next(schedule)

    best = _optimise(opt, train_loss, val_auroc, [m],
                     TrainConfig(early_stop=True, eval_every=1, patience=2,
                                 n_epochs=20, lr=0.5))
    assert abs(best - 0.90) < 1e-9
    assert len(seen) == 4              # stopped early, not all 20 epochs
    assert torch.allclose(m.weight, seen[1])   # restored to best-val snapshot


def test_optimise_no_early_stop_returns_nan_and_runs_all_epochs():
    torch.manual_seed(0)
    m = torch.nn.Linear(2, 1, bias=False)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = torch.ones(1, 2)
    calls = {"n": 0}

    def train_loss():
        calls["n"] += 1
        return (m(x) ** 2).mean()

    import math
    out = _optimise(opt, train_loss, lambda: 0.5, [m], TrainConfig(n_epochs=7))
    assert math.isnan(out)
    assert calls["n"] == 7


# --------------------------------------------------------------------------- #
# Integration — full run path on the small fixture (3 epochs)                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("embed", ["table", "rotor"])
def test_run_smoke_returns_finite_auroc_and_provenance(embed):
    r = run("bitcoin_otc", seed=0, embed=embed, shuffle=False,
            head="bilinear", dedup=True, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    for key in ("lr", "weight_decay", "grad_clip", "class_weight",
                "early_stop", "embed", "n_triads", "n_params"):
        assert key in r
    assert r["embed"] == embed
    assert r["val_auroc"] is None      # early_stop off -> not populated


def test_run_tuned_recipe_path_runs_and_populates_val():
    """class-weight + clip + wd + early-stop all on the same code path."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False,
            head="bilinear", dedup=True,
            cfg=TrainConfig(n_epochs=4, weight_decay=1e-4, grad_clip=1.0,
                            class_weight=True, early_stop=True, eval_every=1))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["val_auroc"] is not None
    assert r["weight_decay"] == 1e-4 and r["grad_clip"] == 1.0
    assert r["class_weight"] is True


@pytest.mark.parametrize("channel", ["both", "quaternion", "clifford"])
def test_run_geom_attn_head_smoke(channel):
    """The geometric attention head runs end-to-end on the rotor line."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False,
            head="geom_attn", dedup=True, geom_channel=channel,
            cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["geom_channel"] == channel
    assert r["head"] == "geom_attn"
