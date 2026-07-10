"""Tests for the synthetic LiNGAM/HSiKAN operator harness (the 12 guards the spec requires)."""
from __future__ import annotations

import numpy as np

from hymeko_rl.eval.causal.hsikan_mechanism import signed_adjacency_split
from hymeko_rl.eval.causal.lingam_operator_harness import (
    HarnessConfig,
    _fit_linear,
    _sample_condition,
    _standardise_splits,
    generate_signed_dag,
    prepare,
    run_condition,
    run_harness,
    sample_sem,
)
from hymeko_rl.experiments.incidence_scramble import scramble_signed_operator

_SMOKE = dict(seeds=1, n=10, samples=200, epochs=40)


# 1 — B acyclic + zero diagonal
def test_dag_is_acyclic_zero_diagonal() -> None:
    b = generate_signed_dag(16, seed=0)
    assert np.all(np.diag(b) == 0.0)
    assert np.all(np.triu(b) == 0.0)                # strictly lower triangular ⇒ acyclic in node order


# 2 — A⁺ − A⁻ reconstructs B
def test_split_reconstructs_generated_B() -> None:
    b = generate_signed_dag(16, seed=1)
    a_pos, a_neg = signed_adjacency_split(b)
    assert np.allclose(a_pos - a_neg, b)


# 3 — SEM respects topological order (a late edge cannot change earlier nodes)
def test_sem_respects_topological_order() -> None:
    b1 = np.zeros((6, 6))
    b1[3, 1] = 0.9
    b2 = b1.copy()
    b2[5, 4] = 0.8                                   # a strictly-later edge (effect 5 from cause 4)
    x1 = sample_sem(b1, 200, kind="linear", seed=0)
    x2 = sample_sem(b2, 200, kind="linear", seed=0)
    assert np.allclose(x1[:, :5], x2[:, :5])         # nodes 0..4 unaffected by the later edge
    assert not np.allclose(x1[:, 5], x2[:, 5])       # the sink (node 5) does change


# 4 — target masking prevents identity leakage
def test_masking_zeroes_the_sink() -> None:
    x = np.random.default_rng(0).standard_normal((20, 6))
    p = prepare(x, x[:, 5].copy(), sink=5)
    assert np.all(p.x_masked[:, 5] == 0.0)           # sink never in the model input
    assert p.x_linear.shape[1] == 5                  # sink column dropped for the linear predictor
    assert not np.all(p.x_leaky[:, 5] == 0.0)        # the leaky probe keeps it (to prove masking matters)


def test_leakage_guard_fires_on_sink_target() -> None:
    # an MLP that SEES the unmasked sink must cheat (~0 MSE) while the masked model does not.
    r = run_condition("nonlinear", HarnessConfig(**_SMOKE))
    assert r["leakage"]["guard_ok"] is True
    assert r["leakage"]["leaky_test"] < 0.05 <= r["leakage"]["masked_test"]


# 5 — operator scramble preserves per-sign counts + weight multiset (+ zero diagonal)
def test_operator_scramble_preserves_counts_and_weights() -> None:
    a_pos, a_neg = signed_adjacency_split(generate_signed_dag(16, seed=2))
    sp, sn = scramble_signed_operator(a_pos, a_neg, seed=0)
    for orig, scr in ((a_pos, sp), (a_neg, sn)):
        assert int((scr != 0).sum()) == int((orig != 0).sum())               # nonzero count per sign
        assert np.allclose(np.sort(scr[scr != 0]), np.sort(orig[orig != 0]))  # weight multiset per sign
    assert np.all(np.diag(sp) == 0.0) and np.all(np.diag(sn) == 0.0)


# 6 — scramble changes incidence
def test_operator_scramble_changes_incidence() -> None:
    a_pos, a_neg = signed_adjacency_split(generate_signed_dag(20, density=0.5, seed=3))
    changed = False
    for seed in range(4):
        sp, sn = scramble_signed_operator(a_pos, a_neg, seed=seed)
        if not (np.array_equal(sp != 0, a_pos != 0) and np.array_equal(sn != 0, a_neg != 0)):
            changed = True
    assert changed


# 7 / 8 — determinism
def test_operator_scramble_deterministic_and_seed_sensitive() -> None:
    a_pos, a_neg = signed_adjacency_split(generate_signed_dag(20, density=0.5, seed=4))
    a1 = scramble_signed_operator(a_pos, a_neg, seed=5)
    a2 = scramble_signed_operator(a_pos, a_neg, seed=5)
    a3 = scramble_signed_operator(a_pos, a_neg, seed=6)
    assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])       # same seed → same
    assert not (np.array_equal(a1[0], a3[0]) and np.array_equal(a1[1], a3[1])) # different seed → different


# 9 — linear SEM sanity: the linear predictor is strong on linear data
def test_linear_predictor_strong_on_linear_sem() -> None:
    b = generate_signed_dag(12, density=0.5, seed=0)
    sink = 11
    xs, ys = zip(*(_sample_condition("linear", b, sink, 600, sample_seed=k) for k in range(3)))
    tr, _va, te = _standardise_splits(list(xs), list(ys), sink)
    res = _fit_linear(tr, te)
    assert res["r2_test"] > 0.5                       # linear structure is linearly predictable


# 10 — flat control: shuffled target ⇒ no model predicts it; operator irrelevant (collapse ≈ 0)
def test_flat_control_has_no_artificial_signal() -> None:
    r = run_condition("flat", HarnessConfig(**_SMOKE))
    hk = r["models"]["hsikan_correct"]["test_median"]
    assert hk > 0.5                                   # shuffled target is not predictable
    assert abs(r["scramble_collapse_frac"]) < 0.2     # scrambling the operator does ~nothing


# 11 — result table completeness
def test_harness_report_complete() -> None:
    r = run_harness(HarnessConfig(**_SMOKE))
    assert set(r["conditions"]) == {"linear", "nonlinear", "flat"}
    for cond in r["conditions"].values():
        assert set(cond["models"]) == {"linear", "mlp", "deepsets", "hsikan_correct", "hsikan_scrambled"}
    assert r["verdict"]["label"] in ("SUPPORTED", "WEAKENED", "FALSIFIED")


# 12 (no existing CIP causal tests break) is verified by running the CIP suite in CI, not duplicated here.
