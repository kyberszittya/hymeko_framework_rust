"""Tests for the leakage-free HSiKAN rotor driver and its tuning knobs.

Covers the new `TrainConfig` levers (weight_decay, grad_clip, class-weighted
BCE, early-stop) added on 2026-06-17 to close the SiGAT gap. The load-bearing
leakage gate (shuffle -> chance) is exercised at experiment scale, not here:
toy epochs do not converge, so a unit-scale ~0.5 assertion would be flaky. These
tests pin the *pure* helpers deterministically and smoke the full `run` path on
the small bitcoin_otc fixture at 3 epochs.

Run: pytest -p no:randomly hymeko_neuro/tests/test_hsikan_rotor.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_neuro.data import datasets
from hymeko_neuro.experiments.runs.run_hsikan_rotor import (
    TrainConfig,
    _drop_train_pairs,
    _optimise,
    _pair,
    _pos_weight,
    run,
)

CPU = torch.device("cpu")


def test_dedup_helpers_are_the_datasets_single_source():
    """Guard against §6.5 #1 re-duplication: the private names are aliases of the
    canonical datasets-layer functions, not a third local copy."""
    assert _drop_train_pairs is datasets.drop_train_pairs
    assert _pair is datasets.undirected_pair


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


def test_run_geom_attn_fires_on_trained_with_aligned_state():
    """on_trained fires once for geom_attn, handing self-consistent handles that
    summarise_gate can consume (drives the gate-inspection diagnostic)."""
    from hymeko_neuro.hyperedge.geometric_triad_attention import summarise_gate

    seen: list = []
    run("bitcoin_otc", seed=0, embed="rotor", shuffle=False,
        head="geom_attn", dedup=True, geom_channel="both",
        cfg=TrainConfig(n_epochs=3), on_trained=lambda st: seen.append(st))
    assert len(seen) == 1
    st = seen[0]
    assert st.pool.channel == "both"
    assert st.h_v.shape[1] == st.temb.shape[1]          # shared hidden width
    assert st.inc_vertex.shape == st.inc_triad.shape
    s = summarise_gate(st.pool, st.h_v, st.temb, st.inc_vertex, st.inc_triad,
                       st.inc_balance)
    assert s["n_incidences"] == st.inc_vertex.shape[0]
    assert 0.0 < s["gate_sigma"] < 1.0


def test_run_bilinear_head_ignores_on_trained():
    """on_trained is geom_attn-only: the bilinear path must not fire it."""
    seen: list = []
    run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="bilinear",
        dedup=True, cfg=TrainConfig(n_epochs=3), on_trained=lambda st: seen.append(st))
    assert seen == []


def test_run_woken_geom_attn_smoke_and_live_weights():
    """The woken readout (init x1.0 + learn-scale + sign-aware) runs end-to-end,
    records its provenance, and — unlike the dead legacy score — has live weights."""
    from hymeko_neuro.hyperedge.geometric_triad_attention import summarise_gate

    seen: list = []
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="geom_attn",
            dedup=True, geom_channel="both", geom_init_scale=1.0,
            geom_learn_scale=True, geom_sign_aware=True,
            cfg=TrainConfig(n_epochs=3), on_trained=lambda st: seen.append(st))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["geom_init_scale"] == 1.0 and r["geom_learn_scale"] is True
    assert r["geom_sign_aware"] is True
    st = seen[0]
    assert st.inc_balance is not None
    s = summarise_gate(st.pool, st.h_v, st.temb, st.inc_vertex, st.inc_triad,
                       st.inc_balance)
    assert s["w_frac_dead"] < 1.0          # the score is no longer all-dead


def test_run_rotor_rel_head_smoke_and_provenance():
    """The rotor-relative head (bilinear + projected relative rotor) runs
    end-to-end and records n_refs/head provenance."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="rotor_rel",
            dedup=True, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["head"] == "rotor_rel" and r["n_refs"] == 1


def test_run_n_refs_two_changes_embedding_path():
    """n_refs=2 (less-lossy embedding, route B) runs with the bilinear head."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="bilinear",
            dedup=True, n_refs=2, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["n_refs"] == 2


def test_rotor_rel_requires_rotor_embed():
    """rotor_rel needs the rotor q; the table embedding has none → raises."""
    with pytest.raises(ValueError):
        run("bitcoin_otc", seed=0, embed="table", shuffle=False, head="rotor_rel",
            dedup=True, cfg=TrainConfig(n_epochs=2))


def test_run_cycle_k4_smoke_and_provenance():
    """Pooling closed 4-cycles (reused construct_k_arrays) runs end-to-end on the
    rotor line and records the cycle provenance."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="bilinear",
            dedup=True, cycle_k=4, max_cycles=500, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["cycle_k"] == 4 and r["max_cycles"] == 500
    assert r["n_triads"] > 0          # 4-cycles were actually enumerated


def test_cycle_k_requires_endpoint_head():
    """The incidence head needs the SignedTriad list; --cycle-k → endpoint only."""
    with pytest.raises(ValueError):
        run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="incidence",
            dedup=True, cycle_k=4, max_cycles=200, cfg=TrainConfig(n_epochs=2))


def test_run_rotor_propagation_smoke_and_provenance():
    """Signed slerp rotor propagation (neighbourhood-aware input) runs end-to-end
    on the rotor line and records the round count."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="bilinear",
            dedup=True, rotor_prop_rounds=2, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["rotor_prop_rounds"] == 2


def test_run_enriched_feature_spec_smoke_and_provenance():
    """The rotor input accepts a leakage-free feature spec (degree+walk+cycle);
    runs end-to-end and records the spec in provenance."""
    spec = ("degree", "walk_k3", "cycle_k3")
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="bilinear",
            dedup=True, feature_spec=spec, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["feature_spec"] == list(spec)


def test_run_rotor_propagation_learnable_smoke_and_provenance():
    """Learnable per-block self-retention runs end-to-end and is recorded; the
    retention parameter is a trainable param of the model."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head="bilinear",
            dedup=True, rotor_prop_rounds=2, rotor_prop_self_weight=4.0,
            rotor_prop_learnable=True, cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["rotor_prop_learnable"] is True


@pytest.mark.parametrize("head", ["complex", "rotate"])
def test_run_head_ablation_smokes_on_propagated_input(head):
    """The ablation heads (ComplEx replaces bilinear; RotatE-geodesic adds to it)
    run end-to-end on the propagated rotor input."""
    r = run("bitcoin_otc", seed=0, embed="rotor", shuffle=False, head=head,
            dedup=True, rotor_prop_rounds=2, rotor_prop_self_weight=4.0,
            cfg=TrainConfig(n_epochs=3))
    assert 0.0 <= r["test_auroc"] <= 1.0
    assert r["head"] == head


def test_rotate_head_requires_rotor_embed():
    """rotate reads the endpoint rotors; the table embedding has none → raises."""
    with pytest.raises(ValueError):
        run("bitcoin_otc", seed=0, embed="table", shuffle=False, head="rotate",
            dedup=True, cfg=TrainConfig(n_epochs=2))
