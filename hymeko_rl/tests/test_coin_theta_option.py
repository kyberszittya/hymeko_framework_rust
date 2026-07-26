"""Tests for the coin 6-D torque-θ option adapter (teacher-to-RL campaign).

Stage 0 layer here: frozen θ semantics, the θ normaliser (round-trip + always-legal), and the ANTI-ALIASING contract —
the Bellman action is the proposal centre θ_0, never the search-selected θ_exec. Later stages (dataset split isolation,
provenance under real physics) add to this file as they are built.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option.semantics import (
    DELIVERY_CFG, DIM, ThetaBox, ThetaProvenance, option_semantics, theta_bounds)
from hymeko_rl.coin_delivery.theta_option.search import ThetaCandidateGenerator
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL
from hymeko_rl.option_rl.core import OptionReplayBuffer, OptionTransition
from hymeko_rl.option_rl.proposal import FixedBudgetSearch

_BANK_PATH = "reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"


# ───────────────────────────── θ box normaliser ─────────────────────────────
def test_theta_box_roundtrip_identity_on_legal_theta():
    box = ThetaBox()
    lo, hi = theta_bounds()
    rng = np.random.default_rng(0)
    thetas = lo + rng.random((200, DIM)) * (hi - lo)          # legal θ inside the box
    for t in thetas:
        z = box.norm(t)
        assert np.all(z >= -1.0 - 1e-6) and np.all(z <= 1.0 + 1e-6)
        back = box.denorm(z)
        assert np.allclose(back, t, atol=1e-5), (t, back)


def test_theta_box_denorm_is_always_legal_even_off_box():
    box = ThetaBox()
    lo, hi = theta_bounds()
    rng = np.random.default_rng(1)
    z = rng.uniform(-4.0, 4.0, (500, DIM))                    # z far outside [-1,1]
    theta = np.asarray([box.denorm(zi) for zi in z])
    assert np.all(theta >= lo - 1e-6) and np.all(theta <= hi + 1e-6)


def test_theta_box_clip_pulls_out_of_box_into_box():
    box = ThetaBox()
    lo, hi = theta_bounds()
    over = hi + 5.0
    under = lo - 5.0
    assert np.allclose(box.clip(over), hi, atol=1e-6)
    assert np.allclose(box.clip(under), lo, atol=1e-6)


# ───────────────────────────── frozen semantics ─────────────────────────────
def test_option_semantics_matches_frozen_delivery_config():
    sem = option_semantics()
    assert sem["dim"] == DIM == 6
    assert len(sem["components"]) == 6
    lo, hi = theta_bounds()
    for i, c in enumerate(sem["components"]):
        assert c["lo"] == float(lo[i]) and c["hi"] == float(hi[i])
    assert tuple(DELIVERY_CFG.lo) == (0.0, 0.0, -0.10, 1.0, 4.0, 0.0)
    assert tuple(DELIVERY_CFG.hi) == (0.25, 0.30, 0.10, 28.0, 48.0, 4.0)
    assert DELIVERY_CFG.horizon == 60 and DELIVERY_CFG.deliver is True
    k6 = sem["termination_and_k6"]
    assert k6["CENTER_TOL_m"] == CENTER_TOL == 0.02
    assert k6["SETTLE_VEL_mps"] == SETTLE_VEL == 0.06
    assert k6["HELD_DWELL_steps"] == HELD_DWELL == 6
    assert sem["frozen_split"]["development"] == ["s1", "s3"]
    assert sem["frozen_split"]["held_out"] == ["s4", "s7"]


# ───────────────────────────── ANTI-ALIASING (Bellman action = θ_0, not θ_exec) ─────────────────────────────
class _StubScorer:
    """Physics-free scorer for the structural aliasing test: prefers candidates NEAR a hidden target θ (so the search
    provably MOVES away from an off-target centre). Conforms to option_rl.CandidateScorer."""

    def __init__(self, target):
        self.target = np.asarray(target, np.float64)

    def score(self, candidate, rng):
        d = float(np.linalg.norm(np.asarray(candidate, np.float64) - self.target))
        return -d, {"k6_delivered": bool(d < 1e-3), "dtz_end": d}


def test_fixed_budget_search_preserves_center_and_moves_selected():
    """The search must (a) keep the input centre as `.center` unchanged and (b) be able to select a DIFFERENT θ_exec —
    if selected silently overwrote center, this test would fail."""
    box = ThetaBox()
    lo, hi = theta_bounds()
    center = box.clip(lo + 0.2 * (hi - lo))                   # an off-target centre
    target = box.clip(lo + 0.8 * (hi - lo))                   # where the search should pull toward
    gen = ThetaCandidateGenerator(box=box)
    sel = FixedBudgetSearch(generator=gen, scorer=_StubScorer(target), budget=32).select(center, np.random.default_rng(3))
    assert np.allclose(sel.center, center, atol=1e-6)         # centre (Bellman action) preserved
    assert not np.allclose(sel.selected, sel.center, atol=1e-4)   # θ_exec genuinely differs
    # θ_exec is closer to target than θ_0 was ⇒ the search is load-bearing
    assert np.linalg.norm(sel.selected - target) < np.linalg.norm(center - target)


def test_budget_zero_executes_center_directly():
    box = ThetaBox()
    center = box.clip(theta_bounds()[0] + 0.3)
    sel = FixedBudgetSearch(generator=ThetaCandidateGenerator(box=box), scorer=_StubScorer(center),
                            budget=0).select(center, np.random.default_rng(0))
    assert np.allclose(sel.selected, sel.center, atol=1e-6) and sel.budget == 0


def test_theta_provenance_keeps_center_as_bellman_action():
    box = ThetaBox()
    center = box.clip(theta_bounds()[0] + 0.1)
    selected = box.clip(theta_bounds()[0] + 0.25)
    prov = ThetaProvenance(center=center, selected=selected, score=1.0, budget=8,
                           outcome={"k6_delivered": True, "dtz_end": 0.01})
    d = prov.as_dict()
    assert d["theta_center"] == [round(float(x), 5) for x in center]
    assert d["theta_selected"] == [round(float(x), 5) for x in selected]
    assert prov.displacement(box) > 0.0                      # they differ


def test_replay_buffer_bellman_action_is_center_not_selected():
    """End-to-end anti-aliasing through the engine: the OptionTransition.action fed to the replay buffer is θ_0; θ_exec
    lives only in provenance. Sampling the batch returns θ_0 — never θ_exec. A regression that logged θ_exec as the
    action would flip this assertion."""
    box = ThetaBox()
    center = box.norm(box.clip(theta_bounds()[0] + 0.15))     # normalised θ_0 (the network action space)
    selected = box.norm(box.clip(theta_bounds()[0] + 0.30))   # a DIFFERENT θ_exec
    buf = OptionReplayBuffer()
    for _ in range(8):
        buf.add(OptionTransition(s=np.zeros(4, np.float32), action=center, reward=1.0, tau=60.0,
                                 s_next=np.zeros(4, np.float32), terminal=1.0, end="handoff",
                                 provenance={"theta_center": center, "theta_selected": selected}))
    bs, ba, br, bt, bs2, bd = buf.sample(4, np.random.default_rng(0))
    for a in ba.numpy():
        assert np.allclose(a, center, atol=1e-6)              # Bellman action == θ_0
        assert not np.allclose(a, selected, atol=1e-4)        # and NOT θ_exec
    # provenance retains θ_exec separately
    assert np.allclose(buf.provenance[0]["theta_selected"], selected, atol=1e-6)


# ───────────────────────────── Stage 1: teacher bank ─────────────────────────────
def _synth_metrics(*, k6=True, dwell=8, dtz_end=0.01, qdot=1.5, coin_speed=0.5):
    return {"k6_delivered": bool(k6), "k6_max_dwell": int(dwell), "touched": True, "dtz_start": 0.10,
            "dtz_end": float(dtz_end), "gap_closed": 0.9, "forward": 0.05, "terminal_coin_speed": 0.0,
            "peak_qdot": float(qdot), "peak_coin_speed": float(coin_speed), "contact_lost_steps": 0,
            "lost_before_release": 0}


def test_phase_of_boundaries():
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import _phase_of
    theta = [0.1, 0.2, 0.0, 8.0, 20.0, 1.0]                   # ramp=8, release=20
    assert _phase_of(1, theta) == "PUSH" and _phase_of(8, theta) == "PUSH"
    assert _phase_of(9, theta) == "BRAKE" and _phase_of(20, theta) == "BRAKE"
    assert _phase_of(21, theta) == "RELEASE" and _phase_of(60, theta) == "RELEASE"


def test_outcome_summary_reports_frozen_k6_verdict():
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import _outcome_summary
    s = _outcome_summary(_synth_metrics(k6=True, dwell=10), DELIVERY_CFG)
    assert s["k6_delivered"] is True and s["k6_max_dwell"] == 10 and s["delivery_success"] is True
    # a motion-contract breach must NOT be delivery_success even with K6
    s2 = _outcome_summary(_synth_metrics(k6=True, qdot=99.0), DELIVERY_CFG)
    assert s2["delivery_success"] is False


@pytest.mark.skipif(not os.path.exists(_BANK_PATH), reason="teacher_bank.json not built")
def test_teacher_bank_artifact_gate_split_and_holdout_isolation():
    bank = json.load(open(_BANK_PATH))
    if bank.get("smoke"):
        pytest.skip("bank artifact is a smoke run (partial states)")
    assert bank["gate"]["passed"] is True
    assert bank["gate"]["k6_by_frozen_monitor_only"] and bank["gate"]["no_pin_teleport_or_coin_edit"]
    dev = [e for e in bank["states"] if e["split"] == "development"]
    held = [e for e in bank["states"] if e["split"] == "held_out"]
    assert [e["tag"] for e in dev] == ["s1", "s3"] and [e["tag"] for e in held] == ["s4", "s7"]
    # every canonical θ delivers frozen K6 and replays; held-out is eval-only (NO basin augmentation)
    for e in bank["states"]:
        assert e["k6_delivered"] is True and e["replay_ok"] is True and e["deterministic"] is True
        assert len(e["canonical_theta_vec"]) == DIM
    for e in held:
        assert "basin_candidates" not in e, f"held-out {e['tag']} must not be augmented"
    for e in dev:
        assert "basin_candidates" in e and e["n_basin_delivering"] >= 1


@pytest.mark.slow
def test_reproduce_held_out_state_delivers_without_basin():
    """Live physics: reproduce a held-out cradle (s7) end-to-end — it must deliver frozen K6, replay deterministically,
    and receive NO basin augmentation (held-out is eval-only)."""
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import CERTIFIED_SEEDS, load_harness, reproduce_state
    e = reproduce_state(load_harness(), 3, CERTIFIED_SEEDS[3], augment=False)  # idx 3 = s7 held-out
    assert e["tag"] == "s7" and e["split"] == "held_out"
    assert e["k6_delivered"] is True and e["replay_ok"] is True and e["deterministic"] is True
    assert "basin_candidates" not in e


# ───────────────────────────── Stage 2: structured causal dataset ─────────────────────────────
_DATASET_PATH = "reports/2026-07-27-coin-teacher-to-rl/dataset_contract.json"


def _synth_feat():
    from hymeko_rl.coin_delivery.theta_option.dataset import FEATURE_ORDER
    sizes = {"dtz": 1, "e_par": 2, "coin_xy": 2, "coin_vel": 2, "straddle": 1, "fn": 2, "normal": 4, "xc_rel": 4,
             "q": 4, "qdot": 4, "prev_tau": 4, "slew_head": 8, "saturated": 4}
    return {g: np.arange(sizes[g], dtype=np.float64) + 1.0 for g in FEATURE_ORDER}


def test_feature_names_match_feature_dim():
    from hymeko_rl.coin_delivery.theta_option.dataset import feature_names, flatten_features
    names = feature_names()
    vec = flatten_features(_synth_feat())
    assert len(names) == vec.shape[0] == 42


def test_flatten_features_deterministic_and_scaled():
    from hymeko_rl.coin_delivery.theta_option.dataset import NORM_SCALES, flatten_features
    feat = _synth_feat()
    a = flatten_features(feat, normalise=True)
    b = flatten_features(feat, normalise=True)
    assert np.array_equal(a, b)                                  # deterministic
    raw = flatten_features(feat, normalise=False)
    assert not np.allclose(a, raw)                              # normalisation changed something
    # dtz (first entry) divided by its physical scale; slew_head is NOT re-scaled (already normalised)
    assert np.isclose(a[0], feat["dtz"][0] / NORM_SCALES["dtz"])


def test_history_streaming_equals_batch_on_probe_shape():
    """The temporal encoder over the (K,6) causal history probe must give identical embeddings whether fed step-by-step
    (deploy/streaming) or as a batch sequence — the closing contract for B2."""
    import torch
    from hymeko_rl.option_rl.temporal import LSTMTemporalEncoder
    torch.manual_seed(0)
    enc = LSTMTemporalEncoder(in_dim=6, hidden=16, out_dim=8)
    X = np.random.default_rng(0).standard_normal((8, 6)).astype(np.float32)   # (HISTORY_K, |HIST_FEATURES|)
    embs, _ = enc.forward(torch.as_tensor(X)[None])            # batch: (1, K, 8)
    h, outs = None, []
    for t in range(X.shape[0]):
        e, h = enc.update(torch.as_tensor(X[t]), h)
        outs.append(e[0])
    stream = torch.stack(outs)
    assert torch.allclose(embs[0], stream, atol=1e-5)


def test_dataset_row_split_isolation_invariant():
    from hymeko_rl.coin_delivery.theta_option.dataset import DatasetRow, ThetaDataset, contract_summary
    good = [DatasetRow("s1", "train", "canonical", np.zeros(42, np.float32), np.zeros((8, 6), np.float32),
                       np.zeros(6), np.zeros(6), False, True),
            DatasetRow("s4", "eval", "canonical", np.zeros(42, np.float32), np.zeros((8, 6), np.float32),
                       np.zeros(6), np.zeros(6), True, True)]
    ds = ThetaDataset(rows=good, contract={"split_counts": {"train": 1, "val": 0, "eval": 1}, "n_by_tag_split": {}})
    assert contract_summary(ds)["split_isolation_ok"] is True
    bad = good + [DatasetRow("s7", "train", "canonical", np.zeros(42, np.float32), np.zeros((8, 6), np.float32),
                             np.zeros(6), np.zeros(6), True, True)]           # held-out (eval_only) leaked into train
    ds_bad = ThetaDataset(rows=bad, contract={"split_counts": {}, "n_by_tag_split": {}})
    assert contract_summary(ds_bad)["split_isolation_ok"] is False


@pytest.mark.skipif(not os.path.exists(_DATASET_PATH), reason="dataset_contract.json not built")
def test_dataset_contract_artifact_invariants():
    c = json.load(open(_DATASET_PATH))
    assert c["split_isolation_ok"] is True and c["all_hashes_match"] is True
    assert c["feature_dim"] == 42 and c["history"]["k"] == 8
    # held-out cradles appear ONLY in eval; θ labels are legal
    lo, hi = theta_bounds()
    for r in c["rows"]:
        th = np.asarray(r["theta"], np.float64)
        assert np.all(th >= lo - 1e-5) and np.all(th <= hi + 1e-5)
        if r["tag"] in ("s4", "s7"):
            assert r["split"] == "eval" and r["eval_only"] is True
    assert any(r["split"] == "train" and r["tag"] in ("s1", "s3") for r in c["rows"])


@pytest.mark.slow
def test_structured_features_and_history_deterministic_live():
    """Live physics: the causal feature vector and history probe are deterministic (same snapshot → identical), and the
    history has the frozen (K,6) shape."""
    from hymeko_rl.coin_delivery.theta_option.dataset import HISTORY_K, causal_history, flatten_features, structured_features
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import CERTIFIED_SEEDS, load_harness
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot
    snap, _ = acquire_snapshot(load_harness(), CERTIFIED_SEEDS[0])           # s1
    f1 = flatten_features(structured_features(snap))
    f2 = flatten_features(structured_features(snap))
    assert np.array_equal(f1, f2) and f1.shape[0] == 42
    h1, h2 = causal_history(snap), causal_history(snap)
    assert h1.shape == (HISTORY_K, 6) and np.array_equal(h1, h2)
