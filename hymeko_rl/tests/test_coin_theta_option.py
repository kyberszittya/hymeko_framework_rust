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


# ───────────────────────────── Stage 3: BC proposal actor ─────────────────────────────
def _tiny_dataset(n_train=12, n_val=4, seed=0):
    from hymeko_rl.coin_delivery.theta_option.dataset import DatasetRow, ThetaDataset
    box = ThetaBox()
    lo, hi = theta_bounds()
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_train + n_val):
        theta = lo + rng.random(DIM) * (hi - lo)
        rows.append(DatasetRow(tag="s1" if i % 2 else "s3", split="train" if i < n_train else "val",
                               kind="basin_delivering", features=rng.standard_normal(42).astype(np.float32),
                               history=rng.standard_normal((8, 6)).astype(np.float32),
                               theta=box.clip(theta), theta_norm=box.norm(theta), eval_only=False, k6_delivered=True))
    return ThetaDataset(rows=rows, contract={})


def test_bc_obs_dim_per_variant():
    from hymeko_rl.coin_delivery.theta_option.proposal import obs_dim
    assert obs_dim("B0") == 42 and obs_dim("B1") == 42 + 48 and obs_dim("B2") == 42 + 8


def test_bc_fit_produces_legal_theta_and_full_validity():
    from hymeko_rl.coin_delivery.theta_option.proposal import fit_bc, offline_metrics
    ds = _tiny_dataset()
    lo, hi = theta_bounds()
    for v in ("B0", "B1", "B2"):
        prop, fz, _tl = fit_bc(ds, v, epochs=60, lr=1e-3, seed=0)
        obs = fz.obs(ds.rows[0].features, ds.rows[0].history)
        th = prop.center(obs)
        assert np.all(th >= lo - 1e-4) and np.all(th <= hi + 1e-4)      # legal option
        m = offline_metrics(prop, fz, ds, "val")
        assert m["bounded_validity"] == 1.0                            # Tanh output always in-box


def test_bc_fit_deterministic_same_seed():
    from hymeko_rl.coin_delivery.theta_option.proposal import fit_bc
    ds = _tiny_dataset()
    p1, f1, _ = fit_bc(ds, "B0", epochs=60, lr=1e-3, seed=1)
    p2, f2, _ = fit_bc(ds, "B0", epochs=60, lr=1e-3, seed=1)
    obs = f1.obs(ds.rows[0].features, ds.rows[0].history)
    assert np.allclose(p1.center(obs), p2.center(obs), atol=1e-6)


def test_bc_save_load_roundtrip(tmp_path):
    from hymeko_rl.coin_delivery.theta_option.proposal import fit_bc, load_proposal, save_proposal
    ds = _tiny_dataset()
    for v in ("B0", "B2"):                                              # cover the flat + LSTM-featurizer paths
        prop, fz, _ = fit_bc(ds, v, epochs=60, lr=1e-3, seed=0)
        p = str(tmp_path / f"bc_{v}.pt")
        save_proposal(prop, fz, p)
        prop2, fz2 = load_proposal(p)
        obs1 = fz.obs(ds.rows[0].features, ds.rows[0].history)
        obs2 = fz2.obs(ds.rows[0].features, ds.rows[0].history)
        assert np.allclose(obs1, obs2, atol=1e-6)
        assert np.allclose(prop.center(obs1), prop2.center(obs2), atol=1e-6)


# ───────────────────────────── Stage 4: update-0 no-regression ─────────────────────────────
_UPDATE0_PATH = "reports/2026-07-27-coin-teacher-to-rl/update_zero.json"


@pytest.mark.skipif(not os.path.exists(_UPDATE0_PATH), reason="update_zero.json not built")
def test_update_zero_artifact_gate_and_controls():
    d = json.load(open(_UPDATE0_PATH))
    g = d["gate"]
    # the ORACLE (teacher θ + same search) must validate that search+physics deliver 4/4 from a correct θ_0
    assert g["oracle_total_k6"] == g["n_states"] and g["oracle_validates_search_and_physics"] is True
    # the actor must be load-bearing (informed strictly beats the uninformed box-centre control)
    assert g["informed_total_k6"] > g["uninformed_total_k6"] and g["actor_load_bearing"] is True
    assert g["no_hidden_teacher_fallback"] is True
    # verdict/authorisation must be internally consistent
    assert d["authorises_rl"] == bool(d["verdict"] == "UPDATE_ZERO_MOBILE_TEACHER_NO_REGRESSION_PASS" and g["passed"])
    if not g["passed"]:
        assert d["authorises_rl"] is False


@pytest.mark.slow
def test_oracle_teacher_theta_delivers_under_fixed_search():
    """Live physics: the centre-inclusive fixed search seeded with the recorded teacher θ delivers frozen K6 on a held-out
    cradle (s4) — isolating any held-out actor miss to θ_0 quality (coverage), not the search/physics. Also regresses the
    centre-inclusion fix: an std=0.15 jitter escapes s4's narrow basin, so a centre-excluding search would fail here."""
    bank = json.load(open(_BANK_PATH))
    if bank.get("smoke"):
        pytest.skip("bank is a smoke run")
    from hymeko_rl.coin_delivery.theta_option.search import fixed_search_select
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import CERTIFIED_SEEDS, acquire_snapshot, load_harness
    s4 = [e for e in bank["states"] if e["tag"] == "s4"][0]
    snap, _ = acquire_snapshot(load_harness(), CERTIFIED_SEEDS[2])
    teacher = np.asarray(s4["canonical_theta_vec"], np.float64)
    prov = fixed_search_select(snap, teacher, np.random.default_rng(0), budget=8)
    assert prov.outcome["delivery_success"] is True                    # teacher θ + centre-inclusive search delivers
    assert np.allclose(prov.center, teacher, atol=1e-4)                # centre (Bellman action) preserved


# ───────────────────────────── centre-inclusive fixed search (correctness fix) ─────────────────────────────
class _CountingCentreOptimalScorer:
    """Physics-free scorer where the CENTRE is strictly optimal and every other candidate is worse; counts evaluations.
    Conforms to option_rl.CandidateScorer, injected into fixed_search_select."""

    def __init__(self, center):
        self.center = np.asarray(center, np.float64)
        self.calls = 0

    def score(self, candidate, rng):
        self.calls += 1
        d = float(np.linalg.norm(np.asarray(candidate, np.float64) - self.center))
        return -d, {"k6_delivered": bool(d < 1e-9), "dtz_end": d, "delivery_success": bool(d < 1e-9)}


def test_fixed_search_is_centre_inclusive_and_budget_exact():
    """Discriminating test for the centre-inclusion correctness fix: (1) the centre is always evaluated; (2) when the
    centre is optimal and every jittered neighbour is worse, the search RETURNS the centre; (3) the total number of
    candidate evaluations equals the declared budget (centre + budget-1 jitters). A centre-EXCLUDING search would return
    a worse neighbour and evaluate the wrong count."""
    from hymeko_rl.coin_delivery.theta_option.search import fixed_search_select, search_semantics
    box = ThetaBox()
    center = box.clip(theta_bounds()[0] + 0.1)
    for budget in (0, 1, 4, 8):
        sc = _CountingCentreOptimalScorer(center)
        prov = fixed_search_select(None, center, np.random.default_rng(0), budget=budget, scorer=sc)
        assert sc.calls == max(1, budget)                              # total evaluations == declared budget
        assert np.allclose(prov.center, center, atol=1e-6)             # Bellman action preserved
        assert np.allclose(prov.selected, prov.center, atol=1e-6)      # centre optimal ⇒ search returns the centre
        assert prov.outcome["delivery_success"] is True
    ss = search_semantics(8)
    assert ss["n_total_candidates"] == 8 and ss["n_jittered_candidates"] == 7
    assert ss["centre_always_evaluated"] and ss["budget_counts_total_candidates"]


# ───────────────────────────── dev-cradle expansion: dedup ─────────────────────────────
class _StubCradle:
    def __init__(self, seed, hsh, fp):
        self.seed = int(seed)
        self.hash = hsh
        self.fingerprint = np.asarray(fp, np.float64)


def test_dedup_exact_near_and_holdout_exclusion():
    """The certified→unique funnel: exact dedup by state hash, near dedup by fingerprint L2, and exclusion of the frozen
    held-out cradles + their near-duplicates. Raw certification count must NOT be mistaken for unique-cradle count."""
    from hymeko_rl.coin_delivery.theta_option.cradle_expansion import dedup_and_split
    pool = [
        _StubCradle(15000, "HELD", [0.0, 10.0, 0.0]),     # held-out (s4)
        _StubCradle(100, "A", [0.0, 0.0, 0.0]),           # unique dev rep
        _StubCradle(200, "A", [0.0, 0.0, 0.0]),           # EXACT dup of 100 (same hash)
        _StubCradle(300, "B", [10.0, 0.0, 0.0]),          # unique dev rep (far)
        _StubCradle(400, "C", [10.1, 0.0, 0.0]),          # NEAR dup of 300 (dist 0.1 < tol)
        _StubCradle(500, "D", [0.0, 10.1, 0.0]),          # NEAR dup of the held-out 15000
    ]
    d = dedup_and_split(pool, near_tol=0.5, heldout_seeds=(15000,))
    assert d["dev_eligible_seeds"] == [100, 300]
    by_seed = {r["seed"]: r for r in d["rows"]}
    assert by_seed[200]["duplicate_of"] == 100 and by_seed[200]["dev_eligible"] is False
    assert by_seed[400]["near_dup_of"] == 300 and by_seed[400]["dev_eligible"] is False
    assert by_seed[15000]["held_out"] is True and by_seed[15000]["dev_eligible"] is False
    assert by_seed[500]["dev_eligible"] is False           # near-dup of a held-out cradle
    assert d["n_certified"] == 6 and d["n_hash_unique"] == 5 and d["n_dev_eligible"] == 2
