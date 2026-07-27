"""K0 — the KINETIC snapshot / replanning contract tests.

Pure-logic tests (schema, admissibility algebra, warm-start window) run without physics. Integration tests use the FAST
committed-bank s1 snapshot (``build_panel(_load_harness(), BANK)["s1"].snap`` — no certified-straddle re-acquisition) and
assert the load-bearing K0 invariants:
  * ``roll_until`` mirrors the frozen ``velocity_rollout`` step kernel bit-for-bit;
  * ``TransportSnapshot.branch()`` is deterministic;
  * ``kinetic_observe`` is batch == streaming and carries no future / teacher / K6 information;
  * ``freeze_kinetic_entry`` is deterministic (stable content hash) and lands on an admissible dual-contact state;
  * ``receding_horizon_relabel`` exports only a length-4, slew-normalised FIRST action, deterministically.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.hybrid_approach import ApproachParams, HybridApproachController
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout

_FORBIDDEN = ("future", "teacher", "k6", "landing", "theta", "release_step", "reward", "success")


# ── pure logic (no physics) ──────────────────────────────────────────────────────────────────────────────────────────
def test_feature_schema_frozen_and_no_future_leak():
    names = kc.FEATURE_NAMES
    assert len(names) == len(set(names)), "duplicate feature names"
    assert len(names) == 41                                              # frozen policy-input dimension
    for nm in names:                                                    # the schema cannot name a future / teacher / K6 quantity
        assert not any(bad in nm.lower() for bad in _FORBIDDEN), f"feature '{nm}' names a forbidden quantity"
    assert "dist_to_corridor" in names and "slip_l" in names and "slip_r" in names


def test_transport_admissibility_algebra():
    ok = kc.TransportAdmissibility(dual_contact=True, straddle=True, straddle_dot=-0.9, fn_min=0.3,
                                   qdot_max=1.0, motion_ok=True)
    assert ok.admissible
    for bad in (kc.TransportAdmissibility(False, True, -0.9, 0.3, 1.0, True),      # lost a tip
                kc.TransportAdmissibility(True, False, 0.4, 0.3, 1.0, True),       # not straddling
                kc.TransportAdmissibility(True, True, -0.9, 0.3, 4.0, False)):     # motion-contract breach
        assert not bad.admissible


def test_windowed_delivery_cfg_warmstarts_at_canonical_theta():
    win = (0.03, 0.05, 0.03, 3.0, 3.0, 0.6)
    cfg = kc._windowed_delivery_cfg(kc.S1_CANONICAL_THETA, win, DELIVERY_CFG, pop=16, iters=3)
    lo, hi, th = np.asarray(cfg.lo), np.asarray(cfg.hi), np.asarray(kc.S1_CANONICAL_THETA)
    assert np.all(lo <= th) and np.all(th <= hi)                        # canonical θ inside the window
    assert np.all(lo >= np.asarray(DELIVERY_CFG.lo) - 1e-9)             # window clamped to the base box
    assert np.all(hi <= np.asarray(DELIVERY_CFG.hi) + 1e-9)
    assert cfg.pop == 16 and cfg.iters == 3 and cfg.elite >= 2 and cfg.deliver


# ── integration (fast committed-bank s1 snapshot) ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def s1_snap():
    from hymeko_rl.experiments.coin_r9_causal_rl import BANK, _load_harness, build_panel
    with open(BANK) as f:
        bank = json.load(f)
    return {ps.tag: ps for ps in build_panel(_load_harness(), bank)}["s1"].snap


@pytest.fixture(scope="module")
def s1_entry(s1_snap):
    return kc.freeze_kinetic_entry(s1_snap)


def _kinetic_controller():
    ap = ApproachParams(qdot_approach=2.4, acquire_squeeze=0.10, kinetic_transport=True, v_floor=0.26,
                        kinetic_vcap=1.8, kinetic_squeeze=0.10, kinetic_entry_v=0.10, kinetic_max_steps=50)
    return ap


def test_roll_until_mirrors_velocity_rollout_trajectory(s1_snap):
    """A full-horizon roll_until (never-firing predicate) reproduces velocity_rollout's coin_trace bit-for-bit — proof the
    capture loop is a faithful mirror of the frozen step kernel, not a divergent reimplementation."""
    ap = _kinetic_controller()
    c_vr = HybridApproachController(s1_snap, TipTransportParams(), ap, DELIVERY_CFG)
    m = velocity_rollout(s1_snap, c_vr, DELIVERY_CFG)
    c_ru = HybridApproachController(s1_snap, TipTransportParams(), ap, DELIVERY_CFG)
    cap = kc.roll_until(s1_snap, c_ru, DELIVERY_CFG, stop_when=lambda c, rl, t: False)
    assert not cap.fired and cap.t == DELIVERY_CFG.horizon
    assert np.allclose(np.asarray(cap.coin_trace), np.asarray(m["coin_trace"]), atol=0.0, rtol=0.0)


def test_transport_snapshot_branch_bit_identity(s1_snap):
    rl = s1_snap.branch()
    ts = kc.TransportSnapshot.from_live(rl, s1_snap.stack, s1_snap.prev_tau)
    b1, b2 = ts.branch(), ts.branch()
    assert np.array_equal(b1.inner.data.qpos, b2.inner.data.qpos)
    assert np.array_equal(b1.inner.data.qvel, b2.inner.data.qvel)
    assert ts.q_hold.shape == (4,) and ts.prev_tau.shape == (4,) and ts.lo.shape == (4,)


def test_kinetic_observe_batch_equals_streaming(s1_snap):
    """Stream kinetic_observe during a rollout, capturing (deepcopy(rl), hist snapshot) at several steps; then recompute the
    SAME feature vectors in batch from the captured states. Batch must equal streaming exactly (the deploy-time invariant)."""
    import copy
    ap = _kinetic_controller()
    c = HybridApproachController(s1_snap, TipTransportParams(), ap, DELIVERY_CFG)
    streamed: list = []
    captured: list = []
    hist: list = []

    def hook(rl, t):
        feat, hframe = kc.kinetic_observe(rl, hist)
        if t in (3, 6, 9, 12):
            captured.append((copy.deepcopy(rl), [list(x) for x in hist]))
            streamed.append(feat.copy())
        hist.append(hframe)

    velocity_rollout(s1_snap, c, DELIVERY_CFG, frame_hook=hook)
    assert len(streamed) >= 2                                           # the rollout reached the sampled steps
    for (rl_b, hist_b), feat_s in zip(captured, streamed):
        feat_b, _ = kc.kinetic_observe(rl_b, hist_b)                   # batch recompute from the captured state
        assert np.array_equal(feat_b, feat_s)                          # batch == streaming, bit-for-bit
        assert feat_b.shape[0] == len(kc.FEATURE_NAMES)


def test_freeze_kinetic_entry_deterministic_and_admissible(s1_entry, s1_snap):
    e2 = kc.freeze_kinetic_entry(s1_snap)
    assert s1_entry.state_hash == e2.state_hash                         # deterministic content hash
    assert s1_entry.entry_step >= 1 and len(s1_entry.state_hash) == 16
    assert s1_entry.admissibility.dual_contact                          # KINETIC entry requires bilateral contact
    assert s1_entry.admissibility.motion_ok                             # within the joint-speed hard cap
    assert s1_entry.entry_v_par >= ApproachParams().kinetic_entry_v - 1e-3   # entered while genuinely moving


def test_kinetic_bank_neighbourhood_deterministic_and_diverse(s1_entry):
    """The K1 neighbourhood generator is deterministic (same seed ⇒ same admissible states + descriptors) and produces a
    physically-diverse, admissibility-gated 4/8/4 set — legal perturbed-control branches, not state edits."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_bank as kb
    acc1, rej1 = kb.generate_neighbourhood(s1_entry.tsnap)
    acc2, rej2 = kb.generate_neighbourhood(s1_entry.tsnap)
    assert len(acc1) + len(rej1) == len(kb.NEIGHBOURHOOD_SPECS) == 16
    assert len(acc1) >= 12                                              # most perturbations stay admissible
    assert [a["provenance"]["label"] for a in acc1] == [a["provenance"]["label"] for a in acc2]   # deterministic set
    assert all(a1["descriptor"] == a2["descriptor"] for a1, a2 in zip(acc1, acc2))                 # deterministic descriptors
    for a in acc1:                                                     # every accepted state is a branchable admissible snapshot
        assert a["tsnap"].admissibility().admissible
        assert a["tsnap"].branch().inner.data.qpos.shape[0] >= 4
    vpar = [a["descriptor"]["v_par"] for a in acc1]
    fnmin = [a["descriptor"]["fn_min"] for a in acc1]
    assert max(vpar) - min(vpar) > 0.1 and max(fnmin) - min(fnmin) > 0.5   # genuine spread (velocity + contact-force range)
    cats = {a["provenance"]["category"] for a in acc1}
    assert cats == {"easy", "medium", "edge"}                          # all three strata represented


def test_receding_horizon_relabel_first_action_only_deterministic(s1_entry):
    r = kc.receding_horizon_relabel(s1_entry.tsnap, budget=0)
    assert r.first_action.shape == (4,) and r.first_action_norm.shape == (4,)
    assert np.abs(r.first_action_norm).max() <= 1.0 + 1e-9             # slew-normalised into the tanh-actor range
    assert isinstance(r.delivers_k6, bool) and r.source == "warm_theta"
    assert "k6_max_dwell" in r.metrics and "dtz_end" in r.metrics       # provenance carries the K6 verdict inputs
    r2 = kc.receding_horizon_relabel(s1_entry.tsnap, budget=0)
    assert np.array_equal(r.first_action, r2.first_action)             # deterministic label
    assert len(r.theta) == 6                                            # θ is provenance, not fed to any policy input
