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


def test_sample_specs_deterministic_and_covers_strata():
    """The K1-A candidate sampler is deterministic and spans the easy/medium/edge strata; it perturbs only the first three θ
    dims (squeeze/forward/balance), leaving ramp/release/brake fixed."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_bank as kb
    s1 = kb.sample_specs(48)
    s2 = kb.sample_specs(48)
    assert len(s1) == 48
    assert [s.label for s in s1] == [s.label for s in s2]                # deterministic
    assert {s.category for s in s1} == {"easy", "medium", "edge"}        # all three strata sampled
    assert all(s.dtheta[3] == 0.0 and s.dtheta[4] == 0.0 and s.dtheta[5] == 0.0 for s in s1)   # only sqz/fwd/bal perturbed
    assert any(s.category == "edge" and s.dtheta[0] < 0 and s.dtheta[1] > 0.05 for s in s1)     # a near-slip edge mode present


def test_labeled_state_carries_41d_obs_no_leak(s1_entry):
    """A K1-A labelled state carries the canonical 41-D observation (history = frames before the state) and an admissible
    snapshot; the observation is finite and never carries teacher/future/K6 info (structural + purity of kinetic_observe)."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_bank as kb
    spec = kb.PerturbSpec("easy", "probe", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 3, 0.0)
    rec, prov = kb.labeled_state(s1_entry.tsnap, spec, seed=1)
    assert rec is not None and prov["admissible"]                       # an easy 3-step advance stays admissible
    assert rec["obs"].shape[0] == len(kc.FEATURE_NAMES) == 41 and np.all(np.isfinite(rec["obs"]))
    assert rec["tsnap"].admissibility().admissible
    rec2, _ = kb.labeled_state(s1_entry.tsnap, spec, seed=1)            # deterministic obs
    assert np.array_equal(rec["obs"], rec2["obs"])


def test_clone_batch_equals_streaming_and_bounded():
    """The K2 clone's recurrent forward is batch == streaming (a sequence at once equals step-by-step with the carried hidden
    state), deterministic on replay, and tanh-bounded — the deploy-time controls."""
    import numpy as _np
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_clone as kcl
    from hymeko_rl.experiments.coin_kinetic_k2_clone import BANK
    bank = json.loads(BANK.read_text())
    model, norm, hist = kcl.train_clone(bank, kcl.CloneTrainConfig(epochs=40), seed=0)
    assert hist[-1] < hist[0]                                            # BC loss decreased
    obs = _np.array([r["obs"] for r in bank], _np.float64)
    model.eval()
    with torch.no_grad():
        batched = model(torch.tensor(norm.apply(obs), dtype=torch.float32).view(1, -1, kcl.OBS_DIM))[0]
        batched = batched.view(-1, kcl.ACT_DIM).numpy()
    actor = kcl.CloneActor(model, norm)
    actor.reset()
    streamed = _np.array([actor.act(o) for o in obs], _np.float64)
    assert float(_np.max(_np.abs(batched - streamed))) < 1e-5           # batch == streaming
    actor.reset()
    streamed2 = _np.array([actor.act(o) for o in obs], _np.float64)
    assert _np.array_equal(streamed, streamed2)                         # deterministic hidden-state reset + replay
    assert _np.abs(streamed).max() <= 1.0 + 1e-6                        # tanh-bounded action basis


def test_kinetic_clone_controller_frozen_approach_and_bounded(s1_snap):
    """The clone controller leaves the frozen APPROACH untouched (its APPROACH-phase Δτ equals the frozen scaffold's), the
    clone acts ONLY in the KINETIC phase, and every emitted Δτ is within the slew bound — no teacher in the loop."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option import kinetic_clone as kcl
    from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
    from hymeko_rl.experiments.coin_kinetic_k2_clone import BANK
    bank = json.loads(BANK.read_text())
    model, norm, _h = kcl.train_clone(bank, kcl.CloneTrainConfig(epochs=10), seed=0)
    clone = kcl.KineticCloneController(s1_snap, kcl.CloneActor(model, norm))
    ref = HybridApproachController(s1_snap, TipTransportParams(), ApproachParams(kinetic_transport=True), DELIVERY_CFG)
    clone.reset()
    ref.reset()
    rl_c, rl_r = s1_snap.branch(), s1_snap.branch()
    prev_c = _np.asarray(s1_snap.prev_tau, _np.float64).copy()
    from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
    from hymeko_rl.env.motion_contract import govern_torque
    import mujoco
    def _gc(_m, d):
        d.ctrl[:4] = govern_torque(d.ctrl[:4], d.qvel[:4], s1_snap.stack.gov)
    mujoco.set_mjcb_control(_gc)
    try:
        for t in range(1, 8):                                          # first few steps are the frozen APPROACH
            dc = clone.dtau_for_step(rl_c, t, prev_c)
            assert _np.max(_np.abs(dc)) <= clone.slew + 1e-9           # bounded by slew
            if clone.clone_trace[-1]["kind"] == "KINETIC_CLONE":
                break
            dr = ref.dtau_for_step(rl_r, t, prev_c)
            assert _np.allclose(dc, dr, atol=0.0)                      # APPROACH Δτ identical to the frozen scaffold
            prev_c = _np.clip(prev_c + dc, s1_snap.lo, s1_snap.hi)
            step_ablation(rl_c, _np.asarray(prev_c, _np.float32), "A")
            step_ablation(rl_r, _np.asarray(prev_c, _np.float32), "A")
    finally:
        mujoco.set_mjcb_control(None)
    assert all(r["kind"] in ("FROZEN", "KINETIC_CLONE") for r in clone.clone_trace)


def test_residual_policy_zero_init_and_reward_ignores_raw_vpar():
    """R0/R1 pure checks: a zero-initialised residual policy outputs exactly 0 (update-zero identity precondition), and the
    task-tied reward does NOT reward raw v_par magnitude (only progress / close+moving / K6, minus stall/clamp/etc.)."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option import kinetic_rl as krl
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import NormStats
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualActor, ResidualBounds, ResidualPolicy
    ract = ResidualActor(ResidualPolicy(zero_init=True), NormStats(_np.zeros(41), _np.ones(41)))
    assert _np.abs(ract.act(_np.random.randn(41))).max() == 0.0 and ResidualBounds().alpha >= 0
    m = {"peak_qdot": 2.0, "peak_coin_speed": 0.4, "k6_delivered": False}
    slow = [{"kind": "KINETIC_CLONE", "v_par": 0.15, "fn_l": 1.0, "fn_r": 1.0, "dtz_mm": 50.0}]
    fast = [{"kind": "KINETIC_CLONE", "v_par": 0.60, "fn_l": 1.0, "fn_r": 1.0, "dtz_mm": 50.0}]
    r_slow, _ = krl.kinetic_reward(m, 40.0, slow, baseline_mm=46.2)
    r_fast, _ = krl.kinetic_reward(m, 40.0, fast, baseline_mm=46.2)
    assert r_slow == r_fast                                             # same landing ⇒ same reward regardless of v_par magnitude
    stalled = [{"kind": "KINETIC_CLONE", "v_par": -0.05, "fn_l": 1.0, "fn_r": 1.0, "dtz_mm": 50.0}]
    r_stall, dec = krl.kinetic_reward(m, 40.0, stalled, baseline_mm=46.2)
    assert r_stall < r_slow and dec["stalls"] == 1                      # a stall is penalised


def test_residual_update_zero_identity_full_chain(s1_snap):
    """R0 gate as a test: a zero residual over the clone reproduces the K2 clone bit-for-bit on the full frozen chain."""
    import numpy as _np
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_rl as krl
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticClone, KineticCloneController, NormStats
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    from hymeko_rl.experiments.coin_kinetic_r1_rl import CLONE_CKPT
    ckpt = torch.load(CLONE_CKPT, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    norm = NormStats(_np.array(ckpt["norm"]["mean"]), _np.array(ckpt["norm"]["std"]))
    m_clone = velocity_rollout(s1_snap, KineticCloneController(s1_snap, CloneActor(model, norm)), DELIVERY_CFG)
    m_zero, _dtz, _kin = krl.deploy_residual(s1_snap, CloneActor(model, norm), _np.zeros(4), ResidualBounds(alpha=0.15))
    assert _np.array_equal(_np.asarray(m_clone["coin_trace"]), _np.asarray(m_zero["coin_trace"]))   # bit-identical
    assert bool(m_clone["k6_delivered"]) == bool(m_zero["k6_delivered"])


def test_reward2_light_contact_progress_and_state_dependent_contact_loss():
    """R2 reward pure checks: light-contact forward progress is rewarded, a heavy clamp is penalised, and the contact-loss
    penalty is STATE-DEPENDENT — losing contact far from the zone is punished, a close-and-moving release is not."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
    light = [{"dtz_mm": 70.0, "v_par": 0.3, "fn_l": 1.0, "fn_r": 1.0}, {"dtz_mm": 66.0, "v_par": 0.3, "fn_l": 1.0, "fn_r": 1.0}]
    r_light, _d = krl2.reward2(light, entry_dtz=74.0, min_dtz=50.0, k6=False, safe=True)
    assert r_light[0] > 0 and r_light[1] < r_light[0] + 100                 # light-contact progress is rewarded per step
    clamp = [{"dtz_mm": 70.0, "v_par": 0.3, "fn_l": 5.0, "fn_r": 5.0}]      # heavy clamp (fn>4) — no follow, a clamp penalty
    r_clamp, dc = krl2.reward2(clamp, entry_dtz=74.0, min_dtz=50.0, k6=False, safe=True)
    assert dc["clamp"] > 0 and r_clamp[0] < 0
    far = krl2.reward2([{"dtz_mm": 60.0, "v_par": 0.3, "fn_l": 1.0, "fn_r": 1.0}], 74.0, 46.0, False, True)[1]
    near = krl2.reward2([{"dtz_mm": 26.0, "v_par": 0.1, "fn_l": 1.0, "fn_r": 1.0}], 30.0, 18.0, False, True)[1]
    assert near["terminal"] > far["terminal"]                              # close+moving release ≫ early contact loss
    assert near["released"] and not far["released"]


def test_temporal_residual_update_zero_identity(s1_snap):
    """R2 update-zero: a zero-initialised per-step temporal residual reproduces the K2 clone bit-for-bit on the full chain."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticClone, KineticCloneController, NormStats
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
        AUG_DIM, KineticTemporalResidualController, TemporalResidualPolicy, deterministic_residual)
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    from hymeko_rl.experiments.coin_kinetic_r1_rl import CLONE_CKPT
    import torch
    assert AUG_DIM == 41 + 64 + 4
    ckpt = torch.load(CLONE_CKPT, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    norm = NormStats(_np.array(ckpt["norm"]["mean"]), _np.array(ckpt["norm"]["std"]))
    mc = velocity_rollout(s1_snap, KineticCloneController(s1_snap, CloneActor(model, norm)), DELIVERY_CFG)
    zpol = deterministic_residual(TemporalResidualPolicy(zero_init=True))
    ctrl = KineticTemporalResidualController(s1_snap, CloneActor(model, norm), zpol, ResidualBounds(alpha=0.15))
    mz = velocity_rollout(s1_snap, ctrl, DELIVERY_CFG)
    assert _np.array_equal(_np.asarray(mc["coin_trace"]), _np.asarray(mz["coin_trace"]))   # bit-identical
    assert ctrl.aug_trace and ctrl.aug_trace[0][0].shape[0] == AUG_DIM                      # augmented state has clone hidden


def test_champion_key3_stall_aware_and_reward3_envelope():
    """R3-B pure checks: the stall-aware champion ranks a clean-moving policy above a closer-but-stalled one, and reward3 adds a
    velocity-envelope penalty when v_par falls below the distance-dependent floor."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
    clean = krl3.champion_key3(min_dtz=40, exit_dtz=52, k6=False, safe=True, released=False, jerk=0.1,
                               has_clamp=False, has_stall=False, has_reversal=False)
    stalled_closer = krl3.champion_key3(min_dtz=45, exit_dtz=47, k6=False, safe=True, released=False, jerk=0.1,
                                        has_clamp=False, has_stall=True, has_reversal=True)
    assert clean > stalled_closer                                      # cleanliness ranks above a lower contact-exit dtz
    below = [{"dtz_mm": 50.0, "v_par": 0.05, "fn_l": 0.5, "fn_r": 0.5}]   # v_par 0.05 << envelope floor 0.18 at 50 mm
    _r, d_below = krl3.reward3(below, entry_dtz=55.0, min_dtz=45.0, k6=False, safe=True)
    above = [{"dtz_mm": 50.0, "v_par": 0.30, "fn_l": 0.5, "fn_r": 0.5}]
    _r2, d_above = krl3.reward3(above, entry_dtz=55.0, min_dtz=45.0, k6=False, safe=True)
    assert d_below["envelope"] > 0 and d_above["envelope"] == 0        # slow-gripping is penalised; fast is not
    assert krl3.envelope_floor(50.0) >= krl3.envelope_floor(25.0)      # the floor relaxes toward the release corridor


@pytest.fixture(scope="module")
def r3b_frontiers(s1_snap):
    """The clone (zero-init temporal residual) as a fast deterministic 'champion': capture its healthy interior frontiers
    (63–73 mm) and its edge state (58–63 mm, the 61 mm contact-loss boundary), plus the uninterrupted coin trace — the shared
    fixture for the four hybrid-boundary restart tests."""
    import numpy as _np
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticClone, NormStats
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
        KineticTemporalResidualController, TemporalResidualPolicy, deterministic_residual)
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    from hymeko_rl.experiments.coin_kinetic_r1_rl import CLONE_CKPT
    ckpt = torch.load(CLONE_CKPT, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    norm = NormStats(_np.array(ckpt["norm"]["mean"]), _np.array(ckpt["norm"]["std"]))
    zpol = TemporalResidualPolicy(zero_init=True)
    b = ResidualBounds(alpha=0.15)
    healthy, _b1 = krl3.capture_frontiers(s1_snap, CloneActor(model, norm), zpol, b, dtz_lo=63.0, dtz_hi=73.0)
    _h2, boundary = krl3.capture_frontiers(s1_snap, CloneActor(model, norm), zpol, b, dtz_lo=58.0, dtz_hi=63.0)
    uc = _np.asarray(velocity_rollout(s1_snap, KineticTemporalResidualController(
        s1_snap, CloneActor(model, norm), deterministic_residual(zpol), b), DELIVERY_CFG)["coin_trace"])
    return {"model": model, "norm": norm, "bounds": b, "zpol": zpol, "healthy": healthy, "boundary": boundary, "uc": uc}


def test_healthy_frontier_restart_one_step_identity(r3b_frontiers):
    """HEALTHY_FRONTIER_RESTART_ONE_STEP_IDENTITY: restarting from a healthy frontier reproduces the uninterrupted continuation
    for one step (bit-tight; the off-by-one guard). Multi-step divergence is chaotic (stiff contact) and NOT asserted."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import KineticTemporalResidualController, deterministic_residual
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
    fx = r3b_frontiers
    assert fx["healthy"]
    f = fx["healthy"][0]
    rc = KineticTemporalResidualController(f, CloneActor(fx["model"], fx["norm"]), deterministic_residual(fx["zpol"]),
                                           fx["bounds"], start_kinetic=f.start_state())
    rct = _np.asarray(velocity_rollout(f, rc, DELIVERY_CFG)["coin_trace"])
    one_step = float(_np.linalg.norm(rct[0] - fx["uc"][f.step_index - 1]))     # coin_trace[i] = coin AFTER step i+1
    assert one_step < 5e-5                                                     # < 0.05 mm — one-step restart is exact


def test_healthy_frontier_restart_continuation(r3b_frontiers):
    """HEALTHY_FRONTIER_RESTART_CONTINUATION: a healthy frontier restarts into MULTIPLE valid KINETIC transitions (interior,
    not a mode boundary), moving and physically consistent."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
    fx = r3b_frontiers
    assert fx["healthy"] and all(f.restart_steps >= krl3.N_EXIT_MARGIN for f in fx["healthy"])
    assert all(f.v_par > 0 and 0.0 < f.guard_margin_fn < krl3.krl2.FN_LIGHT for f in fx["healthy"])


def test_boundary_frontier_rejected(r3b_frontiers):
    """BOUNDARY_FRONTIER_REJECTED: the ~61 mm edge-of-contact-loss snapshot is classified as a hybrid-boundary state (restart <
    N_EXIT_MARGIN), NOT admitted as a curriculum frontier — the hybrid-guard-boundary-aliasing case."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
    fx = r3b_frontiers
    assert fx["boundary"] and any(58.0 <= f.dtz_mm <= 63.0 for f in fx["boundary"])
    assert all(f.restart_steps < krl3.N_EXIT_MARGIN for f in fx["boundary"])   # boundary states restart-alias (few/no steps)


def test_r2_champion_frontier_capture_properties(r3b_frontiers):
    """R2_CHAMPION_FRONTIER_CAPTURE: several healthy frontiers captured with the required properties — positive v_par, light
    non-degenerate force, no stall, complete restorable causal state (clone hidden + prev residual + prev_tau), restart yields
    real KINETIC transitions."""
    fx = r3b_frontiers
    assert len(fx["healthy"]) >= 2
    for f in fx["healthy"]:
        assert f.clone_hidden is not None and f.prev_res.shape[0] == 4 and f.prev_tau.shape[0] == 4
        assert f.v_par > 0 and f.guard_margin_fn > 0 and f.frames_since_entry >= 2 and f.restart_steps >= 2


def test_receding_horizon_relabel_first_action_only_deterministic(s1_entry):
    r = kc.receding_horizon_relabel(s1_entry.tsnap, budget=0)
    assert r.first_action.shape == (4,) and r.first_action_norm.shape == (4,)
    assert np.abs(r.first_action_norm).max() <= 1.0 + 1e-9             # slew-normalised into the tanh-actor range
    assert isinstance(r.delivers_k6, bool) and r.source == "warm_theta"
    assert "k6_max_dwell" in r.metrics and "dtz_end" in r.metrics       # provenance carries the K6 verdict inputs
    r2 = kc.receding_horizon_relabel(s1_entry.tsnap, budget=0)
    assert np.array_equal(r.first_action, r2.first_action)             # deterministic label
    assert len(r.theta) == 6                                            # θ is provenance, not fed to any policy input


# ----- Authority audit (learning-free reachability of the ≤30 mm corridor under three residual-authority families) -----

def test_authority_reachability_pass_predicate():
    """AUTHORITY_REACHABILITY_PASS is a conjunction: reaching the corridor is not enough — it must be clean (no stall/reversal/
    clamp), still moving (+v_par), light (Fn < 2 N), and safe. Each single violation flips the verdict to False."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_authority as ka
    ok = {"min_dtz": 28.0, "exit_v": 0.3, "exit_fn": 0.1, "stalls": 0, "reversals": 0, "clamps": 0, "safe": True}
    assert ka.reachability_pass(ok)
    assert not ka.reachability_pass({**ok, "min_dtz": 31.0})            # 1 mm past the corridor → fail
    assert not ka.reachability_pass({**ok, "exit_v": 0.0})              # not moving at exit → fail
    assert not ka.reachability_pass({**ok, "exit_fn": ka.FN_LIGHT})     # contact not light → fail
    assert not ka.reachability_pass({**ok, "stalls": 1})               # stalled to get there → fail
    assert not ka.reachability_pass({**ok, "reversals": 1})            # sign-reversed v_par → fail
    assert not ka.reachability_pass({**ok, "clamps": 1})               # clamped the coin → fail
    assert not ka.reachability_pass({**ok, "safe": False})             # unsafe (qdot/coin speed) → fail


@pytest.fixture(scope="module")
def authority_restart(r3b_frontiers):
    """A healthy clone-interior frontier (reused from the R3-B capture) as the shared restart point for the authority tests."""
    fx = r3b_frontiers
    assert fx["healthy"]
    return {"f": fx["healthy"][0], "model": fx["model"], "norm": fx["norm"], "bounds": fx["bounds"]}


def test_authority_zero_residual_reduces_to_clone_both_families(authority_restart):
    """Update-zero identity across the new controller AND the structured basis: a zero residual sequence reproduces the frozen
    clone's restart continuation BIT-FOR-BIT for both A0 (per-joint) and A2 (structured coin-following) — so the A2 basis maps a
    zero coefficient to a zero Δτ. A non-zero A2 sequence must DIVERGE, proving the structured mapping is live (not a no-op)."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option import kinetic_authority as ka
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    fx = authority_restart
    f = fx["f"]

    def _trace(family: str, seq: _np.ndarray) -> _np.ndarray:
        ctrl = ka.KineticAuthorityController(f, CloneActor(fx["model"], fx["norm"]), ka.SequenceResidual(seq), family, 0.25,
                                             start_kinetic=f.start_state())
        return _np.asarray(velocity_rollout(f, ctrl, DELIVERY_CFG)["coin_trace"])
    a0_zero = _trace("A0", _np.zeros((8, ka.ACT_DIM)))
    a2_zero = _trace("A2", _np.zeros((8, ka.A2_DIM)))
    assert a0_zero.shape == a2_zero.shape and _np.array_equal(a0_zero, a2_zero)     # both zero-residual paths = the clone
    a2_live = _trace("A2", _np.full((8, ka.A2_DIM), 0.8))
    assert a2_live.shape == a2_zero.shape and not _np.allclose(a2_live, a2_zero)    # a real coefficient moves the coin path


def test_authority_cem_deterministic_and_bounded(authority_restart):
    """The CEM reachability search is deterministic (fixed RNG) and reports a well-formed, safe, bounded result: same (family, α)
    twice → identical metrics; family/alpha echoed; min_dtz finite and no worse than the clone's own reach."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_authority as ka
    fx = authority_restart
    tiny = ka.AuthorityCEMConfig(horizon=6, pop=8, iters=2)
    r1 = ka.authority_cem(fx["f"], fx["model"], fx["norm"], "A2", 0.25, bounds=fx["bounds"], cfg=tiny)
    r2 = ka.authority_cem(fx["f"], fx["model"], fx["norm"], "A2", 0.25, bounds=fx["bounds"], cfg=tiny)
    assert r1 == r2                                                     # deterministic given the seed
    assert r1["family"] == "A2" and r1["alpha"] == 0.25
    assert r1["safe"] and 0.0 < r1["min_dtz_mm"] < 1e4                  # finite, safe reachability


def test_a2_basis_matrix_single_source_matches_structured_u(authority_restart):
    """`a2_structured_u` is exactly `a2_basis_matrix · coeffs` (the DRY refactor is behaviour-identical): the matrix columns are
    the 5 structured directions, and the two share one source. A full-rank 4×5 basis spans R⁴ (every joint direction is
    expressible), which is why the A2-orthogonal projection residual is the meaningful 'missing direction' probe."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option import kinetic_authority as ka
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    fx = authority_restart
    rl = fx["f"].branch()
    e_par = _np.asarray(rl.inner.direction_to_zone()[0], _np.float64)
    _ = CloneActor(fx["model"], fx["norm"])                              # keep the fixture's model alive alongside the branch
    basis = ka.a2_basis_matrix(rl, e_par)
    assert basis.shape == (4, 5)
    for c in (_np.array([1.0, 0, 0, 0, 0]), _np.array([0.2, -0.3, 0.5, -0.1, 0.4]), _np.zeros(5)):
        assert _np.allclose(ka.a2_structured_u(rl, c, e_par), basis @ c)   # structured_u == basis · coeffs, exactly
    assert _np.linalg.matrix_rank(basis) == 4                            # spans R⁴ ⇒ any teacher direction is in the A2 span


# ----- Teacher-torque-span diagnostic (WHY the bounded residual saturates ~6 mm short) -----

def test_project_onto_span_full_rank_zero_and_deficient_residual():
    """`project_onto_span`: a full-rank basis leaves ~0 orthogonal residual (direction expressible); a rank-deficient basis
    leaves exactly the out-of-span norm; coeffs reconstruct the projection."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option import kinetic_torque_span as kts
    d = _np.array([0.3, -0.2, 0.1, 0.4])
    full = _np.random.default_rng(1).standard_normal((4, 5))
    coeffs, ortho, dn = kts.project_onto_span(d, full)
    assert ortho < 1e-9 and abs(dn - _np.linalg.norm(d)) < 1e-12 and _np.allclose(full @ coeffs, d)
    deficient = _np.zeros((4, 5))
    deficient[0, 0] = deficient[1, 1] = 1.0                              # spans only the first two axes
    _c, ortho_def, _dn = kts.project_onto_span(d, deficient)
    assert abs(ortho_def - _np.hypot(0.1, 0.4)) < 1e-9                   # exactly the (axis-2, axis-3) out-of-span part


@pytest.fixture(scope="module")
def teacher_decomp(s1_snap):
    """Roll the delivering teacher θ (K0 entry+full_cem) from the frozen KINETIC entry and decompose it against the clone."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_contract as _kc
    from hymeko_rl.coin_delivery.theta_option import kinetic_torque_span as kts
    from hymeko_rl.experiments.coin_kinetic_r1_rl import CLONE_CKPT
    theta = [0.0, 0.2714, -0.057, 16.5378, 8.7978, 3.4604]
    entry = _kc.freeze_kinetic_entry(s1_snap, seed=_kc.S1_SEED)
    model, norm = kts.load_clone(CLONE_CKPT)
    m, steps = kts.decompose_teacher_vs_clone(entry.tsnap, theta, model, norm)
    return {"entry": entry, "theta": theta, "metrics": m, "steps": steps, "summary": kts.summarize_decomposition(steps)}


def test_teacher_theta_controller_reproduces_rollout_primitive(teacher_decomp, s1_snap):
    """The teacher-θ controller rolled through the deploy kernel reproduces `rollout_primitive(entry, θ)` BIT-FOR-BIT — the
    delivering trajectory the decomposition reads is the real teacher, and that teacher DELIVERS strict K6."""
    import numpy as _np

    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    from hymeko_rl.coin_delivery.theta_option import kinetic_contract as _kc
    fx = teacher_decomp
    rp = rollout_primitive(fx["entry"].tsnap, tuple(fx["theta"]), _kc.DELIVERY_CFG)
    assert _np.array_equal(_np.asarray(fx["metrics"]["coin_trace"]), _np.asarray(rp["coin_trace"]))   # bit-identical kernel
    assert fx["metrics"]["k6_delivered"] and fx["metrics"]["dtz_end"] * 1000 < 25.0                   # the teacher delivers


def test_teacher_correction_is_magnitude_gap_in_a2_span(teacher_decomp):
    """The central diagnostic: the delivering teacher's per-step correction over the clone lies ENTIRELY in the A2 structured
    span (0 orthogonal residual — the direction is expressible) yet its MAGNITUDE far exceeds the residual bound α = 0.15 (and
    even 2α). So the missing ingredient is authority MAGNITUDE, not a missing action direction."""
    s = teacher_decomp["summary"]
    assert s["transport_steps"] > 0
    assert s["max_a2_ortho"] < 1e-3                                     # direction in span (A0 = R⁴ trivially; A2 too)
    assert s["frac_exceeds_2alpha0"] >= 0.5 and s["max_d_inf"] > 0.30   # correction magnitude >> the α/2α residual bound


# ----- R3-C action-preserving authority unlock (the mandatory pre-RL gate) -----

@pytest.fixture(scope="module")
def clone_mn():
    """The frozen K2 clone (model, norm) — the shared scaffold for the R3-C construction tests."""
    import numpy as _np
    import torch

    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import KineticClone, NormStats
    from hymeko_rl.experiments.coin_kinetic_r1_rl import CLONE_CKPT
    ckpt = torch.load(CLONE_CKPT, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "norm": NormStats(_np.array(ckpt["norm"]["mean"]), _np.array(ckpt["norm"]["std"]))}


def test_zero_init_detactor_exact_zero():
    """`zero_init_detactor` yields a DetActor that outputs EXACTLY 0 for arbitrary inputs (the bit-exact zero expansion head)."""
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM
    from hymeko_rl.coin_delivery.theta_option.kinetic_rl2 import ACT_DIM
    from hymeko_rl.option_rl.agents import make_actor
    actor = ku.zero_init_detactor(make_actor("td3", AUG_DIM, ACT_DIM))
    with torch.no_grad():
        for x in (torch.zeros(4, AUG_DIM), torch.randn(4, AUG_DIM), 10.0 * torch.randn(4, AUG_DIM)):
            assert float(actor(x).abs().max()) == 0.0


def test_authority_expansion_update_zero_identity(s1_snap, clone_mn):
    """`AUTHORITY_EXPANSION_UPDATE_ZERO_IDENTITY`: with a zero expansion head, the unlock controller is BIT-IDENTICAL to the
    frozen R2 residual controller (`u = clip(u_clone + 0.15·a_R2, −1, 1)`) for BOTH families — regardless of β. Uses an arbitrary
    fixed frozen r2_fn (the identity is a property of the construction, not of the specific R2 champion)."""
    import numpy as _np
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
        AUG_DIM, KineticTemporalResidualController, TemporalResidualPolicy, deterministic_residual)
    from hymeko_rl.coin_delivery.theta_option.kinetic_rl2 import ACT_DIM
    from hymeko_rl.option_rl.agents import make_actor
    m, norm = clone_mn["model"], clone_mn["norm"]
    torch.manual_seed(7)
    r2_fn = deterministic_residual(TemporalResidualPolicy())            # an arbitrary FIXED frozen R2-style head (non-zero)
    b = ResidualBounds(alpha=ku.ALPHA0)
    ref = velocity_rollout(s1_snap, KineticTemporalResidualController(s1_snap, CloneActor(m, norm), r2_fn, b), DELIVERY_CFG)
    for beta in (ku.C1_BETA, ku.C2_BETA):
        exp = ku.zero_init_detactor(make_actor("td3", AUG_DIM, ACT_DIM))
        ctrl = ku.AuthorityUnlockController(s1_snap, CloneActor(m, norm), deterministic_residual(exp), b,
                                            r2_fn=r2_fn, beta=beta)
        got = velocity_rollout(s1_snap, ctrl, DELIVERY_CFG)
        assert _np.array_equal(_np.asarray(got["coin_trace"]), _np.asarray(ref["coin_trace"]))   # bit-identical, any β


def test_authority_unlock_action_slew_safe_under_large_beta(s1_snap, clone_mn):
    """Safety is the final clip + slew + governor, INDEPENDENT of β: a non-zero expansion at the largest β keeps every applied
    Δτ within the slew bound and the rollout inside the motion contract (peak_qdot ≤ 3, coin speed ≤ 1.5)."""
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
        TemporalResidualPolicy, deterministic_residual)
    m, norm = clone_mn["model"], clone_mn["norm"]
    torch.manual_seed(3)
    r2_fn = deterministic_residual(TemporalResidualPolicy())
    exp_fn = deterministic_residual(TemporalResidualPolicy())           # a real non-zero expansion head
    ctrl = ku.AuthorityUnlockController(s1_snap, CloneActor(m, norm), exp_fn, ResidualBounds(alpha=ku.ALPHA0),
                                        r2_fn=r2_fn, beta=ku.C2_BETA)
    mo = velocity_rollout(s1_snap, ctrl, DELIVERY_CFG)
    slew = float(s1_snap.stack.tau_rate * s1_snap.stack.control_dt)
    kin = [r for r in ctrl.residual_trace]
    assert kin and all(max(abs(x) for x in r["u"]) <= 1.0 + 1e-9 for r in kin)          # every action in [-1,1]
    assert mo["peak_qdot"] <= 3.0 and mo["peak_coin_speed"] <= 1.5 and slew > 0         # motion contract holds at β=1.85


def test_authority_unlock_deterministic_replay(s1_snap, clone_mn):
    """Deterministic reset/replay: two rollouts of the same unlock controller policy produce bit-identical coin traces."""
    import numpy as _np
    import torch

    from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
        TemporalResidualPolicy, deterministic_residual)
    m, norm = clone_mn["model"], clone_mn["norm"]
    torch.manual_seed(11)
    r2_fn = deterministic_residual(TemporalResidualPolicy())
    exp_fn = deterministic_residual(TemporalResidualPolicy())
    b = ResidualBounds(alpha=ku.ALPHA0)

    def _roll():
        c = ku.AuthorityUnlockController(s1_snap, CloneActor(m, norm), exp_fn, b, r2_fn=r2_fn, beta=ku.C1_BETA)
        return _np.asarray(velocity_rollout(s1_snap, c, DELIVERY_CFG)["coin_trace"])
    assert _np.array_equal(_roll(), _roll())


def test_champion_key_unlock_and_expl_noise_scaling():
    """Pure R3-C checks: the freeze order ranks strict K6 above everything and cleanliness above min_dtz; the exploration σ is
    β-scaled so the APPLIED noise β·σ is β-invariant (a larger β is not larger physical exploration)."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
    k6 = ku.champion_key_unlock(k6=True, safe=True, clean=False, min_dtz=45, released=False)
    clean_close = ku.champion_key_unlock(k6=False, safe=True, clean=True, min_dtz=30, released=True)
    dirty_closer = ku.champion_key_unlock(k6=False, safe=True, clean=False, min_dtz=25, released=True)
    assert k6 > clean_close > dirty_closer                              # strict K6 ≻ … ≻ clean ≻ (closer-but-dirty)
    assert abs(ku.expl_noise_for(ku.C1_BETA) * ku.C1_BETA - ku.R2_APPLIED_NOISE) < 1e-9    # applied noise β-invariant
    assert ku.expl_noise_for(ku.C2_BETA) < ku.expl_noise_for(ku.C1_BETA)                   # larger β ⇒ smaller normalised σ


# ----- Explicit APPROACH→KINETIC handoff-reset contract (H0 direct vs H1 explicit reset) -----

@pytest.fixture(scope="module")
def r2_fn_from_ckpt():
    """A frozen R2 residual fn rebuilt from a committed multiseed checkpoint (no 300-option regen)."""
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
    from hymeko_rl.experiments.coin_kinetic_ablation import CKPT_DIR, _rebuild
    ck = json.load(open(CKPT_DIR / "seed_02" / "checkpoint.json"))
    return deterministic_residual(_rebuild(ck["r2_champ_state"]))


def test_handoff_reset_online_equals_frozen_entry(s1_snap, clone_mn, r2_fn_from_ckpt):
    """`HANDOFF_RESET_EXPLICIT` + `ONLINE_FROZEN_ENTRY_EQUIVALENCE`: H1 emits exactly one HANDOFF_RESET before the first
    KINETIC_CLONE, and its online post-reset state reproduces the offline frozen entry BIT-EXACTLY (dtz/qpos/prev_tau Δ = 0).
    So the frozen entry is a legitimate first-class online mode, not a privileged snapshot."""
    import numpy as _np

    from hymeko_rl.coin_delivery.theta_option import kinetic_contract as _kc
    from hymeko_rl.coin_delivery.theta_option import kinetic_handoff_reset as hr
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    m, norm = clone_mn["model"], clone_mn["norm"]
    ctrl = hr.HandoffResetTemporalController(s1_snap, CloneActor(m, norm), r2_fn_from_ckpt, ResidualBounds())
    velocity_rollout(s1_snap, ctrl, DELIVERY_CFG)
    kinds = [r["kind"] for r in ctrl.clone_trace]
    assert kinds.count("HANDOFF_RESET") == 1                                   # exactly one explicit reset
    assert kinds.index("HANDOFF_RESET") < kinds.index("KINETIC_CLONE")         # …before the first policy action
    entry = _kc.freeze_kinetic_entry(s1_snap)
    e_q = _np.asarray(entry.tsnap.branch().inner.data.qpos[:4])
    online_q = _online_qpos(s1_snap, m, norm, r2_fn_from_ckpt)
    assert float(_np.max(_np.abs(e_q - online_q))) < 1e-9                      # online post-reset qpos == frozen entry, bit-exact


def _online_qpos(snap, model, norm, r2_fn):
    """The qpos immediately after H1's HANDOFF_RESET step (the online frozen entry)."""
    import copy as _copy

    import mujoco as _mj
    import numpy as _np

    from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
    from hymeko_rl.coin_delivery.theta_option import kinetic_handoff_reset as hr
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.env.motion_contract import govern_torque
    ctrl = hr.HandoffResetTemporalController(snap, CloneActor(model, norm), r2_fn, ResidualBounds())
    ctrl.reset()
    rl, prev = snap.branch(), _np.asarray(snap.prev_tau, _np.float64).copy()
    _mj.set_mjcb_control(lambda _m, dt: dt.ctrl[:4].__setitem__(slice(None), govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)))
    try:
        for t in range(1, DELIVERY_CFG.horizon + 1):
            nb = len([r for r in ctrl.clone_trace if r["kind"] == "HANDOFF_RESET"])
            prev = _np.clip(prev + ctrl.dtau_for_step(rl, t, prev), snap.lo, snap.hi)
            step_ablation(rl, _np.asarray(prev, _np.float32), "A")
            if len([r for r in ctrl.clone_trace if r["kind"] == "HANDOFF_RESET"]) > nb:
                return _np.asarray(_copy.deepcopy(rl).inner.data.qpos[:4])
    finally:
        _mj.set_mjcb_control(None)
    return _np.zeros(4)


def test_direct_vs_reset_handoff_are_distinct_contracts(s1_snap, clone_mn, r2_fn_from_ckpt):
    """H0 DIRECT_HANDOFF and H1 EXPLICIT_HANDOFF_RESET are genuinely different controllers: the direct chain has no reset event
    and reaches a different min_dtz than the reset chain (the extra servo step is load-bearing, not cosmetic)."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_handoff_reset as hr
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import KineticTemporalResidualController
    from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
    m, norm = clone_mn["model"], clone_mn["norm"]
    b = ResidualBounds()
    h0c = KineticTemporalResidualController(s1_snap, CloneActor(m, norm), r2_fn_from_ckpt, b)
    h0 = velocity_rollout(s1_snap, h0c, DELIVERY_CFG)
    h1c = hr.HandoffResetTemporalController(s1_snap, CloneActor(m, norm), r2_fn_from_ckpt, b)
    h1 = velocity_rollout(s1_snap, h1c, DELIVERY_CFG)
    assert not any(r["kind"] == "HANDOFF_RESET" for r in h0c.clone_trace)      # H0 (direct) has no reset event
    assert any(r["kind"] == "HANDOFF_RESET" for r in h1c.clone_trace)          # H1 (explicit) does
    assert abs(_min_dtz_mm(s1_snap, h0) - _min_dtz_mm(s1_snap, h1)) > 1.0      # the two contracts reach materially different states


def test_r2_h1_collect_and_eval_wellformed(s1_snap, clone_mn):
    """The R2-under-H1 training hooks reuse the tested curriculum through the H1 controller: `make_collect_r2_h1` yields
    per-step transitions and `make_eval_r2_h1` returns a well-formed champion (k6_strict, single HANDOFF_RESET) — with a
    zero-init residual, no training needed."""
    from hymeko_rl.coin_delivery.theta_option import kinetic_r2_h1 as r2h1
    from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
    from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM
    from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import zero_init_detactor
    from hymeko_rl.option_rl.agents import make_actor
    m, norm = clone_mn["model"], clone_mn["norm"]

    def cf():
        return CloneActor(m, norm)
    bounds, w = ResidualBounds(alpha=r2h1.R2_ALPHA), krl2.Reward2Weights()
    actor = zero_init_detactor(make_actor("td3", AUG_DIM, ACT_DIM))
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
    trans = r2h1.make_collect_r2_h1(s1_snap, cf, bounds, w)(deterministic_residual(actor))
    assert isinstance(trans, list) and trans and len(trans[0]) == 5            # (s,a,r,s2,done) per-step transitions
    key, aux = r2h1.make_eval_r2_h1(s1_snap, cf, bounds, w)(actor)
    assert "k6_strict" in aux and isinstance(aux["k6_strict"], bool) and key[0] == int(aux["k6_strict"])
