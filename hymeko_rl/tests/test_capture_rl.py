"""Tests for the Stage-2 verified capture-RL infrastructure (clean, reusable pieces only).

Fast: outcome-independent medoid selection, panel determinism, zero-residual helper. Physics: the bit-exact
zero-residual identity AND that the real zero-init actor outputs exactly zero (the correctness cornerstone), the real
medoid delivering strict K6, the causal observation schema, and perturb_ready keeping the coin canonical. No training
loop, reward, or exploration policy is tested here — the reward-driven policy is not yet achieved (see the audit report);
these tests pin the machinery's correctness for the next session's structure-preserving work.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC, build_home_snapshot

CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")
S1B = Path("reports/2026-07-28-moving-precapture-dynamic-handoff/moving_precapture.json")


# ── fast ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_medoid_is_outcome_independent_and_central():
    a = mp.CaptureParams(n=3, s=0.10, preload_start=0.1, bmax=0.3, residual=np.zeros((2, 4)))
    b = mp.CaptureParams(n=3, s=0.40, preload_start=0.5, bmax=0.5, residual=np.zeros((2, 4)))
    c = mp.CaptureParams(n=3, s=0.70, preload_start=0.9, bmax=0.7, residual=np.zeros((2, 4)))
    assert crl.freeze_scaffold([a, b, c]) is b and crl.freeze_scaffold([c, b, a]) is b   # central, order-independent


def test_panel_deterministic_and_nominal_first():
    p1 = crl.perturbation_panel(n=8, seed=90210)
    p2 = crl.perturbation_panel(n=8, seed=90210)
    assert len(p1) == 8 and np.allclose(p1[3].dq, p2[3].dq) and np.allclose(p1[5].dtau, p2[5].dtau)
    assert np.allclose(p1[0].dq, 0.0) and np.allclose(p1[0].dv, 0.0) and np.allclose(p1[0].dtau, 0.0)


def test_zero_residual_helper():
    assert np.array_equal(crl.zero_residual(np.zeros(31)), np.zeros(4))


# ── physics ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild
    from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    coin = _coin_xy(cradle.branch())
    home = build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)
    ready = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin),
                                pga.TransitConfig()).ready_snapshot
    ref = mp.HandoffReference.from_cradle(cradle)
    r2_fn = deterministic_residual(_rebuild(json.load(open(CKPT))["r2_actor_state"]))
    down = mp.FrozenDownstream(model, norm, r2_fn, cradle.stack)
    sols = [mp.CaptureParams(n=r["params"]["n"], s=r["params"]["s"], preload_start=r["params"]["preload_start"],
                             bmax=r["params"]["bmax"], residual=np.array(r["params"]["residual"]))
            for r in json.load(open(S1B))["seeds"]]
    return {"cradle": cradle, "ready": ready, "ref": ref, "down": down, "coin": coin, "pi0": crl.freeze_scaffold(sols)}


def test_zero_residual_is_scaffold_bit_exact(rig):
    """Cornerstone: the residual roll with zero residual reproduces the scaffold's rollout bit-exact."""
    ref, pi0, coin, stack = rig["ref"], rig["pi0"], rig["coin"], rig["cradle"].stack
    scaffold = mp.PhaseShapeCapture(rig["ready"], ref, stack).roll(pi0)
    res = crl.ResidualCaptureRoll(rig["ready"], ref, stack, pi0, coin).rollout(crl.zero_residual)
    sd, rd = scaffold.snapshot.branch().inner.data, res["snapshot"].branch().inner.data
    assert np.array_equal(sd.qpos, rd.qpos) and np.array_equal(sd.qvel, rd.qvel)
    assert np.allclose(scaffold.snapshot.prev_tau, res["prev"], atol=0.0)


def test_zero_init_actor_outputs_exactly_zero(rig):
    """The real zero-init actor (not a mocked zero vector) outputs EXACTLY zero — so the composed action is bit-exact
    scaffold. This is the gap the composition-only identity test did not cover."""
    res = crl.ResidualCaptureRoll(rig["ready"], rig["ref"], rig["cradle"].stack, rig["pi0"], rig["coin"]).rollout(
        crl.zero_residual)
    fn = crl.policy_residual(crl.make_zero_actor(1))
    out = np.array([fn(o) for o in res["obs"]])
    assert np.all(out == 0.0)


def test_real_medoid_zero_residual_delivers_strict_k6(rig):
    """REAL_MEDOID_SCAFFOLD_IDENTITY: the medoid scaffold with zero learned residual delivers strict K6, and its
    terminal preload is NOT near tau_star (so a cradle-centred preload metric is invalid for this scaffold)."""
    ref, pi0, coin, stack = rig["ref"], rig["pi0"], rig["coin"], rig["cradle"].stack
    res = crl.ResidualCaptureRoll(rig["ready"], ref, stack, pi0, coin).rollout(crl.zero_residual)
    k6, md, safe = rig["down"].deliver(res["snapshot"])
    assert k6 and safe and md < 10.0
    assert float(np.linalg.norm(res["prev"] - ref.tau_star)) > 0.5      # terminal preload far from cradle tau_star


def test_observation_schema_and_no_leak(rig):
    res = crl.ResidualCaptureRoll(rig["ready"], rig["ref"], rig["cradle"].stack, rig["pi0"], rig["coin"]).rollout(
        crl.zero_residual)
    obs = np.array(res["obs"])
    assert obs.shape == (rig["pi0"].steps, crl.OBS_DIM) and not np.isnan(obs).any()
    assert 0.0 <= obs[0, 30] <= 1.0 and obs[-1, 30] == pytest.approx(1.0)


def test_perturb_ready_keeps_coin_canonical(rig):
    pert = crl.Perturbation(np.full(4, 0.01), np.full(4, 0.02), np.full(4, 0.01))
    snap = crl.perturb_ready(rig["ready"], rig["cradle"].stack, pert)
    d0, d1 = rig["ready"].branch().inner.data, snap.branch().inner.data
    assert np.allclose(d1.qpos[4:6], d0.qpos[4:6]) and not np.allclose(d1.qpos[:4], d0.qpos[:4])
    assert np.allclose(np.asarray(snap.prev_tau), np.asarray(rig["ready"].prev_tau) + pert.dtau)
