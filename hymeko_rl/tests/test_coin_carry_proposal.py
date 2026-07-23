"""Tests for the multimodality-preserving proposal + fixed local-search wrapper (Stage 4c/5 library)."""
import json

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_carry_option import make_option_actor  # noqa: F401  (kept for parity of setup helpers)
from hymeko_rl.coin_delivery.coin_carry_proposal import (
    canonical_label,
    denorm_theta,
    fit_proposal,
    kmeans,
    norm_theta,
    search_select,
)
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, DIM, T_MAX, T_MIN
from hymeko_rl.coin_delivery.coin_late_start import LateStart, reconstruct_handoff
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"


def _setup():
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    r = cfg["banks"]["late_dev"]["rows"][0]
    ls = LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5])
    return pi0, base, ls


def _bank():
    z = np.load(f"{D}/carry_option_teacher_bank_v1.npz")
    return z["obs"].astype(np.float32), z["theta"].astype(np.float32)


def test_norm_denorm_roundtrip():
    th = np.array([[3, -3, 2, -2, 1, -1, 0, 0, 2.5, -2.5, 1.5, -1.5, 2, 10, 18]], np.float32)
    assert np.allclose(denorm_theta(norm_theta(th)), th, atol=1e-4)
    z = norm_theta(th)
    assert np.abs(z[:, :12]).max() <= 1.0 + 1e-6 and 0.0 <= z[:, 12:].min() and z[:, 12:].max() <= 1.0 + 1e-6


def test_kmeans_shapes():
    torch.manual_seed(0)
    X = np.random.randn(40, DIM).astype(np.float32)
    lab, med = kmeans(X, 5, seed=0)
    assert lab.shape == (40,) and lab.min() >= 0 and lab.max() < 5 and len(med) == 5 and len(set(med)) == 5


def test_proposal_predicts_legal_theta_and_residual_helps():
    obs, th = _bank()
    prop, info = fit_proposal(obs, th, K=8, clf_epochs=120, res_epochs=120, seed=0)
    out = prop.theta(obs[:16])
    assert out.shape == (16, DIM)
    assert np.abs(out[:, :12]).max() <= A_BOUND + 1e-4                     # amplitudes legal
    assert out[:, 12:].min() >= T_MIN - 1e-3 and out[:, 12:].max() <= T_MAX + 1e-3   # durations legal
    assert prop.theta(obs[0]).shape == (DIM,)                             # single-obs path
    ids = prop._ids(obs)
    mse_med = float(((norm_theta(prop.templates[ids]) - norm_theta(th)) ** 2).sum(1).mean())
    mse_res = float(((norm_theta(prop.theta(obs)) - norm_theta(th)) ** 2).sum(1).mean())
    assert mse_res < mse_med                                              # template+residual fits better than template-only


def test_search_select_b0_is_center_bpos_is_legal():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    center = np.array([2, -2, 1, -1, 0, 0, 1, -1, 0.5, -0.5, 0, 0, 5, 6, 7], np.float32)
    th0, out0 = search_select(rl, gate, center, pi0, base, np.random.default_rng(0), b=0, horizon=60)
    assert np.array_equal(th0, center) and "k6" in out0                   # b=0 executes the center directly
    thb, outb = search_select(rl, gate, center, pi0, base, np.random.default_rng(1), b=6, horizon=60)
    assert np.abs(thb[:12]).max() <= A_BOUND + 1e-4 and T_MIN - 1e-3 <= thb[12:].min()   # selected candidate is legal θ
    assert thb.shape == (DIM,) and "k6" in outb


def test_canonical_label_abstains_or_returns_legal():
    pi0, base, ls = _setup()
    rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360)
    th, out = canonical_label(rl, gate, pi0, base, np.random.default_rng(2), shots=8, horizon=80)
    assert (th is None) or (th.shape == (DIM,) and np.abs(th[:12]).max() <= A_BOUND + 1e-4)
    if th is not None:
        assert int(out["k6"]) == 1 or int(out["reached_handoff"]) == 1   # only confident (K6/handoff) labels, never least-bad
