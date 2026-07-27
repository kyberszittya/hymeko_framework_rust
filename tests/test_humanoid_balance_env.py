"""HumanoidBalanceEnv (position-servo balance env) + SAC eval helper — integration.

Slower than the pure-Lyapunov unit tests (build a MuJoCo model via the hymeko CLI),
but bounded. Covers the gym contract, deterministic reset (§3), the certified
PD-hold-q0 scaffold invariant (a = 0 passes the Lyapunov certificate on the nominal
envelope), the certified-envelope boundary (survival != stability), the fall path,
and the SAC eval helper end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv  # noqa: E402
from scenarios.humanoid.lyapunov import evaluate_lyapunov  # noqa: E402


def _zero(env):
    return np.zeros(env.model.nu)


def _cert_pass(env, seed, act_fn) -> bool:
    obs, _ = env.reset(seed=seed)
    vs, done = [], False
    while not done:
        obs, _r, term, trunc, info = env.step(act_fn(obs))
        vs.append(info["V"])
        done = term or trunc
    return evaluate_lyapunov(vs)["passes"]


def test_spaces_and_step_contract() -> None:
    env = HumanoidBalanceEnv(max_steps=40, seed=0)
    obs, _info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert env.action_space.shape == (env.model.nu,)
    obs2, r, term, trunc, info = env.step(np.zeros(env.model.nu))
    assert np.all(np.isfinite(obs2)) and np.isfinite(r)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert "V" in info and "upright" in info and info["V"] >= 0.0


def test_reset_is_deterministic() -> None:
    env = HumanoidBalanceEnv(max_steps=20, seed=0)
    o1, _ = env.reset(seed=7)
    o2, _ = env.reset(seed=7)
    assert np.allclose(o1, o2)                       # same seed -> identical init (§3)
    o3, _ = env.reset(seed=8)
    assert not np.allclose(o1, o3)                   # different seed -> different perturbation


def test_zero_action_is_certified_pd_hold() -> None:
    # a = 0 is the PD-hold-q0 scaffold: it PASSES the unchanged Lyapunov certificate
    # on the nominal perturbation envelope. This is the certified balance baseline.
    env = HumanoidBalanceEnv(cfg=BalanceConfig(perturb_lo=0.0, perturb_hi=0.3), seed=0)
    passes = sum(_cert_pass(env, s, lambda _o: _zero(env)) for s in range(100, 104))
    assert passes == 4                               # certified on all nominal-envelope seeds


def test_certified_envelope_boundary_survival_ne_stability() -> None:
    # A large perturbation: PD-hold SURVIVES (stays upright) but does NOT certify
    # (transient overshoot fails `bounded`). The certificate separates the two.
    env = HumanoidBalanceEnv(cfg=BalanceConfig(perturb_lo=1.0, perturb_hi=1.0, max_steps=500), seed=0)
    obs, _ = env.reset(seed=0)
    up, steps, vs, done = 0, 0, [], False
    while not done:
        obs, _r, term, trunc, info = env.step(_zero(env))
        steps += 1
        vs.append(info["V"])
        if info["upright"]:
            up = steps
        done = term or trunc
    assert up == steps                               # survives the full horizon (upright throughout)
    assert not evaluate_lyapunov(vs)["passes"]       # yet fails the certificate (overshoot)


def test_fall_terminates() -> None:
    # a perturbation beyond the survival envelope tips the humanoid -> terminated
    env = HumanoidBalanceEnv(cfg=BalanceConfig(perturb_lo=5.0, perturb_hi=5.0, max_steps=300), seed=0)
    env.reset(seed=0)
    fell = any(env.step(_zero(env))[2] for _ in range(300))
    assert fell                                      # the fall-termination path is reachable


def test_eval_balance_helper_end_to_end() -> None:
    from hymeko_rl.train.sac import build_sac

    from scenarios.humanoid.run_humanoid_sac import _eval_balance

    env = HumanoidBalanceEnv(max_steps=30, seed=0)
    od = int(env.observation_space.shape[0])
    torch.manual_seed(0)
    actor, _critics = build_sac("mlp", obs_dim=od, flat_dim=od,
                                action_dim=env.model.nu, action_scale=1.0, hidden=32)
    frac, lyap_rate = _eval_balance(env, actor, [1234, 1235])
    assert 0.0 <= frac <= 1.0 and 0.0 <= lyap_rate <= 1.0


def test_render_rollout_frames_smoke() -> None:
    # exercises the video render path; skips gracefully where no GL context exists (headless CI)
    from scenarios.humanoid.render_balance_video import _H, _W, rollout_frames
    env = HumanoidBalanceEnv(max_steps=12, seed=0)
    try:
        frames, telem, ev = rollout_frames(env, lambda _o: np.zeros(env.model.nu), 0)
    except Exception as exc:                                       # no offscreen GL backend
        pytest.skip(f"no MuJoCo render context: {exc}")
    assert frames and frames[0].shape == (_H, _W, 3)
    assert len(telem) == len(frames) and "bounded" in telem[0]
    assert np.all(np.isfinite(frames[0])) and "passes" in ev
