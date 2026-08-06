"""Tests for SAC: the squashed-Gaussian actor, twin soft-Q build, and the train loop.

Unit: actor sample is bounded with a finite (corrected) log-prob; action_mean is deterministic; build_sac
makes twin critics for mlp/hsikan. Integration: train_sac runs end-to-end on a tiny budget → finite curve.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.train.ddpg import QCritic
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.train.sac import (
    AlphaMode, SACConfig, SquashedGaussianActor, build_sac, scheduled_alpha, train_sac,
)

_MJCF = emit_cartpole_mjcf()


def _env(max_steps: int = 200) -> InvertedPendulumEnv:
    return InvertedPendulumEnv(mjcf=_MJCF, max_steps=max_steps)


def test_squashed_actor_sample_bounded_with_finite_logprob() -> None:
    torch.manual_seed(0)
    env = _env()
    actor, _ = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                         action_scale=10.0, hidden=16)
    assert isinstance(actor, SquashedGaussianActor)
    obs = torch.randn(8, env.hg.n_vertices, 2)
    a, logp = actor.sample(obs)
    assert a.shape == (8, 1) and a.abs().max() <= 10.0           # tanh-squashed into ±scale
    assert logp.shape == (8,) and torch.isfinite(logp).all()     # corrected log-prob, no NaN
    mean = actor.action_mean(obs)
    assert mean.shape == (8, 1) and mean.abs().max() <= 10.0      # deterministic, also bounded


@pytest.mark.parametrize("kind", ["mlp", "hsikan"])
def test_build_sac_twin_critics(kind: str) -> None:
    env = _env()
    kw = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, critics = build_sac(kind, obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                               action_scale=10.0, n_critics=2, hidden=16, **kw)
    assert len(critics) == 2 and all(isinstance(c, QCritic) for c in critics)
    q = critics[0](torch.randn(3, env.hg.n_vertices, 2), torch.zeros(3, 1))
    assert q.shape == (3,) and torch.isfinite(q).all()


def test_train_sac_runs_and_returns_finite_curve() -> None:
    torch.manual_seed(0)
    env = _env(max_steps=50)
    actor, critics = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                               action_scale=10.0, hidden=16)
    hist = train_sac(actor, critics, env, SACConfig(total_steps=300, start_steps=50, batch_size=16,
                                                    capacity=500, eval_every=150, n_eval=2, seed=0))
    assert len(hist) == 2 and all(np.isfinite(hist))


def test_bc_anchor_pulls_action_mean_toward_demos() -> None:
    """The SAC+BC anchor (bc_coef>0 + offline_data) holds ``action_mean`` near the demos while SAC explores — the
    TD3+BC analogue. Without it, un-anchored SAC drifts (the 2026-07-14 'SAC degrades' config artifact)."""
    env = _env(max_steps=50)
    demo_s = np.stack([env.reset(seed=i)[0] for i in range(64)]).astype(np.float32)   # (64, n_vertices, 2)
    demo_a = np.full((64, 1), 4.0, dtype=np.float32)                                   # a constant target action

    def bc_mse(actor: object) -> float:
        with torch.no_grad():
            pred = actor.action_mean(torch.as_tensor(demo_s))   # type: ignore[attr-defined]
            return float(((pred - torch.as_tensor(demo_a)) ** 2).mean())

    results = {}
    for coef in (0.0, 5.0):
        torch.manual_seed(0)
        actor, critics = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                                   action_scale=10.0, hidden=16)
        before = bc_mse(actor)
        train_sac(actor, critics, env, SACConfig(total_steps=400, start_steps=50, batch_size=16, capacity=500,
                                                 eval_every=1000, log_every=0, seed=0, bc_coef=coef),
                  offline_data=(demo_s, demo_a))
        results[coef] = (before, bc_mse(actor))
    # anchored SAC pulls action_mean toward the demos; plain SAC does not track them
    assert results[5.0][1] < results[5.0][0]                    # bc_coef>0: MSE decreases (clone held)
    assert results[5.0][1] < results[0.0][1]                    # anchored ends far closer to the demos than plain SAC


def test_scheduled_alpha_fixed_is_constant() -> None:
    """FIXED α ignores step and returns init_alpha (the entropy-control correction, 2026-07-15)."""
    a0 = scheduled_alpha(AlphaMode.FIXED, 0, 1000, 0.05, 0.005, 0.6)
    aN = scheduled_alpha(AlphaMode.FIXED, 999, 1000, 0.05, 0.005, 0.6)
    assert a0 == pytest.approx(0.05) and aN == pytest.approx(0.05)


def test_scheduled_alpha_anneal_linear_then_holds() -> None:
    """ANNEAL is init_alpha at step 0, alpha_final at anneal_frac·total, and holds after."""
    total, initv, finalv, frac = 1000, 0.1, 0.005, 0.6
    assert scheduled_alpha(AlphaMode.ANNEAL, 0, total, initv, finalv, frac) == pytest.approx(initv)
    mid = scheduled_alpha(AlphaMode.ANNEAL, 300, total, initv, finalv, frac)   # halfway through the ramp
    assert initv > mid > finalv                                                # strictly decreasing on the ramp
    assert scheduled_alpha(AlphaMode.ANNEAL, 600, total, initv, finalv, frac) == pytest.approx(finalv)
    assert scheduled_alpha(AlphaMode.ANNEAL, 900, total, initv, finalv, frac) == pytest.approx(finalv)  # held after


def test_scheduled_alpha_rejects_auto() -> None:
    """AUTO must not route through the scheduler — it tracks the learned log_alpha (fails loudly if misused)."""
    with pytest.raises(ValueError, match="AUTO tracks"):
        scheduled_alpha(AlphaMode.AUTO, 0, 1000, 0.1, 0.005, 0.6)


def test_stable_config_is_the_single_source_of_the_correction() -> None:
    """SACConfig.stable() = the coin-toss-validated F-SAC-1 correction as ONE preset (no per-experiment re-typing):
    init_alpha 0.1, ANNEAL, lr 3e-4, anti-divergence defaults kept. The RAW default stays AUTO so the F-SAC-9
    collapse discriminators still reproduce the collapse. (The stable stack fixes the critic divergence, not the
    off-policy delivery collapse — F-SAC-4/8/9/10 / F-PP-009.)"""
    c = SACConfig.stable(total_steps=1000, seed=3)
    assert c.init_alpha == 0.1 and c.alpha_mode is AlphaMode.ANNEAL
    assert c.actor_lr == 3e-4 and c.critic_lr == 3e-4
    assert c.reward_norm and c.max_grad_norm > 0            # anti-divergence defaults preserved
    assert c.total_steps == 1000 and c.seed == 3
    assert SACConfig(total_steps=1000).alpha_mode is AlphaMode.AUTO   # raw default = classic SAC (collapse repro)
    assert SACConfig.stable(total_steps=500, rollout_anchor_coef=20.0).rollout_anchor_coef == 20.0
    assert SACConfig.stable(total_steps=500, start_steps=42).start_steps == 42   # overrides thread through


def test_rollout_anchor_pulls_action_mean_toward_teacher_on_visited_states() -> None:
    """The rollout-state DAgger anchor (rollout_anchor_coef>0 + a `.reset()/.action(env)` teacher) holds the actor's
    ``action_mean`` near the teacher on the states the actor itself visits — the 2026-07-15 covariate-shift fix. A
    constant teacher makes this checkable: with the anchor the visited-state mean tracks the teacher, without it drifts."""
    env = _env(max_steps=50)

    class _ConstTeacher:                       # `.reset()`/`.action(env)` contract (PushDemonstrator-shaped), constant
        def reset(self) -> None: ...
        def action(self, _env: object) -> np.ndarray:
            return np.full((1,), 3.0, dtype=np.float32)

    def visited_mean_err(actor: object) -> float:
        errs = []
        o, _ = env.reset(seed=7)
        for _ in range(40):
            with torch.no_grad():
                m = actor.action_mean(torch.as_tensor(o[None], dtype=torch.float32))   # type: ignore[attr-defined]
            errs.append(float((m - 3.0) ** 2))
            o, _r, term, trunc, _ = env.step(np.zeros(1, dtype=np.float32))
            if term or trunc:
                o, _ = env.reset(seed=7)
        return float(np.mean(errs))

    results = {}
    for coef in (0.0, 5.0):
        torch.manual_seed(0)
        actor, critics = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                                   action_scale=10.0, hidden=16)
        train_sac(actor, critics, env,
                  SACConfig(total_steps=400, start_steps=50, batch_size=16, capacity=500, eval_every=1000,
                            log_every=0, seed=0, rollout_anchor_coef=coef, rollout_anchor_capacity=500),
                  dagger_teacher=_ConstTeacher())
        results[coef] = visited_mean_err(actor)
    assert results[5.0] < results[0.0]         # the rollout anchor pulls the visited-state mean toward the teacher


def test_train_sac_greedy_rollout_runs_finite() -> None:
    """greedy_rollout steps the env with action_mean (no exploration) — the F-SAC-9 discriminator that makes the
    rollout-anchor states self-consistent with greedy eval. Must train end-to-end to a finite curve."""
    torch.manual_seed(0)
    env = _env(max_steps=50)
    actor, critics = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                               action_scale=10.0, hidden=16)
    hist = train_sac(actor, critics, env, SACConfig(total_steps=300, start_steps=50, batch_size=16, capacity=500,
                                                    eval_every=150, n_eval=2, seed=0, log_every=0, greedy_rollout=True))
    assert len(hist) == 2 and all(np.isfinite(hist))


@pytest.mark.parametrize("mode", [AlphaMode.FIXED, AlphaMode.ANNEAL])
def test_train_sac_runs_under_entropy_control_modes(mode: AlphaMode) -> None:
    """FIXED/ANNEAL entropy-control modes train end-to-end to a finite curve (no α-tuning divergence)."""
    torch.manual_seed(0)
    env = _env(max_steps=50)
    actor, critics = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                               action_scale=10.0, hidden=16)
    hist = train_sac(actor, critics, env,
                     SACConfig(total_steps=300, start_steps=50, batch_size=16, capacity=500, eval_every=150,
                               n_eval=2, seed=0, log_every=0, init_alpha=0.1, alpha_mode=mode, alpha_final=0.005))
    assert len(hist) == 2 and all(np.isfinite(hist))
