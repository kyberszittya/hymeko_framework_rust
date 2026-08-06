"""HSiKAN structural diagnostic: a healthy policy reads healthy; an injected blow-up is flagged and localised to
its named structural component; a forward crash on a diverged policy is captured, not raised."""
from __future__ import annotations

import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.hsikan_diagnose import diagnose, format_diagnosis
from hymeko_rl.agents.policy import build_policy


def _policy_and_obs() -> tuple[object, torch.Tensor]:
    torch.manual_seed(0)
    env = PlanarGraspEnv(robot=None, max_steps=40)
    feat = int(env.observation_space.shape[1])   # type: ignore[index]
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg, hidden=16)
    obs, _ = env.reset(seed=0)
    return ac, torch.as_tensor(obs[None], dtype=torch.float32)


def test_healthy_policy_reads_healthy() -> None:
    ac, obs = _policy_and_obs()
    diag = diagnose(ac, sample_obs=obs)  # type: ignore[arg-type]
    assert diag.healthy and not diag.bad and diag.forward_error is None
    assert "HEALTHY" in format_diagnosis(diag)


def test_injected_blowup_is_localised() -> None:
    ac, obs = _policy_and_obs()
    dict(ac.named_parameters())["actor_backbone.layers.0.w_neg.weight"].data.view(-1)[0] = float("inf")
    diag = diagnose(ac, sample_obs=obs)  # type: ignore[arg-type]
    assert not diag.healthy
    roles = [p.role for p in diag.bad]
    assert any("W-" in r and "L0" in r for r in roles), roles      # localised to the up-chain agg, layer 0
    # the diverged forward is captured as a signal, never raised
    assert diag.forward_error is not None and "DIVERGED" in format_diagnosis(diag)


def test_denormal_is_flagged() -> None:
    ac, obs = _policy_and_obs()
    dict(ac.named_parameters())["actor_backbone.layers.0.w_self.weight"].data.view(-1)[0] = 1e-40  # subnormal
    diag = diagnose(ac, sample_obs=obs)  # type: ignore[arg-type]
    assert not diag.healthy
    assert any(p.n_denormal > 0 for p in diag.bad)
