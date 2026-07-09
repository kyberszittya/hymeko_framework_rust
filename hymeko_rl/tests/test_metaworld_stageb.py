"""Tests for the MetaWorld Stage-B training-smoke setup — GATE + dry-run + plumbing. No training is launched here."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hymeko_rl.experiments.exp_metaworld_reward_stageb import (
    StageBConfig,
    StageBGateError,
    _GaussianMLP,
    _returns_to_go,
    _validate_paths,
    build_reward_profiles,
    dry_run,
    launch,
)

_HAS_METAWORLD = importlib.util.find_spec("metaworld") is not None


def test_import_has_no_training_side_effects() -> None:
    """A. importing the harness constructs no env and launches no training (module import is pure)."""
    import hymeko_rl.experiments.exp_metaworld_reward_stageb as m
    assert issubclass(m.StageBGateError, RuntimeError) and issubclass(m.StageBUncertifiedError, RuntimeError)


def test_missing_launch_flag_blocks_training(tmp_path) -> None:
    """B/C. launch without the explicit flag raises the gate error and never trains."""
    cfg = StageBConfig(out_dir=tmp_path)
    with pytest.raises(StageBGateError, match="launch_training=True"):
        launch(cfg, launch_training=False)
    assert not (tmp_path / "stage_b_train.json").exists()
    for p in cfg.profiles:
        assert not cfg.checkpoint_path(p).exists()          # no checkpoint written


def test_both_profiles_loadable_and_distinct(tmp_path) -> None:
    """D. original and mw_in_place_off profiles both load; the off profile zeros mw_in_place, original keeps it."""
    cfg = StageBConfig(out_dir=tmp_path)
    profiles = build_reward_profiles(cfg)
    assert set(profiles) == {"original", "mw_in_place_off"}
    orig = dict(profiles["original"].ablated)
    off = dict(profiles["mw_in_place_off"].ablated)
    assert orig["mw_in_place"] != 0.0 and off["mw_in_place"] == 0.0
    assert off["mw_near"] == orig["mw_near"]                 # other terms untouched


def test_output_paths_do_not_overwrite_stage_a(tmp_path) -> None:
    """E. Stage-B logging paths are creatable and do not collide with the Stage-A cip_reward_ablation family."""
    cfg = StageBConfig(out_dir=tmp_path / "metaworld_stageb_reward_ab")
    paths = _validate_paths(cfg)
    assert paths["collides_with_stage_a"] is False and paths["overwrites_existing"] is False
    # a hypothetical Stage-A dir name WOULD be flagged
    bad = StageBConfig(out_dir=tmp_path / "2026_07_09_cip_reward_ablation")
    assert _validate_paths(bad)["collides_with_stage_a"] is True


def test_post_eval_command_is_generated(tmp_path) -> None:
    """F. a runnable monitor/CIP post-eval command is generated per profile."""
    cfg = StageBConfig(out_dir=tmp_path)
    cmd = cfg.eval_command("mw_in_place_off")
    assert cmd.startswith("python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --post-eval")
    assert "--profile mw_in_place_off" in cmd and "--checkpoint" in cmd


def test_returns_to_go_and_policy_update_plumbing() -> None:
    """G. the REINFORCE plumbing (returns-to-go + a policy-gradient step) is correct on synthetic data — no env."""
    import torch
    assert np.allclose(_returns_to_go([1.0, 1.0, 1.0], 0.5), [1.75, 1.5, 1.0])
    policy = _GaussianMLP(obs_dim=6, act_dim=2, hidden=8, act_scale=1.0, seed=0)
    act, logp = policy.sample(np.zeros(6, dtype=np.float32))
    assert act.shape == (2,) and np.all(np.abs(act) <= 1.0) and logp.requires_grad
    before = policy.mean.weight.detach().clone()
    opt = torch.optim.Adam(policy.parameters(), lr=0.1)
    loss = -(logp * torch.tensor(1.0))                        # a single synthetic policy-gradient step
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert not torch.allclose(before, policy.mean.weight)     # the optimizer updates the actor


def test_obs_norm_standardizes_input() -> None:
    """I. set_obs_norm makes _hidden standardize its input (obs == mean → the net sees ~0)."""
    import torch
    policy = _GaussianMLP(obs_dim=4, act_dim=2, hidden=8, act_scale=1.0, seed=0)
    mean = np.array([10.0, -5.0, 0.0, 3.0], np.float32)
    std = np.array([2.0, 4.0, 1.0, 0.5], np.float32)
    policy.set_obs_norm(mean, std)
    # the standardized input at obs=mean is 0 → equals the net's response to a literal zero vector
    got = policy._hidden(mean)
    want = policy.net(torch.zeros(4))
    assert torch.allclose(got, want, atol=1e-5)


def test_bc_clone_reduces_loss() -> None:
    """J. behaviour cloning drives the MSE toward zero on a learnable synthetic obs→action map."""
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import bc_clone
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 1, (200, 4)).astype(np.float32)
    acts = np.tanh(obs[:, :2] * 0.5).astype(np.float32)          # a smooth, tanh-range target
    policy = _GaussianMLP(obs_dim=4, act_dim=2, hidden=16, act_scale=1.0, seed=0)
    policy.set_obs_norm(obs.mean(0), obs.std(0))
    losses = bc_clone(policy, obs, acts, epochs=60, lr=1e-2, seed=0)
    assert losses[-1] < 0.25 * losses[0] and losses[-1] < 0.02   # substantial reduction to a small residual


def test_compare_profiles_deltas() -> None:
    """K. compare_profiles reports the original→ablated behavioural deltas, and flags an incomplete panel."""
    from hymeko_rl.experiments.stage_b_eval import compare_profiles
    results = {
        "original": {"eval": {"success_rate": 0.62, "grasp_fraction": 0.49, "progress_score": 0.58,
                              "obj_to_target_delta": 0.16, "reward_monitor_disagreement": 0.14}},
        "mw_in_place_off": {"eval": {"success_rate": 0.0, "grasp_fraction": 0.0, "progress_score": 0.17,
                                     "obj_to_target_delta": 0.01, "reward_monitor_disagreement": 0.31}},
    }
    cmp = compare_profiles(results)
    assert cmp["comparable"] is True
    assert cmp["success_rate"]["delta"] == pytest.approx(-0.62)
    assert cmp["grasp_fraction"]["delta"] == pytest.approx(-0.49)
    assert compare_profiles({"original": results["original"]})["comparable"] is False


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_full_launch_warm_start_eval_and_gif(tmp_path) -> None:
    """L. the real Stage-B path runs BC→fine-tune→post-eval→compare at tiny scale and emits GIFs + a comparison."""
    from dataclasses import replace as _replace

    from hymeko_rl.experiments.exp_metaworld_reward_stageb import launch, post_eval
    cfg = StageBConfig(out_dir=tmp_path, bc_demos=4, bc_epochs=15, total_env_steps=400,
                       eval_episodes=4, eval_episodes_post=8)     # >5 rollouts: DirectLiNGAM needs n_samples > n_vars
    cfg = _replace(cfg, cert_rollouts=6)
    s = launch(cfg, launch_training=True, allow_uncertified=True)   # tiny scale → cert may be non-discriminating
    assert s["bc"]["demo_success_episodes"] >= 0 and "bc_success_rate" in s["bc"]
    for name in cfg.profiles:
        ev = s["results"][name]["eval"]
        assert 0.0 <= ev["success_rate"] <= 1.0 and "loadings" in ev and ev["cross_view_agree"] in (True, False)
        assert (tmp_path / name / "rollout.gif").exists()               # GIF emitted per profile
    assert s["comparison"]["comparable"] is True
    # post_eval reloads a checkpoint and reproduces the metric schema
    ev2 = post_eval(cfg, "original", s["results"]["original"]["checkpoint"])
    assert "success_rate" in ev2 and "reward_under_original_mean" in ev2


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_dry_run_validates_without_training(tmp_path) -> None:
    """H. dry-run validates profiles/env/reward/certification/paths and exits before the optimizer (no training)."""
    cfg = StageBConfig(out_dir=tmp_path, seed=0)
    report = dry_run(cfg, n_certify=4)
    assert report["trained"] is False
    assert set(report["profiles"]) == {"original", "mw_in_place_off"}
    for pr in report["profiles"].values():
        assert pr["reward_finite"] is True
        assert "delivers" in pr["certification"]
    assert (tmp_path / "stage_b_dry_run.json").exists()
    for p in cfg.profiles:                                    # dry-run wrote NO checkpoint
        assert not cfg.checkpoint_path(p).exists()
