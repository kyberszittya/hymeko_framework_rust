"""Tests for the cart-pole sim interface (load a stored policy, run it, render).

Unit: a policy stored to .hymeko loads back and acts identically (bit-exact weights ⇒ identical mean action),
with the architecture inferred from shapes. Integration: render_run writes a GIF + trajectory PNG (skipped if
no GL context for offscreen MuJoCo).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hymeko_rl.train.ddpg import build_offpolicy
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.agents.policy import build_policy
from hymeko_rl.agents.policy_store import policy_to_hymeko
from hymeko_rl.viz.render_inverted_pendulum import load_policy_from_hymeko, render_run
from hymeko_rl.train.sac import build_sac

_MJCF = emit_cartpole_mjcf()


def _env(max_steps: int = 200) -> InvertedPendulumEnv:
    return InvertedPendulumEnv(mjcf=_MJCF, max_steps=max_steps)


def test_load_from_hymeko_reconstructs_and_acts_identically(tmp_path: Path) -> None:
    torch.manual_seed(0)
    env = _env()
    ac = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=env.hg, hidden=24, n_layers=2)
    p = policy_to_hymeko(ac.state_dict(), tmp_path / "p.hymeko", tier="auto")
    loaded = load_policy_from_hymeko(p, env)
    # architecture inferred from shapes:
    assert loaded.actor_mean.in_features == 24  # hidden
    obs = torch.randn(3, env.hg.n_vertices, 2)
    assert torch.allclose(ac.action_mean(obs), loaded.action_mean(obs), atol=0.0)  # bit-exact weights


@pytest.mark.parametrize("algo", ["ppo", "ddpg", "sac"])
def test_load_reconstructs_each_actor_type(tmp_path: Path, algo: str) -> None:
    """A saved PPO / DDPG / SAC actor reloads as the right class with bit-exact greedy actions — the
    render dispatches on the stored keys, not on a single assumed architecture."""
    torch.manual_seed(0)
    env = _env()
    hg = env.hg
    if algo == "ppo":
        mod = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=hg, hidden=16)
    elif algo == "ddpg":
        mod = build_offpolicy("hsikan", obs_dim=2, flat_dim=hg.n_vertices * 2, action_dim=1,
                              action_scale=10.0, hidden=16, hg_state=hg)[0]
    else:
        mod = build_sac("hsikan", obs_dim=2, flat_dim=hg.n_vertices * 2, action_dim=1,
                        action_scale=10.0, hidden=16, hg_state=hg)[0]
    p = policy_to_hymeko(mod.state_dict(), tmp_path / "p.hymeko", meta={"algo": algo, "backbone": "hsikan"})
    ac = load_policy_from_hymeko(p, env)
    x = torch.zeros(2, hg.n_vertices, 2)
    assert torch.equal(ac.action_mean(x), mod.action_mean(x))


def test_load_rejects_mismatched_vertex_count(tmp_path: Path) -> None:
    torch.manual_seed(0)
    env = _env()
    # a stored policy whose a_pos is 3×3 cannot load into the 2-vertex cart-pole env.
    sd = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=env.hg, hidden=8).state_dict()
    sd["actor_backbone.a_pos"] = torch.zeros(3, 3)
    p = policy_to_hymeko(sd, tmp_path / "bad.hymeko")
    with pytest.raises(ValueError):
        load_policy_from_hymeko(p, env)


def test_render_run_writes_gif_and_trajectory(tmp_path: Path) -> None:
    torch.manual_seed(0)
    env = _env(max_steps=12)
    ac = build_policy("hsikan", obs_dim=2, action_dim=1, hg_state=env.hg, hidden=16)
    try:
        paths = render_run(ac, env, tmp_path / "run", seed=0, fps=20)
    except Exception as e:  # noqa: BLE001 — offscreen GL may be unavailable on a headless host
        pytest.skip(f"offscreen render unavailable: {type(e).__name__}: {e}")
    assert paths["gif"].is_file() and paths["gif"].stat().st_size > 0
    assert paths["trajectory"].is_file() and paths["trajectory"].stat().st_size > 0
