"""3-agent MultiTreeChannel CTDE — the drop-in interface + gradients into all agents.

Plus the off-policy (TD3) collaborative coin-toss: the deterministic channel-coupled actor + centralized twin
Q-critics, and that they train finite through ``train_offpolicy`` (the 2026-07-03 build)."""
import numpy as np
import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.multichannel_ctde import (
    DeterministicMultiChannelActor,
    build_collaborative_offpolicy,
    build_multichannel_collaborative,
)


def _env() -> PlanarGraspEnv:
    return PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3, task_graph=True)


def _scale(env: PlanarGraspEnv) -> float:
    return float(np.max(np.abs(env.action_space.high)))   # type: ignore[union-attr]


def test_ctde_interface_shapes() -> None:
    env = _env()
    ac = build_multichannel_collaborative(env, hidden=32)
    obs, _ = env.reset(seed=0)
    ob = torch.as_tensor(obs[None], dtype=torch.float32)
    assert ac.action_mean(ob).shape == (1, env.n_actions)      # per-arm means concatenated -> full action
    assert ac.value(ob).shape == (1,)
    a, lp, v = ac.act(ob)
    assert a.shape == (1, env.n_actions) and lp.shape == (1,) and v.shape == (1,)
    lp2, ent, v2 = ac.evaluate(ob, a)
    assert lp2.shape == (1,) and ent.shape == (1,) and v2.shape == (1,)
    assert ac.n_parameters() > 0


def test_ctde_gradients_reach_all_three_agents() -> None:
    env = _env()
    ac = build_multichannel_collaborative(env, hidden=32)
    obs, _ = env.reset(seed=0)
    ob = torch.as_tensor(obs[None], dtype=torch.float32)
    means, value = ac._heads(ob)
    (means.sum() + value.sum()).backward()
    assert any(p.grad is not None for p in ac.actor_backbones[0].parameters())   # actor 1
    assert any(p.grad is not None for p in ac.actor_backbones[1].parameters())   # actor 2
    assert any(p.grad is not None for p in ac.critic_backbone.parameters())      # critic
    assert any(p.grad is not None for p in ac.channel.parameters())             # the coordination channel


def test_ctde_sa_hsikan_backbone_is_cheaper_and_runs() -> None:
    # the SA-HSiKAN (Bᴸ-collapse) arms: same drop-in interface, fewer params than plain HSiKAN.
    env = _env()
    ob = torch.as_tensor(env.reset(seed=0)[0][None], dtype=torch.float32)
    sa = build_multichannel_collaborative(env, kind="sa_hsikan", hidden=32)
    assert sa.action_mean(ob).shape == (1, env.n_actions) and sa.value(ob).shape == (1,)
    hs = build_multichannel_collaborative(env, kind="hsikan", hidden=32)
    assert sa.n_parameters() < hs.n_parameters()                # the collapse is the cheaper backbone


def test_ctde_rejects_unknown_backbone() -> None:
    try:
        build_multichannel_collaborative(_env(), kind="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown backbone kind")


# ── off-policy (TD3) collaborative coin-toss: deterministic channel-coupled actor + centralized twin critics ──


def test_det_ctde_actor_shape_bounds_and_alias() -> None:
    env = _env()
    actor, _ = build_collaborative_offpolicy(env, kind="hsikan", hidden=32)
    ob = torch.as_tensor(env.reset(seed=0)[0][None], dtype=torch.float32)
    a = actor(ob)
    assert a.shape == (1, env.n_actions)                              # per-arm heads concatenated → joint action
    assert torch.allclose(a, actor.action_mean(ob))                  # action_mean is the deterministic mean alias
    assert bool((a.abs() <= _scale(env) + 1e-5).all())               # bounded by action_scale (tanh)
    assert actor.action_dim == env.n_actions                         # the off-policy contract width
    assert actor.n_parameters() > 0


def test_det_ctde_actor_is_deterministic() -> None:
    env = _env()
    actor, _ = build_collaborative_offpolicy(env, kind="hsikan", hidden=32)
    actor.eval()
    ob = torch.as_tensor(env.reset(seed=1)[0][None], dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(actor(ob), actor(ob))                  # no sampling — same input → same action


def test_det_ctde_no_single_backbone_attr() -> None:
    # The trainer's shared-trunk detection uses getattr(actor, "backbone", None); a multi-backbone CTDE actor must
    # expose NO single .backbone so it takes the (correct) non-shared path.
    env = _env()
    actor, _ = build_collaborative_offpolicy(env, kind="hsikan", hidden=32)
    assert getattr(actor, "backbone", None) is None


def test_build_collaborative_offpolicy_critics() -> None:
    env = _env()
    actor, critics = build_collaborative_offpolicy(env, kind="hsikan", hidden=32, n_critics=2)
    assert isinstance(actor, DeterministicMultiChannelActor) and len(critics) == 2
    ob = torch.as_tensor(env.reset(seed=0)[0][None], dtype=torch.float32)
    q = critics[0](ob, actor(ob))                                    # centralized critic ingests the FULL joint action
    assert q.shape == (1,)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in critics[0].modules())   # anti-overestimation default ON


def test_det_ctde_gradients_reach_both_arms_and_channel() -> None:
    env = _env()
    actor, _ = build_collaborative_offpolicy(env, kind="hsikan", hidden=32)
    ob = torch.as_tensor(env.reset(seed=0)[0][None], dtype=torch.float32)
    actor(ob).sum().backward()
    assert any(p.grad is not None for p in actor.actor_backbones[0].parameters())   # arm L reasoning
    assert any(p.grad is not None for p in actor.actor_backbones[1].parameters())   # arm R reasoning
    assert any(p.grad is not None for p in actor.channel.parameters())              # the coordination channel


def test_collaborative_offpolicy_trains_finite() -> None:
    # Integration: the deterministic CTDE actor + centralized twin critics train through train_offpolicy (pure
    # TD3, tiny budget) on the real PlanarGraspEnv — stays finite and the actor moves. Would fail before the
    # action_dim / backbone-optional generalization of the trainer.
    from hymeko_rl.ddpg import td3_config, train_offpolicy

    env = _env()
    actor, critics = build_collaborative_offpolicy(env, kind="hsikan", hidden=32, n_critics=2)
    before = [p.detach().clone() for p in actor.parameters()]
    cfg = td3_config(total_steps=120, start_steps=16, batch_size=16, eval_every=9_999, n_eval=1,
                     warm_start=True, update_every=2, seed=0)   # eval_every > total → no cartpole eval_balance
    hist = train_offpolicy(actor, critics, env, cfg)
    assert isinstance(hist, list)
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    assert any(not torch.allclose(b, p) for b, p in zip(before, actor.parameters()))
