"""Unified off-policy framework: TD3+BC config + the k-critic / mixed-HSiKAN axes + the BC-term integration."""
import numpy as np

from hymeko_rl.train.ddpg import (
    adaptive_bc_config,
    build_offpolicy,
    td3_bc_config,
    td3_config,
    train_offpolicy,
)
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf


def _cartpole() -> tuple[InvertedPendulumEnv, int, int]:
    env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    ss = env.observation_space.shape
    assert ss is not None
    return env, int(ss[0]), int(ss[1])


def test_td3_bc_config_fields() -> None:
    c = td3_bc_config()
    assert c.n_critics == 2 and c.policy_delay == 2 and c.target_noise > 0   # TD3 axes
    assert c.bc_coef > 0.0 and c.warm_start and c.critic_warmup == 2000       # BC bridge
    assert td3_bc_config(bc_coef=5.0, total_steps=1234).bc_coef == 5.0
    assert td3_config().bc_coef == 0.0                                        # plain TD3 has no BC term


def test_k_critic_builds_ensemble() -> None:
    env, nv, feat = _cartpole()
    _actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                      action_scale=env.force_mag, n_critics=4)
    assert len(critics) == 4                                                  # k-critic ensemble axis


def test_mixed_hsikan_backbone_builds() -> None:
    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mixture", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2, hg_state=env.hg)
    assert len(critics) == 2 and abs(actor.action_scale - env.force_mag) < 1e-6


def test_td3bc_train_runs_with_bc_term() -> None:
    # Integration: critic_warmup < total_steps, so the actor updates and the BC anchor term is exercised.
    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    rng = np.random.default_rng(0)
    bc_obs = rng.standard_normal((48, nv, feat)).astype(np.float32)
    bc_act = rng.uniform(-env.force_mag, env.force_mag, (48, 1)).astype(np.float32)
    cfg = td3_bc_config(total_steps=150, start_steps=20, critic_warmup=10, batch_size=32,
                        eval_every=150, n_eval=2, seed=0)
    hist = train_offpolicy(actor, critics, env, cfg, offline_data=(bc_obs, bc_act))
    assert len(hist) >= 1 and all(np.isfinite(h) for h in hist)


def test_adaptive_bc_config_and_schedule_runs() -> None:
    c = adaptive_bc_config()
    assert c.adaptive_bc and c.bc_coef > 0.0 and c.warm_start and c.n_critics == 2
    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    rng = np.random.default_rng(0)
    bc_obs = rng.standard_normal((48, nv, feat)).astype(np.float32)
    bc_act = rng.uniform(-env.force_mag, env.force_mag, (48, 1)).astype(np.float32)
    cfg = adaptive_bc_config(total_steps=200, start_steps=20, critic_warmup=10, batch_size=32,
                             eval_every=80, n_eval=2, seed=0)   # >=2 evals so the BC weight reschedules
    hist = train_offpolicy(actor, critics, env, cfg, offline_data=(bc_obs, bc_act))
    assert len(hist) >= 2 and all(np.isfinite(h) for h in hist)


def test_shared_trunk_shares_one_backbone() -> None:
    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2, shared_trunk=True, hg_state=env.hg)
    assert actor.backbone is critics[0].backbone is critics[1].backbone   # ONE shared trunk object
    assert actor.detach_backbone


def test_shared_trunk_actor_grad_excludes_trunk() -> None:
    import torch
    env, nv, feat = _cartpole()
    actor, _critics = build_offpolicy("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                      action_scale=env.force_mag, n_critics=1, shared_trunk=True, hg_state=env.hg)
    actor(torch.randn(3, nv, feat)).sum().backward()                     # an actor-side gradient
    assert all(p.grad is None for p in actor.backbone.parameters())      # ...does NOT reach the shared trunk
    assert any(p.grad is not None for p in actor.head.parameters())      # ...but trains the actor head


def test_shared_trunk_trains_clean() -> None:
    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("hsikan", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2, shared_trunk=True, hg_state=env.hg)
    cfg = td3_bc_config(total_steps=120, start_steps=20, critic_warmup=10, batch_size=32,
                        eval_every=120, n_eval=2, seed=0)
    hist = train_offpolicy(actor, critics, env, cfg)
    assert len(hist) >= 1 and all(np.isfinite(h) for h in hist)


def test_update_every_gates_the_gradient_updates() -> None:
    # The UTD knob: update_every > total_steps => the gradient block never fires => actor unchanged from init;
    # update_every=1 => it trains and the actor moves. Proves update_every decimates the (expensive) backward.
    import torch

    env, nv, feat = _cartpole()

    def _fresh() -> tuple[object, object]:
        return build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                               action_scale=env.force_mag, n_critics=2)

    a_none, c_none = _fresh()
    before = [p.detach().clone() for p in a_none.parameters()]                      # type: ignore[attr-defined]
    train_offpolicy(a_none, c_none, env,                                            # type: ignore[arg-type]
                    td3_config(total_steps=60, batch_size=16, eval_every=60, n_eval=1,
                               warm_start=True, update_every=1000))
    assert all(torch.allclose(b, p) for b, p in zip(before, a_none.parameters())), \
        "no update should fire when update_every > total_steps"                     # type: ignore[attr-defined]

    a_all, c_all = _fresh()
    before1 = [p.detach().clone() for p in a_all.parameters()]                      # type: ignore[attr-defined]
    train_offpolicy(a_all, c_all, env,                                              # type: ignore[arg-type]
                    td3_config(total_steps=60, batch_size=16, eval_every=60, n_eval=1,
                               warm_start=True, update_every=1))
    assert any(not torch.allclose(b, p) for b, p in zip(before1, a_all.parameters())), \
        "update_every=1 must actually train the actor"                             # type: ignore[attr-defined]


def test_update_every_rejects_below_one() -> None:
    import pytest

    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    with pytest.raises(ValueError):
        train_offpolicy(actor, critics, env, td3_config(total_steps=10, update_every=0))


def test_hetero_dual_critic_builds_and_trains() -> None:
    # The multi-timescale dual-critic: MLP (fast) + SA-HSiKAN (slow) — builds heterogeneous, trains on the async
    # cadence (1, 4), stays finite. Exercises the per-critic optimizer + per-cadence target-polyak path.
    import torch

    from hymeko_rl.train.ddpg import build_hetero_offpolicy

    env, nv, feat = _cartpole()
    actor, critics = build_hetero_offpolicy("sa_hsikan", ["mlp", "sa_hsikan"], obs_dim=feat, flat_dim=nv * feat,
                                            action_dim=1, action_scale=env.force_mag, hg_state=env.hg)
    assert len(critics) == 2
    cfg = td3_config(total_steps=160, start_steps=20, batch_size=32, eval_every=160, n_eval=2,
                     warm_start=True, critic_update_every=(1, 4), seed=0)
    hist = train_offpolicy(actor, critics, env, cfg)
    assert len(hist) >= 1 and all(np.isfinite(h) for h in hist)
    assert all(torch.isfinite(p).all() for p in actor.parameters())   # no divergence


def test_hetero_critic_update_every_length_checked() -> None:
    import pytest

    from hymeko_rl.train.ddpg import build_hetero_offpolicy

    env, nv, feat = _cartpole()
    actor, critics = build_hetero_offpolicy("mlp", ["mlp", "mlp"], obs_dim=feat, flat_dim=nv * feat,
                                            action_dim=1, action_scale=env.force_mag)
    with pytest.raises(ValueError):                                   # 3 cadences for 2 critics
        train_offpolicy(actor, critics, env, td3_config(total_steps=40, critic_update_every=(1, 2, 4)))


def test_critic_layernorm_present_and_trains() -> None:
    # The anti-overestimation LayerNorm: opt-in (default off = unchanged), present when requested, trains finite.
    import torch

    from hymeko_rl.train.ddpg import QCritic

    env, nv, feat = _cartpole()
    _a0, c0 = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1, action_scale=env.force_mag,
                              n_critics=2)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in c0[0].modules())          # default: no LN
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2, critic_layernorm=True)
    assert all(any(isinstance(m, torch.nn.LayerNorm) for m in c.modules()) for c in critics)  # LN in every critic
    assert isinstance(critics[0], QCritic)
    hist = train_offpolicy(actor, critics, env, td3_config(total_steps=80, batch_size=16, eval_every=80, n_eval=1,
                                                           warm_start=True, seed=0))
    assert len(hist) >= 1 and all(np.isfinite(h) for h in hist)


def test_critic_huber_wired_default_off_and_changes_optimization() -> None:
    # The Huber critic-loss option (framework fix for the spiky-terminal-reward critic overshoot,
    # report 2026-07-05-qterm-collapse-rootcause): default OFF (MSE, existing runs bit-unchanged), settable via
    # config, and — trained under identical seeds — it must produce a DIFFERENT (finite) actor than MSE, proving
    # the loss is actually swapped (not a no-op). smooth_l1 is 0.5·x² in the |δ|<1 region vs MSE's x², so the
    # critic gradient differs even on cartpole's non-spiky reward.
    import torch

    from hymeko_rl.train.ddpg import OffPolicyConfig

    assert OffPolicyConfig().critic_huber is False               # default = MSE (no behaviour change)
    assert td3_config(critic_huber=True).critic_huber is True     # settable

    def _run(huber: bool) -> list[torch.Tensor]:
        env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
        ss = env.observation_space.shape
        assert ss is not None
        nv, feat = int(ss[0]), int(ss[1])
        torch.manual_seed(0)
        np.random.seed(0)
        actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                         action_scale=env.force_mag, n_critics=2)
        cfg = td3_config(total_steps=120, start_steps=16, batch_size=16, eval_every=9_999, n_eval=1,
                         warm_start=True, update_every=1, seed=0, critic_huber=huber)
        train_offpolicy(actor, critics, env, cfg)
        return [p.detach().clone() for p in actor.parameters()]

    mse, hub = _run(False), _run(True)
    assert all(torch.isfinite(p).all() for p in mse + hub)                       # both paths stay finite
    assert any(not torch.allclose(m, h) for m, h in zip(mse, hub))               # Huber actually changes training


def test_vec_offpolicy_rejects_without_make_env() -> None:
    import pytest

    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    with pytest.raises(ValueError):
        train_offpolicy(actor, critics, env, td3_config(total_steps=40), n_envs=4)


def test_vec_offpolicy_trains_finite() -> None:
    # Vectorized OFF-POLICY (TD3) collection: N envs, batched action-selection, add_batch — trains finite, nets
    # return on CPU. The update path (compiled closures / LayerNorm / hetero) is shared with the single-env path.
    import torch

    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf

    mjcf = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mjcf)
    ss = env.observation_space.shape
    assert ss is not None
    nv, feat = int(ss[0]), int(ss[1])
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    before = [p.detach().clone() for p in actor.parameters()]
    cfg = td3_config(total_steps=160, start_steps=16, batch_size=16, eval_every=9_999, n_eval=1,
                     warm_start=True, update_every=2, seed=0)
    hist = train_offpolicy(actor, critics, env, cfg, n_envs=8,
                           make_env=lambda: InvertedPendulumEnv(mjcf=mjcf))
    assert isinstance(hist, list)
    assert all(torch.isfinite(p).all() for p in actor.parameters())                # no divergence
    assert any(not torch.allclose(b, p) for b, p in zip(before, actor.parameters()))  # it actually trained


def test_deterministic_actor_action_dim_matches_head() -> None:
    # The action_dim property (added 2026-07-03) is what train_offpolicy reads instead of head.out_features, so a
    # multi-head CTDE actor can satisfy the same contract. For the single-head actor it must equal head.out_features.
    env, nv, feat = _cartpole()
    actor, _ = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1, action_scale=env.force_mag,
                               n_critics=2)
    assert actor.action_dim == actor.head.out_features == 1


def test_cuda_compile_interleaved_graphs_no_overwrite() -> None:
    # Regression (2026-07-03): pure-TD3 on CUDA with compile=True runs the critic-loss and actor-loss as two
    # reduce-overhead CUDA graphs sharing one memory pool. Once the actor update fires (critic_warmup=0,
    # policy_delay reached), the actor graph — which re-invokes critic[0] — overwrote buffers the just-run
    # critic graph's autograd still held, raising "accessing tensor output of CUDAGraphs that has been
    # overwritten by a subsequent run" at a_loss.backward(). Fixed by cudagraph_mark_step_begin() per update.
    # Would crash against the prior implementation; passes now. CUDA-only (the bug is in the CUDA-graph path).
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required — the crash lives in the reduce-overhead CUDA-graph path")

    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    cfg = td3_config(total_steps=60, start_steps=8, batch_size=32, eval_every=9_999, n_eval=1,
                     warm_start=True, update_every=1, policy_delay=2, device="cuda", compile=True, seed=0)
    hist = train_offpolicy(actor, critics, env, cfg)   # must not raise the CUDA-graph overwrite error
    assert isinstance(hist, list)
    assert all(torch.isfinite(p).all() for p in actor.parameters())   # ran clean, no divergence


def test_polyak_foreach_matches_loop() -> None:
    # The _foreach_ fused polyak (launch-bound fix) must be numerically IDENTICAL to the per-tensor loop.
    import copy

    import torch

    from hymeko_rl.train.ddpg import _polyak

    torch.manual_seed(0)
    src = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    tgt = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    tgt_ref = copy.deepcopy(tgt)
    tau = 0.1
    with torch.no_grad():                                        # the old per-tensor loop = the reference
        for tp, sp in zip(tgt_ref.parameters(), src.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)
    _polyak(tgt, src, tau)                                       # the fused foreach path
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(tgt.parameters(), tgt_ref.parameters()))


# ── asymmetric-CTDE (MADDPG) privileged critic ──


def test_priv_critic_forward_two_and_three_arg() -> None:
    # Unit: priv_dim=0 critic is the plain Q(s,a) (2-arg); priv_dim>0 critic ingests z (3-arg), head wider by z.
    import torch

    from hymeko_rl.train.ddpg import QCritic

    class _BB(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1)[:, :6]

    c0 = QCritic(_BB(), 6, 4, priv_dim=0)
    cp = QCritic(_BB(), 6, 4, priv_dim=5)
    obs, a, z = torch.zeros(2, 3, 2), torch.zeros(2, 4), torch.zeros(2, 5)
    assert c0(obs, a).shape == (2,) and cp(obs, a, z).shape == (2,)
    assert c0.priv_dim == 0 and cp.priv_dim == 5
    lin0 = next(m for m in c0.head if isinstance(m, torch.nn.Linear))
    linp = next(m for m in cp.head if isinstance(m, torch.nn.Linear))
    assert linp.in_features - lin0.in_features == 5          # exactly priv_dim wider


def test_priv_critic_on_nonpriv_env_raises() -> None:
    # Fail-loud guard: a critic with priv_dim>0 on an env that cannot supply z (cart-pole has no
    # privileged_state) must raise a clear error, not crash deep in a cat shape-mismatch.
    import pytest

    env, nv, feat = _cartpole()
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=2)
    critics[0].priv_dim = 5                                  # simulate a mis-wired privileged critic
    cfg = td3_config(total_steps=40, start_steps=8, batch_size=16, eval_every=9_999, n_eval=1, seed=0)
    with pytest.raises(ValueError, match="privileged_state"):
        train_offpolicy(actor, critics, env, cfg)
