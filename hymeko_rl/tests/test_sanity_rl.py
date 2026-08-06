"""The fast RL sanity testbed (contextual bandit): correct reward, and every backbone trains in the loop."""
import math

import torch

from hymeko_rl.experiments.sanity_rl import BanditConfig, ContextualBandit, run_bandit_sanity


def test_optimal_action_is_zero_reward() -> None:
    b = ContextualBandit(BanditConfig(target="structural"))
    ctx = b.sample(8)
    assert torch.allclose(b.reward(ctx, b.optimal(ctx)), torch.zeros(8), atol=1e-6)   # optimal => reward 0


def test_shapes() -> None:
    cfg = BanditConfig()
    b = ContextualBandit(cfg)
    ctx = b.sample(5)
    assert ctx.shape == (5, cfg.n_vertices, cfg.feat)
    assert b.optimal(ctx).shape == (5, cfg.action_dim)
    assert b.reward(ctx, b.optimal(ctx)).shape == (5,)
    assert (b.reward(ctx, torch.zeros(5, cfg.action_dim)) <= 0).all()                 # any action <= optimal


def test_sanity_runs_every_backbone_in_the_rl_loop() -> None:
    # the point of the testbed: all backbones train through REINFORCE, fast, and return finite results.
    res = run_bandit_sanity(("mlp", "hsikan"), steps=120, cfg=BanditConfig(target="flat"))
    for kind, m in res.items():
        assert math.isfinite(m["reward"]) and m["params"] > 0
        assert m["train_s"] >= 0.0 and m["deploy_ms"] >= 0.0                          # both perf metrics present
        assert m["reward"] > -1.0                                                     # learned, not diverged


def test_variant_triples_and_enhancements() -> None:
    # the (label, kind, kwargs) form drives the cr-vs-cheby + SA-HSiKAN enhancement comparison.
    res = run_bandit_sanity((("hsikan-cheby", "hsikan", {"activation": "cr_cheby"}),
                             ("sa_hsikan", "sa_hsikan", {})), steps=40, cfg=BanditConfig(target="structural"))
    assert set(res) == {"hsikan-cheby", "sa_hsikan"}
