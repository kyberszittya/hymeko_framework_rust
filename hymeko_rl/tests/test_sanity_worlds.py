"""The multi-step (grid/hex nav) + multi-agent (collab) RL sanity worlds: correct dynamics, and the backbones
train through the sequential / cooperative loop."""
import torch

from hymeko_rl.sanity_worlds import (
    CollabBandit,
    LatticeNav,
    WorldConfig,
    run_collab_sanity,
    run_world_sanity,
)


def test_lattice_shapes_grid_and_hex() -> None:
    for lat in ("grid", "hex"):
        w = LatticeNav(WorldConfig(lattice=lat, size=3))
        assert w.n == 9
        assert w.reset(5).shape == (5, 9, 3)
        obs, r, done = w.step(torch.zeros(5, 2))
        assert obs.shape == (5, 9, 3) and r.shape == (5,)


def test_lattice_reward_zero_at_goal() -> None:
    w = LatticeNav(WorldConfig(size=3))
    w.reset(4)
    w._agent = w.pos[w._goal]                                   # place the agent on its goal
    _, r, _ = w.step(torch.zeros(4, 2))                         # zero action -> stays put
    assert torch.allclose(r, torch.zeros(4), atol=1e-5)


def test_hex_has_more_edges_than_grid() -> None:
    g = LatticeNav(WorldConfig(lattice="grid", size=3))
    h = LatticeNav(WorldConfig(lattice="hex", size=3))
    assert int(h.hg.edges.shape[0]) > int(g.hg.edges.shape[0])  # 6-neighbour > 4-neighbour


def test_collab_reward_zero_when_coordinated() -> None:
    b = CollabBandit(WorldConfig())
    ctx = b.sample(6)
    t = b.target(ctx)
    assert torch.allclose(b.reward(ctx, t, torch.zeros_like(t)), torch.zeros(6), atol=1e-6)   # a_a+a_b == target


def test_runners_execute() -> None:
    r = run_world_sanity(("mlp",), cfg=WorldConfig(size=3), iters=8)
    assert "mlp" in r and r["mlp"]["params"] > 0
    c = run_collab_sanity("mlp", iters=40)
    assert c["reward"] <= 0.0 and int(c["params"]) > 0          # type: ignore[call-overload]
