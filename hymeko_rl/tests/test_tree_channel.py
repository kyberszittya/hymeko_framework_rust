"""TreeChannel — the dynamic cross-channel between parallel actor/critic graphs on the fixed hypergraph."""
import torch

from hymeko_rl.agents.tree_channel import TreeChannel, adjacency_from_hg


def test_forward_shapes() -> None:
    ch = TreeChannel(16, torch.ones(4, 4, dtype=torch.bool))
    oa, oc = ch(torch.randn(3, 4, 16), torch.randn(3, 4, 16))
    assert oa.shape == (3, 4, 16) and oc.shape == (3, 4, 16)


def test_channel_respects_the_fixed_mask() -> None:
    # node 0 isolated (self-loop only); nodes 1,2 coupled — routing must obey the fixed candidate edges.
    adj = torch.tensor([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=torch.bool)
    ch = TreeChannel(8, adj)
    torch.manual_seed(0)
    ha, hc = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
    out1, _ = ch(ha, hc)
    hc2 = hc.clone()
    hc2[0, 1] += 5.0                                            # perturb node 1's critic feature
    out2, _ = ch(ha, hc2)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)    # node 0 (not coupled to 1) is unaffected
    assert not torch.allclose(out1[0, 1], out2[0, 1], atol=1e-5)  # node 1 IS affected


def test_channel_is_state_adaptive() -> None:
    ch = TreeChannel(8, torch.ones(3, 3, dtype=torch.bool))
    torch.manual_seed(0)
    o1, _ = ch(torch.randn(1, 3, 8), torch.randn(1, 3, 8))
    o2, _ = ch(torch.randn(1, 3, 8), torch.randn(1, 3, 8))
    assert not torch.allclose(o1, o2)                          # different states -> different routing


def test_channel_gradients_flow_both_graphs() -> None:
    ch = TreeChannel(8, torch.ones(3, 3, dtype=torch.bool))
    ha = torch.randn(2, 3, 8, requires_grad=True)
    hc = torch.randn(2, 3, 8, requires_grad=True)
    oa, oc = ch(ha, hc)
    (oa.sum() + oc.sum()).backward()
    assert ha.grad is not None and hc.grad is not None
    assert any(p.grad is not None for p in ch.parameters())


def test_rejects_bad_adj() -> None:
    for bad in (torch.ones(3, 4, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
        try:
            TreeChannel(8, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for adj shape {tuple(bad.shape)}")


def test_adjacency_from_hg_is_symmetric_with_self_loops() -> None:
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
    env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    adj = adjacency_from_hg(env.hg)
    n = int(env.hg.n_vertices)
    assert adj.shape == (n, n) and adj.dtype == torch.bool
    assert bool(adj.diag().all())                             # self-loops (no all-masked softmax row)
    assert bool((adj == adj.T).all())                         # coupling is symmetric
    # the channel builds + runs on the real robot adjacency
    oa, _ = TreeChannel(8, adj)(torch.randn(1, n, 8), torch.randn(1, n, 8))
    assert oa.shape == (1, n, 8) and bool(torch.isfinite(oa).all())


def test_multi_channel_forward_and_grads() -> None:
    from hymeko_rl.agents.tree_channel import MultiTreeChannel
    ch = MultiTreeChannel(8, torch.ones(4, 4, dtype=torch.bool), n_agents=3)
    feats = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]
    outs = ch(feats)
    assert len(outs) == 3 and all(o.shape == (2, 4, 8) for o in outs)
    sum(o.sum() for o in outs).backward()   # type: ignore[union-attr]
    assert all(f.grad is not None for f in feats)                 # every agent graph gets gradient
    assert any(p.grad is not None for p in ch.parameters())


def test_multi_channel_state_adaptive() -> None:
    from hymeko_rl.agents.tree_channel import MultiTreeChannel
    ch = MultiTreeChannel(8, torch.ones(3, 3, dtype=torch.bool), n_agents=3)
    torch.manual_seed(0)
    o1 = ch([torch.randn(1, 3, 8) for _ in range(3)])
    o2 = ch([torch.randn(1, 3, 8) for _ in range(3)])
    assert not torch.allclose(o1[0], o2[0])


def test_multi_channel_rejects_bad() -> None:
    from hymeko_rl.agents.tree_channel import MultiTreeChannel
    try:
        MultiTreeChannel(8, torch.ones(3, 3, dtype=torch.bool), n_agents=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n_agents=1")
    ch = MultiTreeChannel(8, torch.ones(3, 3, dtype=torch.bool), n_agents=3)
    try:
        ch([torch.randn(1, 3, 8)])                                # wrong number of agents
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong agent count")


def test_multi_channel_3agent_on_galambos_hypergraph() -> None:
    # 2 actors + 1 critic on the galambos joint hypergraph — the coin-toss collaboration.
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    from hymeko_rl.agents.tree_channel import MultiTreeChannel, adjacency_from_hg
    env = PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3, task_graph=True)
    adj = adjacency_from_hg(env.hg)
    n = int(env.hg.n_vertices)
    outs = MultiTreeChannel(8, adj, n_agents=3)([torch.randn(1, n, 8) for _ in range(3)])
    assert len(outs) == 3 and all(bool(torch.isfinite(o).all()) for o in outs)
