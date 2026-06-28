"""SA-HSiKAN backbone — the B^L holonomy-collapse of HSiKAN as a SAC backbone (the launch-bound fix)."""
import torch

from hymeko_rl.policy import POLICY_KINDS, StructuralActorBackbone, sa_hsikan_backbone


class _StubHG:
    """Minimal hg_state: a 3-vertex signed graph (1->2 +, 2->3 +, 3->1 -)."""
    n_vertices = 3

    def dense_signed_adj(self, device: str = "cpu"):
        a_pos = torch.tensor([[0.0, 1, 0], [0, 0, 1], [0, 0, 0]])
        a_neg = torch.tensor([[0.0, 0, 0], [0, 0, 0], [1, 0, 0]])
        return a_pos, a_neg


def test_sa_hsikan_registered() -> None:
    assert "sa_hsikan" in POLICY_KINDS


def test_sa_hsikan_backbone_shapes_and_holonomy() -> None:
    hg = _StubHG()
    mod, feat = sa_hsikan_backbone(2, hg_state=hg, hidden=4, walk_len=2)
    assert feat == 4 and isinstance(mod, StructuralActorBackbone)
    # B^L is the precomputed signed L-hop holonomy operator = matrix_power(A+ - A-, walk_len).
    a_pos, a_neg = hg.dense_signed_adj()
    assert torch.allclose(mod.bl, torch.matrix_power(a_pos - a_neg, 2))
    x = torch.randn(5, 3, 2)                                  # (B, N, in_feat)
    assert mod.node_activations(x).shape == (5, 3, 4)         # per-vertex (the per_node head reads this)
    assert mod(x).shape == (5, 4)                             # pooled (the pooled head reads this)
    # the factory ignores HSiKAN-only kwargs (skip/incidence) without error.
    mod2, _ = sa_hsikan_backbone(2, hg_state=hg, hidden=4, skip="highway", incidence="learned")
    assert mod2(x).shape == (5, 4)


def test_sa_hsikan_as_sac_actor() -> None:
    from hymeko_rl.sac import build_sac
    actor, critics = build_sac("sa_hsikan", obs_dim=2, flat_dim=6, action_dim=2, action_scale=1.0,
                               hidden=4, hg_state=_StubHG())
    x = torch.randn(5, 3, 2)
    assert actor.action_mean(x).shape == (5, 2)              # deterministic greedy
    action, log_prob = actor.sample(x)                       # reparameterised
    assert action.shape == (5, 2) and log_prob.shape == (5,)
    assert len(critics) == 2
