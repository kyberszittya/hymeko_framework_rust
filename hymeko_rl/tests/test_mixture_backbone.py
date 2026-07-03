"""The HSiKAN + MLP mixture-of-experts backbone: a per-state gate blends a structural (HSiKAN) and a
structure-blind (MLP) expert in one policy."""
import torch

from hymeko_rl.agents.policy import POLICY_KINDS, MixtureBackbone, build_policy
from hymeko_rl.experiments.structural_probe import build_chain_graph


def test_mixture_registered() -> None:
    assert "mixture" in POLICY_KINDS


def test_mixture_forward_shape_and_gate_range() -> None:
    hg = build_chain_graph(6, seed=0)
    ac = build_policy("mixture", obs_dim=3, action_dim=2, hg_state=hg, hidden=16)
    x = torch.randn(4, 6, 3)
    assert ac.action_mean(x).shape == (4, 2)
    bk = next(m for m in ac.modules() if isinstance(m, MixtureBackbone))
    g = bk.gate_value(x)
    assert g.shape == (4, 1)
    assert bool((g > 0).all()) and bool((g < 1).all())         # a genuine soft mix, never degenerate


def test_mixture_has_both_experts_and_gate() -> None:
    bk = MixtureBackbone(3, 16, build_chain_graph(6, seed=0))
    names = {n for n, _ in bk.named_parameters()}
    assert any("hsikan" in n for n in names)
    assert any("mlp" in n for n in names)
    assert any("gate" in n for n in names)


def test_gate_selects_the_expert() -> None:
    # forcing g->1 must reproduce the HSiKAN expert; g->0 the MLP expert (the blend is exactly the gate).
    bk = MixtureBackbone(3, 16, build_chain_graph(6, seed=1))
    x = torch.randn(4, 6, 3)
    with torch.no_grad():
        h_hsi, h_mlp = bk.hsikan(x), bk.mlp(x)
        assert not torch.allclose(h_hsi, h_mlp)                # the two experts genuinely differ
        bk.gate.weight.zero_()
        bk.gate.bias.fill_(12.0)                               # g≈1 -> all HSiKAN
        assert torch.allclose(bk(x), h_hsi, atol=1e-2)
        bk.gate.bias.fill_(-12.0)                              # g≈0 -> all MLP
        assert torch.allclose(bk(x), h_mlp, atol=1e-2)
