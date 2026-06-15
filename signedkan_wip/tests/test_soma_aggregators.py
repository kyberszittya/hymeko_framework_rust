"""Unit tests for the upgraded RicciStim backbone aggregators.

Run: pytest -p no:randomly signedkan_wip/tests/test_soma_aggregators.py
"""
from __future__ import annotations

import torch

from signedkan_wip.src.hymeko_gomb.soma.vision.aggregators import (
    CrossScalePyramid,
    HighwayGate,
    LearnedBranchMixer,
)


def test_mixer_weights_simplex_and_uniform_init() -> None:
    torch.manual_seed(0)
    mx = LearnedBranchMixer(3)
    w = mx.weights()
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    assert bool((w >= 0).all())
    # alpha = 0 (init) -> uniform mean, not a bare sum
    b = [torch.randn(5, 4) for _ in range(3)]
    out = mx(b)
    assert out.shape == (5, 4)
    assert torch.allclose(out, (b[0] + b[1] + b[2]) / 3, atol=1e-5)


def test_mixer_rejects_wrong_branch_count() -> None:
    mx = LearnedBranchMixer(3)
    try:
        mx([torch.randn(2, 4), torch.randn(2, 4)])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_highway_gate_can_carry_the_skip() -> None:
    hw = HighwayGate(4)
    with torch.no_grad():
        hw.gate.weight.zero_()
        hw.gate.bias.fill_(-100.0)   # T -> 0  =>  y -> skip
    m, s = torch.randn(3, 4), torch.randn(3, 4)
    out = hw(m, s)
    assert out.shape == (3, 4)
    assert torch.allclose(out, s, atol=1e-3)


def test_highway_gate_is_differentiable() -> None:
    hw = HighwayGate(4)
    m = torch.randn(3, 4, requires_grad=True)
    s = torch.randn(3, 4)
    hw(m, s).sum().backward()
    assert m.grad is not None


def test_pyramid_fuses_parent_into_children() -> None:
    py = CrossScalePyramid(4)
    feat = torch.randn(3, 4)
    parent = torch.tensor([-1, 0, 0])
    scales = torch.tensor([0, 1, 1])
    out = py(feat, parent, scales)
    assert out.shape == (3, 4)
    assert torch.allclose(out[0], feat[0])                       # root unchanged
    assert torch.allclose(out[1], feat[1] + py.up(feat[0]), atol=1e-5)
    assert torch.allclose(out[2], feat[2] + py.up(feat[0]), atol=1e-5)


def test_pyramid_topdown_two_levels() -> None:
    # 0(scale0) -> 1(scale1) -> 2(scale2): the grandchild must see the *updated*
    # parent (which itself absorbed the root), i.e. fusion is a true cascade.
    py = CrossScalePyramid(4)
    feat = torch.randn(3, 4)
    parent = torch.tensor([-1, 0, 1])
    scales = torch.tensor([0, 1, 2])
    out = py(feat, parent, scales)
    f1 = feat[1] + py.up(feat[0])
    assert torch.allclose(out[1], f1, atol=1e-5)
    assert torch.allclose(out[2], feat[2] + py.up(f1), atol=1e-5)


def test_pyramid_noop_when_no_parents() -> None:
    py = CrossScalePyramid(4)
    feat = torch.randn(4, 4)
    out = py(feat, torch.tensor([-1, -1, -1, -1]), torch.tensor([0, 0, 0, 0]))
    assert torch.allclose(out, feat)


def test_pyramid_is_differentiable() -> None:
    py = CrossScalePyramid(4)
    feat = torch.randn(3, 4, requires_grad=True)
    py(feat, torch.tensor([-1, 0, 0]), torch.tensor([0, 1, 1])).sum().backward()
    assert feat.grad is not None


def test_backbone_upgraded_forward_and_backward() -> None:
    # Integration smoke: the full backbone with all three upgrades enabled runs
    # end-to-end on a tiny image and is differentiable.
    from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_backbone import (
        RicciStimBackbone,
    )
    torch.manual_seed(0)
    bb = RicciStimBackbone(
        image_h=8, image_w=8, patch_size_initial=4, d_hidden=8, max_depth=1,
        use_arity_mixer=True, use_highway=True, use_pyramid=True,
    )
    img = torch.rand(1, 8, 8)
    feats, tree = bb(img)
    assert feats.shape == (tree.n_anchors, 8)
    feats.sum().backward()
    # at least one upgrade module received gradient
    assert any(p.grad is not None for p in bb.branch_mixer.parameters())
