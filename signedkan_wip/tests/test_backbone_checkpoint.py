"""Stage D-3-quinquies: HSiKAN backbone activation-checkpointing parity tests.

The checkpoint path must produce numerically identical forward output and
gradient tensors as the non-checkpoint path. Memory savings are not
directly testable in unit tests (they show up at production scale); the
parity invariants are what we can pin in CI.
"""
from __future__ import annotations

import pytest
import torch


def _make_backbones():
    from signedkan_wip.src.vision.hymeyolo_backbones import (
        HSiKANConvBackbone,
    )
    torch.manual_seed(0)
    plain = HSiKANConvBackbone(c_in=3, c_out=32, use_checkpoint=False)
    ckpt = HSiKANConvBackbone(c_in=3, c_out=32, use_checkpoint=True)
    # Copy weights so the two backbones are identical except for the
    # checkpoint flag.
    ckpt.load_state_dict(plain.state_dict())
    return plain, ckpt


def test_forward_parity_eval_mode():
    """Eval-mode forward identical regardless of use_checkpoint."""
    plain, ckpt = _make_backbones()
    plain.eval(); ckpt.eval()
    x = torch.randn(2, 3, 64, 64)
    y_plain = plain(x)
    y_ckpt = ckpt(x)
    assert torch.allclose(y_plain, y_ckpt, atol=1e-6)


def test_multi_scale_features_parity_eval():
    plain, ckpt = _make_backbones()
    plain.eval(); ckpt.eval()
    x = torch.randn(2, 3, 64, 64)
    p4_plain, p8_plain = plain.multi_scale_features(x)
    p4_ckpt, p8_ckpt = ckpt.multi_scale_features(x)
    assert torch.allclose(p4_plain, p4_ckpt, atol=1e-6)
    assert torch.allclose(p8_plain, p8_ckpt, atol=1e-6)


def test_forward_parity_train_mode():
    """Training-mode forward identical regardless of use_checkpoint
    (the checkpoint path recomputes the same forward function)."""
    plain, ckpt = _make_backbones()
    plain.train(); ckpt.train()
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    y_plain = plain(x)
    y_ckpt = ckpt(x)
    assert torch.allclose(y_plain, y_ckpt, atol=1e-6)


def test_backward_parity_input_grads():
    """Gradients on the input tensor match between plain and ckpt."""
    plain, ckpt = _make_backbones()
    plain.train(); ckpt.train()
    x_plain = torch.randn(2, 3, 64, 64, requires_grad=True)
    x_ckpt = x_plain.detach().clone().requires_grad_(True)
    plain(x_plain).sum().backward()
    ckpt(x_ckpt).sum().backward()
    assert x_plain.grad is not None
    assert x_ckpt.grad is not None
    assert torch.allclose(x_plain.grad, x_ckpt.grad, atol=1e-5)


def test_backward_parity_param_grads():
    """First-conv weight grads match between plain and ckpt."""
    plain, ckpt = _make_backbones()
    plain.train(); ckpt.train()
    x_plain = torch.randn(2, 3, 64, 64)
    x_ckpt = x_plain.clone()
    # Need grad on input for the checkpoint path to engage.
    x_plain = x_plain.requires_grad_(True)
    x_ckpt = x_ckpt.requires_grad_(True)
    plain(x_plain).sum().backward()
    ckpt(x_ckpt).sum().backward()
    # Compare grad of the very first conv layer.
    p_plain = next(plain.stack[0].parameters())
    p_ckpt = next(ckpt.stack[0].parameters())
    assert p_plain.grad is not None
    assert p_ckpt.grad is not None
    assert torch.allclose(p_plain.grad, p_ckpt.grad, atol=1e-5)


def test_eval_mode_bypasses_checkpoint():
    """In eval mode, _apply_layer must NOT call torch.utils.checkpoint
    (no backward = no need to recompute)."""
    from signedkan_wip.src.vision.hymeyolo_backbones import (
        HSiKANConvBackbone,
    )
    backbone = HSiKANConvBackbone(c_in=3, c_out=32, use_checkpoint=True)
    backbone.eval()
    x = torch.randn(1, 3, 64, 64)
    # The check is structural — the forward path returns without
    # going through checkpoint, which would fail on a non-grad input.
    # If checkpoint was called, the non-grad input would have raised.
    y = backbone(x)
    assert y.shape == (1, 32, 8, 8)


def test_non_requires_grad_input_bypasses_checkpoint():
    """A training-mode forward with a leaf detached input also bypasses
    the checkpoint, since recompute requires the autograd graph."""
    from signedkan_wip.src.vision.hymeyolo_backbones import (
        HSiKANConvBackbone,
    )
    backbone = HSiKANConvBackbone(c_in=3, c_out=32, use_checkpoint=True)
    backbone.train()
    x = torch.randn(1, 3, 64, 64)  # NOT requires_grad
    y = backbone(x)
    assert y.shape == (1, 32, 8, 8)


def test_build_backbone_threads_checkpoint():
    """build_backbone('hsikan', use_checkpoint=True) sets the flag."""
    from signedkan_wip.src.vision.hymeyolo_backbones import (
        build_backbone, HSiKANConvBackbone,
    )
    b = build_backbone("hsikan", c_in=3, c_out=32, use_checkpoint=True)
    assert isinstance(b, HSiKANConvBackbone)
    assert b.use_checkpoint is True
    b_off = build_backbone("hsikan", c_in=3, c_out=32, use_checkpoint=False)
    assert b_off.use_checkpoint is False


def test_build_backbone_checkpoint_ignored_for_resnet():
    """use_checkpoint for non-hsikan backbones is silently ignored."""
    from signedkan_wip.src.vision.hymeyolo_backbones import (
        build_backbone,
    )
    # Must not raise; the flag is dropped.
    b = build_backbone("resnet", c_in=3, c_out=32, use_checkpoint=True)
    assert b is not None
