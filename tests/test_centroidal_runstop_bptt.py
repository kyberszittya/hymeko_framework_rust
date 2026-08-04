"""Gradient-based RL (BPTT via differentiable simulation) for run-stop — beats tuned-linear (and CEM).

Backprop-through-time on a torch copy of the run-stop dynamics, with a surrogate aligned to the true objective
(a *tail* speed penalty, not final-step-only). Its 1-hidden-layer net converts to the numpy policy vector, so the
SAME held-out ``evaluate`` scores it against the tuned-linear baseline. The conversion parity is pinned so the
comparison is honest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scenarios.humanoid.centroidal_runstop import (
    PolicyConfig,
    RunStopConfig,
    episode,
    evaluate,
    mixed_set,
    policy_actions,
)
from scenarios.humanoid.centroidal_runstop_bptt import BpttPolicy, train_bptt


@pytest.fixture(scope="module")
def bptt_params() -> tuple:
    cfg = RunStopConfig()
    return train_bptt(cfg, iters=300).to_numpy_params(), cfg, PolicyConfig(hidden=24)


def test_param_conversion_parity() -> None:
    """The torch policy and its numpy-flattened params produce identical actions (fair, shared-metric comparison)."""
    cfg, pc = RunStopConfig(), PolicyConfig(hidden=24)
    torch.manual_seed(3)
    pol = BpttPolicy(cfg)
    feats = np.random.RandomState(4).standard_normal((30, 5)).astype(np.float32)
    with torch.no_grad():
        tfx, ta = pol(torch.as_tensor(feats))
    nfx, na = policy_actions(pol.to_numpy_params(), feats, pc, cfg)
    assert np.allclose(tfx.numpy(), nfx, atol=1e-5) and np.allclose(ta.numpy(), na, atol=1e-5)


def test_bptt_beats_tuned_linear_on_held_out(bptt_params) -> None:
    """The gradient policy stops far more reliably than the best single linear gain, held-out, without falling."""
    params, cfg, pc = bptt_params
    report = evaluate(params, cfg, pc, offset=0.5)
    states = mixed_set(cfg, offset=0.5)
    best_lin = max(episode(None, states, cfg, pc, gains=(kp, kd))[0].mean()
                   for kp in (2, 4, 6, 8, 10, 12) for kd in (1, 2, 3, 4, 5))
    assert report["stop_success"] > best_lin + 0.1
    assert report["fall_rate"] <= 0.05


def test_bptt_training_is_deterministic() -> None:
    cfg = RunStopConfig()
    a = train_bptt(cfg, iters=20).to_numpy_params()
    b = train_bptt(cfg, iters=20).to_numpy_params()
    assert np.allclose(a, b)
