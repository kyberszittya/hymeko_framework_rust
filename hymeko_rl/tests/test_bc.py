"""Behaviour cloning: it fits the actor mean, and — never run blind (§3) — it emits per-epoch progress so the
BC phase is not a dark gap before off-policy training."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.agents.policy import build_policy
from hymeko_rl.train.bc import behaviour_clone


def _tiny() -> tuple:
    ac = build_policy("mlp", obs_dim=2, action_dim=2, hidden=16)
    obs = np.zeros((64, 2), dtype=np.float32)
    acts = np.zeros((64, 2), dtype=np.float32)
    return ac, obs, acts


def test_behaviour_clone_reduces_loss_and_returns_per_epoch() -> None:
    ac, obs, acts = _tiny()
    losses = behaviour_clone(ac, obs, acts, n_epochs=40, log_every=0)   # silent
    assert len(losses) == 40 and all(np.isfinite(losses))
    assert losses[-1] <= losses[0]                                       # fit improves toward the (zero) target


def test_behaviour_clone_logs_progress_every_log_every(capsys: "pytest.CaptureFixture[str]") -> None:
    """Regression (§3): the BC phase emits a flushed [bc] line every log_every epochs (and log_every=0 silences)."""
    ac, obs, acts = _tiny()
    behaviour_clone(ac, obs, acts, n_epochs=50, log_every=25)
    out = capsys.readouterr().out
    assert "[bc] epoch" in out, "BC ran without a progress line (dark gap — violates never-run-blind)"
    assert "loss=" in out and "ETA" in out                              # progress + the loss tripwire
    assert out.count("[bc] epoch") >= 2                                  # ~every 25 of 50 epochs

    ac2, obs2, acts2 = _tiny()
    behaviour_clone(ac2, obs2, acts2, n_epochs=30, log_every=0)
    assert "[bc]" not in capsys.readouterr().out                        # 0 silences it


def test_behaviour_clone_device_path_returns_cpu_model() -> None:
    """Regression (2026-07-05 GPU lever): ``device="auto"`` trains (cuda when available, else cpu) and the
    model comes back ON THE CPU, so downstream eval/state_dict consumers are unaffected."""
    import torch

    ac, obs, acts = _tiny()
    losses = behaviour_clone(ac, obs, acts, n_epochs=10, log_every=0, device="auto")
    assert len(losses) == 10 and all(np.isfinite(losses)) and losses[-1] <= losses[0]
    assert all(p.device.type == "cpu" for p in ac.parameters()), "model must return to CPU after BC"
    # greedy eval path (CPU tensors) must work immediately after a device-trained clone
    out = ac.action_mean(torch.zeros(1, 2))
    assert out.shape == (1, 2) and bool(torch.isfinite(out).all())
