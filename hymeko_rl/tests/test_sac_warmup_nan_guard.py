"""§4 regression: distinguish the benign warm-up logging condition from a real non-finite update.

Before the first gradient update (step <= start_steps) the trainer must log ``crit=N/A`` (not ``nan``); once updates
begin the critic loss must be finite, and a genuinely non-finite optimized loss must ABORT rather than silently corrupt
the policy. The previous ``crit=nan`` print was logging-only (last_c initialised to nan before any update ran)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac


def _build(*, total_steps: int, start_steps: int, log_every: int = 0):
    env = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    ss = env.observation_space.shape
    assert ss is not None
    torch.manual_seed(0)
    np.random.seed(0)
    actor, critics = build_sac("mlp", obs_dim=ss[0] * ss[1], flat_dim=ss[0] * ss[1], action_dim=1,
                               action_scale=env.force_mag, n_critics=2, hidden=32, device="cpu")
    cfg = SACConfig(total_steps=total_steps, start_steps=start_steps, batch_size=32, eval_every=total_steps,
                    n_eval=3, log_every=log_every, seed=0)
    return env, actor, critics, cfg


def test_warmup_logs_na_not_nan(capsys: pytest.CaptureFixture[str]) -> None:
    """Log lines before the first update show crit=N/A; lines after show a finite number — never a bare nan."""
    env, actor, critics, cfg = _build(total_steps=400, start_steps=200, log_every=50)
    train_sac(actor, critics, env, cfg)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "[sac] step" in ln]
    pre = [ln for ln in lines if int(ln.split("step")[1].split("/")[0]) <= 200]
    post = [ln for ln in lines if int(ln.split("step")[1].split("/")[0]) > 200]
    assert pre and all("crit=N/A" in ln for ln in pre)             # warm-up: N/A, never nan
    assert not any("crit=nan" in ln for ln in lines)               # nan is never printed
    assert post and all("crit=N/A" not in ln for ln in post)       # post-warmup: a real number


def test_healthy_run_stays_finite_and_does_not_abort() -> None:
    env, actor, critics, cfg = _build(total_steps=500, start_steps=100)
    hist = train_sac(actor, critics, env, cfg)                     # must not raise
    assert all(np.isfinite(hist))
    assert all(torch.isfinite(p).all() for p in actor.parameters())


def test_nonfinite_update_aborts() -> None:
    """A genuinely non-finite critic loss (poisoned critic) aborts with a clear error, not silent corruption."""
    env, actor, critics, cfg = _build(total_steps=400, start_steps=50)
    with torch.no_grad():
        next(iter(critics[0].parameters())).mul_(float("nan"))     # poison → the first real update is non-finite
    with pytest.raises(RuntimeError, match="non-finite"):
        train_sac(actor, critics, env, cfg)
