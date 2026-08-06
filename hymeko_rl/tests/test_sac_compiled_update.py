"""Tests for the compiled + rate-decoupled SAC update (2026-07-17).

The 2026-07-17 profile showed the structural SAC per-step cost is 99.9% the B=256 gradient update. `train_sac`
gained two update-cost knobs — `compile` (torch.compile CUDA-graphs; CUDA-only, pure speedup) and `update_every`
(fewer gradient steps; a §6.5 #19 sample-efficiency change). These tests pin the invariants a CPU can check:
the refactor changed no math (determinism + compile-CPU-noop parity), and update_every scales the update count.
The GPU speedup itself is a kato15 smoke, not a unit test."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac


def _run(*, update_every: int = 1, compile: bool = False, total_steps: int = 1500) -> tuple[list[float], float]:
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    ss = env.observation_space.shape
    assert ss is not None
    torch.manual_seed(0)
    np.random.seed(0)
    actor, critics = build_sac("mlp", obs_dim=ss[0] * ss[1], flat_dim=ss[0] * ss[1], action_dim=1,
                               action_scale=env.force_mag, n_critics=2, hidden=32, device="cpu")
    cfg = SACConfig(total_steps=total_steps, start_steps=200, batch_size=32, eval_every=total_steps,
                    n_eval=3, log_every=0, seed=0, update_every=update_every, compile=compile)
    history = train_sac(actor, critics, env, cfg)
    return history, float(sum(float(p.detach().sum()) for p in actor.parameters()))


def test_refactor_is_deterministic_and_finite() -> None:
    """The extracted `_update_once` path is RNG-stable: same seed → identical params, finite eval curve."""
    (h1, p1), (h2, p2) = _run(), _run()
    assert p1 == pytest.approx(p2, abs=1e-9)          # bit-identical params (no RNG-order drift from the refactor)
    assert all(np.isfinite(h1)) and h1 == pytest.approx(h2)


def test_compile_flag_is_cpu_noop_parity() -> None:
    """`compile=True` is guarded to CUDA — on CPU it must fall back to eager and reproduce `compile=False` exactly."""
    _, p_eager = _run(compile=False)
    _, p_compiled = _run(compile=True)
    assert p_eager == pytest.approx(p_compiled, abs=1e-9)


def test_update_every_scales_update_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """`update_every=N` does 1/N the gradient updates. Count `_polyak` calls (once per critic per update)."""
    import hymeko_rl.train.sac as sac_mod

    calls = {"n": 0}
    real_polyak = sac_mod._polyak

    def counting_polyak(*a: object, **k: object) -> None:
        calls["n"] += 1
        return real_polyak(*a, **k)   # type: ignore[arg-type]

    monkeypatch.setattr(sac_mod, "_polyak", counting_polyak)
    calls["n"] = 0
    _run(update_every=1)
    n1 = calls["n"]
    calls["n"] = 0
    _run(update_every=2)
    n2 = calls["n"]
    assert n1 > 0
    # 1300 qualifying steps → 1300 vs 650 updates (×2 critics). Allow ±1 update of slack on the ratio.
    assert abs(n2 - n1 / 2) <= 2, f"update_every=2 did {n2} polyaks, expected ≈{n1 / 2} (half of {n1})"


def test_update_every_validation() -> None:
    """update_every < 1 is a caller bug — fail loud, not with a silent no-update loop."""
    with pytest.raises(ValueError, match="update_every"):
        _run(update_every=0, total_steps=300)
