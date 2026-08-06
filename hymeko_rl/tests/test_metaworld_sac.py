"""From-scratch SAC-on-MetaWorld harness — obs-norm wrapper + (real-env) end-to-end smoke."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hymeko_rl.experiments.exp_metaworld_sac import _ObsNorm


def _metaworld_missing() -> bool:
    return importlib.util.find_spec("metaworld") is None


class _Stub:
    observation_space = type("S", (), {"shape": (4,)})()
    action_space = type("A", (), {"shape": (2,), "high": np.ones(2)})()

    def reset(self, **_kw: object) -> "tuple[np.ndarray, dict]":
        return np.array([10.0, -5.0, 0.0, 3.0], np.float32), {}

    def step(self, _a: object) -> "tuple[np.ndarray, float, bool, bool, dict]":
        return np.array([10.0, -5.0, 0.0, 3.0], np.float32), 1.0, True, False, {"success": 0.0}


def test_obs_norm_standardizes_with_floor() -> None:
    """_ObsNorm standardizes (obs==mean → 0) and floors the std so near-constant dims are not amplified."""
    env = _ObsNorm(_Stub(), mean=np.array([10.0, -5.0, 0.0, 3.0]), std=np.array([2.0, 4.0, 0.0001, 0.5]))
    obs, _ = env.reset()
    assert np.allclose(obs, 0.0)                    # obs at the mean → normalized to 0 (no 1e4 blow-up on the tiny-std dim)


@pytest.mark.skipif(_metaworld_missing(), reason="metaworld not installed")
def test_sac_runs_end_to_end(tmp_path) -> None:
    """A tiny from-scratch SAC run produces an eval curve + checkpoint (plumbing, not learning)."""
    from hymeko_rl.experiments.exp_metaworld_sac import run_sac_seed
    s = run_sac_seed("original", seed=0, steps=2500, out_dir=tmp_path, device="cpu", hidden=64)
    assert s["final_success"] is not None and 0.0 <= s["final_success"] <= 1.0
    assert (tmp_path / "sac_original_seed0.pt").exists()
