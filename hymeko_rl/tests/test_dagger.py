"""Plumbing + label-sanity tests for the lock-step DAgger loop (`hymeko_rl.train.dagger`).

Requires the built `hymeko` CLI + MuJoCo (run on kato15 with `MUJOCO_GL=egl`). Tiny scale: verifies the
label-sanity gate passes (lock-step `_lift_xy` latch coherent, labels finite/in-bounds/deterministic) and that
one end-to-end DAgger run writes evidence-complete artifacts (BC0 + each round + best + final checkpoints).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.experiments.pick_place_dagger import _run_dagger
from hymeko_rl.experiments.training_spec import TrainingSpec
from hymeko_rl.train.dagger import label_sanity
from hymeko_rl.viz.render_pick_place import fanuc_pick_env


def test_label_sanity_lockstep_ok() -> None:
    """Lock-step expert labels must be finite, in-bounds, deterministic, `_lift_xy`-coherent, and the roll must
    actually reach the committed phase (else the latch went untested)."""
    env = fanuc_pick_env()
    lo = np.asarray(env.action_space.low, dtype=np.float64)
    hi = np.asarray(env.action_space.high, dtype=np.float64)
    rep = label_sanity(env, lambda e: e.expert_action, n_episodes=2, action_lo=lo, action_hi=hi)
    assert rep["finite"] and rep["in_bounds"] and rep["deterministic"] and rep["latch_coherent"]
    assert rep["ok"] and rep["committed_steps"] > 0


@pytest.mark.parametrize("warm_start", [True, False])
def test_dagger_plumbing_and_evidence(tmp_path, warm_start) -> None:
    """Tiny end-to-end DAgger, BOTH re-BC paths (warm continue vs fresh rebuild): evidence-complete artifacts,
    best-checkpoint present."""
    spec = TrainingSpec.from_hymeko("data/robotics/pick_place_dagger.hymeko")
    summary = _run_dagger(spec, kind="mlp", hidden=32, smoke=True, seeds=(0,),
                          warm_start=warm_start, base=str(tmp_path))
    assert summary["algorithm"] == "dagger"
    assert summary["label_sanity"]["ok"] is True
    assert "place_median" in summary and 0.0 <= summary["place_median"] <= 1.0

    exp = Path(summary["dir"])
    assert (exp / "results.json").exists() and (exp / "run.log").exists()
    pol = exp / "policies"
    for stage in ("bc0", "d1", "best", "final"):                       # evidence-complete checkpoints
        assert (pol / f"pick_place_dagger_mlp_s0_{stage}.pt").exists(), f"missing {stage} checkpoint"
    data = json.loads((exp / "results.json").read_text())
    assert data["seeds"][0]["curve"][0]["stage"] == "bc0"
