"""Regression guards for the canonical pick-place reconciliation (2026-07-16, canonical_integration/pick_place).

These prevent the pipeline from silently reverting to an older abstraction (§12): the canonical loader must keep
loading a TD3+BC DeterministicActor (G1), and the canonical evaluator must keep reporting the far/near split with
separate reached/grasped/placed metrics (G2) — never one ambiguous `success` that re-opens the 0.875-vs-0.167 trap.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_TD3BC = Path("experiments/2026_07_13_02_55_fanuc_pick_td3bc_hsikan/policies/fanuc_pick_td3bc_hsikan_s0.pt")


def test_registry_specs_carry_full_metadata() -> None:
    """Every canonical artifact is declared with provenance; loadable ones have a checkpoint or are scripted (G4/G5)."""
    from hymeko_rl.experiments.canonical_pick_place import REGISTRY
    names = {s.name for s in REGISTRY}
    assert {"scripted_v3_expert", "ff_dagger_base", "td3bc_s0", "plain_sac_negative"} <= names
    for s in REGISTRY:
        for field in ("method", "architecture", "action_abstraction", "obs_schema", "env_version",
                      "source_experiment", "selection_rule", "fallback"):
            assert getattr(s, field), f"{s.name} missing {field}"
        assert s.fallback == "none"                                  # no silent scripted fallback (routing-verified)
        if s.loadable and s.method != "scripted":
            assert s.checkpoint, f"loadable {s.name} must name a checkpoint"


@pytest.mark.skipif(not _TD3BC.exists(), reason="cached TD3+BC checkpoint not present")
def test_canonical_loader_loads_td3bc_deterministic_actor() -> None:
    """G1: load_pick_policy must build a TD3+BC DeterministicActor into a working action_fn (was fail-loud before)."""
    import numpy as np

    from hymeko_rl.experiments.gripper_pick_bc import load_pick_policy
    from hymeko_rl.viz.render_pick_place import fanuc_pick_env
    env = fanuc_pick_env(expert_version=3, require_settle=True, max_steps=1000)
    fn = load_pick_policy(str(_TD3BC), env, kind="hsikan")
    obs, _ = env.reset(seed=20016)
    a = np.asarray(fn(env, obs))
    assert a.shape == (env.n_actions,)                               # a usable 7-dim action (env clips grip)


def test_canonical_evaluator_reports_far_near_split() -> None:
    """G2: evaluate_canonical must return the split metrics separately — never one ambiguous `success`."""
    from hymeko_rl.experiments.canonical_pick_place import evaluate_canonical
    from hymeko_rl.viz.render_pick_place import expert_action_fn
    m = evaluate_canonical(expert_action_fn(), n=4, seed0=20_000)
    for k in ("reached_full", "reached_far", "grasped_far", "placed_real_far", "safety_no_divergence", "n_far"):
        assert k in m, f"canonical metric {k} missing — the far/near split regressed"
    assert "success" not in m                                        # the overloaded metric must not reappear
    assert 0.0 <= m["placed_real_far"] <= 1.0 and m["n_far"] >= 1
