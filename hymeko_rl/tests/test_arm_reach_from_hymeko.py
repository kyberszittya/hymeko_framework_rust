"""The reaching env built from a ``.hymeko`` robot — the canonical hymeko→mjcf path.

One ``.hymeko`` source produces the MuJoCo scene, the kinematic hypergraph, and the
observation space (the Kato "one source, three products" story), end to end. These tests
exercise that the **emitted** arm — not the hand-authored ``arm_world`` arm — loads,
is controllable, and is reachable by the computed-torque expert at its native 6 DOF.

Skips cleanly when the ``hymeko`` CLI has not been built (``cargo build -p hymeko_cli``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.env.arm_reach_env import _bounds_with_fallback, ArmReachEnv
from hymeko_rl.env.arm_world import _hymeko_cli, emit_arm_mjcf

_REPO = Path(__file__).resolve().parents[2]
_ARM = _REPO / "data" / "robotics" / "anthropomorphic_arm.hymeko"


def _cli_available() -> bool:
    try:
        _hymeko_cli()
        return True
    except FileNotFoundError:
        return False


requires_cli = pytest.mark.skipif(
    not _cli_available(), reason="hymeko CLI not built (cargo build -p hymeko_cli)")


# ── pure helper (no CLI needed) ──────────────────────────────────────────────
def test_bounds_fallback_replaces_only_degenerate_elements() -> None:
    limited = np.array([True, False, True], dtype=bool)
    lo = np.array([-2.0, 0.0, 0.5], dtype=np.float32)
    hi = np.array([2.0, 0.0, 0.4], dtype=np.float32)   # idx2 is limited-but-degenerate
    out_lo, out_hi = _bounds_with_fallback(limited, lo, hi, 9.0)
    # idx0: genuinely limited → preserved; idx1: unlimited → fallback; idx2: hi<=lo → fallback.
    assert out_lo.tolist() == [-2.0, -9.0, -9.0]
    assert out_hi.tolist() == [2.0, 9.0, 9.0]
    assert np.all(out_hi > out_lo)


# ── emitted-arm integration (CLI required) ───────────────────────────────────
@requires_cli
def test_emitted_mjcf_is_articulated() -> None:
    """The bridge emits a 6-DoF, non-degenerate scene (guards B-004/B-005 at the env)."""
    mjcf = emit_arm_mjcf(_ARM)
    assert mjcf.count("<joint ") == 6
    # canonical anthropomorphic arm: local axes are X and Z only (world Y comes from j1's twist).
    assert 'axis="1 0 0"' in mjcf and 'axis="0 0 1"' in mjcf   # mixed axes survived
    assert '<body name="world"' not in mjcf                    # world→worldbody


@requires_cli
def test_emitted_arm_loads_and_is_controllable() -> None:
    env = ArmReachEnv.from_hymeko(_ARM, max_steps=60)
    # native 6 DOF, one vertex per body (7), 8 features per vertex.
    assert env.n_actions == 6
    assert env.observation_space.shape == (env.hg.n_vertices, 8)
    # the emitted motors now carry their declared joint efforts (j0/j1 ±500 N·m via
    # joint0_limit, j2-jtool ±50 inherited from joint_rev_limit) — not the env fallback.
    assert bool(np.all(env.model.actuator_ctrllimited))
    assert sorted(set(env.action_space.high.tolist())) == [50.0, 500.0]
    assert np.all(env.action_space.high > env.action_space.low)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.hg.n_vertices, 8)
    # the joint-range fallback yields a real (non-home) target, not FK(0).
    assert float(np.linalg.norm(info["target"])) > 0.05


@requires_cli
def test_emitted_arm_expert_reaches() -> None:
    """The computed-torque expert drives the EE substantially toward the target on the
    emitted 6-DoF arm — the proof that the canonical scene is actually controllable."""
    ratios = []
    for seed in range(5):
        env = ArmReachEnv.from_hymeko(_ARM, max_steps=150, reach_thresh=0.05)
        env.reset(seed=seed)
        d0 = float(np.linalg.norm(env._ee_pos() - env._target))
        dist = d0
        for _ in range(150):
            _, _, term, trunc, info = env.step(env.expert_action)
            dist = info["dist"]
            if term or trunc:
                break
        assert dist <= d0, f"seed {seed}: expert made it worse ({d0:.3f}→{dist:.3f})"
        ratios.append(dist / d0)
    # robust over seeds: the expert closes most of the gap (observed median ≈ 0.12).
    assert float(np.median(ratios)) < 0.4, f"weak reaching: ratios={ratios}"


@requires_cli
def test_emitted_and_authored_arms_share_the_env() -> None:
    """Same env class, two scene sources — both now declare their actuator ctrlranges, so
    neither needs the fallback: arm_world from its <motor ctrlrange>, the emitted arm from
    the .hymeko joint efforts carried through extract_joint_limits → emit_mjcf."""
    authored = ArmReachEnv(control_mode="torque")          # arm_world, 4 DOF, ±25
    emitted = ArmReachEnv.from_hymeko(_ARM)                 # hymeko, 6 DOF, ±500/±50
    assert authored.n_actions == 4 and emitted.n_actions == 6
    assert bool(np.all(authored.model.actuator_ctrllimited))
    assert bool(np.all(emitted.model.actuator_ctrllimited))
    # the emitted bounds come from the declared efforts, not arm_world's ±25.
    assert max(emitted.action_space.high.tolist()) == 500.0
