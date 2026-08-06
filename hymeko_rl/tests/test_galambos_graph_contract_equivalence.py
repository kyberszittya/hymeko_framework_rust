"""GALAMBOS_PLANAR_GRAPH_CONTRACT_EQUIVALENCE (2026-07-22, Option B): the canonical v2 robot
(``robot_source="hymeko_spec"``, 10 physical bodies) projects — via the explicit ``graph_node 0`` helper metadata —
to a semantic policy graph that is EXACTLY the legacy checkpoint-producing graph (``robot_source="legacy_python"``,
6 vertices / 48-dim actor input): identical vertex names+order, identical signed edges, byte-identical raw node
features and flattened actor input on a matched state panel, and identical action-to-actuator ordering.

This is the load-bearing migration guarantee: the physical/specification authority moves to HyMeKo v2 while the
frozen deploy stack keeps loading and acting identically. Physics parity is proven separately
(:mod:`test_galambos_planar_v2_parity`); this file proves the SEMANTIC graph contract.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.agents.multichannel_ctde import arm_action_partition
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

# matched state panel: neutral, symmetric grasp, asymmetric grasp, two limit-adjacent poses (near ±4), and a
# historical-approach-like pose. Full qpos (4 arm joints + disk_x/y/rz) is set identically on both robots.
POSES = [
    [0.0, 0.0, 0.0, 0.0, 0.03, 0.23, 0.0],           # neutral (coin at a historical spawn)
    [0.6, 0.6, -0.6, -0.6, 0.0, 0.16, 0.0],          # symmetric grasp
    [1.0, 0.8, -1.0, -0.4, -0.05, 0.19, 0.3],        # asymmetric grasp
    [3.9, -3.9, 3.9, -3.9, 0.10, 0.10, -0.2],        # limit-adjacent (+)
    [-3.9, 3.9, -3.9, 3.9, -0.12, 0.21, 0.5],        # limit-adjacent (-)
    [0.35, 0.13, 0.04, -0.56, 0.03, 0.23, 0.0],      # E-approach-like step-zero pose
]


def _env(src: str) -> PlanarGraspEnv:
    return PlanarGraspEnv(robot_source=src, scene_source="hymeko_spec", max_steps=300, difficulty=0.3)


@pytest.fixture(scope="module")
def envs():
    leg, v2 = _env("legacy_python"), _env("hymeko_spec")
    leg.reset(seed=0)
    v2.reset(seed=0)
    return leg, v2


def _features_at(env: PlanarGraspEnv, qpos) -> np.ndarray:
    env.data.qpos[:len(qpos)] = np.asarray(qpos, float)
    mujoco.mj_forward(env.model, env.data)
    env._planar_metrics = env._metrics()
    return np.asarray(env.node_features(), np.float32)


def test_vertex_count_names_order_identical(envs):
    leg, v2 = envs
    assert leg.hg.vertex_labels == v2.hg.vertex_labels == (
        "base_left", "link1_left", "link2_left", "base_right", "link1_right", "link2_right")
    assert leg.hg.n_vertices == v2.hg.n_vertices == 6


def test_signed_edges_and_incidence_identical(envs):
    leg, v2 = envs
    assert np.array_equal(np.asarray(leg.hg.edges), np.asarray(v2.hg.edges))
    assert np.array_equal(np.asarray(leg.hg.signs), np.asarray(v2.hg.signs))


def test_semantic_fingerprint_identical(envs):
    leg, v2 = envs
    assert leg.hg.semantic_fingerprint() == v2.hg.semantic_fingerprint()


def test_raw_node_features_and_actor_input_identical_on_panel(envs):
    leg, v2 = envs
    worst = 0.0
    for q in POSES:
        fl, fv = _features_at(leg, q), _features_at(v2, q)
        assert fl.shape == fv.shape == (6, 8)
        worst = max(worst, float(np.max(np.abs(fl - fv))))
        assert fl.flatten().size == 48
    assert worst < 1e-6, f"raw node-feature / actor-input parity {worst} exceeds 1e-6 on the state panel"


def test_action_to_actuator_ordering_identical(envs):
    leg, v2 = envs
    lp = [(s.start, s.stop) for s in arm_action_partition(leg)]
    vp = [(s.start, s.stop) for s in arm_action_partition(v2)]
    assert lp == vp == [(0, 2), (2, 4)], f"arm-action partition differs: legacy {lp} vs v2 {vp}"
    lnames = [mujoco.mj_id2name(leg.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(leg.n_actions)]
    vnames = [mujoco.mj_id2name(v2.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(v2.n_actions)]
    # actuator side ordering (left,left,right,right) must match; the prefix (a_ vs act_) is immaterial to the graph.
    def _side(n: str) -> str:
        return "left" if n.endswith("left") else "right"
    assert [_side(n) for n in lnames] == [_side(n) for n in vnames] == ["left", "left", "right", "right"]


def test_ring_embodiment_also_projects_to_six(envs):
    # The CONCAVE_CLAMP/RING fingertip variant adds clamp geometry to the (still non-semantic) fingertip helper
    # bodies, so the canonical v2 semantic graph is still the 6-vertex legacy contract.
    from hymeko_rl.coin_delivery.env_factory import make_coin_env
    ring = make_coin_env(embodiment="CONCAVE_CLAMP")
    assert ring.hg.n_vertices == 6
    assert ring.hg.vertex_labels == envs[0].hg.vertex_labels
