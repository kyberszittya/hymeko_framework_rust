"""MuJoCo simulation bridge for HSiKAN.

Runs a small kinematic mechanism in MuJoCo, drives joints with a
deterministic (sinusoidal) controller, and extracts per-timestep state
as continuous vertex/edge features that HSiKAN can consume alongside
the static cycle-pool topology.

State extracted per body (vertex):
    xyz       : position in world frame (3-vec)
    quat      : orientation (4-vec, wxyz)
    lin_vel   : linear velocity (3-vec)
    ang_vel   : angular velocity (3-vec)
                                                            => 13 features

State extracted per joint (edge):
    qpos      : joint position (1 scalar for 1-DOF joints)
    qvel      : joint velocity (1 scalar)
    ctrl      : control input (1 scalar)
                                                            => 3 features

The simulation loop produces a (T_steps, V, 13) tensor of vertex
features and a (T_steps, E, 3) tensor of edge features per trajectory.

These plug into HSiKAN as additional per-vertex / per-edge inputs
alongside the cycle structure, enabling tasks like:

  - **Forward kinematics**: predict end-effector pose from joint angles
  - **Next-state prediction**: predict (qpos, qvel) at t+dt from t
  - **Joint torque estimation**: predict τ from observed qpos, qvel
  - **Anomaly detection**: flag trajectories whose graph-context
    embedding deviates from typical state distribution

Usage
-----
>>> sim = MuJoCoBridge.canonical_4dof_arm()
>>> states = sim.run(duration=2.0, controller="sine")
>>> states.vertex_features.shape   # (T, V, 13)
>>> states.edge_features.shape     # (T, E, 3)
>>> g = sim.kinematic_graph()       # SignedGraph (revolute=+1, prismatic=-1)
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mujoco

from .datasets import SignedGraph


# Canonical 4-DOF arm (used in the existing paper figure scenario).
_4DOF_MJCF = """
<mujoco model="hsikan_4dof_arm">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default>
    <joint damping="2.0" limited="true" range="-2.5 2.5"/>
    <geom rgba="0.55 0.58 0.62 1" friction="0.8 0.05 0.0005"/>
    <position kp="40" kv="4" ctrlrange="-2 2"/>
  </default>
  <worldbody>
    <body name="base_link" pos="0 0 0.1">
      <geom type="cylinder" size="0.06 0.10"/>
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <body name="shoulder_link" pos="0 0 0.10">
        <geom type="box" size="0.04 0.04 0.20" pos="0 0 0.20"/>
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <body name="elbow_link" pos="0 0 0.40">
          <geom type="box" size="0.035 0.035 0.18" pos="0 0 0.18"/>
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <body name="wrist_link" pos="0 0 0.36">
            <geom type="cylinder" size="0.03 0.06" pos="0 0 0.06"/>
            <joint name="j4" type="hinge" axis="0 0 1"/>
            <body name="flange_link" pos="0 0 0.12">
              <geom type="cylinder" size="0.04 0.02" rgba="0.22 0.46 0.78 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="aj1" joint="j1"/>
    <position name="aj2" joint="j2"/>
    <position name="aj3" joint="j3"/>
    <position name="aj4" joint="j4"/>
  </actuator>
</mujoco>
"""


# Canonical 4-bar linkage (planar, equality-constrained closure).
_4BAR_MJCF = """
<mujoco model="hsikan_four_bar">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.1" limited="false"/>
    <geom rgba="0.55 0.58 0.62 1" friction="0.8 0.05 0.0005"/>
  </default>
  <worldbody>
    <body name="ground" pos="0 0 0.5">
      <geom type="box" size="0.3 0.02 0.02"/>
      <body name="crank" pos="-0.2 0 0">
        <joint name="j_gc" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.015"/>
        <body name="coupler" pos="0.15 0 0">
          <joint name="j_cc" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.013"/>
          <body name="rocker" pos="0.30 0 0">
            <joint name="j_cr" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.0 0 -0.20" size="0.014"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <!-- Rocker tip pinned to the right end of the ground bar. -->
    <connect body1="rocker" body2="ground" anchor="0 0 -0.20"/>
  </equality>
</mujoco>
"""


@dataclass
class SimulationStates:
    """Per-trajectory state tensors."""
    timestamps: np.ndarray         # (T,) seconds
    vertex_features: np.ndarray    # (T, V, 13) xyz+quat+lin_vel+ang_vel per body
    edge_features: np.ndarray      # (T, E, 3) qpos+qvel+ctrl per joint
    body_names: list[str]
    joint_names: list[str]


class MuJoCoBridge:
    """Wraps a MuJoCo model and provides:
      - kinematic_graph(): SignedGraph extraction (joints as edges)
      - run(): physics rollout, returns SimulationStates"""

    def __init__(self, mjcf: str):
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self._mjcf = mjcf

    @classmethod
    def canonical_4dof_arm(cls) -> "MuJoCoBridge":
        return cls(_4DOF_MJCF)

    @classmethod
    def canonical_4bar(cls) -> "MuJoCoBridge":
        return cls(_4BAR_MJCF)

    @classmethod
    def from_mjcf_string(cls, mjcf: str) -> "MuJoCoBridge":
        return cls(mjcf)

    @classmethod
    def from_mjcf_file(cls, path: str | Path) -> "MuJoCoBridge":
        return cls(Path(path).read_text())

    # ------------------------------------------------------------------
    # Static graph extraction.

    def kinematic_graph(self) -> SignedGraph:
        """Build SignedGraph from MuJoCo body parent pointers + joint
        types. Edges = joint connections between parent body and child
        body. Sign = +1 (hinge/ball/slide_rotational) or −1 (slide_linear).
        Body 0 is always 'world'; we drop it from the graph."""
        m = self.model
        # Body parent map: m.body_parentid[i] is parent body id.
        # World body is 0; skip.
        body_parents = m.body_parentid
        body_jntadr = m.body_jntadr
        body_jntnum = m.body_jntnum
        # Collect (parent, child, sign) triples for each non-world body
        # that has at least one joint.
        edges, signs = [], []
        for b in range(1, m.nbody):
            parent = body_parents[b]
            jnt_start = body_jntadr[b]
            jnt_n = body_jntnum[b]
            if jnt_n <= 0:
                continue   # rigid attachment, no joint
            for j in range(jnt_start, jnt_start + jnt_n):
                jtype = int(m.jnt_type[j])
                # mjJNT_HINGE=3, mjJNT_SLIDE=2, mjJNT_BALL=1, mjJNT_FREE=0
                if jtype == 2:    # slide
                    sign = -1
                else:             # hinge/ball/free → rotational/free
                    sign = +1
                edges.append((int(parent), int(b)))
                signs.append(sign)
        # Re-index to drop body 0 (world) if it has no edges.
        unique_v = sorted(set(v for e in edges for v in e))
        if 0 not in unique_v and m.nbody > 1:
            unique_v = [0] + unique_v
        idx_map = {v: i for i, v in enumerate(unique_v)}
        edges_re = np.array(
            [(idx_map[u], idx_map[v]) for u, v in edges], dtype=np.int64,
        ) if edges else np.zeros((0, 2), dtype=np.int64)
        return SignedGraph(
            edges=edges_re,
            signs=np.array(signs, dtype=np.int8) if signs else np.zeros((0,), dtype=np.int8),
            n_nodes=len(unique_v),
        )

    # ------------------------------------------------------------------
    # Dynamic state rollout.

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def _capture_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture per-body and per-joint state at the current sim time."""
        m, d = self.model, self.data
        nbody = m.nbody
        # Per-body: xyz (3), quat (4), lin_vel (3), ang_vel (3) = 13
        v_feat = np.zeros((nbody, 13), dtype=np.float32)
        for b in range(nbody):
            v_feat[b, :3]    = d.xpos[b]
            v_feat[b, 3:7]   = d.xquat[b]
            v_feat[b, 7:10]  = d.cvel[b, 3:6]   # linear part of cvel
            v_feat[b, 10:13] = d.cvel[b, 0:3]   # angular part
        # Per-joint: qpos (1), qvel (1), ctrl (1) = 3
        # Only 1-DOF joints — multi-DOF joints (ball/free) take more slots
        # and are out of scope for this simple bridge.
        njnt = m.njnt
        e_feat = np.zeros((njnt, 3), dtype=np.float32)
        for j in range(njnt):
            jtype = int(m.jnt_type[j])
            if jtype not in (2, 3):  # not slide or hinge → skip
                continue
            qadr = m.jnt_qposadr[j]
            vadr = m.jnt_dofadr[j]
            e_feat[j, 0] = d.qpos[qadr]
            e_feat[j, 1] = d.qvel[vadr]
            # Find the actuator targeting this joint, if any.
            for ai in range(m.nu):
                if int(m.actuator_trnid[ai, 0]) == j:
                    e_feat[j, 2] = d.ctrl[ai]
                    break
        return v_feat, e_feat

    def run(self, duration: float = 2.0,
              controller: str = "sine",
              record_every: int = 5) -> SimulationStates:
        """Roll out the simulation for ``duration`` seconds.

        ``controller``: "sine" → all actuators driven by sin(t * f)
                         "zero" → no actuation (passive dynamics)
        ``record_every``: capture every Nth simulation step (default 5
                          → 100 Hz capture from a 500 Hz sim).
        """
        m, d = self.model, self.data
        self.reset()
        n_steps = int(duration / m.opt.timestep)
        v_buf, e_buf, t_buf = [], [], []
        body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                       for i in range(m.nbody)]
        joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                        for i in range(m.njnt)]
        for step in range(n_steps):
            t = step * m.opt.timestep
            if controller == "sine":
                for ai in range(m.nu):
                    f = 0.5 + 0.2 * ai
                    d.ctrl[ai] = 0.6 * np.sin(2 * np.pi * f * t)
            mujoco.mj_step(m, d)
            if step % record_every == 0:
                v, e = self._capture_state()
                v_buf.append(v); e_buf.append(e); t_buf.append(t)
        return SimulationStates(
            timestamps=np.array(t_buf, dtype=np.float32),
            vertex_features=np.array(v_buf, dtype=np.float32),
            edge_features=np.array(e_buf, dtype=np.float32),
            body_names=body_names,
            joint_names=joint_names,
        )


# --- Demo: run the canonical 4-DOF arm + 4-bar and report shapes ---

if __name__ == "__main__":
    print("=== 4-DOF arm (serial) ===")
    sim = MuJoCoBridge.canonical_4dof_arm()
    g = sim.kinematic_graph()
    print(f"  graph: {g.stats()}")
    s = sim.run(duration=2.0, controller="sine")
    print(f"  trajectory: T={len(s.timestamps)} samples  "
          f"V={s.vertex_features.shape[1]}  E={s.edge_features.shape[1]}")
    print(f"  bodies: {s.body_names}")
    print(f"  joints: {s.joint_names}")
    # Print end-effector trajectory amplitude.
    flange_idx = s.body_names.index("flange_link")
    flange_xyz = s.vertex_features[:, flange_idx, :3]
    def _ptp(a):
        return float(a.max() - a.min())
    print(f"  flange XYZ range: x={_ptp(flange_xyz[:,0]):.3f}  "
          f"y={_ptp(flange_xyz[:,1]):.3f}  z={_ptp(flange_xyz[:,2]):.3f}")

    print("\n=== 4-bar linkage (closed loop via equality constraint) ===")
    sim2 = MuJoCoBridge.canonical_4bar()
    g2 = sim2.kinematic_graph()
    print(f"  graph: {g2.stats()}")
    s2 = sim2.run(duration=2.0, controller="zero", record_every=5)
    print(f"  trajectory: T={len(s2.timestamps)} samples  "
          f"V={s2.vertex_features.shape[1]}  E={s2.edge_features.shape[1]}")
