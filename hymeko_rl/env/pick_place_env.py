"""Pick-and-place env: the 6-DOF HyMeKo arm + a gripper end-effector grasps a box and places it.

Faithful decomposition (Phase 0): the arm **and** the gripper are ONE HyMeKo source via *import* —
``data/robotics/arm_gripper_import.hymeko`` ``@``-imports ``anthropomorphic_arm.hymeko`` (no arm-kinematics
duplication) and attaches the gripper fingers + prismatic grip joints to the imported ``arm.tool`` (HyMeKo
cross-model kinematic composition). It is emitted, and ``compose_pick_place_scene`` adds the table, a
``<freejoint>`` box, and the target. So the environment is a robotic arm + a gripper + an object + a plane, and
the policy reads per-vertex features on the **arm+gripper** kinematic hypergraph (``HypergraphState.from_mjcf``,
9 vertices) — the same contract as every other scenario. (``arm_gripper.hymeko`` is the self-contained
equivalent, kept as a fallback.)

Action = the arm's joint targets + one ``grip`` command (mapped to the two fingers oppositely). Reward is a
dense approach→grasp→lift→place shaping; the declarative ``.hymeko`` reward + staged FSM are Phase 2, and the
reach-to-grasp *control* (the arm posing the gripper at the object) is Phase 1 (behaviour cloning).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import emit_arm_mjcf
from hymeko_rl.env.constants import Physics
from hymeko_rl.env.gripper_world import compose_pick_place_scene
from hymeko_rl.env.ik import DampedPoseIK
from hymeko_rl.env.reward import PickMetrics, RewardSpec
from hymeko_rl.agents.hypergraph_state import HypergraphState

_ARM = "data/robotics/arm_gripper_import.hymeko"   # imports anthropomorphic_arm + attaches the gripper (no dup)
# per-vertex features: [qpos, qvel, x, y, z, (object-vertex)(3), grasping, lifted_norm]. The last two are
# the TASK PHASE (is the gripper holding the box? how high?) broadcast to every vertex — they make the obs
# Markovian for the grasp→lift→place transition a learned policy must time (without them PPO learns only
# approach+contact; see reports/2026-06-23-fanuc-lrmate-top-down-grasp.md).
_FEAT = 10
_ARM_JOINTS = ("j0", "j1", "j2", "j3", "j4", "jtool")   # the emitted 6-DOF arm
_GRIP = 0.035
_OPEN = -0.014          # negative grip opens the fingers PAST rest → wide clearance over the box on descent


class PickPlaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """6-DOF arm + gripper grasping a box. Observation ``(n_vertices, 8)`` per arm/gripper-link vertex:
    ``[qpos, qvel, x, y, z, (object - vertex)(3)]``. Action = ``[6 arm joint targets, grip]``.

    # Preconditions ``frame_skip >= 1``, ``max_steps >= 1``.
    # Postconditions ``step`` returns the gym 5-tuple; ``info["death"]`` set when the object leaves the table."""

    metadata = {"render_modes": []}

    def __init__(self, *, robot: str = _ARM, box_half: float = 0.02, box_mass: float = 0.05,
                 box_z_half: float | None = None,
                 target_xy: tuple[float, float] = (0.14, 0.05), frame_skip: int = 5,
                 max_steps: int = 200, table_z: float = 0.0, table_top: float | None = None,
                 mount_height: float = 0.45,
                 obj_radius: tuple[float, float] = (0.10, 0.18),
                 obj_angle: tuple[float, float] = (-0.5, 0.5),
                 arm_home: tuple[float, ...] | None = None,
                 lift_thresh: float = 0.06, place_radius: float = 0.06,
                 ground_penalty: float = 2.0, finger_friction: float = 3.0,
                 reward_profile: str | None = None, target_bin: bool = False) -> None:
        super().__init__()
        if frame_skip < 1 or max_steps < 1:
            raise ValueError("frame_skip/max_steps must be >= 1")
        arm_gripper = emit_arm_mjcf(robot, name="arm_gripper", control_mode="position")  # arm+gripper, one source
        self.hg = HypergraphState.from_mjcf(arm_gripper, is_path=False)
        self._mjcf = compose_pick_place_scene(arm_gripper, box_half=box_half, box_mass=box_mass,
                                              box_z_half=box_z_half, target_xy=target_xy, table_z=table_z,
                                              table_top=table_top, pedestal=mount_height, target_bin=target_bin)
        self.model = mujoco.MjModel.from_xml_string(self._mjcf)
        self.model.actuator_ctrllimited[:] = 0
        # Pedestal mount: the arm must sit ABOVE its workspace. Floor-mounted, a top-down grasp near the base
        # forces the arm to fold onto itself and into the ground (self-collision link_0↔link_3, link_1↔floor —
        # measured), which physically blocks the kinematically-valid pose. Raising the base resolves it.
        self.mount_height = float(mount_height)
        _base = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
        if _base >= 0:
            self.model.body_pos[_base, 2] += self.mount_height
        # Stable integration (blow-up fix, measured 2026-06-30): the emitted ~1e-3 s sub-step lets the position
        # servo DETONATE on contact penetration (qacc ~1e4 -> the box is ejected upward -> a false "lift" that
        # silently inflated eval_success from a true 0.125 to 0.875). Halve the sub-step and raise the substep
        # count so the control interval (frame_skip * timestep) is preserved EXACTLY; the policy sees an
        # identical control interface, so trained weights transfer without re-training.
        _stable_dt = Physics.STABLE_DT
        _control_dt = self.model.opt.timestep * int(frame_skip)
        if self.model.opt.timestep > _stable_dt:
            substeps = max(1, round(_control_dt / _stable_dt))
            self.model.opt.timestep = _control_dt / substeps
            frame_skip = substeps
        self.data = mujoco.MjData(self.model)
        self.frame_skip, self.max_steps = int(frame_skip), int(max_steps)
        self.box_half, self.table_z = box_half, float(table_z)
        self._box_zhalf = float(box_half if box_z_half is None else box_z_half)  # object half-HEIGHT (tall can)
        # the work surface (table top if elevated, else the ground); the box rests + is grasped here.
        self._surf = float(table_z if table_top is None else table_top)
        self.ground_penalty = float(ground_penalty)
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))
        self.obj_radius = (float(obj_radius[0]), float(obj_radius[1]))
        self.obj_angle = (float(obj_angle[0]), float(obj_angle[1]))
        # a bent, non-singular "ready" posture to reset to (the all-zeros vertical home is a kinematic
        # singularity where IK steps sideways); None keeps the zero home (back-compat for the old arm).
        self._arm_home = None if arm_home is None else np.asarray(arm_home, dtype=np.float64)
        self.lift_thresh, self.place_radius = float(lift_thresh), float(place_radius)

        def bid(n: str) -> int:
            return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n))

        def jid(n: str) -> int:
            return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n))

        self._b_obj, self._b_tool = bid("object"), bid("tool")
        self._b_fl, self._b_fr = bid("finger_l"), bid("finger_r")
        self._obj_qadr = int(self.model.jnt_qposadr[jid("obj")])
        # actuator-id per joint (robust to emit order); gripper joints + arm joints.
        self._act_of: dict[str, int] = {}
        for a in range(self.model.nu):
            j = int(self.model.actuator_trnid[a, 0])
            self._act_of[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""] = a
        # The emitter's default position servo (kp=40) is too weak to hold this heavy arm (5 kg + 2 kg links)
        # at a folded top-down grasp pose — it sags and never tracks the IK target. Brute-force stiffness
        # destabilises the integrator (NaN QACC), so instead **gravity-compensate** the manipulator bodies
        # (MuJoCo ``body_gravcomp``) and use a moderate, stable position gain: the servo then only fights
        # inertia, tracks the IK target, and holds the grasp.
        for nm in (*( "base_link", "link_0", "link_1", "link_2", "link_3", "link_4", "tool",
                      "finger_l", "finger_r"),):
            b = bid(nm)
            if b >= 0:
                self.model.body_gravcomp[b] = 1.0
        gains = {j: (45.0, 9.0) for j in _ARM_JOINTS}                  # tuned for reliable grasp+lift
        gains.update({"grip_l": (250.0, 10.0), "grip_r": (250.0, 10.0)})  # firm squeeze to hold the box on lift
        for jn, (kp, kv) in gains.items():
            a = self._act_of.get(jn, -1)
            if a >= 0:
                self.model.actuator_gainprm[a, 0] = kp
                self.model.actuator_biasprm[a, 1] = -kp
                self.model.actuator_biasprm[a, 2] = -kv
        # finger-object friction so the grasp holds during the lift (an "evil" env lowers this → slippery).
        for b in (self._b_fl, self._b_fr):
            for g in range(int(self.model.body_geomadr[b]),
                           int(self.model.body_geomadr[b]) + int(self.model.body_geomnum[b])):
                self.model.geom_friction[g, 0] = float(finger_friction)
        # gripper-link vertices: (vertex, qadr, dadr) for every non-free joint (arm + grip).
        self._jnt: list[tuple[int, int, int]] = []
        for j in range(self.model.njnt):
            if int(self.model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
                continue
            v = int(self.model.jnt_bodyid[j]) - 1
            self._jnt.append((v, int(self.model.jnt_qposadr[j]), int(self.model.jnt_dofadr[j])))
        # arm joint ranges -> action bounds (+ a grip channel).
        lo, hi = [], []
        for jn in _ARM_JOINTS:
            r = self.model.jnt_range[jid(jn)]
            lo.append(float(r[0]) if r[0] < r[1] else -np.pi)
            hi.append(float(r[1]) if r[0] < r[1] else np.pi)
        self._a_lo = np.array(lo + [0.0], np.float32)
        self._a_hi = np.array(hi + [_GRIP], np.float32)
        self.action_space = spaces.Box(self._a_lo, self._a_hi)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.hg.n_vertices, _FEAT), np.float32)
        self.n_actions = len(_ARM_JOINTS) + 1            # 6 arm joint targets + grip (BC/policy sizing)
        self._ik = DampedPoseIK(self.model, self._b_tool, len(_ARM_JOINTS))
        self._manip = {bid(n) for n in ("base_link", "link_0", "link_1", "link_2", "link_3", "link_4",
                                        "tool", "finger_l", "finger_r")}
        # surfaces the robot must NOT touch: the floor (ground) and the table top. (The box may rest on the
        # table — that is the object, not the manipulator; only arm/gripper contact with these is forbidden.)
        self._no_touch = {g for g in (int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")),
                                      int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")))
                          if g >= 0}
        self._step = 0
        self._grasp_hold = 0           # consecutive steps both fingers have held the box (grip-settle gate)
        self._lift_xy: tuple[float, float] | None = None   # grasp xy captured at commit (lift straight up)
        self.difficulty = 0.0          # curriculum knob (0 easy → 1 hard); read at reset, see set_difficulty
        self._obj_radius0, self._obj_angle0 = self.obj_radius, self.obj_angle   # the configured (easy) spawn
        # optional declarative reward (Phase 2): the whole reward read from a .hymeko task profile. When set,
        # step() scores via RewardSpec.evaluate (reading the cached PickMetrics) instead of the procedural sum.
        self._reward_spec = RewardSpec.from_hymeko(reward_profile) if reward_profile else None
        self._pick_metrics: PickMetrics | None = None
        self._grasped_ever = False                     # latched once both fingers first hold the box
        self._spawn_xy = (0.0, 0.0)                     # object xy at reset (for the pre-grasp disturbance penalty)

    def set_difficulty(self, d: float) -> None:
        """Curriculum knob in [0, 1]: widens the object spawn from the configured easy range toward the reach
        edge and wider angles. Takes effect on the NEXT ``reset`` (no rebuild). 0 leaves the configured range.

        # Postconditions ``self.difficulty`` set; ``obj_radius``/``obj_angle`` interpolated easy→hard."""
        self.difficulty = float(np.clip(d, 0.0, 1.0))
        (r_lo, r_hi), (a_lo, a_hi) = self._obj_radius0, self._obj_angle0
        push = 0.10 * self.difficulty                              # object pushed out to the reach edge
        widen = 1.0 + 1.5 * self.difficulty                        # angle range widened
        self.obj_radius = (r_lo + push, r_hi + push)
        self.obj_angle = (a_lo * widen, a_hi * widen)

    def _ik_pose_valid(self, d: mujoco.MjData) -> bool:
        """Reject IK solutions where a manipulator link self-collides or hits the floor (body 0 = world).
        Finger-object contact and contact-free hovers are allowed."""
        for i in range(int(d.ncon)):
            c = d.contact[i]
            b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
            m1, m2 = b1 in self._manip, b2 in self._manip
            if (m1 and m2) or (m1 and b2 == 0) or (m2 and b1 == 0):
                return False
        return True

    def _obj_xyz(self) -> np.ndarray:
        return np.asarray(self.data.xpos[self._b_obj], dtype=np.float32)

    def _both_contact(self) -> tuple[bool, bool]:
        """(left, right) finger-object contact, classified by body (robust to emitted geom names)."""
        left = right = False
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[int(con.geom1)])
            b2 = int(self.model.geom_bodyid[int(con.geom2)])
            if self._b_obj not in (b1, b2):
                continue
            other = b2 if b1 == self._b_obj else b1
            if other == self._b_fl:
                left = True
            elif other == self._b_fr:
                right = True
        return left, right

    def _ground_contact(self) -> bool:
        """True if any manipulator link (arm or gripper) touches a no-touch surface — the floor OR the table.
        The robot must approach/grasp the box without crashing the arm or gripper into either surface."""
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            surf = g1 if g1 in self._no_touch else g2 if g2 in self._no_touch else -1
            if surf < 0:
                continue
            other = int(self.model.geom_bodyid[g2 if surf == g1 else g1])
            if other in self._manip:
                return True
        return False

    def node_features(self) -> np.ndarray:
        feat = np.zeros((self.hg.n_vertices, _FEAT), dtype=np.float32)
        ox, oy, oz = (float(c) for c in self._obj_xyz())
        for v, qadr, dadr in self._jnt:
            if 0 <= v < self.hg.n_vertices:
                feat[v, 0] = float(self.data.qpos[qadr])
                feat[v, 1] = float(self.data.qvel[dadr])
        for v in range(self.hg.n_vertices):
            x, y, z = (float(c) for c in self.data.xpos[v + 1])
            feat[v, 2:5] = (x, y, z)
            feat[v, 5:8] = (ox - x, oy - y, oz - z)
        # task phase (broadcast to every vertex): is the box grasped, and how high is it lifted.
        left, right = self._both_contact()
        grasping = 1.0 if (left and right) else 0.0
        lifted = max(0.0, oz - (self._surf + self._box_zhalf))
        feat[:, 8] = grasping
        feat[:, 9] = float(np.clip(lifted / max(self.lift_thresh, 1e-6), 0.0, 2.0))
        return feat

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self._arm_home is not None:
            self.data.qpos[:len(_ARM_JOINTS)] = self._arm_home
        # a spot on the table within the arm's TOP-DOWN reach envelope (smaller than its planar reach).
        r = float(self.np_random.uniform(*self.obj_radius))
        th = float(self.np_random.uniform(*self.obj_angle))
        q = self._obj_qadr
        self.data.qpos[q:q + 3] = [r * np.cos(th), r * np.sin(th), self._surf + self._box_zhalf]
        self.data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._grasp_hold = 0
        self._lift_xy = None
        self._grasped_ever = False
        o0 = self._obj_xyz()
        self._spawn_xy = (float(o0[0]), float(o0[1]))
        return self.node_features(), {"obj_to_target": self._obj_to_target()}

    def _obj_to_target(self) -> float:
        o = self._obj_xyz()
        return float(np.hypot(o[0] - self.target_xy[0], o[1] - self.target_xy[1]))

    # tool height above the work surface at which the fingers straddle the box's UPPER body (finger tips ~1 cm
    # above the table; the grasped box is gravity-compensated so the arm does not sag the tips into the table).
    _GRASP_TOOL_DZ = 0.115

    @property
    def expert_action(self) -> np.ndarray:
        """A scripted reach→grasp→lift→transport→place demonstrator (for behaviour cloning, Phase 1).

        A converged DLS pose IK (:class:`DampedPoseIK`) drives the tool to a staged target with the gripper
        held **top-down** (tool z-axis → world −z); the grip opens/closes per phase. Phase is read from state
        (above-object, grasped, lifted, at-target), so it is a single-valued closed-loop function of state."""
        obj = self._obj_xyz()
        tool = np.asarray(self.data.xpos[self._b_tool], dtype=np.float64)
        grasped = all(self._both_contact())
        committed = grasped and self._grasp_hold >= 12       # grip-settle: hold + squeeze before lifting
        lifted = bool(obj[2] - (self._surf + self._box_zhalf) > 0.04)
        tx, ty = self.target_xy
        grasp_z = self._surf + self.box_half + self._GRASP_TOOL_DZ
        horiz = float(np.hypot(tool[0] - obj[0], tool[1] - obj[1]))
        near_place = float(np.hypot(obj[0] - tx, obj[1] - ty)) < 0.05

        hover_z = grasp_z + 0.14                              # hover well ABOVE the box, descend straight down
        lift_clear = grasp_z + 0.13                           # clearance height to lift to before transporting
        if not committed:
            self._lift_xy = None                             # re-capture the grasp xy on the next commit
        if committed:
            if self._lift_xy is None:                        # capture the grasp xy ONCE (lift from a FIXED
                self._lift_xy = (float(tool[0]), float(tool[1]))  # point — never chase the drifting tool xy)
            lx, ly = self._lift_xy
            if tool[2] < lift_clear - 0.02 and not lifted:   # 1) lift STRAIGHT UP from the captured grasp xy
                tgt, grip = np.array([lx, ly, lift_clear]), _GRIP
            elif not near_place:                             # 2) transport at height
                tgt, grip = np.array([tx, ty, lift_clear]), _GRIP
            else:                                            # 3) lower + release
                tgt, grip = np.array([tx, ty, grasp_z]), (_GRIP if not lifted else 0.0)
        elif grasped:                                        # DWELL: hold position + keep squeezing to settle
            tgt, grip = np.array([obj[0], obj[1], grasp_z]), _GRIP
        elif horiz > 0.025:                                  # get above the object (centered), fingers WIDE
            tgt, grip = np.array([obj[0], obj[1], hover_z]), _OPEN
        elif tool[2] > grasp_z + 0.02:                       # descend STRAIGHT down, fingers WIDE (clear box)
            tgt, grip = np.array([obj[0], obj[1], grasp_z]), _OPEN
        else:                                                # at the object: close
            tgt, grip = np.array([obj[0], obj[1], grasp_z]), _GRIP

        q_now = np.asarray(self.data.qpos[:len(_ARM_JOINTS)], dtype=np.float64)
        # Solve the full staged goal (strong position + down-orientation) and rate-limit the JOINT step (a
        # Cartesian-waypoint target starves the position term when the orientation correction is large). Fast
        # on the free approach, gentle while carrying so the lift does not slip the box.
        q_des = self._ik.solve(q_now, tgt, down=True, iters=120)
        rate = 0.28 if committed else 0.4
        q_des = q_now + np.clip(q_des - q_now, -rate, rate)
        return np.asarray(np.concatenate([q_des, [grip]]), dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.clip(np.asarray(action, dtype=np.float32), self._a_lo, self._a_hi)
        grip = float(a[-1])
        for jn, target in zip(_ARM_JOINTS, a[:-1]):
            self.data.ctrl[self._act_of[jn]] = float(target)
        self.data.ctrl[self._act_of["grip_l"]] = grip        # +grip closes finger_l toward centre
        self.data.ctrl[self._act_of["grip_r"]] = -grip       # -grip closes finger_r toward centre
        # While the box is held, gravity-compensate it so its weight does not sag the arm (which would drive
        # the fingers down into the table); released → falls normally onto the target. Removes the sag at source.
        self.model.body_gravcomp[self._b_obj] = 1.0 if all(self._both_contact()) else 0.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        o = self._obj_xyz()
        tool = np.asarray(self.data.xpos[self._b_tool], dtype=np.float32)
        approach = float(np.linalg.norm(tool - o))
        left, right = self._both_contact()
        self._grasp_hold = self._grasp_hold + 1 if (left and right) else 0
        lifted = float(max(0.0, o[2] - (self._surf + self._box_zhalf)))
        to_target = self._obj_to_target()
        reached = bool(lifted > self.lift_thresh and to_target < self.place_radius)
        grasping = bool(left and right)
        if grasping:
            self._grasped_ever = True
        on_ground = self._ground_contact()                 # arm/gripper touching the floor OR table
        # Contact while positioned OVER the object (descending to grasp it) is fine — the box sits on the
        # table. The failure to avoid is the gripper crashing into the table/floor on the APPROACH, i.e. while
        # NOT over the object (or while carrying it elsewhere). That is what gets penalised.
        tool_horiz = float(np.hypot(tool[0] - o[0], tool[1] - o[1]))
        approach_contact = on_ground and tool_horiz > 0.06
        # pre-grasp disturbance: BEFORE the object is ever grasped, any displacement of it is a penalty (no
        # nudging/pushing it around — reach it cleanly and only move it once held). 0 once grasped.
        pre_grasp_disturb = (0.0 if self._grasped_ever
                             else float(np.hypot(o[0] - self._spawn_xy[0], o[1] - self._spawn_xy[1])))
        self._pick_metrics = PickMetrics(
            approach=approach, left=float(left), right=float(right), lifted=lifted,
            lift_thresh=self.lift_thresh, to_target=to_target, reached=reached,
            approach_contact=approach_contact, pre_grasp_disturb=pre_grasp_disturb)
        if self._reward_spec is not None:                  # declarative reward (Phase 2) — reads PickMetrics
            reward = self._reward_spec.evaluate(self, approach, a)
        else:                                              # procedural reward (parity-tested equal to the spec)
            reward = (-approach + 0.5 * (left + right) + 5.0 * min(lifted, self.lift_thresh)
                      + (-to_target if lifted > 0.02 else 0.0) + (20.0 if reached else 0.0)
                      - (self.ground_penalty if approach_contact else 0.0)
                      - 3.0 * pre_grasp_disturb)
        death = bool(o[2] < self._surf - 0.05 or np.hypot(o[0], o[1]) > 0.8)
        terminated = bool(reached or death)
        truncated = self._step >= self.max_steps
        info = {"obj_to_target": to_target, "lifted": lifted, "both_contact": grasping,
                "reached": reached, "death": death, "on_ground": on_ground,
                "approach_contact": approach_contact}
        return self.node_features(), float(reward), terminated, truncated, info
