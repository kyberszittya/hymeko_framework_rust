"""Inverted-pendulum (cart-pole) balancing env — the clean testbed for the HSiKAN actor-critic.

The cart-pole is described in HyMeKo (``data/robotics/inverted_pendulum.hymeko``): a cart slides on a
rail (prismatic, AXIS_X) and a pole hinges on it (continuous, AXIS_Y). The **same source** yields the
MuJoCo scene *and* the 2-vertex kinematic hypergraph (cart, pole) the HSiKAN actor/critic message-passes
over — so this isolates the architecture on the canonical control benchmark, decoupled from the hard
Galambos grasp task.

A *separate class* from :class:`ArmReachEnv` (the task is structurally different — balancing, not
reaching), but it speaks the same gymnasium contract and the same per-vertex obs shape ``(N, feat)``, so
the **shared** PPO loop (``hymeko_rl.ppo.train_ppo``) and the **shared** backbone swap
(``build_policy("hsikan" | "mlp")``) drive it unchanged: fix the algorithm, ablate the architecture.

The emitter actuates every non-fixed joint; the canonical inverted pendulum is under-actuated, so the
pole's motor is stripped (``strip_actuators``) — only the cart's rail force is the action.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from hymeko_rl.env.arm_world import emit_arm_mjcf, strip_actuators
from hymeko_rl.hypergraph_state import HypergraphState

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_HYMEKO = _REPO / "data" / "robotics" / "inverted_pendulum.hymeko"


def emit_cartpole_mjcf(hymeko_path: str | Path = _DEFAULT_HYMEKO, *, name: str = "cartpole") -> str:
    """Emit the cart-pole MJCF once (one `hymeko` CLI call), to share across vectorized workers.

    # Postconditions returns the raw emitted scene; :class:`InvertedPendulumEnv` strips the pole motor.
    """
    return emit_arm_mjcf(hymeko_path, name=name)

# Per-vertex obs width: each link node carries its driving joint's (qpos, qvel). The MLP baseline
# flattens (N, 2) -> 2N, recovering the classic [x, x_dot, theta, theta_dot] cart-pole observation.
_NODE_FEAT = 2


class InvertedPendulumEnv(gym.Env[np.ndarray, np.ndarray]):
    """Cart-pole balancing on the HyMeKo-described inverted pendulum.

    # Preconditions
    ``angle_limit > 0``, ``cart_limit > 0``, ``force_mag > 0``, ``frame_skip >= 1``, ``max_steps >= 1``.
    # Invariants
    ``observation`` is ``(n_vertices, 2)`` float32 (per-vertex ``[qpos, qvel]``); ``action`` is ``(1,)``
    the cart force, clipped to ``[-force_mag, force_mag]``. The pole is passive (its motor is stripped).
    # Postconditions (``step``)
    ``terminated`` iff the pole leaves ``|theta| <= angle_limit`` or the cart leaves ``|x| <= cart_limit``;
    ``reward`` is ``+1`` while alive (the canonical InvertedPendulum objective).
    """

    metadata = {"render_modes": []}

    def __init__(self, *, hymeko_path: str | Path = _DEFAULT_HYMEKO, name: str = "cartpole",
                 mjcf: str | None = None,
                 angle_limit: float = 0.2, cart_limit: float = 1.0, force_mag: float = 10.0,
                 init_noise: float = 0.05, frame_skip: int = 20, max_steps: int = 200) -> None:
        super().__init__()
        if angle_limit <= 0 or cart_limit <= 0 or force_mag <= 0:
            raise ValueError("angle_limit/cart_limit/force_mag must be > 0")
        if frame_skip < 1 or max_steps < 1:
            raise ValueError("frame_skip/max_steps must be >= 1")
        # One source → scene + hypergraph. The pole motor is stripped → passive pole (under-actuated).
        # ``mjcf`` lets a caller pass a pre-emitted scene (emit once, share across N vectorized workers),
        # so building a fleet of envs does not pay N `hymeko` CLI subprocess calls.
        if mjcf is None:
            mjcf = emit_arm_mjcf(hymeko_path, name=name)
        mjcf = strip_actuators(mjcf, ["hinge"])
        self._mjcf = mjcf   # kept so the renderer can re-skin the scene (decorate_scene)
        self.model = mujoco.MjModel.from_xml_string(mjcf)
        self.data = mujoco.MjData(self.model)
        self.hg = HypergraphState.from_mjcf(mjcf, is_path=False)
        if int(self.model.nu) != 1:
            raise ValueError(f"expected a single (cart) actuator after stripping; got nu={self.model.nu}")
        # joint -> hypergraph vertex (its child body, remapped body b -> vertex b-1), and joint -> qpos
        # address, so node_features and the termination read the right DOFs regardless of body order.
        self._jnt_vtx = np.array(
            [int(self.model.jnt_bodyid[j]) - 1 for j in range(self.model.njnt)], dtype=np.int64)
        self._jnt_qadr = np.array(
            [int(self.model.jnt_qposadr[j]) for j in range(self.model.njnt)], dtype=np.int64)
        # the rail (cart) and hinge (pole) qpos addresses, by joint name (robust to ordering).
        self._cart_q = int(self.model.jnt_qposadr[self._jid("rail")])
        self._pole_q = int(self.model.jnt_qposadr[self._jid("hinge")])
        self.angle_limit = float(angle_limit)
        self.cart_limit = float(cart_limit)
        self.force_mag = float(force_mag)
        self.init_noise = float(init_noise)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self._step = 0
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.hg.n_vertices, _NODE_FEAT), dtype=np.float32)
        self.action_space = spaces.Box(-force_mag, force_mag, shape=(1,), dtype=np.float32)

    def _jid(self, joint: str) -> int:
        jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint))
        if jid < 0:
            raise ValueError(f"joint {joint!r} not in the emitted cart-pole model")
        return jid

    def node_features(self) -> np.ndarray:
        """Per-vertex ``(N, 2)`` obs on the cart-pole hypergraph: each link node carries its driving
        joint's ``(qpos, qvel)``. Flattened, this is the classic ``[x, x_dot, theta, theta_dot]`` obs —
        so the MLP baseline and the HSiKAN backbone read the same signal (a pure backbone swap)."""
        feat = np.zeros((self.hg.n_vertices, _NODE_FEAT), dtype=np.float32)
        for j in range(self.model.njnt):
            v = int(self._jnt_vtx[j])
            qadr = int(self._jnt_qadr[j])
            feat[v, 0] = self.data.qpos[qadr]
            feat[v, 1] = self.data.qvel[self.model.jnt_dofadr[j]]
        return feat

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.data.qpos[:] = self.np_random.uniform(-self.init_noise, self.init_noise, self.model.nq)
        self.data.qvel[:] = self.np_random.uniform(-self.init_noise, self.init_noise, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self.node_features(), {}

    def step(self, action: np.ndarray,
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ctrl = np.clip(np.asarray(action, dtype=np.float32).reshape(1), -self.force_mag, self.force_mag)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step += 1
        cart_x = float(self.data.qpos[self._cart_q])
        pole_th = float(self.data.qpos[self._pole_q])
        fell = abs(pole_th) > self.angle_limit
        out = abs(cart_x) > self.cart_limit
        terminated = bool(fell or out)
        truncated = self._step >= self.max_steps
        reward = 0.0 if terminated else 1.0     # +1 per upright step (canonical InvertedPendulum)
        return (self.node_features(), reward, terminated, truncated,
                {"cart_x": cart_x, "pole_angle": pole_th, "fell": fell, "out_of_bounds": out})
