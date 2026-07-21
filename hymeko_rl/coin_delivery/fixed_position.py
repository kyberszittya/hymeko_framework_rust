"""Fixed-initial-position Coin Delivery replay — deterministic problem replay + exact-state replay.

Two explicit replay modes over the canonical neutral-start Coin chain (approach → handoff → transport):

* **Deterministic problem replay** — a ``seed`` regenerates the exact deterministic problem (neutral arm pose, coin
  pose, target zone) through the env's own ``reset(seed)``; we read out and hash the canonical problem definition.
* **Exact state replay** — a checked :class:`CoinInitialState` JSON sets the MuJoCo physics state EXACTLY, so a
  replay is a pure function of the state (bit-identical where the physics + policy are deterministic).

Coordinate conventions (the planar Coin env, ``model.nq == 7``):

    qpos = [ j1_left, j2_left, j1_right, j2_right,  disk_x, disk_y, disk_rz ]
             └────────── arm (4 hinges) ──────────┘ └── planar coin (x, y, yaw) ──┘

The delivery **target** is the zone site ``(x, y)`` with half-width ``zone_half``; the control step is
``control_dt = frame_skip (20) * timestep (5e-4) = 0.01 s``. The coin is **planar** (two slides + one hinge), NOT a
6-DoF freejoint — a ``coin_quaternion`` must be a yaw-only rotation and the coin ``z`` must lie in the table plane, or
the replay fails loud (a non-planar request is rejected, never silently flattened).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# Canonical planar layout (verified against the built model, seed-independent).
ARM_NQ = 4                       # j1_left, j2_left, j1_right, j2_right
DISK_X_QADR = 4                  # disk_x, disk_y, disk_rz follow the 4 arm hinges
CONTROL_DT = 0.01                # frame_skip(20) * timestep(5e-4)
TABLE_Z = 0.02                   # the planar coin's fixed table-plane height (disk half-extent); z must match this
_QUAT_PLANAR_TOL = 1e-6          # a coin_quaternion must be yaw-only within this tolerance
_ZERO_TOL = 1e-9

_CKPT_MANIFEST = {               # the frozen deploy checkpoints the replay is allowed to load (sha256 prefixes)
    "e_approach": ("experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt", "7dbbf1a7782f"),
    "handoff_transport": ("experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt", "8955e8db8ac1"),
    "frozen_transport": ("experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt", "8bd73d8cbea0"),
}


class InvalidInitialState(ValueError):
    """Raised when a requested initial state is malformed, non-planar, in contact when neutral is required, has the
    coin overlapping the target, or otherwise physically invalid. Never silently repaired."""


@dataclass(frozen=True)
class CoinInitialState:
    """A fully-specified Coin initial condition (the exact-state replay schema).

    # Preconditions ``arm_qpos``/``arm_qvel`` have length 4; ``coin_position``/velocities length 3;
      ``coin_quaternion`` length 4; ``target_position`` length 3; all finite; the quaternion is yaw-only (planar);
      ``control_dt`` matches the env. # Invariants immutable; ``problem_hash`` is a pure function of the canonical
      fields.
    """
    embodiment: str = "POINT"
    arm_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    arm_qvel: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    coin_position: tuple[float, float, float] = (0.0, 0.0, TABLE_Z)
    coin_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)   # (w, x, y, z), yaw-only
    coin_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coin_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_position: tuple[float, float, float] = (0.0, 0.0, 0.004)
    zone_half: float = 0.04
    control_dt: float = CONTROL_DT
    provenance: dict[str, Any] = field(default_factory=dict)

    # ---- construction / (de)serialisation -------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CoinInitialState":
        known = set(cls.__dataclass_fields__)
        extra = set(d) - known
        if extra:
            raise InvalidInitialState(f"unknown initial-state keys {sorted(extra)}; known={sorted(known)}")
        _scalar = ("embodiment", "control_dt", "zone_half", "provenance")

        def _coerce(k: str, v: Any) -> Any:                         # array-like fields → float tuples; scalars as-is
            return v if k in _scalar or not isinstance(v, (list, tuple)) else tuple(float(x) for x in v)
        st = cls(**{k: _coerce(k, v) for k, v in d.items()})
        st.validate()
        return st

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    # ---- validation (fail loud, §2) -------------------------------------------------------------------------------
    def validate(self) -> None:
        """Reject a malformed / non-planar / non-finite state. # Errors :class:`InvalidInitialState`."""
        def _finite(name: str, v: Any, n: int) -> np.ndarray:
            a = np.asarray(v, dtype=np.float64)
            if a.shape != (n,):
                raise InvalidInitialState(f"{name} must have {n} dims, got shape {a.shape}")
            if not np.all(np.isfinite(a)):
                raise InvalidInitialState(f"{name} contains non-finite values: {a.tolist()}")
            return a
        _finite("arm_qpos", self.arm_qpos, ARM_NQ)
        _finite("arm_qvel", self.arm_qvel, ARM_NQ)
        _finite("coin_position", self.coin_position, 3)
        q = _finite("coin_quaternion", self.coin_quaternion, 4)
        _finite("coin_linear_velocity", self.coin_linear_velocity, 3)
        _finite("coin_angular_velocity", self.coin_angular_velocity, 3)
        _finite("target_position", self.target_position, 3)
        if not math.isfinite(self.control_dt) or self.control_dt <= 0:
            raise InvalidInitialState(f"control_dt must be finite and > 0, got {self.control_dt}")
        if abs(self.control_dt - CONTROL_DT) > 1e-9:
            raise InvalidInitialState(f"control_dt {self.control_dt} != env control_dt {CONTROL_DT} (schema mismatch)")
        # quaternion must be a unit yaw-only rotation (planar coin): axis component along z only
        nq = float(np.linalg.norm(q))
        if abs(nq - 1.0) > 1e-6:
            raise InvalidInitialState(f"coin_quaternion must be unit norm, got |q|={nq:.6f}")
        if abs(q[1]) > _QUAT_PLANAR_TOL or abs(q[2]) > _QUAT_PLANAR_TOL:
            raise InvalidInitialState(f"coin_quaternion must be yaw-only (x,y components ~0); got {q.tolist()} "
                                      f"(the planar coin cannot tip out of the table plane)")
        if abs(float(self.coin_position[2]) - TABLE_Z) > 1e-6:
            raise InvalidInitialState(f"coin z {self.coin_position[2]} must equal the table plane {TABLE_Z} "
                                      f"(the planar coin cannot leave the table)")
        # angular velocity must be about z only (planar)
        wv = np.asarray(self.coin_angular_velocity, dtype=np.float64)
        if abs(wv[0]) > _ZERO_TOL or abs(wv[1]) > _ZERO_TOL:
            raise InvalidInitialState(f"coin_angular_velocity must be about z only; got {wv.tolist()}")
        if abs(float(self.coin_linear_velocity[2])) > _ZERO_TOL:
            raise InvalidInitialState(f"coin_linear_velocity z must be 0 (planar); got {self.coin_linear_velocity}")

    # ---- derived planar quantities --------------------------------------------------------------------------------
    @property
    def coin_yaw(self) -> float:
        """The single planar rotation angle (about +z) encoded by ``coin_quaternion`` = (cos θ/2, 0, 0, sin θ/2)."""
        w, _x, _y, z = self.coin_quaternion
        return 2.0 * math.atan2(z, w)

    @property
    def coin_yaw_rate(self) -> float:
        return float(self.coin_angular_velocity[2])

    def qpos(self) -> np.ndarray:
        """The full 7-vector ``[arm(4), disk_x, disk_y, disk_rz]``."""
        return np.array([*self.arm_qpos, self.coin_position[0], self.coin_position[1], self.coin_yaw],
                        dtype=np.float64)

    def qvel(self) -> np.ndarray:
        return np.array([*self.arm_qvel, self.coin_linear_velocity[0], self.coin_linear_velocity[1],
                         self.coin_yaw_rate], dtype=np.float64)

    def target_xy(self) -> tuple[float, float]:
        return float(self.target_position[0]), float(self.target_position[1])

    def problem_hash(self) -> str:
        """A deterministic hash of the canonical problem definition (arm, coin, target, dt, embodiment)."""
        canon = {"embodiment": self.embodiment, "arm_qpos": [round(x, 9) for x in self.arm_qpos],
                 "arm_qvel": [round(x, 9) for x in self.arm_qvel],
                 "coin": [round(self.coin_position[0], 9), round(self.coin_position[1], 9), round(self.coin_yaw, 9)],
                 "coin_vel": [round(x, 9) for x in (*self.coin_linear_velocity[:2], self.coin_yaw_rate)],
                 "target": [round(self.target_xy()[0], 9), round(self.target_xy()[1], 9)],
                 "zone_half": round(self.zone_half, 9), "control_dt": round(self.control_dt, 9)}
        return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]


# ── env <-> state ────────────────────────────────────────────────────────────────────────────────────────────────────
def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def extract_initial_state(env: Any, cf: Any, *, embodiment: str = "POINT") -> CoinInitialState:
    """Read the current (already-reset) env into a :class:`CoinInitialState` — the canonical export of a problem.

    # Preconditions ``cf._env`` is a stepped planar Coin env. # Postconditions the returned state, re-applied via
    :func:`apply_initial_state`, reproduces the same qpos/qvel/zone (verified bit-identical in the replay tests).
    """
    inner = cf._env
    d = inner.data
    yaw = float(d.qpos[DISK_X_QADR + 2])
    st = CoinInitialState(
        embodiment=embodiment,
        arm_qpos=tuple(float(x) for x in d.qpos[:ARM_NQ]),
        arm_qvel=tuple(float(x) for x in d.qvel[:ARM_NQ]),
        coin_position=(float(d.qpos[DISK_X_QADR]), float(d.qpos[DISK_X_QADR + 1]), TABLE_Z),
        coin_quaternion=_yaw_to_quat(yaw),
        coin_linear_velocity=(float(d.qvel[DISK_X_QADR]), float(d.qvel[DISK_X_QADR + 1]), 0.0),
        coin_angular_velocity=(0.0, 0.0, float(d.qvel[DISK_X_QADR + 2])),
        target_position=(float(inner._zone_x), float(inner._zone_y), 0.004),
        zone_half=float(inner._zone_half),
        control_dt=CONTROL_DT,
    )
    st.validate()
    return st


def apply_initial_state(env: Any, cf: Any, state: CoinInitialState, *, seed_hint: int = 0) -> None:
    """Set the env EXACTLY to ``state`` (arm qpos/qvel, planar coin pose/vel, target zone), in place.

    Drives the env's OWN full reset path with the coin/zone **sampling pinned** to the state's values, so the whole
    stack (outer neutral env → ContactFormationEnv bookkeeping → planar inner) initialises consistently around the
    requested coin and target — then overrides the remaining DoF the neutral reset zeroed (arm pose/velocity, coin
    yaw/velocity, zone half-width). This makes the replay a self-consistent function of the state (no stale
    seed-``seed_hint`` bookkeeping leaks into the transport phase). ``seed_hint`` only seeds the outer RNG, which the
    seeded-eval path does not consult once the coin/zone are pinned — so the result is independent of ``seed_hint``.

    # Preconditions ``state`` is valid. # Postconditions ``cf._env`` is at ``state``; the next rollout evolves from
    exactly here, reproducing the seed run bit-for-bit when ``state`` was extracted from that seed.
    """
    import mujoco
    state.validate()
    inner = cf._env
    cx, cy = float(state.coin_position[0]), float(state.coin_position[1])
    tx, ty = state.target_xy()
    orig_coin, orig_zone = inner._sample_coin_xy, inner._sample_zone_xy
    inner._sample_coin_xy = lambda: (cx, cy)                     # pin the sampled coin/zone so the normal reset path
    inner._sample_zone_xy = lambda: (tx, ty)                     # builds a consistent outer/cf/inner state around them
    try:
        env.set_stage(0)
        env.reset(seed=seed_hint)
    finally:
        inner._sample_coin_xy, inner._sample_zone_xy = orig_coin, orig_zone
    inner.data.qpos[:ARM_NQ] = state.arm_qpos                    # override the DoF a neutral reset zeroed
    inner.data.qpos[DISK_X_QADR + 2] = state.coin_yaw
    inner.data.qvel[:] = state.qvel()
    inner._zone_half = float(state.zone_half)
    if inner._zone_site >= 0:
        inner.model.site_pos[inner._zone_site] = [tx, ty, 0.004]
    mujoco.mj_forward(inner.model, inner.data)
    inner._planar_metrics = inner._metrics()


# ── fingerprints / hashes ────────────────────────────────────────────────────────────────────────────────────────────
def env_fingerprint(cf: Any) -> dict[str, Any]:
    """A deterministic fingerprint of the built env geometry (model nq/nu, geom sizes, obs/action dims)."""
    inner = cf._env
    return {"model_nq": int(inner.model.nq), "model_nu": int(inner.model.nu),
            "obs_space": list(inner.observation_space.shape), "action_space": list(inner.action_space.shape),
            "geom_fp": hashlib.sha256(np.ascontiguousarray(inner.model.geom_size).tobytes()).hexdigest()[:12]}


def verify_checkpoint_hashes() -> dict[str, str]:
    """Verify every deploy checkpoint against the pinned manifest; fail loud on any mismatch (§2)."""
    import os
    got = {}
    for name, (path, prefix) in _CKPT_MANIFEST.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint missing for {name}: {path} (external artifact; see manifest)")
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if not h.startswith(prefix):
            raise InvalidInitialState(f"checkpoint hash mismatch for {name}: {path} is {h[:12]}, expected {prefix}")
        got[name] = h[:12]
    return got
