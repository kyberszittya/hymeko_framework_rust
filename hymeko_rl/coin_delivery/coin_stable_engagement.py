"""STABLE_OBJECT_ENGAGEMENT_V1 — the hybrid deployable phase gate that replaces the rejected premature-activation
predicate (``coin_phase_gate.PhaseGate``, contract ``d739e8af``, which armed on unilateral acquisition brushes on
11/14 trajectories).

Two arm paths, disarm/hysteresis shared:

- **BILATERAL fast path** — arm on ``left_contact AND right_contact`` held ``bilateral_arm_after`` consecutive steps
  (grasp-style transport).
- **UNILATERAL co-motion slow path** — arm on the SAME contacting side held ``uni_arm_after`` consecutive steps AND
  the coin demonstrably co-moving with that fingertip over a trailing kinematic window (push-valid transport, e.g.
  seed 1447 which never forms a bilateral grasp).

The gate generates NO actions — it returns a multiplier ``g_t ∈ {0, 1}`` scaling the residual, so ``g=0`` reproduces
the frozen base exactly. It has internal memory (counters + kinematic history) → the full controller state is
:class:`EngagementState` (serialized as ``PHASE_GATE_CONTROLLER_STATE_V2``).

Deployable signals only (§3): left/right robot-attributed fingertip contact; coin position (canonical observation);
fingertip position (FK from measured joint state — MuJoCo site ``tip_left``/``tip_right``); short causal history.
Finite-difference coin/tip motion is causal from those. It uses NO ``disk_to_zone``, target, success, seed, trajectory
id, planner state, future obs, or hidden state. Thresholds are derived from geometry/jitter, NOT rollout success (§4).
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

# geometry constants (canonical coin scenario): disk radius 0.02 m; sim numerical jitter ~0.
_DISK_RADIUS = 0.02


class EngagementMode(str, Enum):
    EARLY_CONTROL = "EARLY_CONTROL"
    LATE_CONTROL_ARMED = "LATE_CONTROL_ARMED"
    REACQUIRE = "REACQUIRE"
    TERMINAL = "TERMINAL"


class ArmMechanism(str, Enum):
    BILATERAL_FAST = "BILATERAL_FAST"
    UNILATERAL_COMOTION = "UNILATERAL_COMOTION"


@dataclass(frozen=True)
class StableEngagementConfig:
    """Hybrid gate thresholds. Geometry/jitter-derived (§4), NOT tuned on rollout success. Frozen → new SHA on change.

    - ``coin_motion_floor`` = disk_radius × 0.025 = 5e-4 m: coin must move beyond numerical/sensor jitter in the window.
    - ``comotion_dot_min`` = cos(45°) ≈ 0.707: coin displacement within 45° of the fingertip displacement.
    - ``slip_bound`` = disk_radius / 2 = 0.01 m: relative slip over the window bounded by half a coin radius.
    """
    bilateral_arm_after: int = 3
    uni_arm_after: int = 6
    disarm_after: int = 2
    kin_window: int = 4
    coin_motion_floor: float = _DISK_RADIUS * 0.025     # 5e-4 m
    comotion_dot_min: float = 0.707                      # cos 45°
    slip_bound: float = _DISK_RADIUS * 0.5              # 0.01 m
    threshold_provenance: str = ("floor=disk_r*0.025; dot_min=cos45; slip=disk_r/2; disk_r=0.02m; sim jitter~0; "
                                 "NOT tuned on headline/validation success")

    def __post_init__(self) -> None:
        if self.uni_arm_after <= self.bilateral_arm_after:
            raise ValueError("uni_arm_after must exceed bilateral_arm_after (slow path is stricter)")
        if self.kin_window < 1 or self.disarm_after < 1:
            raise ValueError("kin_window and disarm_after must be >= 1")


@dataclass
class EngagementState:
    """PHASE_GATE_CONTROLLER_STATE_V2 — the complete causal memory needed to reproduce the hybrid gate transition."""
    mode: str = EngagementMode.EARLY_CONTROL.value
    bilateral_counter: int = 0
    uni_counter: int = 0
    uni_side: "str | None" = None
    loss_counter: int = 0
    coin_hist: list = field(default_factory=list)         # trailing coin xy, newest last (len <= kin_window+1)
    ltip_hist: list = field(default_factory=list)         # trailing left-tip xy
    rtip_hist: list = field(default_factory=list)         # trailing right-tip xy
    last_arm_mechanism: "str | None" = None
    comotion_ok: bool = False                             # last computed co-motion qualification flag


class StableEngagementGate:
    """Hybrid bilateral/unilateral-co-motion gate. :meth:`update` advances one step and returns ``(g_t, mechanism)``.

    # Preconditions: ``coin_xy``/``ltip_xy``/``rtip_xy`` are deployable planar positions (m); ``lc``/``rc`` the
      per-side robot-attributed contact booleans for the step just executed. # Postconditions: returns ``g_t = 1.0``
      iff in ``LATE_CONTROL_ARMED`` after this step, else 0.0, and the mechanism that armed it (or None).
      # Invariants: never produces an action; co-motion is required only to ARM, never to STAY armed (§5 — settling
      must not disarm when the coin correctly stops).
    """

    def __init__(self, config: StableEngagementConfig | None = None) -> None:
        self.cfg = config or StableEngagementConfig()
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        self.s = EngagementState()
        self._coin = deque(maxlen=c.kin_window + 1)
        self._ltip = deque(maxlen=c.kin_window + 1)
        self._rtip = deque(maxlen=c.kin_window + 1)
        self.arm_count = 0
        self.disarm_count = 0
        self.steps = 0

    # ---- co-motion test (§4) ----
    def _comotion_ok(self, side: str) -> bool:
        c = self.cfg
        if len(self._coin) < c.kin_window + 1:
            return False
        dcoin = np.asarray(self._coin[-1]) - np.asarray(self._coin[0])
        tip = self._ltip if side == "L" else self._rtip
        dtip = np.asarray(tip[-1]) - np.asarray(tip[0])
        ncoin = float(np.linalg.norm(dcoin))
        if ncoin < c.coin_motion_floor:                 # coin must move beyond jitter
            return False
        ntip = float(np.linalg.norm(dtip))
        dot = float(dcoin @ dtip) / (ncoin * ntip + 1e-12)
        if dot < c.comotion_dot_min:                    # directional agreement
            return False
        if float(np.linalg.norm(dcoin - dtip)) > c.slip_bound:   # bounded relative slip
            return False
        return True

    def update(self, lc: bool, rc: bool, coin_xy, ltip_xy, rtip_xy, terminated: bool = False):
        c = self.cfg
        self.steps += 1
        self._coin.append(np.asarray(coin_xy, float)[:2].copy())
        self._ltip.append(np.asarray(ltip_xy, float)[:2].copy())
        self._rtip.append(np.asarray(rtip_xy, float)[:2].copy())
        if terminated:
            self.s.mode = EngagementMode.TERMINAL.value
            self._sync(); return 0.0, None
        if self.s.mode == EngagementMode.TERMINAL.value:
            return 0.0, None

        bilateral = lc and rc
        self.s.bilateral_counter = self.s.bilateral_counter + 1 if bilateral else 0
        side = "L" if (lc and not rc) else ("R" if (rc and not lc) else None)
        if side is not None and side == self.s.uni_side:
            self.s.uni_counter += 1
        elif side is not None:
            self.s.uni_side = side; self.s.uni_counter = 1     # side change resets (no alternating accumulation)
        else:
            self.s.uni_side = None; self.s.uni_counter = 0
        self.s.loss_counter = self.s.loss_counter + 1 if not (lc or rc) else 0

        mechanism = None
        if self.s.mode in (EngagementMode.EARLY_CONTROL.value, EngagementMode.REACQUIRE.value):
            if self.s.bilateral_counter >= c.bilateral_arm_after:
                mechanism = ArmMechanism.BILATERAL_FAST.value
            elif (self.s.uni_side is not None and self.s.uni_counter >= c.uni_arm_after
                  and self._comotion_ok(self.s.uni_side)):
                mechanism = ArmMechanism.UNILATERAL_COMOTION.value
            if mechanism is not None:
                self.s.mode = EngagementMode.LATE_CONTROL_ARMED.value
                self.s.last_arm_mechanism = mechanism
                self.s.loss_counter = 0
                self.arm_count += 1
        elif self.s.mode == EngagementMode.LATE_CONTROL_ARMED.value:
            if self.s.loss_counter >= c.disarm_after:           # co-motion NOT required to STAY armed (§5)
                self.s.mode = EngagementMode.REACQUIRE.value
                self.s.bilateral_counter = 0; self.s.uni_counter = 0; self.s.uni_side = None
                self.disarm_count += 1
        self.s.comotion_ok = (self.s.uni_side is not None and self._comotion_ok(self.s.uni_side))
        self._sync()
        return (1.0 if self.s.mode == EngagementMode.LATE_CONTROL_ARMED.value else 0.0), mechanism

    def _sync(self) -> None:
        self.s.coin_hist = [x.tolist() for x in self._coin]
        self.s.ltip_hist = [x.tolist() for x in self._ltip]
        self.s.rtip_hist = [x.tolist() for x in self._rtip]

    @property
    def gate(self) -> float:
        return 1.0 if self.s.mode == EngagementMode.LATE_CONTROL_ARMED.value else 0.0

    # ---- PHASE_GATE_CONTROLLER_STATE_V2 (§6) ----
    def state_v2(self) -> dict:
        return asdict(self.s)

    def load_state_v2(self, st: dict) -> None:
        self.s = EngagementState(**st)
        self._coin = deque([np.asarray(x, float) for x in self.s.coin_hist], maxlen=self.cfg.kin_window + 1)
        self._ltip = deque([np.asarray(x, float) for x in self.s.ltip_hist], maxlen=self.cfg.kin_window + 1)
        self._rtip = deque([np.asarray(x, float) for x in self.s.rtip_hist], maxlen=self.cfg.kin_window + 1)

    def contract_v2(self) -> dict:
        return {"gate": "coin_stable_engagement.StableEngagementGate", "schema": "PHASE_GATE_CONTROLLER_STATE_V2",
                "config": asdict(self.cfg),
                "state_fields": ["mode", "bilateral_counter", "uni_counter", "uni_side", "loss_counter",
                                 "coin_hist", "ltip_hist", "rtip_hist", "last_arm_mechanism", "comotion_ok"],
                "arm_paths": {"BILATERAL_FAST": "left AND right >= bilateral_arm_after",
                              "UNILATERAL_COMOTION": "same side >= uni_arm_after AND co-motion(coin,tip) qualifies"},
                "disarm": "complete contact loss >= disarm_after -> REACQUIRE (co-motion NOT required to stay armed)",
                "deployable_signals": ["left_contact", "right_contact", "coin_xy", "tip_xy(FK)", "causal_history"],
                "forbidden_inputs": ["disk_to_zone", "target", "success", "seed", "trajectory_id", "planner_state",
                                     "future_obs", "hidden_mujoco_state"]}

    def contract_v2_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.contract_v2(), sort_keys=True).encode()).hexdigest()


def stable_engagement_signals(inner):
    """Extract the deployable hybrid-gate signals from the coin inner env: per-side contact, coin xy, and left/right
    fingertip xy via FK (MuJoCo site ``tip_left``/``tip_right`` = ``inner._tip_sites``). Copies the live views."""
    m = inner._planar_metrics
    lc, rc = bool(m.left_contact), bool(m.right_contact)
    coin = np.array(m.disk_pos[:2], float)
    ltip = np.array(inner.data.site_xpos[inner._tip_sites[0]][:2], float)
    rtip = np.array(inner.data.site_xpos[inner._tip_sites[1]][:2], float)
    return lc, rc, coin, ltip, rtip
