"""K0 — the KINETIC snapshot / replanning contract for the first learned s1 delivery.

The mechanical picture is settled (see `reports/2026-07-27-coin-release-guard-g0.md` and the local-atlas scout): the frozen
`APPROACH_MOMENTUM_BUILD` reaches teacher-close momentum, the `VELOCITY_FLOOR/KINETIC` mode transports the coin *while
moving* (no stiction stall), and the θ-primitive teacher is a strict-K6 positive control on s1. The single missing skill is a
learned, causal KINETIC coin-following torque policy. This module is the CONTRACT the learning stack (K1–K4) is built on — it
does NOT train anything. It provides, with no state edits and no oracle-in-deployment:

  * ``TransportSnapshot`` — a duck-typed, deterministically-branchable snapshot of an ARBITRARY mid-transport state (the θ
    teacher's ``CradleSnapshot`` interface without the B_v straddle gate), so the proven teacher can replan from a moving,
    lightly-gripped coin.
  * ``roll_until`` — a higher-order capture loop that mirrors the frozen ``velocity_rollout`` step kernel EXACTLY (so its
    branches are bit-identical) but exposes the mid-loop ``prev_tau`` the frame-hook cannot see.
  * ``freeze_kinetic_entry`` — the deterministic, content-hashed frozen KINETIC-entry state on dev s1.
  * ``kinetic_observe`` — the SINGLE canonical policy-input extractor, used identically by the offline bank builder (batch)
    and the deployed controller (streaming). Reads only the live state + short causal history — no future / teacher / K6 leak.
  * ``receding_horizon_relabel`` — the receding-horizon teacher relabel: replan from a transport snapshot, export ONLY the
    first causal torque action as a label, and report the strict-K6 verdict. The teacher's lookahead is baked into the label,
    never into the policy input.

Nothing here alters physics, the motion contract, the K6 monitor, or the split. Torque only ever enters through the frozen
governed ``step_ablation`` + ``govern_torque`` stack. The teacher is used OFFLINE to label; deployment (K2+) is the learned
clone only.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.contact_velocity import geom_planar_velocity, primary_fingertip_contacts, state_hash
from hymeko_rl.coin_delivery.forward_displacement import (
    ForwardConfig, _coin_xy, _schedule_increment, cem_search, delivery_success, rollout_primitive)
from hymeko_rl.coin_delivery.mobile_conditioning import _fingertip_geoms
from hymeko_rl.coin_delivery.theta_option.hybrid_approach import (
    KINETIC, ApproachParams, HybridApproachController)
from hymeko_rl.coin_delivery.theta_option.residual_adapter import _coin_spin
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.env.motion_contract import govern_torque

# dev s1 anchor (frozen seed) + its scout-certified canonical delivering θ (basin centre for warm-started replanning).
S1_SEED = 14250
S1_CANONICAL_THETA = (0.0515, 0.2179, 0.0164, 12.8078, 9.2369, 3.0388)
_HIST = 3                               # short causal response-history depth (a momentary state may hide a sliding coin)

# canonical policy-input schema (frozen order); the leak-guard test asserts none names a future / teacher / K6 quantity.
FEATURE_NAMES: tuple[str, ...] = (
    "dtz", "v_par", "v_lat", "spin", "fn_l", "fn_r",
    "relpos_par_l", "relpos_lat_l", "relvel_par_l", "relvel_lat_l", "slip_l",
    "relpos_par_r", "relpos_lat_r", "relvel_par_r", "relvel_lat_r", "slip_r",
    "q0", "q1", "q2", "q3", "qd0", "qd1", "qd2", "qd3",
    "prev0", "prev1", "prev2", "prev3", "dist_to_corridor",
) + tuple(f"hist{k}_{n}" for k in range(_HIST) for n in ("dtz", "v_par", "fn_l", "fn_r"))
RELEASE_CORRIDOR_HI = ApproachParams().kinetic_release_hi    # 0.026 m — the close-and-moving release corridor upper edge


class KineticEntryError(RuntimeError):
    """The frozen KINETIC-entry state could not be produced (no certified s1 snapshot, or the controller never entered the
    KINETIC phase). Halt — do NOT fabricate an entry state."""


# --------------------------------------------------------------------------------------------------------------------
# TransportSnapshot — an arbitrary mid-transport state, branchable exactly like a CradleSnapshot (no straddle gate)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class TransportAdmissibility:
    """A QUERY over a transport state (never a silent raise): is it a teacher-replannable dual-contact straddle within the
    motion contract? # Invariants: read-only; computed on a fresh branch, the stored env is never stepped in place."""

    dual_contact: bool
    straddle: bool
    straddle_dot: float
    fn_min: float
    qdot_max: float
    motion_ok: bool

    @property
    def admissible(self) -> bool:
        """Replannable iff both tips carry the coin in a straddle within the joint-speed hard cap."""
        return bool(self.dual_contact and self.straddle and self.motion_ok)


@dataclass
class TransportSnapshot:
    """A frozen mid-transport state with EXACT deepcopy branching. Duck-types the subset of ``CradleSnapshot`` the θ teacher
    (``rollout_primitive`` / ``passive_drift`` / ``cem_search``) reads — ``branch``/``stack``/``prev_tau``/``q_hold``/
    ``lo``/``hi`` — but is built from ANY live mid-transport ``rl`` (no B_v ``_assert_admissible_frames`` gate; admissibility
    is a query). # Preconditions: ``rl`` a deepcopy owned by this snapshot; ``prev_tau`` the exact rate-limited torque applied
    at capture. # Postconditions: ``branch()`` is bit-identical across calls (deterministic deepcopy); the stored env is never
    stepped in place. # Invariants: every teacher rollout restores this identical state."""

    _rl: Any
    stack: Any
    prev_tau: np.ndarray
    q_hold: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    joint_vel_hard: float = 3.0                         # motion-contract joint-speed hard cap (admissibility gate)

    def branch(self) -> Any:
        """A fresh deepcopy restoring the identical snapshot state (MuJoCo data + warm-start + contacts + pin)."""
        return copy.deepcopy(self._rl)

    def admissibility(self) -> TransportAdmissibility:
        """Query the branchable state for teacher-replannability (dual contact ∧ straddle ∧ motion-contract). # Postconditions:
        pure — reads a fresh branch, mutates nothing."""
        rl = self.branch()
        con = primary_fingertip_contacts(rl)
        both = con["left"] is not None and con["right"] is not None
        straddle_dot = float(con["left"]["n"] @ con["right"]["n"]) if both else 1.0
        fn_min = min(float(con["left"]["fn"]) if con["left"] else 0.0,
                     float(con["right"]["fn"]) if con["right"] else 0.0)
        qdot = float(np.max(np.abs(rl.inner.data.qvel[:4])))
        return TransportAdmissibility(dual_contact=both, straddle=bool(straddle_dot < 0.0),
                                      straddle_dot=round(straddle_dot, 4), fn_min=round(fn_min, 4),
                                      qdot_max=round(qdot, 4), motion_ok=bool(qdot <= self.joint_vel_hard))

    @classmethod
    def from_live(cls, rl: Any, stack: Any, prev_tau: np.ndarray, *, joint_vel_hard: float = 3.0) -> "TransportSnapshot":
        """Own a deepcopy of ``rl`` and capture the actuator ctrl range + a hold-in-place servo target. # Preconditions:
        ``rl`` a live dual-contact mid-transport env; ``prev_tau`` its last rate-limited applied torque."""
        owned = copy.deepcopy(rl)
        m = owned.inner.model
        return cls(_rl=owned, stack=stack, prev_tau=np.asarray(prev_tau, np.float64).copy(),
                   q_hold=owned.inner.data.qpos[:4].astype(np.float64).copy(),
                   lo=m.actuator_ctrlrange[:4, 0].copy(), hi=m.actuator_ctrlrange[:4, 1].copy(),
                   joint_vel_hard=float(joint_vel_hard))


# --------------------------------------------------------------------------------------------------------------------
# roll_until — higher-order capture loop mirroring the frozen velocity_rollout step kernel (bit-identical branches)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Capture:
    """A captured branch point from ``roll_until``. # Invariants: ``prev_tau`` is the exact rate-limited torque applied at the
    capture step (the mid-loop value the frame-hook cannot observe); ``rl`` is a deepcopy at the END of that step."""

    rl: Any
    prev_tau: np.ndarray
    t: int
    fired: bool
    coin_trace: list = field(default_factory=list)


def roll_until(snap: Any, controller: Any, cfg: ForwardConfig,
               *, stop_when: Callable[[Any, Any, int], bool]) -> Capture:
    """Run the velocity-servo controller through the FROZEN governed physics (identical step kernel to ``velocity_rollout``:
    ``dtau = controller.dtau_for_step`` → ``prev_tau = clip(prev_tau + dtau, lo, hi)`` → governed ``step_ablation``), stopping
    at the first step ``stop_when(controller, rl, t)`` fires (or the horizon). Captures the exact ``prev_tau`` and a deepcopy
    of ``rl`` at that step. # Preconditions: ``snap`` exposes ``branch``/``prev_tau``/``lo``/``hi``/``stack.gov``.
    # Postconditions: no pin/teleport/coin edit; a full horizon with a never-firing predicate reproduces ``velocity_rollout``'s
    ``coin_trace`` bit-for-bit (tested). # Invariants: the global control callback is always cleared."""
    controller.reset()
    rl = snap.branch()
    prev_tau = np.asarray(snap.prev_tau, np.float64).copy()
    trace: list = []

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    try:
        for t in range(1, cfg.horizon + 1):
            dtau = controller.dtau_for_step(rl, t, prev_tau)
            prev_tau = np.clip(prev_tau + dtau, snap.lo, snap.hi)
            step_ablation(rl, np.asarray(prev_tau, np.float32), "A")
            trace.append(_coin_xy(rl).copy())
            if stop_when(controller, rl, t):
                return Capture(copy.deepcopy(rl), prev_tau.copy(), int(t), True, [c.tolist() for c in trace])
    finally:
        mujoco.set_mjcb_control(None)
    return Capture(copy.deepcopy(rl), prev_tau.copy(), int(cfg.horizon), False, [c.tolist() for c in trace])


# --------------------------------------------------------------------------------------------------------------------
# freeze_kinetic_entry — the deterministic, content-hashed frozen KINETIC-entry state on dev s1
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class KineticEntry:
    """The frozen KINETIC-entry state: the first step at which the frozen ``HybridApproachController`` hands APPROACH → KINETIC
    on the given cradle. # Invariants: same (seed, approach params, cfg) ⇒ bit-identical ``state_hash``."""

    tsnap: TransportSnapshot
    seed: int
    entry_step: int
    state_hash: str
    entry_dtz: float
    entry_v_par: float
    admissibility: TransportAdmissibility
    source_snap: Any = None                             # the parent CradleSnapshot (for the bank builder / positive control)

    def summary(self) -> dict:
        return {"seed": self.seed, "entry_step": self.entry_step, "state_hash": self.state_hash,
                "entry_dtz_mm": round(self.entry_dtz * 1000, 2), "entry_v_par": round(self.entry_v_par, 4),
                "admissible": self.admissibility.admissible, "straddle_dot": self.admissibility.straddle_dot,
                "fn_min": self.admissibility.fn_min, "qdot_max": self.admissibility.qdot_max}


def freeze_kinetic_entry(source_snap: Any, *, seed: int = S1_SEED, approach: ApproachParams | None = None,
                         cfg: ForwardConfig = DELIVERY_CFG) -> KineticEntry:
    """Roll the frozen ``HybridApproachController(kinetic_transport=True)`` from ``source_snap`` until it enters the KINETIC
    phase, and freeze that state as a branchable ``TransportSnapshot`` with a content hash. # Preconditions: ``source_snap`` a
    certified s1 ``CradleSnapshot``. # Postconditions: deterministic (same inputs ⇒ same hash). # Raises: ``KineticEntryError``
    if the controller never enters KINETIC within the horizon."""
    ap = approach if approach is not None else ApproachParams(kinetic_transport=True)
    if not ap.kinetic_transport:
        raise KineticEntryError("approach params must have kinetic_transport=True to reach the KINETIC entry")
    controller = HybridApproachController(source_snap, approach=ap)
    cap = roll_until(source_snap, controller, cfg, stop_when=lambda c, rl, t: c.phase == KINETIC)
    if not cap.fired:
        raise KineticEntryError(f"controller never entered KINETIC within horizon {cfg.horizon} (seed {seed})")
    tsnap = TransportSnapshot.from_live(cap.rl, source_snap.stack, cap.prev_tau)
    rl = cap.rl
    _u, dtz = rl.inner.direction_to_zone()
    vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    v_par = float(np.dot(vel, np.asarray(_u, np.float64)[:2]))
    return KineticEntry(tsnap=tsnap, seed=seed, entry_step=cap.t, state_hash=state_hash(rl),
                        entry_dtz=float(dtz), entry_v_par=v_par, admissibility=tsnap.admissibility(),
                        source_snap=source_snap)


# --------------------------------------------------------------------------------------------------------------------
# kinetic_observe — the SINGLE canonical policy-input extractor (batch == streaming; no future / teacher / K6 leak)
# --------------------------------------------------------------------------------------------------------------------
def _side_relatives(rl: Any, gid: int, coin_xy: np.ndarray, coin_vel: np.ndarray, e_par: np.ndarray, e_lat: np.ndarray,
                    con_side: "dict | None") -> list[float]:
    """Per-side tip→coin relative position/velocity in the (e_par, e_lat) target frame + tangential slip at the contact.
    # Postconditions: length-5 [relpos_par, relpos_lat, relvel_par, relvel_lat, slip]; slip is 0 when that tip is not in
    contact (no contact frame to resolve the tangential rate against)."""
    d = rl.inner.data
    tip_xy = np.asarray(d.geom_xpos[int(gid)][:2], np.float64)
    tip_vel = geom_planar_velocity(rl.inner.model, d, int(gid))
    relpos = tip_xy - coin_xy
    relvel = tip_vel - coin_vel
    slip = float(np.asarray(con_side["t"], np.float64) @ relvel) if con_side is not None else 0.0
    return [float(relpos @ e_par), float(relpos @ e_lat), float(relvel @ e_par), float(relvel @ e_lat), slip]


def kinetic_observe(rl: Any, hist: list) -> "tuple[np.ndarray, list]":
    """The canonical KINETIC policy input: coin target-frame kinematics (dtz, v_par, v_lat, spin), per-side contact force,
    per-side tip→coin relative pos/vel + slip, arm q/qdot, previous applied torque, distance-to-release-corridor, and the last
    ``_HIST`` compact (dtz, v_par, fn_l, fn_r) frames. Used IDENTICALLY by the batch bank builder and the streaming deployed
    controller (the ``batch == streaming`` invariant). # Preconditions: live ``rl``; ``hist`` the running compact-frame buffer
    (newest last). # Postconditions: ``feat`` has ``len(FEATURE_NAMES)`` entries in the frozen order; ``hframe`` is the compact
    frame to append; NO future / teacher / K6 information is read (the function has no access to any of it)."""
    u, dtz = rl.inner.direction_to_zone()
    e_par = np.asarray(u, np.float64)[:2]
    n = float(np.linalg.norm(e_par))
    e_par = e_par / n if n > 1e-9 else np.array([1.0, 0.0])
    e_lat = np.array([-e_par[1], e_par[0]], np.float64)
    coin_xy = _coin_xy(rl)
    coin_vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    con = primary_fingertip_contacts(rl)
    fnl = float(con["left"]["fn"]) if con["left"] is not None else 0.0
    fnr = float(con["right"]["fn"]) if con["right"] is not None else 0.0
    gl, gr = _fingertip_geoms(rl.inner.model)
    cur = [float(dtz), float(coin_vel @ e_par), float(coin_vel @ e_lat), _coin_spin(rl), fnl, fnr]
    left = _side_relatives(rl, gl, coin_xy, coin_vel, e_par, e_lat, con["left"])
    right = _side_relatives(rl, gr, coin_xy, coin_vel, e_par, e_lat, con["right"])
    q = list(np.asarray(rl.inner.data.qpos[:4], np.float64))
    qd = list(np.asarray(rl.inner.data.qvel[:4], np.float64))
    pv = list(np.asarray(rl.inner.data.ctrl[:4], np.float64))
    dist_corridor = [float(dtz) - RELEASE_CORRIDOR_HI]
    hframe = [float(dtz), float(coin_vel @ e_par), fnl, fnr]
    h: list = []
    for k in range(_HIST):
        h += hist[-1 - k] if len(hist) > k else hframe
    feat = np.array(cur + left + right + q + qd + pv + dist_corridor + h, np.float64)
    assert feat.shape[0] == len(FEATURE_NAMES), f"feature dim {feat.shape[0]} != schema {len(FEATURE_NAMES)}"
    return feat, hframe


# --------------------------------------------------------------------------------------------------------------------
# receding_horizon_relabel — first-action teacher label from a transport snapshot (offline; no oracle in deployment)
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class RelabelResult:
    """A receding-horizon teacher relabel. # Invariants: ``first_action`` (and its normalised form) is the ONLY label; ``theta``
    / ``metrics`` are provenance (the teacher's lookahead), never fed to the policy input."""

    first_action: np.ndarray                 # length-4 Δτ — the sole causal label
    first_action_norm: np.ndarray            # first_action / step ∈ [-1, 1] (tanh-actor convention)
    delivers_k6: bool
    admissible: bool
    theta: list
    source: str                              # "warm_theta" | "local_cem" (provenance)
    metrics: dict = field(default_factory=dict)


def _windowed_delivery_cfg(warm_theta: Any, window: Any, base: ForwardConfig, *, pop: int, iters: int) -> ForwardConfig:
    """A ``ForwardConfig`` whose θ box is a small window around ``warm_theta`` (clamped to the base box) so ``cem_search``'s
    box-centre mean warm-starts at ``warm_theta`` — a local receding-horizon replan around the known-delivering basin."""
    w = np.asarray(warm_theta, np.float64)
    win = np.asarray(window, np.float64)
    lo = np.maximum(w - win, np.asarray(base.lo, np.float64))
    hi = np.minimum(w + win, np.asarray(base.hi, np.float64))
    return replace(base, lo=tuple(lo), hi=tuple(hi), pop=int(pop), iters=int(iters),
                   elite=max(2, int(pop) // 4), init_std=tuple(0.5 * (hi - lo) + 1e-6))


def _first_action(tsnap: TransportSnapshot, theta: Any, cfg: ForwardConfig) -> np.ndarray:
    """The teacher's FIRST slew-admissible Δτ from ``tsnap`` under ``theta`` — the sole exported label (t=1 of the replan,
    built from the live geometry via the frozen ``_schedule_increment``). # Postconditions: length-4, slew-clipped."""
    rl = tsnap.branch()
    e_par = np.asarray(rl.inner.direction_to_zone()[0], np.float64)
    step = float(tsnap.stack.tau_rate * tsnap.stack.control_dt)
    return np.asarray(_schedule_increment(rl, np.asarray(theta, np.float64), 1, e_par, step, _coin_xy(rl)), np.float64)


def receding_horizon_relabel(tsnap: TransportSnapshot, *, warm_theta: Any = S1_CANONICAL_THETA, budget: int = 0,
                             cfg: ForwardConfig = DELIVERY_CFG, window: Any = None,
                             pop: int = 16, iters: int = 3) -> RelabelResult:
    """Replan the proven θ-primitive teacher from ``tsnap`` and export ONLY the first causal torque action as a label, with the
    strict-K6 verdict. ``budget<=0`` executes ``warm_theta`` directly (cheap positive-control probe); ``budget>0`` runs a small
    warm-started local CEM around ``warm_theta`` (the receding-horizon replan). Deterministic given (tsnap, warm_theta, cfg,
    seeds). # Preconditions: ``tsnap`` a dual-contact transport snapshot. # Postconditions: the teacher's future never enters
    the label consumer's policy input; no state edit / pin / hidden force."""
    adm = tsnap.admissibility()
    step = float(tsnap.stack.tau_rate * tsnap.stack.control_dt)
    # TransportSnapshot structurally duck-types the CradleSnapshot rollout interface the teacher reads
    # (branch/stack/prev_tau/lo/hi/q_hold); the type: ignore[arg-type] is scoped to these three teacher calls only.
    if budget <= 0:
        theta = tuple(float(x) for x in warm_theta)
        metrics = rollout_primitive(tsnap, theta, cfg)                     # type: ignore[arg-type]
        source = "warm_theta"
    else:
        win = window if window is not None else (0.03, 0.05, 0.03, 3.0, 3.0, 0.6)
        res = cem_search(tsnap, _windowed_delivery_cfg(warm_theta, win, cfg, pop=pop, iters=iters))  # type: ignore[arg-type]
        theta = tuple(float(x) for x in (res["best_theta"] or list(warm_theta)))
        metrics = res["best_metrics"] or rollout_primitive(tsnap, theta, cfg)   # type: ignore[arg-type]
        source = "local_cem"
    first = _first_action(tsnap, theta, cfg)
    return RelabelResult(first_action=first, first_action_norm=np.clip(first, -step, step) / step,
                         delivers_k6=bool(delivery_success(metrics, cfg)), admissible=adm.admissible,
                         theta=[round(float(x), 5) for x in theta], source=source, metrics=metrics)
