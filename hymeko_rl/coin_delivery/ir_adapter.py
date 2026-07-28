"""Coin-domain adapter for the R11.2 HyMeKo IR: the MuJoCo <-> IR boundary.

This is the only place that touches simulator types. It reads a live coin rollout into a domain-generic
:class:`RolloutState`, declares the ``EXACT_ZERO_HOME_V1`` initial condition, certifies a fresh zero home, decides coin-pose
admissibility (the certificate-filtered distribution predicate), measures an energy ledger over a reach, and stamps a
:class:`RolloutProvenance` + strict-K6 :class:`SuccessCertificate` on every instrumented rollout. All energy terms are
**measured, not optimized** (some with documented unit-mass / coarse-integral proxies pending the R11.8 calibration).

The ``rig``/``rl`` objects are dynamic MuJoCo wrappers with no stubs; they are typed ``Any`` at this boundary only — the
generic IR (``hymeko_rl.ir``) stays fully typed and simulator-free.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments import coin_zero_home_rrt as R
from hymeko_rl.ir import (
    AdmissibilityResult,
    EnergyTransitionCertificate,
    HybridMode,
    InitialCondition,
    InitialConditionCertificate,
    InitialDistribution,
    MeasuredEnergyLedger,
    ModeTrace,
    RolloutProvenance,
    RolloutState,
    SuccessCertificate,
    build_mode_trace,
)

# EXACT_ZERO_HOME_V1: both 2R arms fully extended at q=[0,0,0,0], everything else at rest / empty. Immutable constant.
EXACT_ZERO_HOME_V1 = InitialCondition(name="EXACT_ZERO_HOME_V1", q_expected=np.zeros(4, dtype=np.float64))


def read_rollout_state(rl: Any, *, prev_tau: "np.ndarray | None" = None, rollout_step: int = 0,
                       has_snapshot: bool = False, has_teacher: bool = False,
                       memory_empty: bool = True) -> RolloutState:
    """Extract a domain-generic :class:`RolloutState` from a live coin rollout.

    The coin velocity is read from the true generalized velocities ``d.qvel[4:]`` — NOT ``_planar_metrics.disk_vel``,
    which is a finite-difference metric carrying ~1e-3 residual noise even when the object is exactly at rest. The
    rollout ``step`` is passed explicitly (0 for a fresh home): ``d.time`` reflects the coin-settling performed while
    *building* the home (~1.7 s), which is construction cost, not rollout progress, so it must not drive the step count.

    Preconditions: ``rl`` is a rollout with ``inner.data`` (MuJoCo); ``rollout_step >= 0``. Postconditions: the returned
    state's velocities/contacts reflect the live sim; ``prev_tau`` defaults to zero (the fresh-reset torque).
    """
    d = rl.inner.data
    con = primary_fingertip_contacts(rl)
    n_contacts = int(con["left"] is not None) + int(con["right"] is not None)
    tau = np.zeros(4, np.float64) if prev_tau is None else np.asarray(prev_tau, np.float64)
    return RolloutState(
        q=np.asarray(d.qpos[:4], np.float64), qdot=np.asarray(d.qvel[:4], np.float64), prev_tau=tau,
        object_vel=np.asarray(d.qvel[4:], np.float64), n_task_contacts=n_contacts,
        controller_memory_empty=memory_empty, step=int(rollout_step), has_snapshot_parent=has_snapshot,
        has_teacher_state=has_teacher)


def certify_zero_home(rig: dict, coin_xy: "np.ndarray | None" = None) -> InitialConditionCertificate:
    """Certify a freshly-built exact zero home against ``EXACT_ZERO_HOME_V1``. This is the R11.2 first-class check."""
    home, _coin = Z._home_with_coin(rig, coin_xy)
    return EXACT_ZERO_HOME_V1.certify(read_rollout_state(home.branch()))


def coin_admissibility(rig: dict, coin_xy: np.ndarray, cfg: "pga.TransitConfig | None" = None) -> AdmissibilityResult:
    """Certificate-filtered admissibility of a coin pose: the exact zero home must be collision-free and a non-empty
    precontact goal set must exist. (The full RRT-reach-within-budget check is the R11.3 demonstration-bank step.)"""
    cfg = cfg or pga.TransitConfig()
    home, coin = Z._home_with_coin(rig, coin_xy)
    arm_l, arm_r = pga.build_arms(home, coin)
    if not R._collision_free(np.zeros(4), coin, arm_l, arm_r, cfg):
        return AdmissibilityResult(False, "start_in_collision")

    def coll(q: np.ndarray) -> bool:
        return R._collision_free(q, coin, arm_l, arm_r, cfg)

    goals = R._straddle_goal_set(coin, arm_l, arm_r, pga.CoinStraddleTargets(coin=coin), coll)
    return AdmissibilityResult(True, "ADMISSIBLE") if goals else AdmissibilityResult(False, "no_precontact_goal_set")


def make_coin_distribution(rig: dict, lo: np.ndarray, hi: np.ndarray) -> InitialDistribution:
    """A certificate-filtered coin :class:`InitialDistribution` over an absolute-pose box; the predicate is
    :func:`coin_admissibility`, so inadmissible poses are ``INVALID_INITIAL_CONDITION`` rather than hard negatives."""
    return InitialDistribution(name="D_coin", lo=np.asarray(lo, np.float64), hi=np.asarray(hi, np.float64),
                               predicate=lambda pose: coin_admissibility(rig, pose))


def _robot_ke(rl: Any) -> float:
    """Measured robot kinetic energy 0.5 v_a^T M_aa v_a over the 4 arm DOFs (true mass matrix via mj_fullM)."""
    m, d = rl.inner.model, rl.inner.data
    full = np.zeros((m.nv, m.nv), np.float64)
    mujoco.mj_fullM(m, d, full)  # mujoco>=3 signature is (model, data, dst); densifies d.qM into dst
    v = np.asarray(d.qvel[:4], np.float64)
    return float(0.5 * v @ full[:4, :4] @ v)


def _object_ke(rl: Any) -> float:
    """Measured coin specific kinetic energy 0.5|v|^2 from the true generalized coin velocity ``qvel[4:6]`` (unit-mass
    proxy — the true coin mass enters at R11.8; not the noisy ``disk_vel`` metric)."""
    v = np.asarray(rl.inner.data.qvel[4:6], np.float64)
    return float(0.5 * v @ v)


@dataclass
class EnergyProbe:
    """Read-only per-waypoint work accumulator for the reach. Plugged in as the reach ``frame_hook`` (called after each
    waypoint), it integrates positive/negative actuator work W+/W- = sum max(+/- tau.qdot, 0) * dt_waypoint. Coarse
    (waypoint-resolution, not substep) and documented as such — the R11.2 ledger asserts completeness, not accuracy."""

    w_pos: float = 0.0
    w_neg: float = 0.0
    _t_last: float = 0.0

    def __call__(self, rl: Any, i: int) -> None:
        d = rl.inner.data
        power = float(np.asarray(d.ctrl[:4], np.float64) @ np.asarray(d.qvel[:4], np.float64))
        dt = max(0.0, float(d.time) - self._t_last)
        self._t_last = float(d.time)
        if power >= 0.0:
            self.w_pos += power * dt
        else:
            self.w_neg += -power * dt


def measured_reach_ledger(ke_pre: float, rl_post: Any, probe: EnergyProbe) -> MeasuredEnergyLedger:
    """Assemble a complete measured ledger for the reach segment (M0->M2). PE is 0 for the planar top-down task (constant
    height); contact impulse is 0 (no task contact during reach); dissipation proxy is the negative actuator work."""
    ke_post = _robot_ke(rl_post) + _object_ke(rl_post)
    return MeasuredEnergyLedger(
        robot_ke=_robot_ke(rl_post), object_ke=_object_ke(rl_post), potential_energy=0.0,
        w_actuator_pos=probe.w_pos, w_actuator_neg=probe.w_neg, contact_impulse=0.0,
        dissipation_proxy=probe.w_neg, energy_pre=ke_pre, energy_post=ke_post,
        numerical_residual=ke_post - (ke_pre + probe.w_pos - probe.w_neg))


def zero_home_reach_trace(*, captured: bool, k6: bool) -> ModeTrace:
    """The mode trace actually achieved: always M0->M1->M2 (reach); +M3 if the capture ran; +M4..M7 if it reached K6."""
    modes = [HybridMode.ZERO_HOME, HybridMode.FREE_REACH, HybridMode.PRECONTACT_ALIGNMENT]
    if captured:
        modes.append(HybridMode.CAPTURE)
    if k6:
        modes += [HybridMode.CONTROLLED_DELIVERY, HybridMode.TARGET_ENTRY, HybridMode.SETTLE, HybridMode.K6_SUCCESS]
    return build_mode_trace(modes)


def _git_sha() -> str:
    """Best-effort short-circuit to the working-tree HEAD; ``UNKNOWN`` if git is unavailable (kept out of the hash's way)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or "UNKNOWN"
    except (subprocess.SubprocessError, OSError):
        return "UNKNOWN"


def k6_success_certificate(outcome: Any, provenance_hash: str) -> SuccessCertificate:
    """Bind a strict-K6 outcome to the provenance hash of the rollout that produced it."""
    return SuccessCertificate(outcome_name="STRICT_K6", success=bool(outcome.k6),
                              metric_mm=float(outcome.min_dtz_mm), safe=bool(outcome.safe),
                              provenance_hash=provenance_hash)


def instrument_reach_rrt(rig: dict, coin_xy: "np.ndarray | None" = None, seed: int = 0,
                         git_sha: "str | None" = None) -> "dict | None":
    """Run the RRT reach + re-solved capture and emit the full IR bundle: IC certificate, measured energy certificate,
    mode trace, provenance, and the strict-K6 success certificate. Returns None when no reach/goal exists.

    Postconditions: on success, ``ic_certificate.valid`` for a genuine zero home; ``provenance.content_hash`` links the
    success certificate to this rollout.
    """
    cfg = replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    home, coin = Z._home_with_coin(rig, coin_xy)
    ic_cert = EXACT_ZERO_HOME_V1.certify(read_rollout_state(home.branch()))
    probe = EnergyProbe()
    res = R.reach_rrt(rig, cfg, coin_xy=coin_xy, seed=seed, frame_hook=probe)
    if res is None:
        return None
    ledger = measured_reach_ledger(0.0, res["ready"].branch() if hasattr(res["ready"], "branch") else home.branch(),
                                   probe)
    energy_cert = EnergyTransitionCertificate(ledger)
    o = type("O", (), {"k6": res["capture_k6"], "min_dtz_mm": res["min_dtz_mm"], "safe": res["safe"]})()
    trace = zero_home_reach_trace(captured=True, k6=bool(res["capture_k6"]))
    prov = RolloutProvenance(git_sha=git_sha or _git_sha(), seed=seed, ic_certificate=ic_cert,
                             coin_pose=np.asarray(coin, np.float64), target_pose=None, mode_trace=trace,
                             n_transitions=len(trace.modes) - 1, energy_ledger_complete=ledger.is_complete())
    return {"reach": res, "ic_certificate": ic_cert, "energy_certificate": energy_cert, "mode_trace": trace,
            "provenance": prov, "k6_certificate": k6_success_certificate(o, prov.content_hash())}
