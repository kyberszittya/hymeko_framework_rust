"""The R11.3 per-scenario demonstration pipeline: EXACT_ZERO_HOME -> IC certificate -> admissibility -> deployed RRT reach
-> precontact handoff -> per-instance CEM teacher -> strict K6 / classified failure -> provenance + measured energy ledger.

This composes the *deployed* reach primitives (from ``coin_zero_home_rrt``) with the coin ``ir_adapter`` and the CEM
teacher (``moving_precapture.plan_capture``), running the reach and exactly one capture so the full teacher signals
(contacts, capture params, tube metrics) are recorded. The zone (delivery target) is relocated before the capture when a
scenario specifies a target; whether the relocation is honored is recorded. The CEM capture is a *training teacher only*,
labelled as such in the provenance (``teacher_identity``); no rollout is ever marked teacher-free. Generation only: no
BC/RL/refinement here.

The teacher-handoff split (per the R11.3 teacher-generation contract): a geometrically valid, well-tracked RRT reach can
land outside the frozen CEM teacher's admissible handoff manifold. ``handoff_admissible`` (a geometric READY predicate) and
``coin_progress = entry_dtz - min_dtz`` let the classifier separate PRECONTACT_HANDOFF_INVALID from CAPTURE/DELIVERY teacher
failures.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

from hymeko_rl.coin_delivery import ir_adapter as A
from hymeko_rl.coin_delivery.demo_bank.failure_class import ClassifyThresholds, FailureClass, RolloutSignals, classify
from hymeko_rl.coin_delivery.demo_bank.record import SUCCESS_LABEL, DemonstrationRecord
from hymeko_rl.coin_delivery.demo_bank.scenario import CoinTargetScenario, ScenarioSplit, curriculum_stage
from hymeko_rl.coin_delivery.forward_displacement import primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments import coin_zero_home_rrt as R
from hymeko_rl.ir import (
    EnergyTransitionCertificate,
    HandoffDescriptor,
    HybridMode,
    InitialConditionCertificate,
    RolloutProvenance,
)

SCHEMA_VERSION = "r11.3-v2"
TEACHER_IDENTITY = "CEM_phase_shape_capture"
_NO_TEACHER = "none"


def _finite_or_none(x: float, ndigits: int = 4) -> Optional[float]:
    """Round a measured scalar, or None if it is not finite — so no-capture rollouts serialize to valid, stable JSON."""
    return round(float(x), ndigits) if math.isfinite(float(x)) else None


@dataclass(frozen=True)
class PipelineConfig:
    """Pilot pipeline knobs. ``teacher_budget`` caps the CEM re-solve seeds; the reach config keeps precontact coin motion
    at ~0 mm (the governed transit already satisfies the <=1 mm contract).

    ``grasp_objective`` selects the capture-teacher mode (R11.4A integration): the default is the grasp-aware objective
    (ranks candidates by grasp class → delivery, hybrid elite) so the pipeline no longer accepts an ungrasped nudge as a
    capture success. Setting it to ``None`` restores the bit-exact grasp-agnostic (min_dtz/K6) capture for regression.
    """

    thresholds: ClassifyThresholds = ClassifyThresholds()
    teacher_budget: int = 3
    substeps: int = 6
    hold_steps: int = 160
    grasp_objective: "GraspObjective | None" = GraspObjective()


def _zone_xy(rl: Any) -> tuple[float, float]:
    return (float(rl.inner._zone_x), float(rl.inner._zone_y))


def _relocate_zone(rl: Any, target_xy: "np.ndarray | None") -> tuple[tuple[float, float], bool]:
    """Set the delivery zone to ``target_xy`` (if given) and report (zone_xy, honored)."""
    if target_xy is None:
        return _zone_xy(rl), True
    rl.inner._zone_x, rl.inner._zone_y = float(target_xy[0]), float(target_xy[1])
    zone = _zone_xy(rl)
    return zone, bool(np.allclose(zone, np.asarray(target_xy, float), atol=1e-6))


def _handoff_admissible(coin: np.ndarray, tip_l: np.ndarray, tip_r: np.ndarray,
                        r_lo: float = 0.03, r_hi: float = 0.09) -> bool:
    """Geometric READY predicate for the teacher's capture manifold: both fingertips within a graspable shell of the coin
    and on opposite sides (straddling). A documented proxy for 'the teacher could be expected to grasp from here'."""
    vl, vr = np.asarray(tip_l, float) - coin, np.asarray(tip_r, float) - coin
    dl, dr = float(np.linalg.norm(vl)), float(np.linalg.norm(vr))
    if not (r_lo <= dl <= r_hi and r_lo <= dr <= r_hi):
        return False
    return float(vl @ vr) < 0.0                                   # opposite sides -> angle > 90 deg


def _min_link_clr_mm(coin: np.ndarray, q: np.ndarray, arm_l: pga.PlanarArm2R, arm_r: pga.PlanarArm2R) -> float:
    """Minimum distance (mm) from either arm's link capsules to the coin at configuration ``q``."""
    best = 9.0
    for arm, qq in ((arm_l, q[:2]), (arm_r, q[2:])):
        b, e, t = arm.link_points(qq)
        best = min(best, pga._seg_point_dist(b, e, coin), pga._seg_point_dist(e, t, coin))
    return round(best * 1000, 1)


@dataclass
class _ReachCapture:
    ready: Any
    reach: dict[str, float]
    reach_contacts: int
    n_goals: int
    planning_time_s: float
    min_link_clr_mm: float
    handoff: HandoffDescriptor
    handoff_admissible: bool
    energy_cert: EnergyTransitionCertificate
    energy_complete: bool
    zone: tuple[float, float]
    target_honored: bool
    result: Any  # CaptureResult


def _do_reach_and_capture(rig: dict[str, Any], scenario: CoinTargetScenario, coin: np.ndarray, home: Any,
                          cfg: pga.TransitConfig, config: PipelineConfig,
                          seed: int) -> "tuple[Optional[str], Optional[_ReachCapture]]":
    """Run the deployed RRT reach (scenario zone injected) then one CEM capture. Returns (reason, None) on a reach failure
    (``GOAL_SET_EMPTY`` / ``RRT_PLANNING_FAILURE``) or (None, _ReachCapture) on success."""
    stack = rig["cradle"].stack
    arm_l, arm_r = pga.build_arms(home, coin)
    rl = home.branch()
    zone, honored = _relocate_zone(rl, scenario.target_xy)
    gl, gr = pga._fingertip_geoms(rl.inner.model)
    lo, hi = np.asarray(home.lo, float), np.asarray(home.hi, float)

    def coll(q: np.ndarray) -> bool:
        return R._collision_free(q, coin, arm_l, arm_r, cfg)

    rng = np.random.default_rng(seed)
    goals = R._straddle_goal_set(coin, arm_l, arm_r, pga.CoinStraddleTargets(coin=coin), coll)
    if not goals or not coll(np.zeros(4)):
        return "GOAL_SET_EMPTY", None
    t0 = time.perf_counter()
    raw = R.rrt_connect(np.zeros(4), goals, coll, rng)
    planning_time_s = round(time.perf_counter() - t0, 3)
    if raw is None:
        return "RRT_PLANNING_FAILURE", None
    qref = R._densify(R._shortcut(raw, coll, rng))
    probe = A.EnergyProbe()
    prev, minclr, pert = Z._servo(rl, qref, stack, lo, hi, cfg, coin, gl, gr, frame_hook=probe)
    rc = _assemble_reach(rig, rl, coin, arm_l, arm_r, gl, gr, qref, raw, prev, minclr, pert, probe,
                         zone, honored, len(goals), planning_time_s, config, seed, stack)
    return None, rc


def _assemble_reach(rig: dict[str, Any], rl: Any, coin: np.ndarray, arm_l: pga.PlanarArm2R, arm_r: pga.PlanarArm2R,
                    gl: int, gr: int, qref: np.ndarray, raw: list, prev: np.ndarray, minclr: float, pert: float,
                    probe: Any, zone: tuple[float, float], honored: bool, n_goals: int, planning_time_s: float,
                    config: PipelineConfig, seed: int, stack: Any) -> _ReachCapture:
    """Build the reach metrics + handoff + measured ledger, then run one CEM capture (keeps the reach fn under budget)."""
    d = rl.inner.data
    straddle = pga.CoinStraddleTargets(coin=coin).precontact()
    tip_l, tip_r = np.asarray(d.geom_xpos[gl][:2], float), np.asarray(d.geom_xpos[gr][:2], float)
    reach = {"rrt_waypoints": float(len(raw)), "path_waypoints": float(len(qref)),
             "reach_qerr": round(float(np.linalg.norm(d.qpos[:4] - qref[-1])), 4),
             "min_tip_coin_clr_mm": round(minclr * 1000, 1), "coin_moved_before_capture_mm": round(pert * 1000, 2),
             "tip_l_err_mm": round(float(np.linalg.norm(tip_l - straddle.tip_left)) * 1000, 1),
             "tip_r_err_mm": round(float(np.linalg.norm(tip_r - straddle.tip_right)) * 1000, 1)}
    con = primary_fingertip_contacts(rl)
    reach_contacts = int(con["left"] is not None) + int(con["right"] is not None)
    handoff = HandoffDescriptor(q=np.asarray(d.qpos[:4], float), qdot=np.asarray(d.qvel[:4], float), prev_tau=prev.copy(),
                                x_coin=np.asarray(coin, float), xdot_coin=np.zeros(2), x_goal=np.asarray(zone, float),
                                n_contacts=reach_contacts, e_phase=0.0, mode_from=HybridMode.PRECONTACT_ALIGNMENT,
                                mode_to=HybridMode.CAPTURE)
    ledger = A.measured_reach_ledger(0.0, rl, probe)
    ready = kc.TransportSnapshot.from_live(copy.deepcopy(rl), stack, prev.copy())
    result = _best_capture(rig, ready, seed, config.teacher_budget, config.grasp_objective)
    return _ReachCapture(ready=ready, reach=reach, reach_contacts=reach_contacts, n_goals=n_goals,
                         planning_time_s=planning_time_s,
                         min_link_clr_mm=_min_link_clr_mm(coin, np.asarray(d.qpos[:4], float), arm_l, arm_r),
                         handoff=handoff, handoff_admissible=_handoff_admissible(coin, tip_l, tip_r),
                         energy_cert=EnergyTransitionCertificate(ledger), energy_complete=ledger.is_complete(),
                         zone=zone, target_honored=honored, result=result)


def _best_capture(rig: dict[str, Any], ready: Any, seed: int, budget: int,
                  grasp_objective: "GraspObjective | None") -> Any:
    """CEM teacher re-solve over up to ``budget`` seeds. Grasp-aware (default) ranks candidates by grasp class → delivery
    (consistent with the in-CEM ranking) and stops only on a certified grasped-K6; ``grasp_objective=None`` restores the
    bit-exact grasp-agnostic ``(k6, min_dtz)`` re-solve."""
    from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
    if grasp_objective is None:
        return _best_capture_legacy(rig, ready, seed, budget, mp)
    spec = mp.CaptureSearchSpec(grasp_objective=grasp_objective)
    best: "tuple[Any, Any] | None" = None
    for sd in range(seed, seed + budget):
        res = mp.plan_capture(ready, rig["ref"], rig["stack"], rig["down"], seed=sd, spec=spec)
        key = mp.rank_capture(res.outcome, grasp_objective)
        if best is None or key < best[0]:
            best = (key, res)
        if res.outcome.k6 and res.outcome.min_dtz_mm < 10 and mp.is_certified_grasp(res.outcome, grasp_objective):
            break
    assert best is not None
    return best[1]


def _best_capture_legacy(rig: dict[str, Any], ready: Any, seed: int, budget: int, mp: Any) -> Any:
    """Bit-exact prior grasp-agnostic re-solve: best by ``(k6, -min_dtz)``, early-exit on a tight K6."""
    best = None
    for sd in range(seed, seed + budget):
        res = mp.plan_capture(ready, rig["ref"], rig["stack"], rig["down"], seed=sd)
        if best is None or (res.outcome.k6, -res.outcome.min_dtz_mm) > (best.outcome.k6, -best.outcome.min_dtz_mm):
            best = res
        if res.outcome.k6 and res.outcome.min_dtz_mm < 10:
            break
    return best


def _ledger_dict(cert: EnergyTransitionCertificate) -> dict[str, float]:
    led = cert.ledger
    return {"robot_ke": round(led.robot_ke, 6), "object_ke": round(led.object_ke, 6),
            "potential_energy": round(led.potential_energy, 6), "w_actuator_pos": round(led.w_actuator_pos, 6),
            "w_actuator_neg": round(led.w_actuator_neg, 6), "contact_impulse": round(led.contact_impulse, 6),
            "dissipation_proxy": round(led.dissipation_proxy, 6), "energy_pre": round(led.energy_pre, 6),
            "energy_post": round(led.energy_post, 6), "numerical_residual": round(led.numerical_residual, 6),
            "balance_residual": round(led.balance_residual(), 6)}


def _no_teacher_record(scenario: CoinTargetScenario, seed: int, ic: InitialConditionCertificate, coin: np.ndarray,
                       zone: tuple[float, float], git_sha: str, *, admissible: bool, reason: str,
                       label: str) -> DemonstrationRecord:
    """A record for a rollout that never reached the teacher (inadmissible start, empty goal set, or planning failure)."""
    rel = scenario.relative_goal(np.asarray(zone, float))
    trace = A.zero_home_reach_trace(captured=False, k6=False)
    prov = RolloutProvenance(git_sha=git_sha, seed=seed, ic_certificate=ic, coin_pose=np.asarray(coin, float),
                             target_pose=scenario.target_xy, mode_trace=trace, n_transitions=0,
                             energy_ledger_complete=False)
    return DemonstrationRecord(
        schema_version=SCHEMA_VERSION, scenario_id=scenario.scenario_id, split=scenario.split.value,
        kind=scenario.kind, curriculum_stage=curriculum_stage(scenario.kind), seed=seed, sample_id=scenario.scenario_id,
        ic_valid=ic.valid, ic_violations=ic.violations, admissible=admissible, rejection_reason=reason,
        coin_pose=(float(coin[0]), float(coin[1])),
        target_pose=None if scenario.target_xy is None else (float(scenario.target_xy[0]), float(scenario.target_xy[1])),
        relative_goal=(float(rel[0]), float(rel[1])), zone_pose=zone, target_relocation_honored=False,
        reach_found=False, goal_set_empty=(label == "GOAL_SET_EMPTY"), n_goals=0, planning_time_s=0.0, reach=None,
        min_link_clr_mm=None, premature_contacts=0, handoff_complete=False, handoff_admissible=False,
        mode_trace=tuple(int(m) for m in trace.modes), teacher_identity=_NO_TEACHER, teacher_seed=-1, teacher_params={},
        entry_obs={}, contact_sequence=(), contacts=0, cos_dir=None, vel_scale=None, dtau=None, k6=False,
        entry_dtz_mm=None, min_dtz_mm=None, coin_progress_mm=None, safe=True, outcome_label=label, energy_ledger={},
        energy_verdicts=(), energy_measurement_complete=False, git_sha=git_sha, provenance_hash=prov.content_hash(),
        k6_provenance_link=False)


def _integrity_override(label: str, k6: bool, k6_link: bool, energy_complete: bool) -> str:
    """Integrity failures override the physics label: an incomplete energy ledger, or a K6 whose success certificate does
    not link to the provenance hash, is a pipeline failure regardless of the physics outcome."""
    if not energy_complete:
        return FailureClass.ENERGY_LEDGER_INCOMPLETE.value
    if k6 and not k6_link:
        return FailureClass.PROVENANCE_FAILURE.value
    return label


def _build_record(scenario: CoinTargetScenario, seed: int, ic: InitialConditionCertificate, coin: np.ndarray,
                  rc: _ReachCapture, thr: ClassifyThresholds, git_sha: str) -> DemonstrationRecord:
    """Assemble the full success/failure demonstration record from a completed reach+capture."""
    o, p = rc.result.outcome, rc.result.params
    entry_dtz_mm = _finite_or_none(float(rc.ready.branch().inner.direction_to_zone()[1]) * 1000, 2)
    min_dtz_mm = _finite_or_none(o.min_dtz_mm, 3)
    coin_progress = (None if entry_dtz_mm is None or min_dtz_mm is None
                     else round(entry_dtz_mm - min_dtz_mm, 3))
    sig = RolloutSignals(goal_set_empty=False, reach_found=True, reach_qerr_rad=rc.reach["reach_qerr"],
                         precontact_coin_motion_mm=rc.reach["coin_moved_before_capture_mm"],
                         premature_contacts=rc.reach_contacts, safe=bool(o.safe),
                         handoff_admissible=rc.handoff_admissible, k6=bool(o.k6),
                         coin_progress_mm=coin_progress if coin_progress is not None else 0.0,
                         min_dtz_mm=float(o.min_dtz_mm))
    fc = classify(sig, thr)
    physics_label = SUCCESS_LABEL if fc is None else fc.value
    trace = A.zero_home_reach_trace(captured=True, k6=bool(o.k6))
    prov = RolloutProvenance(git_sha=git_sha, seed=seed, ic_certificate=ic, coin_pose=np.asarray(coin, float),
                             target_pose=scenario.target_xy, mode_trace=trace,
                             n_transitions=len(trace.modes) - 1, energy_ledger_complete=rc.energy_complete)
    phash = prov.content_hash()
    k6_cert = A.k6_success_certificate(o, phash)
    k6_link = bool(o.k6) and k6_cert.provenance_hash == phash
    label = _integrity_override(physics_label, bool(o.k6), k6_link, rc.energy_complete)
    rel = scenario.relative_goal(np.asarray(rc.zone, float))
    return DemonstrationRecord(
        schema_version=SCHEMA_VERSION, scenario_id=scenario.scenario_id, split=scenario.split.value,
        kind=scenario.kind, curriculum_stage=curriculum_stage(scenario.kind), seed=seed, sample_id=scenario.scenario_id,
        ic_valid=ic.valid, ic_violations=ic.violations, admissible=True, rejection_reason="ADMISSIBLE",
        coin_pose=(float(coin[0]), float(coin[1])),
        target_pose=None if scenario.target_xy is None else (float(scenario.target_xy[0]), float(scenario.target_xy[1])),
        relative_goal=(float(rel[0]), float(rel[1])), zone_pose=rc.zone, target_relocation_honored=rc.target_honored,
        reach_found=True, goal_set_empty=False, n_goals=rc.n_goals, planning_time_s=rc.planning_time_s, reach=rc.reach,
        min_link_clr_mm=rc.min_link_clr_mm, premature_contacts=rc.reach_contacts,
        handoff_complete=rc.handoff.is_complete(), handoff_admissible=rc.handoff_admissible,
        mode_trace=tuple(int(m) for m in trace.modes), teacher_identity=TEACHER_IDENTITY,
        teacher_seed=int(rc.result.seed),
        teacher_params={"n": float(p.n), "s": float(p.s), "preload_start": float(p.preload_start),
                        "bmax": float(p.bmax), "steps": float(p.steps), "kp": float(p.kp), "kd": float(p.kd)},
        entry_obs={"entry_dtz_mm": entry_dtz_mm, "cos_dir": _finite_or_none(o.cos_dir),
                   "vel_scale": _finite_or_none(o.vel_scale), "dtau": _finite_or_none(o.dtau)},
        contact_sequence=(rc.reach_contacts, int(o.contacts)), contacts=int(o.contacts),
        cos_dir=_finite_or_none(o.cos_dir), vel_scale=_finite_or_none(o.vel_scale), dtau=_finite_or_none(o.dtau),
        k6=bool(o.k6), entry_dtz_mm=entry_dtz_mm, min_dtz_mm=min_dtz_mm, coin_progress_mm=coin_progress,
        safe=bool(o.safe), outcome_label=label, energy_ledger=_ledger_dict(rc.energy_cert),
        energy_verdicts=rc.energy_cert.verdicts, energy_measurement_complete=rc.energy_cert.is_measurement_complete(),
        git_sha=git_sha, provenance_hash=phash, k6_provenance_link=k6_link)


def run_scenario(rig: dict[str, Any], scenario: CoinTargetScenario, seed: int = 0,
                 config: PipelineConfig = PipelineConfig()) -> DemonstrationRecord:
    """Run one scenario through the full certified pipeline and return its demonstration record.

    Preconditions: ``scenario.coin_xy`` is a valid coin pose to attempt. Postconditions: every returned record carries a
    valid-or-invalid IC certificate + a provenance hash; INVALID-split / inadmissible scenarios are rejected before
    planning (no teacher use) and never counted in the success denominator.
    """
    cfg = replace(pga.TransitConfig(), substeps=config.substeps, hold_steps=config.hold_steps)
    git_sha = A._git_sha()
    home, coin = Z._home_with_coin(rig, scenario.coin_xy)
    ic = A.EXACT_ZERO_HOME_V1.certify(A.read_rollout_state(home.branch()))
    adm = A.coin_admissibility(rig, scenario.coin_xy, cfg)
    zone0 = _zone_xy(home.branch())
    if scenario.split == ScenarioSplit.INVALID or not adm.admissible:
        reason = "INVALID_INITIAL_CONDITION" if scenario.split == ScenarioSplit.INVALID else adm.reason
        return _no_teacher_record(scenario, seed, ic, coin, zone0, git_sha, admissible=False, reason=reason,
                                  label="INVALID_INITIAL_CONDITION")
    reason, rc = _do_reach_and_capture(rig, scenario, coin, home, cfg, config, seed)
    if rc is None:
        assert reason is not None, "a failed reach must carry a reason"
        return _no_teacher_record(scenario, seed, ic, coin, zone0, git_sha, admissible=True, reason="ADMISSIBLE",
                                  label=reason)
    return _build_record(scenario, seed, ic, coin, rc, config.thresholds, git_sha)
