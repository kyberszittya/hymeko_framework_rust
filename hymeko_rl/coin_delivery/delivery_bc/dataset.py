"""R11.4B dataset — deterministic teacher-replay theta-extraction + the conditioning descriptor + scenario split.

The certified demonstration bank persists the *input* side (handoff q/qdot/prev_tau, coin pose+vel, capture/contact
descriptors, delivery *outcome*) but NOT the winning theta. This module re-runs the frozen teacher on the certified
handoff (bit-for-bit deterministic given the seed) to recover ``DeliveryResult.theta`` — the BC regression target — and
assembles the standardizable input descriptor. Extraction and eval share ``reconstruct_capture`` so the closed-loop
harness starts from the exact same handoff state the teacher used.

Preconditions: ``sid`` is a bank scenario id; ``seed`` reproduces a certified grasp (the certification's ``selected_seed``
for a recovered scenario, or a first-certified seed for a frozen-R2 anchor).
Postconditions: ``extract_sample`` returns a ``BcSample`` whose ``theta`` reproduces the certified K6 delivery, or
``None`` if the seed does not yield a certified grasp / K6 delivery (never a fabricated label).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.evaluate import rollout_theta
from hymeko_rl.coin_delivery.delivery_teacher import solver as _solver
from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

FEATURE_NAMES: tuple[str, ...] = (
    "q0", "q1", "q2", "q3", "qd0", "qd1", "qd2", "qd3",
    "coin_x", "coin_y", "coin_vx", "coin_vy",
    "target_x", "target_y", "c2t_x", "c2t_y",
    "ptau0", "ptau1", "ptau2", "ptau3",
    "cap_n", "cap_s", "cap_preload", "cap_bmax",
    "bilateral_dwell", "contact_delay", "relvel1", "relvel2", "coin_disp_mm", "terminal_coin_speed",
)
N_FEATURES = len(FEATURE_NAMES)
THETA_DIM = 6


@dataclass(frozen=True)
class BcSample:
    """One conditioned-BC training row: the input descriptor, the certified teacher theta, and its delivery outcome."""

    scenario_id: str
    split: str
    seed: int
    source: str                 # "recovered" | "anchor"
    x: list[float]
    theta: list[float]
    k6: bool
    dtz_mm: float

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> "BcSample":
        return BcSample(scenario_id=d["scenario_id"], split=d["split"], seed=int(d["seed"]), source=d["source"],
                        x=[float(v) for v in d["x"]], theta=[float(v) for v in d["theta"]],
                        k6=bool(d["k6"]), dtz_mm=float(d["dtz_mm"]))


def bc_context() -> "tuple[Any, PipelineConfig, GraspObjective]":
    """The frozen (cfg, conf, obj) used by BOTH the certification and this extraction. The rig is built FRESH per
    scenario (see ``fresh_rig``): a rig reused across scenarios accumulates MuJoCo state and makes reconstruction
    order-dependent, so (scenario, seed) must own a clean rig to be reproducible across the extract and eval phases."""
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=1, grasp_objective=obj)
    return cfg, conf, obj


def fresh_rig() -> "dict[str, Any]":
    """A clean rig for one scenario's reconstruction — reproducible given (scenario, seed), history-independent."""
    return _rig()


def _f(v: "float | None") -> float:
    """None / NaN contact metrics (e.g. a missing second-contact relvel) map to 0.0 — an explicit, documented default."""
    return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def zone_of(snap: Any) -> "tuple[float, float]":
    """The model's canonical delivery zone centre (used when a scenario has no relocated target)."""
    inner = snap.branch().inner
    return float(inner._zone_x), float(inner._zone_y)


def target_of(scen: Any, snap: Any) -> np.ndarray:
    """The scenario's delivery target: the relocated ``target_xy`` if set, else the canonical zone centre."""
    if scen.target_xy is not None:
        return np.asarray(scen.target_xy, np.float64)
    return np.asarray(zone_of(snap), np.float64)


def descriptor(scen: Any, rc: Any, snap: Any) -> np.ndarray:
    """Assemble the ``N_FEATURES`` conditioning vector from the certified handoff + capture/contact + target geometry.

    Postcondition: shape ``(N_FEATURES,)``, ordered exactly as ``FEATURE_NAMES``.
    """
    rl = snap.branch()
    data = rl.inner.data
    q = np.asarray(data.qpos[:4], np.float64)
    qd = np.asarray(data.qvel[:4], np.float64)
    coin = np.asarray(_coin_xy(rl), np.float64)
    coin_vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    ptau = np.asarray(snap.prev_tau, np.float64)
    target = target_of(scen, snap)
    oc, params = rc.result.outcome, rc.result.params
    x = np.concatenate([
        q, qd, coin, coin_vel, target, target - coin, ptau,
        np.array([float(params.n), _f(params.s), _f(params.preload_start), _f(params.bmax),
                  float(oc.bilateral_dwell), float(oc.left_right_contact_delay),
                  _f(oc.first_contact_relvel), _f(oc.second_contact_relvel),
                  _f(oc.coin_disp_capture_mm), _f(oc.terminal_coin_speed)], np.float64),
    ])
    assert x.shape == (N_FEATURES,), f"descriptor shape {x.shape} != ({N_FEATURES},)"
    return x


def reconstruct_capture(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective,
                        scen: Any, seed: int) -> "Any | None":
    """Re-run reach+capture for ``scen`` at ``seed``; return the capture record if it is a certified grasp, else None.

    Deterministic: the same (scen, seed) reproduces the identical handoff snapshot the certification used.
    """
    home, coin = Z._home_with_coin(rig, scen.coin_xy)
    _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
    if rc is None or not is_certified_grasp(rc.result.outcome, obj):
        return None
    return rc


def scenario_by_id(sid: str) -> Any:
    return next(s for s in build_bank_scenarios() if s.scenario_id == sid)


def best_theta_full(snap: Any, tspec: Any, restarts: int) -> "tuple[bool, float, np.ndarray]":
    """Best-of-``restarts`` frozen-teacher delivery, returning the FULL-PRECISION winning theta and its
    ``rollout_theta``-measured outcome (k6, dtz).

    ``DeliveryResult.theta`` rounds to 4 dp, and some deliveries are chaotically sensitive (a <1e-4 theta perturbation
    lands the coin in a different basin — measured 7.99 mm vs 54.27 mm on bank_c1_-0.03_+0.00), so the rounded theta
    would not reproduce K6 under the eval harness. The winner is VERIFIED with the same ``rollout_theta`` the closed-loop
    gate uses, so every stored label is guaranteed to deliver K6 at eval. Early-exits on the first K6 (as the teacher).
    """
    cfg = _solver._config(tspec)
    best: "tuple[bool, float, np.ndarray] | None" = None
    for s in range(restarts):
        theta, _m = _solver._search(snap, cfg, tspec, s)
        r = rollout_theta(snap, np.asarray(theta, np.float64))              # verify with the EVAL harness
        cand = (bool(r["k6"]), float(r["dtz_mm"]), np.asarray(theta, np.float64))
        if best is None or (cand[0], -cand[1]) > (best[0], -best[1]):
            best = cand
        if cand[0]:
            break
    assert best is not None
    return best


def extract_sample(cfg: Any, conf: PipelineConfig, obj: GraspObjective,
                   sid: str, split: str, seed: int, source: str, restarts: int = 11) -> "BcSample | None":
    """Reconstruct the certified handoff (fresh rig) and replay the frozen teacher (R=``restarts``) to recover the
    full-precision winning theta, verified to deliver K6 under ``rollout_theta``.

    Returns None if the seed yields no certified grasp, or no restart reproduces K6 (a mislabeled target is worse than a
    missing one — the caller reports the omission).
    """
    scen = scenario_by_id(sid)
    rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scen, seed)
    if rc is None:
        return None
    snap = rc.result.outcome.snapshot
    k6, dtz, theta = best_theta_full(snap, full_transport_spec(), restarts)
    if not k6:
        return None
    x = descriptor(scen, rc, snap)
    return BcSample(scenario_id=sid, split=split, seed=seed, source=source,
                    x=[float(v) for v in x], theta=[float(t) for t in theta], k6=True, dtz_mm=float(dtz))
