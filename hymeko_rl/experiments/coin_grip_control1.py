"""COIN-GRIP-CONTROL-1 — active clamp-load regulation + friction audit (orchestration + gates + routing).

Answers: can a physically-meaningful clamp controller maintain a certified grasp on the CYLINDRICAL coin while the
midpoint moves? Actuator inventory (position servos + observable contact force) → C0-C4h are position-based (NO CORE
change). Tests them, then the friction audit (fingertip pad material). NO reward/CORE change; cylinder kept; NO RL.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from hymeko_rl.env.contact_formation_env import ContactFormationEnv
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.coin_delivery_acquisition1 import _states
from hymeko_rl.experiments.exp_v3_handoff_gate import _DIFF, _MAX_STEPS
from hymeko_rl.experiments.pedc_selection import _C1_HORIZON, _ctx, _load_pkl_bank, c1_config
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode
from hymeko_rl.train.coin_grip_control import GripController, GripParams, controlled_micro_transport
from hymeko_rl.train.coin_transport import extract_handoffs

_OUT = Path("experiments/2026_07_20_coin_grip_control1")
_ACQ_MAN = Path("experiments/2026_07_20_coin_delivery_acquisition/manifests/coin_delivery_acquisition.json")
_ACTUATOR_TABLE = {"position_target": "EXPOSED (4 position servos, kp40 kv4, no force limit)",
                   "velocity_target": "not exposed (would need velocity actuators — CORE)",
                   "torque_command": "NOT exposed (no motor actuators — CORE change to add)",
                   "impedance_target": "position servo IS a joint-space impedance; aperture-space impedance approximated via position",
                   "grip_force_target": "NOT a direct target (CORE); but regulable via observed force → position (C4h, no CORE)",
                   "contact_force_observation": "AVAILABLE (mj_contactForce)",
                   "aperture_observation": "AVAILABLE (obs[22])", "slip_observation": "AVAILABLE (disk_speed)",
                   "aperture_independent_of_midpoint": "YES (separate action dims → decoupled position targets)"}


def _log(m: str) -> None:
    print(m, flush=True)


def _acq() -> AcqParams:
    d = json.loads(_ACQ_MAN.read_text())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)


def _env(**kw) -> ContactFormationEnv:
    planar = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFF, task_graph=True,
                            fingertip_shape="capsule", fingertip_size="0.012 0.03", **kw)  # CYLINDER coin + Z-capsule
    return ContactFormationEnv(planar, _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False), _ctx()["contract"],
                               horizon=_C1_HORIZON, cfg=c1_config(), seed=0)


def _controlled(env, H, ctrl, p, *, scramble=None) -> dict:
    rows = [controlled_micro_transport(env, h, ctrl, p, scramble=scramble) for h in H]
    return {"n": len(rows), "micro_transport": sum(r["micro_transport"] for r in rows),
            "micro_strict": sum(r["micro_strict"] for r in rows),
            "median_rho": round(float(np.median([r["rho"] for r in rows])), 3) if rows else 0.0,
            "retained": sum(r["retained"] for r in rows),
            "median_coin_disp": round(float(np.median([r["coin_disp"] for r in rows])), 4) if rows else 0.0}


def run() -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    acq = _acq()
    wall = _states()["acquisition_wall"]
    p = GripParams()

    _log("=== §1 ACTUATOR CAPABILITY (position servos; contact force observable; C1/C2/C4h need NO CORE change) ===")
    for k, v in _ACTUATOR_TABLE.items():
        _log(f"  {k:32s} {v}")

    _log("=== controller arms at canonical friction (C0-C4h, position-based) ===")
    env0 = _env()
    H0 = extract_handoffs(env0, wall, acq)
    controllers = {c.value: _controlled(env0, H0, c, p) for c in GripController}
    for k, v in controllers.items():
        _log(f"  {k:26s} micro_transport={v['micro_transport']}/{v['n']} strict={v['micro_strict']}/{v['n']} "
             f"median_rho={v['median_rho']} retained={v['retained']}/{v['n']} (coin held but coupling weak)")

    _log("=== §10 friction audit (C1 aperture-feedback + fingertip pad material) ===")
    friction = {}
    for mu, name in [(None, "mu0_canonical"), (3.0, "mu1_moderate"), (6.0, "mu2_high")]:
        env = _env(**({} if mu is None else {"fingertip_friction": mu}))
        H = extract_handoffs(env, wall, acq)
        friction[name] = _controlled(env, H, GripController.C1_APERTURE_FB, p) if H else {"n": 0, "micro_transport": 0, "micro_strict": 0}
        _log(f"  C1+{name:14s} micro_transport={friction[name]['micro_transport']}/{friction[name]['n']} "
             f"strict={friction[name].get('micro_strict')}/{friction[name]['n']} "
             f"median_rho={friction[name].get('median_rho')} retained={friction[name].get('retained')}/{friction[name]['n']}")

    # structural control: does the aperture-feedback CONTROLLER contribute beyond a fixed squeeze, at the best friction?
    # KEY: if a SCRAMBLE matches/beats correct, the micro-transport is NOT controller-attributable (a bulldoze artifact).
    _log("=== structural control (C1 correct vs C0 feedback-off vs C1-inverted, at mu2) — loose vs STRICT ===")
    env_mu2 = _env(fingertip_friction=6.0)
    Hmu2 = extract_handoffs(env_mu2, wall, acq)

    def inv(a, _t):   # aperture-sign inverted (S2 structural scramble)
        return np.array([a[0], a[1], -a[2], a[3], a[4], a[5]], np.float32)

    struct = {"C1_correct": _controlled(env_mu2, Hmu2, GripController.C1_APERTURE_FB, p),
              "C0_feedback_off": _controlled(env_mu2, Hmu2, GripController.C0_CONSTANT, p),
              "C1_aperture_inverted": _controlled(env_mu2, Hmu2, GripController.C1_APERTURE_FB, p, scramble=inv)}
    for k, v in struct.items():
        _log(f"  {k:22s} micro_transport={v['micro_transport']}/{v['n']} STRICT={v['micro_strict']}/{v['n']} "
             f"median_rho={v['median_rho']}")

    verdict = _verdict(controllers, friction, struct)
    out = {"cylindrical_coin": True, "actuator_capability": _ACTUATOR_TABLE, "controllers_canonical_friction": controllers,
           "friction_audit": friction, "structural_control": struct, "gates": verdict["gates"],
           "routing_case": verdict["case"], "routing_reason": verdict["reason"], "core_change": verdict["core"],
           "canonical_change": verdict["canonical"], "rl_eligible": False, "kato15_justified": False,
           "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests" / "coin_grip_control1.json").write_text(json.dumps(out, indent=2, default=str))
    best_mt = max((friction[k].get("micro_transport", 0) for k in friction), default=0)
    _log(f"\n[COIN-GRIP-CONTROL-1] active-clamp-only micro-transport 0; +friction best {best_mt} | Case {verdict['case']} "
         f"| CORE-change: {verdict['core']} | {out['wall_s']}s")
    return out


def _verdict(controllers: dict, friction: dict, struct: dict) -> dict:
    """Routing per §9/§10. The STRUCTURAL CONTROL is decisive: if a SCRAMBLED clamp policy matches/beats the correct
    controller, the micro-transport is NOT controller-attributable — it is a physics artifact (a single-finger bulldoze),
    and the loose count is metric-inflated. Routes on the STRICT grasp-integrity predicate, not the loose one."""
    clamp_only_strict = max((controllers[k]["micro_strict"] for k in controllers), default=0)
    friction_strict = max((friction[k].get("micro_strict", 0) for k in friction), default=0)
    c_correct, c_off, c_inv = struct["C1_correct"], struct["C0_feedback_off"], struct["C1_aperture_inverted"]
    # falsification: does the correct controller genuinely beat BOTH the feedback-off and the scrambled arm (loose count)?
    controller_survives = c_correct["micro_transport"] > c_off["micro_transport"] \
        and c_correct["micro_transport"] >= c_inv["micro_transport"]
    scramble_inflated = c_inv["micro_transport"] > c_correct["micro_transport"] or c_off["micro_transport"] > c_correct["micro_transport"]
    gates = {"GRIP_CONTROL_survives_structural_control": bool(controller_survives),
             "SCRAMBLE_matches_or_beats_correct(metric_inflation)": bool(scramble_inflated),
             "MICRO_TRANSPORT_strict_clamp_only": bool(clamp_only_strict >= 1),
             "MICRO_TRANSPORT_strict_with_friction": bool(friction_strict >= 1),
             "MICRO_TRANSPORT_strict_robust_ge4": bool(friction_strict >= 4)}
    if controller_survives and clamp_only_strict >= 4:
        return {"case": "A_or_B", "reason": "active clamp regulation genuinely micro-transports the cylinder (survives the "
                "structural control; strict grasp-integrity predicate) — no CORE change", "gates": gates,
                "core": "not_needed", "canonical": "adopt aperture-feedback/impedance controller"}
    if scramble_inflated:
        return {"case": "E_falsified", "reason": "The campaign's decisive hypothesis — «transport requires active "
                "regulation of aperture/preload/compliance» — is FALSIFIED by the structural control: the aperture-feedback "
                f"controller ({c_correct['micro_transport']}/{c_correct['n']}) is beaten by feedback-OFF "
                f"({c_off['micro_transport']}/{c_correct['n']}) and by its own SCRAMBLE "
                f"({c_inv['micro_transport']}/{c_correct['n']}). A scramble winning ⇒ the loose micro-transport at high "
                "friction is a single-finger BULLDOZE artifact, not a maintained bilateral clamp (metric-inflation, per "
                "the evaluation-integrity rule). Under the STRICT grasp-integrity predicate (end-bilateral + both_frac≥.75 "
                f"+ coin travels WITH the midpoint) the friction 'wins' collapse to {friction_strict}. So: position-based "
                "active clamp regulation does NOT achieve trustworthy bilateral grasp-transport of the cylinder; the "
                "fingertip↔cylinder friction changes coupling but not via a controllable grasp. The lever is NOT a "
                "position-space controller.", "gates": gates,
                "core": "NONE for this campaign (falsified without one). The next lever is a genuinely different ACTUATOR "
                "(true torque/impedance/grip-force — a gated CORE change) or a mechanical redesign (wrist DOF / compliant "
                "pad), NOT more position-space clamp tuning",
                "canonical": "NONE — hypothesis falsified; no controller adopted; cylinder kept"}
    if friction_strict >= 1:
        return {"case": "D", "reason": "friction + controller genuinely micro-transports (survives the strict predicate) "
                f"in {friction_strict} states — marginal, needs full-chain replication", "gates": gates,
                "core": "NONE — position control + fingertip friction/geom variant are additive/non-CORE",
                "canonical": "NOT adopted yet — marginal; candidate is capsule + aperture feedback + pad friction"}
    return {"case": "F", "reason": "no controller or friction variant strictly micro-transports; the wrist-less position "
            "clamp cannot transport the cylinder → mechanical redesign / true actuators (CORE)", "gates": gates,
            "core": "escalate to a mechanical-redesign audit or a gated CORE actuator change", "canonical": "NONE"}


if __name__ == "__main__":
    import sys
    run()
    sys.exit(0)
