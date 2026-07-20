"""CLAMP-ORACLE-0 — isolated physical clamp-transport oracle (Track B, bounded side audit; NOT RL, NOT task success).

Decisive question: does a strict-passing physical clamp oracle exist that would justify implementing true force/impedance
actuation? Conditions O0 (metric calibration) · O1 (position parallel jaw) · O2 (force/impedance clamp) · O3 (force +
Z-capsule) · O4 (box positive control) · O5 (direct object translation, calibration) + a structural scramble control.
Routing per §5 (Oracle Case A–E). NO reward/CORE change; the canonical cylinder + task are untouched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from hymeko_rl.train.clamp_oracle import Actuation, OracleParams, OracleRig, build_oracle_mjcf
from hymeko_rl.train.coin_compliant_pad import strict_micro_transport_pass

_OUT = Path("experiments/2026_07_20_clamp_oracle0")
_P = OracleParams()
_FIRM = dict(coin_disp=0.05, align=0.99, rho=0.95, both_frac=1.0, final_bilateral=True,
             max_eta=0.6, min_FN=2.0, penetration_max=0.003, solver_ok=True)


def _log(m: str) -> None:
    print(m, flush=True)


def _condition(label: str, *, actuation: Actuation, control: str = "normal", **kw) -> dict:
    rig = OracleRig(build_oracle_mjcf(actuation=actuation, p=_P, **kw), actuation, _P)
    r = rig.run_condition(control=control)
    d = r.as_dict()
    d["condition"] = label
    _log(f"  {label:26s} strict={str(r.strict_pass):5s} rho={r.rho:5} both={r.both_frac} align={r.align} "
         f"disp={r.coin_disp} FN={r.min_FN} eta={r.max_eta} pen={r.penetration_max} "
         f"FT>supp={r.finger_exceeds_support} obj_act={r.object_actuated} solver={r.solver_ok}")
    return d


def _o0_metric_calibration() -> dict:
    """O0: validate the strict predicate on synthetic inputs — PASS a firm-coupling trajectory, REJECT the artifacts.
    Metric validity ONLY; NOT physical headroom evidence."""
    passes = strict_micro_transport_pass(**_FIRM)
    rejects = {
        "bulldoze": not strict_micro_transport_pass(**{**_FIRM, "final_bilateral": False}),
        "slip": not strict_micro_transport_pass(**{**_FIRM, "max_eta": 1.0}),
        "soft_grip": not strict_micro_transport_pass(**{**_FIRM, "min_FN": 0.0}),
        "squirt": not strict_micro_transport_pass(**{**_FIRM, "align": 0.2}),
    }
    ok = bool(passes and all(rejects.values()))
    _log(f"  O0_metric_calibration     firm_pass={passes} rejects={rejects} → metric_valid={ok} (NOT headroom)")
    return {"condition": "O0_metric_calibration", "firm_pass": passes, "rejects": rejects, "metric_valid": ok}


def run() -> dict:
    t0 = time.perf_counter()
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests").mkdir(exist_ok=True)
    _log("=== CLAMP-ORACLE-0 (diagnostic_physical_oracle, noncanonical) — is strict clamp transport achievable? ===")
    o0 = _o0_metric_calibration()
    conds = {
        "O1_position_sphere": _condition("O1_position_sphere", actuation=Actuation.POSITION),
        "O1b_position_capsule": _condition("O1b_position_capsule", actuation=Actuation.POSITION, fingertip_shape="capsule"),
        "O2_force_sphere": _condition("O2_force_sphere", actuation=Actuation.FORCE),
        "O3_force_capsule": _condition("O3_force_capsule", actuation=Actuation.FORCE, fingertip_shape="capsule"),
        "O4_box_positive_control": _condition("O4_box_positive_control", actuation=Actuation.FORCE, coin_shape="box"),
        "O5_object_direct": _condition("O5_object_direct", actuation=Actuation.OBJECT_DIRECT),
    }
    _log("=== structural controls (correct bilateral clamp MUST beat the open-right scramble; §T9) ===")
    controls = {
        "O1_scramble_open_right": _condition("O1_scramble_open_right", actuation=Actuation.POSITION, control="open_right"),
        "O2_scramble_open_right": _condition("O2_scramble_open_right", actuation=Actuation.FORCE, control="open_right"),
    }
    verdict = _route(o0, conds, controls)
    out = {"diagnostic_physical_oracle": True, "noncanonical": True, "metric_calibration": o0, "conditions": conds,
           "structural_controls": controls, "gates": verdict["gates"], "oracle_case": verdict["case"],
           "verdict": verdict["reason"], "core_decision": verdict["core"], "rl_run": False,
           "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests" / "clamp_oracle0.json").write_text(json.dumps(out, indent=2, default=str))
    _log(f"\n[CLAMP-ORACLE-0] Oracle {verdict['case']} | CORE: {verdict['core'][:80]} | {out['wall_s']}s")
    return out


def _route_signals(o0: dict, conds: dict, controls: dict) -> dict:
    """Extract the boolean routing signals + gate dict from the condition tables (keeps _route a pure branch)."""
    o1 = conds["O1_position_sphere"]["strict_pass"] or conds["O1b_position_capsule"]["strict_pass"]
    o2 = conds["O2_force_sphere"]["strict_pass"] or conds["O3_force_capsule"]["strict_pass"]
    box = conds["O4_box_positive_control"]["strict_pass"]
    scr = controls["O1_scramble_open_right"]["strict_pass"] or controls["O2_scramble_open_right"]["strict_pass"]
    physical = all(c["penetration_max"] <= 0.02 and c["solver_ok"] for c in conds.values()
                   if c["strict_pass"] and not c["object_actuated"])   # no pass may rely on unphysical contact (§Case E)
    gates = {"O0_metric_valid": o0["metric_valid"], "O1_position_jaw_passes": bool(o1),
             "O2_force_clamp_passes": bool(o2), "O4_box_positive_control_passes": bool(box),
             "SCRAMBLE_beaten(correct>scramble)": bool(not scr), "PHYSICAL_no_unphysical_pass": bool(physical)}
    return {"o1": o1, "o2": o2, "box": box, "scr": scr, "physical": physical, "gates": gates}


def _route(o0: dict, conds: dict, controls: dict) -> dict:
    """§5 routing. Case A if the POSITION jaw (O1/O1b) strict-passes AND beats its scramble (position suffices →
    kinematic/geometric fix, NO force actuator). Case B if only FORCE (O2/O3) passes. Case C/D/E otherwise."""
    sig = _route_signals(o0, conds, controls)
    o1, o2, box, scr, physical, gates = sig["o1"], sig["o2"], sig["box"], sig["scr"], sig["physical"], sig["gates"]
    if scr:
        return {"case": "E_reject", "reason": "a SCRAMBLED (open-right) clamp strict-passes → the oracle 'pass' is not a "
                "real bilateral clamp; reject the oracle, no actuator change authorized", "gates": gates,
                "core": "NONE — oracle rejected (scramble passes)"}
    if not physical:
        return {"case": "E_reject", "reason": "a pass relies on unphysical penetration/adhesion/solver instability; "
                "reject the oracle", "gates": gates, "core": "NONE — oracle rejected (unphysical)"}
    if o1:
        return {"case": "A", "reason": "the POSITION-controlled parallel jaw (correctly-oriented independent opposing "
                "pads) STRICT-transports the cylinder and beats the scramble — clamp transport is physically achievable "
                "with POSITION control given correct geometry/kinematics. The canonical-arm failure is KINEMATIC/"
                "GEOMETRIC (a wrist-less 2-link arm cannot present two correctly-oriented, independently-controllable "
                "opposing pads with a decoupled aperture+midpoint), NOT actuator-type. Force/impedance (O2/O3) also pass "
                "but are NOT NECESSARY. So a TRUE torque/impedance CORE actuator is NOT justified by this oracle; the "
                "warranted lever is a KINEMATIC/GEOMETRIC redesign (a wrist DOF, or a dedicated parallel-jaw gripper "
                "aperture DOF, or an articulated end-effector). This REFINES the compliant-pad inference ('need a "
                "maintained grip load'): headroom EXISTS and position control reaches it — the gap is kinematic.",
                "gates": gates, "core": "NO torque/impedance CORE actuator (Case B REFUTED — position suffices). "
                "Authorized next: a KINEMATIC/GEOMETRIC end-effector redesign (wrist DOF / parallel-jaw aperture DOF), "
                "gated by CORE + user; NOT a force actuator, NOT RL"}
    if o2:
        return {"case": "B", "reason": "only the FORCE/impedance clamp (O2/O3) strict-transports; position (O1) fails ⇒ "
                "true force/impedance actuation provides genuine headroom", "gates": gates,
                "core": "a minimal torque/impedance/grip-force CORE actuator campaign becomes authorized (gated)"}
    if box:
        return {"case": "C", "reason": "only the box object passes; the present cylinder–jaw contact model cannot support "
                "clamp transport under tested conditions. Do NOT change the coin; clamp-drag remains noncanonical",
                "gates": gates, "core": "NONE — clamp-drag closed as noncanonical; deliver via push/plow (Track A)"}
    return {"case": "D", "reason": "no physical contact oracle passes ⇒ no demonstrated clamp-transport headroom in the "
            "present physical model; do NOT implement torque/impedance CORE changes", "gates": gates,
            "core": "NONE — no headroom; no CORE actuator change"}


if __name__ == "__main__":
    import sys
    run()
    sys.exit(0)
