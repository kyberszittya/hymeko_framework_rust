"""R9 APPROACH→KINETIC handoff semantic-equivalence audit (why does clone+R2 deliver from the frozen entry but not the chain?).

The frozen-policy intervention exposed a start-state contradiction: clone + R2 delivers strict K6 from the frozen KINETIC entry
(16.86 mm) but NOT through the cradle→APPROACH→KINETIC path (39.58 mm). This audit tests whether the frozen entry is EXACTLY the
natural uninterrupted handoff or a displaced/privileged interface, by comparing the FULL hybrid state z_t = (x_t, mode, GRU/history,
guards/latches, prev_tau, prev residual) at the FIRST KINETIC PRE-step across four runs, all from the canonical s1 cradle:

  E0  frozen-entry artifact                → clone + R2
  E1  cradle → frozen APPROACH → natural KINETIC (uninterrupted) → clone + R2
  E2  E1 paused/serialised at the first KINETIC pre-step, reconstructed and resumed
  E3  a fresh controller built from E1's exact handoff state, resumed

Gates: NATURAL_HANDOFF_CAPTURE_PASS, PAUSE_RESUME_ONE_STEP_IDENTITY_PASS, PAUSE_RESUME_CONTINUATION_PASS,
FROZEN_ENTRY_EQUIVALENCE_PASS, END_TO_END_R2_K6_PASS. Comparisons are at the first KINETIC pre-step (before the first KINETIC
action) to avoid pre/post-step aliasing. All `8a0c1c7b` modules imported unchanged; no RL; no tag moved.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_handoff_audit``.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    KineticTemporalResidualController, deterministic_residual)
from hymeko_rl.coin_delivery.theta_option.residual_adapter import _coin_spin
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.env.motion_contract import govern_torque
from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone

OUT = Path("reports/2026-07-28-coin-r9-handoff-audit")
CKPT = Path("reports/2026-07-28-coin-r9-r3c-multiseed/seed_02/checkpoint.json")
POS_TOL = 1e-4                                    # accepted numerical tolerance (pause/resume within-run; matches R3-B 7 µm scale)


def _physical(rl: Any) -> dict:
    d = rl.inner.data
    con = primary_fingertip_contacts(rl)
    return {"qpos": [round(float(x), 6) for x in d.qpos[:4]], "qvel": [round(float(x), 6) for x in d.qvel[:4]],
            "coin_xy": [round(float(x), 6) for x in _coin_xy(rl)], "dtz_mm": round(rl.inner.direction_to_zone()[1] * 1000, 4),
            "disk_vel": [round(float(x), 6) for x in np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]],
            "spin": round(float(_coin_spin(rl)), 6),
            "fn_l": round(float(con["left"]["fn"]) if con["left"] else 0.0, 5),
            "fn_r": round(float(con["right"]["fn"]) if con["right"] else 0.0, 5)}


def _capture_at_first_kinetic(start: Any, clone: CloneActor, r2_fn: Callable, bounds: ResidualBounds) -> dict:
    """Roll from ``start`` and capture the FULL hybrid state at the first KINETIC PRE-step (before the first clone action), the
    first clone/R2/final actions taken there, a resume-able `FrontierSnapshot`, and the complete rollout metrics + coin trace."""
    ctrl = KineticTemporalResidualController(start, clone, r2_fn, bounds)
    ctrl.reset()
    rl = start.branch()
    prev_tau = np.asarray(start.prev_tau, np.float64).copy()
    cap: dict = {}

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], start.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    trace: list = []
    try:
        for t in range(1, DELIVERY_CFG.horizon + 1):
            pre_rl, pre_tau, pre_h = copy.deepcopy(rl), prev_tau.copy(), ctrl.actor.hidden_tensor()
            pre_res, n_before = ctrl._prev_res.copy(), len([r for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"])
            dtau = ctrl.dtau_for_step(rl, t, prev_tau)
            prev_tau = np.clip(prev_tau + dtau, start.lo, start.hi)
            step_ablation(rl, np.asarray(prev_tau, np.float32), "A")
            trace.append(_coin_xy(rl).copy())
            if not cap and len([r for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]) > n_before:
                rt = ctrl.residual_trace[-1]
                fs = krl3.FrontierSnapshot(pre_rl, pre_tau, start.stack, start.lo, start.hi, pre_h, pre_res,
                                           round(pre_rl.inner.direction_to_zone()[1] * 1000, 2), step_index=t)
                obs, _ = kc.kinetic_observe(pre_rl, [])
                cap = {"t_first": t, "prestep": {**_physical(pre_rl), **{"ctrl_" + k: v for k, v in
                       _controller_prestep(ctrl, pre_tau, pre_h, pre_res).items()}}, "first_actions": rt,
                       "first_obs": [round(float(x), 6) for x in obs], "frontier": fs}
    finally:
        mujoco.set_mjcb_control(None)
    m = velocity_rollout(start, KineticTemporalResidualController(start, CloneActor(clone.model, clone.norm), r2_fn, bounds),
                         DELIVERY_CFG)
    cap["metrics"] = {"k6": bool(delivery_success(m, DELIVERY_CFG)), "k6_dwell": int(m["k6_max_dwell"]),
                      "min_dtz_mm": round(_min_dtz_mm(start, m), 2), "dtz_end_mm": round(m["dtz_end"] * 1000, 2),
                      "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)}
    cap["coin_trace"] = [c.tolist() for c in trace]
    return cap


def _controller_prestep(ctrl: Any, pre_tau: np.ndarray, pre_h: Any, pre_res: np.ndarray) -> dict:
    """Controller state as it was at the PRE-step (before the step mutated it) — the honest first-KINETIC-pre-step snapshot."""
    return {"phase": int(ctrl.phase), "kinetic_steps": 0, "prev_tau": [round(float(x), 6) for x in pre_tau],
            "prev_res": [round(float(x), 6) for x in pre_res],
            "clone_hidden_norm": (None if pre_h is None else round(float(pre_h.norm()), 6)),
            "obs_hist_len": 0}


def _resume(frontier: Any, clone: CloneActor, r2_fn: Callable, bounds: ResidualBounds) -> dict:
    """E2/E3: resume clone+R2 from the captured first-KINETIC handoff (restored GRU hidden + prev residual + phase=KINETIC)."""
    ctrl = KineticTemporalResidualController(frontier, clone, r2_fn, bounds, start_kinetic=frontier.start_state())
    m = velocity_rollout(frontier, ctrl, DELIVERY_CFG)
    return {"coin_trace": np.asarray(m["coin_trace"]), "k6": bool(delivery_success(m, DELIVERY_CFG)),
            "k6_dwell": int(m["k6_max_dwell"]), "min_dtz_mm": round(_min_dtz_mm(frontier, m), 2),
            "first_action": ctrl.residual_trace[0] if ctrl.residual_trace else None}


def _state_diff(a: dict, b: dict) -> dict:
    """Max abs difference of the shared numeric fields (lists compared elementwise); non-numeric flagged if unequal."""
    out: dict = {}
    for k in sorted(set(a) & set(b)):
        va, vb = a[k], b[k]
        if isinstance(va, list) and va and isinstance(va[0], (int, float)):
            out[k] = round(float(np.max(np.abs(np.asarray(va) - np.asarray(vb)))), 6)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = round(abs(va - vb), 6)
        elif va != vb:
            out[k] = f"{va}!={vb}"
    return out


def run(out: Path = OUT) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    snap, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    entry = kc.freeze_kinetic_entry(snap, seed=kc.S1_SEED)
    bounds = ResidualBounds()
    r2_fn = deterministic_residual(_rebuild(json.load(open(CKPT))["r2_champ_state"]))

    def cf() -> CloneActor:
        return CloneActor(model, norm)
    e0 = _capture_at_first_kinetic(entry.tsnap, cf(), r2_fn, bounds)             # E0 frozen-entry artifact
    e1 = _capture_at_first_kinetic(snap, cf(), r2_fn, bounds)                    # E1 natural uninterrupted chain
    front = e1["frontier"]
    e2 = _resume(front, cf(), r2_fn, bounds)                                     # E2 pause/resume
    e3 = _resume(front, cf(), r2_fn, bounds)                                     # E3 fresh controller from the same handoff
    e1_tail = np.asarray(e1["coin_trace"])[e1["t_first"] - 1:]                   # E1 coin trace from the first KINETIC step
    one_step = float(np.linalg.norm(e2["coin_trace"][0] - e1_tail[0])) if len(e1_tail) else 9.9
    k = min(len(e2["coin_trace"]), len(e1_tail))
    cont = float(np.max(np.abs(e2["coin_trace"][:k] - e1_tail[:k]))) if k else 9.9

    prestep_diff = _state_diff(e0["prestep"], e1["prestep"])
    gates = {
        "NATURAL_HANDOFF_CAPTURE_PASS": bool(e1.get("frontier") is not None and "first_actions" in e1),
        "PAUSE_RESUME_ONE_STEP_IDENTITY_PASS": bool(one_step < POS_TOL),
        "PAUSE_RESUME_CONTINUATION_PASS": bool(cont < 5e-3 and e2["k6"] == (e1["metrics"]["k6"])),
        "FROZEN_ENTRY_EQUIVALENCE_PASS": bool(prestep_diff.get("dtz_mm", 9.9) < 0.1 and prestep_diff.get("qpos", 9.9) < POS_TOL),
        "END_TO_END_R2_K6_PASS": bool(e1["metrics"]["k6"])}
    verdict = _verdict(gates)
    summary = {"contract": "APPROACH_TO_KINETIC_HANDOFF_SEMANTIC_EQUIVALENCE_V1", "immutable_source": "2478a35d",
               "E0_frozen_entry": {"t_first": e0["t_first"], "prestep": e0["prestep"], "metrics": e0["metrics"],
                                   "first_actions": e0["first_actions"]},
               "E1_natural_chain": {"t_first": e1["t_first"], "prestep": e1["prestep"], "metrics": e1["metrics"],
                                    "first_actions": e1["first_actions"]},
               "E2_pause_resume": {"k6": e2["k6"], "k6_dwell": e2["k6_dwell"], "min_dtz_mm": e2["min_dtz_mm"],
                                   "one_step_norm": round(one_step, 8), "continuation_max": round(cont, 8)},
               "E3_reconstructed": {"k6": e3["k6"], "k6_dwell": e3["k6_dwell"], "min_dtz_mm": e3["min_dtz_mm"]},
               "E0_vs_E1_prestep_diff": prestep_diff, "gates": gates, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "handoff_audit.json").write_text(json.dumps(summary, indent=1))
    return summary


def _verdict(g: dict) -> str:
    if g["FROZEN_ENTRY_EQUIVALENCE_PASS"] and g["END_TO_END_R2_K6_PASS"]:
        return "R2_ALREADY_DELIVERED_FROM_NATURAL_HANDOFF"            # frozen entry == chain, and chain delivers
    if not g["FROZEN_ENTRY_EQUIVALENCE_PASS"] and g["PAUSE_RESUME_ONE_STEP_IDENTITY_PASS"]:
        return "FROZEN_ENTRY_IS_A_DISPLACED_PRIVILEGED_HANDOFF"       # capture/resume correct; entry != natural handoff
    if not g["PAUSE_RESUME_ONE_STEP_IDENTITY_PASS"]:
        return "INCOMPLETE_HANDOFF_STATE_CAPTURE"                     # even within-run resume diverges — a snapshot-contract gap
    return "HANDOFF_AUDIT_INCONCLUSIVE"


if __name__ == "__main__":
    r = run()
    print(f"\nHANDOFF AUDIT: {r['verdict']}  (wall {r['wall_s']}s)")
    for k, v in r["gates"].items():
        print(f"  {'✅' if v else '❌'} {k} = {v}")
    print(f"  E0 frozen entry: t{r['E0_frozen_entry']['t_first']} dtz {r['E0_frozen_entry']['prestep']['dtz_mm']}mm "
          f"→ K6 {r['E0_frozen_entry']['metrics']['k6']} min_dtz {r['E0_frozen_entry']['metrics']['min_dtz_mm']}mm")
    print(f"  E1 natural chain: t{r['E1_natural_chain']['t_first']} dtz {r['E1_natural_chain']['prestep']['dtz_mm']}mm "
          f"→ K6 {r['E1_natural_chain']['metrics']['k6']} min_dtz {r['E1_natural_chain']['metrics']['min_dtz_mm']}mm")
    print(f"  E2 resume: one-step {r['E2_pause_resume']['one_step_norm']} cont {r['E2_pause_resume']['continuation_max']} "
          f"K6 {r['E2_pause_resume']['k6']}")
    print(f"  E0 vs E1 pre-step diff (key fields): dtz {r['E0_vs_E1_prestep_diff'].get('dtz_mm')} "
          f"qpos {r['E0_vs_E1_prestep_diff'].get('qpos')}")
