"""R9 explicit-handoff-reset audit — legitimise the frozen-entry interface as an ONLINE end-to-end controller mode.

Runs the two boundary contracts (`kinetic_handoff_reset`) end-to-end from the canonical s1 cradle, teacher-free, with no
interruption or snapshot injection, for clone+R2 AND the FULL C1 policy:

  H0  DIRECT_HANDOFF          (the natural chain; the audit's E1)
  H1  EXPLICIT_HANDOFF_RESET  (one explicit online transition-servo step, then the policy)

Gates: HANDOFF_RESET_EXPLICIT_PASS (the reset is a distinct mode/event with its own trace, before the first policy step),
ONLINE_FROZEN_ENTRY_EQUIVALENCE_PASS (the online post-reset state equals the offline frozen entry), END_TO_END_R2_K6_UNDER_
EXPLICIT_RESET (the full cradle→APPROACH→HANDOFF_RESET→clone+R2→coast→K6 chain reaches strict K6). All `8a0c1c7b` modules imported
unchanged; no tag moved; no RL.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_handoff_reset_audit``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.coin_strict_markov_ablation import step_ablation
from hymeko_rl.coin_delivery.forward_displacement import delivery_success
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_handoff_reset as hr
from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import C1_BETA
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import KineticTemporalResidualController, deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.env.motion_contract import govern_torque
from hymeko_rl.experiments.coin_kinetic_ablation import CKPT_DIR, _rebuild
from hymeko_rl.experiments.coin_kinetic_handoff_audit import _physical
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone

OUT = Path("reports/2026-07-28-coin-r9-handoff-reset")
CKPT = CKPT_DIR / "seed_02" / "checkpoint.json"
EQ_TOL = 5e-3                                     # online post-reset vs offline frozen-entry (same servo step; near-exact)


def _metrics(start: Any, ctrl: Any) -> dict:
    m = velocity_rollout(start, ctrl, DELIVERY_CFG)
    n_reset = len([r for r in ctrl.clone_trace if r["kind"] == "HANDOFF_RESET"])
    kinds = [r["kind"] for r in ctrl.clone_trace]
    reset_before_policy = ("HANDOFF_RESET" in kinds and "KINETIC_CLONE" in kinds
                           and kinds.index("HANDOFF_RESET") < kinds.index("KINETIC_CLONE")) if n_reset else False
    return {"k6": bool(delivery_success(m, DELIVERY_CFG)), "k6_dwell": int(m["k6_max_dwell"]),
            "min_dtz_mm": round(_min_dtz_mm(start, m), 2), "dtz_end_mm": round(m["dtz_end"] * 1000, 2),
            "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5), "n_handoff_reset": n_reset,
            "reset_before_policy": bool(reset_before_policy)}


def _post_reset_state(snap: Any, ctrl_factory: Callable[[Any], Any]) -> dict:
    """Roll H1 from the cradle and capture the physical state immediately AFTER the HANDOFF_RESET step (= the online frozen
    entry), before the first policy action — mirroring the frozen kernel so the capture is causal."""
    ctrl = ctrl_factory(snap)
    ctrl.reset()
    rl = snap.branch()
    prev_tau = np.asarray(snap.prev_tau, np.float64).copy()

    def _gcb(_mo: Any, dt: Any) -> None:
        dt.ctrl[:4] = govern_torque(dt.ctrl[:4], dt.qvel[:4], snap.stack.gov)
    mujoco.set_mjcb_control(_gcb)
    cap: dict = {}
    try:
        for t in range(1, DELIVERY_CFG.horizon + 1):
            n_before = len([r for r in ctrl.clone_trace if r["kind"] == "HANDOFF_RESET"])
            dtau = ctrl.dtau_for_step(rl, t, prev_tau)
            prev_tau = np.clip(prev_tau + dtau, snap.lo, snap.hi)
            step_ablation(rl, np.asarray(prev_tau, np.float32), "A")
            if not cap and len([r for r in ctrl.clone_trace if r["kind"] == "HANDOFF_RESET"]) > n_before:
                cap = {**_physical(rl), "prev_tau": [round(float(x), 6) for x in prev_tau], "t_reset": int(t)}
                break
    finally:
        mujoco.set_mjcb_control(None)
    return cap


def _equiv(a: dict, b: dict) -> dict:
    keys = ("dtz_mm", "qpos", "qvel", "coin_xy", "disk_vel", "spin", "fn_l", "fn_r", "prev_tau")
    out: dict = {}
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if isinstance(va, list):
            out[k] = round(float(np.max(np.abs(np.asarray(va) - np.asarray(vb)))), 6)
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = round(abs(va - vb), 6)
    return out


def run(out: Path = OUT) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    snap, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    entry = kc.freeze_kinetic_entry(snap, seed=kc.S1_SEED)
    bounds = ResidualBounds()
    ck = json.load(open(CKPT))
    r2_fn = deterministic_residual(_rebuild(ck["r2_champ_state"]))
    exp_fn = deterministic_residual(_rebuild(ck["expansion_state"]))

    def cf() -> CloneActor:
        return CloneActor(model, norm)

    runs = {
        "H0_direct_cloneR2": _metrics(snap, KineticTemporalResidualController(snap, cf(), r2_fn, bounds)),
        "H1_reset_cloneR2": _metrics(snap, hr.HandoffResetTemporalController(snap, cf(), r2_fn, bounds)),
        "H0_direct_FULL": _metrics(snap, _unlock(snap, cf(), exp_fn, r2_fn, bounds, reset=False)),
        "H1_reset_FULL": _metrics(snap, _unlock(snap, cf(), exp_fn, r2_fn, bounds, reset=True)),
        "E0_frozen_cloneR2": _metrics(entry.tsnap, KineticTemporalResidualController(entry.tsnap, cf(), r2_fn, bounds))}
    online = _post_reset_state(snap, lambda s: hr.HandoffResetTemporalController(s, cf(), r2_fn, bounds))
    frozen = {**_physical(entry.tsnap.branch()), "prev_tau": [round(float(x), 6) for x in entry.tsnap.prev_tau]}
    equiv = _equiv(online, frozen)

    gates = {
        "HANDOFF_RESET_EXPLICIT_PASS": bool(runs["H1_reset_cloneR2"]["n_handoff_reset"] == 1
                                            and runs["H1_reset_cloneR2"]["reset_before_policy"]),
        "ONLINE_FROZEN_ENTRY_EQUIVALENCE_PASS": bool(equiv.get("dtz_mm", 9.9) < 0.5
                                                     and max(equiv.get(k, 9.9) for k in ("qpos", "prev_tau")) < EQ_TOL),
        "END_TO_END_R2_K6_UNDER_EXPLICIT_RESET_PASS": bool(runs["H1_reset_cloneR2"]["k6"]
                                                           and runs["H1_reset_cloneR2"]["safe"])}
    verdict = _verdict(runs, gates)
    summary = {"contract": "EXPLICIT_HANDOFF_RESET_ONLINE_EQUIVALENCE_V1", "immutable_source": "85c5eca6",
               "beta": C1_BETA, "runs": runs, "online_post_reset": online, "frozen_entry": frozen,
               "online_vs_frozen_equiv": equiv, "gates": gates, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "handoff_reset_audit.json").write_text(json.dumps(summary, indent=1))
    return summary


def _unlock(snap: Any, clone: CloneActor, exp_fn: Callable, r2_fn: Callable, bounds: ResidualBounds, *, reset: bool) -> Any:
    cls = hr.HandoffResetUnlockController if reset else _plain_unlock_cls()
    return cls(snap, clone, exp_fn, bounds, r2_fn=r2_fn, beta=C1_BETA)


def _plain_unlock_cls() -> Any:
    from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import AuthorityUnlockController
    return AuthorityUnlockController


def _verdict(runs: dict, gates: dict) -> str:
    if not gates["END_TO_END_R2_K6_UNDER_EXPLICIT_RESET_PASS"] and not runs["H1_reset_FULL"]["k6"]:
        return "NATURAL_HANDOFF_RL_CLOSURE_REQUIRED"                  # neither delivers end-to-end under either contract
    if gates["END_TO_END_R2_K6_UNDER_EXPLICIT_RESET_PASS"]:
        return "R2_IS_FIRST_LEARNED_K6_UNDER_EXPLICIT_HANDOFF_RESET"  # clone+R2 delivers e2e; R3-C = dwell/robustness refinement
    return "ONLY_FULL_DELIVERS_UNDER_EXPLICIT_RESET"                  # expansion load-bearing on the natural chain


if __name__ == "__main__":
    r = run()
    print(f"\nHANDOFF-RESET AUDIT: {r['verdict']}  (wall {r['wall_s']}s)")
    for k, v in r["gates"].items():
        print(f"  {'✅' if v else '❌'} {k} = {v}")
    for name, m in r["runs"].items():
        print(f"  {name:22s} min_dtz {m['min_dtz_mm']:6.2f}mm  K6 {str(m['k6']):5s}  dwell {m['k6_dwell']:2d}  "
              f"reset {m['n_handoff_reset']}  safe {m['safe']}")
    print(f"  online-post-reset vs frozen-entry: dtz Δ{r['online_vs_frozen_equiv'].get('dtz_mm')} "
          f"qpos Δ{r['online_vs_frozen_equiv'].get('qpos')} prev_tau Δ{r['online_vs_frozen_equiv'].get('prev_tau')}")
