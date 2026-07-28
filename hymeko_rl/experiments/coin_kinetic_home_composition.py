"""R9 HOME-start composition audit — can the frozen stack deliver strict K6 from a collision-free HOME posture (no learning)?

Extends the reproduced cradle-start result (R2 under H1 = 22/24 K6) to a stricter compositional benchmark. `HOME_STATE_V1` is a
legal home posture — both arms retracted to a fixed home q, qdot = 0, prev_tau = 0, NO contact, fresh history/recurrent state, coin
at the canonical s1 cradle. The decisive first question (before any RL): does the EXISTING frozen APPROACH reach the same
handoff-basin from HOME that R2 works from? If yes ⇒ `HOME_START_END_TO_END_K6_COMPOSITION_PASS` and no new RL. If no ⇒ only the
missing upstream skill (HOME → stable precontact/approach-entry) needs learning, with HANDOFF_RESET / R2 / release-coast / K6 /
physics / safety all frozen.

Gates: HOME_REACH (arms reach precontact / KINETIC entry), HOME_TO_HANDOFF (exactly one HANDOFF_RESET), HOME_TO_K6 (teacher-free
strict K6, no snapshot injection). All `8a0c1c7b`/`41510cac` modules imported unchanged; this audit MODIFIES no downstream policy.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_home_composition``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, delivery_success, primary_fingertip_contacts
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_handoff_reset import HandoffResetTemporalController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

OUT = Path("reports/2026-07-28-coin-r9-home-composition")
CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")
HOME_Q = np.array([-0.9, -1.4, 0.0, 2.7])        # HOME_STATE_V1: both arms retracted, no contact (tips ~130–160 mm from coin)


def build_home_state(cradle: Any) -> Any:
    """A `HOME_STATE_V1` snapshot: coin at the canonical s1 cradle, arms at the fixed home q, qdot = 0, prev_tau = 0, no contact.
    # Postconditions: branchable snapshot duck-typing the rollout interface; no straddle assumed (built via `from_live`)."""
    rl = cradle.branch()
    d, m = rl.inner.data, rl.inner.model
    coin = _coin_xy(cradle.branch())
    d.qpos[:4] = HOME_Q
    d.qpos[4:6] = coin
    d.qpos[6] = 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)
    return kc.TransportSnapshot.from_live(rl, cradle.stack, np.zeros(4))


def compose(home: Any, clone_factory: Any, r2_fn: Any) -> dict:
    """Roll the frozen chain (APPROACH → [HANDOFF_RESET] → R2) from HOME, teacher-free, no snapshot injection. Reports whether it
    reaches KINETIC / the handoff, and the K6 outcome."""
    ctrl = HandoffResetTemporalController(home, clone_factory(), r2_fn, ResidualBounds(alpha=0.15))
    m = velocity_rollout(home, ctrl, DELIVERY_CFG)
    kinds = [r["kind"] for r in ctrl.clone_trace]
    con = primary_fingertip_contacts(home.branch())
    reached_kinetic = "KINETIC_CLONE" in kinds
    return {"home_contacts": [con["left"] is not None, con["right"] is not None],
            "kind_counts": {k: kinds.count(k) for k in sorted(set(kinds))}, "reached_kinetic": bool(reached_kinetic),
            "n_handoff_reset": kinds.count("HANDOFF_RESET"), "min_dtz_mm": round(_min_dtz_mm(home, m), 2),
            "k6": bool(delivery_success(m, DELIVERY_CFG)), "k6_dwell": int(m["k6_max_dwell"]),
            "dtz_end_mm": round(m["dtz_end"] * 1000, 2),
            "safe": bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)}


def run(out: Path = OUT) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    ck = json.load(open(CKPT))
    r2_fn = deterministic_residual(_rebuild(ck["r2_actor_state"]))

    def cf() -> CloneActor:
        return CloneActor(model, norm)
    home = build_home_state(cradle)
    res = compose(home, cf, r2_fn)
    gates = {"HOME_REACH_PASS": bool(res["reached_kinetic"]),
             "HOME_TO_HANDOFF_PASS": bool(res["n_handoff_reset"] == 1 and res["reached_kinetic"]),
             "HOME_TO_K6_PASS": bool(res["k6"] and res["safe"])}
    verdict = ("HOME_START_END_TO_END_K6_COMPOSITION_PASS" if all(gates.values())
               else "HOME_START_COMPOSITION_NEEDS_UPSTREAM_REACH_SKILL")
    summary = {"contract": "HOME_STATE_V1", "immutable_source": "d55f5017", "home_q": HOME_Q.tolist(),
               "policy_checkpoint": str(CKPT), "composition": res, "gates": gates, "verdict": verdict}
    out.mkdir(parents=True, exist_ok=True)
    (out / "home_composition.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run()
    print(f"\nHOME COMPOSITION: {r['verdict']}")
    for k, v in r["gates"].items():
        print(f"  {'✅' if v else '❌'} {k} = {v}")
    c = r["composition"]
    print(f"  HOME contacts {c['home_contacts']} → kinds {c['kind_counts']}; reached_kinetic {c['reached_kinetic']}; "
          f"HANDOFF_RESET {c['n_handoff_reset']}; min_dtz {c['min_dtz_mm']}mm; K6 {c['k6']}")
