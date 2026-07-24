"""Freeze COIN_DYNAMICS_CONTRACT_V4_CONTACT_AGILE = V3_AGILE + active-braking governor (over_hard_brake) + a
task-independent SUSTAINED-CONTACT gate. Selection lexicographic, NEVER delivery. After freeze, NO V4 tuning on K6/
zone-entry — that would destroy the clean separation.

Sustained-contact gate (neutral, delivery-free): drive the arm with the shared GovernedArm PD toward a target BEYOND the
coin so it pushes SUSTAINEDLY against the passive coin (repeated contact loading), then release; measure peak + integrated
overspeed during contact, return-below-soft time, and post-release settling. The bounded sweep is over ``over_hard_brake``
only (the V3 base is already frozen) — least-intervention selection: the smallest over_hard_brake that passes the gate.
The frozen artifact records dynamics+controller params, the full A1–A6 + sustained-contact table, traces, control rate/
substeps, and the GovernedArm source hash.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.env.governed_arm import GovernedArm, V3Stack  # noqa: E402
from hymeko_rl.env.motion_contract import MotionLimits  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
N, LIM = 4, MotionLimits()


def _sustained_contact_gate(pi0, base, forbidden, stack: V3Stack, n_states=4):
    """Drive the arm into and through the coin (sustained push) via the shared PD, then release. Delivery-free — we
    measure ONLY motion: peak + integrated overspeed during contact, return-below-soft, post-release settling."""
    peak, integ, contact_seen, returned, settled = 0.0, 0.0, 0, True, True
    for si in range(n_states):
        rl, _g = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None, seed_lo=14000 + 300 * si, tries=2)
        m, d = rl.inner.model, rl.inner.data
        with GovernedArm(m, d, stack, n=N) as arm:
            q0 = d.qpos[:N].copy()
            push_to = q0 + 0.6                                   # a target BEYOND the coin ⇒ sustained contact loading
            for _ in range(120):
                arm.pd_step(push_to)
                v = float(np.max(np.abs(d.qvel[:N])))
                peak = max(peak, v)
                integ += max(0.0, v / LIM.joint_vel_safe - 1.0) * stack.control_dt
                contact_seen += int(bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact))
            for k in range(120):                                 # release: DISENGAGE toward home, then settle freely
                arm.pd_step(q0)
                if np.max(np.abs(d.qvel[:N])) < LIM.terminal_joint_vel:
                    break
            settled = settled and (np.max(np.abs(d.qvel[:N])) < LIM.terminal_joint_vel * 2)
        returned = returned and (peak <= LIM.joint_vel_hard * 1.15)
    return {"contact_peak_vel": round(peak, 2), "integrated_overspeed": round(integ, 3), "contact_frames": contact_seen,
            "post_release_settled": bool(settled),
            "ok": bool(peak <= LIM.joint_vel_hard * 1.15 and integ < 0.4 and settled and contact_seen > 0)}


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    v3 = json.load(open(f"{OUT}/dynamics_contract_v3_agile.json"))["frozen_contract"]
    gh = hashlib.sha256(open("hymeko_rl/env/governed_arm.py", "rb").read()).hexdigest()[:12]
    base_kw = dict(qdot_soft=v3["qdot_soft"], qdot_hard=v3["qdot_hard"], armature=v3["armature"],
                   damping=v3["damping"], friction=v3["friction"], kp=v3["kp"], tau_rate=v3["tau_rate"])
    # bounded local sweep around V3: active braking + PD velocity gain (kv) — the settling gate needs a well-damped PD
    # (V3's kv=12 at kp=120 is underdamped). Lexicographic: pass the gate, then least over_hard_brake, then least kv.
    table, chosen = [], None
    for ohb in (1.5, 2.0, 3.0):
        for kv in (12.0, 24.0, 36.0):
            stack = V3Stack(**base_kw, kv=kv, over_hard_brake=ohb)
            g = _sustained_contact_gate(pi0, base, forbidden, stack)
            table.append({"over_hard_brake": ohb, "kv": kv, "sustained_contact": g})
            print(f"ohb {ohb} kv {kv}: contact_peak {g['contact_peak_vel']} int_over {g['integrated_overspeed']} "
                  f"settled {g['post_release_settled']} PASS={g['ok']}", flush=True)
            if g["ok"] and chosen is None:
                chosen = (ohb, kv)
    frozen, verdict = None, "NO_CONTACT_ROBUST_CONFIG_FOUND"
    if chosen is not None:
        ohb, kv = chosen
        frozen = {"dynamics_contract": "COIN_DYNAMICS_CONTRACT_V4_CONTACT_AGILE", "based_on": "V3_AGILE",
                  "governor_arm_source_sha": gh, **base_kw, "kv": kv, "over_hard_brake": ohb,
                  "control_dt": 0.01, "substeps": 20, "joint_vel_hard": LIM.joint_vel_hard,
                  "joint_vel_safe": LIM.joint_vel_safe, "terminal_joint_vel": LIM.terminal_joint_vel}
        verdict = "V4_CONTACT_AGILE_FROZEN"
    json.dump({"contract": "COIN_DYNAMICS_CONTRACT_V4_FREEZE", "date": "2026-07-25",
               "discipline": "V3 base + sustained-contact gate; least-intervention over_hard_brake; NO delivery in selection; FROZEN — no K6 tuning after",
               "v3_base": v3, "sustained_contact_table": table, "frozen_contract": frozen, "verdict": verdict},
              open(f"{OUT}/dynamics_contract_v4.json", "w"), indent=1, default=float)
    print(f"\n→ {verdict}" + (f"  (over_hard_brake {chosen}, damp {frozen['damping']}, gov {frozen['qdot_hard']})" if frozen else ""))
    print(f"artifact: {OUT}/dynamics_contract_v4.json\nCOIN_DYNAMICS_V4_DONE")
    return frozen is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
