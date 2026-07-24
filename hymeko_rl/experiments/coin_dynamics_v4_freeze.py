"""Freeze COIN_DYNAMICS_CONTRACT_V4_CONTACT_AGILE = V3_AGILE + active-braking governor (over_hard_brake) + a
task-independent GENUINE-SUSTAINED-CONTACT gate. Selection lexicographic, NEVER delivery. After freeze, NO V4 tuning on
K6/zone-entry — that would destroy the clean dynamics↔control separation.

Driver correction (2026-07-25): the earlier ``GovernedArm.pd_step(q0+const)`` gate recorded contact_frames==0 — it drove
raw ``mj_step`` and BYPASSED the coin env's ``step_ablation``, so ``_planar_metrics`` never updated and the "gate" measured
a FREE-SPACE swing (that whole V4 negative is VOID). The corrected driver is ``motion_robust_carry`` (the C1 controller)
with **braking OFF** — it acquires genuine contact through ``step_ablation`` and then keeps PUSHING (pure transport), the
WORST case for the actuation stack: velocity is held legal by the GOVERNOR + over_hard_brake ALONE (no high-level brake).
It is delivery-agnostic — it may approach and load the coin, but K6 / zone-entry NEVER enter the gate or the selection.

Design: over_hard_brake's active-braking branch is DORMANT below qdot_hard (verified: govern_torque zeroes no torque
below hard), so it changes NOTHING in the free-space / agility / tracking / reversal / settling regime — all of which run
below qdot_hard. Therefore the frozen V3 gates are inherited BY CONSTRUCTION; the only new intervention swept is
over_hard_brake (kv is held at the V3-frozen value). Lexicographic selection: all hard+agility gates PASS (inherited) →
sustained-contact PASS → least intervention (smallest over_hard_brake) → least contact overspeed integral.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from hymeko_rl.coin_delivery.motion_robust_expert import CarryControllerConfig, motion_robust_carry  # noqa: E402
from hymeko_rl.env.governed_arm import GovernedArm, V3Stack  # noqa: E402
from hymeko_rl.env.motion_contract import MotionLimits  # noqa: E402
from hymeko_rl.experiments.video_coin_variants import _reconstruct, _setup  # noqa: E402

OUT = "reports/2026-07-25-coin-dynamics-contract-v2"
N, LIM = 4, MotionLimits()

# ── PRE-DECLARED thresholds (fixed BEFORE observing any result; NEVER changed post-hoc to force a pass) ──────────────
NOMINAL_HARD = LIM.joint_vel_hard          # 3.0 rad/s — nominal hard velocity limit
TRANSIENT_TOL = 0.15                        # 15% allowed transient overshoot (pre-declared in the contract)
ABS_GATE = NOMINAL_HARD * (1.0 + TRANSIENT_TOL)   # 3.45 rad/s — absolute contact-phase peak-velocity gate
INTEG_GATE = 0.4                            # integrated-overspeed ceiling (contact frames)
SAFE = LIM.joint_vel_safe                   # 2.0 rad/s — the safe band; recovery must return below this
MIN_CONTACT_FRAMES = 10                     # genuine sustained contact: ≥10 control frames …
MIN_CONTACT_FRAC = 0.25                     # … OR ≥25% of the loading window in contact
RECOVERY_MAX = 200                          # control steps allowed to return below SAFE after unloading
SAT_FRAC_GATE = 0.5                         # no PERSISTENT torque saturation (≤50% of control steps)
GATE_HORIZON = 200                          # loading-window length (acquire + sustained transport push)
OHB_SWEEP = (1.5, 2.0, 3.0)                 # the ONLY swept intervention (active-braking gain)


def _recovery_to_safe(rl, stack: V3Stack, q0):
    """After the sustained load, DISENGAGE to home through the shared GovernedArm and count control steps until the arm
    returns below the safe band. Reads only ``d.qvel`` (updated by mj_step), so raw GovernedArm driving is valid here."""
    m, d = rl.inner.model, rl.inner.data
    with GovernedArm(m, d, stack, n=N) as arm:
        for k in range(RECOVERY_MAX):
            arm.pd_step(q0)
            if float(np.max(np.abs(d.qvel[:N]))) < SAFE:
                return k + 1, True
    return RECOVERY_MAX, False


def _sustained_contact_gate(pi0, base, forbidden, stack: V3Stack, n_states=4):
    """Certify the DYNAMICS in the genuine sustained-contact regime, delivery-free. Returns per-state contact-conditioned
    motion + the aggregate acceptance decision. K6 / zone are never read."""
    cfg = CarryControllerConfig(sustained_press=True, enable_braking=False, replan_every=4)   # PRESS into coin; governor-only vel ctrl
    per_state = []
    for si in range(n_states):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 300 * si, tries=2)
        q0 = rl.inner.data.qpos[:N].copy()
        o = motion_robust_carry(rl, gate, pi0, base, stack, horizon=GATE_HORIZON, cfg=cfg)
        rec_frames, rec_ok = _recovery_to_safe(rl, stack, q0)
        loaded = bool(o["contact_frames"] >= MIN_CONTACT_FRAMES or o["contact_frac"] >= MIN_CONTACT_FRAC)
        per_state.append({
            "state": si, "genuinely_loaded": loaded, "contact_frames": o["contact_frames"],
            "contact_frac": o["contact_frac"], "peak_joint_vel_whole": o["peak_joint_vel"],
            "peak_joint_vel_in_contact": o["peak_joint_vel_in_contact"],
            "integrated_overspeed_in_contact": o["integrated_overspeed_in_contact"],
            "peak_contact_normal_force": o["peak_contact_normal_force"], "terminal_joint_vel": o["terminal_joint_vel"],
            "recovery_frames": rec_frames, "recovery_time_s": round(rec_frames * stack.control_dt, 3), "recovery_ok": rec_ok,
            "governor_active_frac": o["governor_active_frac"], "active_brake_frac": o["active_brake_frac"],
            "torque_saturation_frac": o["torque_saturation_frac"]})
    n_loaded = sum(s["genuinely_loaded"] for s in per_state)
    n_any_contact = sum(s["contact_frames"] > 0 for s in per_state)
    genuine = n_loaded >= max(1, n_states // 2)                       # majority of states SUSTAIN contact (≥10f / ≥0.25 frac)
    any_contact = n_any_contact > 0                                  # contact occurred AT ALL (driver reaches the coin)
    # evaluate motion on the states that actually made contact (peak velocity ON CONTACT FRAMES only, per the directive)
    contact_states = [s for s in per_state if s["contact_frames"] > 0] or per_state
    peak_c = max(s["peak_joint_vel_in_contact"] for s in contact_states)
    integ_c = max(s["integrated_overspeed_in_contact"] for s in contact_states)
    recovery_ok = all(s["recovery_ok"] for s in per_state)
    sat_max = max(s["torque_saturation_frac"] for s in per_state)
    # "is the stack motion-legal WHERE contact occurs" — clean even under intermittent contact
    motion_legal_in_contact = bool(any_contact and peak_c <= ABS_GATE and integ_c < INTEG_GATE and recovery_ok)
    return {
        "genuine_contact": genuine, "any_contact": any_contact, "n_loaded_states": n_loaded,
        "n_contact_states": n_any_contact, "motion_legal_in_contact": motion_legal_in_contact, "n_states": n_states,
        "peak_vel_in_contact": round(peak_c, 2), "integrated_overspeed_in_contact": round(integ_c, 3),
        "peak_vel_whole": round(max(s["peak_joint_vel_whole"] for s in per_state), 2),
        "peak_contact_normal_force": round(max(s["peak_contact_normal_force"] for s in per_state), 2),
        "max_recovery_frames": max(s["recovery_frames"] for s in per_state),
        "recovery_all_ok": recovery_ok, "max_torque_saturation_frac": round(sat_max, 3),
        "governor_active_frac": round(float(np.mean([s["governor_active_frac"] for s in per_state])), 3),
        "active_brake_frac": round(float(np.mean([s["active_brake_frac"] for s in per_state])), 3),
        "per_state": per_state,
        # ACCEPTANCE: genuine sustained contact ∧ contact-phase peak ≤ ABS_GATE ∧ integ ≤ INTEG_GATE ∧ recovers to safe
        #             ∧ no persistent saturation. (V3 free-space/agility/tracking/reversal/settling inherited by
        #             construction — over_hard_brake is dormant below qdot_hard.)
        "ok": bool(genuine and peak_c <= ABS_GATE and integ_c < INTEG_GATE and recovery_ok and sat_max <= SAT_FRAC_GATE)}


def main():
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    pi0, base, forbidden = _setup()
    v3doc = json.load(open(f"{OUT}/dynamics_contract_v3_agile.json"))
    assert v3doc["verdict"] == "V3_AGILE_FROZEN", f"V3 not frozen: {v3doc['verdict']}"   # inherit only from a frozen V3
    v3 = v3doc["frozen_contract"]
    gh = hashlib.sha256(open("hymeko_rl/env/governed_arm.py", "rb").read()).hexdigest()[:12]
    base_kw = dict(qdot_soft=v3["qdot_soft"], qdot_hard=v3["qdot_hard"], armature=v3["armature"],
                   damping=v3["damping"], friction=v3["friction"], kp=v3["kp"], kv=v3["kv"], tau_rate=v3["tau_rate"])

    print(f"thresholds (pre-declared): ABS_GATE {ABS_GATE} (= {NOMINAL_HARD}·{1 + TRANSIENT_TOL}) | INTEG {INTEG_GATE} | "
          f"SAFE {SAFE} | genuine ≥{MIN_CONTACT_FRAMES} frames or ≥{MIN_CONTACT_FRAC} frac | recover ≤{RECOVERY_MAX} steps", flush=True)
    table, chosen, any_genuine, any_contact, all_legal = [], None, False, False, True
    for ohb in OHB_SWEEP:                                            # sweep ONLY the new intervention; kv held at V3-frozen
        stack = V3Stack(**base_kw, over_hard_brake=ohb)
        g = _sustained_contact_gate(pi0, base, forbidden, stack)
        any_genuine = any_genuine or g["genuine_contact"]
        any_contact = any_contact or g["any_contact"]
        all_legal = all_legal and g["motion_legal_in_contact"]
        table.append({"over_hard_brake": ohb, "kv": v3["kv"], "sustained_contact": g})
        print(f"ohb {ohb}: sustained {g['genuine_contact']} ({g['n_loaded_states']}/{g['n_states']}) | "
              f"any-contact {g['n_contact_states']}/{g['n_states']} | peak_contact {g['peak_vel_in_contact']} (≤{ABS_GATE}) | "
              f"integ {g['integrated_overspeed_in_contact']} | recover {g['max_recovery_frames']}f ok={g['recovery_all_ok']} | "
              f"sat {g['max_torque_saturation_frac']} | Fn {g['peak_contact_normal_force']} | legal-in-contact "
              f"{g['motion_legal_in_contact']} | PASS={g['ok']}", flush=True)
        if g["ok"] and chosen is None:
            chosen = ohb                                             # least-intervention: first (smallest) ohb that passes

    # Verdict ladder (per the overnight directive):
    #   no contact at all         → INVALID (harness: the driver never reached the coin)
    #   sustained + all gates PASS → FROZEN
    #   sustained but gates FAIL   → real negative (NOT contact-robust)
    #   contact occurs, motion legal where it occurs, but NOT sustained across the majority → PARTIAL, HALT for a design
    #     decision: the single-tip / low-friction disk cannot SUSTAIN continuous contact (the coin squirts away — which is
    #     why the coin TASK is push-and-coast / intermittent-contact). The pre-declared ≥10-frame/≥0.25-frac SUSTAINED
    #     criterion is mismatched to that intermittent regime; relaxing it now = changing a threshold after seeing results
    #     (forbidden). Stop the coin branch; do NOT freeze; do NOT launch C2.
    if not any_contact:
        verdict, frozen = "INVALID_GATE_NO_CONTACT_ESTABLISHED", None
    elif chosen is not None:
        frozen = {"dynamics_contract": "COIN_DYNAMICS_CONTRACT_V4_CONTACT_AGILE", "based_on": "V3_AGILE",
                  "governor_arm_source_sha": gh, **base_kw, "over_hard_brake": chosen,
                  "control_dt": 0.01, "substeps": 20,
                  "nominal_joint_vel_hard": NOMINAL_HARD, "transient_tolerance": TRANSIENT_TOL, "abs_velocity_gate": ABS_GATE,
                  "joint_vel_safe": SAFE, "terminal_joint_vel": LIM.terminal_joint_vel,
                  "v3_gates_inherited": "over_hard_brake dormant below qdot_hard ⇒ free-space/agility/tracking/reversal/settling identical to V3_AGILE_FROZEN"}
        verdict = "V4_CONTACT_AGILE_FROZEN"
    elif any_genuine:
        verdict, frozen = "CURRENT_ACTUATION_STACK_NOT_CONTACT_ROBUST_UNDER_REALISTIC_AGILE_LIMITS", None
    else:
        verdict = ("GENUINE_CONTACT_INTERMITTENT_MOTION_LEGAL_BUT_NOT_SUSTAINED__HALT_FOR_GATE_DECISION"
                   if all_legal else "GENUINE_CONTACT_INTERMITTENT_NOT_SUSTAINED__HALT_FOR_GATE_DECISION")
        frozen = None

    json.dump({"contract": "COIN_DYNAMICS_CONTRACT_V4_FREEZE", "date": "2026-07-25",
               "discipline": "V3 base + genuine-sustained-contact gate (motion_robust_carry, braking OFF, delivery-free); "
                             "least-intervention over_hard_brake; NO delivery in selection; FROZEN — no K6 tuning after",
               "pre_declared_thresholds": {"nominal_hard": NOMINAL_HARD, "transient_tolerance": TRANSIENT_TOL,
                                           "abs_velocity_gate": ABS_GATE, "integ_gate": INTEG_GATE, "safe": SAFE,
                                           "min_contact_frames": MIN_CONTACT_FRAMES, "min_contact_frac": MIN_CONTACT_FRAC,
                                           "recovery_max_steps": RECOVERY_MAX, "sat_frac_gate": SAT_FRAC_GATE},
               "v3_base": v3, "v3_verdict": v3doc["verdict"], "sustained_contact_table": table,
               "frozen_contract": frozen, "verdict": verdict},
              open(f"{OUT}/dynamics_contract_v4.json", "w"), indent=1, default=float)
    print(f"\n→ {verdict}" + (f"  (over_hard_brake {chosen}, damp {v3['damping']}, qdot_hard {v3['qdot_hard']}, kv {v3['kv']})" if frozen else ""))
    print(f"artifact: {OUT}/dynamics_contract_v4.json\nCOIN_DYNAMICS_V4_DONE")
    return frozen is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
