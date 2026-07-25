"""Freeze COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT = V3_AGILE + active-braking governor (over_hard_brake) + a
task-independent INTERMITTENT-CONTACT gate. Selection lexicographic, NEVER delivery.

Why intermittent, not sustained (user decision 2026-07-25): the earlier SUSTAINED-contact gate found genuine contact
(Fn ~17 N) that was MOTION-LEGAL where it occurred but lasted only ~6–8 frames — a single tip on a LOW-MASS, LOW-FRICTION
disk cannot hold continuous contact: the coin squirts away. That is not "the arm can't move it"; it is the correct
physical CONTACT CLASS of the coin task — push-and-coast / INTERMITTENT contact, not continuous-pressure grasp-transport.
Demanding sustained contact of a single-tip/low-friction disk would be certifying a grasp the geometry does not support.
The SUSTAINED / GRASP-TRANSPORT criterion is deferred to a SEPARATE future benchmark (two-sided clamp / gripper / concave
tip / graspable object). This gate certifies the regime the task ACTUALLY uses.

Intermittent-contact acceptance (all delivery-free — K6 / zone never read):
  * genuine contact occurs on the majority of states;
  * motion-legal in EVERY contact episode (contact-phase peak joint vel ≤ ABS_GATE, integrated overspeed < INTEG_GATE);
  * no runaway arm velocity (⊂ the peak gate);
  * object (coin) velocity bounded during contact (not launched);
  * arm returns below the safe band after unloading;
  * RE-CONTACT is stably executable (≥ MIN_EPISODES contact episodes across the panel);
  * terminal certificate — after settling, BOTH the arm and the coin are near-rest;
  * no persistent torque saturation.
Also REPORTED (force decomposition — a normal-force peak alone does not prove useful push): normal vs tangential force,
contact normal impulse. over_hard_brake is DORMANT below qdot_hard, so V3 free-space/agility/tracking/reversal/settling
gates are inherited BY CONSTRUCTION; only over_hard_brake is swept (kv held at the V3-frozen value). Lexicographic:
all gates PASS → least intervention (smallest over_hard_brake) → least contact overspeed integral.
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
COIN_SPEED_MAX = 1.0                        # m/s — object not LAUNCHED during contact (a controllable push is slow)
COIN_TERMINAL_MAX = 0.15                    # m/s — terminal object near-rest (≈3× terminal_ee_speed; settle certificate)
MIN_EPISODES = 2                            # re-contact demonstrated: ≥2 contact episodes across the contact states
RECOVERY_MAX = 200                          # control steps allowed to return below SAFE after unloading
SETTLE_STEPS = 60                           # steps to let the coin coast to rest after the arm disengages (terminal cert)
SAT_FRAC_GATE = 0.5                         # no PERSISTENT torque saturation (≤50% of control steps)
GATE_HORIZON = 200                          # contact-window length (acquire + intermittent oscillating push)
OHB_SWEEP = (1.5, 2.0, 3.0)                 # the ONLY swept intervention (active-braking gain)


def _recovery_and_settle(rl, stack: V3Stack, q0):
    """After the contact window, DISENGAGE the arm to home through the shared GovernedArm and count control steps until it
    returns below the safe band; keep holding home for a settle window so the (now-unloaded) coin coasts to rest, then read
    the terminal ARM velocity and terminal COIN speed. GovernedArm.pd_step does mj_step on the full model, so the coin
    evolves; ``_planar_metrics`` recomputes on read → the terminal coin speed is valid after settling."""
    m, d = rl.inner.model, rl.inner.data
    rec_frames, rec_ok = RECOVERY_MAX, False
    with GovernedArm(m, d, stack, n=N) as arm:
        for k in range(RECOVERY_MAX):
            arm.pd_step(q0)
            if not rec_ok and float(np.max(np.abs(d.qvel[:N]))) < SAFE:
                rec_frames, rec_ok = k + 1, True
                break
        for _ in range(SETTLE_STEPS):                                # hold home; the disengaged coin coasts to rest
            arm.pd_step(q0)
    term_arm = float(np.max(np.abs(d.qvel[:N])))
    term_coin = float(np.linalg.norm(np.asarray(rl.inner._planar_metrics.disk_vel, np.float32)[:2]))
    return rec_frames, rec_ok, round(term_arm, 3), round(term_coin, 3)


def _intermittent_contact_gate(pi0, base, forbidden, stack: V3Stack, n_states=4):
    """Certify the DYNAMICS in the genuine INTERMITTENT-contact regime, delivery-free. Returns per-state contact-
    conditioned motion + force decomposition + terminal certificate + the aggregate acceptance decision. K6/zone never read."""
    cfg = CarryControllerConfig(sustained_press=True, enable_braking=False, replan_every=4)   # oscillating push; governor-only vel ctrl
    per_state = []
    for si in range(n_states):
        rl, gate = _reconstruct(pi0, base, forbidden, coin_shape="cylinder", hx=0.020, hy=None,
                                seed_lo=14000 + 300 * si, tries=2)
        q0 = rl.inner.data.qpos[:N].copy()
        o = motion_robust_carry(rl, gate, pi0, base, stack, horizon=GATE_HORIZON, cfg=cfg)
        rec_frames, rec_ok, term_arm, term_coin = _recovery_and_settle(rl, stack, q0)
        per_state.append({
            "state": si, "contact_frames": o["contact_frames"], "contact_frac": o["contact_frac"],
            "n_contact_episodes": o["n_contact_episodes"], "peak_joint_vel_whole": o["peak_joint_vel"],
            "peak_joint_vel_in_contact": o["peak_joint_vel_in_contact"],
            "integrated_overspeed_in_contact": o["integrated_overspeed_in_contact"],
            "peak_contact_normal_force": o["peak_contact_normal_force"],
            "peak_contact_tangential_force": o["peak_contact_tangential_force"],
            "contact_normal_impulse": o["contact_normal_impulse"], "peak_coin_speed": o["peak_coin_speed"],
            "recovery_frames": rec_frames, "recovery_time_s": round(rec_frames * stack.control_dt, 3), "recovery_ok": rec_ok,
            "terminal_arm_vel": term_arm, "terminal_coin_speed": term_coin,
            "governor_active_frac": o["governor_active_frac"], "active_brake_frac": o["active_brake_frac"],
            "torque_saturation_frac": o["torque_saturation_frac"]})
    n_contact = sum(s["contact_frames"] > 0 for s in per_state)
    genuine = n_contact >= max(1, (n_states + 1) // 2)               # contact on the MAJORITY of states (driver valid)
    contact_states = [s for s in per_state if s["contact_frames"] > 0] or per_state
    peak_c = max(s["peak_joint_vel_in_contact"] for s in contact_states)
    integ_c = max(s["integrated_overspeed_in_contact"] for s in contact_states)
    peak_coin = max(s["peak_coin_speed"] for s in contact_states)
    total_episodes = sum(s["n_contact_episodes"] for s in per_state)
    recovery_ok = all(s["recovery_ok"] for s in per_state)
    term_arm_ok = all(s["terminal_arm_vel"] < SAFE for s in per_state)
    term_coin_ok = all(s["terminal_coin_speed"] < COIN_TERMINAL_MAX for s in per_state)
    sat_max = max(s["torque_saturation_frac"] for s in per_state)
    return {
        "genuine_contact": genuine, "n_contact_states": n_contact, "n_states": n_states,
        "peak_vel_in_contact": round(peak_c, 2), "integrated_overspeed_in_contact": round(integ_c, 3),
        "peak_vel_whole": round(max(s["peak_joint_vel_whole"] for s in per_state), 2),
        "peak_coin_speed": round(peak_coin, 3), "total_contact_episodes": total_episodes,
        "peak_contact_normal_force": round(max(s["peak_contact_normal_force"] for s in per_state), 2),
        "peak_contact_tangential_force": round(max(s["peak_contact_tangential_force"] for s in per_state), 2),
        "max_contact_normal_impulse": round(max(s["contact_normal_impulse"] for s in per_state), 4),
        "max_recovery_frames": max(s["recovery_frames"] for s in per_state), "recovery_all_ok": recovery_ok,
        "terminal_arm_ok": term_arm_ok, "terminal_coin_ok": term_coin_ok,
        "max_terminal_coin_speed": round(max(s["terminal_coin_speed"] for s in per_state), 3),
        "max_torque_saturation_frac": round(sat_max, 3),
        "governor_active_frac": round(float(np.mean([s["governor_active_frac"] for s in per_state])), 3),
        "active_brake_frac": round(float(np.mean([s["active_brake_frac"] for s in per_state])), 3),
        "per_state": per_state,
        # INTERMITTENT-CONTACT ACCEPTANCE: genuine contact (majority) ∧ every episode motion-legal (peak ≤ ABS_GATE,
        # integ < INTEG_GATE) ∧ object not launched (peak coin speed ≤ COIN_SPEED_MAX) ∧ arm recovers to safe ∧ re-contact
        # demonstrated (≥ MIN_EPISODES) ∧ terminal certificate (arm + coin near-rest) ∧ no persistent saturation.
        "ok": bool(genuine and peak_c <= ABS_GATE and integ_c < INTEG_GATE and peak_coin <= COIN_SPEED_MAX
                   and recovery_ok and total_episodes >= MIN_EPISODES and term_arm_ok and term_coin_ok
                   and sat_max <= SAT_FRAC_GATE)}


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
          f"SAFE {SAFE} | coin_speed ≤{COIN_SPEED_MAX} | coin_terminal <{COIN_TERMINAL_MAX} | ≥{MIN_EPISODES} episodes | "
          f"recover ≤{RECOVERY_MAX} steps", flush=True)
    table, chosen, any_contact = [], None, False
    for ohb in OHB_SWEEP:                                            # sweep ONLY the new intervention; kv held at V3-frozen
        stack = V3Stack(**base_kw, over_hard_brake=ohb)
        g = _intermittent_contact_gate(pi0, base, forbidden, stack)
        any_contact = any_contact or g["n_contact_states"] > 0
        table.append({"over_hard_brake": ohb, "kv": v3["kv"], "intermittent_contact": g})
        print(f"ohb {ohb}: genuine {g['genuine_contact']} ({g['n_contact_states']}/{g['n_states']}) | "
              f"episodes {g['total_contact_episodes']} | peak_contact {g['peak_vel_in_contact']} (≤{ABS_GATE}) | "
              f"integ {g['integrated_overspeed_in_contact']} | coin_pk {g['peak_coin_speed']} | recover "
              f"{g['max_recovery_frames']}f ok={g['recovery_all_ok']} | term arm={g['terminal_arm_ok']} coin={g['terminal_coin_ok']} "
              f"({g['max_terminal_coin_speed']}) | Fn {g['peak_contact_normal_force']} Ft {g['peak_contact_tangential_force']} "
              f"imp {g['max_contact_normal_impulse']} | sat {g['max_torque_saturation_frac']} | PASS={g['ok']}", flush=True)
        if g["ok"] and chosen is None:
            chosen = ohb                                             # least-intervention: first (smallest) ohb that passes

    # Verdict ladder:
    #   no contact at all          → INVALID (harness: the driver never reached the coin)
    #   intermittent gates PASS     → V4_INTERMITTENT_CONTACT_FROZEN
    #   contact occurs, gates FAIL  → real negative (stack not intermittent-contact-robust under these limits)
    if not any_contact:
        verdict, frozen = "INVALID_GATE_NO_CONTACT_ESTABLISHED", None
    elif chosen is not None:
        frozen = {"dynamics_contract": "COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT", "based_on": "V3_AGILE",
                  "contact_class": "INTERMITTENT (push-and-coast); SUSTAINED/grasp deferred to a separate gripping benchmark",
                  "governor_arm_source_sha": gh, **base_kw, "over_hard_brake": chosen,
                  "control_dt": 0.01, "substeps": 20,
                  "nominal_joint_vel_hard": NOMINAL_HARD, "transient_tolerance": TRANSIENT_TOL, "abs_velocity_gate": ABS_GATE,
                  "joint_vel_safe": SAFE, "coin_speed_max": COIN_SPEED_MAX, "coin_terminal_max": COIN_TERMINAL_MAX,
                  "min_contact_episodes": MIN_EPISODES, "terminal_joint_vel": LIM.terminal_joint_vel,
                  "v3_gates_inherited": "over_hard_brake dormant below qdot_hard ⇒ free-space/agility/tracking/reversal/settling identical to V3_AGILE_FROZEN"}
        verdict = "V4_INTERMITTENT_CONTACT_FROZEN"
    else:
        verdict, frozen = "ACTUATION_STACK_NOT_INTERMITTENT_CONTACT_ROBUST_UNDER_REALISTIC_AGILE_LIMITS", None

    json.dump({"contract": "COIN_DYNAMICS_CONTRACT_V4_INTERMITTENT_CONTACT_FREEZE", "date": "2026-07-25",
               "contact_class": "INTERMITTENT (push-and-coast) — the regime the coin task actually uses; SUSTAINED/grasp "
                                "is a SEPARATE future benchmark (two-sided clamp / gripper / concave tip / graspable object)",
               "discipline": "V3 base + intermittent-contact gate (motion_robust_carry oscillating press, delivery-free); "
                             "least-intervention over_hard_brake; NO delivery in selection; FROZEN — no K6 tuning after",
               "pre_declared_thresholds": {"nominal_hard": NOMINAL_HARD, "transient_tolerance": TRANSIENT_TOL,
                                           "abs_velocity_gate": ABS_GATE, "integ_gate": INTEG_GATE, "safe": SAFE,
                                           "coin_speed_max": COIN_SPEED_MAX, "coin_terminal_max": COIN_TERMINAL_MAX,
                                           "min_contact_episodes": MIN_EPISODES, "recovery_max_steps": RECOVERY_MAX,
                                           "settle_steps": SETTLE_STEPS, "sat_frac_gate": SAT_FRAC_GATE},
               "v3_base": v3, "v3_verdict": v3doc["verdict"], "intermittent_contact_table": table,
               "frozen_contract": frozen, "verdict": verdict},
              open(f"{OUT}/dynamics_contract_v4.json", "w"), indent=1, default=float)
    print(f"\n→ {verdict}" + (f"  (over_hard_brake {chosen}, damp {v3['damping']}, qdot_hard {v3['qdot_hard']}, kv {v3['kv']})" if frozen else ""))
    print(f"artifact: {OUT}/dynamics_contract_v4.json\nCOIN_DYNAMICS_V4_DONE")
    return frozen is not None


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
