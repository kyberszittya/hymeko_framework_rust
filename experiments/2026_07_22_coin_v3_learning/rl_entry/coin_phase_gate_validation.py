"""§4 PHASE_GATE_RUNTIME_CONTRACT validation. Roll the FROZEN pi_0 (and certified K=6 deliveries) through the canonical
neutral env, feed ONLY the deployable robot-attributed contact to the PhaseGate, and report the §4 statistics + verify
the required behaviours. The gate uses no seed / dtz / future info; dtz-derived PHASE labels here are DIAGNOSTIC ONLY.
"""
import glob
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_phase_gate import PhaseGate, PhaseGateConfig, robot_attributed_contact  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, load_frozen_clip_actor  # noqa: E402
from hymeko_rl.experiments.coin_neutral_start import neutral_env  # noqa: E402

PI0_CKPT = sys.argv[3] if len(sys.argv) > 3 else None
BASE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/phase_gate_val.json"
INIT_SUCCESS = {1011, 1447, 1568}
CENTER_TOL, ENTRY_TOL, SETTLE = 0.02, 0.05, 0.06


def load_pi0():
    """Load the IMMUTABLE frozen pi_0 from the persisted checkpoint (file-SHA prefix 1902454c) — not a rebuild."""
    pi0 = load_frozen_clip_actor(PI0_CKPT, freeze=True)
    sha = hashlib.sha256(open(PI0_CKPT, "rb").read()).hexdigest()
    return None, pi0, sha


def phase_label(touched_ever, dtz, strict):
    if not touched_ever:
        return "APPROACH"
    if dtz > ENTRY_TOL:
        return "TRANSPORT"
    if dtz > CENTER_TOL:
        return "TARGET_ENTRY"
    return "DWELL" if strict > 0 else "SETTLING"


def roll_gate(actor, seed, *, replay_actions=None, horizon=360):
    """Roll actor (or replay recorded actions) from neutral; drive the gate on deployable contact. Returns the trace."""
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    env.set_stage(0); env.reset(seed=int(seed))
    gate = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    ew = ActorEvalWrap(actor) if actor is not None else None
    trace = []
    touched_ever = False; strict = 0; first_contact = -1
    n = horizon if replay_actions is None else min(horizon, len(replay_actions))
    for t in range(n):
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        g_now = gate.gate                                   # multiplier the residual WOULD get for this action
        a = replay_actions[t] if replay_actions is not None else ew.act(nf)
        inner.step(np.clip(np.asarray(a, np.float32), -4, 4))
        m = inner._planar_metrics
        contact = robot_attributed_contact(inner)
        if contact and first_contact < 0:
            first_contact = t
        touched_ever = touched_ever or contact
        dtz = float(m.disk_to_zone)
        speed = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
        strict = strict + 1 if (dtz <= CENTER_TOL and speed < SETTLE) else 0
        delivered = strict >= 6 and touched_ever
        g_next = gate.update(contact, terminated=delivered)
        trace.append({"t": t, "contact": contact, "g_applied": g_now, "g_next": g_next,
                      "state": gate.state.value, "dtz": round(dtz, 4), "strict": strict,
                      "phase": phase_label(touched_ever, dtz, strict)})
        if delivered:
            break
    return trace, first_contact, gate


def stats(trace, first_contact):
    """§4 statistics for one trajectory."""
    active = [s for s in trace if s["state"] == "LATE_CONTROL_ARMED"]
    first_act = next((s["t"] for s in trace if s["g_next"] == 1.0), -1)
    # activation phase = phase at the step the gate first arms
    act_phase = next((s["phase"] for s in trace if s["g_next"] == 1.0), None)
    # false early activation: armed while no contact had ever occurred (before first_contact)
    false_early = sum(1 for s in trace if s["g_next"] == 1.0 and (first_contact < 0 or s["t"] < first_contact))
    # toggles = transitions of g between consecutive steps (chatter proxy)
    gs = [s["g_next"] for s in trace]
    toggles = sum(1 for i in range(1, len(gs)) if gs[i] != gs[i - 1])
    time_by_phase = {}
    for s in active:
        time_by_phase[s["phase"]] = time_by_phase.get(s["phase"], 0) + 1
    return {"len": len(trace), "first_contact": first_contact, "first_activation": first_act,
            "activation_phase": act_phase, "false_early_activations": false_early,
            "deactivations": sum(1 for s in trace if s["state"] == "REACQUIRE" and s["g_next"] == 0.0
                                 and s["g_applied"] == 1.0),
            "reacquire_entries": _count_state_entries(trace, "REACQUIRE"),
            "rearm_entries": max(0, _count_state_entries(trace, "LATE_CONTROL_ARMED") - 1),
            "toggles": toggles, "active_steps": len(active), "active_by_phase": time_by_phase,
            "activated": first_act >= 0}


def _count_state_entries(trace, name):
    prev = None; c = 0
    for s in trace:
        if s["state"] == name and prev != name:
            c += 1
        prev = s["state"]
    return c


def hysteresis_honored(trace, arm_after=3, disarm_after=2):
    """STRUCTURAL anti-chatter guarantee: every ARM transition was preceded by ``arm_after`` consecutive contact
    steps, every DISARM (→REACQUIRE) by ``disarm_after`` consecutive no-contact steps. No sub-window toggling."""
    contacts = [s["contact"] for s in trace]
    states = [s["state"] for s in trace]
    for i in range(1, len(states)):
        if states[i] == "LATE_CONTROL_ARMED" and states[i - 1] != "LATE_CONTROL_ARMED":
            if not all(contacts[max(0, i - arm_after + 1):i + 1]):        # last arm_after incl. this step all contact
                return False
        if states[i] == "REACQUIRE" and states[i - 1] == "LATE_CONTROL_ARMED":
            if any(contacts[max(0, i - disarm_after + 1):i + 1]):          # last disarm_after all no-contact
                return False
    return True


def main():
    bc, pi0, sha = load_pi0()
    gate0 = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    out = {"pi0_file_sha256": sha, "pi0_sha_prefix": sha[:8], "gate_contract": gate0.contract(),
           "gate_contract_sha256": gate0.contract_sha256(), "classes": {}}
    print(f"pi0 file SHA prefix = {sha[:8]} (expected 1902454c)", flush=True)
    print(f"gate contract SHA = {gate0.contract_sha256()[:16]}", flush=True)

    fail_seeds = [s for s in HEADLINE if s not in INIT_SUCCESS]
    certs = sorted(glob.glob(BASE + "/traj_*.npz"))[:5]

    def run_class(name, items, replay=False):
        rows = []
        for it in items:
            if replay:
                seed = int(it.split("traj_")[1].split(".npz")[0]); acts = np.load(it)["act"]
                tr, fc, g = roll_gate(None, seed, replay_actions=acts)
            else:
                seed = int(it); tr, fc, g = roll_gate(pi0, seed)
            st = stats(tr, fc); st["seed"] = seed
            st["hysteresis_honored"] = hysteresis_honored(tr)
            rows.append(st)
            print(f"  [{name}] seed {seed}: contact@{fc} activate@{st['first_activation']} "
                  f"phase={st['activation_phase']} false_early={st['false_early_activations']} "
                  f"reacq={st['reacquire_entries']} rearm={st['rearm_entries']} toggles={st['toggles']} "
                  f"hyst_ok={st['hysteresis_honored']} active={st['active_steps']}/{st['len']}", flush=True)
        out["classes"][name] = rows
        return rows

    s_ok = run_class("success_pi0_K6", sorted(INIT_SUCCESS))
    s_fail = run_class("fail_pi0", fail_seeds)
    s_cert = run_class("certified_delivery", certs, replay=True)

    allrows = s_ok + s_fail + s_cert
    # §4 required behaviours
    never_before_contact = all(r["false_early_activations"] == 0 for r in allrows)
    # arms only after arm_after consecutive contact: first_activation >= first_contact + (arm_after-1) when activated
    arms_after_stable = all((r["first_activation"] < 0) or (r["first_contact"] >= 0 and
                            r["first_activation"] >= r["first_contact"] + 2) for r in allrows)
    # active during transport/entry/settling on the successful K6 deliveries (contact sustained there)
    active_in_late = all(sum(r["active_by_phase"].get(p, 0) for p in ("TRANSPORT", "TARGET_ENTRY", "SETTLING", "DWELL")) > 0
                         for r in s_ok if r["activated"])
    # STRUCTURAL anti-chatter: every arm/disarm honored the full hysteresis window (no sub-window toggling). The
    # toggle COUNT is reported as a statistic (legitimate re-acquisitions on pi_0's bouncy transport grasp), NOT gated.
    hyst_ok = all(r["hysteresis_honored"] for r in allrows)
    max_toggles = max(r["toggles"] for r in allrows)
    total_reacq = sum(r["reacquire_entries"] for r in allrows)
    activated_any = any(r["activated"] for r in allrows)
    contract_pass = bool(never_before_contact and arms_after_stable and hyst_ok and activated_any and
                         active_in_late and sha[:8] == "1902454c")
    out["checks"] = {"never_active_before_contact": never_before_contact,
                     "arms_only_after_stable_contact": arms_after_stable,
                     "active_through_late_phases_on_success": active_in_late,
                     "hysteresis_honored_no_subwindow_chatter": hyst_ok,
                     "max_toggles_reported": max_toggles, "total_reacquisitions_reported": total_reacq,
                     "activated_at_all": activated_any,
                     "sha_matches_1902454c": sha[:8] == "1902454c"}
    verdict = "PHASE_GATE_RUNTIME_CONTRACT_PASS" if contract_pass else "PHASE_GATE_RUNTIME_CONTRACT_REVIEW"
    out["verdict"] = verdict
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nchecks:", json.dumps(out["checks"]), flush=True)
    print(verdict, flush=True); print("GATE_VAL_DONE", flush=True)


if __name__ == "__main__":
    main()
