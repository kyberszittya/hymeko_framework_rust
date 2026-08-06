"""§7 unilateral stress panel: does the `left_contact OR right_contact` predicate ARM the residual on THREE
consecutive UNILATERAL (single-finger) contact steps during genuine CONTACT_ACQUISITION (before a bilateral grasp
ever forms)? Records left/right SEPARATELY on real pi_0 + certified acquisition dynamics, and probes the named
synthetic contact cases. Reports the material rate of premature-unilateral arming.
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_phase_gate import PhaseGate, PhaseGateConfig  # noqa: E402
from hymeko_rl.coin_delivery.coin_v3_seed_banks import HEADLINE  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import ActorEvalWrap, load_frozen_clip_actor  # noqa: E402
from hymeko_rl.experiments.coin_neutral_start import neutral_env  # noqa: E402

PI0 = sys.argv[1]; BASE = sys.argv[2]; OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/unilat.json"


def roll_lr(actor, seed, *, replay=None, horizon=360):
    """Roll; record per-step (left, right) contact separately + the gate arm events with their arming-window makeup."""
    env, cf = neutral_env(prefix_steps=0); inner = cf._env
    env.set_stage(0); env.reset(seed=int(seed))
    gate = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    ew = ActorEvalWrap(actor) if actor is not None else None
    L, R, states, dtzs = [], [], [], []
    bilateral_ever = False; arm_events = []
    n = horizon if replay is None else min(horizon, len(replay))
    for t in range(n):
        nf = np.asarray(inner.node_features(), np.float32).flatten()
        a = replay[t] if replay is not None else ew.act(nf)
        inner.step(np.clip(np.asarray(a, np.float32), -4, 4))
        m = inner._planar_metrics
        lc, rc = bool(m.left_contact), bool(m.right_contact)
        L.append(lc); R.append(rc)
        bilateral_ever = bilateral_ever or (lc and rc)
        dtz = float(m.disk_to_zone); dtzs.append(dtz)
        prev_state = gate.state.value
        speed = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
        strict_ok = dtz <= 0.02 and speed < 0.06
        gate.update(lc or rc, terminated=(strict_ok and t > 6 and (lc or rc)))
        states.append(gate.state.value)
        if gate.state.value == "LATE_CONTROL_ARMED" and prev_state != "LATE_CONTROL_ARMED":
            # arming window = the arm_after steps ending at t
            w = slice(max(0, t - 2), t + 1)
            win_L, win_R = L[w], R[w]
            unilateral_window = all(not (a and b) for a, b in zip(win_L, win_R))     # never bilateral in the window
            only_one_side = (all(win_L) and not any(win_R)) or (all(win_R) and not any(win_L))
            arm_events.append({"t": t, "bilateral_ever_before": bilateral_ever and not (lc and rc),
                               "unilateral_window": unilateral_window, "single_side_window": only_one_side,
                               "bilateral_at_arm": lc and rc,
                               # premature = armed on a unilateral-only window with NO prior bilateral grasp
                               "premature_unilateral_acquisition": unilateral_window and not _bilateral_before(L, R, t)})
        if replay is None and dtz <= 0.02 and t > 6:
            pass
    return {"L": L, "R": R, "dtz": dtzs, "states": states, "arm_events": arm_events, "bilateral_ever": bilateral_ever}


def _bilateral_before(L, R, t):
    return any(L[i] and R[i] for i in range(0, max(0, t - 2)))


def candidate_gate(L, R, dtz, *, arm_pred, arm_after=3, disarm_after=2, entry_tol=0.05):
    """Offline replay of a CANDIDATE arm predicate over recorded (L,R). arm_pred in {'or','and'}. disarm = complete
    loss (not (L or R)). Returns arm events with acquisition-vs-transport labelling. Does NOT touch the deployed gate.
    """
    mode = "EARLY"; cstreak = 0; lstreak = 0; bilat_ever = False
    arms = []; active = 0; active_transport = 0
    for t in range(len(L)):
        lc, rc = L[t], R[t]
        bilat_ever = bilat_ever or (lc and rc)
        armp = (lc and rc) if arm_pred == "and" else (lc or rc)
        cstreak = cstreak + 1 if armp else 0
        lstreak = lstreak + 1 if not (lc or rc) else 0
        prev = mode
        if mode in ("EARLY", "REACQUIRE") and cstreak >= arm_after:
            mode = "ARMED"; lstreak = 0
        elif mode == "ARMED" and lstreak >= disarm_after:
            mode = "REACQUIRE"; cstreak = 0
        if mode == "ARMED":
            active += 1
            if dtz[t] <= entry_tol or _bilateral_before(L, R, t):
                active_transport += 1
        if mode == "ARMED" and prev != "ARMED":
            # premature = armed before any bilateral grasp formed (acquisition brush)
            arms.append({"t": t, "premature_acq": not _bilateral_before(L, R, t) and not (lc and rc)})
    return {"n_arm": len(arms), "premature_acq": sum(a["premature_acq"] for a in arms),
            "active": active, "active_transport": active_transport}


def synthetic(name, contacts_lr):
    """Feed a named (left,right) sequence; return whether/when it arms and if the arming window was unilateral."""
    g = PhaseGate(PhaseGateConfig(arm_after=3, disarm_after=2))
    Ls = [c[0] for c in contacts_lr]; Rs = [c[1] for c in contacts_lr]
    armed_at = -1; uni = None
    for t, (lc, rc) in enumerate(contacts_lr):
        prev = g.state.value
        g.update(lc or rc)
        if g.state.value == "LATE_CONTROL_ARMED" and prev != "LATE_CONTROL_ARMED":
            armed_at = t
            w = slice(max(0, t - 2), t + 1)
            uni = all(not (a and b) for a, b in zip(Ls[w], Rs[w]))
            break
    return {"case": name, "armed_at": armed_at, "unilateral_arm_window": uni}


def main():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    out = {"real": {}, "synthetic": [], "predicate": "left_contact OR right_contact"}
    # named synthetic cases (§7)
    U, N, B = (True, False), (False, False), (True, True)   # unilateral-left, none, bilateral
    UR = (False, True)
    cases = [
        ("unilateral_transient_2", [N, U, U, N, N]),
        ("unilateral_3steps", [N, U, U, U, N]),
        ("alternating_LR", [U, UR, U, UR, U, UR]),
        ("bilateral_3steps", [N, B, B, B, N]),
        ("contact_bounce", [B, B, N, B, B, N, B]),
        ("acquire_unilateral_then_bilateral", [N, U, U, U, B, B, B]),
    ]
    for nm, seq in cases:
        r = synthetic(nm, seq); out["synthetic"].append(r)
        print(f"  [syn] {nm:<38} armed_at={r['armed_at']:>2} unilateral_window={r['unilateral_arm_window']}", flush=True)

    # real acquisition dynamics: pi_0 on all headline + a few certified. Compare deployed OR-gate vs BILATERAL candidate.
    prem = 0; total_arms = 0; uni_windows = 0
    cand = {"or": {"prem_traj": 0, "arms": 0, "active_transport": 0}, "and": {"prem_traj": 0, "arms": 0, "active_transport": 0}}
    trajs = [("pi0", s, None) for s in sorted(HEADLINE)] + \
            [("cert", int(f.split("traj_")[1].split(".npz")[0]), f)
             for f in sorted(glob.glob(BASE + "/traj_*.npz"))[:5]]
    for tag, s, f in trajs:
        r = roll_lr(pi0 if f is None else None, s, replay=(np.load(f)["act"] if f else None))
        ev = r["arm_events"]
        p = sum(e["premature_unilateral_acquisition"] for e in ev)
        u = sum(e["unilateral_window"] for e in ev)
        prem += (1 if p > 0 else 0); total_arms += len(ev); uni_windows += u
        cor = candidate_gate(r["L"], r["R"], r["dtz"], arm_pred="or")
        cand_and = candidate_gate(r["L"], r["R"], r["dtz"], arm_pred="and")
        for key, cc in (("or", cor), ("and", cand_and)):
            cand[key]["prem_traj"] += (1 if cc["premature_acq"] > 0 else 0)
            cand[key]["arms"] += cc["n_arm"]; cand[key]["active_transport"] += cc["active_transport"]
        out["real"][f"{tag}_{s}"] = {"or_arms": cor["n_arm"], "or_premature": cor["premature_acq"],
                                     "and_arms": cand_and["n_arm"], "and_premature": cand_and["premature_acq"],
                                     "and_active_transport": cand_and["active_transport"]}
        print(f"  [real] {tag} {s}: OR arms={cor['n_arm']} prem={cor['premature_acq']} | "
              f"BILATERAL arms={cand_and['n_arm']} prem={cand_and['premature_acq']} "
              f"active_transport={cand_and['active_transport']}", flush=True)

    n_traj = len(trajs)
    out["refinement_bilateral_and"] = cand
    material_rate = prem / n_traj
    # "materially" = premature unilateral acquisition arming on a non-trivial fraction of trajectories
    premature = material_rate >= 0.15
    out["summary"] = {"n_traj": n_traj, "traj_with_premature_unilateral_acq": prem,
                      "material_rate": round(material_rate, 3), "total_arm_events": total_arms,
                      "total_unilateral_windows": uni_windows,
                      "unilateral_3steps_arms_synthetic": out["synthetic"][1]["armed_at"] >= 0,
                      "material_premature": premature}
    verdict = "PHASE_GATE_PREMATURE_UNILATERAL_ACTIVATION" if premature else "UNILATERAL_STRESS_CLEAN"
    out["verdict"] = verdict
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsummary: {json.dumps(out['summary'])}", flush=True)
    print(verdict, flush=True); print("UNILAT_DONE", flush=True)


if __name__ == "__main__":
    main()
