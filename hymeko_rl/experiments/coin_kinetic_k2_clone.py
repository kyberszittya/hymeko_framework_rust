"""K2 — train the KINETIC feedback clone (K2-A) and run the teacher-free closed-loop s1 smoke (K2-B).

K2-A: one normal BC fit of the 32 K1-A feedback labels (+ a 3-seed audit for stability). K2-B: deploy the clone in the FROZEN
chain (frozen APPROACH → learned KINETIC clone → frozen G0/release → frozen coast → frozen K6) with NO teacher/CEM, and grade
by the CLOSED-LOOP behaviour, not action-R². Two DISTINCT gates:
  * LOCAL_KINETIC_FEEDBACK_SKILL_PASS — teacher-free, the clone keeps the coin moving (no clamp/stall), does not drift into a
    terminal-failure state, and gets measurably past the 48–51 mm hand plateau (ideally into the 20–30 mm close+moving
    corridor). A positive gate EVEN IF final K6 is absent.
  * FIRST_LEARNED_S1_K6_DELIVERY — the clone alone reaches strict K6 (great, but NOT a prerequisite for RL).

This does NOT start RL or DAgger — it reports and stops. Run:
``python -m hymeko_rl.experiments.coin_kinetic_k2_clone``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option import kinetic_clone as kcl
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

BANK = Path("reports/2026-07-28-coin-r9-k1a-bank/feedback_labels.json")
OUT = Path("reports/2026-07-28-coin-r9-k2-clone")
PLATEAU_MM = 48.0                                  # the measured hand-tuning plateau (48–51 mm); "past" ⇒ below this
CORRIDOR_MM = 30.0                                 # the close+moving release corridor upper edge (20–30 mm)
STALL_VPAR = 0.0                                   # a transport step with v_par ≤ this is a stall/terminal-failure drift


def _streaming_equals_batch(model: kcl.KineticClone, norm: kcl.NormStats, obs_seq: np.ndarray) -> float:
    """Max |batched − streamed| over a sequence — the deploy-time determinism/consistency control (should be ~0)."""
    x = torch.from_numpy(norm.apply(obs_seq).astype(np.float32))
    model.eval()
    with torch.no_grad():
        batched, _ = model(x.view(1, -1, kcl.OBS_DIM))
        batched = batched.view(-1, kcl.ACT_DIM).numpy()
    actor = kcl.CloneActor(model, norm)
    actor.reset()
    streamed = np.array([actor.act(o) for o in obs_seq], np.float64)
    return float(np.max(np.abs(batched - streamed)))


def _transport_profile(clone_trace: list[dict]) -> dict:
    """Summarise the KINETIC-transport segment: entry/exit steps, v_par & contact-force profile, stall/terminal-failure drift."""
    kin = [r for r in clone_trace if r["kind"] == "KINETIC_CLONE"]
    if not kin:
        return {"kinetic_steps": 0, "entry_step": None}
    vpar = [r["v_par"] for r in kin]
    fn_min = [min(r["fn_l"], r["fn_r"]) for r in kin]
    return {"kinetic_steps": len(kin), "entry_step": kin[0]["t"], "exit_step": kin[-1]["t"],
            "entry_dtz_mm": kin[0]["dtz_mm"], "exit_dtz_mm": kin[-1]["dtz_mm"],
            "v_par_min": round(min(vpar), 4), "v_par_max": round(max(vpar), 4), "v_par_final": round(vpar[-1], 4),
            "fn_min_transport": round(min(fn_min), 4), "fn_max_transport": round(max(fn_min), 4),
            "stall_steps": int(sum(1 for v in vpar if v <= STALL_VPAR)),
            "sign_reversals": int(sum(1 for i in range(1, len(vpar)) if vpar[i] * vpar[i - 1] < 0))}


def _deploy(snap: Any, actor: kcl.CloneActor) -> "tuple[dict, dict, list]":
    """Teacher-free closed-loop deploy of the clone in the frozen chain; returns (metrics, transport_profile, clone_trace)."""
    controller = kcl.KineticCloneController(snap, actor)
    m = velocity_rollout(snap, controller, kc.DELIVERY_CFG)
    return m, _transport_profile(controller.clone_trace), controller.clone_trace


def _grade(m: dict, prof: dict, min_dtz: float) -> dict:
    """The two split gates + the RL-branch signal, from the closed-loop behaviour."""
    moved = prof.get("kinetic_steps", 0) > 0
    kept_moving = bool(moved and prof.get("stall_steps", 1) == 0 and prof.get("v_par_min", -1.0) > STALL_VPAR - 1e-9)
    past_plateau = bool(min_dtz < PLATEAU_MM)
    reached_corridor = bool(min_dtz <= CORRIDOR_MM)
    safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
    local_pass = bool(kept_moving and past_plateau and safe)         # teacher-free by construction
    k6 = bool(m["k6_delivered"] and safe)
    if k6:
        gate = "FIRST_LEARNED_S1_K6_DELIVERY"
    elif local_pass:
        gate = "LOCAL_KINETIC_FEEDBACK_SKILL_PASS"
    elif kept_moving:
        gate = "TRANSPORT_MAINTAINED_SHORT_OF_PLATEAU"               # moving but not past 48 mm
    else:
        gate = "CLONE_STALLS"                                        # stall/clamp ⇒ DAgger territory
    # RL-branch signal per the pre-registered decision tree
    if not kept_moving:
        nxt = "SHORT_DAGGER"
    elif not k6:
        nxt = "TD3_VS_SAC"                                           # maintains transport / 20–50 mm ⇒ RL now
    else:
        nxt = "TD3_VS_SAC_ROBUSTIFY"
    return {"gate": gate, "next": nxt, "kept_moving": kept_moving, "past_plateau": past_plateau,
            "reached_corridor": reached_corridor, "k6_delivered": k6, "safe": safe, "min_dtz_mm": round(min_dtz, 2)}


def run() -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    bank = json.loads(BANK.read_text())
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")

    # K2-A: main fit (seed 0) + a 3-seed audit
    model, norm, hist = kcl.train_clone(bank, seed=0)
    obs_seq = np.array([r["obs"] for r in bank], np.float64)
    stream_gap = _streaming_equals_batch(model, norm, obs_seq)

    # K2-B: teacher-free closed-loop smoke (seed-0 clone)
    m, prof, trace = _deploy(snap, kcl.CloneActor(model, norm))
    min_dtz = _min_dtz_mm(snap, m)
    grade = _grade(m, prof, min_dtz)

    # seed audit — stability of the fit + closed-loop reach across seeds
    audit = []
    for s in (0, 1, 2):
        ms, ns, hs = kcl.train_clone(bank, seed=s)
        md, pd, _tr = _deploy(snap, kcl.CloneActor(ms, ns))
        mdtz = _min_dtz_mm(snap, md)
        audit.append({"seed": s, "final_loss": round(hs[-1], 6), "min_dtz_mm": round(mdtz, 2),
                      "kinetic_steps": pd.get("kinetic_steps", 0), "v_par_min": pd.get("v_par_min"),
                      "k6": bool(md["k6_delivered"])})

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "norm": norm.as_dict(), "hidden": 64}, OUT / "clone_seed0.pt")
    out = {"contract": "COIN_KINETIC_K2_CLONE_V1", "seed": kc.S1_SEED, "n_labels": len(bank),
           "bc_fit": {"final_loss": round(hist[-1], 6), "loss_at_0": round(hist[0], 6),
                      "loss_curve_tail": [round(x, 6) for x in hist[-5:]]},
           "streaming_equals_batch_max_gap": stream_gap,
           "closed_loop": {"k6_delivered": bool(m["k6_delivered"]), "min_dtz_mm": round(min_dtz, 2),
                           "dtz_end_mm": round(float(m["dtz_end"]) * 1000, 2), "peak_qdot": round(m["peak_qdot"], 3),
                           "peak_coin_speed": round(m["peak_coin_speed"], 3), "transport": prof},
           "grade": grade, "seed_audit": audit, "event_trace": trace, "wall_s": round(time.time() - t0, 1)}
    (OUT / "k2_clone.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    g, cl, pr = r["grade"], r["closed_loop"], r["closed_loop"]["transport"]
    print(f"\nGATE: {g['gate']}  →  next: {g['next']}   (wall {r['wall_s']}s)\n")
    print(f"  BC fit: loss {r['bc_fit']['loss_at_0']} → {r['bc_fit']['final_loss']}  |  stream==batch gap {r['streaming_equals_batch_max_gap']:.2e}")
    print(f"  closed-loop: K6={cl['k6_delivered']}  min_dtz={cl['min_dtz_mm']}mm  peak_qdot={cl['peak_qdot']}  peak_v={cl['peak_coin_speed']}")
    print(f"  transport: {pr.get('kinetic_steps')} steps  dtz {pr.get('entry_dtz_mm')}→{pr.get('exit_dtz_mm')}mm  "
          f"v_par[{pr.get('v_par_min')},{pr.get('v_par_max')}]  fn[{pr.get('fn_min_transport')},{pr.get('fn_max_transport')}]  "
          f"stalls={pr.get('stall_steps')} reversals={pr.get('sign_reversals')}")
    print(f"  gate flags: kept_moving={g['kept_moving']} past_plateau({PLATEAU_MM})={g['past_plateau']} corridor={g['reached_corridor']} k6={g['k6_delivered']}")
    print("  seed audit:")
    for a in r["seed_audit"]:
        print(f"    seed {a['seed']}: loss {a['final_loss']}  min_dtz {a['min_dtz_mm']}mm  kin_steps {a['kinetic_steps']}  v_par_min {a['v_par_min']}  K6={a['k6']}")
