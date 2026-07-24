"""BALLTIP_COLLISION_ON_V1 — Stage B1 capability decomposition (measurement only, NO training).

Evaluates the PHYSICALLY HONEST ball-tip robot (spherical fingertip r0.020, inter-arm collision ENABLED, no filtering,
no penalty) on a matched held-out panel with FOUR controllers, decomposing where capability is lost — so Stage B2 can
name the adaptation boundary rather than reading the clamp-controller's transfer number as the ball's ceiling.

Controllers (collision-on physics throughout):
  1. clamp_proposal_b8   — the FROZEN clamp proposal θ_center + b=8 Gaussian search (the deployed, clamp-developed stack)
  2. random_shooting_64  — full structured random shooting, uniform over the action language (no clamp bias)
  3. expert_192          — strong structured search expert at the validated larger budget (the achievable ceiling probe)
  4. geometry_probed     — an explicit geometry-aware push: finite-difference the disk's contact response to each joint,
                           push toward the zone (the joint→disk map is contact-mediated ⇒ probed, not analytic)

Per controller, reported separately: candidate support coverage (frac of shots reaching handoff), contact-retention
(touched), handoff rate, settling|handoff (K6 among handoff-reachers = the FROZEN clamp pi_0's settling skill on the
ball), final K6, containment exit, inter-arm contact rate (real physics contacts, collision-on), option duration, and
solved-set overlap with the clamp robot. The clamp robot is preserved as the frozen canonical reference.
"""
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_carry_proposal import load_proposal, search_select  # noqa: E402
from hymeko_rl.coin_delivery.coin_carry_structured import (  # noqa: E402
    A_BOUND, T_MAX, T_MIN, structured_carry_rollout, structured_random_best_with_support)
from hymeko_rl.coin_delivery.coin_late_start import (  # noqa: E402
    LateStart, build_boundary_panel, reconstruct_handoff)
from hymeko_rl.coin_delivery.coin_markov_ablation_train import make_late_actor55_from_pi0  # noqa: E402
from hymeko_rl.coin_delivery.coin_robot_variant import (  # noqa: E402
    build_variant_rl, interarm_contact_count, min_interarm_clearance, transplant_handoff)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"
OUT = "reports/2026-07-24-balltip-b1-capability"
FAMS = ("contact_retention", "transport", "braking")
PROP_CKPT = f"{D}/carry_proposal_refined.pt"
SEARCH_SEED = 9000
HORIZON = 160
BASELINE = {"tag": "executable-hymeko-option-rl-v1", "commit": "772a11a4"}
BALL = "balltip_nofilter"                                    # = BALLTIP_COLLISION_ON_V1 (ball r0.020, collision ON)


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _committed_metrics(rl, gate, pi0, base, theta):
    """Re-run the committed option, tracking inter-arm CONTACT steps (real physics) and min clearance (diagnostic).
    Deep-copies the gate — structured_carry_rollout mutates it, and a shared gate would contaminate later controllers."""
    gate = copy.deepcopy(gate)
    m, d = rl.inner.model, rl.inner.data
    acc = {"contact_steps": 0, "n": 0, "min_clr": float("inf")}

    def hook(_p, _s):
        acc["contact_steps"] += int(interarm_contact_count(m, d) > 0)
        acc["min_clr"] = min(acc["min_clr"], min_interarm_clearance(m, d))
        acc["n"] += 1

    o = structured_carry_rollout(rl, gate, pi0, base, np.asarray(theta, np.float32), horizon=HORIZON, frame_hook=hook)
    o["contact_rate"] = round(acc["contact_steps"] / max(1, acc["n"]), 4)
    o["min_clr"] = round(float(acc["min_clr"]), 5)
    return o


def geometry_probed_theta(rl, mag=2.0, k_probe=6, push_mag=2.5, t_push=11, t_brake=6, t_rel=3):
    """Explicit geometry-aware push (controller 4). Finite-difference d(disk_xy)/d(joint torque) with a short contact
    probe, then least-squares the push that drives the disk along the coin→zone direction. Brake = −0.4·push; no release."""
    dir_vec, _dist = rl.inner.direction_to_zone()
    disk0 = np.asarray(rl.inner._planar_metrics.disk_pos, np.float32)[:2].copy()
    J = np.zeros((2, 4), np.float32)
    for j in range(4):
        r2 = copy.deepcopy(rl)
        a = np.zeros(4, np.float32)
        a[j] = mag
        for _ in range(k_probe):
            r2.step(a)
        J[:, j] = (np.asarray(r2.inner._planar_metrics.disk_pos, np.float32)[:2] - disk0) / (mag * k_probe)
    a_push = np.linalg.lstsq(J, np.asarray(dir_vec, np.float32), rcond=None)[0]
    nrm = float(np.linalg.norm(a_push)) + 1e-9
    a_push = np.clip(a_push / nrm * push_mag, -A_BOUND, A_BOUND).astype(np.float32)
    dur = np.clip([t_push, t_brake, t_rel], T_MIN, T_MAX).astype(np.float32)
    return np.concatenate([a_push, dur[:1], (-0.4 * a_push), dur[1:2], np.zeros(4, np.float32), dur[2:3]]).astype(np.float32)


def eval_controllers(ball, gate_c, c_center, i, pi0, base, rebuild):
    """Run all four controllers on the ball robot at a matched start. ``rebuild()`` returns a fresh matched ball env
    (each rollout mutates its env)."""
    rng = lambda: np.random.default_rng(SEARCH_SEED + i)  # noqa: E731  (same seed across controllers for a matched search)
    out = {}
    # 1. clamp proposal + b=8 (Gaussian around the frozen clamp proposal center) — with support
    _t1, _o1, sup1 = structured_random_best_with_support(copy.deepcopy(ball), copy.deepcopy(gate_c), pi0, base, rng(),
                                                         shots=8, horizon=HORIZON, center=c_center)
    th1, o1 = search_select(copy.deepcopy(ball), copy.deepcopy(gate_c), c_center, pi0, base, rng(), b=8, horizon=HORIZON)
    out["clamp_proposal_b8"] = (_committed_metrics(rebuild(), gate_c, pi0, base, th1), sup1 / 8)
    # 2. full structured random shooting (uniform, 64) — with support
    th2, _o2, sup2 = structured_random_best_with_support(copy.deepcopy(ball), copy.deepcopy(gate_c), pi0, base, rng(),
                                                        shots=64, horizon=HORIZON)
    out["random_shooting_64"] = (_committed_metrics(rebuild(), gate_c, pi0, base, th2), sup2 / 64)
    # 3. strong expert at the validated larger budget (uniform, 192) — the ceiling probe, with support
    th3, _o3, sup3 = structured_random_best_with_support(copy.deepcopy(ball), copy.deepcopy(gate_c), pi0, base, rng(),
                                                        shots=192, horizon=HORIZON)
    out["expert_192"] = (_committed_metrics(rebuild(), gate_c, pi0, base, th3), sup3 / 192)
    # 4. explicit geometry-aware push (single controller — support N/A)
    th4 = geometry_probed_theta(rebuild())
    out["geometry_probed"] = (_committed_metrics(rebuild(), gate_c, pi0, base, th4), float("nan"))
    return out


def summarize(recs, clamp_solved, n):
    """Per-controller aggregate decomposition."""
    out = {}
    for ctrl, rs in recs.items():
        handoff = [r for r in rs if r["reached_handoff"]]
        solved = sorted(r["i"] for r in rs if r["k6"])
        sup = [r["support"] for r in rs if not np.isnan(r["support"])]
        out[ctrl] = {
            "n": n,
            "candidate_support": round(float(np.mean(sup)), 4) if sup else None,
            "contact_retention_rate": round(sum(r["touched"] for r in rs) / n, 4),
            "handoff_rate": round(len(handoff) / n, 4),
            "settling_given_handoff": round(sum(r["k6"] for r in handoff) / len(handoff), 4) if handoff else None,
            "k6": sum(r["k6"] for r in rs), "k6_rate": round(sum(r["k6"] for r in rs) / n, 4),
            "mean_contain_exit": round(float(np.mean([r["contain_exit_ct"] for r in rs])), 3),
            "mean_interarm_contact_rate": round(float(np.mean([r["contact_rate"] for r in rs])), 4),
            "mean_option_duration": round(float(np.mean([r["completion"] for r in rs])), 2),
            "mean_min_clearance": round(float(np.mean([r["min_clr"] for r in rs])), 5),
            "solved_states": solved,
            "solved_overlap_with_clamp": sorted(set(solved) & set(clamp_solved)),
        }
    return out


def main(smoke=False):
    import torch
    torch.set_num_threads(1)
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json"))
    pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    base = make_late_actor55_from_pi0(pi0, trainable=False)
    prop = load_proposal(PROP_CKPT)
    forbidden = {b.seed for b in _bank(cfg["banks"]["late_train"])} | {b.seed for b in _bank(cfg["banks"]["late_dev"])}
    raw, _c, _s = build_boundary_panel(pi0, range(14000, 15200), forbidden, want=4 if smoke else 24, families=FAMS,
                                       strict_primary=(0,), strict_fill=(), per_seed_cap=3)
    panel = [(*reconstruct_handoff(pi0, ls, horizon=360)[:2], ls) for ls in raw]
    print(f"[balltip B1] {len(panel)} matched states | robot {BALL} (ball r0.020, collision ON) | search seed {SEARCH_SEED}", flush=True)

    ctrls = ["clamp_proposal_b8", "random_shooting_64", "expert_192", "geometry_probed"]
    recs = {c: [] for c in ctrls}
    clamp_solved = []
    for i, (rl_c, gate_c, ls) in enumerate(panel):
        c_center = prop.theta(rl_c.obs())
        # clamp reference (frozen clamp robot + clamp proposal b=8) — for solved-set overlap
        _thc, oc = search_select(copy.deepcopy(rl_c), copy.deepcopy(gate_c), c_center, pi0, base,
                                 np.random.default_rng(SEARCH_SEED + i), b=8, horizon=HORIZON)
        if oc["k6"]:
            clamp_solved.append(i)

        def rebuild():
            return transplant_handoff(build_variant_rl(BALL, seed=int(ls.seed)), rl_c)  # noqa: B023

        for ctrl, (o, sup) in eval_controllers(rebuild(), gate_c, c_center, i, pi0, base, rebuild).items():
            recs[ctrl].append({"i": i, "seed": int(ls.seed), "support": sup, **{k: o[k] for k in
                               ("k6", "reached_handoff", "touched", "contain_exit_ct", "completion", "contact_rate", "min_clr")}})
        done = i + 1
        if done % 4 == 0 or i == 0:
            print(f"  [{done}/{len(panel)}] ball K6 " + " ".join(
                f"{c.split('_')[0][:4]}:{sum(r['k6'] for r in recs[c])}" for c in ctrls)
                + f" | clamp_ref {len(clamp_solved)}", flush=True)

    summary = summarize(recs, clamp_solved, len(panel))
    manifest = {"contract": "BALLTIP_COLLISION_ON_V1", "stage": "B1", "date": "2026-07-24", "smoke": smoke,
                "baseline": BASELINE, "robot": "ball r0.020 collision-ON (no filter, no penalty)", "n_states": len(panel),
                "clamp_solved_states": clamp_solved, "summary": summary, "records": recs}
    json.dump(manifest, open(f"{OUT}/b1_capability.json", "w"), indent=1, default=float)

    print("\n== BALLTIP_COLLISION_ON_V1 — Stage B1 capability decomposition ==", flush=True)
    hdr = f"{'controller':20} {'support':>8} {'contact':>8} {'handoff':>8} {'K6|hand':>8} {'K6':>7} {'exit':>6} {'ia_ctc':>7} {'dur':>6}"
    print(hdr)
    print("-" * len(hdr))
    for c in ctrls:
        s = summary[c]
        sup = "n/a" if s["candidate_support"] is None else f"{s['candidate_support']:.3f}"
        sgh = "n/a" if s["settling_given_handoff"] is None else f"{s['settling_given_handoff']:.3f}"
        print(f"{c:20} {sup:>8} {s['contact_retention_rate']:>8.3f} {s['handoff_rate']:>8.3f} {sgh:>8} "
              f"{s['k6']:>3}/{s['n']:<3} {s['mean_contain_exit']:>6.2f} {s['mean_interarm_contact_rate']:>7.3f} {s['mean_option_duration']:>6.1f}", flush=True)
    print(f"\n  clamp reference solved {len(clamp_solved)}/{len(panel)}: {clamp_solved}", flush=True)
    for c in ctrls:
        print(f"  {c:20} solved {summary[c]['solved_states']} | ∩clamp {summary[c]['solved_overlap_with_clamp']}", flush=True)
    print(f"\n  artifact: {OUT}/b1_capability.json\nBALLTIP_B1_DONE", flush=True)
    return manifest


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
