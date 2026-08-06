"""Deterministic receding-horizon feedback expert for the Coin delivery pilot (§3-§5).

The strengthened teacher's open-loop CEM suffix is a feedforward, time-indexed *sequence*: its per-step action is not
a function of the observation, so it is un-clonable (measured `FULL_ACTION_BC_COMPETENCE_BLOCKED`, 2026-07-22). This
expert instead produces a STATE-FEEDBACK law by MPC-style replanning: at every queried state it forward-searches a
bounded action sequence over the simulator, executes/labels only the FIRST action, advances the real environment one
step, and replans from the newly reached state. The label is therefore a function of the state that produced it.

Action space = the deployed actor's 4-dim `inner.step` arm command (== the `ACTION_CONTRACT` label), so the expert's
first action, the supervised label, and the deployed action all live in the same space (no env.step/6-dim aliasing).
Scoring reuses the strict K=6 `DeliveryCertifier` over `_cert_step` — the exact certificate the deployed policy is
graded by (proven consistent with `eval_bc_delivery`). Forward search uses the simulator state internally; the stored
learner KEY is the exact `FULL_ACTION_OBS_HISTORY_V1` input.
"""
from __future__ import annotations

import numpy as np

CENTER_TOL = 0.02
SETTLE_VEL = 0.06
CTRL_LIM = 4.0
ACT_DIM = 4


def _snapshot(inner):
    return inner.data.qpos.copy(), inner.data.qvel.copy()


def _restore(inner, qpos, qvel):
    import mujoco
    inner.data.qpos[:] = qpos
    inner.data.qvel[:] = qvel
    mujoco.mj_forward(inner.model, inner.data)


def _score_horizon(inner, cf, qpos, qvel, actions_h, clearance, carry_touched, carry_dwell):
    """Roll ``actions_h`` (H, 4) open-loop from the snapshot via ``inner.step``; score the bounded horizon by the strict
    K=6 certificate carried from the real state (touched + in-progress dwell). Returns the lexicographic components."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step
    _restore(inner, qpos, qvel)
    cert = DeliveryCertifier(initial_clearance=clearance)
    cert.robot_touched = bool(carry_touched)
    cert.delivery_dwell = int(carry_dwell)
    best_dwell = int(carry_dwell)
    min_dtz = 9.0
    min_speed = 9.0
    strict = False
    entry_step = -1                                          # first in-plan step the coin enters the zone (dtz<=0.04)
    for t in range(len(actions_h)):
        cert.update(_cert_step(inner, cf))
        best_dwell = max(best_dwell, cert.delivery_dwell)
        m = inner._planar_metrics
        dtz = float(m.disk_to_zone)
        min_dtz = min(min_dtz, dtz)
        if entry_step < 0 and dtz <= 0.04:
            entry_step = t
        sp = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
        min_speed = min(min_speed, sp)
        if cert.delivery_certified:
            strict = True
            break
        inner.step(np.asarray(actions_h[t], np.float32))
        if not np.all(np.isfinite(inner.data.qvel)):
            return {"strict": False, "dwell": best_dwell, "min_dtz": min_dtz, "min_speed": 9.0, "effort": 9.0,
                    "entry_step": entry_step, "horizon": len(actions_h)}
    return {"strict": strict, "dwell": best_dwell, "min_dtz": min_dtz, "min_speed": min_speed,
            "effort": float(np.mean(np.abs(actions_h))), "entry_step": entry_step, "horizon": len(actions_h)}


def _lexo(r):
    """strict → max dwell → min dtz → min speed → min effort (higher is better)."""
    return (int(r["strict"]), r["dwell"], -r["min_dtz"], -r.get("min_speed", 9.0), -r.get("effort", 9.0))


def plan_first_action(inner, cf, clearance, touched, dwell, seed, *, horizon=15, pop=32, iters=6, elite=8,
                      sigma0=0.6, warm=None):
    """CEM over the H×4 arm-action sequence from the CURRENT ``inner`` state; return (first_action[4], best_result).

    Leaves the real ``inner`` state UNCHANGED (snapshots + restores around the search). ``warm`` (H,4) is an optional
    proposal warm start (e.g. a retained open-loop suffix segment). Determinism: fixed ``seed`` + config ⇒ fixed CEM."""
    qpos, qvel = _snapshot(inner)
    dim = horizon * ACT_DIM
    mean = (np.asarray(warm, np.float32).flatten() if warm is not None else np.zeros(dim, np.float32))
    if mean.size != dim:
        mean = np.zeros(dim, np.float32)
    sigma = np.full(dim, sigma0, np.float32)
    rng = np.random.default_rng(seed)
    best = None
    best_seq = None
    any_strict = False                                       # did ANY sampled candidate reach strict K=6 in-plan?
    for _it in range(iters):
        cand = np.clip(rng.normal(mean, sigma, size=(pop, dim)), -CTRL_LIM, CTRL_LIM).astype(np.float32)
        cand[0] = mean                                       # keep the incumbent
        scored = []
        for c in cand:
            r = _score_horizon(inner, cf, qpos, qvel, c.reshape(horizon, ACT_DIM), clearance, touched, dwell)
            any_strict = any_strict or bool(r["strict"])
            scored.append((_lexo(r), c, r))
        scored.sort(key=lambda x: x[0])
        el = np.stack([c for _k, c, _r in scored[-elite:]])
        mean, sigma = el.mean(0), el.std(0) + 0.05
        if best is None or scored[-1][0] > _lexo(best):
            best, best_seq = scored[-1][2], scored[-1][1].copy()
    _restore(inner, qpos, qvel)
    best["any_candidate_strict"] = bool(any_strict)          # §3 horizon-mechanism probe
    return best_seq.reshape(horizon, ACT_DIM)[0].astype(np.float32), best


# ---- §5 closed-loop receding-horizon rollout from true canonical neutral ----------------------------------------

def receding_horizon_rollout(seed, *, horizon=15, pop=48, iters=8, elite=8, plan_seed_base=0, max_steps=360,
                             approach_cap=160, grasp_hold=3):
    """True-neutral → frozen E-approach to the bilateral-grasp handoff (the declared teacher approach; the strict-K=6
    scorer has no approach gradient, memory ``spec-reward``) → then the §3 receding-horizon CEM expert REPLANS every
    step (4-dim ``inner.step``, unchanged strict-K=6 certificate), executing only the first action and discarding the
    suffix. Returns the full executed action sequence + per-step diagnostics + outcome. No open-loop suffix labels."""
    import time

    import torch

    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.coin_delivery.full_action_obs_history import ObsHistoryV1
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, _e_approach_actor, neutral_env
    e = _e_approach_actor()
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    env.set_stage(0)
    env.reset(seed=int(seed))
    clearance = _clearance(inner)
    cert = DeliveryCertifier(initial_clearance=clearance)
    hist = ObsHistoryV1()
    hist.reset(np.asarray(inner.node_features(), np.float32).flatten())
    executed: list = []
    feats: list = []
    dwell_seq: list = []
    cem_scores: list = []
    latencies: list = []
    n_cand: list = []
    start_dtz = float(inner._planar_metrics.disk_to_zone)
    bi = 0
    grasped = False
    handoff_step = None
    first_contact = bilateral = entered = centered = False
    contact_loss = exit_after_entry = False
    was_bilateral = was_entered = False
    max_dwell = 0
    reason = "timeout"
    # §3 within-plan mechanism probes + §2 per-seed diagnostics
    plan_any_strict = plan_sel_strict = n_plans = 0
    max_plan_dwell = 0
    min_plan_dtz = 9.0
    entry_remaining: list = []
    max_coin_speed_near = 0.0
    min_dtz_overall = 9.0
    for step in range(max_steps):
        cert.update(_cert_step(inner, cf))
        m = inner._planar_metrics
        dtz = float(m.disk_to_zone)
        lc, rc = bool(m.left_contact), bool(m.right_contact)
        first_contact = first_contact or lc or rc
        now_bi = lc and rc
        bilateral = bilateral or now_bi
        if was_bilateral and not now_bi and grasped:
            contact_loss = True
        was_bilateral = now_bi
        entered = entered or dtz <= 0.04
        centered = centered or dtz <= CENTER_TOL
        if was_entered and dtz > 0.04:
            exit_after_entry = True
        was_entered = dtz <= 0.04
        max_dwell = max(max_dwell, int(cert.delivery_dwell))
        dwell_seq.append(int(cert.delivery_dwell))
        min_dtz_overall = min(min_dtz_overall, dtz)
        if dtz < 0.06:                                        # coin speed near the target (overshoot indicator)
            csp = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
            max_coin_speed_near = max(max_coin_speed_near, csp)
        if cert.delivery_certified:
            reason = "strict_success"
            break
        if not grasped and (bi < grasp_hold) and step < approach_cap:
            a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
            a = np.asarray(a, np.float32)
            bi = bi + 1 if now_bi else 0
            if bi >= grasp_hold:
                grasped = True
                handoff_step = step
            cem_scores.append(None)
            latencies.append(0.0)
            n_cand.append(0)
        else:
            grasped = True
            if handoff_step is None:
                handoff_step = step
            t0 = time.time()
            a, res = plan_first_action(inner, cf, clearance, cert.robot_touched, cert.delivery_dwell,
                                       plan_seed_base + step, horizon=horizon, pop=pop, iters=iters, elite=elite)
            latencies.append(round(time.time() - t0, 4))
            n_cand.append(pop * iters)
            cem_scores.append(round(-res["min_dtz"] + res["dwell"], 4))
            n_plans += 1
            plan_any_strict += int(res.get("any_candidate_strict", False))
            plan_sel_strict += int(res["strict"])
            max_plan_dwell = max(max_plan_dwell, int(res["dwell"]))
            min_plan_dtz = min(min_plan_dtz, float(res["min_dtz"]))
            if res.get("entry_step", -1) >= 0:
                entry_remaining.append(int(res["horizon"] - res["entry_step"]))
        feats.append(hist.feature().copy())
        inner.step(np.asarray(a, np.float32))
        executed.append(np.asarray(a, np.float32).copy())
        hist.push(np.asarray(inner.node_features(), np.float32).flatten(), a)
    strict = reason == "strict_success"
    fail_class = _classify_failure(strict, first_contact, bilateral, grasped, entered, centered, max_dwell,
                                   contact_loss, exit_after_entry, start_dtz,
                                   float(inner._planar_metrics.disk_to_zone))
    return {"seed": int(seed), "strict": bool(strict), "reason": reason, "steps": len(executed),
            "handoff_step": handoff_step, "first_contact": bool(first_contact), "bilateral": bool(bilateral),
            "entered": bool(entered), "centered": bool(centered), "max_dwell": int(max_dwell),
            "contact_loss_after_acq": bool(contact_loss), "target_exit_after_entry": bool(exit_after_entry),
            "failure_class": fail_class,
            "mean_action_mag": round(float(np.mean([np.linalg.norm(a) for a in executed])) if executed else 0.0, 4),
            "mean_plan_latency": round(float(np.mean([x for x in latencies if x > 0])) if any(latencies) else 0.0, 4),
            "total_candidate_evals": int(sum(n_cand)), "planner_calls": int(n_plans),
            "final_approach_dtz": round(float(min_dtz_overall), 4),
            "max_coin_speed_near_target": round(float(max_coin_speed_near), 4),
            # §3 within-plan horizon-mechanism probes
            "plan_any_strict_frac": round(plan_any_strict / n_plans, 4) if n_plans else 0.0,
            "plan_sel_strict_frac": round(plan_sel_strict / n_plans, 4) if n_plans else 0.0,
            "max_plan_dwell": int(max_plan_dwell), "min_plan_dtz": round(float(min_plan_dtz), 4),
            "entry_remaining_med": int(np.median(entry_remaining)) if entry_remaining else -1,
            "executed": np.asarray(executed, np.float32), "dwell_seq": dwell_seq}


def _classify_failure(strict, fc, bil, grasped, entered, centered, max_dwell, contact_loss, exit_after, start_dtz, end_dtz):
    if strict:
        return None
    if not fc:
        return "approach_failure"
    if not bil:
        return "no_bilateral_grasp"
    if contact_loss and not centered:
        return "contact_loss_recovery_failure"
    if not entered:
        return "transport_failure" if end_dtz < start_dtz - 0.02 else "transport_failure"
    if not centered:
        return "target_entry_failure"
    if exit_after and max_dwell == 0:
        return "target_exit_failure"
    if centered and max_dwell == 0:
        return "settling_failure"
    if 0 < max_dwell < 6:
        return "dwell_recovery_failure"
    return "timeout"


def replay_certify(seed, executed, *, max_steps=360):
    """Reset true neutral with ``seed``, replay the exact executed action sequence WITHOUT replanning, and verify the
    unchanged strict-K=6 certificate. Returns (certified, cert_step, max_dwell). Catches replay nondeterminism."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, neutral_env
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    env.set_stage(0)
    env.reset(seed=int(seed))
    cert = DeliveryCertifier(initial_clearance=_clearance(inner))
    md = 0
    for t, a in enumerate(executed):
        cert.update(_cert_step(inner, cf))
        md = max(md, int(cert.delivery_dwell))
        if cert.delivery_certified:
            return True, t, md
        inner.step(np.asarray(a, np.float32))
    cert.update(_cert_step(inner, cf))
    return bool(cert.delivery_certified), len(executed), max(md, int(cert.delivery_dwell))


# ---- state capture (representative per-phase states with their FULL_ACTION_OBS_HISTORY_V1 key) -------------------

_PHASES = ["APPROACH", "CONTACT_ACQUISITION", "BILATERAL_OR_STABLE_CONTACT", "TRANSPORT",
           "TARGET_ENTRY", "SETTLING", "STRICT_DWELL"]


def _phase_code(inner, dwell, touched_ever, start_dtz):
    m = inner._planar_metrics
    dtz = float(m.disk_to_zone)
    if dwell > 0:
        return 6
    if dtz <= CENTER_TOL:
        return 5
    if dtz <= 0.04:
        return 4
    if touched_ever and dtz < start_dtz - 0.01:
        return 3
    if bool(m.left_contact and m.right_contact):
        return 2
    if bool(m.left_contact or m.right_contact) or touched_ever:
        return 1
    return 0


def capture_states(seeds, *, want_phases=(3, 4, 5, 6), per_phase=2, horizon=360):
    """Drive neutral→transport with the frozen E-approach + handoff transport, snapshotting (qpos, qvel, clearance,
    touched, dwell, obs_history feature, phase) at representative states of the requested phases. These are the
    fixed states the first-action stability gate (§4) probes. Deterministic per seed."""
    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.coin_delivery.full_action_dataset import _handoff_transport
    from hymeko_rl.coin_delivery.full_action_obs_history import ObsHistoryV1
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn
    from hymeko_rl.experiments.coin_neutral_start import _cert_step, _clearance, _e_approach_actor, neutral_env
    e = _e_approach_actor()
    tfn = _greedy_action_fn(_handoff_transport())
    import torch
    captured = []
    for s in seeds:
        counts = {p: 0 for p in want_phases}                 # per-SEED quota → cross-seed diversity in the probe set
        env, cf = neutral_env(prefix_steps=0)
        inner = cf._env
        env.set_stage(0)
        env.reset(seed=int(s))
        clearance = _clearance(inner)
        cert = DeliveryCertifier(initial_clearance=clearance)
        hist = ObsHistoryV1()
        hist.reset(np.asarray(inner.node_features(), np.float32).flatten())
        touched = False
        start_dtz = float(inner._planar_metrics.disk_to_zone)
        bi = 0
        in_transport = False
        o = None
        for _k in range(horizon):
            cert.update(_cert_step(inner, cf))
            m = inner._planar_metrics
            touched = touched or bool(m.left_contact or m.right_contact)
            ph = _phase_code(inner, cert.delivery_dwell, touched, start_dtz)
            if ph in want_phases and counts[ph] < per_phase:
                qpos, qvel = _snapshot(inner)
                captured.append({"seed": int(s), "phase": int(ph), "phase_name": _PHASES[ph],
                                 "qpos": qpos, "qvel": qvel, "clearance": float(clearance),
                                 "touched": bool(touched), "dwell": int(cert.delivery_dwell),
                                 "obs_hist": hist.feature().copy()})
                counts[ph] += 1
            if cert.delivery_certified:
                break
            if not in_transport and bi >= 3:                 # E-approach → transport handoff
                cf._prev_coin = np.asarray(inner._planar_metrics.disk_pos[:2], np.float64)
                cf._t = 0
                cf._both_hist = []
                env._suffix_t = 0
                env._prev_dtz = env._dtz()
                env._prev_both = env._both()
                o = cf._obs(np.zeros(4, np.float32))
                in_transport = True
            if in_transport:
                a6 = np.asarray(tfn(env, o, None), np.float32)
                o = env.step(a6)[0]
            else:
                bi = bi + 1 if (m.left_contact and m.right_contact) else 0
                a = e.action_mean(torch.as_tensor(np.asarray(inner.node_features(), np.float32)[None]))[0].detach().numpy()
                inner.step(np.asarray(a, np.float32))
            hist.push(np.asarray(inner.node_features(), np.float32).flatten(),
                      np.asarray(inner.data.ctrl[:4], np.float32))
    return captured


# ---- §4 first-action stability -----------------------------------------------------------------------------------

def _pairwise_cosine(mat):
    """Mean pairwise cosine similarity of the rows of ``mat`` (N, d). 1.0 = all agree in direction."""
    x = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    g = x @ x.T
    n = len(mat)
    iu = np.triu_indices(n, k=1)
    return float(g[iu].mean()) if n > 1 else 1.0


def _select(results):
    """Deterministic selection over per-seed (first_action, planning result): best predicted lexo, action tie-break."""
    keyed = sorted(results, key=lambda fa_r: (_lexo(fa_r[1]), tuple(np.round(fa_r[0], 6))))
    return keyed[-1]


def first_action_stability(inner, cf, state, *, n_seeds=6, horizon=15, pop=32, iters=6, elite=8):
    """Run the expert repeatedly (n_seeds search seeds) from ONE fixed captured state; report first-action agreement
    + the deterministically selected action + its predicted outcome. Restores the state before each plan."""
    firsts = []
    results = []
    for sd in range(n_seeds):
        _restore(inner, state["qpos"], state["qvel"])
        fa, res = plan_first_action(inner, cf, state["clearance"], state["touched"], state["dwell"], sd,
                                    horizon=horizon, pop=pop, iters=iters, elite=elite)
        firsts.append(fa)
        results.append((fa, res))
    fa_mat = np.stack(firsts)
    sel_fa, sel_res = _select(results)
    mean_mag = float(np.linalg.norm(fa_mat.mean(0)))
    std_abs = float(fa_mat.std(0).mean())
    # magnitude-aware conflict: cosine is meaningless when the optimal action is ~0 (settle = "hold still"). A state
    # CONFLICTS only if the action is non-trivial (|mean| > MAG_FLOOR) AND directions disagree (cos < COS_MIN) AND the
    # absolute spread is not tiny (std/|mean| large). Near-zero or tight-cluster actions are STABLE (well-defined label).
    mag_floor = 0.35
    conflicting = (mean_mag > mag_floor) and (_pairwise_cosine(fa_mat) < 0.5) and (std_abs / (mean_mag + 1e-9) > 0.6)
    return {"seed": state["seed"], "phase": state["phase"], "phase_name": state["phase_name"],
            "first_mean": fa_mat.mean(0).round(4).tolist(), "mean_mag": round(mean_mag, 4),
            "first_std_abs": round(std_abs, 4),
            "first_std_norm": round(std_abs / CTRL_LIM, 4),
            "rel_spread": round(std_abs / (mean_mag + 1e-9), 4),
            "pairwise_cosine": round(_pairwise_cosine(fa_mat), 4),
            "near_zero_action": bool(mean_mag <= mag_floor),
            "conflicting": bool(conflicting),
            "selected": sel_fa.round(4).tolist(),
            "predicted": {"strict": sel_res["strict"], "dwell": int(sel_res["dwell"]),
                          "min_dtz": round(sel_res["min_dtz"], 4), "min_speed": round(sel_res.get("min_speed", 9.0), 4)}}


def _run_stability_state(args):
    state, cfg = args
    from hymeko_rl.experiments.coin_neutral_start import neutral_env
    env, cf = neutral_env(prefix_steps=0)
    inner = cf._env
    return first_action_stability(inner, cf, state, **cfg)


def main():
    import argparse
    import json
    import multiprocessing as mp

    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1011, 1045, 1174, 1278, 1447])
    ap.add_argument("--n-search-seeds", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--per-phase", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cos-thresh", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    from hymeko_rl.coin_delivery.full_action_obs_history import contract_spec
    states = capture_states(args.seeds, per_phase=args.per_phase)
    print(f"captured {len(states)} representative states across phases "
          f"{sorted({s['phase_name'] for s in states})}", flush=True)
    cfg = {"n_seeds": args.n_search_seeds, "horizon": args.horizon, "pop": args.pop, "iters": args.iters}
    with mp.Pool(args.workers) as pool:
        rows = pool.map(_run_stability_state, [(s, cfg) for s in states])
    for r in rows:
        tag = "NEAR_ZERO" if r["near_zero_action"] else ("CONFLICT" if r["conflicting"] else "stable")
        print(f"  {r['phase_name']:<20} seed {r['seed']}: |mean|={r['mean_mag']:.3f} cos={r['pairwise_cosine']:.3f} "
              f"rel_spread={r['rel_spread']:.3f} [{tag}] pred(strict={r['predicted']['strict']},"
              f"dwell={r['predicted']['dwell']},dtz={r['predicted']['min_dtz']})", flush=True)
    # gate: magnitude-aware. A critical-phase state blocks only if it CONFLICTS (non-trivial action, disagreeing
    # direction, large relative spread) — cosine on near-zero settle/dwell actions is not counted.
    crit = [r for r in rows if r["phase"] in (3, 4, 5, 6)]
    conflicts = [r for r in crit if r["conflicting"]]
    passed = len(conflicts) == 0
    verdict = "FEEDBACK_EXPERT_FIRST_ACTION_STABILITY_PASS" if passed else "FEEDBACK_EXPERT_OBSERVATION_ALIASING_BLOCKED"
    summary = {"verdict": verdict, "n_states": len(states), "n_critical": len(crit), "n_conflicting": len(conflicts),
               "conflicting_states": [{"phase": r["phase_name"], "seed": r["seed"], "mean_mag": r["mean_mag"],
                                       "cos": r["pairwise_cosine"], "rel_spread": r["rel_spread"]} for r in conflicts],
               "config": cfg, "obs_contract_sha": contract_spec()["sha256"], "rows": rows}
    json.dump(summary, open(args.out, "w"), indent=1)
    print(f"\n{verdict}  conflicting {len(conflicts)}/{len(crit)} critical-phase states", flush=True)


if __name__ == "__main__":
    main()
