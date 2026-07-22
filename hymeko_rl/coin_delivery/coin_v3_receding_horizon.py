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
    for t in range(len(actions_h)):
        cert.update(_cert_step(inner, cf))
        best_dwell = max(best_dwell, cert.delivery_dwell)
        m = inner._planar_metrics
        min_dtz = min(min_dtz, float(m.disk_to_zone))
        sp = float(np.linalg.norm(inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]))
        min_speed = min(min_speed, sp)
        if cert.delivery_certified:
            strict = True
            break
        inner.step(np.asarray(actions_h[t], np.float32))
        if not np.all(np.isfinite(inner.data.qvel)):
            return {"strict": False, "dwell": best_dwell, "min_dtz": min_dtz, "min_speed": 9.0, "effort": 9.0}
    return {"strict": strict, "dwell": best_dwell, "min_dtz": min_dtz, "min_speed": min_speed,
            "effort": float(np.mean(np.abs(actions_h)))}


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
    for _it in range(iters):
        cand = np.clip(rng.normal(mean, sigma, size=(pop, dim)), -CTRL_LIM, CTRL_LIM).astype(np.float32)
        cand[0] = mean                                       # keep the incumbent
        scored = []
        for c in cand:
            r = _score_horizon(inner, cf, qpos, qvel, c.reshape(horizon, ACT_DIM), clearance, touched, dwell)
            scored.append((_lexo(r), c, r))
        scored.sort(key=lambda x: x[0])
        el = np.stack([c for _k, c, _r in scored[-elite:]])
        mean, sigma = el.mean(0), el.std(0) + 0.05
        if best is None or scored[-1][0] > _lexo(best):
            best, best_seq = scored[-1][2], scored[-1][1].copy()
    _restore(inner, qpos, qvel)
    return best_seq.reshape(horizon, ACT_DIM)[0].astype(np.float32), best


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
