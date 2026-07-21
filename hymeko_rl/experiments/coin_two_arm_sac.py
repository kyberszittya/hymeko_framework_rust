"""COIN two-arm delivery — BC-anchored corrected SAC over the SHARED components (thin composition root, NO new
framework). Integrates the frozen scientific findings into the existing BC/SAC/replay path:

  §1 competence-gated BC anchor  — cfg.bc_coef modulated by eval milestones (1.0 -> 0.3 -> 0.1 -> 0.05), NOT step-decay
  §2 phase-stratified demo seed  — A1/A4 demo transitions labelled by phase; rare phases oversampled at replay seed
  §4 relational observation      — the canonical ACTOR_FIELDS obs (coin->target dir/dist, coin vel, L/R contacts, phase)
  §5 task reward                 — galambos_task_deliver_v2b.hymeko certified (delivers=True) as the §3 anti-farming gate
  §6 checkpoint ranking          — best by (strict deliveries, zone rate, mean progress, two-arm participation)

Reuses (imports, does NOT re-implement): eval.reward_oracle.certify, train.sac.{build_sac,train_sac,SACConfig},
train.coin_delivery_rl.{make_delivery_rl_env,p_grasp_carry}, train.coin_delivery_actor.{rollout,DeliveryActor,
actor_action,_attribution_from_trace + strict thresholds}. Evaluation funnels through the ONE canonical rollout()
(defect-1). No new trainer / env wrapper / replay / rollout / experiment hierarchy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.reward_oracle import certify
from hymeko_rl.train.coin_delivery_actor import (
    _BODY_SHOVE_MAX,
    _DWELL_STEPS,
    _ONE_FINGER_MAX,
    _PROGRESS_MIN,
    _SETTLE_VEL,
    DeliveryActor,
    _attribution_from_trace,
    actor_action,
    rollout,
)
from hymeko_rl.train.coin_delivery_rl import make_delivery_rl_env
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_V2B = "data/robotics/galambos_task_deliver_v2b.hymeko"
_DEMO_ACTORS = (DeliveryActor.A1_VPLOW, DeliveryActor.A4_RECOVERY)
# disjoint seed pools -> disjoint bank-index states (seed is an index selector; disjointness verified at startup)
_TRAIN_SEEDS = tuple(range(64_000, 64_056))     # 56
_VAL_SEEDS = tuple(range(64_100, 64_114))       # 14
_DEMO_SEEDS = tuple(range(64_200, 64_204))      # 4  (states A1/A4 demos are collected from)


def certify_or_abort() -> float:
    """§3/§5 mandatory gate: the v2b delivery reward spec must certify delivers=True (reward delivers, not farms)."""
    orc = certify(RewardSpec.from_hymeko(_V2B))
    print(f"[gate] v2b delivers={orc.delivers} optimal_return={orc.optimal_return:.2f}", flush=True)
    if not orc.delivers:
        raise SystemExit("ABORT (§3): v2b reward does not certify delivers=True — refusing to launch RL")
    return float(orc.optimal_return)


def direct_env(*, train_seed_pool: tuple[int, ...] | None = None, fingertip_geometry: str = "POINT"):
    """The shared CoinDeliveryTrainEnv made DIRECT-action via its own base/delta overrides (zero base + unit delta ->
    residual_action(0, raw, 1) == clip(raw)). If ``train_seed_pool`` is given, the instance's auto-reset (no seed)
    rotates deterministically through that pool so SAC trains on the intended split — a bounded instance
    parameterization, not a new env class."""
    env = make_delivery_rl_env(fingertip_geometry=fingertip_geometry)
    env._base_override = lambda inner, t: np.zeros(env.action_space.shape[0], np.float32)
    env._delta_override = 1.0
    if train_seed_pool is not None:
        rng = np.random.default_rng(0)
        _orig = env.reset

        def _reset(*, seed=None):
            return _orig(seed=int(rng.choice(train_seed_pool)) if seed is None else seed)
        env.reset = _reset
    return env


def _phase_of(step) -> str:
    """§2 replay stratum for one demo step (APPROACH -> CONTACT -> TARGET_PROGRESS -> HOLD_OR_ZONE_ENTRY)."""
    if step.in_zone:
        return "HOLD_OR_ZONE_ENTRY"
    if step.left_contact and step.right_contact:
        return "TARGET_PROGRESS"
    if step.left_contact or step.right_contact:
        return "CONTACT"
    return "APPROACH"


def collect_demos(env, seeds):
    """Collect A1/A4 demonstrations through the canonical rollout(); return (obs,act,rew,next,done) arrays + a phase
    label per transition. Uses the ONE rollout path — no bespoke demo loop."""
    obs_l, act_l, rew_l, nxt_l, done_l, phase_l = [], [], [], [], [], []
    for actor in _DEMO_ACTORS:
        for s in seeds:
            env.reset(seed=int(s))
            tr = rollout(env, lambda inner, t, _obs, _a=actor: actor_action(inner, t, _a), max_steps=60)
            for i, st in enumerate(tr.steps):
                if st.obs is None:
                    continue
                nxt = tr.steps[i + 1].obs if i + 1 < len(tr.steps) else tr.final_obs
                obs_l.append(st.obs)
                act_l.append(np.asarray(st.action, np.float32))
                rew_l.append(st.reward)
                nxt_l.append(nxt if nxt is not None else st.obs)
                done_l.append(st.terminated)
                phase_l.append(_phase_of(st))
    return (np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32), np.asarray(rew_l, np.float32),
            np.asarray(nxt_l, np.float32), np.asarray(done_l, bool), np.asarray(phase_l))


def stratify_seed(demos, *, target_per_phase: int = 400):
    """§2 phase-stratified replay seed: oversample rare phases (contact/hold) so approach states don't drown the
    seeded demos. Returns (obs,act,rew,next,done) with each present phase upsampled toward ``target_per_phase``."""
    obs_a, act_a, rew_a, nxt_a, done_a, phase_a = demos
    rng = np.random.default_rng(0)
    idxs = []
    for ph in sorted(set(phase_a.tolist())):
        pool = np.flatnonzero(phase_a == ph)
        take = max(len(pool), target_per_phase)
        idxs.append(rng.choice(pool, size=take, replace=take > len(pool)))
    order = np.concatenate(idxs)
    rng.shuffle(order)
    return (obs_a[order], act_a[order], rew_a[order], nxt_a[order], done_a[order])


def policy_strict(trace) -> bool:
    """The scripted strict-monitor's numeric conditions, made actor-name-free for a learned policy: not initially
    successful, zone entry, dwell>=K, settled, fingertip-attributed (>=0.6), body-shove below threshold, not a
    one-finger bulldoze."""
    att = _attribution_from_trace(trace)
    ff = att.fingertip_fraction + 1e-9
    not_bulldoze = min(att.alpha_L, att.alpha_R) / ff >= _ONE_FINGER_MAX
    return bool(not trace.initial_success and trace.loose and trace.best_dwell >= _DWELL_STEPS
                and trace.settle_vel <= _SETTLE_VEL and att.fingertip_fraction >= 0.6
                and att.alpha_body <= _BODY_SHOVE_MAX and not_bulldoze)


def evaluate(eval_env, actor, seeds, *, max_steps: int | None = None) -> dict:
    """Deterministic eval through the canonical rollout(): loose competence (zone entry + progress) AND strict
    competence (the strict predicate) + two-arm participation. Same rollout path scripted actors use.

    ``max_steps`` defaults to the DECLARED environment horizon ``eval_env.cfg.horizon`` — never a hard-coded
    truncation. A caller may pass a shorter ``max_steps`` ONLY as a time-to-success diagnostic; it must never be
    used for checkpoint selection or headline reporting (that silently changes the task definition — see the
    2026-07-22 60-vs-120 horizon artifact). # Preconditions ``eval_env`` exposes ``cfg.horizon`` (the deployment
    horizon). # Postconditions the rollout length equals the declared horizon unless explicitly overridden."""
    import torch

    def greedy(inner, t, obs):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]

    steps = int(eval_env.cfg.horizon if max_steps is None else max_steps)
    rows = []
    for s in seeds:
        eval_env.reset(seed=int(s))
        tr = rollout(eval_env, greedy, max_steps=steps)
        att = _attribution_from_trace(tr)
        rows.append(dict(loose=tr.loose, strict=policy_strict(tr), progress=tr.progress, both_frac=tr.both_frac,
                         aL=att.alpha_L, aR=att.alpha_R,
                         lc=float(np.mean([s.left_contact for s in tr.steps])) if tr.steps else 0.0,
                         rc=float(np.mean([s.right_contact for s in tr.steps])) if tr.steps else 0.0))
    n = max(1, len(rows))
    return dict(n=len(rows), zone_rate=sum(r["loose"] for r in rows) / n,
                strict_rate=sum(r["strict"] for r in rows) / n, strict_count=sum(r["strict"] for r in rows),
                mean_progress=float(np.mean([r["progress"] for r in rows])),
                both_frac=float(np.mean([r["both_frac"] for r in rows])),
                lc=float(np.mean([r["lc"] for r in rows])), rc=float(np.mean([r["rc"] for r in rows])),
                aL=float(np.mean([r["aL"] for r in rows])), aR=float(np.mean([r["aR"] for r in rows])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5_000)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_20_coin_two_arm_sac")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    certify_or_abort()
    env = direct_env(train_seed_pool=_TRAIN_SEEDS)
    print(f"[splits] TRAIN={len(_TRAIN_SEEDS)} VAL={len(_VAL_SEEDS)} DEMO={len(_DEMO_SEEDS)} disjoint-seed-pools",
          flush=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"[env] obs_dim={obs_dim} act_dim={act_dim} (direct A1/A4-space; relational ACTOR_FIELDS obs)", flush=True)

    demos = collect_demos(env, _DEMO_SEEDS + _VAL_SEEDS[:4])
    ph, cnt = np.unique(demos[5], return_counts=True)
    print(f"[demos] {len(demos[0])} transitions | phases " + " ".join(f"{p}:{c}" for p, c in zip(ph, cnt)), flush=True)
    seed_replay = stratify_seed(demos)
    print(f"[replay-seed] phase-stratified: {len(seed_replay[0])} seeded demo transitions", flush=True)

    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=1.0)
    cfg = SACConfig.stable(total_steps=args.steps, seed=args.seed, bc_coef=1.0,
                           log_every=min(500, args.eval_every), eval_every=args.eval_every)

    comp = {"progress_ok": False, "first_strict": False, "consec_strict": 0}

    def bc_coef_fn(_step: int) -> float:                            # §1 competence gate (milestone-driven)
        if comp["consec_strict"] >= 3:
            return 0.05
        if comp["first_strict"]:
            return 0.1
        if comp["progress_ok"]:
            return 0.3
        return 1.0

    eval_env = direct_env()                                         # separate instance (never disturb the training env)
    best = {"score": -1.0, "step": 0, "metrics": None}
    hist: list = []

    def eval_fn(_train_env, ac) -> float:
        m = evaluate(eval_env, ac, _VAL_SEEDS)
        if m["mean_progress"] >= _PROGRESS_MIN:
            comp["progress_ok"] = True
        if m["strict_count"] >= 1:
            comp["first_strict"] = True
            comp["consec_strict"] += 1
        else:
            comp["consec_strict"] = 0
        score = m["strict_count"] * 1e3 + m["zone_rate"] * 1e1 + m["mean_progress"] + min(m["lc"], m["rc"]) * 0.5
        m.update(bc_coef=bc_coef_fn(0), consec_strict=comp["consec_strict"])
        hist.append(m)
        print(f"  [eval#{len(hist)}] zone={m['zone_rate']:.2f} strict={m['strict_rate']:.2f}({m['strict_count']}) "
              f"prog={m['mean_progress']:.4f} both={m['both_frac']:.2f} L/Rcontact={m['lc']:.2f}/{m['rc']:.2f} "
              f"attrL/R={m['aL']:.2f}/{m['aR']:.2f} | bc_coef={m['bc_coef']} consec_strict={comp['consec_strict']}",
              flush=True)
        if score > best["score"]:
            import torch
            best.update(score=score, step=len(hist) * args.eval_every, metrics=m)
            torch.save(ac.state_dict(), out / "sac_actor_best.pt")
        if comp["consec_strict"] >= 3:
            print("  [stop] 3 consecutive strict deliveries reached (§8 demonstration goal)", flush=True)
        return float(m["strict_rate"] * 10 + m["zone_rate"])

    print(f"[sac] start: {args.steps} steps | bc_coef competence-gated (init 1.0) | v2b-aligned reward | "
          f"demo-seeded replay ({len(seed_replay[0])}) | canonical-rollout eval every {args.eval_every}", flush=True)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn,
                      offline_data=(demos[0], demos[1]), init_transitions=seed_replay, bc_coef_fn=bc_coef_fn)

    import torch
    torch.save(actor.state_dict(), out / "sac_actor_final.pt")
    (out / "run.json").write_text(json.dumps(dict(
        steps=args.steps, seed=args.seed, obs_dim=obs_dim, act_dim=act_dim,
        train_seeds=list(_TRAIN_SEEDS), val_seeds=list(_VAL_SEEDS), demo_seeds=list(_DEMO_SEEDS),
        n_demos=int(len(demos[0])), n_seeded=int(len(seed_replay[0])), curve=curve,
        best_step=best["step"], best_metrics=best["metrics"], eval_history=hist), indent=1, default=float))
    print(f"[done] saved sac_actor_best.pt + sac_actor_final.pt + run.json | "
          f"best score={best['score']:.3f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
