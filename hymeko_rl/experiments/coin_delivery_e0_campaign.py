"""E0 push/coast Coin-Delivery: oracle reproduction → demonstrations → learned campaign (task-semantics fix).

Coin Delivery ≠ Grasp Delivery (see :mod:`hymeko_rl.coin_delivery.delivery_certificate`). This module executes, on
the simplest embodiment E0 (passive concave ring, no wrist/closure DoF):

  §4 reproduce  — freeze the strongest clear-start E0 states, verify each with N deterministic restores against
                  COIN_DELIVERY_STRICT, prove zero-action delivers 0/N (robot-attribution), disjoint footprints.
  §5 demos      — record the successful scripted push/coast trajectories on the canonical action/obs schema.
  §6 learn      — BC-init a policy from the demos, then canonical SAC around the successful states (bands + rehearsal).
  §7 causal     — on identical states compare scripted oracle / BC-init / trained / zero-action.

Force closure is NEVER required. Reuses :func:`coin_wristed_delivery.make_wristed_delivery_env` (E0) and the shared
:class:`DeliveryCertifier`; no new simulator, no CORE change.
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.delivery_certificate import CertStep, DeliveryCertifier, DeliveryThresholds
from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.pad_actuation import Phase

_HOLD = np.array([0.0, 0.0, 0.0, 0.55, 0.0, 0.0], np.float32)      # arm holds, pads squeezed (no targetward push)
_ZERO_BASE = np.zeros(6, np.float32)                              # true no-op: no push, no squeeze


def _e0_env():
    from hymeko_rl.experiments.coin_wristed_delivery import make_wristed_delivery_env
    return make_wristed_delivery_env("E0")


def _clearance(inner) -> float:
    disk_r = float(inner.model.geom_size[inner._disk_geom][0])
    return float(inner.planar_metrics.disk_to_zone) - (disk_r + float(inner._zone_half))


def _cert_step(inner, cf) -> CertStep:
    """Read one certificate step from the live env metrics (real fields only)."""
    met = inner._planar_metrics
    lg = getattr(met, "legality", None)
    lf = bool(lg.left_fingertip_contact) if lg is not None else bool(met.left_contact)
    rf = bool(lg.right_fingertip_contact) if lg is not None else bool(met.right_contact)
    body = bool(lg.arm_body_contact) if lg is not None else False
    imp = float(lg.arm_body_contact_impulse) if lg is not None else 0.0
    v = inner.data.qvel[inner._disk_x_adr:inner._disk_x_adr + 2]
    from hymeko_rl.experiments.coin_wristed_delivery import _both_pad_contact
    _both, fl, fr = _both_pad_contact(cf)
    return CertStep(disk_to_zone=float(met.disk_to_zone), disk_speed=float(np.linalg.norm(v)),
                    left_fingertip=lf, right_fingertip=rf, arm_body_contact=body, arm_body_impulse=imp,
                    force_left=fl, force_right=fr)


def delivery_rollout(env, cf, *, seed, action_fn=None, zero_action=False, max_steps=260, th=None, record=False):
    """Roll one E0 delivery episode; certify with :class:`DeliveryCertifier`. ``action_fn(env, obs, phase)`` supplies a
    learned residual (default zero residual = pure scripted push); ``zero_action`` forces a true no-op base (control)."""
    th = th or DeliveryThresholds()
    reset_out = env.reset(seed=seed)                              # (obs, info) or bare obs — normalise
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    inner = cf._env
    clr = _clearance(inner)
    cert = DeliveryCertifier(initial_clearance=clr, th=th)
    brake_r = float(inner._zone_half) * 1.5
    st = {"p": Phase.APPROACH}

    def base(innr, _t):
        if zero_action:
            return _ZERO_BASE
        d, _n = innr.direction_to_zone()
        p = st["p"]
        dtz_b = float(innr._planar_metrics.disk_to_zone)
        if p is Phase.APPROACH:
            return np.array([d[0], d[1], 0.0, 0.6, 0.0, 0.0], np.float32)
        if p in (Phase.WRIST_ALIGN, Phase.PAD_CLOSE, Phase.FORCE_HOLD):
            return _HOLD
        if p is Phase.TRANSPORT:
            s = float(min(1.0, dtz_b / brake_r))
            return np.array([d[0] * s, d[1] * s, 0.0, 0.6, 0.0, 0.0], np.float32)
        return np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0], np.float32)                 # BRAKE: zero transport, hold

    env._base_override = base
    trace = {"obs": [], "act_base": [], "coin": [], "vel": [], "contact": []} if record else None
    n_act = env.action_space.shape[0]
    for _t in range(max_steps):
        cf.set_phase(st["p"] if st["p"] is not Phase.BRAKE else Phase.FORCE_HOLD)
        met = inner._planar_metrics
        dtz = float(met.disk_to_zone)
        s = _cert_step(inner, cf)
        cert.update(s)
        raw = (np.zeros(n_act, np.float32) if (zero_action or action_fn is None)
               else np.asarray(action_fn(env, obs, st["p"]), np.float32).reshape(-1))
        if record:
            trace["obs"].append(np.asarray(obs, np.float32).copy())
            trace["act_base"].append(base(inner, _t).copy())      # the executed push (direct-action BC target)
            trace["coin"].append(np.asarray(met.disk_pos, np.float32).copy())
            trace["vel"].append(float(s.disk_speed))
            trace["contact"].append((s.left_fingertip, s.right_fingertip))
        # phase machine (delivery: no release — hold in zone)
        p = st["p"]
        both = s.left_fingertip and s.right_fingertip
        if p is Phase.APPROACH and (met.left_contact or met.right_contact):
            p = Phase.PAD_CLOSE
        elif p is Phase.PAD_CLOSE and both:
            p = Phase.FORCE_HOLD
        elif p is Phase.FORCE_HOLD and (both or (met.left_contact and met.right_contact)):
            p = Phase.TRANSPORT
        elif p is Phase.TRANSPORT and dtz <= brake_r:
            p = Phase.BRAKE
        st["p"] = p
        obs = env.step(raw)[0]
        if cert.delivery_certified:
            break
    out = cert.summary()
    out["seed"] = seed
    if record:
        out["trace"] = {k: (np.asarray(v) if k != "contact" else v) for k, v in trace.items()}
    return out


def direct_e0_env(*, train_seed_pool=None):
    """E0 delivery env made DIRECT-action (base=0, delta=1 → a_exec = clip(tanh(raw))). If ``train_seed_pool`` is given,
    seedless auto-reset rotates deterministically through it (so SAC trains on the intended split)."""
    env, cf = _e0_env()
    env._base_override = lambda inner, t: np.zeros(env.action_space.shape[0], np.float32)
    env._delta_override = 1.0
    if train_seed_pool is not None:
        rng = np.random.default_rng(0)
        _orig = env.reset

        def _reset(*, seed=None, options=None):
            return _orig(seed=int(rng.choice(train_seed_pool)) if seed is None else seed)
        env.reset = _reset
    return env, cf


def collect_e0_demos(states, *, th=None):
    """§5: record the successful scripted push/coast trajectories as (obs, executed_action) for BC. The executed action
    is ``clip(scripted_base)`` (the direct-action target a policy must reproduce). Only certified deliveries are kept."""
    from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig
    env, cf = _e0_env()
    cfg = DeliveryRLConfig()
    obs_l, act_l = [], []
    kept = 0
    for seed in states:
        r = delivery_rollout(env, cf, seed=seed, th=th, record=True)
        if not r["delivery_certified"]:
            continue
        kept += 1
        tr = r["trace"]
        for o, base in zip(tr["obs"], tr["act_base"]):
            obs_l.append(o)
            act_l.append(np.clip(base, cfg.lo, cfg.hi))
    return np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32), kept


def _greedy_action_fn(actor):
    import torch

    def fn(env, obs, _phase):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).numpy()[0]
    return fn


def evaluate_policy(actor, seeds, *, th=None, env_cf=None) -> dict:
    """§7 eval: COIN_DELIVERY_STRICT rate of a learned actor on fixed states (direct-action env)."""
    env, cf = env_cf or direct_e0_env()
    fn = _greedy_action_fn(actor)
    rows = [delivery_rollout(env, cf, seed=int(s), action_fn=fn, th=th) for s in seeds]
    n = max(1, len(rows))
    return {"n": len(rows), "delivery_rate": sum(r["delivery_certified"] for r in rows) / n,
            "delivery_count": sum(r["delivery_certified"] for r in rows),
            "best_dwell": max((r["best_delivery_dwell"] for r in rows), default=0),
            "grasp_count": sum(r["grasp_certified"] for r in rows)}


def bc_fit(actor, demos, *, epochs=300, lr=1e-3, seed=0):
    """BC-initialise the actor: MSE(action_mean(obs), demo_action). Deterministic (seeded); returns the loss curve."""
    import torch
    torch.manual_seed(seed)
    obs = torch.as_tensor(demos[0], dtype=torch.float32)
    act = torch.as_tensor(demos[1], dtype=torch.float32)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((actor.action_mean(obs) - act) ** 2)
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            curve.append(float(loss))
    return curve


def reproduce_state(seed, *, n=10, th=None) -> dict:
    """§4: reproduce one clear-start state under N deterministic restores + N zero-action controls; freeze provenance."""
    env, cf = _e0_env()
    inner = cf._env
    scripted = [delivery_rollout(env, cf, seed=seed, th=th) for _ in range(n)]
    zero = [delivery_rollout(env, cf, seed=seed, zero_action=True, th=th) for _ in range(n)]
    env.reset(seed=seed)
    clr = _clearance(inner)
    # StateId anchor: the restored bank snapshot hash + compiled-model hash
    snap = getattr(cf, "_last_snap", None)
    snap_h = snapshot_hash(snap) if snap is not None else "n/a"
    import hashlib
    model_h = hashlib.sha256(np.ascontiguousarray(inner.model.geom_size).tobytes()
                             + np.ascontiguousarray(inner.model.jnt_type).tobytes()).hexdigest()[:12]
    n_deliver = sum(r["delivery_certified"] for r in scripted)
    n_zero = sum(r["delivery_certified"] for r in zero)
    return {"seed": seed, "initial_clearance": round(clr, 5), "footprints_disjoint": clr > 0.0,
            "snapshot_sha256": snap_h[:12], "model_sha256": model_h,
            "scripted_delivery": f"{n_deliver}/{n}", "zero_action_delivery": f"{n_zero}/{n}",
            "best_delivery_dwell": max(r["best_delivery_dwell"] for r in scripted),
            "grasp_certified_any": any(r["grasp_certified"] for r in scripted),
            "n_deliver": n_deliver, "n_zero": n_zero, "n": n}


def scan_delivery_states(seed_lo=1000, seed_hi=1700):
    """Return the clear-start (positive-clearance) E0 states that certify COIN_DELIVERY_STRICT, with their clearance."""
    env, cf = _e0_env()
    out = []
    for s in range(seed_lo, seed_hi):
        r = delivery_rollout(env, cf, seed=s)
        if r["initial_clearance"] > 0.005 and r["delivery_certified"]:
            out.append((s, round(r["initial_clearance"], 4)))
    return out


# §7 headline eval states (clearance >= +0.030, deliverable by the scripted oracle); disjoint from the train pool
_HEADLINE = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)     # 9 clear starts >= +0.030 (up to +0.0698)


def train_campaign(*, steps=20_000, seed=0, out="experiments/2026_07_21_coin_e0_learned", eval_every=2_000):
    """§5–§7: demos → BC-init → bounded SAC on E0 (direct-action) → causal-validity eval. Live progress; checkpoints."""
    import json
    from pathlib import Path

    import torch

    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    states = scan_delivery_states()
    train_states = [s for s, _c in states if s not in _HEADLINE]        # demo/train pool disjoint from eval
    print(f"[states] {len(states)} clear-start delivery states | train_pool={len(train_states)} "
          f"headline(>=+0.030)={_HEADLINE}", flush=True)
    demos = collect_e0_demos(train_states)
    print(f"[demos] {demos[2]} certified trajectories -> {len(demos[0])} (obs,act) pairs", flush=True)
    if len(demos[0]) < 20:
        raise SystemExit("BLOCKED: too few demonstration pairs to BC-init")

    env, cf = direct_e0_env(train_seed_pool=tuple(train_states))
    eval_cf = direct_e0_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=1.0)

    bc_curve = bc_fit(actor, (demos[0], demos[1]), epochs=400, seed=seed)
    torch.save(actor.state_dict(), outp / "bc_init.pt")
    bc_eval = evaluate_policy(actor, _HEADLINE, env_cf=(eval_cf[0], eval_cf[1]))
    print(f"[bc-init] loss {bc_curve[0]:.3f}->{bc_curve[-1]:.4f} | headline delivery "
          f"{bc_eval['delivery_count']}/{bc_eval['n']} (best_dwell={bc_eval['best_dwell']})", flush=True)

    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0, log_every=min(500, eval_every),
                           eval_every=eval_every)
    best = {"score": -1.0, "count": -1, "metrics": bc_eval}
    hist = []
    comp = {"best_count": 0}

    def bc_coef_fn(_step: int) -> float:                            # competence-gated anchor with a FLOOR (anti-collapse)
        if comp["best_count"] >= 6:
            return 0.3
        if comp["best_count"] >= 3:
            return 0.5
        return 1.0

    def eval_fn(_e, ac) -> float:
        m = evaluate_policy(ac, _HEADLINE, env_cf=(eval_cf[0], eval_cf[1]))
        hist.append(m)
        comp["best_count"] = max(comp["best_count"], m["delivery_count"])
        print(f"  [eval#{len(hist)}] headline delivery {m['delivery_count']}/{m['n']} "
              f"rate={m['delivery_rate']:.2f} best_dwell={m['best_dwell']} grasp={m['grasp_count']} "
              f"bc_coef={bc_coef_fn(0)}", flush=True)
        score = m["delivery_count"] * 100 + m["best_dwell"]
        if score > best["score"]:
            best.update(score=score, count=m["delivery_count"], metrics=m)
            torch.save(ac.state_dict(), outp / "sac_actor_best.pt")
        return float(m["delivery_rate"])

    print(f"[sac] start {steps} steps | direct-action E0 | demo-anchored (bc_coef competence-gated, floor 0.3) | "
          f"eval every {eval_every} | reward=canonical delivery_reward, selection=COIN_DELIVERY_STRICT", flush=True)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(demos[0], demos[1]),
                      bc_coef_fn=bc_coef_fn)
    torch.save(actor.state_dict(), outp / "sac_actor_final.pt")
    (outp / "run.json").write_text(json.dumps(dict(
        steps=steps, seed=seed, obs_dim=obs_dim, act_dim=act_dim, n_demos=int(len(demos[0])),
        train_states=train_states, headline=list(_HEADLINE), bc_curve=bc_curve, bc_eval=bc_eval,
        curve=curve, best=best["metrics"], eval_history=hist), indent=1, default=float))
    print(f"[done] best headline delivery {best['count']}/{len(_HEADLINE)} | saved bc_init/sac_actor_best/final + run.json",
          flush=True)
    return best


def main() -> None:
    import argparse
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reproduce", "train"])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_e0_learned")
    a = ap.parse_args()
    if a.cmd == "reproduce":
        rows = [reproduce_state(s, n=10) for s in _HEADLINE]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "frozen_states.json").write_text(json.dumps(rows, indent=1, default=str))
        for r in rows:
            print(f"seed {r['seed']} clr={r['initial_clearance']:+.4f} disjoint={r['footprints_disjoint']} "
                  f"scripted={r['scripted_delivery']} zero={r['zero_action_delivery']} "
                  f"dwell={r['best_delivery_dwell']} grasp={r['grasp_certified_any']} model={r['model_sha256']}",
                  flush=True)
    else:
        train_campaign(steps=a.steps, seed=a.seed, out=a.out)


if __name__ == "__main__":
    main()
