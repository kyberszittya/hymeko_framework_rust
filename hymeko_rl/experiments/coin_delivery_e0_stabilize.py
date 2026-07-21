"""§5–§6 COMPETENCE-STABILIZATION for learned E0 Coin Delivery.

Preserve the competence already observed (static-BC SAC peaked at 6/9 then collapsed as α annealed to 0.005). One
explicit stabilized mode over the EXISTING canonical SAC (no new trainer): a **persistent** demonstration anchor
(bc_coef held constant, never decayed), an **entropy floor** (α held at the competence-associated level, resolved
from the committed 6/9 peak: α≈0.0367), and **dev-bank competence checkpointing** (checkpoint by COIN_DELIVERY_STRICT
coverage on a hash-disjoint dev bank; retain the best). Matched CONTROL = the previous static-BC protocol (α→0.005).

Reuses `coin_delivery_e0_campaign` (env, demos, certifier eval) and `train.sac`. No embodiment / certificate / grasp
change; no contact/relay/critic/replay/n-step reopen; no hyperparameter grid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_delivery_e0_campaign import (
    bc_fit, collect_e0_demos, direct_e0_env, evaluate_policy)
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_ALPHA_FLOOR = 0.0367          # α at the last reproducible 6/9 pre-collapse checkpoint (static run, step 16000)
_BC_COEF = 1.0                 # the static coefficient that produced the 6/9 peak — held CONSTANT (persistent anchor)
_SPLIT = "experiments/2026_07_21_coin_e0_stabilize/state_split.json"


def _load_split():
    d = json.loads(Path(_SPLIT).read_text())
    return ([s for s, _ in d["headline"]], [s for s, _ in d["dev"]], [s for s, _ in d["train"]])


def _coverage(actor, seeds, eval_cf) -> dict:
    """COIN_DELIVERY_STRICT coverage on a fixed state set (deterministic; count of states delivered)."""
    return evaluate_policy(actor, seeds, env_cf=eval_cf)


def train_arm(*, seed, arm, steps, eval_every, demos, dev, headline, train_pool, outdir):
    """One matched run. arm='STABILIZED' → α floor + persistent BC + dev-best retention; 'CONTROL' → α→0.005 static-BC.
    Both: same env / demos / splits / budget / persistent bc_coef; checkpoint by DEV coverage; final eval on HEADLINE."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    env, cf = direct_e0_env(train_seed_pool=tuple(train_pool))
    eval_cf = direct_e0_env()
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    actor, critics = build_sac("mlp", obs_dim=od, flat_dim=od, action_dim=ad, action_scale=1.0)
    bc_fit(actor, (demos[0], demos[1]), epochs=400, seed=seed)
    bc_head = _coverage(actor, headline, (eval_cf[0], eval_cf[1]))["delivery_count"]

    alpha_final = _ALPHA_FLOOR if arm == "STABILIZED" else 0.005
    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=_BC_COEF, alpha_final=alpha_final,
                           log_every=eval_every, eval_every=eval_every)
    best = {"dev": -1, "step": 0}
    hist = []
    ckpt = out / f"{arm.lower()}_s{seed}_best.pt"

    def eval_fn(_e, ac) -> float:
        m = _coverage(ac, dev, (eval_cf[0], eval_cf[1]))
        hist.append(m["delivery_count"])
        if m["delivery_count"] > best["dev"]:                       # lexicographic-1: dev COIN_DELIVERY_STRICT coverage
            best.update(dev=m["delivery_count"], step=len(hist) * eval_every)
            torch.save(ac.state_dict(), ckpt)                       # retain best (deployed checkpoint)
        print(f"  [{arm} s{seed} eval#{len(hist)}] dev {m['delivery_count']}/{len(dev)} "
              f"(best {best['dev']}) alpha_floor={alpha_final}", flush=True)
        return float(m["delivery_count"])

    print(f"[{arm} s{seed}] BC-init headline {bc_head}/9 | steps={steps} alpha_final={alpha_final} "
          f"bc_coef={_BC_COEF}(persistent)", flush=True)
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn, offline_data=(demos[0], demos[1]),
              bc_coef_fn=lambda _s: _BC_COEF)                       # persistent anchor: never decays
    final_dev = hist[-1] if hist else 0
    # evaluate the RETAINED BEST checkpoint on the locked headline (9 states × 10 deterministic restores)
    best_actor, _ = build_sac("mlp", obs_dim=od, flat_dim=od, action_dim=ad, action_scale=1.0)
    best_actor.load_state_dict(torch.load(ckpt, weights_only=True))
    head_rows = _headline_per_state(best_actor, headline, (eval_cf[0], eval_cf[1]))
    return {"arm": arm, "seed": seed, "bc_head": bc_head, "peak_dev": max(hist, default=0), "final_dev": final_dev,
            "dev_curve": hist, "best_step": best["step"], "ckpt": str(ckpt),
            "headline_states_8of10": sum(v >= 8 for v in head_rows.values()), "headline_per_state": head_rows}


def _headline_per_state(actor, headline, eval_cf) -> dict:
    """Per-locked-state 10-restore COIN_DELIVERY_STRICT count for the retained-best checkpoint (§7 criterion)."""
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _greedy_action_fn, delivery_rollout
    env, cf = eval_cf
    fn = _greedy_action_fn(actor)
    rows = {}
    for s in headline:
        rs = [delivery_rollout(env, cf, seed=int(s), action_fn=fn) for _ in range(10)]
        rows[int(s)] = int(sum(r["delivery_certified"] for r in rs))
    return rows


def campaign(*, seeds=(0, 1, 2, 3), reps=2, steps=24_000, eval_every=2_000,
            outdir="experiments/2026_07_21_coin_e0_stabilize"):
    """§6 matched campaign: CONTROL vs STABILIZED across seeds×reps. Primary endpoint = best reproducible headline
    coverage; secondary = final retained dev coverage, #states>=8/10, peak-to-final degradation."""
    headline, dev, train_pool = _load_split()
    demos = collect_e0_demos(train_pool)
    print(f"[campaign] headline={len(headline)} dev={len(dev)} train={len(train_pool)} | demos={len(demos[0])} pairs "
          f"| seeds={seeds} reps={reps} steps={steps}", flush=True)
    results = []
    for arm in ("CONTROL", "STABILIZED"):
        for seed in seeds:
            for rep in range(reps):
                r = train_arm(seed=seed * 10 + rep, arm=arm, steps=steps, eval_every=eval_every, demos=demos,
                              dev=dev, headline=headline, train_pool=train_pool, outdir=outdir)
                results.append(r)
                print(f"  => {arm} s{seed}.{rep}: peak_dev={r['peak_dev']} final_dev={r['final_dev']} "
                      f"headline>=8/10={r['headline_states_8of10']}", flush=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "campaign.json").write_text(json.dumps(results, indent=1, default=float))
    _summarize(results)
    return results


def _summarize(results) -> None:
    for arm in ("CONTROL", "STABILIZED"):
        rs = [r for r in results if r["arm"] == arm]
        peak = [r["peak_dev"] for r in rs]
        final = [r["final_dev"] for r in rs]
        h8 = [r["headline_states_8of10"] for r in rs]
        print(f"[{arm}] n={len(rs)} peak_dev med={int(np.median(peak))} final_dev med={int(np.median(final))} "
              f"degrade med={int(np.median(np.array(peak) - np.array(final)))} | headline>=8/10 states: "
              f"max={max(h8)} any-run-with-1+={sum(v >= 1 for v in h8)}/{len(rs)}", flush=True)


def verify_learned(ckpt, seed, *, render_dir=None) -> int:
    """§9 canonical deterministic verification: restore the frozen state, ASSERT positive clearance, load the learned
    checkpoint, run deterministic learned actions, print COIN_DELIVERY_STRICT, render, return 0 on success else 1."""
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _clearance, _greedy_action_fn, delivery_rollout
    env, cf = direct_e0_env()
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    actor, _ = build_sac("mlp", obs_dim=od, flat_dim=od, action_dim=ad, action_scale=1.0)
    actor.load_state_dict(torch.load(ckpt, weights_only=True))
    env.reset(seed=int(seed))
    clr = _clearance(cf._env)
    print(f"[verify] state seed={seed} initial signed clearance = {clr:+.4f}", flush=True)
    if clr <= 0.0:
        print("[verify] FAIL: initial footprints not disjoint (clearance <= 0)", flush=True)
        return 1
    fn = _greedy_action_fn(actor)
    rs = [delivery_rollout(env, cf, seed=int(seed), action_fn=fn) for _ in range(10)]
    n = sum(r["delivery_certified"] for r in rs)
    zero = sum(delivery_rollout(env, cf, seed=int(seed), zero_action=True)["delivery_certified"] for _ in range(10))
    print(f"[verify] LEARNED policy COIN_DELIVERY_STRICT = {n}/10 | zero-action = {zero}/10 | "
          f"grasp={sum(r['grasp_certified'] for r in rs)}/10", flush=True)
    ok = n >= 8 and zero == 0 and clr >= 0.030
    if render_dir is not None and ok:
        _render_learned(env, cf, actor, int(seed), clr, render_dir)
    print(f"[verify] {'PASS' if ok else 'FAIL'}: learned>=8/10 ∧ zero==0/10 ∧ clearance>=+0.030", flush=True)
    return 0 if ok else 1


def _render_learned(env, cf, actor, seed, clr, render_dir) -> None:
    """Render the learned clear-start delivery to GIF + MP4 with the annotated §9 HUD."""

    from hymeko_rl.coin_delivery.delivery_certificate import DeliveryCertifier
    from hymeko_rl.eval.evaluate import render_episode_gif
    from hymeko_rl.experiments.coin_delivery_e0_campaign import _cert_step, _greedy_action_fn
    inner = cf._env
    env.model, env.data, env.max_steps = inner.model, inner.data, 240
    Path(render_dir).mkdir(parents=True, exist_ok=True)
    fn = _greedy_action_fn(actor)
    st = {"cert": None, "phase": "START — COIN OUTSIDE TARGET"}

    def act(e, obs):
        s = _cert_step(inner, cf)
        if st["cert"] is None:
            st["cert"] = DeliveryCertifier(initial_clearance=clr)
        st["cert"].update(s)
        if st["cert"].delivery_certified:
            st["phase"] = "CERTIFIED DELIVERY"
        elif s.disk_to_zone <= 0.02:
            st["phase"] = "ZONE ENTRY / SETTLED"
        elif s.left_fingertip or s.right_fingertip:
            st["phase"] = "ROBOT CONTACT / TARGETWARD PUSH"
        return fn(e, obs, None)

    def hud(d):
        return [f"LEARNED POLICY  clearance {clr:+.4f}", st["phase"]]
    gif = render_episode_gif(env, act, f"{render_dir}/coin_delivery_learned_clear_start.gif",
                             seed=seed, width=600, height=440, fps=16, overlay=hud)
    print(f"[render] wrote {gif}", flush=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="campaign", choices=["campaign", "verify"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=24_000)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_e0_stabilize")
    ap.add_argument("--ckpt", default="experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt")
    ap.add_argument("--seed", type=int, default=1174)
    ap.add_argument("--render-dir", default=None)
    a = ap.parse_args()
    if a.cmd == "verify":
        raise SystemExit(verify_learned(a.ckpt, a.seed, render_dir=a.render_dir))
    campaign(seeds=tuple(a.seeds), reps=a.reps, steps=a.steps, outdir=a.out)


if __name__ == "__main__":
    main()
