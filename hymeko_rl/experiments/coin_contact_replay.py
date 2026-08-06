"""COIN contact-stratified replay experiment — matched CONTROL vs STRATIFIED continuation from sac_actor_best.pt.

Single experimental variable: the replay SAMPLER. Everything else identical (env, K0, obs schema, action semantics,
delivery-v2b reward, strict predicate, actor+critic architecture, SAC hyperparameters, BC competence gate, state splits,
eval path, seed). Reuses the canonical shared pieces; the only new code is the contact-quality STRATUM labelling of the
demo corpus (the sampler itself now lives in train.replay.ReplayBuffer.sample_stratified). No new replay buffer / trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.experiments.coin_two_arm_sac import (
    _DEMO_SEEDS,
    _VAL_SEEDS,
    certify_or_abort,
    direct_env,
    evaluate,
    policy_strict,
)
from hymeko_rl.train.coin_delivery_actor import (
    _ONE_FINGER_MAX,
    DeliveryActor,
    _attribution_from_trace,
    actor_action,
    rollout,
)
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

# stratum ids (0 == ONLINE, reserved by the buffer)
STRATA = {"CERTIFIED_BILATERAL": 1, "HIGH_QUALITY_CONTACT": 2, "RECOVERY": 3,
          "CONTRASTIVE_BULLDOZE": 4, "GENERAL_PROGRESS": 5}
STRATA_WEIGHTS = {1: 0.35, 2: 0.25, 3: 0.15, 4: 0.15, 5: 0.10}
GATE_CONFIRM = 2       # STRATIFIED->UNIFORM only after certified competence at this many consecutive evals (hysteresis)
# competence-state -> sampler mapping (reuses the existing bc_coef gate; NO new classifier, NO CONTROL input):
#   weak / pre-certified  (bc_coef in {1.0, 0.3}, or first_strict but consec_strict < GATE_CONFIRM) -> STRATIFIED
#   established competence (consec_strict >= GATE_CONFIRM, bc_coef 0.1/0.05, confirmed twice)         -> UNIFORM (irreversible)


def gate_step(gate: dict, comp: dict, *, eval_idx: int, step: int, bc_coef: float) -> str:
    """Update the STRATIFIED->UNIFORM replay gate from the run's OWN competence state ``comp`` (never the matched CONTROL
    result). The switch is irreversible and fires once certified competence is confirmed at ``GATE_CONFIRM`` consecutive
    deterministic evals (hysteresis against a single noisy eval). Records + returns the sampler mode.

    # Preconditions ``comp`` holds ``consec_strict``. # Postconditions once ``mode=='uniform'`` it never reverts."""
    if gate["mode"] == "stratified" and comp["consec_strict"] >= GATE_CONFIRM:
        gate.update(mode="uniform", switch_step=int(step), switch_eval=int(eval_idx), switch_bc=float(bc_coef),
                    switch_consec=int(comp["consec_strict"]),
                    reason=f"consec_strict>={GATE_CONFIRM} (established certified competence, confirmed)")
    gate["history"].append(dict(eval=int(eval_idx), step=int(step), mode=gate["mode"], bc_coef=float(bc_coef),
                                consec_strict=int(comp["consec_strict"])))
    return gate["mode"]


def new_gate() -> dict:
    return dict(mode="stratified", switch_step=None, switch_eval=None, switch_bc=None, switch_consec=None,
                reason=None, history=[])
_CORPUS_SEEDS = tuple(range(64_000, 64_056)) + _DEMO_SEEDS            # TRAIN + DEMO states (disjoint from VAL eval)
_EVAL_SEEDS = _DEMO_SEEDS + _VAL_SEEDS                                # §9: 4 DEMO + 14 VAL
_DEMO_ACTORS = (DeliveryActor.A1_VPLOW, DeliveryActor.A4_RECOVERY)
_CKPT = Path("experiments/2026_07_20_coin_two_arm_sac_100k/sac_actor_best.pt")


def _episode_quality(trace) -> dict:
    att = _attribution_from_trace(trace)
    ff = att.fingertip_fraction
    clean = (min(att.alpha_L, att.alpha_R) / (ff + 1e-9)) >= _ONE_FINGER_MAX
    return dict(attribution=float(ff), body=float(att.alpha_body), clean=bool(clean),
                bilateral=bool(trace.both_frac > 0.0), strict=bool(policy_strict(trace)), zone=bool(trace.loose))


def _recovery_steps(steps) -> set[int]:
    """Transition indices inside a loss-of-both-contact -> recontact window (the A4 recovery signature)."""
    both = [bool(s.left_contact and s.right_contact) for s in steps]
    out, lost_at = set(), None
    for i in range(1, len(steps)):
        if both[i - 1] and not both[i]:
            lost_at = i
        if lost_at is not None and both[i] and not both[i - 1]:
            out.update(range(lost_at, i + 1))
            lost_at = None
    return out


def _stratum(q: dict, in_recovery: bool) -> int:
    if in_recovery:
        return STRATA["RECOVERY"]
    certified = q["strict"] or (q["zone"] and q["attribution"] >= 0.60 and q["body"] <= 0.20
                                and q["clean"] and q["bilateral"])
    if certified:
        return STRATA["CERTIFIED_BILATERAL"]
    if q["attribution"] >= 0.60 and q["body"] <= 0.20 and q["clean"]:
        return STRATA["HIGH_QUALITY_CONTACT"]
    if q["zone"] and (q["attribution"] < 0.60 or not q["clean"] or not q["bilateral"]):
        return STRATA["CONTRASTIVE_BULLDOZE"]
    return STRATA["GENERAL_PROGRESS"]


def build_corpus(env):
    """A1/A4 demonstrations through the canonical rollout(), each transition tagged with its contact-quality stratum
    (episode-level quality + per-step recovery detection). Returns (obs,act,rew,next,done, tags)."""
    obs_l, act_l, rew_l, nxt_l, done_l, tag_l = [], [], [], [], [], []
    for actor in _DEMO_ACTORS:
        for s in _CORPUS_SEEDS:
            env.reset(seed=int(s))
            tr = rollout(env, lambda inner, t, _o, _a=actor: actor_action(inner, t, _a), max_steps=60)
            q = _episode_quality(tr)
            rec = _recovery_steps(tr.steps)
            for i, st in enumerate(tr.steps):
                if st.obs is None:
                    continue
                nxt = tr.steps[i + 1].obs if i + 1 < len(tr.steps) else tr.final_obs
                obs_l.append(st.obs)
                act_l.append(np.asarray(st.action, np.float32))
                rew_l.append(st.reward)
                nxt_l.append(nxt if nxt is not None else st.obs)
                done_l.append(st.terminated)
                tag_l.append(_stratum(q, i in rec))
    return (np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32), np.asarray(rew_l, np.float32),
            np.asarray(nxt_l, np.float32), np.asarray(done_l, bool), np.asarray(tag_l, np.int16))


def corpus_stratum_counts(tags) -> dict:
    inv = {v: k for k, v in STRATA.items()}
    u, c = np.unique(tags, return_counts=True)
    return {inv.get(int(t), f"tag{t}"): int(n) for t, n in zip(u, c)}


def make_actor(obs_dim, act_dim):
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=1.0)
    actor.load_state_dict(torch.load(_CKPT, map_location="cpu"))       # CONTINUE from the certified checkpoint
    return actor, critics


def _eval_state_64102(eval_env, actor) -> bool:
    eval_env.reset(seed=64_102)
    tr = rollout(eval_env, lambda inner, t, obs: actor.action_mean(
        torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32)).detach().numpy()[0], max_steps=60)
    return bool(policy_strict(tr))


def run_arm(sampler: str, steps: int, out: Path, seed: int = 0) -> dict:
    certify_or_abort()
    env = direct_env(train_seed_pool=tuple(range(64_000, 64_056)))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    corpus = build_corpus(env)
    counts = corpus_stratum_counts(corpus[5])
    print(f"[{sampler}] corpus {len(corpus[0])} transitions | strata {counts}", flush=True)

    actor, critics = make_actor(obs_dim, act_dim)
    cfg = SACConfig.stable(total_steps=steps, seed=seed, bc_coef=1.0, log_every=1000, eval_every=2500)
    comp = {"progress_ok": False, "first_strict": False, "consec_strict": 0}

    def bc_coef_fn(_s):
        return 0.05 if comp["consec_strict"] >= 3 else 0.1 if comp["first_strict"] else 0.3 if comp["progress_ok"] else 1.0

    def demo_frac_fn(step):                                            # §5 explicit ratio (STRATIFIED only): 50% then 25%
        return 0.50 if step < 25_000 else 0.25

    eval_env = direct_env()
    best = {"score": -1.0, "metrics": None}
    hist = []
    gate = new_gate()

    def eval_fn(_e, ac):
        m = evaluate(eval_env, ac, _EVAL_SEEDS)                        # canonical-rollout eval on 4 DEMO + 14 VAL
        m["s64102_strict"] = _eval_state_64102(eval_env, ac)
        m["bc_coef"] = bc_coef_fn(0)
        if m["mean_progress"] >= 0.02:
            comp["progress_ok"] = True
        comp["consec_strict"] = comp["consec_strict"] + 1 if m["strict_count"] >= 1 else 0
        if m["strict_count"] >= 1:
            comp["first_strict"] = True
        hist.append(m)
        if sampler == "gated":                                         # online competence gate (own-run signal only)
            prev = gate["mode"]
            mode = gate_step(gate, comp, eval_idx=len(hist), step=len(hist) * 2500, bc_coef=m["bc_coef"])
            m["sampler_mode"] = mode
            if prev != mode:
                print(f"  [gated] SWITCH {prev}->{mode} @ eval#{len(hist)} step {len(hist)*2500} "
                      f"bc_coef={m['bc_coef']} consec_strict={comp['consec_strict']} ({gate['reason']})", flush=True)
        score = m["strict_count"] * 1e3 + m["zone_rate"] * 1e1 + m["mean_progress"]
        if score > best["score"]:
            best.update(score=score, metrics=m, step=len(hist) * 2500)
            torch.save(ac.state_dict(), out / "actor_best.pt")
        print(f"  [{sampler} eval#{len(hist)}] strict={m['strict_count']} zone={m['zone_rate']:.2f} "
              f"P(attr|zone)~{m.get('lc',0):.2f} 64102={'Y' if m['s64102_strict'] else 'n'} "
              f"prog={m['mean_progress']:.4f} bc={m['bc_coef']} mode={m.get('sampler_mode','-')}", flush=True)
        return float(m["strict_count"] + m["zone_rate"])

    kw = dict(eval_fn=eval_fn, offline_data=(corpus[0], corpus[1]),
              init_transitions=corpus[:5], bc_coef_fn=bc_coef_fn)
    if sampler in ("stratified", "gated"):
        kw.update(init_transition_tags=corpus[5], demo_frac_fn=demo_frac_fn, strata_weights=STRATA_WEIGHTS)
    if sampler == "gated":                                             # start STRATIFIED; flip to UNIFORM on the gate
        kw["sampler_gate_fn"] = lambda _step: gate["mode"] == "stratified"
    out.mkdir(parents=True, exist_ok=True)
    curve = train_sac(actor, critics, env, cfg, **kw)
    torch.save(actor.state_dict(), out / "actor_final.pt")
    result = dict(sampler=sampler, steps=steps, corpus_size=int(len(corpus[0])), corpus_strata=counts,
                  curve=curve, best_step=best.get("step"), best_metrics=best["metrics"], eval_history=hist,
                  gate=dict(switched=gate["switch_step"] is not None, switch_step=gate["switch_step"],
                            switch_eval=gate["switch_eval"], switch_bc=gate["switch_bc"], reason=gate["reason"],
                            mode_history=[h["mode"] for h in gate["history"]]) if sampler == "gated" else None)
    (out / "run.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[{sampler}] done | best strict={best['metrics']['strict_count'] if best['metrics'] else 0} "
          f"@ step {best.get('step')}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["control", "stratified", "gated"], required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_arm(a.sampler, a.steps, Path(a.out), a.seed)


if __name__ == "__main__":
    main()
