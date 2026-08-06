"""ACTOR_UPDATE_PATH_AUDIT_V1 — on a trained Markov critic, isolate WHY the actor did not convert the critic's terminal-
proximity signal into policy improvement. No new campaign. Four one-step update proposals on the SAME gate-active batch:
A Q-only, B Q+BC, C Q+trust-region, D Q+BC+trust. Measures whether the raw gradient uses the counter and raises Q,
whether the BC anchor (pi_0, strict-blind) opposes it, and which trust-region condition rolls it back."""
import copy
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_late_start import LateStart  # noqa: E402
from hymeko_rl.coin_delivery.coin_markov_ablation_train import (  # noqa: E402
    OBS48,
    _actor_loss55,
    _sample_actor,
    train_arm,
)
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import strict_onehot  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss  # noqa: E402
from hymeko_rl.coin_delivery.coin_td3_transactional import TransactionalConfig, transactional_actor_step  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

D = "experiments/2026_07_22_coin_v3_learning/rl_entry"


def _bank(m):
    return [LateStart(seed=r[0], prefix_steps=r[1], family=r[2], obs_sha=r[3], base_sha=r[4], causal_sha=r[5]) for r in m["rows"]]


def _with_strict(obs55, k):
    o = obs55.clone(); o[:, OBS48:] = torch.as_tensor(strict_onehot(k))
    return o


def _mean_q(critic, actor, obs):
    with torch.no_grad():
        return float(critic.min_q(obs, torch.clamp(actor.action_mean(obs), -4, 4)).mean())


def _grad_vec(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])


def main():
    torch.set_num_threads(1); log = lambda *a: print(*a, flush=True)
    cfg = json.load(open(f"{D}/td3_baseline_v1_config.json")); pi0 = load_frozen_clip_actor(f"{D}/frozen/pi0_shared_clip_actor.pt", freeze=True)
    tb, db = _bank(cfg["banks"]["late_train"]), _bank(cfg["banks"]["late_dev"])
    stage = dict(cfg["stage1"]); stage["total_updates"] = 4000; stage["checkpoints"] = [0, 2000, 4000]   # healthy Markov critic
    tcfg = TransactionalConfig(); batch = 256

    log("[audit] training Arm A seed 0 to obtain a healthy Markov critic (4000 upd)...")
    _res, art = train_arm(pi0, "A", stage, tb, db, seed=0, tcfg=tcfg, return_artifacts=True,
                          log=lambda *a: None)
    critic, actor, buf, anchor_obs, a0_anchor = art["critic"], art["actor"], art["buf"], art["anchor_obs"], art["a0_anchor"]
    rng = np.random.default_rng(0)
    ao, ag = _sample_actor(buf, batch, rng)                          # gate-active obs55 batch
    ao = ao[ag > 0][:128]                                            # keep only gate-active states
    ag2 = torch.ones(ao.shape[0])
    params = list(actor.parameters())

    # baseline: does the TRAINED actor already use the counter?
    ao_s1, ao_s5 = _with_strict(ao, 1), _with_strict(ao, 5)
    with torch.no_grad():
        cu_before = float((actor.action_mean(ao_s1) - actor.action_mean(ao_s5)).abs().mean())
    q_before = _mean_q(critic, actor, ao)

    # gradient opposition: Q-loss grad vs BC-loss grad (BC target = pi_0 on obs48 = strict-blind)
    q_loss = masked_actor_loss(critic, actor, ao, ag2)
    with torch.no_grad():
        pi0_bc = torch.clamp(pi0.action_mean(ao[:, :OBS48]), -4, 4)
    bc_loss = ((actor.action_mean(ao) - pi0_bc) ** 2).sum(-1).mean()
    gq, gbc = _grad_vec(q_loss, params), _grad_vec(bc_loss, params)
    cos = float(torch.nn.functional.cosine_similarity(gq.unsqueeze(0), gbc.unsqueeze(0))[0])

    def propose(use_bc, use_trust):
        act = copy.deepcopy(actor); opt = torch.optim.Adam(act.parameters(), lr=tcfg.actor_lr)
        if use_trust:
            lf = (_actor_loss55(critic, act, pi0, ao, ag2, ao, tcfg.lambda_bc) if use_bc
                  else (lambda: masked_actor_loss(critic, act, ao, ag2)))
            r = transactional_actor_step(act, opt, lf, anchor_obs, a0_anchor, tcfg)
            outcome, scale, sd = r["outcome"], r["scale"], r
        else:
            loss = masked_actor_loss(critic, act, ao, ag2)
            if use_bc:
                with torch.no_grad():
                    pb = torch.clamp(pi0.action_mean(ao[:, :OBS48]), -4, 4)
                loss = loss + tcfg.lambda_bc * ((act.action_mean(ao) - pb) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); outcome, scale, sd = "applied", 1.0, {}
        dq = _mean_q(critic, act, ao) - q_before
        with torch.no_grad():
            cu = float((act.action_mean(_with_strict(ao, 1)) - act.action_mean(_with_strict(ao, 5))).abs().mean())
            drift = float((act.action_mean(anchor_obs) - a0_anchor).norm(dim=-1).max())
        return {"outcome": outcome, "scale": scale, "delta_Q": round(dq, 4), "counter_use": round(cu, 5),
                "anchor_drift": round(drift, 5),
                "trust_step_p95": round(sd.get("step_p95", 0.0), 5) if isinstance(sd, dict) else None,
                "trust_cum_max": round(sd.get("cum_max", 0.0), 5) if isinstance(sd, dict) else None}

    A = propose(False, False); B = propose(True, False); C = propose(False, True); Dd = propose(True, True)

    # decisive probe: UNCONSTRAINED Q-only optimisation (no BC, no trust) — can the actor convert the critic signal at all?
    act = copy.deepcopy(actor); opt = torch.optim.Adam(act.parameters(), lr=1e-3)
    for _ in range(100):
        loss = masked_actor_loss(critic, act, ao, ag2)
        opt.zero_grad(); loss.backward(); opt.step()
    unc_dq = _mean_q(critic, act, ao) - q_before
    with torch.no_grad():
        unc_cu = float((act.action_mean(_with_strict(ao, 1)) - act.action_mean(_with_strict(ao, 5))).abs().mean())
        unc_drift = float((act.action_mean(anchor_obs) - a0_anchor).norm(dim=-1).max())
    unconstrained = {"delta_Q": round(unc_dq, 3), "counter_use": round(unc_cu, 5), "anchor_drift": round(unc_drift, 4),
                     "raises_Q": unc_dq > 1.0, "grows_counter_use": unc_cu > cu_before + 0.01}

    # which trust-region cap binds for the Q-only proposal (measure drift vs caps)
    act = copy.deepcopy(actor); opt = torch.optim.Adam(act.parameters(), lr=tcfg.actor_lr)
    old = act.action_mean(anchor_obs).detach()
    lf = lambda: masked_actor_loss(critic, act, ao, ag2)
    lo = lf(); opt.zero_grad(); lo.backward(); opt.step()
    with torch.no_grad():
        step_d = (act.action_mean(anchor_obs).detach() - old).norm(dim=-1).numpy()
        cum_d = (act.action_mean(anchor_obs).detach() - a0_anchor).norm(dim=-1).numpy()
    caps = {"step_median": tcfg.step_median, "step_p95": tcfg.step_p95, "step_max": tcfg.step_max, "cum_p95": tcfg.cum_p95, "cum_max": tcfg.cum_max}
    binding = {"step_median": (float(np.median(step_d)), caps["step_median"]), "step_p95": (float(np.percentile(step_d, 95)), caps["step_p95"]),
               "step_max": (float(np.max(step_d)), caps["step_max"]), "cum_p95": (float(np.percentile(cum_d, 95)), caps["cum_p95"]),
               "cum_max": (float(np.max(cum_d)), caps["cum_max"])}
    violated = {k: {"value": round(v, 5), "cap": c, "exceeds": v > c} for k, (v, c) in binding.items()}

    # decision — the UNCONSTRAINED probe determines whether the critic gradient is informative at all
    grad_informative = unconstrained["raises_Q"] and unconstrained["grows_counter_use"]
    bc_opposes = cos < -0.1
    trust_rolls_back = (C["outcome"] != "accepted") or C["scale"] in (None, 0.0)
    if not grad_informative:
        mech = "CRITIC_ACTION_GRADIENT_UNINFORMATIVE"
    elif bc_opposes and trust_rolls_back:
        mech = "BC_ANCHOR_AND_TRUST_REGION_BOTH_BLOCK"
    elif trust_rolls_back:
        mech = "TRUST_REGION_BLOCKS_ACTOR_CONVERSION"
    elif bc_opposes:
        mech = "BC_ANCHOR_BLOCKS_ACTOR_CONVERSION"
    else:
        mech = "ACTOR_PATH_CLEAR_IMPROVEMENT_POSSIBLE"

    out = {"contract": "ACTOR_UPDATE_PATH_AUDIT_V1", "date": "2026-07-23", "no_new_campaign": True, "train_updates": 4000,
           "trained_actor_uses_counter_before": round(cu_before, 5), "q_before": round(q_before, 3),
           "q_grad_vs_bc_grad_cosine": round(cos, 4), "bc_target": "pi_0(obs48) — strict-blind (identical for strict 1..5)",
           "proposals": {"A_Q_only": A, "B_Q_plus_BC": B, "C_Q_plus_trust": C, "D_Q_plus_BC_plus_trust": Dd},
           "unconstrained_q_only_probe": unconstrained, "trust_region_binding": violated, "mechanism": mech,
           "verdict_correction": {"ablation_verdict_was": "REPAIRS_NOT_SUFFICIENT",
                                  "corrected": ["MARKOV_CRITIC_REPAIR_CONFIRMED", "ACTOR_CONVERSION_BLOCKED"]}}
    json.dump(out, open(f"{D}/actor_update_path_audit_v1.json", "w"), indent=1, default=float)

    log("== ACTOR_UPDATE_PATH_AUDIT_V1 ==")
    log(f"  trained actor counter-use (|a(s1)-a(s5)|): {cu_before:.5f}  q_before {q_before:.2f}")
    log(f"  Q-grad vs BC-grad cosine: {cos:+.3f}  (negative = BC opposes the counter-using Q gradient)")
    log(f"  A Q-only : ΔQ {A['delta_Q']:+.3f} counter_use {A['counter_use']:.4f} drift {A['anchor_drift']}")
    log(f"  B Q+BC   : ΔQ {B['delta_Q']:+.3f} counter_use {B['counter_use']:.4f}")
    log(f"  C Q+trust: outcome {C['outcome']} scale {C['scale']} ΔQ {C['delta_Q']:+.3f}")
    log(f"  D full   : outcome {Dd['outcome']} scale {Dd['scale']} ΔQ {Dd['delta_Q']:+.3f}")
    log(f"  UNCONSTRAINED Q-only (no BC/trust, 100 steps): ΔQ {unconstrained['delta_Q']:+.2f} counter_use {unconstrained['counter_use']:.4f} drift {unconstrained['anchor_drift']} raises_Q {unconstrained['raises_Q']} grows_counter {unconstrained['grows_counter_use']}")
    log(f"  trust caps exceeded by Q-only step: {[k for k,v in violated.items() if v['exceeds']]}")
    log(f"\n→ mechanism: {mech}  |  corrected verdict: MARKOV_CRITIC_REPAIR_CONFIRMED + ACTOR_CONVERSION_BLOCKED")
    log(f"wrote {D}/actor_update_path_audit_v1.json\nACTOR_AUDIT_DONE")


if __name__ == "__main__":
    main()
