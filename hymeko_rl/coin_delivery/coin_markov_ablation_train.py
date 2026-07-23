"""STRICT_COUNTER_MARKOV_REPAIR_ABLATION_V1 — obs55 transactional-TD3 trainer (Arm A / Arm B). Same transactional
update, banks, families, horizon, n-step, target smoothing, exploration, budget as the historical Stage-1b; the ONLY
differences are the exact strict-counter one-hot appended to the actor/critic/replay/target state and (Arm B) the
terminal-bonus realignment. Trains under CONTINUATION_STRICT (inherit handoff_strict); evaluates both semantics.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_rl_env import HELD_DWELL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_strict_markov_ablation import (
    STRICT_K,
    base_spec_no_terminal,
    step_ablation,
    strict_onehot,
)
from hymeko_rl.coin_delivery.coin_td3_contracts import (
    CoherentNoise,
    LateReplayBuffer,
    LateTwinCritic,
    td3_target_action,
)
from hymeko_rl.coin_delivery.coin_td3_transactional import (
    TransactionalConfig,
    critic_authorization,
    transactional_actor_step,
)
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss
from hymeko_rl.coin_delivery.rl_clip_actor import ClipDeterministicActor, make_backbone

ACTION_SCALE, GAMMA = 4.0, 0.99
OBS48, AUG = 48, STRICT_K + 1                                    # 48 + 7 = 55
ENTRY_TOL, CENTER_TOL, SETTLE_VEL = 0.05, 0.02, 0.06


# ── obs55 actor: exact pi_0 copy, seven strict-input columns zero-initialised (update-0 ≡ pi_0), trainable ──
def make_late_actor55_from_pi0(pi0, *, trainable=True):
    feat = pi0.head.in_features
    late = ClipDeterministicActor(make_backbone(OBS48 + AUG, feat), feat, pi0.action_dim, pi0.action_scale)
    with torch.no_grad():
        late.backbone[0].weight.zero_()
        late.backbone[0].weight[:, :OBS48].copy_(pi0.backbone[0].weight)   # 48 physical columns exact; 7 strict columns 0
        late.backbone[0].bias.copy_(pi0.backbone[0].bias)
        late.backbone[2].load_state_dict(pi0.backbone[2].state_dict())
        late.head.load_state_dict(pi0.head.state_dict())
    for p in late.parameters():
        p.requires_grad_(bool(trainable))
    (late.train() if trainable else late.eval())
    return late


def _aug(obs48, strict):
    return np.concatenate([np.asarray(obs48, np.float32), strict_onehot(int(strict))]).astype(np.float32)


def _det(actor, obs):
    with torch.no_grad():
        return torch.clamp(actor.action_mean(torch.as_tensor(np.asarray(obs, np.float32)[None]))[0], -ACTION_SCALE, ACTION_SCALE).numpy()


def verify_update0(pi0, actor55, obs48_bank):
    """actor55(obs48, strict=k) == pi0(obs48) for every dev obs and k∈0..K — the zero-init contract."""
    dmax = dmean = 0.0
    for o in obs48_bank:
        a0 = _det(pi0, o)
        for k in range(STRICT_K + 1):
            d = np.abs(_det(actor55, _aug(o, k)) - a0)
            dmax = max(dmax, float(d.max())); dmean += float(d.mean())
    return {"max_action_diff": dmax, "mean_action_diff": round(dmean / (len(obs48_bank) * (STRICT_K + 1)), 12)}


# ── collect one late episode under the ablation env, storing obs55 transitions ──
def collect_ablation(pi0, actor55, ls, arm, cnoise, *, horizon, explore, reset_strict=False):
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    if reset_strict:
        rl._strict = 0
    base_spec = base_spec_no_terminal(rl) if arm == "B" else None
    bonus_state = [False]
    if cnoise is not None:
        cnoise.reset()
    trs = []
    for k in range(horizon):
        gate_on = gate.gate == 1.0
        o48 = rl.obs().astype(np.float32); s = int(rl._strict); o55 = _aug(o48, s)
        a = _det(actor55, o55) if gate_on else _det(pi0, o48)
        if gate_on and explore and cnoise is not None:
            a = cnoise.perturb(a)
        r, term, trunc = step_ablation(rl, a, arm, base_spec=base_spec, bonus_state=bonus_state)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        o48n = rl.obs().astype(np.float32); sn = int(rl._strict); o55n = _aug(o48n, sn)
        cut = (k == horizon - 1)
        trs.append({"obs": o55, "action": a.astype(np.float32), "reward": float(r), "obs_next": o55n,
                    "terminated": bool(term), "truncated": bool(trunc or (cut and not term)),
                    "gate_on": bool(gate_on), "gate_on_next": bool(gate.gate == 1.0),
                    "strict": s, "strict_next": sn, "family": ls.family})
        if term or trunc:
            break
    return trs


def _sample_nstep(buf, batch, n, gamma, rng):
    """n-step batch over obs55 transitions. Returns O,A,R,Bt,M,GP,BG (bootstrap obs55 + gate flag)."""
    trajs = [t for t in buf.trajectories if t]
    O, A, R, Bt, M, GP, BG = [], [], [], [], [], [], []
    for _ in range(batch):
        tr = trajs[rng.integers(len(trajs))]; t = int(rng.integers(len(tr)))
        G, disc, mask = 0.0, 1.0, 1; boot, bg = tr[t]["obs_next"], tr[t]["gate_on_next"]
        for i in range(n):
            idx = t + i
            if idx >= len(tr):
                break
            x = tr[idx]; G += disc * x["reward"]; boot, bg = x["obs_next"], x["gate_on_next"]; disc *= gamma
            if x["terminated"]:
                mask = 0; break
            if x["truncated"]:
                mask = 1; break
        O.append(tr[t]["obs"]); A.append(tr[t]["action"]); R.append(G); Bt.append(boot); M.append(mask); GP.append(disc); BG.append(bg)
    f = lambda z: torch.as_tensor(np.asarray(z, np.float32))
    return f(O), f(A), f(R), f(Bt), f(M), f(GP), torch.as_tensor(np.asarray(BG, bool))


def _target_action55(pi0, actor_t55, boot55, bg, *, std, clip, gen):
    """TD3 target action: smoothed pi_late_target55 on gate-on bootstraps, exact pi_0 (physical 48) on gate-off."""
    a = td3_target_action(actor_t55, boot55, smoothing_std=std, smoothing_clip=clip, generator=gen)
    if (~bg).any():
        with torch.no_grad():
            a0 = torch.clamp(pi0.action_mean(boot55[:, :OBS48]), -ACTION_SCALE, ACTION_SCALE)
        a = torch.where(bg.unsqueeze(-1), a, a0)
    return a


def _sample_actor(buf, batch, rng):
    trajs = [t for t in buf.trajectories if t]; O, GT = [], []
    for _ in range(batch):
        tr = trajs[rng.integers(len(trajs))]; x = tr[int(rng.integers(len(tr)))]
        O.append(x["obs"]); GT.append(1.0 if x["gate_on"] else 0.0)
    return torch.as_tensor(np.asarray(O, np.float32)), torch.as_tensor(np.asarray(GT, np.float32))


def _actor_loss55(critic, actor55, pi0, actor_obs, actor_gate, bc_obs, lambda_bc):
    def fn():
        with torch.no_grad():
            pi0_bc = torch.clamp(pi0.action_mean(bc_obs[:, :OBS48]), -ACTION_SCALE, ACTION_SCALE)
        bc = ((actor55(bc_obs) - pi0_bc) ** 2).sum(-1).mean()
        return masked_actor_loss(critic, actor55, actor_obs, actor_gate) + lambda_bc * bc
    return fn


# ── dual-semantics evaluation ──
def _ladder_row(rl, gate, pi0, actor55, arm, horizon):
    base_spec = base_spec_no_terminal(rl) if arm == "B" else None; bonus_state = [False]
    md = int(rl._strict); ret = 0.0; disc = 1.0; entered = rl._dtz() <= ENTRY_TOL
    exit_ct = 0; was_in = rl._dtz() <= ENTRY_TOL; entry_vel = None; touched = rl._touched
    for _k in range(horizon):
        gate_on = gate.gate == 1.0; o48 = rl.obs(); s = int(rl._strict)
        a = _det(actor55, _aug(o48, s)) if gate_on else _det(pi0, o48)
        r, term, trunc = step_ablation(rl, a, arm, base_spec=base_spec, bonus_state=bonus_state)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        ret += disc * r; disc *= GAMMA; md = max(md, int(rl._strict)); touched = touched or rl._touched
        dtz = rl._dtz()
        if dtz <= ENTRY_TOL and not entered:
            entry_vel = rl._speed()
        entered = entered or dtz <= ENTRY_TOL
        if was_in and dtz > ENTRY_TOL:
            exit_ct += 1
        was_in = dtz <= ENTRY_TOL
        if term or trunc:
            break
    return {"max_dwell": md, "k6": int(md >= HELD_DWELL and touched), "k1": int(md >= 1), "k3": int(md >= 3),
            "k5": int(md >= 5), "entered": int(entered), "exit_ct": exit_ct, "ret": round(ret, 3),
            "entry_vel": round(float(entry_vel), 4) if entry_vel is not None else 0.0,
            "full_contain": int(md >= 1)}


def eval_ablation(pi0, actor55, dev_bank, arm, *, horizon, families, reset_strict):
    rows = [_ladder_row(*reconstruct_handoff(pi0, ls, horizon=360)[:2], pi0, actor55, arm, horizon)
            for ls in dev_bank if ls.family in families]
    # reset_strict handled by re-reconstructing with strict cleared
    if reset_strict:
        rows = []
        for ls in dev_bank:
            if ls.family not in families:
                continue
            rl, gate, _h, _r = reconstruct_handoff(pi0, ls, horizon=360); rl._strict = 0
            rows.append(_ladder_row(rl, gate, pi0, actor55, arm, horizon))
    agg = lambda k: round(float(np.mean([r[k] for r in rows])), 4) if rows else 0.0
    return {"n": len(rows), "k6_rate": agg("k6"), "k1": agg("k1"), "k3": agg("k3"), "k5": agg("k5"),
            "mean_max_dwell": agg("max_dwell"), "entry_vel": agg("entry_vel"), "exit_ct": agg("exit_ct"),
            "canonical_return": agg("ret"), "full_contain": agg("full_contain")}


def strict_conditioned_q(critic, actor55, obs48):
    """Q(aug(obs, strict), pi_late(aug)) for strict 0..K from ONE physical state — does the critic represent terminal
    proximity (monotone rising toward K6)?"""
    qs = []
    for k in range(STRICT_K + 1):
        s = torch.as_tensor(_aug(obs48, k)[None])
        with torch.no_grad():
            a = torch.clamp(actor55.action_mean(s), -ACTION_SCALE, ACTION_SCALE)
            q = float(critic.min_q(s, a)[0])
        qs.append(round(q, 3))
    return qs


def train_arm(pi0, arm, cfg_stage, train_bank, dev_bank, *, seed, tcfg=None, log=print):
    tcfg = tcfg or TransactionalConfig()
    torch.manual_seed(seed)
    families = tuple(cfg_stage["families"]); horizon = cfg_stage["horizon"]; n_step = cfg_stage["n_step"]
    warmup = cfg_stage["critic_warmup_steps"]; total = cfg_stage["total_updates"]
    collect_every, eps_per = cfg_stage["collect_every"], cfg_stage["episodes_per_collect"]
    checkpoints = set(cfg_stage["checkpoints"]); tau, batch = 0.005, 256; policy_delay = cfg_stage["policy_delay"]
    exp_init, exp_max = 0.15, 0.30; smoothing_std, smoothing_clip = 0.10, 0.25

    actor = make_late_actor55_from_pi0(pi0, trainable=True)
    actor_t = make_late_actor55_from_pi0(pi0, trainable=False)
    critic = LateTwinCritic(obs_dim=OBS48 + AUG); critic_t = copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=tcfg.actor_lr); c_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    buf = LateReplayBuffer(); rng_c = np.random.default_rng(seed); rng_r = np.random.default_rng(seed + 1)
    gen = torch.Generator().manual_seed(seed)
    starts = [ls for ls in train_bank if ls.family in families]
    # obs55 anchor bank (inherit handoff strict) + a0 (update-0 ≡ pi_0)
    anchor = []
    for ls in starts[:64]:
        rl, _g, _h, rec = reconstruct_handoff(pi0, ls, horizon=360); anchor.append(_aug(rec.obs, rl._strict))
    anchor_obs = torch.as_tensor(np.asarray(anchor, np.float32)); a0_anchor = torch.clamp(actor.action_mean(anchor_obs), -ACTION_SCALE, ACTION_SCALE).detach()
    obs48_bank = [reconstruct_handoff(pi0, ls, horizon=360)[3].obs for ls in starts[:16]]

    def collect(std):
        cn = CoherentNoise(std=std, hold_min=2, hold_max=4, seed=int(rng_c.integers(1 << 30)))
        for j in rng_c.integers(0, len(starts), eps_per):
            trs = collect_ablation(pi0, actor, starts[j], arm, cn, horizon=horizon, explore=True)
            if trs:
                buf.add_trajectory(trs)

    collect(exp_init)
    ckpt, acc, rej, td_hist = {}, 0, 0, []
    for step in range(total + 1):
        if step in checkpoints:
            snap = make_late_actor55_from_pi0(pi0, trainable=False); snap.load_state_dict(actor.state_dict())
            auth = critic_authorization(critic, actor, anchor_obs, tcfg)
            cont = eval_ablation(pi0, snap, dev_bank, arm, horizon=horizon, families=families, reset_strict=False)
            reset = eval_ablation(pi0, snap, dev_bank, arm, horizon=horizon, families=families, reset_strict=True)
            with torch.no_grad():
                qc = critic.min_q(anchor_obs, torch.clamp(snap.action_mean(anchor_obs), -4, 4))
                drift = float((snap.action_mean(anchor_obs) - a0_anchor).norm(dim=-1).max())
            ckpt[step] = {"auth": auth, "CONTINUATION_STRICT": cont, "RESET_AT_HANDOFF_STRICT": reset, "accepted": acc,
                          "rejected": rej, "actor_drift": round(drift, 5), "q_mean": round(float(qc.mean()), 2),
                          "q_std": round(float(qc.std()), 2), "q_max": round(float(qc.max()), 2),
                          "td_p90": round(float(np.percentile(td_hist, 90)) if td_hist else 0.0, 3),
                          "strict_q": strict_conditioned_q(critic, snap, obs48_bank[0])}
            log(f"  [{arm} s{seed} ck{step}] auth={auth['authorized']} acc/rej {acc}/{rej} CONT_K6 {cont['k6_rate']} "
                f"RESET_K6 {reset['k6_rate']} dwell {cont['mean_max_dwell']} drift {ckpt[step]['actor_drift']} "
                f"q {ckpt[step]['q_mean']} strictQ {ckpt[step]['strict_q']}")
        if step == total:
            break
        if step > 0 and step % collect_every == 0:
            collect(min(exp_max, exp_init + (exp_max - exp_init) * step / total))
        O, A, R, Bt, M, GP, BG = _sample_nstep(buf, batch, n_step, GAMMA, rng_r)
        with torch.no_grad():
            ta = _target_action55(pi0, actor_t, Bt, BG, std=smoothing_std, clip=smoothing_clip, gen=gen)
            y = R + GP * M * critic_t.min_q(Bt, ta)
        q1, q2 = critic(O, A); c_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        c_opt.zero_grad(); c_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); c_opt.step()
        td_hist.append(float((q1 - y).abs().mean().detach())); td_hist[:] = td_hist[-500:]
        if step >= warmup and step % policy_delay == 0:
            auth = critic_authorization(critic, actor, anchor_obs, tcfg)
            if auth["authorized"]:
                ao, ag = _sample_actor(buf, batch, rng_r); bo, _bg = _sample_actor(buf, batch, rng_r)
                if float(ag.sum()) > 0:
                    lf = _actor_loss55(critic, actor, pi0, ao, ag, bo, tcfg.lambda_bc)
                    r = transactional_actor_step(actor, a_opt, lf, anchor_obs, a0_anchor, tcfg)
                    if r["outcome"] == "accepted":
                        acc += 1
                        with torch.no_grad():
                            for p, pt in zip(actor.parameters(), actor_t.parameters()):
                                pt.mul_(1 - tau).add_(tau * p)
                    else:
                        rej += 1
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_t.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
        if step % 1000 == 0:
            log(f"    [{arm} s{seed}] step {step}/{total} c_loss {c_loss.item():.2f} buf {buf.n_transitions()} acc/rej {acc}/{rej}")

    u0 = verify_update0(pi0, make_late_actor55_from_pi0(pi0, trainable=False), obs48_bank)
    return {"arm": arm, "seed": seed, "update0": u0, "checkpoints": {str(k): v for k, v in ckpt.items()},
            "accepted": acc, "rejected": rej, "critic_ever_authorized": any(ckpt[k]["auth"]["authorized"] for k in ckpt)}
