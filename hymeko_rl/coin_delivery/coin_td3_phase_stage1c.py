"""DYNAMIC_PHASE_TRANSITION_CONTRACT_V1 + Stage-1c trainer. The late controller is conditioned on the CURRENT
per-transition phase (``phase_t`` from :class:`PhaseDetector`), NOT the static ``LateStart.family``:

    actor_input_t  = obs_t ++ one_hot(phase_t)
    critic_input_t = obs_t ++ one_hot(phase_t) ++ executed_action_t
    bootstrap      = obs_boot ++ one_hot(phase_boot)   (phase_boot = stored phase at the bootstrap state)

Actor masking stays on current ``gate_t``; target routing stays on ``gate_tp1`` and ``phase_tp1``. Everything else
(banks, reward, horizon, n-step, smoothing, exploration, transactional trust-region thresholds) is UNCHANGED from
Stage-1b — this run isolates the effect of DYNAMIC phase conditioning.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_phase_conditioning import (
    PHASES,
    PhaseDetector,
    augment,
    make_phase_actor_from_pi0,
    make_phase_critic,
    phase_onehot,
)
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_td3_contracts import CoherentNoise, LateReplayBuffer
from hymeko_rl.coin_delivery.coin_td3_trainer import GAMMA, _det, masked_actor_loss
from hymeko_rl.coin_delivery.coin_td3_transactional import (
    critic_authorization,
    stage1b_gate,
    transactional_actor_step,
)

ACTION_SCALE = 4.0


# ── §1-§3 dynamic-phase collection ──
def collect_late_episode_phase(pi0, pi_late, ls, cnoise, *, horizon: int, explore: bool):
    """Phase-switched late episode with DYNAMIC phase. Stores ``phase_t``/``phase_tp1`` (current per-transition phase),
    never the static ``ls.family``. gate-on ⇒ ``pi_late(obs ++ onehot(phase_t)) + noise``; gate-off ⇒ ``pi_0(obs)``."""
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    det = PhaseDetector()
    if cnoise is not None:
        cnoise.reset()
    o = rec.obs.astype(np.float32); phase_t = det.phase_of(rl); trs = []
    for k in range(horizon):
        gate_on = gate.gate == 1.0
        if gate_on:
            a = _det(pi_late, augment(o, phase_t))
            if explore and cnoise is not None:
                a = cnoise.perturb(a)
        else:
            a = _det(pi0, o)
        o2, r, term, trunc, _ = rl.step(a)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
        gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        phase_tp1 = det.phase_of(rl); cut = (k == horizon - 1)
        trs.append({"obs": o, "phase_t": phase_t, "phase_tp1": phase_tp1, "action": a.astype(np.float32),
                    "reward": float(r), "obs_next": o2.astype(np.float32), "terminated": bool(term),
                    "truncated": bool(trunc or (cut and not term)), "gate_on": bool(gate_on),
                    "gate_on_next": bool(gate.gate == 1.0), "family": ls.family})
        o = o2; phase_t = phase_tp1
        if term or trunc:
            break
    return trs


# ── phase-aware samplers ──
def _flat(buf):
    return [(ti, t) for ti, traj in enumerate(buf.trajectories) for t in range(len(traj))]


def sample_actor_batch_phase(buf, batch, rng):
    flat = _flat(buf)
    if not flat:
        return None
    pick = rng.integers(0, len(flat), batch)
    aug = np.stack([augment(buf.trajectories[flat[j][0]][flat[j][1]]["obs"],
                            buf.trajectories[flat[j][0]][flat[j][1]]["phase_t"]) for j in pick])
    gate = np.array([float(buf.trajectories[flat[j][0]][flat[j][1]]["gate_on"]) for j in pick], np.float32)
    return torch.tensor(aug), torch.tensor(gate)


def sample_gate_on_phase(buf, batch, rng):
    flat = [(ti, t) for ti, traj in enumerate(buf.trajectories) for t in range(len(traj)) if traj[t]["gate_on"]]
    if not flat:
        return None
    pick = rng.integers(0, len(flat), min(batch, len(flat)))
    obs = np.stack([buf.trajectories[flat[j][0]][flat[j][1]]["obs"] for j in pick])
    aug = np.stack([augment(buf.trajectories[flat[j][0]][flat[j][1]]["obs"],
                            buf.trajectories[flat[j][0]][flat[j][1]]["phase_t"]) for j in pick])
    return torch.tensor(obs), torch.tensor(aug)


def _nstep_phase(traj, t, n, gamma):
    G, disc, mask, steps = 0.0, 1.0, 1, 0
    boot, boot_gate, boot_phase = traj[t]["obs_next"], traj[t]["gate_on_next"], traj[t]["phase_tp1"]
    for i in range(n):
        idx = t + i
        if idx >= len(traj):
            break
        tr = traj[idx]
        G += disc * float(tr["reward"]); boot, boot_gate, boot_phase = tr["obs_next"], tr["gate_on_next"], tr["phase_tp1"]
        disc *= gamma; steps += 1
        if tr["terminated"]:
            mask = 0; break
        if tr["truncated"]:
            mask = 1; break
    return G, boot, mask, disc, boot_gate, boot_phase


def sample_nstep_phase_c(buf, batch, n, gamma, rng):
    flat = _flat(buf)
    if not flat:
        raise ValueError("empty buffer")
    pick = rng.integers(0, len(flat), batch)
    A_in, ACT, R, B48, BPH, M, GP, BG = [], [], [], [], [], [], [], []
    for j in pick:
        ti, t = flat[j]; traj = buf.trajectories[ti]
        G, b, m, gp, bg, bph = _nstep_phase(traj, t, n, gamma)
        A_in.append(augment(traj[t]["obs"], traj[t]["phase_t"])); ACT.append(traj[t]["action"]); R.append(G)
        B48.append(b); BPH.append(phase_onehot(bph)); M.append(m); GP.append(gp); BG.append(float(bg))
    T = torch.tensor
    return (T(np.stack(A_in)), T(np.stack(ACT)), T(np.array(R, np.float32)), T(np.stack(B48)),
            T(np.stack(BPH)), T(np.array(M, np.float32)), T(np.array(GP, np.float32)), T(np.array(BG, np.float32)))


def phase_target_action_c(pi0, pi_late_target, boot48, boot_phase_oh, gate_next, *, std, clip, gen):
    """§8 target: gate_next==1 ⇒ smoothed ``pi_late_target(obs_next ++ onehot(phase_tp1))``; ==0 ⇒ frozen ``pi_0``."""
    with torch.no_grad():
        base_pi0 = torch.clamp(pi0.action_mean(boot48), -ACTION_SCALE, ACTION_SCALE)
        aug = torch.cat([boot48, boot_phase_oh], -1)
        base_late = torch.clamp(pi_late_target.action_mean(aug), -ACTION_SCALE, ACTION_SCALE)
    noise = torch.clamp(torch.randn(base_late.shape, generator=gen) * std, -clip, clip)
    late = torch.clamp(base_late + noise, -ACTION_SCALE, ACTION_SCALE)
    g = gate_next.unsqueeze(-1)
    return g * late + (1.0 - g) * base_pi0


# ── phase-aware anchor + eval ──
def build_anchor_bank_phase(pi0, train_bank, families):
    obs48, aug = [], []
    for ls in train_bank:
        if ls.family not in families:
            continue
        rl, _g, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        det = PhaseDetector(); ph = det.phase_of(rl)
        obs48.append(rec.obs.astype(np.float32)); aug.append(augment(rec.obs, ph))
    return torch.tensor(np.stack(obs48)), torch.tensor(np.stack(aug))


def _rollout_metrics_phase(pi0, pi_late, ls, horizon, *, use_late):
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    det = PhaseDetector(); o = rec.obs.astype(np.float32); phase_t = det.phase_of(rl)
    tot, disc = 0.0, 1.0; max_dwell = rl._strict; entered = rl._dtz() <= CENTER_TOL; exited = False
    contact_steps = n = 0; term = trunc = False
    for _k in range(horizon):
        if use_late and gate.gate == 1.0:
            a = _det(pi_late, augment(o, phase_t))
        else:
            a = _det(pi0, o)
        o2, r, term, trunc, _ = rl.step(a); tot += disc * r; disc *= GAMMA
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        phase_t = det.phase_of(rl); m = rl.inner._planar_metrics
        contact_steps += int(bool(m.left_contact or m.right_contact)); n += 1; max_dwell = max(max_dwell, rl._strict)
        if rl._dtz() <= CENTER_TOL:
            entered = True
        elif entered:
            exited = True
        o = o2
        if term or trunc:
            break
    return {"strict_success": int(max_dwell >= HELD_DWELL), "max_dwell": int(max_dwell), "entered": int(entered),
            "exited": int(exited), "contact_retention": contact_steps / max(n, 1), "ret": float(tot)}


def eval_late_controller_phase(pi0, pi_late, dev_bank, *, horizon, families):
    starts = [ls for ls in dev_bank if ls.family in families]
    late = [_rollout_metrics_phase(pi0, pi_late, ls, horizon, use_late=True) for ls in starts]
    base = [_rollout_metrics_phase(pi0, pi0, ls, horizon, use_late=False) for ls in starts]

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0
    keys = ["strict_success", "max_dwell", "entered", "exited", "contact_retention", "ret"]
    out = {"n": len(starts), "late": {k: round(agg(late, k), 3) for k in keys},
           "pi0": {k: round(agg(base, k), 3) for k in keys}}
    out["delta_vs_pi0"] = {k: round(out["late"][k] - out["pi0"][k], 3) for k in keys}
    return out


def _anchor_l2(pi_late, aug_anchor, a0_anchor):
    with torch.no_grad():
        d = (torch.clamp(pi_late.action_mean(aug_anchor), -4, 4) - a0_anchor).norm(dim=-1).numpy()
    return {"median": round(float(np.median(d)), 5), "p95": round(float(np.percentile(d, 95)), 5), "max": round(float(np.max(d)), 5)}


# ── Stage-1c trainer (phase-conditioned, transactional) ──
def train_stage1c(pi0, cfg_stage, train_bank, dev_bank, *, seeds, tcfg, log=print):
    torch.manual_seed(seeds["torch"])
    families = tuple(cfg_stage["families"]); horizon = cfg_stage["horizon"]; n_step = cfg_stage["n_step"]
    warmup, total = cfg_stage["critic_warmup_steps"], cfg_stage["total_updates"]
    collect_every, eps_per = cfg_stage["collect_every"], cfg_stage["episodes_per_collect"]
    checkpoints = set(cfg_stage["checkpoints"]); tau, batch = 0.005, 256
    exp_init, exp_max, smoothing_std, smoothing_clip = 0.15, 0.30, 0.10, 0.25

    pi_late = make_phase_actor_from_pi0(pi0, trainable=True)
    pi_late_target = make_phase_actor_from_pi0(pi0, trainable=False)
    critic = make_phase_critic(); critic_target = copy.deepcopy(critic)
    a_opt = torch.optim.Adam(pi_late.parameters(), lr=tcfg.actor_lr); c_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    buf = LateReplayBuffer(); rng_c = np.random.default_rng(seeds["numpy_collect"]); rng_r = np.random.default_rng(seeds["numpy_replay"])
    gen = torch.Generator().manual_seed(seeds["torch"])
    train_starts = [ls for ls in train_bank if ls.family in families]
    obs48_anchor, aug_anchor = build_anchor_bank_phase(pi0, train_bank, families)
    a0_anchor = torch.clamp(pi0.action_mean(obs48_anchor), -4, 4).detach()
    probe = torch.randn(64, 48, generator=torch.Generator().manual_seed(123)); a0_probe = torch.clamp(pi0.action_mean(probe), -4, 4).detach().numpy()

    def collect(std):
        cn = CoherentNoise(std=std, hold_min=2, hold_max=4, seed=int(rng_c.integers(1 << 30)))
        for j in rng_c.integers(0, len(train_starts), eps_per):
            trs = collect_late_episode_phase(pi0, pi_late, train_starts[int(j)], cn, horizon=horizon, explore=True)
            if trs:
                buf.add_trajectory(trs)

    collect(exp_init)
    ckpt, acc, rej, step_maxes = {}, 0, 0, []
    for step in range(total + 1):
        if step in checkpoints:
            snap = make_phase_actor_from_pi0(pi0, trainable=False); snap.load_state_dict(pi_late.state_dict())
            dev = eval_late_controller_phase(pi0, snap, dev_bank, horizon=horizon, families=families)
            auth = critic_authorization(critic, pi_late, aug_anchor, tcfg); dev["auth"] = auth
            dev["anchor_L2_cumulative"] = _anchor_l2(snap, aug_anchor, a0_anchor)
            dev["anchor_cum_max"] = dev["anchor_L2_cumulative"]["max"]
            dev["anchor_L2_perstep_over_accepted"] = {
                "median": round(float(np.median(step_maxes)), 5) if step_maxes else 0.0,
                "p95": round(float(np.percentile(step_maxes, 95)), 5) if step_maxes else 0.0,
                "max": round(float(np.max(step_maxes)), 5) if step_maxes else 0.0}
            with torch.no_grad():
                pl = torch.clamp(snap.action_mean(torch.cat([probe, torch.zeros(64, len(PHASES))], -1)), -4, 4).numpy()
            dev["probe_Linf_diagnostic"] = round(float(np.abs(pl - a0_probe).max()), 5)
            dev["accepted"], dev["rejected"] = acc, rej; ckpt[step] = dev
            dl = dev["delta_vs_pi0"]
            log(f"  [ckpt {step}] auth={auth['authorized']} acc/rej {acc}/{rej} Δstrict {dl['strict_success']:+.3f} "
                f"Δdwell {dl['max_dwell']:+.2f} Δexit {dl['exited']:+.3f} Δcontact {dl['contact_retention']:+.3f} | "
                f"anchorL2 cum(med/p95/max) {dev['anchor_L2_cumulative']['median']}/{dev['anchor_L2_cumulative']['p95']}/"
                f"{dev['anchor_L2_cumulative']['max']} probeLinf {dev['probe_Linf_diagnostic']}")
        if step == total:
            break
        if step > 0 and step % collect_every == 0:
            collect(min(exp_max, exp_init + (exp_max - exp_init) * step / total))
        A_in, ACT, R, B48, BPH, M, GP, BG = sample_nstep_phase_c(buf, batch, n_step, GAMMA, rng_r)
        with torch.no_grad():
            ta = phase_target_action_c(pi0, pi_late_target, B48, BPH, BG, std=smoothing_std, clip=smoothing_clip, gen=gen)
            aug_boot = torch.cat([B48, BPH], -1)
            y = R + GP * M * critic_target.min_q(aug_boot, ta)
        q1, q2 = critic(A_in, ACT)
        c_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        c_opt.zero_grad(); c_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); c_opt.step()
        if step >= warmup and step % tcfg.policy_delay == 0:
            auth = critic_authorization(critic, pi_late, aug_anchor, tcfg)
            if auth["authorized"]:
                sample = sample_actor_batch_phase(buf, batch, rng_r); bc = sample_gate_on_phase(buf, batch, rng_r)
                if sample is not None and bc is not None and float(sample[1].sum()) > 0:
                    bc48, bcaug = bc

                    def lf(aug_o=sample[0], gate=sample[1], bc48=bc48, bcaug=bcaug):
                        with torch.no_grad():
                            pi0_bc = torch.clamp(pi0.action_mean(bc48), -4, 4)
                        bcl = ((pi_late(bcaug) - pi0_bc) ** 2).sum(-1).mean()
                        return masked_actor_loss(critic, pi_late, aug_o, gate) + tcfg.lambda_bc * bcl
                    r = transactional_actor_step(pi_late, a_opt, lf, aug_anchor, a0_anchor, tcfg)
                    if r["outcome"] == "accepted":
                        acc += 1; step_maxes.append(r["step_max"])
                        with torch.no_grad():
                            for p, pt in zip(pi_late.parameters(), pi_late_target.parameters()):
                                pt.mul_(1 - tau).add_(tau * p)
                    else:
                        rej += 1
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
        if step % 1000 == 0:
            log(f"    step {step}/{total} c_loss {c_loss.item():.3f} buf {buf.n_transitions()} acc/rej {acc}/{rej}")

    ks = sorted(ckpt); ever_auth = any(ckpt[k]["auth"]["authorized"] for k in ks)
    passed = ever_auth and any(stage1b_gate(ckpt[ks[i]], ckpt[ks[i + 1]], tcfg) for i in range(len(ks) - 1))
    return {"checkpoints": {str(k): v for k, v in ckpt.items()}, "stage1c_pass": passed,
            "critic_ever_authorized": ever_auth, "accepted": acc, "rejected": rej,
            "phase_conditioned": True, "dynamic_phase": True}
