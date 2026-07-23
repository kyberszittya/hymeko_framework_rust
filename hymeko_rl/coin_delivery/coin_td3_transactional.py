"""TRANSACTIONAL_TD3_ACTOR_UPDATE_V1 (Stage-1b) — a basin-preserving TD3 actor update.

Every actor update is a TRANSACTION: snapshot actor + optimizer, compute one proposed TD3 step (with a secondary BC
anchor), measure the executed-action change on a FROZEN gate-active anchor bank, and accept only inside a trust region
on per-step and cumulative drift. On violation, restore and retry with backtracking scales on the parameter delta; if
none passes, REJECT (restore, keep training the critic). A critic-authorization gate must pass on a frozen local-action
diagnostic panel before any actor update is enabled — otherwise the actor is never touched
(TD3_CRITIC_NOT_AUTHORIZED_FOR_ACTOR_UPDATE). This directly targets the V1 failure (unanchored actor → full-range drift).

Everything else (banks, reward, phase switch, horizon, n-step, target smoothing, eval protocol) is UNCHANGED.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_phase_switched_late import make_late_actor_from_pi0
from hymeko_rl.coin_delivery.coin_td3_contracts import CoherentNoise, LateReplayBuffer, LateTwinCritic
from hymeko_rl.coin_delivery.coin_td3_trainer import (
    GAMMA,
    _sample_gate_on,
    _sample_nstep_phase,
    collect_late_episode,
    eval_late_controller,
    masked_actor_loss,
    phase_target_action,
    sample_actor_batch,
)

ACTION_SCALE = 4.0


def stage1b_gate(dev_a: dict, dev_b: dict, cfg: "TransactionalConfig") -> bool:
    """Stage-1b pass rule (§12): two CONSECUTIVE checkpoints each BEAT OR MATCH pi_0 (Δstrict ≥ 0 and Δmax_dwell ≥
    −0.05), without contact degradation (Δcontact ≥ −0.05) or target-exit rise (Δexit ≤ +0.05), with the critic
    authorized and the cumulative trust region intact (anchor cum_max ≤ bound)."""
    def ck(d):
        dl = d["delta_vs_pi0"]
        return (d["auth"]["authorized"] and dl["strict_success"] >= 0 and dl["max_dwell"] >= -0.05
                and dl["contact_retention"] >= -0.05 and dl["exited"] <= 0.05
                and d.get("anchor_cum_max", 0.0) <= cfg.cum_max)
    return ck(dev_a) and ck(dev_b)


@dataclass(frozen=True)
class TransactionalConfig:
    actor_lr: float = 1e-5
    policy_delay: int = 4
    lambda_bc: float = 0.1
    # trust region on ||delta_a|| over the frozen anchor bank
    step_median: float = 0.0025
    step_p95: float = 0.005
    step_max: float = 0.010
    cum_p95: float = 0.030
    cum_max: float = 0.060
    backtrack_scales: tuple = (0.5, 0.25, 0.125, 0.0625)
    # critic-authorization gate thresholds (frozen)
    auth_boundary_pref_max: float = 0.5
    auth_twin_disagree_max: float = 0.5
    auth_perturb_order_min: float = 0.5
    auth_eps: float = 0.05

    def manifest(self) -> dict:
        return {"actor_lr": self.actor_lr, "policy_delay": self.policy_delay, "lambda_bc": self.lambda_bc,
                "trust_region": {"step_median": self.step_median, "step_p95": self.step_p95, "step_max": self.step_max,
                                 "cum_p95": self.cum_p95, "cum_max": self.cum_max},
                "backtrack_scales": list(self.backtrack_scales),
                "auth_gate": {"boundary_pref_max": self.auth_boundary_pref_max,
                              "twin_disagree_max": self.auth_twin_disagree_max,
                              "perturb_order_min": self.auth_perturb_order_min, "eps": self.auth_eps}}


def build_anchor_bank(pi0, train_bank, families, *, horizon: int = 360):
    """Frozen gate-active anchor bank = the reconstructed handoff observations of the Stage-1 train late-starts."""
    obs = []
    for ls in train_bank:
        if ls.family not in families:
            continue
        _rl, _g, _h, rec = reconstruct_handoff(pi0, ls, horizon=horizon)
        obs.append(rec.obs.astype(np.float32))
    return torch.tensor(np.stack(obs))


def _actions(actor, obs):
    return torch.clamp(actor.action_mean(obs), -ACTION_SCALE, ACTION_SCALE)


def within_trust_region(step_delta: np.ndarray, cum_delta: np.ndarray, cfg: TransactionalConfig) -> bool:
    """# Postcondition: True iff per-step drift (median/p95/max) AND cumulative drift (p95/max) are all within bounds."""
    return bool(np.median(step_delta) <= cfg.step_median and np.percentile(step_delta, 95) <= cfg.step_p95
                and np.max(step_delta) <= cfg.step_max
                and np.percentile(cum_delta, 95) <= cfg.cum_p95 and np.max(cum_delta) <= cfg.cum_max)


def critic_authorization(critic, actor, anchor_obs, cfg: TransactionalConfig) -> dict:
    """Frozen local-action diagnostic gate: finite Q/grad, no universal bound preference, bounded twin disagreement,
    local-ascent ordering not below chance. Returns the checks + ``authorized``."""
    a_cur = _actions(actor, anchor_obs).detach()
    with torch.no_grad():
        q1, q2 = critic(anchor_obs, a_cur)
    finite_q = bool(torch.isfinite(q1).all() and torch.isfinite(q2).all())
    # action-gradient (robust to a non-finite / action-independent critic)
    a_grad = a_cur.clone().requires_grad_(True)
    q1g, _ = critic(anchor_obs, a_grad)
    if finite_q and q1g.requires_grad:
        q1g.sum().backward(); grad = a_grad.grad
        finite_grad = grad is not None and bool(torch.isfinite(grad).all())
        grad = grad.detach() if grad is not None else torch.zeros_like(a_cur)
    else:
        grad, finite_grad = torch.zeros_like(a_cur), False
    with torch.no_grad():
        twin = (q1 - q2).abs() / (0.5 * (q1.abs() + q2.abs()) + 1e-6)
        twin_disagree = float(twin.mean())
        # candidate local actions: ±eps on each actuator basis + ±eps along the unit gradient
        gnorm = grad.norm(dim=-1, keepdim=True) + 1e-9
        gunit = grad / gnorm
        cands = [a_cur]
        for k in range(4):
            e = torch.zeros_like(a_cur); e[:, k] = cfg.auth_eps
            cands += [torch.clamp(a_cur + e, -4, 4), torch.clamp(a_cur - e, -4, 4)]
        cands += [torch.clamp(a_cur + cfg.auth_eps * gunit, -4, 4), torch.clamp(a_cur - cfg.auth_eps * gunit, -4, 4)]
        qs = torch.stack([critic(anchor_obs, c)[0] for c in cands], dim=1)   # [N, n_cand]
        best = cands[0].new_zeros(a_cur.shape[0], 4)
        best_idx = qs.argmax(dim=1)
        for i in range(a_cur.shape[0]):
            best[i] = cands[int(best_idx[i])][i]
        boundary_pref = float((best.abs() >= 3.9).any(dim=-1).float().mean())
        q_plus = critic(anchor_obs, torch.clamp(a_cur + cfg.auth_eps * gunit, -4, 4))[0]
        perturb_order = float((q_plus > q1).float().mean())     # grad step locally increases Q (self-consistency)
    checks = {"finite_Q": finite_q, "finite_grad": finite_grad,
              "boundary_pref": round(boundary_pref, 3), "boundary_ok": boundary_pref < cfg.auth_boundary_pref_max,
              "twin_disagree": round(twin_disagree, 3), "twin_ok": twin_disagree < cfg.auth_twin_disagree_max,
              "perturb_order": round(perturb_order, 3), "perturb_ok": perturb_order >= cfg.auth_perturb_order_min,
              "q1_mean": round(float(q1.mean()), 3), "q2_mean": round(float(q2.mean()), 3)}
    checks["authorized"] = bool(finite_q and finite_grad and checks["boundary_ok"] and checks["twin_ok"] and checks["perturb_ok"])
    return checks


def transactional_actor_step(pi_late, a_opt, critic, pi0, actor_obs, actor_gate, bc_obs, anchor_obs, a0_anchor, cfg):
    """One TRANSACTIONAL actor update. Returns a dict: outcome ∈ {accepted, rejected}, scale, step/cum drift stats.
    The Q term uses the §1 masked actor loss on ``(actor_obs, actor_gate)`` (only current gate-on rows drive pi_late)."""
    snap_params = {k: v.clone() for k, v in pi_late.state_dict().items()}
    snap_opt = copy.deepcopy(a_opt.state_dict())
    old_anchor = _actions(pi_late, anchor_obs).detach()

    with torch.no_grad():
        pi0_bc = _actions(pi0, bc_obs)
    bc = ((pi_late(bc_obs) - pi0_bc) ** 2).sum(-1).mean()   # secondary BC anchor on gate-active states
    a_loss = masked_actor_loss(critic, pi_late, actor_obs, actor_gate) + cfg.lambda_bc * bc
    a_opt.zero_grad(); a_loss.backward(); nn.utils.clip_grad_norm_(pi_late.parameters(), 1.0); a_opt.step()
    param_delta = {k: (pi_late.state_dict()[k] - snap_params[k]).clone() for k in snap_params}

    def measure():
        prop = _actions(pi_late, anchor_obs).detach()
        step_d = (prop - old_anchor).norm(dim=-1).numpy()
        cum_d = (prop - a0_anchor).norm(dim=-1).numpy()
        return step_d, cum_d

    step_d, cum_d = measure()
    if within_trust_region(step_d, cum_d, cfg):
        return {"outcome": "accepted", "scale": 1.0, "step_p95": float(np.percentile(step_d, 95)),
                "cum_p95": float(np.percentile(cum_d, 95)), "cum_max": float(np.max(cum_d)), "bc": float(bc.detach())}
    for scale in cfg.backtrack_scales:
        with torch.no_grad():
            for k in snap_params:
                pi_late.state_dict()[k].copy_(snap_params[k] + scale * param_delta[k])
        a_opt.load_state_dict(snap_opt)                         # optimizer restored (scaled param delta applied directly)
        step_d, cum_d = measure()
        if within_trust_region(step_d, cum_d, cfg):
            return {"outcome": "accepted", "scale": float(scale), "step_p95": float(np.percentile(step_d, 95)),
                    "cum_p95": float(np.percentile(cum_d, 95)), "cum_max": float(np.max(cum_d)), "bc": float(bc.detach())}
    with torch.no_grad():                                       # reject: restore actor + optimizer
        for k in snap_params:
            pi_late.state_dict()[k].copy_(snap_params[k])
    a_opt.load_state_dict(snap_opt)
    return {"outcome": "rejected", "scale": None, "step_p95": float(np.percentile(step_d, 95)),
            "cum_p95": float(np.percentile(cum_d, 95)), "cum_max": float(np.max(cum_d)), "bc": float(bc.detach())}


def train_stage1b(pi0, cfg_stage, train_bank, dev_bank, *, seeds, tcfg: "TransactionalConfig | None" = None, log=print):
    tcfg = tcfg or TransactionalConfig()
    torch.manual_seed(seeds["torch"])
    families = tuple(cfg_stage["families"]); horizon = cfg_stage["horizon"]; n_step = cfg_stage["n_step"]
    warmup = cfg_stage["critic_warmup_steps"]; total = cfg_stage["total_updates"]
    collect_every, eps_per = cfg_stage["collect_every"], cfg_stage["episodes_per_collect"]
    checkpoints = set(cfg_stage["checkpoints"]); tau, batch = 0.005, 256
    exp_init, exp_max = 0.15, 0.30; smoothing_std, smoothing_clip = 0.10, 0.25

    pi_late = make_late_actor_from_pi0(pi0, trainable=True)
    pi_late_target = make_late_actor_from_pi0(pi0, trainable=False)
    critic = LateTwinCritic(); critic_target = copy.deepcopy(critic)
    a_opt = torch.optim.Adam(pi_late.parameters(), lr=tcfg.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    buf = LateReplayBuffer(); rng_c = np.random.default_rng(seeds["numpy_collect"]); rng_r = np.random.default_rng(seeds["numpy_replay"])
    gen = torch.Generator().manual_seed(seeds["torch"])
    train_starts = [ls for ls in train_bank if ls.family in families]
    anchor_obs = build_anchor_bank(pi0, train_bank, families)
    a0_anchor = _actions(pi0, anchor_obs).detach()
    probe = torch.randn(64, 48, generator=torch.Generator().manual_seed(123))
    a0_probe = _actions(pi0, probe).detach().numpy()

    def collect(std):
        cn = CoherentNoise(std=std, hold_min=2, hold_max=4, seed=int(rng_c.integers(1 << 30)))
        for j in rng_c.integers(0, len(train_starts), eps_per):
            trs = collect_late_episode(pi0, pi_late, train_starts[j], cn, horizon=horizon, explore=True)
            if trs:
                buf.add_trajectory(trs)

    collect(exp_init)
    ckpt_dev, acc, rej, scales = {}, 0, 0, []
    for step in range(total + 1):
        if step in checkpoints:
            snap = make_late_actor_from_pi0(pi0, trainable=False); snap.load_state_dict(pi_late.state_dict())
            dev = eval_late_controller(pi0, snap, dev_bank, horizon=horizon, families=families)
            auth = critic_authorization(critic, pi_late, anchor_obs, tcfg)
            dev["auth"] = auth; dev["calibration_ok"] = auth["finite_Q"] and auth["twin_ok"]
            dev["accepted"], dev["rejected"] = acc, rej
            dev["actor_drift_from_update0"] = round(float(np.abs(_actions(snap, probe).detach().numpy() - a0_probe).max()), 5)
            cum = (_actions(snap, anchor_obs).detach() - a0_anchor).norm(dim=-1).numpy()
            dev["anchor_cum_p95"] = round(float(np.percentile(cum, 95)), 5); dev["anchor_cum_max"] = round(float(np.max(cum)), 5)
            dev["boundary_fraction"] = auth["boundary_pref"]; dev["scale_hist"] = _hist(scales)
            ckpt_dev[step] = dev
            dl = dev["delta_vs_pi0"]
            log(f"  [ckpt {step}] auth={auth['authorized']} acc/rej {acc}/{rej} Δstrict {dl['strict_success']:+.3f} "
                f"Δdwell {dl['max_dwell']:+.2f} Δexit {dl['exited']:+.3f} Δcontact {dl['contact_retention']:+.3f} "
                f"drift {dev['actor_drift_from_update0']:.4f} q1 {auth['q1_mean']:.1f} twin {auth['twin_disagree']:.2f} bound {auth['boundary_pref']:.2f}")
        if step == total:
            break
        if step > 0 and step % collect_every == 0:
            collect(min(exp_max, exp_init + (exp_max - exp_init) * step / total))
        O, A, R, B, M, GP, BG = _sample_nstep_phase(buf, batch, n_step, GAMMA, rng_r)
        with torch.no_grad():
            ta = phase_target_action(pi0, pi_late_target, B, BG, std=smoothing_std, clip=smoothing_clip, gen=gen)
            y = R + GP * M * critic_target.min_q(B, ta)
        q1, q2 = critic(O, A)
        c_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        c_opt.zero_grad(); c_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); c_opt.step()
        if step >= warmup and step % tcfg.policy_delay == 0:
            auth = critic_authorization(critic, pi_late, anchor_obs, tcfg)
            if auth["authorized"]:                              # transactional actor update ONLY when critic authorized
                sample = sample_actor_batch(buf, batch, rng_r); bc_obs = _sample_gate_on(buf, batch, rng_r)
                if sample is not None and bc_obs is not None and float(sample[1].sum()) > 0:
                    r = transactional_actor_step(pi_late, a_opt, critic, pi0, sample[0], sample[1], bc_obs,
                                                 anchor_obs, a0_anchor, tcfg)
                    if r["outcome"] == "accepted":
                        acc += 1; scales.append(r["scale"])
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

    ks = sorted(ckpt_dev)
    ever_auth = any(ckpt_dev[k]["auth"]["authorized"] for k in ks)
    passed = ever_auth and any(stage1b_gate(ckpt_dev[ks[i]], ckpt_dev[ks[i + 1]], tcfg) for i in range(len(ks) - 1))
    return {"checkpoints": {str(k): v for k, v in ckpt_dev.items()}, "stage1b_pass": passed,
            "critic_ever_authorized": ever_auth, "accepted": acc, "rejected": rej,
            "scale_hist": _hist(scales), "transactional_config": tcfg.manifest()}


def _hist(scales):
    from collections import Counter
    return {str(k): v for k, v in sorted(Counter(scales).items())}
