"""PHASE_SWITCHED_TD3_BASELINE_V1 trainer (Stage-1). Collects phase-switched late episodes from deterministically
reconstructed handoffs, trains a twin critic + delayed full-action ``pi_late`` actor with n-step targets, coherent
exploration, and a PHASE-CORRECT target action (gate-on ⇒ smoothed ``pi_late``; gate-off ⇒ frozen ``pi_0``). Evaluates
on disjoint dev late-starts against the frozen ``pi_0`` continuation. No neutral-reset composed eval here.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
from hymeko_rl.coin_delivery.coin_phase_switched_late import make_late_actor_from_pi0
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals
from hymeko_rl.coin_delivery.coin_td3_contracts import CoherentNoise, LateReplayBuffer, LateTwinCritic
from hymeko_rl.coin_delivery.rl_clip_actor import ClipDeterministicActor

ACTION_SCALE, GAMMA = 4.0, 0.99


def _det(actor, o):
    with torch.no_grad():
        return np.clip(actor.action_mean(torch.as_tensor(np.asarray(o, np.float32)[None]))[0].numpy(),
                       -ACTION_SCALE, ACTION_SCALE).astype(np.float32)


# ── phase-switched late-episode collection ──
def collect_late_episode(pi0, pi_late, ls, cnoise: "CoherentNoise | None", *, horizon: int, explore: bool):
    """Reconstruct the handoff (deterministic replay), then run the phase-switched controller for ``horizon`` steps.
    Exploration noise is added ONLY on gate-on steps (where ``pi_late`` acts); gate-off steps are pi_0 exact. Returns the
    transition list (with ``gate_on``/``gate_on_next`` and family)."""
    rl, gate, _hist, rec = reconstruct_handoff(pi0, ls, horizon=360)
    if cnoise is not None:
        cnoise.reset()
    o = rec.obs.astype(np.float32); trs = []
    for k in range(horizon):
        gate_on = gate.gate == 1.0
        if gate_on:
            a = _det(pi_late, o)
            if explore and cnoise is not None:
                a = cnoise.perturb(a)
        else:
            a = _det(pi0, o)
        o2, r, term, trunc, _ = rl.step(a)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
        gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        cut = (k == horizon - 1)                                 # late-horizon boundary = truncation (bootstrap)
        trs.append({"obs": o, "action": a.astype(np.float32), "reward": float(r), "obs_next": o2.astype(np.float32),
                    "terminated": bool(term), "truncated": bool(trunc or (cut and not term)),
                    "gate_on": bool(gate_on), "gate_on_next": bool(gate.gate == 1.0), "family": ls.family})
        o = o2
        if term or trunc:
            break
    return trs


def _nstep_with_gate(traj, t, n, gamma):
    """n-step return + the gate flag at the bootstrap state (to choose the phase-correct target action)."""
    G, disc, mask, steps = 0.0, 1.0, 1, 0
    boot, boot_gate_on = traj[t]["obs_next"], traj[t]["gate_on_next"]
    for i in range(n):
        idx = t + i
        if idx >= len(traj):
            break
        tr = traj[idx]
        G += disc * float(tr["reward"]); boot = tr["obs_next"]; boot_gate_on = tr["gate_on_next"]
        disc *= gamma; steps += 1
        if tr["terminated"]:
            mask = 0; break
        if tr["truncated"]:
            mask = 1; break
    return G, boot, mask, disc, boot_gate_on


def _sample_nstep_phase(buf, batch, n, gamma, rng):
    flat = [(ti, t) for ti, traj in enumerate(buf.trajectories) for t in range(len(traj))]
    pick = rng.integers(0, len(flat), batch)
    O, A, R, B, M, GP, BG = [], [], [], [], [], [], []
    for j in pick:
        ti, t = flat[j]; traj = buf.trajectories[ti]
        G, b, m, gp, bg = _nstep_with_gate(traj, t, n, gamma)
        O.append(traj[t]["obs"]); A.append(traj[t]["action"]); R.append(G); B.append(b); M.append(m); GP.append(gp); BG.append(bg)
    return (torch.tensor(np.stack(O)), torch.tensor(np.stack(A)), torch.tensor(np.array(R, np.float32)),
            torch.tensor(np.stack(B)), torch.tensor(np.array(M, np.float32)), torch.tensor(np.array(GP, np.float32)),
            torch.tensor(np.array(BG, np.float32)))


def phase_target_action(pi0, pi_late_target, obs_next, gate_next, *, std, clip, gen):
    """gate_next==1 ⇒ smoothed pi_late target; gate_next==0 ⇒ frozen pi_0 (no smoothing). Full-action units."""
    with torch.no_grad():
        base_pi0 = torch.clamp(pi0.action_mean(obs_next), -ACTION_SCALE, ACTION_SCALE)
        base_late = torch.clamp(pi_late_target.action_mean(obs_next), -ACTION_SCALE, ACTION_SCALE)
    noise = torch.clamp(torch.randn(base_late.shape, generator=gen) * std, -clip, clip)
    late = torch.clamp(base_late + noise, -ACTION_SCALE, ACTION_SCALE)
    g = gate_next.unsqueeze(-1)
    return g * late + (1.0 - g) * base_pi0


# ── dev evaluation vs frozen pi_0 continuation ──
def _rollout_metrics(pi0, actor, ls, horizon, *, use_late: bool):
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    o = rec.obs.astype(np.float32); tot, disc = 0.0, 1.0
    max_dwell = rl._strict; entered = rl._dtz() <= CENTER_TOL; exited = False
    contact_steps = 0; n = 0; term = trunc = False
    for _k in range(horizon):
        a = _det(actor if (use_late and gate.gate == 1.0) else pi0, o)
        o2, r, term, trunc, _ = rl.step(a); tot += disc * r; disc *= GAMMA
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
        gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        m = rl.inner._planar_metrics
        contact_steps += int(bool(m.left_contact or m.right_contact)); n += 1
        dtz = rl._dtz(); max_dwell = max(max_dwell, rl._strict)
        if dtz <= CENTER_TOL:
            entered = True
        elif entered:
            exited = True
        o = o2
        if term or trunc:
            break
    return {"strict_success": int(max_dwell >= HELD_DWELL), "max_dwell": int(max_dwell), "entered": int(entered),
            "exited": int(exited), "contact_retention": contact_steps / max(n, 1), "ret": float(tot)}


def eval_late_controller(pi0, pi_late, dev_bank, *, horizon: int, families) -> dict:
    """Per-family and pooled metrics for ``pi_late`` and frozen ``pi_0``, from the SAME reconstructed handoffs."""
    starts = [ls for ls in dev_bank if ls.family in families]
    late = [_rollout_metrics(pi0, pi_late, ls, horizon, use_late=True) for ls in starts]
    base = [_rollout_metrics(pi0, pi0, ls, horizon, use_late=False) for ls in starts]

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0
    keys = ["strict_success", "max_dwell", "entered", "exited", "contact_retention", "ret"]
    out = {"n": len(starts), "late": {k: round(agg(late, k), 3) for k in keys},
           "pi0": {k: round(agg(base, k), 3) for k in keys}}
    out["delta_vs_pi0"] = {k: round(out["late"][k] - out["pi0"][k], 3) for k in keys}
    out["by_family"] = {}
    for f in families:
        li = [r for r, ls in zip(late, starts) if ls.family == f]; bi = [r for r, ls in zip(base, starts) if ls.family == f]
        out["by_family"][f] = {"n": len(li), "d_strict": round(agg(li, "strict_success") - agg(bi, "strict_success"), 3),
                               "d_dwell": round(agg(li, "max_dwell") - agg(bi, "max_dwell"), 3),
                               "d_exit": round(agg(li, "exited") - agg(bi, "exited"), 3),
                               "d_contact": round(agg(li, "contact_retention") - agg(bi, "contact_retention"), 3)}
    return out


def stage_gate(dev_a: dict, dev_b: dict) -> bool:
    """Two CONSECUTIVE checkpoints must each: improve strict OR max_dwell vs pi_0; not raise target-exit > +0.05;
    not degrade contact-retention < −0.05; have stable (finite) calibration."""
    def ck(d):
        dl = d["delta_vs_pi0"]
        improve = dl["strict_success"] > 0 or dl["max_dwell"] > 0.05
        return (improve and dl["exited"] <= 0.05 and dl["contact_retention"] >= -0.05
                and d.get("calibration_ok", True))
    return ck(dev_a) and ck(dev_b)


# ── training loop ──
def train_stage1(pi0, cfg_stage, train_bank, dev_bank, *, seeds, log=print):
    torch.manual_seed(seeds["torch"])
    families = tuple(cfg_stage["families"]); horizon = cfg_stage["horizon"]; n_step = cfg_stage["n_step"]
    smoothing_std, smoothing_clip = 0.10, 0.25
    warmup, policy_delay = cfg_stage["critic_warmup_steps"], cfg_stage["policy_delay"]
    total, collect_every, eps_per = cfg_stage["total_updates"], cfg_stage["collect_every"], cfg_stage["episodes_per_collect"]
    checkpoints = set(cfg_stage["checkpoints"])
    tau, batch, exp_init, exp_max = 0.005, 256, 0.15, 0.30

    pi_late = make_late_actor_from_pi0(pi0, trainable=True)
    pi_late_target = make_late_actor_from_pi0(pi0, trainable=False)
    critic = LateTwinCritic(); critic_target = copy.deepcopy(critic)
    a_opt = torch.optim.Adam(pi_late.parameters(), lr=3e-4)
    c_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    buf = LateReplayBuffer(); rng_c = np.random.default_rng(seeds["numpy_collect"]); rng_r = np.random.default_rng(seeds["numpy_replay"])
    train_starts = [ls for ls in train_bank if ls.family in families]
    probe = torch.randn(64, 48, generator=torch.Generator().manual_seed(123))
    a0 = _det_probe(pi0, probe)
    gen = torch.Generator().manual_seed(seeds["torch"])

    def collect(std):
        cn = CoherentNoise(std=std, hold_min=2, hold_max=4, seed=int(rng_c.integers(1 << 30)))
        picks = rng_c.integers(0, len(train_starts), eps_per)
        for j in picks:
            trs = collect_late_episode(pi0, pi_late, train_starts[j], cn, horizon=horizon, explore=True)
            if trs:
                buf.add_trajectory(trs)

    collect(exp_init)                                            # seed the buffer
    ckpt_dev = {}
    for step in range(total + 1):
        if step in checkpoints:
            snap = make_late_actor_from_pi0(pi0, trainable=False); snap.load_state_dict(pi_late.state_dict())
            dev = eval_late_controller(pi0, snap, dev_bank, horizon=horizon, families=families)
            with torch.no_grad():
                q1, q2 = critic(probe[:8], torch.zeros(8, 4))
            dev["calibration_ok"] = bool(torch.isfinite(q1).all() and torch.isfinite(q2).all()
                                         and (q1 - q2).abs().max().item() < 1e4)
            dev["q1_mean"] = round(float(q1.mean()), 3); dev["q2_mean"] = round(float(q2.mean()), 3)
            dev["actor_drift_from_update0"] = round(float(np.abs(_det_probe(snap, probe) - a0).max()), 4)
            ckpt_dev[step] = dev
            log(f"  [ckpt {step}] Δstrict {dev['delta_vs_pi0']['strict_success']:+.3f} Δdwell {dev['delta_vs_pi0']['max_dwell']:+.2f} "
                f"Δexit {dev['delta_vs_pi0']['exited']:+.3f} Δcontact {dev['delta_vs_pi0']['contact_retention']:+.3f} "
                f"drift {dev['actor_drift_from_update0']:.3f} q1 {dev['q1_mean']:.1f} calib {dev['calibration_ok']}")
        if step == total:
            break
        if step > 0 and step % collect_every == 0:
            collect(min(exp_max, exp_init + (exp_max - exp_init) * step / total))
        # critic update (n-step, phase-correct target)
        O, A, R, B, M, GP, BG = _sample_nstep_phase(buf, batch, n_step, GAMMA, rng_r)
        with torch.no_grad():
            ta = phase_target_action(pi0, pi_late_target, B, BG, std=smoothing_std, clip=smoothing_clip, gen=gen)
            qn = critic_target.min_q(B, ta)
            y = R + GP * M * qn
        q1, q2 = critic(O, A)
        c_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        c_opt.zero_grad(); c_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); c_opt.step()
        # delayed actor update on GATE-ON states, after critic warm-up
        if step >= warmup and step % policy_delay == 0:
            on = _sample_gate_on(buf, batch, rng_r)
            if on is not None:
                q1_pi, _ = critic(on, pi_late(on))               # ascend Q1 at gate-on states (pi_late acts there)
                a_loss = -q1_pi.mean()
                a_opt.zero_grad(); a_loss.backward(); nn.utils.clip_grad_norm_(pi_late.parameters(), 1.0); a_opt.step()
                with torch.no_grad():
                    for p, pt in zip(pi_late.parameters(), pi_late_target.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
        with torch.no_grad():
            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
        if step % 1000 == 0:
            log(f"    step {step}/{total} c_loss {c_loss.item():.3f} buf {buf.n_transitions()}")

    ks = sorted(ckpt_dev)
    passed = any(stage_gate(ckpt_dev[ks[i]], ckpt_dev[ks[i + 1]]) for i in range(len(ks) - 1))
    return {"checkpoints": {str(k): v for k, v in ckpt_dev.items()}, "stage1_pass": passed,
            "final_actor_drift": ckpt_dev[ks[-1]]["actor_drift_from_update0"]}


def _det_probe(actor: ClipDeterministicActor, probe: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.clamp(actor.action_mean(probe), -ACTION_SCALE, ACTION_SCALE).numpy()


def _sample_gate_on(buf, batch, rng):
    flat = [(ti, t) for ti, traj in enumerate(buf.trajectories) for t in range(len(traj)) if traj[t]["gate_on"]]
    if not flat:
        return None
    pick = rng.integers(0, len(flat), min(batch, len(flat)))
    return torch.tensor(np.stack([buf.trajectories[flat[j][0]][flat[j][1]]["obs"] for j in pick]))
