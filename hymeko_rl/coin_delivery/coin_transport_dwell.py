"""TRANSPORT_TO_DWELL_TD3_BASELINE_V1 — re-scoped late controller. Control modes are the phases that actually PERSIST:

    control_mode ∈ {transport, braking, settling_dwell}          (target_entry REMOVED — it is a 1-step event)
    contact_flag ∈ {contact_present, contact_lost}               (orthogonal)
    target-entry EVENT features = [inside_target_zone, just_entered, just_exited, distance_to_target, radial_velocity]

Actor/critic conditioning = onehot3(control_mode) ++ onehot2(contact_flag) ++ event(5) = 10 dims (obs_48 ++ 10 = 58).
The conditioning weights are ZERO-init ⇒ update-0 == pi_0. Same transactional TD3 machinery, reward, gate, 4-step target,
0.10/0.25 smoothing, coherent exploration, and trust-region caps as Stage-1b/1c — only the ontology + horizon (60) +
dynamically-balanced sampling change.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_late_start import LateStart, _base, _sha
from hymeko_rl.coin_delivery.coin_phase_conditioning import make_phase_actor_from_pi0, make_phase_critic
from hymeko_rl.coin_delivery.coin_residual_critic_state import ResidualCriticStateV2
from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, SETTLE_VEL, CoinRL4Dof
from hymeko_rl.coin_delivery.coin_stable_engagement import StableEngagementConfig, StableEngagementGate, stable_engagement_signals
from hymeko_rl.coin_delivery.coin_td3_contracts import CoherentNoise, LateReplayBuffer
from hymeko_rl.coin_delivery.coin_td3_trainer import GAMMA, _det, masked_actor_loss
from hymeko_rl.coin_delivery.coin_td3_transactional import critic_authorization, transactional_actor_step

ENTRY_TOL = 0.05
ACTION_SCALE = 4.0
CONTROL_MODES = ("transport", "braking", "settling_dwell")
CONTACT_FLAGS = ("contact_present", "contact_lost")
N_EVENT = 5
N_COND = len(CONTROL_MODES) + len(CONTACT_FLAGS) + N_EVENT      # 3+2+5 = 10
SAMPLE_TARGET = {"transport": 0.5, "braking": 0.3, "settling_dwell": 0.2}
BANK_MIN = {"transport": 20, "braking": 12, "settling_dwell": 8}


def control_mode(dtz, speed, prev_speed, strict) -> str:
    if strict >= 1 or (dtz <= CENTER_TOL and speed < SETTLE_VEL):
        return "settling_dwell"
    if prev_speed - speed > 0.02 and dtz <= 2 * ENTRY_TOL:
        return "braking"
    return "transport"


def contact_flag(lc, rc) -> str:
    return "contact_present" if (lc or rc) else "contact_lost"


class EventStateDetector:
    """(control_mode, contact_flag, event_features) from the current state + running context. Call once per state."""

    def __init__(self):
        self.prev_dtz = None; self.prev_speed = None

    def state_of(self, rl):
        m = rl.inner._planar_metrics
        dtz = float(m.disk_to_zone); speed = float(rl._speed()); lc, rc, strict = bool(m.left_contact), bool(m.right_contact), int(rl._strict)
        pdtz = dtz if self.prev_dtz is None else self.prev_dtz
        pspd = speed if self.prev_speed is None else self.prev_speed
        cm = control_mode(dtz, speed, pspd, strict); cf = contact_flag(lc, rc)
        ev = np.array([float(dtz <= ENTRY_TOL), float(pdtz > ENTRY_TOL >= dtz), float(pdtz <= ENTRY_TOL < dtz),
                       dtz, dtz - pdtz], np.float32)                # inside, just_entered, just_exited, distance, radial_vel
        self.prev_dtz = dtz; self.prev_speed = speed
        return cm, cf, ev


def state_vector(cm: str, cf: str, ev: np.ndarray) -> np.ndarray:
    v = np.zeros(N_COND, np.float32)
    v[CONTROL_MODES.index(cm)] = 1.0
    v[len(CONTROL_MODES) + CONTACT_FLAGS.index(cf)] = 1.0
    v[len(CONTROL_MODES) + len(CONTACT_FLAGS):] = ev
    return v


def augment_td(obs, cm, cf, ev) -> np.ndarray:
    return np.concatenate([np.asarray(obs, np.float32), state_vector(cm, cf, ev)]).astype(np.float32)


def actor_trainable(gate_on: bool, cm: str) -> float:
    return 1.0 if (bool(gate_on) and cm in CONTROL_MODES) else 0.0


# ── §6 rebuild persistent control-mode banks ──
def rebuild_control_mode_bank(pi0, seeds, *, min_persist=2, per_mode=None, horizon=360):
    per_mode = per_mode or {"transport": 40, "braking": 40, "settling_dwell": 40}
    banks = {m: [] for m in CONTROL_MODES}
    for s in seeds:
        if all(len(banks[m]) >= per_mode[m] for m in CONTROL_MODES):
            break
        rl = CoinRL4Dof(horizon=horizon); o = rl.reset(int(s))
        gate = StableEngagementGate(StableEngagementConfig()); det = EventStateDetector()
        hist = ResidualCriticStateV2(); hist.reset(o); rows = []
        for _k in range(horizon):
            cm, _cf, _ev = det.state_of(rl); gd = ReplayControllerStateV2.from_gate(gate).to_dict()
            rows.append((_k, cm, gate.gate == 1.0, o.astype(np.float32), _base(pi0, o).astype(np.float32),
                         hist.feature(gd).astype(np.float32), gd))
            a = rows[-1][4]; o2, _r, term, trunc, _ = rl.step(a); hist.push(o2, a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            o = o2
            if term or trunc:
                break
        seen = set()
        for i in range(len(rows) - min_persist + 1):
            k, cm, g, obs, base, causal, gd = rows[i]
            if not g or cm in seen or len(banks[cm]) >= per_mode[cm]:
                continue
            if all(rows[i + j][2] and rows[i + j][1] == cm for j in range(min_persist)):
                seen.add(cm)
                banks[cm].append(LateStart(seed=int(s), prefix_steps=k, family=cm, obs_sha=_sha(obs),
                                           base_sha=_sha(base), causal_sha=_sha(causal), gate_state=gd))
    return banks, {m: len(banks[m]) for m in CONTROL_MODES}


# ── collection (event-conditioned; horizon 60) ──
def collect_td_episode(pi0, pi_late, ls, cnoise, *, horizon, explore):
    from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    det = EventStateDetector()
    if cnoise is not None:
        cnoise.reset()
    o = rec.obs.astype(np.float32); cm, cf, ev = det.state_of(rl); trs = []
    for k in range(horizon):
        gate_on = gate.gate == 1.0
        if gate_on:
            a = _det(pi_late, augment_td(o, cm, cf, ev))
            if explore and cnoise is not None:
                a = cnoise.perturb(a)
        else:
            a = _det(pi0, o)
        o2, r, term, trunc, _ = rl.step(a)
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        cm2, cf2, ev2 = det.state_of(rl); cut = (k == horizon - 1)
        trs.append({"obs": o, "cm": cm, "cf": cf, "ev": ev, "cm_next": cm2, "cf_next": cf2, "ev_next": ev2,
                    "action": a.astype(np.float32), "reward": float(r), "obs_next": o2.astype(np.float32),
                    "terminated": bool(term), "truncated": bool(trunc or (cut and not term)),
                    "gate_on": bool(gate_on), "gate_on_next": bool(gate.gate == 1.0)})
        o = o2; cm, cf, ev = cm2, cf2, ev2
        if term or trunc:
            break
    return trs


# ── §8 dynamically-balanced actor sampling (50/30/20, not forced equal) ──
def balanced_actor_batch(buf, batch, rng):
    pools = {m: [] for m in CONTROL_MODES}
    for ti, traj in enumerate(buf.trajectories):
        for t in range(len(traj)):
            if traj[t]["gate_on"] and traj[t]["cm"] in pools:
                pools[traj[t]["cm"]].append((ti, t))
    if not any(pools.values()):
        return None
    aug, gate = [], []
    for m in CONTROL_MODES:
        want = int(round(SAMPLE_TARGET[m] * batch))
        if not pools[m] or want == 0:
            continue
        for j in rng.integers(0, len(pools[m]), want):
            ti, t = pools[m][int(j)]; tr = buf.trajectories[ti][t]
            aug.append(augment_td(tr["obs"], tr["cm"], tr["cf"], tr["ev"])); gate.append(actor_trainable(True, tr["cm"]))
    if not aug:
        return None
    return torch.tensor(np.stack(aug)), torch.tensor(np.array(gate, np.float32))


def _flat(buf):
    return [(ti, t) for ti, traj in enumerate(buf.trajectories) for t in range(len(traj))]


def sample_nstep_td(buf, batch, n, gamma, rng):
    flat = _flat(buf); pick = rng.integers(0, len(flat), batch)
    A, ACT, R, B, B48, M, GP, BG = [], [], [], [], [], [], [], []
    for j in pick:
        ti, t = flat[j]; traj = buf.trajectories[ti]
        G, disc, mask, bi = 0.0, 1.0, 1, t
        for i in range(n):
            idx = t + i
            if idx >= len(traj):
                break
            G += disc * float(traj[idx]["reward"]); bi = idx; disc *= gamma
            if traj[idx]["terminated"]:
                mask = 0; break
            if traj[idx]["truncated"]:
                mask = 1; break
        tr, bt = traj[t], traj[bi]                              # bt = last accumulated step; bootstrap on its next-state
        A.append(augment_td(tr["obs"], tr["cm"], tr["cf"], tr["ev"])); ACT.append(tr["action"]); R.append(G)
        B.append(augment_td(bt["obs_next"], bt["cm_next"], bt["cf_next"], bt["ev_next"])); B48.append(bt["obs_next"])
        M.append(mask); GP.append(disc); BG.append(float(bt["gate_on_next"]))
    T = torch.tensor
    return (T(np.stack(A)), T(np.stack(ACT)), T(np.array(R, np.float32)), T(np.stack(B)), T(np.stack(B48).astype(np.float32)),
            T(np.array(M, np.float32)), T(np.array(GP, np.float32)), T(np.array(BG, np.float32)))


def td_target_action(pi0, pi_late_target, boot_aug, boot48, gate_next, *, std, clip, gen):
    with torch.no_grad():
        base_pi0 = torch.clamp(pi0.action_mean(boot48), -ACTION_SCALE, ACTION_SCALE)
        base_late = torch.clamp(pi_late_target.action_mean(boot_aug), -ACTION_SCALE, ACTION_SCALE)
    noise = torch.clamp(torch.randn(base_late.shape, generator=gen) * std, -clip, clip)
    late = torch.clamp(base_late + noise, -ACTION_SCALE, ACTION_SCALE); g = gate_next.unsqueeze(-1)
    return g * late + (1.0 - g) * base_pi0


def build_anchor_td(pi0, banks_flat):
    from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
    obs48, aug = [], []
    for ls in banks_flat:
        rl, _g, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
        cm, cf, ev = EventStateDetector().state_of(rl)
        obs48.append(rec.obs.astype(np.float32)); aug.append(augment_td(rec.obs, cm, cf, ev))
    return torch.tensor(np.stack(obs48)), torch.tensor(np.stack(aug))


# ── §11 eval (transport-to-dwell metrics) ──
def _rollout_td(pi0, pi_late, ls, horizon, *, use_late):
    from hymeko_rl.coin_delivery.coin_late_start import reconstruct_handoff
    rl, gate, _h, rec = reconstruct_handoff(pi0, ls, horizon=360)
    det = EventStateDetector(); o = rec.obs.astype(np.float32); cm, cf, ev = det.state_of(rl)
    tot, disc = 0.0, 1.0; max_dwell = rl._strict; entered = rl._dtz() <= ENTRY_TOL; exited = False
    entry_speed = float("nan"); contact_steps = n = 0; progress = 0.0; brake_dv = []; prev_dtz = rl._dtz(); term = trunc = False
    for _k in range(horizon):
        a = _det(pi_late, augment_td(o, cm, cf, ev)) if (use_late and gate.gate == 1.0) else _det(pi0, o)
        prev_speed = rl._speed(); o2, r, term, trunc, _ = rl.step(a); tot += disc * r; disc *= GAMMA
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner); gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        cm, cf, ev = det.state_of(rl); dtz = rl._dtz(); sp = rl._speed()
        if cm == "transport":
            progress += (prev_dtz - dtz)
        if cm == "braking":
            brake_dv.append(prev_speed - sp)
        if dtz <= ENTRY_TOL and prev_dtz > ENTRY_TOL:
            entered = True; entry_speed = prev_speed
        elif dtz > ENTRY_TOL and prev_dtz <= ENTRY_TOL and entered:
            exited = True
        max_dwell = max(max_dwell, rl._strict); contact_steps += int(bool(rl.inner._planar_metrics.left_contact or rl.inner._planar_metrics.right_contact)); n += 1
        prev_dtz = dtz; o = o2
        if term or trunc:
            break
    return {"strict_success": int(max_dwell >= HELD_DWELL), "max_dwell": int(max_dwell), "entered": int(entered),
            "exited": int(exited), "entry_speed": entry_speed if entered else 0.0,
            "braking_eff": float(np.mean(brake_dv)) if brake_dv else 0.0, "progress": float(progress),
            "contact_retention": contact_steps / max(n, 1), "ret": float(tot)}


def eval_td(pi0, pi_late, dev_banks_flat, *, horizon):
    late = [_rollout_td(pi0, pi_late, ls, horizon, use_late=True) for ls in dev_banks_flat]
    base = [_rollout_td(pi0, pi0, ls, horizon, use_late=False) for ls in dev_banks_flat]

    def agg(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else 0.0
    keys = ["strict_success", "max_dwell", "entered", "exited", "entry_speed", "braking_eff", "progress", "contact_retention", "ret"]
    out = {"n": len(dev_banks_flat), "late": {k: round(agg(late, k), 4) for k in keys}, "pi0": {k: round(agg(base, k), 4) for k in keys}}
    out["delta_vs_pi0"] = {k: round(out["late"][k] - out["pi0"][k], 4) for k in keys}
    return out


def td_gate(a: dict, b: dict) -> bool:
    """§13: two consecutive TRAINED checkpoints improve ≥1 of {entry-rate, braking/entry-speed, max-dwell, strict}
    without materially degrading contact retention (≥ −0.05) or target exit (≤ +0.05)."""
    def ck(d):
        dl = d["delta_vs_pi0"]
        improve = (dl["entered"] > 0 or dl["max_dwell"] > 0.05 or dl["strict_success"] > 0
                   or dl["braking_eff"] > 0 or dl["entry_speed"] < 0)
        return (d.get("accepted", 0) > 0 and d["auth"]["authorized"] and improve
                and dl["contact_retention"] >= -0.05 and dl["exited"] <= 0.05)
    return ck(a) and ck(b)


def train_transport_dwell(pi0, cfg, train_banks, dev_flat, *, seeds, tcfg, log=print):
    torch.manual_seed(seeds["torch"])
    horizon, n_step, warmup, total = cfg["horizon"], cfg["n_step"], cfg["critic_warmup_steps"], cfg["total_updates"]
    collect_every, eps_per, checkpoints = cfg["collect_every"], cfg["episodes_per_collect"], set(cfg["checkpoints"])
    tau, batch, exp_init, exp_max, sstd, sclip = 0.005, 256, 0.15, 0.30, 0.10, 0.25
    pi_late = make_phase_actor_from_pi0(pi0, trainable=True, n_cond=N_COND)
    pi_late_target = make_phase_actor_from_pi0(pi0, trainable=False, n_cond=N_COND)
    critic = make_phase_critic(N_COND); critic_target = copy.deepcopy(critic)
    a_opt = torch.optim.Adam(pi_late.parameters(), lr=tcfg.actor_lr); c_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    buf = LateReplayBuffer(); rng_c = np.random.default_rng(seeds["numpy_collect"]); rng_r = np.random.default_rng(seeds["numpy_replay"])
    gen = torch.Generator().manual_seed(seeds["torch"])
    all_starts = [ls for m in CONTROL_MODES for ls in train_banks[m]]
    obs48_anchor, aug_anchor = build_anchor_td(pi0, all_starts)
    a0_anchor = torch.clamp(pi0.action_mean(obs48_anchor), -4, 4).detach()
    probe = torch.randn(64, 48, generator=torch.Generator().manual_seed(123)); a0_probe = torch.clamp(pi0.action_mean(probe), -4, 4).detach().numpy()

    def collect(std):
        cn = CoherentNoise(std=std, hold_min=2, hold_max=4, seed=int(rng_c.integers(1 << 30)))
        for j in rng_c.integers(0, len(all_starts), eps_per):
            trs = collect_td_episode(pi0, pi_late, all_starts[int(j)], cn, horizon=horizon, explore=True)
            if trs:
                buf.add_trajectory(trs)

    collect(exp_init); ckpt, acc, rej, step_maxes = {}, 0, 0, []
    for step in range(total + 1):
        if step in checkpoints:
            snap = make_phase_actor_from_pi0(pi0, trainable=False, n_cond=N_COND); snap.load_state_dict(pi_late.state_dict())
            dev = eval_td(pi0, snap, dev_flat, horizon=horizon)
            dev["auth"] = critic_authorization(critic, pi_late, aug_anchor, tcfg)
            with torch.no_grad():
                cum = (torch.clamp(snap.action_mean(aug_anchor), -4, 4) - a0_anchor).norm(dim=-1).numpy()
            dev["anchor_cum_max"] = round(float(np.max(cum)), 5); dev["anchor_cum_p95"] = round(float(np.percentile(cum, 95)), 5)
            dev["perstep_max"] = round(float(np.max(step_maxes)), 5) if step_maxes else 0.0
            with torch.no_grad():
                pl = torch.clamp(snap.action_mean(torch.cat([probe, torch.zeros(64, N_COND)], -1)), -4, 4).numpy()
            dev["probe_Linf_diag"] = round(float(np.abs(pl - a0_probe).max()), 5); dev["accepted"], dev["rejected"] = acc, rej
            ckpt[step] = dev; dl = dev["delta_vs_pi0"]
            log(f"  [ckpt {step}] auth={dev['auth']['authorized']} acc/rej {acc}/{rej} Δenter {dl['entered']:+.3f} "
                f"Δdwell {dl['max_dwell']:+.2f} Δstrict {dl['strict_success']:+.3f} Δbrake {dl['braking_eff']:+.4f} "
                f"Δprogress {dl['progress']:+.4f} Δcontact {dl['contact_retention']:+.3f} Δexit {dl['exited']:+.3f} | "
                f"anchorL2cum {dev['anchor_cum_max']} probeLinf {dev['probe_Linf_diag']}")
        if step == total:
            break
        if step > 0 and step % collect_every == 0:
            collect(min(exp_max, exp_init + (exp_max - exp_init) * step / total))
        A, ACT, R, B, B48, M, GP, BG = sample_nstep_td(buf, batch, n_step, GAMMA, rng_r)
        with torch.no_grad():
            ta = td_target_action(pi0, pi_late_target, B, B48, BG, std=sstd, clip=sclip, gen=gen)
            y = R + GP * M * critic_target.min_q(B, ta)
        q1, q2 = critic(A, ACT); c_loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        c_opt.zero_grad(); c_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); c_opt.step()
        if step >= warmup and step % tcfg.policy_delay == 0:
            auth = critic_authorization(critic, pi_late, aug_anchor, tcfg)
            if auth["authorized"]:
                sm = balanced_actor_batch(buf, batch, rng_r)
                if sm is not None and float(sm[1].sum()) > 0:
                    bc = sm  # BC anchor uses the same balanced gate-on states

                    def lf(aug_o=sm[0], gate=sm[1], bc_aug=bc[0]):
                        with torch.no_grad():
                            pi0_bc = torch.clamp(pi0.action_mean(bc_aug[:, :48]), -4, 4)
                        bcl = ((pi_late(bc_aug) - pi0_bc) ** 2).sum(-1).mean()
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
    ks = sorted(ckpt); ever = any(ckpt[k]["auth"]["authorized"] for k in ks)
    passed = ever and any(td_gate(ckpt[ks[i]], ckpt[ks[i + 1]]) for i in range(len(ks) - 1))
    return {"checkpoints": {str(k): v for k, v in ckpt.items()}, "td_pass": passed, "critic_ever_authorized": ever,
            "accepted": acc, "rejected": rej, "control_modes": list(CONTROL_MODES)}
