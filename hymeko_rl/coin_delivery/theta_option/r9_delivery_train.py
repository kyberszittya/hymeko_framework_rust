"""R9 STAGE 3 — per-decision delivery-focused TD3 over the bounded causal residual Δa.

Each control decision (every `control_interval` frames) is one MDP step: state = the 41-D causal state, action = the unit
Δa, reward = a K6-INDEPENDENT physical shaping accumulated over the decision's frames (+ a terminal shaping at the rollout
end). One rollout = one episode (~horizon/K decisions) on a development cradle; the frozen R8 champion supplies the base
`a_R8` per cradle and the warm-started policy's zero-mean Δ reproduces it. Reuses the framework network primitives
(`QNet`/`DetActor`/`_polyak`); nothing here reads the K6 boolean, injects a teacher, edits state, or adds a release bit.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, delivery_success
from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import (
    CAUSAL_DIM, DeltaBounds, R9CausalResidualAdapter)
from hymeko_rl.coin_delivery.theta_option.residual_adapter import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.option_rl.agents import DetActor, QNet, _polyak

ACT_DIM = 3
SPEED_CAP = 0.6                                  # soft coin-speed cap (well under the 1.5 hard motion limit)
CENTER_TOL = 0.02                                # the frozen 20 mm K6 zone (used ONLY for shaping geometry, not the K6 flag)


@dataclass(frozen=True)
class DeliveryReward:
    """K6-INDEPENDENT per-decision + terminal shaping. Curriculum A→C tunes the weights (never the K6 boolean)."""

    w_progress: float = 8.0        # + per-frame dtz reduction
    w_contact: float = 0.05        # + per-frame contact retained
    w_reversal: float = 2.0        # − per-frame coin v_par reversal (negative along-track velocity)
    w_overspeed: float = 1.0       # − per-frame coin speed above SPEED_CAP
    w_variation: float = 0.5       # − per-decision |Δa − Δa_prev| (anti-chatter)
    w_terminal: float = 6.0        # − terminal dtz_end (closer = higher)
    w_lowspeed: float = 3.0        # − terminal coin speed WHEN inside/near the zone (settle, don't fling)
    w_unsafe: float = 10.0         # − hard safety breach

    def decision(self, frames: list, dvar: float) -> float:
        r = -self.w_variation * float(dvar)
        for f in frames:
            r += (self.w_progress * f["ddtz"] + self.w_contact * f["contact"]
                  - self.w_reversal * max(0.0, -f["vpar"]) - self.w_overspeed * max(0.0, f["speed"] - SPEED_CAP))
        return r

    def terminal(self, m: dict, safe: bool) -> float:
        near = float(m["dtz_end"] <= 2.0 * CENTER_TOL)
        return (-self.w_terminal * float(m["dtz_end"]) - self.w_lowspeed * near * float(m["terminal_coin_speed"])
                - (0.0 if safe else self.w_unsafe))


@dataclass
class FlatReplay:
    cap: int = 40000
    _s: list = field(default_factory=list)
    _a: list = field(default_factory=list)
    _r: list = field(default_factory=list)
    _s2: list = field(default_factory=list)
    _d: list = field(default_factory=list)

    def add(self, s: Any, a: Any, r: float, s2: Any, d: float) -> None:
        self._s.append(np.asarray(s, np.float32))
        self._a.append(np.asarray(a, np.float32))
        self._r.append(float(r))
        self._s2.append(np.asarray(s2, np.float32))
        self._d.append(float(d))
        if len(self._s) > self.cap:
            for buf in (self._s, self._a, self._r, self._s2, self._d):
                del buf[0]

    def __len__(self) -> int:
        return len(self._s)

    def sample(self, n: int, rng: np.random.Generator) -> tuple:
        idx = rng.integers(0, len(self._s), n)
        t = lambda b: torch.as_tensor(np.asarray([b[i] for i in idx], np.float32))   # noqa: E731
        return (t(self._s), t(self._a), torch.as_tensor(np.asarray([self._r[i] for i in idx], np.float32)),
                t(self._s2), torch.as_tensor(np.asarray([self._d[i] for i in idx], np.float32)))


class _Collector:
    """Delta actor that queries a DetActor policy (with optional exploration) and RECORDS (causal_state, Δa) per decision."""

    def __init__(self, policy: Any, explore: float, rng: np.random.Generator) -> None:
        self.policy, self.explore, self.rng = policy, explore, rng
        self.records: list = []

    def __call__(self, cs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = self.policy(torch.as_tensor(np.asarray(cs, np.float32)[None]))[0].numpy()
        if self.explore > 0.0:
            a = np.clip(a + self.rng.normal(0, self.explore, ACT_DIM).astype(np.float32), -1, 1)
        self.records.append((np.asarray(cs, np.float32), np.asarray(a, np.float64)))
        return a


def _frame_signal(rl: Any, prev: dict) -> dict:
    u, dtz = rl.inner.direction_to_zone()
    vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    m = rl.inner._planar_metrics
    return {"dtz": float(dtz), "ddtz": float(prev["dtz"] - dtz), "vpar": float(np.dot(vel, np.asarray(u, np.float64)[:2])),
            "speed": float(_coin_speed(rl)), "contact": int(bool(m.left_contact or m.right_contact))}


def rollout_collect(snap: Any, a_r8: np.ndarray, policy: Any, dbounds: DeltaBounds, k: int, explore: float,
                    rng: np.random.Generator) -> tuple:
    """Run ONE rollout with the causal adapter driven by the recording policy; capture per-frame signals. Returns
    (per-decision records, per-frame signals, metrics)."""
    coll = _Collector(policy, explore, rng)
    adapter = R9CausalResidualAdapter(snap, a_r8, coll, TipTransportParams(), ResidualBounds(), dbounds,
                                      control_interval=k, cfg=DELIVERY_CFG)
    frames: list = []
    state = {"prev": None}

    def hook(rl: Any, t: int) -> None:
        u, dtz = rl.inner.direction_to_zone()
        if state["prev"] is None:
            state["prev"] = {"dtz": float(dtz)}
        f = _frame_signal(rl, state["prev"])
        state["prev"] = {"dtz": f["dtz"]}
        frames.append(f)

    m = velocity_rollout(snap, adapter, DELIVERY_CFG, frame_hook=hook)
    return coll.records, frames, m


def build_transitions(records: list, frames: list, m: dict, reward: DeliveryReward, k: int) -> list:
    """Decompose one collected rollout into per-decision TD3 transitions (s, Δa, r, s2, done). Terminal shaping is added to
    the last decision. Δa variation penalty uses consecutive decisions."""
    safe = bool(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0 and np.isfinite(m["dtz_end"]))
    trans, nd = [], len(records)
    for i, (cs, a) in enumerate(records):
        seg = frames[i * k:(i + 1) * k] if frames else []
        dvar = float(np.sum(np.abs(a - records[i - 1][1]))) if i > 0 else 0.0
        r = reward.decision(seg, dvar)
        last = i == nd - 1
        if last:
            r += reward.terminal(m, safe)
        s2 = records[i + 1][0] if not last else cs
        trans.append((cs, a, r, s2, float(last)))
    return trans


@dataclass
class R9TD3Config:
    gamma: float = 0.98
    lr: float = 3e-4
    batch: int = 128
    warmup_rollouts: int = 20
    total_rollouts: int = 400
    updates_per_rollout: int = 8
    eval_every: int = 50
    tau_polyak: float = 0.01
    policy_delay: int = 2
    target_noise: float = 0.1
    noise_clip: float = 0.2
    expl_noise: float = 0.15
    control_interval: int = 4
    reward_scale: float = 10.0


def distill_zero(policy: Any, cs_batch: np.ndarray, *, epochs: int = 300, lr: float = 1e-3, seed: int = 0) -> float:
    """Warm-start: policy mean Δ → 0 on sampled causal states ⇒ update-zero reproduces the frozen R8 champion."""
    torch.manual_seed(seed)
    x = torch.as_tensor(np.asarray(cs_batch, np.float32))
    opt = torch.optim.Adam(policy.parameters(), lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        loss = (policy(x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())


class _RecZero:
    """Records causal states, emits Δa = 0 (the R8-champion trajectory) — for warm-start cs + the replay seed."""

    def __init__(self) -> None:
        self.records: list = []

    def __call__(self, cs: np.ndarray) -> np.ndarray:
        self.records.append((np.asarray(cs, np.float32), np.zeros(ACT_DIM, np.float64)))
        return np.zeros(ACT_DIM, np.float64)


def _champion_pass(snap: Any, a_r8: np.ndarray, k: int) -> tuple:
    """One Δ=0 rollout = the R8 champion; return (records, frames, metrics) for cs + replay seeding (no exploration)."""
    rec = _RecZero()
    adapter = R9CausalResidualAdapter(snap, a_r8, rec, TipTransportParams(), ResidualBounds(), DeltaBounds(),
                                      control_interval=k, cfg=DELIVERY_CFG)
    frames: list = []
    state = {"prev": None}

    def hook(rl: Any, t: int) -> None:
        _u, dtz = rl.inner.direction_to_zone()
        if state["prev"] is None:
            state["prev"] = {"dtz": float(dtz)}
        f = _frame_signal(rl, state["prev"])
        state["prev"] = {"dtz": f["dtz"]}
        frames.append(f)
    m = velocity_rollout(snap, adapter, DELIVERY_CFG, frame_hook=hook)
    return rec.records, frames, m


def _seed_replay(dev: list, reward: DeliveryReward, cfg: R9TD3Config, replay: FlatReplay, rng: np.random.Generator) -> np.ndarray:
    """Seed replay with the R8-champion (Δ=0) transitions + bounded random-Δ perturbations; return cs for warm-start distil."""
    cs_all: list = []
    for _tag, snap, a_r8 in dev:
        rec, frames, m = _champion_pass(snap, a_r8, cfg.control_interval)
        for tr in build_transitions(rec, frames, m, reward, cfg.control_interval):
            replay.add(*[tr[0], tr[1], tr[2] / cfg.reward_scale, tr[3], tr[4]])
        cs_all += [r[0] for r in rec]
    return np.asarray(cs_all, np.float32)


def _collect(dev: list, policy: Any, reward: DeliveryReward, cfg: R9TD3Config, replay: FlatReplay, explore: float,
             rng: np.random.Generator) -> dict:
    """Collect ONE rollout on a random dev cradle, store per-decision transitions, return the rollout's delivery metrics."""
    tag, snap, a_r8 = dev[int(rng.integers(0, len(dev)))]
    rec, frames, m = rollout_collect(snap, a_r8, policy, DeltaBounds(), cfg.control_interval, explore, rng)
    for tr in build_transitions(rec, frames, m, reward, cfg.control_interval):
        replay.add(*[tr[0], tr[1], tr[2] / cfg.reward_scale, tr[3], tr[4]])
    return {"tag": tag, "dtz_end": float(m["dtz_end"]), "k6": bool(delivery_success(m, DELIVERY_CFG)),
            "peak_coin": float(m["peak_coin_speed"]), "peak_q": float(m["peak_qdot"])}


def _dev_eval(policy: Any, dev: list, cfg: R9TD3Config) -> dict:
    """Deterministic (Δa = policy mean, no exploration) delivery eval over the dev cradles. K6 is reported, NOT used to shape."""
    deliveries, dtz, safe = 0, [], 0
    for _tag, snap, a_r8 in dev:
        _rec, _fr, m = rollout_collect(snap, a_r8, policy, DeltaBounds(), cfg.control_interval, 0.0, np.random.default_rng(0))
        deliveries += int(bool(delivery_success(m, DELIVERY_CFG)))
        dtz.append(float(m["dtz_end"]))
        safe += int(m["peak_coin_speed"] <= 1.5 and m["peak_qdot"] <= 3.0)
    return {"k6": deliveries, "mean_dtz_mm": round(float(np.mean(dtz)) * 1000, 1), "safe": safe, "n": len(dev)}


def _td3_update(nets: dict, replay: FlatReplay, cfg: R9TD3Config, rng: np.random.Generator, upd: int) -> None:
    q1, q2, q1t, q2t, at, pol, qopt, aopt = (nets[k] for k in ("q1", "q2", "q1t", "q2t", "at", "pol", "qopt", "aopt"))
    bs, ba, br, bs2, bd = replay.sample(cfg.batch, rng)
    with torch.no_grad():
        noise = (torch.randn_like(ba) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
        a2 = (at(bs2) + noise).clamp(-1, 1)
        y = br + (1 - bd) * cfg.gamma * torch.min(q1t(bs2, a2), q2t(bs2, a2))
    ql = ((q1(bs, ba) - y) ** 2).mean() + ((q2(bs, ba) - y) ** 2).mean()
    qopt.zero_grad()
    ql.backward()
    qopt.step()
    if upd % cfg.policy_delay == 0:
        al = -q1(bs, pol(bs)).mean()
        aopt.zero_grad()
        al.backward()
        aopt.step()
        for tgt, src in ((q1t, q1), (q2t, q2), (at, pol)):
            _polyak(tgt, src, cfg.tau_polyak)


def train_causal_td3(dev: list, reward: DeliveryReward, cfg: R9TD3Config, *, seed: int = 0, log: Any = print) -> tuple:
    """Per-decision TD3 over the causal residual on the DEV cradles. `dev` = [(tag, snap, a_R8)]. Warm-starts Δ→0 (update-0
    = R8 champion), seeds replay with champion + perturbation experience, then trains. Returns (checkpoints, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pol = DetActor(CAUSAL_DIM, ACT_DIM)
    replay = FlatReplay()
    cs_seed = _seed_replay(dev, reward, cfg, replay, rng)
    distill_loss = distill_zero(pol, cs_seed, seed=seed)
    at = DetActor(CAUSAL_DIM, ACT_DIM)
    at.load_state_dict(pol.state_dict())
    q1, q2 = QNet(CAUSAL_DIM, ACT_DIM), QNet(CAUSAL_DIM, ACT_DIM)
    q1t, q2t = QNet(CAUSAL_DIM, ACT_DIM), QNet(CAUSAL_DIM, ACT_DIM)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    nets = {"q1": q1, "q2": q2, "q1t": q1t, "q2t": q2t, "at": at, "pol": pol,
            "qopt": torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), cfg.lr),
            "aopt": torch.optim.Adam(pol.parameters(), cfg.lr)}
    ckpts = {"update0": copy.deepcopy(pol.state_dict()), "best_val": copy.deepcopy(pol.state_dict())}
    hist, best, upd, recent = [], -1e9, 0, []
    for it in range(cfg.total_rollouts):
        expl = cfg.expl_noise if it >= cfg.warmup_rollouts else 0.4
        recent.append(_collect(dev, pol, reward, cfg, replay, expl, rng))
        if len(replay) >= cfg.batch and it >= cfg.warmup_rollouts:
            for _ in range(cfg.updates_per_rollout):
                upd += 1
                _td3_update(nets, replay, cfg, rng, upd)
        if (it + 1) % cfg.eval_every == 0 or it == cfg.total_rollouts - 1:
            ev = _dev_eval(pol, dev, cfg)
            rk = sum(r["k6"] for r in recent[-cfg.eval_every:])
            hist.append({"it": it + 1, **ev, "train_k6_recent": rk})
            log(f"    [r9-td3 it {it+1}/{cfg.total_rollouts}] dev K6 {ev['k6']}/{ev['n']} dtz {ev['mean_dtz_mm']}mm "
                f"safe {ev['safe']} | train_k6 {rk} | replay {len(replay)}")
            score = ev["k6"] - ev["mean_dtz_mm"] / 1000.0                # K6 first, then closeness (dev selection)
            if score > best:
                best = score
                ckpts["best_val"] = copy.deepcopy(pol.state_dict())
    ckpts["final"] = copy.deepcopy(pol.state_dict())
    return ckpts, {"history": hist, "distill_loss": round(distill_loss, 6)}

