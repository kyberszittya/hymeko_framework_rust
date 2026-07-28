"""R2 — per-step state-dependent bounded-residual RL over the frozen clone: reward, collector, matched TD3/SAC, champion.

The reward rewards LIGHT-CONTACT FOLLOWING, not contact duration or raw speed (either would bring back clamping / overshoot):

    r_follow = 1[contact] · 1[Fn < Fn_light] · max(v_par, 0) · Δd_toward_zone

with SEPARATE penalties (heavy clamp, coin slowdown, negative v_par, sign reversal, safety) and a STATE-DEPENDENT contact-loss
term — losing contact far from the zone is punished, releasing close-and-moving is not (else the policy learns to never let go).
Each KINETIC step is one transition; the episode is terminal at the KINETIC exit (the frozen coast/K6 is downstream). Training
is a compact per-step off-policy loop reusing the framework's `DetActor`/`GaussActor`/`QNet`; the champion is chosen
LEXICOGRAPHICALLY (safety ≻ K6 ≻ close+moving release ≻ contact-exit dtz ≻ landing ≻ residual smoothness), never the Q-value or
the raw reward, and the BEST checkpoint is kept (not the last).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    AUG_DIM, KineticTemporalResidualController, deterministic_residual, exploring_residual)
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.option_rl.agents import DetActor, QNet, _polyak, make_actor

FN_LIGHT = 2.0                                  # N — per-side normal force below this counts as light contact (reward-eligible)
FN_CLAMP = 4.0                                  # N — above this is a heavy clamp (penalised)
CORRIDOR_MM = 30.0                              # close+moving release corridor upper edge
RELEASE_VMIN = 0.05                             # m/s — "moving" at release
REWARD_SCALE = 20.0


@dataclass(frozen=True)
class Reward2Weights:
    follow: float = 1.0
    corridor: float = 30.0
    k6: float = 100.0
    release: float = 20.0
    clamp: float = 5.0
    slow: float = 10.0
    neg: float = 10.0
    reversal: float = 5.0
    early: float = 1.0                          # per mm of contact-exit above the corridor (state-dependent contact loss)
    safety: float = 60.0


def reward2(kin: list[dict], entry_dtz: float, min_dtz: float, k6: bool, safe: bool,
            w: Reward2Weights = Reward2Weights()) -> "tuple[list[float], dict]":
    """Per-step rewards over the KINETIC steps + a decomposition. # Preconditions: ``kin`` the KINETIC-clone steps (dtz/v_par/
    fn per step, oldest first); ``entry_dtz`` the dtz at KINETIC entry. # Postconditions: len(rewards) == len(kin); reward is a
    pure function of physical signals (light-contact progress, NOT raw v_par or contact duration)."""
    rewards: list[float] = []
    prev_dtz, prev_vpar = entry_dtz, None
    tot = {"follow": 0.0, "clamp": 0.0, "slow": 0.0, "neg": 0.0, "reversal": 0.0}
    for s in kin:
        dtz, vpar, fn = s["dtz_mm"], s["v_par"], min(s["fn_l"], s["fn_r"])
        ddtz = max(0.0, prev_dtz - dtz)
        follow = w.follow * (1.0 if (fn > 0.05 and fn < FN_LIGHT) else 0.0) * max(vpar, 0.0) * ddtz
        clamp = w.clamp if fn > FN_CLAMP else 0.0
        slow = w.slow * max(0.0, prev_vpar - vpar) if prev_vpar is not None else 0.0
        neg = w.neg if vpar < 0.0 else 0.0
        rev = w.reversal if (prev_vpar is not None and vpar * prev_vpar < 0.0) else 0.0
        rewards.append(follow - clamp - slow - neg - rev)
        for k, v in (("follow", follow), ("clamp", clamp), ("slow", slow), ("neg", neg), ("reversal", rev)):
            tot[k] += v
        prev_dtz, prev_vpar = dtz, vpar
    exit_dtz = kin[-1]["dtz_mm"] if kin else 999.0
    released = bool(exit_dtz <= CORRIDOR_MM and kin and kin[-1]["v_par"] >= RELEASE_VMIN)
    term = ((w.corridor if min_dtz <= CORRIDOR_MM else 0.0) + (w.k6 if (k6 and safe) else 0.0)
            + (w.release if released else 0.0) - w.early * max(0.0, exit_dtz - CORRIDOR_MM)
            - (w.safety if not safe else 0.0))
    if rewards:
        rewards[-1] += term
    return rewards, {**{k: round(v, 2) for k, v in tot.items()}, "terminal": round(term, 2),
                     "exit_dtz_mm": round(exit_dtz, 2), "released": released}


@dataclass
class Episode:
    transitions: list                            # (s, a, r, s2, done)
    min_dtz: float
    exit_dtz: float
    k6: bool
    safe: bool
    residuals: np.ndarray                        # (T, ACT_DIM) executed residuals (for state-dependence / smoothness)
    decomp: dict


def collect_episode(snap: Any, clone: CloneActor, residual_fn: Callable[[np.ndarray], np.ndarray],
                    bounds: ResidualBounds, w: Reward2Weights, cfg: Any = DELIVERY_CFG) -> Episode:
    """Run one full frozen-chain rollout with the per-step residual and build per-step transitions + the physical outcome."""
    controller = KineticTemporalResidualController(snap, clone, residual_fn, bounds)
    m = velocity_rollout(snap, controller, cfg)
    kin = [r for r in controller.clone_trace if r["kind"] == "KINETIC_CLONE"]
    aug = controller.aug_trace
    entry_dtz = kin[0]["dtz_mm"] if kin else 999.0
    min_dtz = _min_dtz_mm(snap, m)
    safe = bool(m["peak_qdot"] <= 3.0 and m["peak_coin_speed"] <= 1.5)
    rewards, decomp = reward2(kin, entry_dtz, min_dtz, bool(m["k6_delivered"]), safe, w)
    trans = []
    for t in range(len(aug)):
        s, a = aug[t]
        s2 = aug[t + 1][0] if t + 1 < len(aug) else np.zeros(AUG_DIM, np.float64)
        trans.append((s.astype(np.float32), a.astype(np.float32), float(rewards[t]) / REWARD_SCALE,
                      s2.astype(np.float32), 1.0 if t == len(aug) - 1 else 0.0))
    return Episode(trans, min_dtz, decomp["exit_dtz_mm"], bool(m["k6_delivered"]), safe,
                   np.array([r for _s, r in aug], np.float64) if aug else np.zeros((0, ACT_DIM)), decomp)


def champion_key(ep_min_dtz: float, exit_dtz: float, k6: bool, safe: bool, released: bool, jerk: float) -> tuple:
    """LEXICOGRAPHIC champion (higher is better): safety ≻ K6 ≻ close+moving release ≻ contact-exit dtz ≻ landing ≻ smoothness.
    Never the Q-value or the raw reward."""
    return (int(safe), int(k6), int(released), -round(exit_dtz, 2), -round(ep_min_dtz, 2), -round(jerk, 4))


class _Replay:
    """A compact flat FIFO replay for the per-step residual transitions."""

    def __init__(self, cap: int = 40000) -> None:
        self.cap, self.buf = cap, []

    def add(self, tr: tuple) -> None:
        self.buf.append(tr)
        if len(self.buf) > self.cap:
            self.buf.pop(0)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, n: int, rng: np.random.Generator) -> tuple:
        idx = rng.integers(0, len(self.buf), n)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (torch.as_tensor(np.array(s)), torch.as_tensor(np.array(a)), torch.as_tensor(np.array(r), dtype=torch.float32),
                torch.as_tensor(np.array(s2)), torch.as_tensor(np.array(d), dtype=torch.float32))


@dataclass
class PerStepConfig:
    gamma: float = 0.97
    tau_polyak: float = 0.01
    lr: float = 3e-4
    batch: int = 128
    warmup_options: int = 20
    total_options: int = 240
    updates_per_option: int = 8
    eval_every: int = 30
    policy_delay: int = 2
    target_noise: float = 0.15
    noise_clip: float = 0.3
    expl_noise: float = 0.25
    alpha: float = 0.1


def _dev_eval(snap: Any, clone: CloneActor, bounds: ResidualBounds, w: Reward2Weights, cfg: Any) -> Callable[[Any], tuple]:
    def ev(actor: Any) -> "tuple[tuple, dict]":
        ep = collect_episode(snap, clone, deterministic_residual(actor), bounds, w, cfg)
        jerk = float(np.mean(np.abs(np.diff(ep.residuals, axis=0)))) if len(ep.residuals) > 1 else 0.0
        key = champion_key(ep.min_dtz, ep.exit_dtz, ep.k6, ep.safe, ep.decomp["released"], jerk)
        return key, {"min_dtz": round(ep.min_dtz, 2), "exit_dtz": round(ep.exit_dtz, 2), "k6": ep.k6,
                     "residual_std": round(float(ep.residuals.std()), 4) if len(ep.residuals) else 0.0, "jerk": round(jerk, 4)}
    return ev


def train_perstep(algo: str, snap: Any, clone: CloneActor, bounds: ResidualBounds, w: Reward2Weights,
                  cfg: PerStepConfig, *, seed: int = 0, warm_actor: Any = None, cfg_env: Any = DELIVERY_CFG,
                  log: Callable = print, collect_override: "Callable | None" = None,
                  champion_override: "Callable | None" = None, stop_when: "Callable | None" = None) -> "tuple[Any, list]":
    """Compact matched per-step TD3/SAC over δ_ψ. Reuses `DetActor`/`GaussActor`/`QNet`/`_polyak`; keeps the BEST checkpoint by
    the lexicographic champion. ``collect_override(rfn) -> transitions`` and ``champion_override(actor) -> (key, aux)`` let a
    curriculum (R3-B) inject its own episode source + champion without duplicating the update loop (defaults = the R2 behaviour).
    ``stop_when(key, aux) -> bool`` (checked at each eval) requests an EARLY STOP once the target is met (R3-C: the first strict
    learned K6) so the champion is frozen immediately without burning the rest of the budget. # Postconditions: returns
    (best_actor, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = warm_actor if warm_actor is not None else make_actor(algo, AUG_DIM, ACT_DIM)
    q1, q2 = QNet(AUG_DIM, ACT_DIM), QNet(AUG_DIM, ACT_DIM)
    q1t, q2t = QNet(AUG_DIM, ACT_DIM), QNet(AUG_DIM, ACT_DIM)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    at = DetActor(AUG_DIM, ACT_DIM) if algo == "td3" else None
    if at is not None:
        at.load_state_dict(actor.state_dict())
    qopt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), cfg.lr)
    aopt = torch.optim.Adam(actor.parameters(), cfg.lr)
    replay = _Replay()
    ev = champion_override if champion_override is not None else _dev_eval(snap, clone, bounds, w, cfg_env)
    best_key, best_actor, history, upd = None, copy.deepcopy(actor), [], 0
    for it in range(cfg.total_options):
        rfn = (exploring_residual(actor, algo, cfg.expl_noise, rng) if it >= cfg.warmup_options
               else (lambda _a: np.clip(rng.uniform(-1, 1, ACT_DIM), -1, 1).astype(np.float32)))
        trans = collect_override(rfn) if collect_override is not None else collect_episode(snap, clone, rfn, bounds, w, cfg_env).transitions
        for tr in trans:
            replay.add(tr)
        if len(replay) >= cfg.batch and it >= cfg.warmup_options:
            for _ in range(cfg.updates_per_option):
                upd += 1
                bs, ba, br, bs2, bd = replay.sample(cfg.batch, rng)
                with torch.no_grad():
                    if algo == "sac":
                        a2, logp2 = actor.sample(bs2)
                        qn = torch.min(q1t(bs2, a2), q2t(bs2, a2)) - cfg.alpha * logp2
                    else:
                        noise = (torch.randn_like(ba) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
                        a2 = (at(bs2) + noise).clamp(-1, 1)
                        qn = torch.min(q1t(bs2, a2), q2t(bs2, a2))
                    y = br + cfg.gamma * (1.0 - bd) * qn
                ql = ((q1(bs, ba) - y) ** 2).mean() + ((q2(bs, ba) - y) ** 2).mean()
                qopt.zero_grad()
                ql.backward()
                qopt.step()
                if algo == "sac":
                    ap, logp = actor.sample(bs)
                    al = (cfg.alpha * logp - torch.min(q1(bs, ap), q2(bs, ap))).mean()
                    aopt.zero_grad()
                    al.backward()
                    aopt.step()
                    _polyak(q1t, q1, cfg.tau_polyak)
                    _polyak(q2t, q2, cfg.tau_polyak)
                elif upd % cfg.policy_delay == 0:
                    al = -q1(bs, actor(bs)).mean()
                    aopt.zero_grad()
                    al.backward()
                    aopt.step()
                    _polyak(q1t, q1, cfg.tau_polyak)
                    _polyak(q2t, q2, cfg.tau_polyak)
                    _polyak(at, actor, cfg.tau_polyak)
        if (it + 1) % cfg.eval_every == 0 or it == cfg.total_options - 1:
            key, aux = ev(actor)
            history.append({"it": it + 1, "champion": list(key), "aux": aux})
            log(f"    [{algo} it {it+1}/{cfg.total_options}] champ {key} {aux} | replay {len(replay)}")
            if best_key is None or key > best_key:
                best_key, best_actor = key, copy.deepcopy(actor)
            if stop_when is not None and stop_when(key, aux):        # target met (R3-C first strict K6) ⇒ freeze immediately
                history.append({"it": it + 1, "early_stop": True, "aux": aux})
                break
    return best_actor, history
