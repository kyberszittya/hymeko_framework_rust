"""R10.2 Stage 2 (Boundary 4b) — conservative structured-option TD3 over the frozen HOME-start capture env.

The training protocol the Boundary-3 review mandated, reusing the ``option_rl`` primitives (``QNet``, ``DetActor``,
``OptionReplayBuffer``, ``smdp_target``, ``_polyak``) rather than the vanilla ``train_semi_mdp`` loop (which has no
warm-up / ranking gate / immutable-positive replay):

  1. **Zero-init actor** — the policy starts AT the scaffold (``theta = 0``), so exploration ``theta = clip(actor(obs)
     + sigma*D*z, -1, 1)`` explores around the proven delivery, not a random policy.
  2. **Immutable positive replay** — the medoid ``theta=0`` K6 deliveries + any K6 exploration transitions are kept in a
     separate never-evicted buffer; the minibatch reserves a positive fraction so the critic cannot collapse to an
     "everything is bad" function (this does NOT falsify the environmental frequency — it only keeps positives visible).
  3. **Critic-only warm-up** — the critics regress the (terminal) reward with the actor frozen at the scaffold.
  4. **Option-ranking gate** — the actor is released ONLY when the critic ranks ``Q(corrective) > Q(medoid) >
     Q(destructive)`` on the frozen paired READY panel; otherwise the run stops without an actor update.
  5. **Conservative actor release** — TD3 (delayed, smoothed) updates over the frozen episode budget.

Each option is terminal, so ``smdp_target`` reduces to the reward (a contextual-bandit critic). Deterministic given the
seed. No cradle-centred ``tau*`` anywhere; the reward lives in :mod:`torque_path_env`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option.kinetic_authority_unlock import zero_init_detactor
from hymeko_rl.coin_delivery.theta_option.torque_path_env import StructuredOptionCaptureEnv
from hymeko_rl.coin_delivery.theta_option.torque_path_option import THETA_DIM
from hymeko_rl.option_rl.agents import DetActor, QNet, _polyak, make_actor
from hymeko_rl.option_rl.core import OptionReplayBuffer, OptionTransition, smdp_target

OBS_DIM = 31


@dataclass
class TD3Config:
    total_episodes: int = 400
    warmup_episodes: int = 40          # exploration-only collection before any update
    critic_warmup_updates: int = 120   # critic-only updates before the ranking gate (must be reachable in the budget)
    eval_every: int = 25
    batch: int = 64
    positive_frac: float = 0.25        # minimum minibatch fraction drawn from the immutable positive buffer
    lr: float = 3e-4
    gamma: float = 0.99
    polyak: float = 0.01
    target_noise: float = 0.1
    noise_clip: float = 0.2
    updates_per_episode: int = 2
    reward_scale: float = 10.0
    policy_delay: int = 2
    rank_thetas: int = 6               # exploration thetas per panel state for the ranking gate
    rank_spearman_min: float = 0.3     # a meaningfully-positive Q-vs-reward rank correlation (sanity beside bucket order)


@dataclass
class _Nets:
    q1: QNet
    q2: QNet
    q1t: QNet
    q2t: QNet
    actor: DetActor
    at: DetActor
    qopt: Any = None
    aopt: Any = None


def _build_nets(cfg: TD3Config, seed: int) -> _Nets:
    torch.manual_seed(seed)
    actor = zero_init_detactor(make_actor("td3", OBS_DIM, THETA_DIM))     # policy starts AT the scaffold (theta=0)
    q1, q2 = QNet(OBS_DIM, THETA_DIM), QNet(OBS_DIM, THETA_DIM)
    q1t, q2t = QNet(OBS_DIM, THETA_DIM), QNet(OBS_DIM, THETA_DIM)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    at = zero_init_detactor(make_actor("td3", OBS_DIM, THETA_DIM))
    at.load_state_dict(actor.state_dict())
    nets = _Nets(q1, q2, q1t, q2t, actor, at)
    nets.qopt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), cfg.lr)
    nets.aopt = torch.optim.Adam(actor.parameters(), cfg.lr)
    return nets


def _act(actor: DetActor, obs: np.ndarray, d_norm: np.ndarray, sigma: float, rng: np.random.Generator,
         explore: bool) -> np.ndarray:
    with torch.no_grad():
        mean = actor(torch.as_tensor(obs[None]).float())[0].numpy()
    if explore:
        mean = mean + sigma * d_norm * rng.standard_normal(THETA_DIM)
    return np.clip(mean, -1.0, 1.0).astype(np.float32)


def _transition(obs: np.ndarray, action: np.ndarray, obs2: np.ndarray, r: float, info: dict,
                scale: float) -> OptionTransition:
    return OptionTransition(s=obs, action=np.asarray(action, np.float32), reward=r / scale, tau=float(info["tau"]),
                            s_next=obs2, terminal=1.0, end="handoff",
                            provenance={"class": info["class"], "k6": info["k6"], "min_dtz": info["min_dtz"]})


def _seed_positives(env: StructuredOptionCaptureEnv, positives: OptionReplayBuffer, cfg: TD3Config) -> int:
    """Anchor the immutable positive buffer with the medoid (theta=0) K6 deliveries across the training states."""
    n = 0
    zero = np.zeros(THETA_DIM, dtype=np.float32)
    for idx in range(len(env)):
        obs = env.reset(idx)
        obs2, r, _, info = env.step(zero)
        if info["class"] == "k6":
            positives.add(_transition(obs, zero, obs2, r, info, cfg.reward_scale))
            n += 1
    return n


def _sample_mixed(main: OptionReplayBuffer, positives: OptionReplayBuffer, cfg: TD3Config,
                  rng: np.random.Generator) -> tuple:
    npos = min(len(positives), int(cfg.batch * cfg.positive_frac)) if len(positives) else 0
    nmain = cfg.batch - npos
    bm = main.sample(nmain, rng)
    if npos == 0:
        return bm
    bp = positives.sample(npos, rng)
    return tuple(torch.cat([m, p], 0) for m, p in zip(bm, bp))


def _critic_step(nets: _Nets, batch: tuple, cfg: TD3Config) -> float:
    bs, ba, br, bt, bs2, bd = batch
    with torch.no_grad():
        noise = (torch.randn_like(ba) * cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
        a2 = (nets.at(bs2) + noise).clamp(-1, 1)
        q_next = torch.min(nets.q1t(bs2, a2), nets.q2t(bs2, a2))
        y = smdp_target(br, cfg.gamma, bt, bd, q_next)
    ql = ((nets.q1(bs, ba) - y) ** 2).mean() + ((nets.q2(bs, ba) - y) ** 2).mean()
    nets.qopt.zero_grad()
    ql.backward()
    nets.qopt.step()
    return float(ql.item())


def _actor_step(nets: _Nets, bs: torch.Tensor, cfg: TD3Config) -> None:
    al = -nets.q1(bs, nets.actor(bs)).mean()
    nets.aopt.zero_grad()
    al.backward()
    nets.aopt.step()
    for tgt, src in ((nets.q1t, nets.q1), (nets.q2t, nets.q2), (nets.at, nets.actor)):
        _polyak(tgt, src, cfg.polyak)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _q(nets: _Nets, obs: np.ndarray, theta: np.ndarray) -> float:
    with torch.no_grad():
        return float(nets.q1(torch.as_tensor(obs[None]).float(), torch.as_tensor(theta[None]).float())[0])


def ranking_gate(nets: _Nets, env: StructuredOptionCaptureEnv, d_norm: np.ndarray, sigma: float,
                 rng: np.random.Generator, cfg: TD3Config, rel_eps: float = 0.05) -> dict:
    """Actor-release precondition: the critic must rank options by their ACTUAL outcome from the same READY state.

    Primary criterion is ``spearman(Q, reward) >= rank_spearman_min`` over all (state, theta) probes (robust when the
    scaffold is already near-optimal so K6 'corrective' samples are rare). Secondary, reported: ``Q(corrective) >
    Q(medoid) > Q(destructive)`` where corrective/destructive are defined *relative to the per-state medoid reward*
    (an option that improves / degrades vs theta=0 on that state)."""
    zero = np.zeros(THETA_DIM, dtype=np.float32)
    qs, rs, q_corr, q_med, q_dest = [], [], [], [], []
    for idx in range(len(env)):
        obs = env.reset(idx)
        env.reset(idx)
        r_med = env.step(zero)[1]
        q_m = _q(nets, obs, zero)
        qs.append(q_m)
        rs.append(r_med)
        q_med.append(q_m)
        for _ in range(cfg.rank_thetas):
            theta = np.clip(sigma * d_norm * rng.standard_normal(THETA_DIM), -1, 1).astype(np.float32)
            env.reset(idx)
            r = env.step(theta)[1]
            q = _q(nets, obs, theta)
            qs.append(q)
            rs.append(r)
            if r > r_med + rel_eps:
                q_corr.append(q)          # improves vs the per-state medoid (theta=0)
            elif r < r_med - rel_eps:
                q_dest.append(q)          # degrades vs the medoid
    sp = _spearman(np.array(qs), np.array(rs))
    mc = float(np.mean(q_corr)) if q_corr else float(np.mean(q_med))
    md = float(np.mean(q_dest)) if q_dest else float(np.mean(q_med)) - 1.0
    mm = float(np.mean(q_med))
    bucket_order_ok = bool(mc >= mm >= md)
    return {"spearman_q_vs_reward": round(sp, 3), "q_corrective": round(mc, 3), "q_medoid": round(mm, 3),
            "q_destructive": round(md, 3), "n_corrective": len(q_corr), "n_destructive": len(q_dest),
            "bucket_order_ok": bucket_order_ok,
            "passed": bool(bucket_order_ok and sp >= cfg.rank_spearman_min)}


def eval_actor(actor: DetActor, env: StructuredOptionCaptureEnv) -> dict:
    """Deterministic (no-exploration) evaluation of ``actor``'s mean option on every panel state. State 0 is nominal."""
    k6, safe, boundary, dtz, nominal_k6 = 0, 0, 0, [], False
    for idx in range(len(env)):
        obs = env.reset(idx)
        with torch.no_grad():
            theta = np.clip(actor(torch.as_tensor(obs[None]).float())[0].numpy(), -1, 1).astype(np.float32)
        _, _, _, info = env.step(theta)
        k6 += int(info["class"] == "k6")
        safe += int(info["safe"])
        boundary += int(info["boundary_route_variation"])
        dtz.append(info["min_dtz"])
        nominal_k6 = nominal_k6 or (idx == 0 and info["class"] == "k6")
    n = len(env)
    return {"n": n, "k6": k6, "k6_rate": round(k6 / n, 3), "safe": safe, "boundary": boundary,
            "mean_min_dtz": round(float(np.mean(dtz)), 2), "nominal_k6": bool(nominal_k6)}


def _learn_updates(nets: _Nets, main: OptionReplayBuffer, positives: OptionReplayBuffer, cfg: TD3Config,
                   rng: np.random.Generator, released: bool, upd: int) -> int:
    """One episode's worth of gradient steps: critics always; the actor (TD3-delayed) only after release. Returns upd."""
    for _ in range(cfg.updates_per_episode):
        upd += 1
        batch = _sample_mixed(main, positives, cfg, rng)
        _critic_step(nets, batch, cfg)
        if released and upd % cfg.policy_delay == 0:
            _actor_step(nets, batch[0], cfg)
    return upd


def train_seed(rig: dict, train_perts: list, eval_perts: list, seed: int, cfg: TD3Config,
               d_norm: np.ndarray, sigma: float, log: Any = print) -> dict:
    """One conservative TD3 seed: zero-init actor -> critic-only warm-up -> ranking gate -> (release) -> eval/25.

    # Postconditions: deterministic given ``seed``; the actor is released ONLY after the ranking gate passes; returns the
    #   history, the gate result, and the pre/post 3-way panel evals."""
    env = StructuredOptionCaptureEnv(rig, train_perts)
    eval_env = StructuredOptionCaptureEnv(rig, eval_perts)
    rng = np.random.default_rng(seed)
    nets = _build_nets(cfg, seed)
    main, positives = OptionReplayBuffer(), OptionReplayBuffer()
    n_anchor = _seed_positives(env, positives, cfg)
    scaffold_eval = eval_actor(zero_init_detactor(make_actor("td3", OBS_DIM, THETA_DIM)), eval_env)

    history, upd, released, gate = [], 0, False, None
    obs = env.reset()
    for it in range(cfg.total_episodes):
        a = _act(nets.actor, obs, d_norm, sigma, rng, explore=True)
        obs2, r, _, info = env.step(a)
        tr = _transition(obs, a, obs2, r, info, cfg.reward_scale)
        main.add(tr)
        if info["class"] == "k6":
            positives.add(tr)
        obs = env.reset()
        if len(main) >= cfg.batch and it >= cfg.warmup_episodes:
            upd = _learn_updates(nets, main, positives, cfg, rng, released, upd)
            if not released and upd >= cfg.critic_warmup_updates:
                gate = ranking_gate(nets, eval_env, d_norm, sigma, rng, cfg)
                released = gate["passed"]
                log(f"    [seed {seed}] ranking gate @upd {upd}: {gate}")
        if (it + 1) % cfg.eval_every == 0 or it == cfg.total_episodes - 1:
            ev = eval_actor(nets.actor, eval_env)
            history.append({"it": it + 1, "released": released, "eval": ev})
            log(f"    [seed {seed} it {it+1}] released={released} eval={ev}")
    return {"seed": seed, "anchor_positives": n_anchor, "scaffold_eval": scaffold_eval, "gate": gate,
            "released": released, "history": history, "post_eval": eval_actor(nets.actor, eval_env)}
