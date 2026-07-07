"""Root-cause probe for the off-policy Q-term collapse (k-arm coin toss, 2026-07-05).

Every off-policy continuation of the BC clone degrades it: delivery decays to ~0 and ``both_contact`` decays in
lockstep — the actor drifts OFF the contact manifold. The overnight session isolated that the actor's
Q-maximization term (``-Q(s, mu(s))``, ddpg.py:360) reduces delivery under *every* BC-anchor coefficient, but
did NOT isolate *why* among competing hypotheses. Asserting one without a discriminating test is the
"most-likely cause in a lab coat" this codebase forbids (CLAUDE.md operating principles).

This module builds the discriminating test. It FREEZES the BC clone ``mu0`` (which removes the online
replay-drift feedback loop *by construction*) and interrogates the critic that policy-evaluation produces on
the clone's own on-distribution data:

  * **M1 rank fidelity** — Spearman(``Q_hat(s, mu0(s))``, MC return-to-go ``G(s)``). Low ⇒ the critic cannot
    even value the clone (**H_fit**).
  * **M3 one clean improvement phase** — ascend ``Q_hat`` on a copy ``mu1`` of ``mu0`` against the *frozen*,
    well-fit critic (no online buffer drift, no reward-norm nonstationarity). Then compare, at the policy
    level, the Q-predicted change against the *true* MC return change and the delivery change:
      - true return DROPS while Q_hat rose ⇒ the gradient climbed a phantom ⇒ **H_ood** (OOD overestimation);
      - true return holds/rises but delivery DROPS ⇒ **H_reward** candidate (reward/metric mismatch to audit);
      - both hold/rise ⇒ the operator is fine offline; the collapse is the online loop ⇒ **H_shift**.
  * **M2 off-manifold Q inflation** (confirmatory) — mean over on-clone states of ``Q_hat(mu0+delta) - Q_hat(mu0)``
    for random ``delta``. Positive ⇒ the value surface *rewards leaving* the clone action = the OOD phantom.

The critic-fit replicates the trainer's Bellman backup term-for-term (clipped double-Q + target smoothing +
``reward_norm`` + gamma; ddpg.py:351-357), with ONE deliberate, documented difference: the target critics are
polyak-updated *every* critic step. The trainer gates critic-target polyak behind the actor-update branch
(ddpg.py:435-437), so a plain ``critic_warmup`` freeze would bootstrap off a random target and would NOT be a
valid policy-evaluation. Here the actor is frozen but the target still tracks — faithful fitted-Q evaluation.

CLI (self-contained run dir + verdict.json + run.log):
    python -m hymeko_rl.eval.critic_probe --clone experiments/.../policies/..._s0.pt
    python -m hymeko_rl.eval.critic_probe --smoke        # fast path check
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.evaluate import experiment_dir, eval_metric, greedy_action_fn, DwellMetric
from hymeko_rl.train.normalize import RunningRMS
from hymeko_rl.train.replay import ReplayBuffer

_DELIVER_HYMEKO = "data/robotics/galambos_task_deliver.hymeko"
_MAX_STEPS = 300
_EVAL_SEED = 9_000


@dataclass(frozen=True)
class ProbeConfig:
    """Knobs for the frozen-clone critic probe. Defaults mirror the failing campaign (``sa_hsikan`` CTDE actor,
    twin LayerNorm critics, TD3 backup). ``smoke`` shrinks every budget to a path check.

    # Preconditions all step/size counts >= 1; ``0 < ood_eps``; ``0 < gamma < 1``.
    """

    difficulty: float = 0.3
    buffer_steps: int = 20_000     # on-clone transitions collected for fitted-Q evaluation
    fit_updates: int = 20_000      # critic SGD steps against the fixed on-clone buffer
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    critic_lr: float = 1e-3
    max_grad_norm: float = 10.0    # critic grad-norm clip (0 = off). Mirrors the trainer (ddpg.py max_grad_norm)
                                   # so the frozen-policy fit is a FAITHFUL diagnosis, not an under-stabilised one.
    noise_scale: float = 0.1       # on-clone exploration sigma (fraction of action_scale) — matches trainer
    target_noise: float = 0.2      # target-policy smoothing sigma (TD3) — matches trainer
    noise_clip: float = 0.5
    reward_norm: bool = True
    n_critics: int = 2
    improve_steps: int = 300       # clean Q-ascent steps on mu1 against the FROZEN critic
    improve_lr: float = 1e-3
    mc_traj: int = 30              # trajectories rolled for MC return-to-go (M1) and true-return (M3)
    mc_states: int = 512           # subsample of (obs, G) pairs scored in M1
    ood_eps: float = 0.15          # off-manifold perturbation (fraction of action_scale) for M2
    ood_dirs: int = 8
    n_eval: int = 50               # delivery-eval episodes for mu0 / mu1
    log_every: int = 4_000
    seed: int = 0
    smoke: bool = False

    def resolved(self) -> "ProbeConfig":
        """Apply the ``smoke`` cap (a path check, not experiment identity). # Postconditions under smoke:
        <=1500 buffer, <=500 fit updates, <=40 improve, <=6 eval."""
        if not self.smoke:
            return self
        return ProbeConfig(**{**asdict(self),
                              "buffer_steps": min(self.buffer_steps, 1_500),
                              "fit_updates": min(self.fit_updates, 500),
                              "improve_steps": min(self.improve_steps, 40),
                              "mc_traj": min(self.mc_traj, 4),
                              "mc_states": min(self.mc_states, 128),
                              "n_eval": min(self.n_eval, 6),
                              "log_every": 200})


@dataclass(frozen=True)
class ProbeVerdict:
    """The discriminating measurements + the hypothesis they support. Every field is a measured number."""

    clone_delivery: float          # deliver(mu0)
    critic_final_loss: float       # fitted-Q evaluation loss at convergence (finite ⇒ fit did not diverge)
    rank_spearman: float           # M1: Spearman(Q_hat(s, mu0(s)), G(s))
    q_pred_rise: float             # M3: Q_hat(mu1) - Q_hat(mu0) over the ascent batch (>=0 by construction)
    true_return_delta: float       # M3: MC return(mu1) - MC return(mu0) (raw deliver reward)
    delivery_delta: float          # M3: deliver(mu1) - deliver(mu0)
    ood_q_inflation: float         # M2: mean_s mean_delta [Q_hat(mu0+delta) - Q_hat(mu0)]  (>0 ⇒ phantom)
    diagnosis: str                 # which hypothesis the numbers support
    hypothesis: str                # one of H_fit / H_ood / H_reward / H_shift


def make_deliver_env(difficulty: float) -> PlanarGraspEnv:
    """The production coin-toss env with the ORACLE-CERTIFIED delivering reward set as the training reward (the
    reward whose Q-term collapses). Delivery is still graded by the reward-independent dwell metric."""
    env = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=difficulty)
    env.reward_spec = RewardSpec.from_hymeko(_DELIVER_HYMEKO)
    return env


def _load_clone(env: PlanarGraspEnv, clone_path: "str | Path") -> Any:
    """Rebuild the CTDE actor of the failing campaign and load the measured clone weights into it (fail-fast on
    an architecture mismatch — ``load_state_dict`` raises rather than silently accepting a wrong shape)."""
    actor, _critics = build_collaborative_offpolicy(env, kind="sa_hsikan", hidden=64)
    state = torch.load(clone_path, map_location="cpu")
    actor.load_state_dict(state)
    actor.eval()
    return actor


def _eval_delivery(env: PlanarGraspEnv, actor: Any, n: int) -> float:
    dwell = int(getattr(env, "success_steps", 1))
    res = eval_metric(env, greedy_action_fn(actor), DwellMetric("in_zone", dwell), n_episodes=n, seed0=_EVAL_SEED)
    return float(sum(res)) / max(1, n)


class QTermCollapseProbe:
    """Freeze the BC clone, fit ``Q^mu0`` faithfully, and run the M1/M3(/M2) discrimination battery.

    # Preconditions ``clone_path`` loads into a :func:`build_collaborative_offpolicy` actor; the env exposes the
      deliver reward. # Postconditions :meth:`run` returns a :class:`ProbeVerdict` and never mutates ``mu0``.
    """

    def __init__(self, clone_path: "str | Path", cfg: ProbeConfig) -> None:
        from gymnasium.spaces import Box
        self.clone_path = str(clone_path)
        self.cfg = cfg.resolved()
        self.env = make_deliver_env(self.cfg.difficulty)
        space = self.env.action_space
        assert isinstance(space, Box)
        self.scale = float(np.max(np.abs(space.high)))                   # action bound
        obs_shape = self.env.observation_space.shape
        assert obs_shape is not None
        self.obs_shape = tuple(int(d) for d in obs_shape)
        self.action_dim = int(self.env.n_actions)
        self.rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.mu0 = _load_clone(self.env, clone_path)                      # frozen throughout
        for p in self.mu0.parameters():
            p.requires_grad_(False)
        # Symmetric critic (priv=off): this diagnostic studies the plain Q(s,a) clone-critic geometry (fitted-Q
        # overshoot, actor-ascent), not the asymmetric-CTDE contract — it probes the critic with (obs, action).
        _, self.critics = build_collaborative_offpolicy(self.env, kind="sa_hsikan", hidden=64, privileged=False)
        self.t_critics = [copy.deepcopy(c) for c in self.critics]
        self.fit_traj: "list[dict[str, float]]" = []      # {u,q,loss} samples of the fitted-Q evaluation

    # ---- data path -----------------------------------------------------------------------------------------
    def _collect_on_clone(self) -> ReplayBuffer:
        """Roll ``mu0 + noise`` and store (s,a,r,s2,d) with the deliver reward — the clone's own visited
        distribution (no actor drift, so no H_shift feedback). Truncation stored NON-terminal (bootstrap past
        the time limit; ddpg.py:519)."""
        cfg = self.cfg
        buf = ReplayBuffer(max(cfg.buffer_steps, cfg.batch_size), self.obs_shape, self.action_dim)
        sigma = cfg.noise_scale * self.scale
        obs, _ = self.env.reset(seed=cfg.seed)
        t0, done_eps = time.perf_counter(), 0
        for step in range(1, cfg.buffer_steps + 1):
            with torch.no_grad():
                mu = self.mu0(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
            act = np.clip(mu + self.rng.normal(0, sigma, mu.shape), -self.scale, self.scale).astype(np.float32)
            nobs, rew, term, trunc, _ = self.env.step(act)
            buf.add(obs, act, float(rew), nobs, bool(term))
            obs = nobs if not (term or trunc) else self.env.reset()[0]
            done_eps += int(term or trunc)
            if cfg.log_every and step % cfg.log_every == 0:
                rate = step / max(1e-9, time.perf_counter() - t0)
                print(f"  [probe:collect] {step}/{cfg.buffer_steps} on-clone steps | {done_eps} eps | "
                      f"{rate:5.0f} steps/s", flush=True)
        return buf

    def fit_clone_critic(self, buf: ReplayBuffer) -> float:
        """Fitted-Q evaluation of the FROZEN clone: train ``self.critics`` to ``Q^mu0`` on the fixed on-clone
        buffer (trainer's TD3 backup, target-polyak EVERY step). Records the fit trajectory in ``self.fit_traj``
        so divergence (rising loss / marching Q, not just a non-finite final value) is detectable.
        # Postconditions returns the final critic loss; ``self.fit_traj`` holds the ``{u,q,loss}`` samples."""
        last, self.fit_traj = _bellman_fit(
            self.critics, self.t_critics, self.mu0, buf, scale=self.scale, rng=self.rng, cfg=self.cfg,
            reward_norm=self.cfg.reward_norm, max_grad_norm=self.cfg.max_grad_norm,
            critic_lr=self.cfg.critic_lr, label="probe:fit")
        return last

    # ---- Monte-Carlo return-to-go (raw deliver reward) ------------------------------------------------------
    def _rollout_returns(self, actor: Any, n_traj: int, seed0: int) -> "tuple[np.ndarray, np.ndarray, float]":
        """Roll ``actor`` greedily for ``n_traj`` episodes; return (obs, return-to-go G, mean episode return).
        Ranking-robust: Spearman is invariant to the constant ``reward_norm`` scale, so G uses the RAW reward."""
        g = self.cfg.gamma
        obs_all: list[np.ndarray] = []
        g_all: list[float] = []
        ep_returns: list[float] = []
        act_fn = greedy_action_fn(actor)
        for ep in range(n_traj):
            obs, _ = self.env.reset(seed=seed0 + ep)
            traj_o: list[np.ndarray] = []
            traj_r: list[float] = []
            for _ in range(self.env.max_steps):
                traj_o.append(obs)
                obs, rew, term, trunc, _ = self.env.step(act_fn(self.env, obs))
                traj_r.append(float(rew))
                if term or trunc:
                    break
            gto = 0.0
            rev: list[float] = []
            for rr in reversed(traj_r):                                  # return-to-go, discounted
                gto = rr + g * gto
                rev.append(gto)
            rev.reverse()
            obs_all.extend(traj_o)
            g_all.extend(rev)
            ep_returns.append(rev[0] if rev else 0.0)
        return (np.asarray(obs_all, dtype=np.float32), np.asarray(g_all, dtype=np.float32),
                float(np.mean(ep_returns)) if ep_returns else 0.0)

    # ---- M1 -------------------------------------------------------------------------------------------------
    def rank_fidelity(self) -> float:
        """M1: Spearman rank correlation between ``Q_hat(s, mu0(s))`` and the MC return-to-go ``G(s)`` on
        held-out on-clone states. # Postconditions returns rho in [-1, 1]."""
        obs, gto, _ = self._rollout_returns(self.mu0, self.cfg.mc_traj, seed0=50_000)
        if len(obs) == 0:
            return 0.0
        idx = self.rng.choice(len(obs), size=min(self.cfg.mc_states, len(obs)), replace=False)
        so = torch.as_tensor(obs[idx], dtype=torch.float32)
        with torch.no_grad():
            q = self.critics[0](so, self.mu0(so)).numpy()
        return _spearman(q, gto[idx])

    # ---- M3 -------------------------------------------------------------------------------------------------
    def one_step_improve(self, buf: ReplayBuffer) -> "tuple[float, Any, torch.Tensor]":
        """M3: ascend the pure Q-term (no BC anchor) on a COPY ``mu1`` of ``mu0`` against the FROZEN critic, over
        on-clone states — the idealized off-policy improvement operator, with the online feedback removed. This
        is exactly the term under investigation (``-Q(s, mu(s))``, ddpg.py:360).

        # Postconditions returns (q_pred_rise, mu1, batch_obs) — deliveries/returns measured by the caller; the
          frozen ``mu0`` is NOT mutated (``mu1`` is a deep copy)."""
        cfg = self.cfg
        mu1 = copy.deepcopy(self.mu0)
        for p in mu1.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(mu1.parameters(), lr=cfg.improve_lr)
        s0, *_ = buf.sample(min(cfg.batch_size, buf.size), generator=self.rng)
        with torch.no_grad():
            q_before = float(self.critics[0](s0, self.mu0(s0)).mean())
        for c in self.critics:                                          # freeze the critic during the ascent:
            c.requires_grad_(False)                                     # its grad still flows to mu1's action
        try:
            for _ in range(cfg.improve_steps):
                s, *_ = buf.sample(cfg.batch_size, generator=self.rng)
                loss = -self.critics[0](s, mu1(s)).mean()               # the actor's Q-term, in isolation
                opt.zero_grad()
                loss.backward()
                opt.step()
        finally:
            for c in self.critics:
                c.requires_grad_(True)
        with torch.no_grad():
            q_after = float(self.critics[0](s0, mu1(s0)).mean())
        mu1.eval()
        return q_after - q_before, mu1, s0

    # ---- M2 (confirmatory) ----------------------------------------------------------------------------------
    def ood_inflation(self, buf: ReplayBuffer) -> float:
        """M2: mean over on-clone states of ``Q_hat(mu0(s) + delta) - Q_hat(mu0(s))`` for random off-manifold
        ``delta`` (``ood_eps`` * action_scale). Positive ⇒ the value surface rewards *leaving* the clone action
        = the OOD phantom the deterministic policy gradient climbs."""
        cfg = self.cfg
        s, *_ = buf.sample(min(cfg.mc_states, buf.size), generator=self.rng)
        with torch.no_grad():
            a0 = self.mu0(s)
            q0 = self.critics[0](s, a0)
            deltas = []
            for _ in range(cfg.ood_dirs):
                pert = torch.clamp(a0 + cfg.ood_eps * self.scale * torch.randn_like(a0),
                                   -self.scale, self.scale)
                deltas.append((self.critics[0](s, pert) - q0).mean())
            return float(torch.stack(deltas).mean())

    # ---- framework-defect localization ----------------------------------------------------------------------
    def stability_scan(self, buf: ReplayBuffer, cells: "list[dict[str, Any]]") -> "list[dict[str, Any]]":
        """Fit ``Q^mu0`` on the SAME on-clone buffer under each stabilizer ``cell`` (``reward_norm`` /
        ``max_grad_norm`` / ``critic_lr`` overrides) and report whether the frozen-policy value fit CONVERGES or
        DIVERGES. Isolates which knob the off-policy critic's instability lives in (the "look into the framework"
        step). Fresh critics per cell, seeded identically so the only difference is the stabilizer under test.

        # Postconditions returns one record per cell with ``diverged`` (trajectory-based), ``final_loss``, and the
          ``q``/``loss`` trajectory."""
        out: "list[dict[str, Any]]" = []
        for cell in cells:
            torch.manual_seed(self.cfg.seed)                            # identical critic init per cell
            _, critics = build_collaborative_offpolicy(self.env, kind="sa_hsikan", hidden=64, privileged=False)
            t_critics = [copy.deepcopy(c) for c in critics]
            label = cell.get("label", "cell")
            print(f"\n  [scan] cell {label}: {json.dumps({k: v for k, v in cell.items() if k != 'label'})}",
                  flush=True)
            final, traj = _bellman_fit(
                critics, t_critics, self.mu0, buf, scale=self.scale,
                rng=np.random.default_rng(self.cfg.seed), cfg=self.cfg,
                reward_norm=bool(cell.get("reward_norm", self.cfg.reward_norm)),
                max_grad_norm=float(cell.get("max_grad_norm", self.cfg.max_grad_norm)),
                critic_lr=float(cell.get("critic_lr", self.cfg.critic_lr)),
                fit_updates=int(cell.get("fit_updates", self.cfg.fit_updates)),
                gamma=cell.get("gamma"), tau=cell.get("tau"), huber=bool(cell.get("huber", False)),
                reward_scale=float(cell.get("reward_scale", 1.0)), label=f"scan:{label}")
            rec = {"label": label, "diverged": _diverged(traj), "final_loss": round(final, 5),
                   "reward_norm": bool(cell.get("reward_norm", self.cfg.reward_norm)),
                   "max_grad_norm": float(cell.get("max_grad_norm", self.cfg.max_grad_norm)),
                   "critic_lr": float(cell.get("critic_lr", self.cfg.critic_lr)),
                   "gamma": float(cell.get("gamma", self.cfg.gamma)), "tau": float(cell.get("tau", self.cfg.tau)),
                   "huber": bool(cell.get("huber", False)), "reward_scale": float(cell.get("reward_scale", 1.0)),
                   "traj": traj}
            out.append(rec)
        return out

    # ---- orchestration --------------------------------------------------------------------------------------
    def run(self) -> ProbeVerdict:
        """Fit ``Q^mu0`` then run M1/M3/M2 and classify the root cause. The whole battery reads through the
        frozen clone, so H_shift (online drift) is excluded by construction and any collapse observed here is a
        property of the critic/operator, not the online loop."""
        cfg = self.cfg
        clone_deliv = _eval_delivery(self.env, self.mu0, cfg.n_eval)
        print(f"  [probe] clone delivery = {clone_deliv:.3f}; collecting on-clone buffer...", flush=True)
        buf = self._collect_on_clone()
        crit_loss = self.fit_clone_critic(buf)
        diverged = _diverged(self.fit_traj)                            # trajectory-based, not just final-finite
        rho = self.rank_fidelity()
        q_rise, mu1, _ = self.one_step_improve(buf)
        deliv1 = _eval_delivery(self.env, mu1, cfg.n_eval)
        _, _, ret0 = self._rollout_returns(self.mu0, cfg.mc_traj, seed0=60_000)
        _, _, ret1 = self._rollout_returns(mu1, cfg.mc_traj, seed0=60_000)
        ood = self.ood_inflation(buf)
        hyp, diag = _classify(clone_deliv, crit_loss, rho, q_rise, ret1 - ret0, deliv1 - clone_deliv, ood,
                              crit_diverged=diverged)
        return ProbeVerdict(clone_delivery=round(clone_deliv, 4), critic_final_loss=round(crit_loss, 5),
                            rank_spearman=round(rho, 4), q_pred_rise=round(q_rise, 5),
                            true_return_delta=round(ret1 - ret0, 4), delivery_delta=round(deliv1 - clone_deliv, 4),
                            ood_q_inflation=round(ood, 5), diagnosis=diag, hypothesis=hyp)


def _bellman_fit(critics: "list[Any]", t_critics: "list[Any]", mu0: Any, buf: ReplayBuffer, *, scale: float,
                 rng: np.random.Generator, cfg: ProbeConfig, reward_norm: bool, max_grad_norm: float,
                 critic_lr: float, fit_updates: "int | None" = None, gamma: "float | None" = None,
                 tau: "float | None" = None, huber: bool = False, reward_scale: float = 1.0,
                 label: str = "fit") -> "tuple[float, list[dict[str, float]]]":
    """Fitted-Q evaluation of a FROZEN policy ``mu0`` on ``buf`` — the trainer's TD3 backup (clipped double-Q +
    target smoothing + optional ``reward_norm``; ddpg.py:351-357) with target-polyak EVERY step and optional
    grad-norm clip (``max_grad_norm``, trainer-faithful). The one engine shared by the primary probe and the
    stability scan (§6.1 — no duplicated Bellman loop). ``gamma``/``tau``/``huber``/``reward_scale`` are the DEEP
    stabilizer axes (contraction strength, target-tracking speed, robust loss, bounded value magnitude) tried when
    the shallow {reward_norm, clip, lr} knobs do not converge. # Postconditions returns (final_loss, trajectory)."""
    n_updates = int(fit_updates if fit_updates is not None else cfg.fit_updates)
    g = cfg.gamma if gamma is None else float(gamma)
    tau_eff = cfg.tau if tau is None else float(tau)
    opt = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=critic_lr)
    reward_rms = RunningRMS()
    tn, nc = cfg.target_noise * scale, cfg.noise_clip * scale
    params = [p for c in critics for p in c.parameters()]
    loss_fn = F.smooth_l1_loss if huber else F.mse_loss                # Huber bounds the +30-terminal TD spikes
    traj: "list[dict[str, float]]" = []
    last, t0 = float("nan"), time.perf_counter()
    for u in range(1, n_updates + 1):
        s, a, r, s2, d = buf.sample(cfg.batch_size, generator=rng)
        r = r * reward_scale                                            # fixed (stationary) reward scaling
        if reward_norm:
            r = reward_rms.normalize(r)
        with torch.no_grad():
            noise = torch.clamp(torch.randn(a.shape) * tn, -nc, nc) if tn > 0 else torch.zeros_like(a)
            a2 = torch.clamp(mu0(s2) + noise, -scale, scale)
            q_next = torch.stack([tc(s2, a2) for tc in t_critics], 0).amin(0)   # clipped double-Q
            y = r + g * (1.0 - d) * q_next
        loss = torch.stack([loss_fn(c(s, a), y) for c in critics]).sum()
        opt.zero_grad()
        loss.backward()   # type: ignore[no-untyped-call]  # torch stubs mark Tensor.backward untyped
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(params, max_grad_norm)
        opt.step()
        for tc, c in zip(t_critics, critics):                          # TARGET POLYAK EVERY STEP (the fix)
            _soft_update(tc, c, tau_eff)
        last = float(loss.detach())
        if cfg.log_every and (u % cfg.log_every == 0 or u == n_updates):   # always capture the FINAL point too,
            with torch.no_grad():                                          # so traj is never empty / endpoint-less
                qm = float(critics[0](s, mu0(s)).mean())
            traj.append({"u": float(u), "q": round(qm, 3), "loss": round(last, 4)})
            rate = u / max(1e-9, time.perf_counter() - t0)
            print(f"  [{label}] {u}/{n_updates} | crit={last:.4g} | Q(mu0)={qm:+.3g} | {rate:5.0f} upd/s",
                  flush=True)
    return last, traj


def _diverged(traj: "list[dict[str, float]]") -> bool:
    """A fitted-Q evaluation of a FIXED policy must converge (loss plateau, Q bounded). Flag divergence from the
    TRAJECTORY (not just a non-finite final value): a non-finite entry, OR a loss whose final value is well above
    its running minimum, OR a |Q| still marching in the last half. This is the H_fit signature the primary run's
    final-loss-only check missed (loss rose 0.39→6.29, Q −16→−60 — clearly diverging, yet finite)."""
    if len(traj) < 3:
        return False
    losses = [t["loss"] for t in traj]
    qs = [abs(t["q"]) for t in traj]
    if any(not np.isfinite(v) for v in losses + qs):
        return True
    loss_blowup = losses[-1] > 3.0 * max(min(losses), 1e-6)            # final loss ≫ its best → not converging
    mid = len(qs) // 2
    q_marching = qs[-1] > 1.5 * max(qs[mid], 1e-6)                     # |Q| still growing in the last half
    return bool(loss_blowup or q_marching)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak (soft) target update, fused over parameter tensors (matches ddpg.py::_polyak)."""
    with torch.no_grad():
        tps: "list[torch.Tensor]" = [p for p in target.parameters()]
        sps: "list[torch.Tensor]" = [p for p in source.parameters()]
        torch._foreach_mul_(tps, 1 - tau)
        torch._foreach_add_(tps, sps, alpha=tau)


def _tied_ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (scipy ``rankdata('average')``): tied values share their mean rank, so a
    constant vector maps to a single rank (zero variance). # Postconditions returns float ranks, same length."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]                                  # dense group index 1..k
    group_starts = np.r_[np.nonzero(obs)[0], len(a)]          # boundaries of equal-value runs
    ranks: np.ndarray = 0.5 * (group_starts[dense] + group_starts[dense - 1] + 1.0)   # mean rank per tie group
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation of two 1-D arrays (tie-averaged rank-transform + Pearson). Scale-invariant, so
    the constant ``reward_norm`` factor between the critic's target and the raw MC return does not affect it.
    # Postconditions returns rho in [-1, 1]; 0.0 when either input has no rank variance (constant / length < 2)."""
    if len(x) < 2:
        return 0.0
    rx = _tied_ranks(x)
    ry = _tied_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return 0.0 if denom == 0.0 else float((rx * ry).sum() / denom)


def _classify(clone_deliv: float, crit_loss: float, rho: float, q_rise: float,
              ret_delta: float, deliv_delta: float, ood: float,
              *, crit_diverged: bool = False, rho_lo: float = 0.3, tol: float = 0.05,
              ret_tol: float = 0.5) -> "tuple[str, str]":
    """Map the measurements to the hypothesis they support (pure function, unit-tested). Order matters: an
    unfittable/DIVERGING critic (H_fit) is checked before reading its gradient (a diverged critic's gradient is
    meaningless, so the downstream H_ood/H_reward/H_shift verdicts would be confounded); a phantom (H_ood: Q rose
    but the true return it predicts FELL) before the secondary reward/metric mismatch candidate (H_reward)."""
    if crit_diverged or not np.isfinite(crit_loss) or rho < rho_lo:
        why = ("the fitted-Q evaluation DIVERGED (loss rising / |Q| marching for a FIXED policy)"
               if crit_diverged else f"the critic cannot rank the clone's own states (rho={rho:.2f} < {rho_lo})")
        return "H_fit", (f"{why} — loss={crit_loss:.3g}, rho={rho:.2f}. The off-policy critic is unstable at the "
                         f"SOURCE: a frozen policy's value function must converge, and this one does not, so its "
                         f"gradient (and every refine built on it) is unreliable. This is a FRAMEWORK defect, not "
                         f"an RL-can't-help result — fix the critic stability (grad-clip / reward-scale / lr / "
                         f"target lag; run stability_scan to localize) before any refine. Downstream H_ood/H_shift "
                         f"verdicts are confounded until the fit converges.")
    if ret_delta < -ret_tol:
        return "H_ood", (f"the critic ranks on-distribution states well (rho={rho:.2f}) but ascending Q "
                         f"(predicted +{q_rise:.3g}) DROVE the true return DOWN ({ret_delta:+.2f}); off-manifold "
                         f"Q inflation={ood:+.3g}. The deterministic policy gradient climbs a phantom: OOD "
                         f"overestimation. Bound OOD Q (in-support gradient / CQL / stronger BC constraint) or "
                         f"accept model-free RL cannot add here.")
    if deliv_delta < -tol:
        return "H_reward", (f"the critic is faithful (rho={rho:.2f}) and ascending Q raised the true return "
                            f"({ret_delta:+.2f}) but delivery FELL ({deliv_delta:+.2f}): the training reward is "
                            f"a reward/metric mismatch candidate. This is NOT permission to redesign the scenario "
                            f"or retune reward first; audit reward timing, termination/truncation, eval labeling, "
                            f"oracle assumptions, and implementation plumbing before any reward change.")
    return "H_shift", (f"one clean improvement against the frozen well-fit critic did NOT degrade the clone "
                       f"(delivery {deliv_delta:+.2f}, return {ret_delta:+.2f}, rho={rho:.2f}): the operator is "
                       f"sound offline. The measured online collapse is therefore the replay-distribution-drift "
                       f"feedback loop (H_shift) — fix with on-policy-ish replay / conservative UTD / slower "
                       f"actor, NOT the critic or the reward.")


def run_probe(clone_path: "str | Path", cfg: ProbeConfig, *, base: str = "experiments") -> "dict[str, Any]":
    """Run the probe, tee stdout + verdict into a self-contained ``experiments/<ts>_qterm_collapse_probe/``.

    # Postconditions writes ``verdict.json`` + ``run.log``; returns the verdict dict (+ ``dir``)."""
    exp = experiment_dir(base, "qterm_collapse_probe")
    from hymeko_rl.train.campaign import tee_stdout
    with tee_stdout(exp / "run.log"):
        print(f"\n===== Q-term collapse root-cause probe =====\nclone: {clone_path}\n"
              f"cfg: {json.dumps(asdict(cfg.resolved()))}", flush=True)
        t0 = time.perf_counter()
        verdict = QTermCollapseProbe(clone_path, cfg).run()
        wall = time.perf_counter() - t0
        out = {**asdict(verdict), "clone_path": str(clone_path), "wall_s": round(wall, 1)}
        print(f"\n=== VERDICT: {verdict.hypothesis} ({wall:.0f}s) ===\n{verdict.diagnosis}", flush=True)
        print(json.dumps(out, indent=2), flush=True)
        (exp / "verdict.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["dir"] = str(exp)
    return out


def default_scan_cells(fit_updates: int) -> "list[dict[str, Any]]":
    """The stabilizer grid for :meth:`QTermCollapseProbe.stability_scan`: the primary probe's setting (A) plus one
    knob turned at a time, so a divergence that vanishes in exactly one cell names the responsible knob."""
    return [
        {"label": "A_baseline_rnormT_clip0", "reward_norm": True, "max_grad_norm": 0.0, "fit_updates": fit_updates},
        {"label": "B_trainer_rnormT_clip10", "reward_norm": True, "max_grad_norm": 10.0, "fit_updates": fit_updates},
        {"label": "C_rnormF_clip10", "reward_norm": False, "max_grad_norm": 10.0, "fit_updates": fit_updates},
        {"label": "D_rnormT_clip10_lr3e-4", "reward_norm": True, "max_grad_norm": 10.0, "critic_lr": 3e-4,
         "fit_updates": fit_updates},
        {"label": "E_rnormF_clip10_lr3e-4", "reward_norm": False, "max_grad_norm": 10.0, "critic_lr": 3e-4,
         "fit_updates": fit_updates},
    ]


def deep_scan_cells(fit_updates: int) -> "list[dict[str, Any]]":
    """The DEEP stabilizer grid — tried when the shallow {reward_norm, clip, lr} knobs all diverge. Each targets a
    distinct divergence mechanism of a bootstrapped value fit: Huber (bounds the +30-terminal TD spikes), lower γ
    (stronger Bellman contraction + smaller value magnitude), slower τ (less aggressive target chasing), fixed
    reward scaling (bounded value magnitude without reward-norm nonstationarity), and their combination."""
    base = {"reward_norm": True, "max_grad_norm": 10.0, "fit_updates": fit_updates}
    return [
        {**base, "label": "F_huber", "huber": True},
        {**base, "label": "G_gamma0.95", "gamma": 0.95},
        {**base, "label": "H_tau0.001", "tau": 0.001},
        {**base, "label": "I_rscale0.1_rnormF", "reward_norm": False, "reward_scale": 0.1},
        {**base, "label": "J_huber_gamma0.95_lr3e-4", "huber": True, "gamma": 0.95, "critic_lr": 3e-4},
    ]


def run_ablation(clone_path: "str | Path", cfg: ProbeConfig, *, scan_updates: int = 8_000,
                 deep: bool = False, base: str = "experiments") -> "dict[str, Any]":
    """Localize the off-policy critic's frozen-policy divergence to a stabilizer knob. Collects ONE on-clone
    buffer (measurement-cache discipline), fits ``Q^mu0`` under each cell (``deep`` → the deep grid), and reports
    which converge. # Postconditions writes ``ablation.json`` + ``run.log`` into a self-contained run dir."""
    from dataclasses import replace
    exp = experiment_dir(base, "qterm_stability_ablation")
    from hymeko_rl.train.campaign import tee_stdout
    with tee_stdout(exp / "run.log"):
        print(f"\n===== Q-term critic STABILITY ablation ({'deep' if deep else 'shallow'}) =====\n"
              f"clone: {clone_path}", flush=True)
        # Finer log cadence than the primary probe: >= 8 trajectory points per cell so _diverged (which needs
        # >= 3) reliably distinguishes a marching Q from a plateau (the primary default 4000 gave only 2 pts/8k).
        fine = replace(cfg, log_every=max(1, min(cfg.log_every, scan_updates // 8)))
        probe = QTermCollapseProbe(clone_path, fine)
        buf = probe._collect_on_clone()
        cells = deep_scan_cells(scan_updates) if deep else default_scan_cells(scan_updates)
        results = probe.stability_scan(buf, cells)
        print("\n=== STABILITY SCAN ===", flush=True)
        for r in results:
            print(f"  {r['label']:28s} diverged={r['diverged']!s:5s} final_loss={r['final_loss']:.4g} "
                  f"(rnorm={r['reward_norm']}, clip={r['max_grad_norm']}, lr={r['critic_lr']})", flush=True)
        converged = [r["label"] for r in results if not r["diverged"]]
        print(f"\nCONVERGED cells: {converged or 'NONE — divergence is not a single-knob fix'}", flush=True)
        out = {"clone_path": str(clone_path), "converged": converged, "cells": results}
        (exp / "ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["dir"] = str(exp)
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clone", default="experiments/2026_07_05_03_29_galambos_coord_ab_deliver/"
                    "policies/galambos_coord_ab_deliver_s0.pt", help="BC clone checkpoint to interrogate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="fast path check (tiny buffer/fit/eval)")
    ap.add_argument("--ablate", action="store_true",
                    help="run the critic-stability scan (localize the frozen-policy divergence to a knob)")
    ap.add_argument("--deep", action="store_true",
                    help="use the deep stabilizer grid (Huber / gamma / tau / reward-scale) instead of the shallow one")
    ap.add_argument("--scan-updates", type=int, default=8_000, help="critic updates per ablation cell")
    a = ap.parse_args(argv)
    if a.ablate:
        run_ablation(a.clone, ProbeConfig(seed=a.seed, smoke=a.smoke), scan_updates=a.scan_updates, deep=a.deep)
    else:
        run_probe(a.clone, ProbeConfig(seed=a.seed, smoke=a.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
