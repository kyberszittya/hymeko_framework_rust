"""Off-policy actor-critic — DDPG and TD3 on the HyMeKo cart-pole (one config-driven core).

The off-policy family beside the on-policy PPO baseline (survey:
``reports/2026-06-21-offpolicy-rl-survey``). A deterministic actor ``μ(s)`` is improved *through* a Q-critic
``Q(s,a)`` (the deterministic policy gradient); critics regress to a Polyak-target Bellman backup over a replay
buffer; exploration is additive Gaussian action noise. The state encoder is the *same* swappable backbone
(``mlp``/``hsikan``/``signedkan``) the PPO policy uses — architecture is orthogonal to the algorithm.

**DDPG and TD3 are one trainer** (§6.5 #1: config, not a forked function). TD3 = DDPG + three settings, each a
fix to DDPG's Q-overestimation/fragility: ``n_critics=2`` (clipped double-Q — target uses the min),
``policy_delay=2`` (delayed actor/target updates), ``target_noise>0`` (target-policy smoothing). DDPG is the
degenerate preset (``n_critics=1, policy_delay=1, target_noise=0``).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.train.normalize import RunningRMS
from hymeko_rl.agents.policy import POLICY_KINDS, _BACKBONES
from hymeko_rl.train.replay import ReplayBuffer
from hymeko_rl.train.train_inverted_pendulum import eval_balance


class DeterministicActor(nn.Module):
    """``μ(s) = action_scale · tanh(head(backbone(s)))`` — a bounded deterministic policy.

    # Invariants output is in ``[-action_scale, action_scale]^{action_dim}``."""

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int, action_scale: float, *,
                 detach_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, action_dim)
        self.action_scale = float(action_scale)
        # Shared-trunk (DrQ/SAC-AE) convention: when the backbone is SHARED with the critics, the actor reads its
        # features with a stop-grad so only the CRITIC's Bellman gradient shapes the structural trunk (the actor's
        # max-Q gradient would otherwise destabilise the shared encoder). The actor still learns — via its head.
        self.detach_backbone = bool(detach_backbone)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(obs)
        if self.detach_backbone:
            feat = feat.detach()
        return self.action_scale * torch.tanh(self.head(feat))

    def head_parameters(self) -> "list[nn.Parameter]":
        """The actor's OWN params (the head) — excludes a shared trunk, for the split optimiser."""
        return list(self.head.parameters())

    @property
    def action_dim(self) -> int:
        """The joint action width. Read by :func:`train_offpolicy` (replay-buffer / target-noise sizing) so a
        multi-head CTDE actor without a single ``head`` can satisfy the same contract by exposing ``action_dim``."""
        return int(self.head.out_features)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias so :func:`hymeko_rl.train.train_inverted_pendulum.eval_balance` can score the greedy policy."""
        return self.forward(obs)


class QCritic(nn.Module):
    """``Q(s,a) = head(concat(backbone(s), a))`` — an action-value over the same backbone family.

    ``layernorm`` inserts a ``LayerNorm`` after the first head linear: it bounds the feature magnitude, which
    bounds Q growth — the standard, low-risk cure for the off-policy **Q-overestimation** that destabilised the
    galambos refine (critic loss spiking to 90, the actor chasing an inflating Q away from the clone)."""

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int, *, layernorm: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        layers: list[nn.Module] = [nn.Linear(feat_dim + action_dim, feat_dim)]
        if layernorm:
            layers.append(nn.LayerNorm(feat_dim))    # bound the feature scale -> bound Q growth (anti-overestimation)
        layers += [nn.ReLU(), nn.Linear(feat_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.head(torch.cat([self.backbone(obs), action], dim=-1)).squeeze(-1)
        return q


def _backbone(kind: str, obs_dim: int, flat_dim: int, **kw: object) -> tuple[nn.Module, int]:
    """Build one backbone of the requested kind (mlp reads the flat obs; hsikan/signedkan the per-vertex obs).

    ``hidden`` is forwarded to the mlp too, so the off-policy MLP baseline can be widened to **params-match**
    the HSiKAN backbone (the standing rule: always compare against a matched-capacity control). ``hg_state``
    and other hsikan-only kwargs are not passed to the mlp.
    """
    if kind == "mlp":
        hidden = kw.get("hidden")
        if hidden is None:
            return _BACKBONES["mlp"](flat_dim)
        assert isinstance(hidden, int)   # callers pass an int width; narrow for the kwargs forward
        return _BACKBONES["mlp"](flat_dim, hidden=hidden)
    return _BACKBONES[kind](obs_dim, **kw)


def build_offpolicy(kind: str, *, obs_dim: int, flat_dim: int, action_dim: int, action_scale: float,
                    n_critics: int = 1, hidden: int = 64, device: torch.device | str = "cpu",
                    shared_trunk: bool = False, critic_layernorm: bool = False,
                    **kw: object) -> tuple[DeterministicActor, list[QCritic]]:
    """Construct ``(actor, [critic, …])`` of ``kind`` (the architecture swap).

    ``n_critics`` is 1 for DDPG, 2 for TD3 (clipped double-Q). ``shared_trunk`` makes the actor + every critic
    share ONE backbone (structural cross-propagation): the critics train it (Bellman) and the actor reads it with
    a stop-grad (``detach_backbone``) — the value gradient shapes the structural representation without the
    actor's max-Q gradient destabilising it. ``device`` moves the nets (signed adjacency buffers travel with them).
    # Preconditions ``kind in POLICY_KINDS``; ``hsikan``/``signedkan``/``sa_hsikan`` require ``hg_state=`` in ``kw``."""
    if kind not in POLICY_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected {POLICY_KINDS}")
    if n_critics < 1:
        raise ValueError(f"n_critics must be >= 1; got {n_critics}")
    if shared_trunk:
        trunk, feat = _backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)
        actor = DeterministicActor(trunk, feat, action_dim, action_scale, detach_backbone=True).to(device)
        critics = [QCritic(trunk, feat, action_dim, layernorm=critic_layernorm).to(device)
                   for _ in range(n_critics)]                                            # SAME trunk object
        return actor, critics
    ab, feat = _backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)
    actor = DeterministicActor(ab, feat, action_dim, action_scale).to(device)
    critics = [QCritic(_backbone(kind, obs_dim, flat_dim, hidden=hidden, **kw)[0], feat, action_dim,
                       layernorm=critic_layernorm).to(device) for _ in range(n_critics)]
    return actor, critics


def build_hetero_offpolicy(actor_kind: str, critic_kinds: "list[str]", *, obs_dim: int, flat_dim: int,
                           action_dim: int, action_scale: float, hidden: int = 64,
                           device: torch.device | str = "cpu", critic_layernorm: bool = False, **kw: object
                           ) -> tuple[DeterministicActor, list[QCritic]]:
    """Actor of ``actor_kind`` + **heterogeneous** critics, one ``QCritic`` per ``critic_kinds`` entry — the
    multi-timescale dual-critic (Hajdu 2026-07-01): e.g. ``critic_kinds=["mlp", "sa_hsikan"]`` pairs a cheap
    fused-MLP critic (updated every step — the reflexive value tracker) with a structural SA-HSiKAN critic
    (updated less often — the deliberative corrector). In TD3's clipped double-Q the *lagging* structural critic
    is a **conservative Q-bound** (attacks the overestimation that blows up a single fast critic), at a fraction
    of the launch-bound cost since it updates rarely. Each critic reads the same ``(B, N, feat)`` obs (the MLP
    flattens it); the per-critic update cadence is set by ``OffPolicyConfig.critic_update_every``.

    # Preconditions ``actor_kind in POLICY_KINDS``; every ``critic_kinds`` entry in ``POLICY_KINDS``; structural
      kinds need ``hg_state=`` in ``kw``. # Postconditions ``len(critics) == len(critic_kinds)``."""
    if actor_kind not in POLICY_KINDS or any(ck not in POLICY_KINDS for ck in critic_kinds):
        raise ValueError(f"kinds must be in {POLICY_KINDS}; got actor={actor_kind!r}, critics={critic_kinds}")
    if not critic_kinds:
        raise ValueError("critic_kinds must be non-empty")
    ab, feat = _backbone(actor_kind, obs_dim, flat_dim, hidden=hidden, **kw)
    actor = DeterministicActor(ab, feat, action_dim, action_scale).to(device)
    critics: list[QCritic] = []
    for ck in critic_kinds:
        cb, cfeat = _backbone(ck, obs_dim, flat_dim, hidden=hidden, **kw)
        critics.append(QCritic(cb, cfeat, action_dim, layernorm=critic_layernorm).to(device))
    return actor, critics


@dataclass(frozen=True)
class OffPolicyConfig:
    """DDPG/TD3 hyperparameters (cart-pole defaults). The TD3 axes default to DDPG (off)."""

    total_steps: int = 30_000
    start_steps: int = 1_000          # uniform-random action warm-up to seed the buffer
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005                # Polyak averaging coefficient for the target nets
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    max_grad_norm: float = 10.0       # gradient-norm clip on actor + critics (prevents the Q-overestimation blow-up)
    reward_norm: bool = True          # normalise rewards by a running RMS → bounds the Q-scale (anti-divergence)
    capacity: int = 100_000
    noise_scale: float = 0.1          # exploration σ as a fraction of action_scale
    eval_every: int = 5_000
    n_eval: int = 10
    seed: int = 0
    critic_update_every: "tuple[int, ...]" = ()   # PER-CRITIC update cadence for the heterogeneous dual-critic
                                      # (build_hetero_offpolicy). () or all-1 = uniform (every critic every update —
                                      # the compiled combined path). e.g. (1, 4) with critics [mlp, sa_hsikan] =
                                      # update the cheap MLP critic every update, the launch-bound SA-HSiKAN critic
                                      # every 4th → cheaper AND its lagging Q is a conservative bound in clipped
                                      # double-Q (attacks overestimation). Length must equal len(critics).
    update_every: int = 1             # UTD (update-to-data) ratio = 1/update_every: run the gradient-update block
                                      # every `update_every` env steps. 1 = every step (SB3/TD3 default). Raise to
                                      # 2–4 to cut wall-time when the per-step backward dominates (launch-bound tiny
                                      # nets — measured 56 min/5e4 on galambos): the physics + action-select still
                                      # run every step, only the backward is decimated. Modest quality tradeoff for
                                      # TD3; measure before trusting a high value (§3).
    log_every: int = 5_000            # LIVE FEEDBACK: emit a progress line (step, critic/actor loss, steps/s, ETA)
                                      # every `log_every` steps. Never run blind — the loss line surfaces a NaN/
                                      # divergence loop instantly, and steps/s makes a stall obvious. 0 = silent
                                      # (short runs / tests where total_steps < log_every never print anyway).
    device: str = "cpu"              # "cpu" (default — safe for short runs/tests), "cuda", or "auto" (cuda if
                                      # available). OPT-IN for long runs: GPU+`compile` pays off only past ~1e4
                                      # steps (compile/capture has fixed cost). These nets are LAUNCH-BOUND (tiny
                                      # ops): raw cuda is *slower* than cpu (measured 53 vs 42 ms/update) — the win
                                      # is `compile` below (CUDA graphs), not the device alone. Internal eval
                                      # (`eval_balance`/`eval_fn`) is CPU-only, so set eval_every > total_steps on
                                      # cuda runs (or rely on an end-of-training eval). Nets return on CPU.
    compile: bool = True              # torch.compile(mode="reduce-overhead") the update step on CUDA — captures the
                                      # launch sequence as a CUDA graph, amortising the dispatch that dominates a
                                      # launch-bound tiny-net update. Measured 8.25x vs cpu (5.1 vs 41.9 ms/update)
                                      # on the RTX 3070. No-op on cpu. Needs a working C++ toolchain (MSVC on Win).
    # --- TD3 axes (DDPG defaults: a single critic, no delay, no target smoothing) ---
    n_critics: int = 1
    policy_delay: int = 1             # update actor/targets every `policy_delay` critic steps (TD3: 2)
    target_noise: float = 0.0         # target-policy smoothing σ as a fraction of action_scale (TD3: ~0.2)
    noise_clip: float = 0.5           # smoothing clip as a fraction of action_scale
    # --- BC warm-start bridge (default off → from-scratch behaviour unchanged) ---
    warm_start: bool = False          # the actor is pre-trained (BC): act with IT from step 0 (skip the
                                      # uniform-random `start_steps` that would overwrite the cloned behaviour).
    critic_warmup: int = 0            # update the CRITIC alone for this many steps before the actor moves, so a
                                      # cold critic's gradients don't destroy the cloned actor (set ~2000 for BC).
    # --- TD3+BC: anchor the deterministic actor to the demos during the policy update (Fujimoto & Gu 2021) ---
    bc_coef: float = 0.0              # weight on ||mu(s) - a_demo||^2 over the offline demos. 0 = plain DDPG/TD3.
                                      # The off-policy analogue of the FAILED PPO anchor: here it holds because the
                                      # critic learns off-policy from replay, so the actor is improved through a Q
                                      # that has seen the visited distribution — not on-policy advantages only.
    adaptive_bc: bool = False         # Adaptive-BC: a FIXED bc_coef is the wrong knob (lit) — too little collapses,
                                      # too much freezes. When on, scale the BC weight up when the eval degrades
                                      # (anchor harder) and down when it improves (loosen), within [0.1, 10]x.


# Back-compat alias (DDPG was the first member); presets below select DDPG vs TD3.
DDPGConfig = OffPolicyConfig


def td3_config(**overrides: object) -> OffPolicyConfig:
    """A TD3 preset: twin critics, delayed policy, target-policy smoothing."""
    base: dict[str, object] = {"n_critics": 2, "policy_delay": 2, "target_noise": 0.2, "noise_clip": 0.5}
    base.update(overrides)
    return OffPolicyConfig(**base)   # type: ignore[arg-type]


def td3_bc_config(**overrides: object) -> OffPolicyConfig:
    """TD3+BC preset (Fujimoto & Gu 2021): TD3 axes + a BC anchor to the demos + the BC warm-start bridge
    (act with the cloned actor from step 0, hold it through a critic warm-up). The off-policy fix for the PPO
    warm-start collapse — pass ``offline_data=(obs, acts)`` to ``train_offpolicy``."""
    base: dict[str, object] = {"n_critics": 2, "policy_delay": 2, "target_noise": 0.2, "noise_clip": 0.5,
                               "bc_coef": 2.5, "warm_start": True, "critic_warmup": 2000}
    base.update(overrides)
    return OffPolicyConfig(**base)   # type: ignore[arg-type]


def adaptive_bc_config(**overrides: object) -> OffPolicyConfig:
    """Adaptive-BC preset: TD3+BC with the BC weight scheduled by online eval (anchor harder on a regress, looser
    on a gain) — the lit fix for a fixed ``bc_coef`` being too little (collapse) or too much (frozen)."""
    base: dict[str, object] = {"adaptive_bc": True}
    base.update(overrides)
    return td3_bc_config(**base)


def _polyak(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak (soft) target update ``θ_t ← (1-τ)θ_t + τθ_s``, **fused** over all parameter tensors.

    The naive per-tensor Python loop (``for tp, sp: tp.mul_(1-τ).add_(sp, τ)``) is launch-bound — 2 tiny kernel
    launches per tensor × ~10 tensors × 3 target nets every update dominates the wall on small nets (measured:
    it was the bulk of a 22 steps/s off-policy loop on a 1.1k-param SA-HSiKAN). ``torch._foreach_*`` collapses
    the whole loop into 2 fused multi-tensor kernels — numerically identical, just far fewer launches."""
    with torch.no_grad():
        tps: list[torch.Tensor] = [p for p in target.parameters()]
        sps: list[torch.Tensor] = [p for p in source.parameters()]
        torch._foreach_mul_(tps, 1 - tau)
        torch._foreach_add_(tps, sps, alpha=tau)


def train_offpolicy(actor: DeterministicActor, critics: list[QCritic], env: Any,
                    cfg: OffPolicyConfig, *,
                    eval_fn: Callable[[Any, Any], float] | None = None,
                    offline_data: "tuple[np.ndarray, np.ndarray] | None" = None,
                    n_envs: int = 1, make_env: "Callable[[], Any] | None" = None) -> list[float]:
    """Train DDPG/TD3 on ``env`` (one core); returns the periodic eval curve.

    Clipped double-Q when ``n_critics>1`` (target uses ``min`` over the target critics); delayed actor/target
    updates every ``policy_delay`` critic steps; target-policy smoothing when ``target_noise>0``. Time-limit
    truncation is stored as **non-terminal**, so the Bellman backup bootstraps past the time limit.

    ``eval_fn`` scores the curve: ``None`` (default) uses :func:`eval_balance` (cart-pole upright-steps,
    behaviour unchanged); a real-topology task injects a return-based eval (env-agnostic). It is called as
    ``eval_fn(env, actor) -> float`` at each ``eval_every`` checkpoint.
    """
    if cfg.update_every < 1:
        raise ValueError(f"update_every must be >= 1; got {cfg.update_every}")
    if n_envs < 1:
        raise ValueError(f"n_envs must be >= 1; got {n_envs}")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if cfg.device == "auto" else cfg.device))
    actor.to(dev)
    for _c in critics:
        _c.to(dev)
    t_actor = copy.deepcopy(actor)
    t_critics = [copy.deepcopy(c) for c in critics]
    # Shared-trunk split (structural cross-propagation): if the actor and critics share ONE backbone, the actor
    # optimises its HEAD only (the trunk is trained by the critics' Bellman loss; the actor reads it detached),
    # and the critic optimiser de-duplicates the shared trunk params (the k critics list it k times).
    shared_trunk = bool(critics) and getattr(actor, "backbone", None) is not None \
        and actor.backbone is critics[0].backbone
    if shared_trunk:
        a_opt = torch.optim.Adam(actor.head_parameters(), lr=cfg.actor_lr)
        seen: set[int] = set()
        c_params: list[nn.Parameter] = []
        for c in critics:
            for p in c.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    c_params.append(p)
        c_opt = torch.optim.Adam(c_params, lr=cfg.critic_lr)
    else:
        a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        c_opt = torch.optim.Adam([p for c in critics for p in c.parameters()], lr=cfg.critic_lr)
    # Heterogeneous dual-critic: per-critic optimizers + cadence (build_hetero_offpolicy). Uniform = the combined
    # c_opt above + the compiled path. Hetero updates each critic on its own schedule and polyaks its target at
    # that cadence, so a rarely-updated structural critic lags → conservative Q-bound in the clipped min.
    cue = cfg.critic_update_every
    hetero = bool(cue) and any(u != 1 for u in cue)
    c_opts: list[torch.optim.Adam] = []
    if hetero:
        if len(cue) != len(critics):
            raise ValueError(f"critic_update_every length {len(cue)} != n_critics {len(critics)}")
        if shared_trunk:
            raise ValueError("heterogeneous critic_update_every is incompatible with shared_trunk")
        c_opts = [torch.optim.Adam(c.parameters(), lr=cfg.critic_lr) for c in critics]
    space_shape = env.observation_space.shape
    assert space_shape is not None
    buf = ReplayBuffer(cfg.capacity, tuple(int(d) for d in space_shape), actor.action_dim)
    scale = actor.action_scale
    sigma, tn, nc = cfg.noise_scale * scale, cfg.target_noise * scale, cfg.noise_clip * scale
    history: list[float] = []
    updates = 0
    bc_scale = 1.0                                                  # Adaptive-BC runtime multiplier on bc_coef
    reward_rms = RunningRMS()                                       # bounds the Q-scale (anti-divergence)
    bc_obs = bc_act = None                                          # TD3+BC: the demos that anchor the actor
    if cfg.bc_coef > 0.0 and offline_data is not None:
        bc_obs = torch.as_tensor(offline_data[0], dtype=torch.float32, device=dev)
        bc_act = torch.as_tensor(offline_data[1], dtype=torch.float32, device=dev)

    # The launch-bound hot path (target + critic loss, actor loss) as closures. On CUDA we torch.compile them
    # with reduce-overhead → the forward is captured as a CUDA graph and its backward is compiled by autograd,
    # amortising the per-op dispatch that dominates a tiny-net update (measured 8.25x). The optimizer/clip/polyak
    # stay eager, and the *modules* are never compiled in place — so checkpoints stay clean (no `_orig_mod.`).
    def _critic_loss(s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, s2: torch.Tensor,
                     d: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a2 = torch.clamp(t_actor(s2) + noise, -scale, scale)
            q_next = torch.stack([tc(s2, a2) for tc in t_critics], 0).amin(0)   # clipped double-Q
            y = r + cfg.gamma * (1 - d) * q_next
        return sum(F.mse_loss(c(s, a), y) for c in critics)   # type: ignore[return-value]

    def _actor_loss(s: torch.Tensor, bo: "torch.Tensor | None", ba: "torch.Tensor | None") -> torch.Tensor:
        al = -critics[0](s, actor(s)).mean()
        if bo is not None and ba is not None:                 # TD3+BC anchor over the demos
            al = al + cfg.bc_coef * bc_scale * F.mse_loss(actor(bo), ba)
        return al

    cudagraph_step = cfg.compile and dev.type == "cuda"
    if cudagraph_step:
        # PERSISTENT compile cache: pin Inductor's on-disk cache to a stable project-local dir so consecutive runs
        # (and later seeds) SKIP kernel compilation — a cache hit turns the ~30-60s cold JIT into a fast load +
        # CUDA-graph capture. fx_graph_cache is on by default; the explicit dir just survives %TEMP% cleanup. (For
        # a *pure* no-JIT deploy of the trained actor, AOTInductor compiles the forward to a standalone .dll — see
        # deploy_policy; that is the inference path, not this training loop.)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                              str(Path(__file__).resolve().parent.parent / ".torchinductor_cache"))
        _critic_loss = torch.compile(_critic_loss, mode="reduce-overhead")   # type: ignore[assignment]
        _actor_loss = torch.compile(_actor_loss, mode="reduce-overhead")     # type: ignore[assignment]
    # NOTE: compiling the B=1 action-selection was MEASURED net-negative end-to-end (56 vs 75 steps/s) despite a
    # 3.2x isolated profile — compiling the actor for both B=1 (select) and B=batch (loss) thrashes torch.compile's
    # CUDA-graph shape guards. Left eager on purpose (measure the pipeline, not the component). 2026-07-02.

    action_dim = int(actor.action_dim)
    # Live feedback (never run blind): steps/s, ETA, and the losses (a NaN/inf surfaces a divergence loop instantly).
    t_last = time.perf_counter()
    step_last, last_c, last_a = 0, float("nan"), float("nan")

    def _update_once() -> None:                              # ONE gradient update — shared by the single & vec paths
        nonlocal updates, last_c, last_a
        if cudagraph_step:
            # reduce-overhead runs each compiled closure as a CUDA graph sharing one static memory pool. This step
            # interleaves TWO graphs (critic loss, then actor loss — which re-invokes critic[0]); without a step
            # marker the actor graph's replay overwrites buffers the just-run critic graph's autograd still holds,
            # raising "accessing tensor output of CUDAGraphs overwritten by a subsequent run" at a_loss.backward().
            # cudagraph_mark_step_begin() roots a fresh step so each update's graphs get non-aliasing memory. The
            # per-step losses are consumed here (backward + .detach()→float host copy) so nothing is held across it.
            torch.compiler.cudagraph_mark_step_begin()
        s, a, r, s2, d = (x.to(dev) for x in buf.sample(cfg.batch_size, generator=rng))
        if cfg.reward_norm:
            r = reward_rms.normalize(r)                       # bounded-scale reward → bounded Q-target (eager)
        noise = (torch.clamp(torch.randn(a.shape, device=dev) * tn, -nc, nc)   # target smoothing, generated eager
                 if tn > 0 else torch.zeros_like(a))         # so no RNG lives inside the CUDA graph
        updates += 1
        if hetero:
            with torch.no_grad():                             # target once: min over ALL target critics (the
                a2 = torch.clamp(t_actor(s2) + noise, -scale, scale)   # rarely-updated structural one lags and
                q_next = torch.stack([tc(s2, a2) for tc in t_critics], 0).amin(0)   # bounds Q conservatively
                y = r + cfg.gamma * (1 - d) * q_next
            for i, c in enumerate(critics):                   # each critic on its own cadence
                if updates % cue[i] == 0:
                    li = F.mse_loss(c(s, a), y)
                    c_opts[i].zero_grad()
                    li.backward()
                    torch.nn.utils.clip_grad_norm_(c.parameters(), cfg.max_grad_norm)
                    c_opts[i].step()
                    _polyak(t_critics[i], c, cfg.tau)         # target lags at the critic's cadence
                    if i == 0:
                        last_c = float(li.detach())
        else:
            c_loss = _critic_loss(s, a, r, s2, d, noise)
            c_opt.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_([p for c in critics for p in c.parameters()], cfg.max_grad_norm)
            c_opt.step()
            last_c = float(c_loss.detach())
        if updates > cfg.critic_warmup and updates % cfg.policy_delay == 0:
            bo = ba = None
            if bc_obs is not None and bc_act is not None:     # TD3+BC anchor: index the demo batch eager
                bi = torch.randint(0, int(bc_obs.shape[0]), (cfg.batch_size,), device=dev)
                bo, ba = bc_obs[bi], bc_act[bi]
            a_loss = _actor_loss(s, bo, ba)
            a_opt.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            a_opt.step()
            last_a = float(a_loss.detach())
            _polyak(t_actor, actor, cfg.tau)
            if not hetero:                                    # hetero polyaks each critic-target in the critic loop
                for tc, c in zip(t_critics, critics):
                    _polyak(tc, c, cfg.tau)

    def _log(step: int) -> None:
        nonlocal t_last, step_last
        now = time.perf_counter()
        rate = (step - step_last) / max(1e-9, now - t_last)
        eta = (cfg.total_steps - step) / max(1e-9, rate)
        # Decompose the actor objective: act = -Q + bc_coef*BC fuses two unrelated signals. Q is the divergence
        # tripwire (Q -> +/-inf = a real Q-scale blow-up); bc is the anchor pull (grows benignly as the policy
        # outgrows the demo). Logging them fused hides which is moving. Cheap (one forward per log line, no_grad).
        diag = ""
        if buf.size >= 64:
            with torch.no_grad():
                sd = buf.sample(min(256, buf.size), generator=rng)[0].to(dev)
                diag = f" Q={float(critics[0](sd, actor(sd)).mean()):+.3g}"
                if bc_obs is not None and bc_act is not None:
                    m = min(256, int(bc_obs.shape[0]))
                    bi = torch.randint(0, int(bc_obs.shape[0]), (m,), device=dev)
                    diag += f" bc={float(F.mse_loss(actor(bc_obs[bi]), bc_act[bi])):.3g}"
        print(f"  [offpolicy] step {step:>7}/{cfg.total_steps} | crit={last_c:.3g} act={last_a:.3g}{diag} | "
              f"{rate:5.0f} steps/s | ETA {eta/60:5.1f} min | buf={buf.size}", flush=True)
        t_last, step_last = now, step

    def _eval(step: int) -> None:
        nonlocal bc_scale
        history.append(eval_balance(env, actor, cfg.n_eval, seed=20_000)
                       if eval_fn is None else eval_fn(env, actor))
        if cfg.adaptive_bc and len(history) >= 2:              # anchor harder on a regress, looser on a gain
            bc_scale = (min(bc_scale * 1.5, 10.0) if history[-1] < history[-2]
                        else max(bc_scale * 0.7, 0.1))

    if n_envs > 1:                                            # VECTORIZED collection (TD3/off-policy) — batch the
        if make_env is None:                                 # B=N action-selection (one sync for N) + add_batch N
            raise ValueError("n_envs > 1 requires make_env to build the worker envs")
        envs = [make_env() for _ in range(n_envs)]
        obs_vec = np.stack([envs[i].reset(seed=cfg.seed + i)[0] for i in range(n_envs)]).astype(np.float32)
        pending, step, next_eval = 0, 0, cfg.eval_every
        while step < cfg.total_steps:
            if step <= cfg.start_steps and not cfg.warm_start:
                acts = rng.uniform(-scale, scale, (n_envs, action_dim)).astype(np.float32)
            else:
                with torch.no_grad():                         # ONE B=N forward + ONE sync (the amortized cost)
                    mu = actor(torch.as_tensor(obs_vec, dtype=torch.float32, device=dev)).cpu().numpy()
                acts = np.clip(mu + rng.normal(0, sigma, mu.shape), -scale, scale).astype(np.float32)
            buf_next = np.empty_like(obs_vec)
            nxt = np.empty_like(obs_vec)
            rews = np.empty(n_envs, dtype=np.float32)
            dones = np.empty(n_envs, dtype=np.float32)
            for i, e in enumerate(envs):                      # physics stays serial; the sync is what amortized
                no, rw, tm, tr, _ = e.step(acts[i])
                rews[i] = float(rw)
                dones[i] = 1.0 if tm else 0.0                 # truncation stored NON-terminal (bootstrap past it)
                buf_next[i] = no                              # actual next obs (terminal if done) — the Bellman target
                nxt[i] = e.reset()[0] if (tm or tr) else no   # what continues next tick (reset on done)
            buf.add_batch(obs_vec, acts, rews, buf_next, dones)
            obs_vec = nxt
            step += n_envs
            if buf.size >= cfg.batch_size and (cfg.warm_start or step > cfg.start_steps):
                pending += n_envs                            # preserve UTD: total updates = total_steps/update_every
                while pending >= cfg.update_every:
                    _update_once()
                    pending -= cfg.update_every
            if cfg.log_every > 0 and step - step_last >= cfg.log_every:
                _log(step)
            if step >= next_eval:
                _eval(step)
                next_eval += cfg.eval_every
        if dev.type != "cpu":                                # return nets on CPU (device-agnostic downstream)
            actor.to("cpu")
            for _c in critics:
                _c.to("cpu")
        return history

    obs, _ = env.reset(seed=cfg.seed)
    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.start_steps and not cfg.warm_start:
            action = rng.uniform(-scale, scale, size=actor.head.out_features).astype(np.float32)
        else:                                                      # warm-started: act with the cloned actor now
            with torch.no_grad():
                mu = actor(torch.as_tensor(obs[None], dtype=torch.float32, device=dev)).squeeze(0).cpu().numpy()
            action = np.clip(mu + rng.normal(0, sigma, mu.shape), -scale, scale).astype(np.float32)
        nobs, rew, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, float(rew), nobs, bool(terminated))   # truncation is NOT a true terminal
        obs = nobs if not (terminated or truncated) else env.reset()[0]

        if (step % cfg.update_every == 0 and buf.size >= cfg.batch_size
                and (cfg.warm_start or step > cfg.start_steps)):
            _update_once()
        if cfg.log_every > 0 and step % cfg.log_every == 0:
            _log(step)
        if step % cfg.eval_every == 0:
            _eval(step)
    if dev.type != "cpu":                                          # return the trained nets on CPU so downstream
        actor.to("cpu")                                            # eval/save/render are device-agnostic (the
        for _c in critics:                                         # caller never has to know training used a GPU)
            _c.to("cpu")
    return history


def run_offpolicy(algo: str = "ddpg", kind: str = "hsikan", *, hidden: int = 64, seed: int = 0,
                  n_eval: int = 20, cfg: OffPolicyConfig | None = None,
                  save: str | None = None) -> dict[str, float | str]:
    """Build env + actor/critics, train DDPG or TD3, evaluate. Mirrors ``run_balance`` (PPO) for comparison."""
    if algo not in ("ddpg", "td3"):
        raise ValueError(f"algo must be 'ddpg' or 'td3'; got {algo!r}")
    if cfg is None:
        cfg = td3_config(seed=seed) if algo == "td3" else OffPolicyConfig(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    space_shape = env.observation_space.shape
    assert space_shape is not None
    n_vertices, feat = int(space_shape[0]), int(space_shape[1])
    kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, critics = build_offpolicy(kind, obs_dim=feat, flat_dim=n_vertices * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=cfg.n_critics, hidden=hidden, **kw)
    floor = eval_balance(InvertedPendulumEnv(mjcf=mj), actor, n_eval, seed=10_000)
    history = train_offpolicy(actor, critics, env, cfg)
    upright = eval_balance(InvertedPendulumEnv(mjcf=mj), actor, n_eval, seed=20_000)
    if save is not None:
        from hymeko_rl.agents.policy_store import policy_to_hymeko
        policy_to_hymeko(actor.state_dict(), save, meta={
            "algo": algo, "backbone": kind, "upright": round(upright, 1), "seed": seed})
    return dict(algo=algo, policy=kind, n_critics=cfg.n_critics,
                n_params=sum(p.numel() for p in actor.parameters()),
                untrained_upright_steps=round(floor, 2), upright_steps=round(upright, 2),
                curve=json.dumps([round(h, 1) for h in history]), max_steps=float(env.max_steps))


def run_ddpg(kind: str = "hsikan", **kw: object) -> dict[str, float | str]:
    """DDPG preset (single critic, no delay/smoothing)."""
    return run_offpolicy("ddpg", kind, **kw)   # type: ignore[arg-type]


def run_td3(kind: str = "hsikan", **kw: object) -> dict[str, float | str]:
    """TD3 preset (twin critics, delayed policy, target smoothing)."""
    return run_offpolicy("td3", kind, **kw)   # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algo", default="td3", choices=["ddpg", "td3"])
    ap.add_argument("--policy", default="hsikan", choices=list(POLICY_KINDS))
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="store the trained policy to this .hymeko (with provenance)")
    a = ap.parse_args(argv)
    cfg = td3_config(total_steps=a.steps, seed=a.seed) if a.algo == "td3" \
        else OffPolicyConfig(total_steps=a.steps, seed=a.seed)
    print(json.dumps(run_offpolicy(a.algo, a.policy, hidden=a.hidden, seed=a.seed, cfg=cfg,
                                   save=a.save), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
