"""Action-diverse replay + measured Monte-Carlo component targets — the fair-retest fix.

The prior vector-critic run was inconclusive for one measured reason: the DAgger replay was near-deterministic
(every stored action ≈ π_DAgger(s)), so no critic could learn how Q varies with the *action* — ∂Q/∂a was pure
extrapolation. This module removes that obstruction in two steps:

1. **State visitation** — roll the frozen actor with **phase-aware** exploration noise (ε ∈ a small sweep;
   APPROACH left undisturbed, CONTACT/PUSH/DELIVERY perturbed) so the visited states carry genuine action
   variation, with a qacc-divergence guard that drops (and counts) blown-up episodes.
2. **Measured MC targets** — for a prioritized sample of visited states, each candidate action (perturbed at each
   ε, plus a random OOD action) is branch-rolled under the *frozen* policy (:func:`branch_component_returns`) to
   get the **exact** discounted component return ``Q^\\pi_c(s, a)``. These are *measured* outcomes, not bootstrap
   guesses — and the random-action branches supply the off-manifold (OOD) shape for free.

No actor is trained here. Output feeds :func:`hymeko_rl.train.vector_critic.train_vector_critics_mc`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from hymeko_rl.env.planar_snapshot import PlanarSnapshot, branch_component_returns, snapshot_planar
from hymeko_rl.eval.critic_benchmark import phase_label
from hymeko_rl.train.search_objective import COMPONENTS, SearchObjective

_ENGAGED = ("CONTACT", "PUSH", "DELIVERY")   # phases where perturbation is applied + preferentially sampled
_PHASE_ID = {"APPROACH": 0, "CONTACT": 1, "PUSH": 2, "DELIVERY": 3}


@dataclass
class DiverseReplayConfig:
    """Knobs for action-diverse replay + measured MC targets. Smoke defaults are small; ``full`` scales them up."""

    n_visit_episodes: int = 40
    seed0: int = 4000                                   # NOT the eval seed 9000 — keep eval states unseen
    max_steps: int = 300
    eps_values: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02)
    approach_scale: float = 0.0                          # never destabilize APPROACH
    engaged_scale: float = 1.0                           # CONTACT/PUSH/DELIVERY perturbation magnitude
    progress_eps: float = 0.002
    near_coin: float = 0.06
    diverge_qacc: float = 5e3
    n_targets: int = 1500
    prioritize_engaged: bool = True
    engaged_fraction: float = 0.75                       # share of targets drawn from engaged phases
    n_ood_per_state: int = 1
    branch_horizon: int = 120
    gamma: float = 0.99
    seed: int = 0
    log_every: int = 250


@dataclass
class _Visited:
    obs: np.ndarray
    z: np.ndarray
    snap: PlanarSnapshot
    phase: str
    both_contact: bool
    arm_body: bool
    min_tip: float
    toward: float


@dataclass
class DiverseReplay:
    data: dict[str, np.ndarray]
    phase_counts: dict[str, int]
    n_visited: int
    n_targets: int
    n_diverged: int
    eps_used: tuple[float, ...]
    probe_pool: list["_Visited"] = field(default_factory=list)   # engaged (CONTACT/PUSH/DELIVERY) states w/ snapshots
    meta: dict[str, Any] = field(default_factory=dict)


def _greedy(actor: Any, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy().astype(np.float32)


def _visit(env: Any, actor: Any, cfg: DiverseReplayConfig, objective: SearchObjective,
           rng: np.random.Generator, log: Callable[[str], None]) -> tuple[list[_Visited], int, dict[str, int]]:
    """Stage A: roll the frozen actor with phase-aware exploration noise, snapshotting every visited state.

    # Postconditions: returns (visited states, #diverged episodes, phase histogram); diverged episodes contribute
      no states (their transitions are dropped, not silently truncated — the count is reported)."""
    lo, hi = env._ctrl_lo.astype(np.float32), env._ctrl_hi.astype(np.float32)
    visited: list[_Visited] = []
    n_diverged = 0
    phase_counts = {p: 0 for p in _PHASE_ID}
    for ep in range(cfg.n_visit_episodes):
        eps = cfg.eps_values[ep % len(cfg.eps_values)]
        obs, info = env.reset(seed=cfg.seed0 + ep)
        toward = 0.0
        ep_states: list[_Visited] = []
        diverged = False
        for _t in range(cfg.max_steps):
            z = env.privileged_state().astype(np.float32)
            m = env._planar_metrics
            both = bool(z[0] > 0.5 and z[1] > 0.5)
            in_zone = bool(z[4] > 0.5)
            min_tip = float(min(m.left_tip_dist, m.right_tip_dist))
            phase = phase_label(min_tip=min_tip, both_contact=both, toward=toward, in_zone=in_zone,
                                near_coin=cfg.near_coin, progress_eps=cfg.progress_eps)
            snap = snapshot_planar(env)
            base_a = _greedy(actor, obs)
            scale = cfg.approach_scale if phase == "APPROACH" else cfg.engaged_scale
            noise = rng.normal(0.0, eps * scale, size=base_a.shape).astype(np.float32) if eps * scale > 0 else 0.0
            act = np.clip(base_a + noise, lo, hi).astype(np.float32)
            ep_states.append(_Visited(obs=obs.astype(np.float32), z=z, snap=snap, phase=phase,
                                      both_contact=both, arm_body=bool(info.get("arm_body_contact_this_step", False)),
                                      min_tip=min_tip, toward=toward))
            nobs, _r, term, trunc, info = env.step(act)
            if float(np.max(np.abs(env.data.qacc))) > cfg.diverge_qacc:
                diverged = True
                break
            toward = max(0.0, float(snap.disk_to_zone) - float(info["disk_to_zone"])) if not np.isnan(snap.disk_to_zone) else 0.0
            obs = nobs
            if term or trunc:
                break
        if diverged:
            n_diverged += 1
            continue
        for s in ep_states:
            phase_counts[s.phase] += 1
        visited.extend(ep_states)
        if cfg.log_every and (ep + 1) % max(1, cfg.n_visit_episodes // 8) == 0:
            log(f"  [replay/visit] ep {ep + 1}/{cfg.n_visit_episodes} | visited={len(visited)} "
                f"| diverged={n_diverged} | phases={phase_counts}")
    return visited, n_diverged, phase_counts


def _sample_targets(visited: list[_Visited], cfg: DiverseReplayConfig,
                    rng: np.random.Generator) -> list[_Visited]:
    """Pick target states, preferring engaged (CONTACT/PUSH/DELIVERY) states so the critics see contact-rich
    action variation (where the scalar critic's gradient was measured to be monitor-misaligned)."""
    n = min(cfg.n_targets, len(visited))
    if not cfg.prioritize_engaged:
        idx = rng.choice(len(visited), size=n, replace=False)
        return [visited[i] for i in idx]
    engaged = [i for i, s in enumerate(visited) if s.phase in _ENGAGED]
    other = [i for i, s in enumerate(visited) if s.phase not in _ENGAGED]
    n_eng = min(len(engaged), int(round(cfg.engaged_fraction * n)))
    n_oth = min(len(other), n - n_eng)
    pick = list(rng.choice(engaged, size=n_eng, replace=False)) if n_eng else []
    pick += list(rng.choice(other, size=n_oth, replace=False)) if n_oth else []
    # top up from whichever pool has slack if one was short
    if len(pick) < n:
        rest = [i for i in range(len(visited)) if i not in set(pick)]
        extra = min(len(rest), n - len(pick))
        if extra:
            pick += list(rng.choice(rest, size=extra, replace=False))
    rng.shuffle(pick)
    return [visited[i] for i in pick]


def generate_action_diverse_replay(env: Any, frozen_actor: Any, cfg: DiverseReplayConfig, *,
                                   objective: SearchObjective | None = None,
                                   log: Callable[[str], None] = print) -> DiverseReplay:
    """Build the action-diverse replay + measured MC component targets.

    # Preconditions: ``frozen_actor`` maps obs→action for ``env``; ``env`` is a ``PlanarGraspEnv`` (snapshotable).
    # Postconditions: ``DiverseReplay.data`` holds, per (state, candidate-action) row: ``obs (N,V,8)``,
      ``action (N,nu)``, ``z (N,5)``, ``next_obs``, ``done``, ``z_next``, ``mc_r_{component} (N,)`` (measured),
      plus log columns ``is_ood, eps, perturb_norm, phase_id, both_contact, arm_body, min_tip, ft_ratio``. No actor
      or critic is trained. Determinism: fixed by ``cfg.seed`` / ``cfg.seed0``."""
    objective = objective or SearchObjective(near_coin=cfg.near_coin, progress_eps=cfg.progress_eps)
    rng = np.random.default_rng(cfg.seed)
    lo, hi = env._ctrl_lo.astype(np.float32), env._ctrl_hi.astype(np.float32)

    log(f"[replay] stage A: visiting states over {cfg.n_visit_episodes} eps, eps sweep {cfg.eps_values} ...")
    visited, n_diverged, phase_counts = _visit(env, frozen_actor, cfg, objective, rng, log)
    if not visited:
        raise RuntimeError("action-diverse replay collected zero visited states (all episodes diverged?)")
    targets = _sample_targets(visited, cfg, rng)
    log(f"[replay] stage B: branch-MC targets for {len(targets)} states "
        f"× ({len(cfg.eps_values)} eps + {cfg.n_ood_per_state} ood), horizon {cfg.branch_horizon} ...")

    cols: dict[str, list] = {k: [] for k in (
        "obs", "action", "next_obs", "done", "z", "z_next", "is_ood", "eps", "perturb_norm", "state_id",
        "phase_id", "both_contact", "arm_body", "min_tip", "ft_ratio", *(f"mc_r_{c}" for c in COMPONENTS))}

    def _add_branch(s: _Visited, act: np.ndarray, eps_tag: float, is_ood: bool, base_a: np.ndarray,
                    sid: int) -> None:
        ret, fs = branch_component_returns(env, frozen_actor, s.snap, act, objective=objective,
                                           gamma=cfg.gamma, horizon=cfg.branch_horizon)
        ft, body = fs["ft_prog_sum"], fs["body_prog_sum"]
        cols["obs"].append(s.obs)
        cols["action"].append(act.astype(np.float32))
        cols["next_obs"].append(fs["next_obs"])
        cols["done"].append(fs["done"])
        cols["z"].append(s.z)
        cols["z_next"].append(fs["z_next"])
        cols["is_ood"].append(np.float32(1.0 if is_ood else 0.0))
        cols["eps"].append(np.float32(eps_tag))
        cols["perturb_norm"].append(np.float32(np.linalg.norm(act - base_a)))
        cols["state_id"].append(np.int64(sid))
        cols["phase_id"].append(np.int64(_PHASE_ID[s.phase]))
        cols["both_contact"].append(np.float32(s.both_contact))
        cols["arm_body"].append(np.float32(s.arm_body))
        cols["min_tip"].append(np.float32(s.min_tip))
        cols["ft_ratio"].append(np.float32(ft / (ft + body + 1e-9)))
        for c in COMPONENTS:
            cols[f"mc_r_{c}"].append(np.float32(ret[c]))

    for i, s in enumerate(targets):
        base_a = _greedy(frozen_actor, s.obs)
        scale = cfg.approach_scale if s.phase == "APPROACH" else cfg.engaged_scale
        for eps in cfg.eps_values:
            noise = rng.normal(0.0, eps * scale, size=base_a.shape).astype(np.float32) if eps * scale > 0 else np.zeros_like(base_a)
            act = np.clip(base_a + noise, lo, hi).astype(np.float32)
            _add_branch(s, act, eps, False, base_a, i)
        for _ in range(cfg.n_ood_per_state):
            act = rng.uniform(lo, hi).astype(np.float32)
            _add_branch(s, act, -1.0, True, base_a, i)
        if cfg.log_every and (i + 1) % cfg.log_every == 0:
            log(f"  [replay/branch] target {i + 1}/{len(targets)} | rows={len(cols['obs'])}")

    data = {}
    for k, v in cols.items():
        arr = np.asarray(v)
        data[k] = arr.astype(np.int64) if k in ("phase_id", "state_id") else arr.astype(np.float32)
    log(f"[replay] done: {len(data['obs'])} (state,action) rows "
        f"({int(data['is_ood'].sum())} OOD) from {len(visited)} visited, {n_diverged} diverged eps")
    probe_pool = [s for s in visited if s.phase in _ENGAGED]
    return DiverseReplay(data=data, phase_counts=phase_counts, n_visited=len(visited),
                         n_targets=len(targets), n_diverged=n_diverged, eps_used=tuple(cfg.eps_values),
                         probe_pool=probe_pool,
                         meta={"branch_horizon": cfg.branch_horizon, "gamma": cfg.gamma})
