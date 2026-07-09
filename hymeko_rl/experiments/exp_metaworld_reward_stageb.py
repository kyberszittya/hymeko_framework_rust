"""Stage B setup: a bounded training-smoke harness for the MetaWorld reward ablation. **GATED — no training here.**

Stage A / comparison / multi-seed established (at the reward-computation level) that ``mw_in_place`` is the dominant
reward driver (SUPPORTED 5/5) and ``mw_grasp`` is inert (NOT_SUPPORTED 5/5). Stage B asks the *policy-learning*
question this setup prepares for:

    Does a policy trained under ``mw_in_place_off`` behave differently from one trained under the original reward?

This module is **setup / dry-run only**. Training requires the explicit ``--launch-training`` flag; a reward whose
scripted demonstrator does not rank success above failure additionally requires ``--allow-uncertified`` (the
``mw_in_place_off`` reward is *expected* to trip this — removing the dominant term may break the reward's ability
to rank delivery, which is exactly the Stage-B hypothesis). The dry-run validates reward profiles, env
construction, the HyMeKo reward-override signal, logging paths, and the monitor/CIP post-eval, then exits **before**
the optimizer.

Reuses the Stage-A machinery (``ablate_reward_spec``, ``_TERM_TO_COMPONENT``, ``hymeko_reward``,
``run_reward_ablation_comparison``) and the MetaWorld goal-observable ENVS — no copied reward or eval code
(CLAUDE.md §6.1). The reward-override env wrapper and the minimal REINFORCE smoke are new because no flat-obs
MetaWorld trainer exists (the repo trainers require 2-D hypergraph observations).

    python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --dry-run          # validate everything, NO training
    python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --launch-training  # train (bounded smoke) — NOT RUN in setup
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.eval.cip.metaworld_reward import _DEFAULT_SPEC, _TERM_TO_COMPONENT
from hymeko_rl.eval.cip.reward_ablation_metaworld import AblatedRewardSpec, ablate_reward_spec

# profile name → reward terms to drop (empty = original). The Stage-B comparison arms.
_PROFILE_DROP: dict[str, tuple[str, ...]] = {
    "original": (),
    "mw_in_place_off": ("mw_in_place",),
    "mw_grasp_off": ("mw_grasp",),
    "mw_dist_off": ("mw_dist",),
}


class StageBGateError(RuntimeError):
    """Raised when training is requested without the explicit ``--launch-training`` safety flag."""


class StageBUncertifiedError(RuntimeError):
    """Raised when a training reward's scripted demonstrator does not rank success above failure and no waiver given."""


@dataclass(frozen=True)
class StageBConfig:
    """The bounded Stage-B training-smoke plan. All budgets are deliberately short — this proves plumbing, not skill."""
    task: str = "pick-place"
    profiles: tuple[str, ...] = ("original", "mw_in_place_off")
    spec_path: str = _DEFAULT_SPEC
    seed: int = 0
    n_seeds: int = 1                     # Stage-B is single-seed by design (a smoke, not a benchmark)
    total_env_steps: int = 3_000         # short budget
    max_steps: int = 180
    eval_episodes: int = 24
    hidden: int = 64
    gamma: float = 0.99
    lr: float = 3e-4
    log_every_ep: int = 5
    wall_time_cap_s: float = 600.0       # hard 10-min cap on the smoke
    cert_noise_max: float = 0.7          # action noise in certification so failures appear (a discriminating gate)
    cert_rollouts: int = 12
    out_dir: Path = field(default_factory=lambda: _default_out_dir())

    @property
    def env_id(self) -> str:
        return f"{self.task}-v3-goal-observable"

    def drop_for(self, profile: str) -> tuple[str, ...]:
        if profile not in _PROFILE_DROP:
            raise ValueError(f"unknown Stage-B profile {profile!r}; known {sorted(_PROFILE_DROP)}")
        return _PROFILE_DROP[profile]

    def log_dir(self, profile: str) -> Path:
        return self.out_dir / profile

    def checkpoint_path(self, profile: str) -> Path:
        return self.log_dir(profile) / "policy.pt"

    def eval_command(self, profile: str) -> str:
        """The exact monitor/CIP post-training evaluation command for a trained profile (run AFTER training)."""
        return (f"python -m hymeko_rl.experiments.exp_metaworld_reward_stageb --post-eval "
                f"--task {self.task} --profile {profile} --checkpoint {self.checkpoint_path(profile)}")


def _default_out_dir() -> Path:
    from hymeko_rl.eval.evaluate import experiment_dir
    # a Stage-B-specific dir — never collides with Stage A's *cip_reward_ablation* artifacts
    return experiment_dir("reports/figures", "metaworld_stageb_reward_ab")


def build_reward_profiles(cfg: StageBConfig) -> "dict[str, AblatedRewardSpec]":
    """Load the declared HyMeKo reward and its per-profile ablations (deterministic, non-mutating)."""
    return {name: ablate_reward_spec(cfg.spec_path, drop=cfg.drop_for(name)) for name in cfg.profiles}


def _weights_vector(spec: AblatedRewardSpec) -> np.ndarray:
    """The ablated weights as a vector aligned with ``spec.term_kinds()`` (same declaration order)."""
    return np.asarray(spec.ablated_weights(), dtype=np.float64)


class HymekoRewardMetaWorld:
    """A thin reward-override around a MetaWorld env: the step reward becomes the HyMeKo ``Σ weight·component``.

    The original env reward is preserved in ``info['env_reward']``; the recomputed HyMeKo reward in
    ``info['hymeko_reward']``. Duck-typed (``reset``/``step``/spaces delegate) so any policy loop can drive it.

    # Preconditions the wrapped env's ``info`` exposes the MetaWorld component keys in :data:`_TERM_TO_COMPONENT`.
    # Postconditions ``step`` returns a finite scalar reward equal to the profile's weighted component sum.
    """

    def __init__(self, env: Any, spec: AblatedRewardSpec) -> None:
        self.env = env
        self._kinds = list(spec.term_kinds())
        self._comp_keys = [_TERM_TO_COMPONENT[k] for k in self._kinds]
        self._weights = _weights_vector(spec)

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def reset(self, **kw: Any) -> Any:
        return self.env.reset(**kw)

    def hymeko_reward(self, info: "dict[str, Any]") -> float:
        comps = np.asarray([sign * float(info.get(key, 0.0)) for key, sign in self._comp_keys])
        return float(self._weights @ comps)

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        hy = self.hymeko_reward(info)
        info = {**info, "env_reward": float(reward), "hymeko_reward": hy}
        return obs, hy, terminated, truncated, info


def make_training_env(cfg: StageBConfig, spec: AblatedRewardSpec, *, render_mode: "str | None" = None) -> Any:
    """Construct the MetaWorld goal-observable env under a HyMeKo reward-override profile."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
        base = ENVS[cfg.env_id](render_mode=render_mode)  # type: ignore[arg-type]  # metaworld narrows to a Literal
    return HymekoRewardMetaWorld(base, spec)


def _roll_scripted(cfg: StageBConfig, spec: AblatedRewardSpec, policy: Any, seed_i: int, noise: float,
                   ) -> "tuple[float, bool]":
    """One noisy scripted episode under the reward-override env → (HyMeKo return, whether MetaWorld success fired).

    Action noise makes some episodes fail, so the success-vs-failure ranking below is discriminating (not vacuous)."""
    env = make_training_env(cfg, spec)
    obs, _ = env.reset(seed=seed_i)
    rng = np.random.default_rng(seed_i)
    ret = 0.0
    ok = False
    for _ in range(cfg.max_steps):
        act = np.asarray(policy.get_action(obs), np.float32) + rng.normal(0.0, noise, 4).astype(np.float32)
        obs, r, term, trunc, info = env.step(np.clip(act, -1.0, 1.0))
        ret += r
        ok = ok or bool(info.get("success", 0.0))
        if term or trunc:
            break
    return ret, ok


def _scripted_delivers(cfg: StageBConfig, spec: AblatedRewardSpec, *, n: "int | None" = None) -> "dict[str, Any]":
    """Does this reward rank delivery above failure? (the MetaWorld analogue of reward_oracle.certify's `delivers`).

    Rolls ``n`` noisy scripted episodes, splits by MetaWorld success, and checks the mean HyMeKo return is higher on
    successful episodes. This is the semantic gate CLAUDE.md §3 mandates before training — extended from the
    galambos oracle to MetaWorld's dense reward (the galambos abstract-MDP oracle does not model this task).
    ``delivers`` is only meaningful when both success and failure episodes appear; ``discriminating`` flags that."""
    import warnings

    from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS
    n = n if n is not None else cfg.cert_rollouts
    policy_name = GENERIC_TASKS[cfg.task]
    succ_ret: list[float] = []
    fail_ret: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import metaworld.policies as mp
        for i in range(n):
            ret, ok = _roll_scripted(cfg, spec, getattr(mp, policy_name)(), cfg.seed + i, cfg.cert_noise_max)
            (succ_ret if ok else fail_ret).append(ret)
    n_succ = len(succ_ret)
    discriminating = n_succ > 0 and len(fail_ret) > 0
    delivers = (n_succ > 0 and (not fail_ret or float(np.mean(succ_ret)) > float(np.mean(fail_ret))))
    return {"delivers": bool(delivers), "discriminating": bool(discriminating), "n_success": n_succ, "n_rollouts": n,
            "mean_return_success": round(float(np.mean(succ_ret)), 4) if succ_ret else None,
            "mean_return_failure": round(float(np.mean(fail_ret)), 4) if fail_ret else None}


def _validate_paths(cfg: StageBConfig) -> "dict[str, Any]":
    """Logging paths are creatable and never overwrite Stage-A (``*cip_reward_ablation*``) artifacts."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for p in cfg.profiles:
        cfg.log_dir(p).mkdir(parents=True, exist_ok=True)
    name = cfg.out_dir.name.lower()
    collides = "cip_reward_ablation" in name           # the Stage-A artifact family
    overwrites = any(cfg.checkpoint_path(p).exists() for p in cfg.profiles)
    return {"out_dir": str(cfg.out_dir), "collides_with_stage_a": collides, "overwrites_existing": overwrites,
            "log_dirs": {p: str(cfg.log_dir(p)) for p in cfg.profiles}}


def dry_run(cfg: StageBConfig, *, n_certify: int = 8) -> "dict[str, Any]":
    """Validate the whole Stage-B plan WITHOUT training: profiles, env+reward, certification, paths, eval commands.

    # Postconditions no optimizer runs, no checkpoint is written, ``trained`` is False."""
    profiles = build_reward_profiles(cfg)
    prof_report: dict[str, Any] = {}
    for name, spec in profiles.items():
        w = dict(spec.ablated)
        env = make_training_env(cfg, spec)
        obs, _ = env.reset(seed=cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        rewards = []
        for _ in range(3):                              # step the env a few times — env+reward validation, NOT training
            a = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            rewards.append(r)
            if term or trunc:
                break
        cert = _scripted_delivers(cfg, spec, n=n_certify)
        prof_report[name] = {"dropped": list(cfg.drop_for(name)), "weights": {k: round(v, 4) for k, v in w.items()},
                             "reward_finite": bool(np.all(np.isfinite(rewards))),
                             "certification": cert, "eval_command": cfg.eval_command(name)}
    report: dict[str, Any] = {"stage": "B", "mode": "dry-run", "trained": False, "task": cfg.task, "env_id": cfg.env_id,
              "seed": cfg.seed, "budget": {"total_env_steps": cfg.total_env_steps, "max_steps": cfg.max_steps,
                                           "eval_episodes": cfg.eval_episodes, "wall_time_cap_s": cfg.wall_time_cap_s},
              "profiles": prof_report, "paths": _validate_paths(cfg),
              "post_eval_commands": {p: cfg.eval_command(p) for p in cfg.profiles}}
    (cfg.out_dir / "stage_b_dry_run.json").write_text(json.dumps(report, indent=2, default=float))
    for name, pr in prof_report.items():
        c = pr["certification"]
        print(f"[stage-b dry-run] {name:16s} finite={pr['reward_finite']} delivers={c['delivers']} "
              f"(succ {c['n_success']}/{c['n_rollouts']})", flush=True)
    print(f"[stage-b dry-run] paths ok (no Stage-A collision={not report['paths']['collides_with_stage_a']}); "
          f"NO training launched.", flush=True)
    return report


class _GaussianMLP:
    """A tiny tanh-squashed Gaussian policy over flat obs — the minimal actor for the bounded REINFORCE smoke."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act_scale: float, *, seed: int) -> None:
        import torch
        torch.manual_seed(seed)
        self._torch = torch
        self._scale = act_scale
        self.net = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh())
        self.mean = torch.nn.Linear(hidden, act_dim)
        self.log_std = torch.nn.Parameter(torch.zeros(act_dim))

    def parameters(self) -> Any:
        return [*self.net.parameters(), *self.mean.parameters(), self.log_std]

    def _dist(self, obs: np.ndarray) -> Any:
        t = self._torch
        h = self.net(t.as_tensor(np.asarray(obs, np.float32)))
        return t.distributions.Normal(self.mean(h), self.log_std.exp())

    def sample(self, obs: np.ndarray) -> "tuple[np.ndarray, Any]":
        t = self._torch
        dist = self._dist(obs)
        raw = dist.sample()                 # DETACHED sample — REINFORCE needs the score-function gradient of
        logp = dist.log_prob(raw).sum()     # log_prob w.r.t. the params; a reparameterized rsample() would cancel it
        act = t.tanh(raw) * self._scale
        return act.detach().numpy().astype(np.float32), logp


def _returns_to_go(rewards: "list[float]", gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float64)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        out[i] = acc
    return out


def _train_reward_smoke(cfg: StageBConfig, env: Any, seed: int, *, log: Any) -> "dict[str, Any]":
    """A bounded REINFORCE smoke on the reward-override env. RUNTIME-UNVERIFIED until the first ``--launch-training``.

    Live-logged per §3 (episode, step k/total, return, loss, steps/s). NOT run during setup."""
    import time

    import torch
    np.random.seed(seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    scale = float(np.max(np.abs(np.asarray(env.action_space.high, np.float64))))
    policy = _GaussianMLP(obs_dim, act_dim, cfg.hidden, scale, seed=seed)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history: list[float] = []
    steps = ep = 0
    t0 = time.time()
    while steps < cfg.total_env_steps and (time.time() - t0) < cfg.wall_time_cap_s:
        obs, _ = env.reset(seed=seed + ep)
        logps: list[Any] = []
        rews: list[float] = []
        for _ in range(cfg.max_steps):
            act, logp = policy.sample(obs)
            obs, r, term, trunc, _ = env.step(act)
            logps.append(logp)
            rews.append(float(r))
            steps += 1
            if term or trunc:
                break
        g = torch.as_tensor(_returns_to_go(rews, cfg.gamma), dtype=torch.float32)
        g = (g - g.mean()) / (g.std() + 1e-8)
        loss = -(torch.stack(logps) * g).sum()
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch autograd is untyped
        opt.step()
        ep += 1
        history.append(float(sum(rews)))
        if ep % cfg.log_every_ep == 0:
            log(f"ep {ep} step {steps}/{cfg.total_env_steps} ret {history[-1]:.2f} "
                f"loss {float(loss):.3f} {steps / max(1e-6, time.time() - t0):.0f} st/s")
    return {"policy": policy, "returns": history, "env_steps": steps, "episodes": ep,
            "wall_s": round(time.time() - t0, 1)}


def launch(cfg: StageBConfig, *, launch_training: bool = False, allow_uncertified: bool = False) -> "dict[str, Any]":
    """Train the Stage-B profiles — GATED. Raises unless ``launch_training`` (and, per profile, certified or waived).

    # Preconditions ``launch_training`` is True; each profile's reward delivers, or ``allow_uncertified`` is set.
    # Postconditions returns a per-profile summary; a checkpoint is written per profile."""
    if not launch_training:
        raise StageBGateError(
            "refusing to train: pass launch_training=True (--launch-training). Stage B is gated by design.")
    import torch

    profiles = build_reward_profiles(cfg)
    summary: dict[str, Any] = {"stage": "B", "mode": "train", "trained": True, "results": {}}
    for name, spec in profiles.items():
        cert = _scripted_delivers(cfg, spec)
        if not cert["delivers"] and not allow_uncertified:
            raise StageBUncertifiedError(
                f"profile {name!r} reward does not rank success above failure ({cert}); pass allow_uncertified=True "
                f"(--allow-uncertified) to train on it deliberately (this is the mw_in_place_off hypothesis).")
        cfg.log_dir(name).mkdir(parents=True, exist_ok=True)
        env = make_training_env(cfg, spec)
        out = _train_reward_smoke(cfg, env, cfg.seed, log=lambda m, _n=name: print(f"[stage-b {_n}] {m}", flush=True))
        torch.save(out["policy"].net.state_dict(), cfg.checkpoint_path(name))
        summary["results"][name] = {"certification": cert, "episodes": out["episodes"], "env_steps": out["env_steps"],
                                    "wall_s": out["wall_s"], "final_return": out["returns"][-1] if out["returns"] else None,
                                    "checkpoint": str(cfg.checkpoint_path(name)), "eval_command": cfg.eval_command(name)}
    (cfg.out_dir / "stage_b_train.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="pick-place")
    ap.add_argument("--profiles", nargs="+", default=["original", "mw_in_place_off"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-env-steps", type=int, default=3_000)
    ap.add_argument("--dry-run", action="store_true", help="validate profiles/env/paths/eval and exit before optimizer")
    ap.add_argument("--launch-training", action="store_true", help="SAFETY GATE: actually train (bounded smoke)")
    ap.add_argument("--allow-uncertified", action="store_true",
                    help="train even if a reward does not rank success above failure (the mw_in_place_off hypothesis)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = StageBConfig(task=a.task, profiles=tuple(a.profiles), seed=a.seed, total_env_steps=a.total_env_steps)
    if a.out:
        cfg = replace(cfg, out_dir=Path(a.out))
    if a.launch_training:
        print(json.dumps({k: v for k, v in launch(cfg, launch_training=True,
                                                  allow_uncertified=a.allow_uncertified).items() if k != "policy"},
                         indent=2, default=str))
        return 0
    if not a.dry_run:
        print("[stage-b] no mode given; defaulting to --dry-run (use --launch-training to train). No training run.",
              flush=True)
    dry_run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
