"""From-scratch SAC on MetaWorld pick-place — the gated Option-E path (original vs monitor_aligned reward).

The flat-obs PPO could not robustly learn even reach from scratch (optimizer-limited). SAC is the sample-efficient
off-policy fix: this reuses the repo's hardened ``train_sac`` (twin soft-Q critics, auto-α, reward-norm, grad-clip,
GPU, live ``[sac]`` logging — no re-implemented trainer, §6.1) with an ``mlp`` actor on MetaWorld's flat obs, an
observation-normalization wrapper (the near-constant-object-dim fix from the optimizer-repair pass — std floor 0.05),
and the reward-override envs (original ``HymekoRewardMetaWorld`` / repaired ``MonitorAlignedEnv``).

    python -m hymeko_rl.experiments.exp_metaworld_sac --reward original --steps 300000 --seed 0
    python -m hymeko_rl.experiments.exp_metaworld_sac --reward monitor_aligned --steps 300000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


class _ObsNorm:
    """Standardize observations with a std floor (near-constant MetaWorld dims must not be amplified into noise)."""

    def __init__(self, env: Any, mean: np.ndarray, std: np.ndarray) -> None:
        self.env = env
        self._mean = mean.astype(np.float32)
        self._std = np.maximum(std, 0.05).astype(np.float32)

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    def _n(self, obs: Any) -> np.ndarray:
        out: np.ndarray = ((np.asarray(obs, np.float32) - self._mean) / self._std).astype(np.float32)
        return out

    def reset(self, **kw: Any) -> Any:
        obs, info = self.env.reset(**kw)
        return self._n(obs), info

    def step(self, action: Any) -> Any:
        obs, r, term, trunc, info = self.env.step(action)
        return self._n(obs), r, term, trunc, info


def _reward_env(cfg: Any, spec: Any, reward: str) -> Any:
    from hymeko_rl.eval.cip.monitor_aligned_reward import MonitorAlignedEnv
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import make_training_env
    base = make_training_env(cfg, spec)                               # HymekoRewardMetaWorld(original weights)
    return MonitorAlignedEnv(base) if reward == "monitor_aligned" else base


def _fit_obs_norm(cfg: Any, spec: Any, n: int = 12) -> "tuple[np.ndarray, np.ndarray]":
    """Fit obs mean/std from scripted-expert rollouts (broad, task-relevant obs distribution)."""
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import collect_scripted_demos
    obs, _acts, _ns = collect_scripted_demos(cfg, spec, n, only_success=False)
    return obs.mean(axis=0), obs.std(axis=0)


def _sac_success_eval(device: Any, n: int = 12, max_steps: int = 180, eval_env: Any = None) -> Any:
    """An ``eval_fn(env, actor)`` returning the greedy success rate over ``n`` episodes (the MetaWorld task monitor).

    ``eval_env``: evaluate on this *dedicated* env instead of the training env passed by ``train_sac``. This is
    **required for MetaWorld**, whose raw goal-observable env enforces a manual-reset-after-truncation contract —
    sharing the training env leaves it truncated after the eval's last episode, so the training loop's next
    ``step`` raises. Pass a fresh env (default ``None`` keeps the legacy shared-env behaviour, fine for cart-pole).
    """
    import torch

    def fn(env: Any, actor: Any) -> float:
        e = eval_env if eval_env is not None else env
        ok = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(n):
                obs, _ = e.reset()
                done = False
                for _ in range(max_steps):
                    with torch.no_grad():
                        a = actor.action_mean(torch.as_tensor(obs[None], dtype=torch.float32, device=device))
                    obs, _r, term, trunc, info = e.step(a.squeeze(0).cpu().numpy())
                    done = done or bool(info.get("success", 0.0))
                    if term or trunc:
                        break
                ok += int(done)
        return ok / max(1, n)
    return fn


def run_sac_seed(reward: str = "original", seed: int = 0, steps: int = 300_000, out_dir: "Path | None" = None,
                 device: str = "cpu", hidden: int = 256) -> "dict[str, Any]":
    """One from-scratch SAC seed on the reward-override MetaWorld env; checkpoints + eval curve + provenance."""
    import torch

    from hymeko_rl.eval.cip.reward_ablation_metaworld import ablate_reward_spec
    from hymeko_rl.eval.evaluate import experiment_dir
    from hymeko_rl.experiments.exp_metaworld_reward_stageb import StageBConfig
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    out = out_dir or experiment_dir("reports/figures", f"metaworld_sac_{reward}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = StageBConfig(task="pick-place", seed=seed)
    spec = ablate_reward_spec(cfg.spec_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = _fit_obs_norm(cfg, spec)
        env = _ObsNorm(_reward_env(cfg, spec, reward), mean, std)
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        scale = float(np.max(np.abs(np.asarray(env.action_space.high, np.float64))))
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=scale,
                               hidden=hidden, device=device)
    sac_cfg = SACConfig(total_steps=steps, seed=seed, log_every=max(1000, steps // 100),
                        eval_every=max(2000, steps // 30), start_steps=min(5000, steps // 4))
    curve = train_sac(actor, critics, env, sac_cfg, eval_fn=_sac_success_eval(device))
    torch.save(actor.state_dict(), out / f"sac_{reward}_seed{seed}.pt")
    summary = {"reward": reward, "seed": seed, "steps": steps, "device": device, "hidden": hidden,
               "obs_dim": obs_dim, "action_dim": act_dim, "success_curve": [round(c, 4) for c in curve],
               "final_success": round(curve[-1], 4) if curve else None,
               "best_success": round(max(curve), 4) if curve else None}
    (out / f"sac_{reward}_seed{seed}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[sac-run] reward={reward} seed={seed} steps={steps} device={device} | final_success={summary['final_success']} "
          f"best_success={summary['best_success']}", flush=True)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reward", default="original", choices=("original", "monitor_aligned"))
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_sac_seed(a.reward, a.seed, a.steps, Path(a.out) if a.out else None, a.device, a.hidden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
