"""CIP baseline (Cao et al. ICLR 2025; Ito et al., katolab) on MetaWorld — SAC + counterfactual data augmentation.

**Arms: `plain` (canonical SAC) vs `cds` (SAC + counterfactual data augmentation).** The augmented arm is **CDS
ONLY — NOT full CIP**: it does not implement Cao's empowerment term (reverse/source policy + causal-weighted
intrinsic reward). Do not report it as "CIP"/"full CIP". A/B is through the ``ReplayAugmentor`` seat on
``train_sac`` — the only difference between the two arms is whether :class:`CdsReplayAugmentor` is attached, so the
comparison is clean (same trainer, same env, same obs-norm, same seed).

This is Direction **A** of the CIP-continuation arc (*get the baseline*); the LLM correction (Ito's contribution)
and the HyMeKo structural corrector are Directions B/C on top of this. It reuses the hardened repo scaffolding
(``train_sac``, ``build_sac``, ``_ObsNorm``, ``_sac_success_eval`` — no re-implemented trainer, §6.1); the only new
piece is the CIP augmentation (:mod:`hymeko_rl.eval.cip.cip_augment`).

Phase-1 use (Mac, watched smoke — proves the mechanism, not the curve)::

    python -m hymeko_rl.experiments.exp_metaworld_cip_baseline --task coffee-push --cds --steps 20000 --seed 0
    python -m hymeko_rl.experiments.exp_metaworld_cip_baseline --task coffee-push        --steps 20000 --seed 0

Phase-2 (kato15, gated): the same entry point at ``--steps 1000000`` over 8 seeds × {plain, cip} × {coffee-push,
dial-turn} is the reproduced baseline campaign — a GPU job, not a Mac run.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def _make_native_env(env_id: str, render_mode: "str | None" = None) -> Any:
    """The raw MetaWorld goal-observable env with its **native** dense reward (not the HyMeKo reward-override)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ENVS  # type: ignore[attr-defined]  # no stubs
        return ENVS[env_id](render_mode=render_mode)  # type: ignore[arg-type]  # metaworld narrows to a Literal


def _fit_native_obs_norm(env_id: str, policy_name: str, *, n: int = 8, max_steps: int = 500, seed: int = 0,
                         ) -> "tuple[np.ndarray, np.ndarray]":
    """Obs mean/std from scripted-expert rollouts on the native env (the obs-norm fix that made MetaWorld SAC learn).

    The scripted expert visits the broad, task-relevant obs distribution; normalising against it (std floor 0.05,
    applied by :class:`_ObsNorm`) is the near-constant-dim fix from the Stage-B optimizer-repair pass."""
    import metaworld.policies as mp
    rows: list[np.ndarray] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            env = _make_native_env(env_id)
            expert = getattr(mp, policy_name)()
            obs, _ = env.reset(seed=seed + 5_000 + i)
            for _ in range(max_steps):
                rows.append(np.asarray(obs, np.float32))
                act = np.clip(np.asarray(expert.get_action(obs), np.float32), -1.0, 1.0)
                obs, _r, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
    arr = np.asarray(rows, np.float32)
    return arr.mean(axis=0), arr.std(axis=0)


def run_cip_seed(task: str = "coffee-push", *, cip: bool = True, seed: int = 0, steps: int = 20_000,
                 device: str = "cpu", hidden: int = 256, horizon: int = 500, out_dir: "Path | None" = None,
                 refresh_every: int = 10_000, sample_n: int = 1500, n_swap_dims: int = 1,
                 eval_every: "int | None" = None, stable: bool = False,
                 corrected: bool = False) -> "dict[str, Any]":
    """One SAC seed on the native-reward MetaWorld task, with (``cip=True``) or without the CIP CDS augmentor.

    Returns the run summary (success curve + provenance + augmentor stats); checkpoints the actor.

    # Preconditions ``task`` in the Ito set (``coffee-push``/``dial-turn``; any GENERIC_TASKS key with a V3 policy
      also works). ``metaworld`` importable.
    """
    import torch

    from hymeko_rl.eval.cip.cip_augment import CdsReplayAugmentor, CipAugmentConfig
    from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS
    from hymeko_rl.eval.evaluate import experiment_dir
    from hymeko_rl.experiments.exp_metaworld_sac import _ObsNorm, _sac_success_eval
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    from hymeko_rl.train.flat_critic import build_flat_sac

    env_id = f"{task}-v3-goal-observable"
    policy_name = GENERIC_TASKS[task]
    out = out_dir or experiment_dir("reports/figures", f"metaworld_cip_{task}")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{'cds' if cip else 'plain'}_{task}_seed{seed}"   # 'cds' = SAC + counterfactual data aug (NOT full CIP)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, std = _fit_native_obs_norm(env_id, policy_name, seed=seed, max_steps=horizon)
        env = _ObsNorm(_make_native_env(env_id), mean, std)
        eval_env = _ObsNorm(_make_native_env(env_id), mean, std)   # dedicated — MetaWorld's reset-after-trunc contract
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        scale = float(np.max(np.abs(np.asarray(env.action_space.high, np.float64))))

    if corrected:
        # Coffee-Push-CORRECTED stack (2026-07-18 SB3 cross-impl audit). Two demonstrated fixes over the old --stable:
        #   (1) reward_norm=False  -> removes the +60 Q-vs-MC calibration inflation on dense reward (was +60, now +5);
        #   (2) early-concat flat critic (build_flat_sac) -> restores dQ/da -> reach-and-HOLD (fixes reach-then-regress);
        #   + SB3-matched auto-alpha (init 1.0, lr 3e-4). Took S1 fixed-mug reach 0/4 -> 2/4 (vs SB3 4/4; residual = init/variance).
        actor, critics = build_flat_sac(obs_dim, act_dim, scale, hidden=hidden, device=device)
        cfg_kw: dict = dict(init_alpha=1.0, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, reward_norm=False)
    else:
        actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=scale,
                                   hidden=hidden, device=device)
        # OLD stable stack (reproduces kato14/15). WARNING: carries BOTH audited defects — reward_norm Q-inflation +
        # late-action-fusion QCritic (reach-then-regress). Kept only to reproduce the historical runs. Prefer --corrected.
        cfg_kw = dict(init_alpha=0.2, actor_lr=3e-4, critic_lr=3e-4) if stable else {}
    sac_cfg = SACConfig(total_steps=steps, seed=seed, log_every=max(1000, steps // 100),
                        eval_every=eval_every if eval_every is not None else max(2000, steps // 10),
                        start_steps=min(5000, steps // 4), **cfg_kw)  # type: ignore[arg-type]
    augmentor = None
    if cip:
        augmentor = CdsReplayAugmentor(obs_dim, act_dim, CipAugmentConfig(
            refresh_every=refresh_every, sample_n=sample_n, n_swap_dims=n_swap_dims, seed=seed))

    print(f"[cds-run] task={task} arm={'CDS' if cip else 'plain'} seed={seed} steps={steps} "
          f"obs_dim={obs_dim} act_dim={act_dim} device={device}", flush=True)
    curve = train_sac(actor, critics, env, sac_cfg,
                      eval_fn=_sac_success_eval(device, max_steps=horizon, eval_env=eval_env), augmentor=augmentor)
    torch.save(actor.state_dict(), out / f"{tag}.pt")
    summary: dict[str, Any] = {
        "task": task, "arm": "cds" if cip else "plain", "seed": seed, "steps": steps, "device": device,
        "hidden": hidden, "horizon": horizon, "obs_dim": obs_dim, "action_dim": act_dim,
        "success_curve": [round(c, 4) for c in curve],
        "final_success": round(curve[-1], 4) if curve else None,
        "best_success": round(max(curve), 4) if curve else None,
        "cds": augmentor.summary() if augmentor is not None else None,
    }
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[cds-run] {tag} done | final={summary['final_success']} best={summary['best_success']} "
          f"| cds={summary['cds']}", flush=True)
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="coffee-push", help="coffee-push | dial-turn | any GENERIC_TASKS key")
    ap.add_argument("--cds", "--cip", dest="cds", action="store_true",
                    help="attach the CDS counterfactual-data-augmentation arm (--cip = deprecated alias); else plain SAC")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=500, help="MetaWorld episode horizon (eval + obs-norm rollouts)")
    ap.add_argument("--refresh-every", type=int, default=10_000)
    ap.add_argument("--sample-n", type=int, default=1500)
    ap.add_argument("--n-swap-dims", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=None, help="eval cadence (default steps//10)")
    ap.add_argument("--stable", action="store_true", help="OLD stable stack (init_alpha 0.2, lr 3e-4) — reproduces kato14/15 (defective)")
    ap.add_argument("--corrected", action="store_true",
                    help="CORRECTED stack (2026-07-18 audit): reward_norm off + early-concat flat critic + init_alpha 1.0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run_cip_seed(a.task, cip=a.cds, seed=a.seed, steps=a.steps, device=a.device, hidden=a.hidden,
                 horizon=a.horizon, out_dir=Path(a.out) if a.out else None,
                 refresh_every=a.refresh_every, sample_n=a.sample_n, n_swap_dims=a.n_swap_dims,
                 eval_every=a.eval_every, stable=a.stable, corrected=a.corrected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
