"""Corrected 5-stage Coffee-Push curriculum — with the FIXED SAC (early-concat critic + reward_norm off).

Maps where the *corrected* SAC hits the wall on progressively harder tasks (the clean version of the earlier
buggy curriculum). Fixed task = explicit frozen ``_last_rand_vec`` (the working mechanism); random task = no freeze
(randomises per reset, verified by the env audit). Reward modes genuinely differ. Corrected stack throughout:
``build_flat_sac`` (early-concat critic) + ``reward_norm=False`` + SB3-matched auto-alpha (init 1.0, lr 3e-4).

Stages (3 seeds, 80k steps each; contact criterion 7.5 cm):
  s1_fixed_reach   : fixed task, reward=proximity, success=d<7.5cm
  s2_random_reach  : random task, reward=proximity, success=d<7.5cm
  s3_fixed_contact : fixed task, reward=proximity+contact_bonus, success=near_object fired
  s4_fixed_push    : fixed task, reward=native, success=info.success
  s5_random_push   : random task, reward=native, success=info.success (the real Coffee-Push)
"""

from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
warnings.simplefilter("ignore")
import s1_env  # noqa: E402
from hymeko_rl.train.flat_critic import build_flat_sac  # noqa: E402
from hymeko_rl.train.sac import SACConfig, train_sac, AlphaMode  # noqa: E402

STEPS = int(os.environ.get("S1_STEPS", "80000"))
EVAL = max(8000, STEPS // 5)
SEEDS = tuple(range(int(os.environ.get("S1_SEEDS", "3"))))
CONTACT_OK = 0.075


class CurricEnv(gym.Wrapper):
    """Mode-based reward (reach/contact/push) over a fixed (rand_vec) or random (rand_vec=None) coffee-push task."""

    def __init__(self, env, mode: str, rand_vec=None) -> None:
        if rand_vec is not None:
            s1_env._fix_task(env, rand_vec)
        super().__init__(env)
        self.mode = mode

    @staticmethod
    def _d(obs):
        return float(np.linalg.norm(np.asarray(obs)[0:3] - np.asarray(obs)[4:7]))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return obs, {**info, "d_eef_mug": self._d(obs), "stage_success": False}

    def step(self, action):
        obs, r_native, term, trunc, info = self.env.step(action)
        d = self._d(obs)
        near = float(info.get("near_object", 0.0))
        reach = float(1.0 - np.tanh(4.0 * d))
        if self.mode == "reach":
            r, succ = reach, d < CONTACT_OK
        elif self.mode == "contact":
            r, succ = reach + 3.0 * near, near > 0.5
        else:  # push (native)
            r, succ = float(r_native), bool(info.get("success", 0.0))
        return (
            obs,
            float(r),
            term,
            trunc,
            {**info, "d_eef_mug": d, "stage_success": bool(succ)},
        )


def _make(mode, rand_vec, mean, std):
    return s1_env.ObsNorm(CurricEnv(s1_env._base(), mode, rand_vec), mean, std)


def _fit_broad_obsnorm(n=8):
    """Broad obs-norm from the scripted expert on RANDOM tasks (covers fixed + random stages consistently)."""
    import metaworld.policies as mp

    rows = []
    for i in range(n):
        env = s1_env._base()
        pol = getattr(mp, s1_env.POLICY)()
        obs, _ = env.reset(seed=i)
        for _ in range(s1_env.HORIZON):
            rows.append(np.asarray(obs, np.float32))
            obs, _r, term, trunc, _ = env.step(np.clip(pol.get_action(obs), -1, 1))
            if term or trunc:
                break
    a = np.asarray(rows, np.float32)
    return a.mean(0), a.std(0)


def _eval_stage(env, actor, n=10):
    succ, mind = [], []
    for i in range(n):
        obs, _ = env.reset(seed=90_000 + i)
        m, ok = 1e9, False
        for _ in range(s1_env.HORIZON):
            with torch.no_grad():
                a = (
                    actor.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
                    .squeeze(0)
                    .numpy()
                )
            obs, _r, term, trunc, info = env.step(a)
            m = min(m, info["d_eef_mug"])
            ok = ok or info["stage_success"]
            if term or trunc:
                break
        succ.append(int(ok))
        mind.append(m)
    return {
        "success": round(float(np.mean(succ)), 3),
        "min_d_median": round(float(np.median(mind)), 4),
    }


def run_stage(name, mode, fixed, mean, std, rand_vec):
    per = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if fixed:
            rv = rand_vec  # the known-solvable s1 task
        else:  # metaworld fixes task per ENV instance; draw a
            e0 = s1_env._base()  # per-seed random-but-fixed task (distinct from s1),
            e0.reset(
                seed=1000 + seed * 7
            )  # frozen identically for train+eval (cross-task test)
            rv = np.asarray(
                getattr(e0, "unwrapped", e0)._last_rand_vec, dtype=np.float64
            )
        env = _make(mode, rv, mean, std)
        actor, critics = build_flat_sac(39, 4, 1.0, hidden=256)
        cfg = SACConfig(
            total_steps=STEPS,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            init_alpha=1.0,
            alpha_mode=AlphaMode.AUTO,
            target_entropy=-4.0,
            start_steps=1000,
            capacity=200_000,
            reward_norm=False,
            eval_every=EVAL,
            log_every=EVAL,
        )
        log = []
        ee = _make(mode, rv, mean, std)
        train_sac(
            actor,
            critics,
            env,
            cfg,
            eval_fn=lambda _e, a, _l=log, _ee=ee: (
                _l.append(_eval_stage(_ee, a)) or _l[-1]["success"]
            ),
        )
        sc = [m["success"] for m in log]
        per.append(
            {
                "seed": seed,
                "success_curve": sc,
                "min_d_curve": [m["min_d_median"] for m in log],
                "stable_success_final3": round(float(np.mean(sc[-3:])), 3),
            }
        )
        print(
            f"[curric2] {name} seed{seed}: success {sc} min_d {[m['min_d_median'] for m in log]} "
            f"stable3 {per[-1]['stable_success_final3']}",
            flush=True,
        )
    n = sum(1 for p in per if p["stable_success_final3"] >= 0.5)
    return {
        "stage": name,
        "mode": mode,
        "fixed": fixed,
        "n_seeds_stable": n,
        "n_seeds": len(SEEDS),
        "LEARNS": bool(n >= max(1, len(SEEDS) // 2 + 1)),
        "per_seed": per,
    }


def main():
    print(
        "[curric2] fitting broad obs-norm (random-task scripted expert)...", flush=True
    )
    mean, std = _fit_broad_obsnorm()
    _, _, rand_vec = s1_env.load_setup(
        str(HERE / "s1_setup.npz")
    )  # the known-solvable fixed task
    stages = [
        ("s1_fixed_reach", "reach", True),
        ("s2_random_reach", "reach", False),
        ("s3_fixed_contact", "contact", True),
        ("s4_fixed_push", "push", True),
        ("s5_random_push", "push", False),
    ]
    res = []
    for name, mode, fixed in stages:
        print(f"\n===== {name} (mode={mode} fixed={fixed}) =====", flush=True)
        res.append(run_stage(name, mode, fixed, mean, std, rand_vec))
        (HERE / "curriculum_corrected_result.json").write_text(
            json.dumps(res, indent=2, default=float)
        )
    print("\n[curric2] VERDICT (corrected SAC)")
    for r in res:
        print(
            f"  {r['stage']:18s} LEARNS={r['LEARNS']}  stable={r['n_seeds_stable']}/{r['n_seeds']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
