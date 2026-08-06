"""Our train_sac on the exact S1 env — 4 variants (auto-alpha / fixed-0.1 / fixed-0.2 / no-reward-norm), 2 seeds.

Decisive cross-impl test vs SB3 (s1_sb3.py) on the byte-identical S1 env. Grades success at the PHYSICAL contact
distance (`near_object` / d<7.5cm), because d<5cm is below the mug's contact geometry (expert min 6.85cm) — the 5cm
number is reported too, to show it is ~0 for BOTH implementations (impossible, not a defect). Rich per-eval logging.
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")
import s1_env  # noqa: E402
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac, AlphaMode  # noqa: E402

STEPS = 40_000
CONTACT_OK = 0.075  # physically-valid "reached contact" threshold (near_object fires ~0.069-0.073)


def eval_metrics(env, actor, critics, gamma, n=8):
    md, contact_rate, near_rate, act, logstds, q1s, q2s, tds, rews = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(n):
        obs, _ = env.reset(seed=70_000 + i)
        m, near = 1e9, 0
        steps = 0
        for _ in range(s1_env.HORIZON):
            st = torch.as_tensor(obs[None], dtype=torch.float32)
            with torch.no_grad():
                a = actor.action_mean(st).squeeze(0).numpy()
                a_s, _ = actor.sample(st)
                h = actor.backbone(st)
                logstds.append(float(actor.log_std(h).clamp(-20, 2).mean().item()))
                q1 = float(
                    critics[0](st, torch.as_tensor(a[None], dtype=torch.float32)).item()
                )
                q2 = float(
                    critics[1](st, torch.as_tensor(a[None], dtype=torch.float32)).item()
                )
            act.append(a_s.squeeze(0).numpy())
            nobs, r, term, trunc, info = env.step(a)
            m = min(m, info["d_eef_mug"])
            near += int(info.get("near_object", 0))
            rews.append(r)
            with torch.no_grad():
                a2, _ = actor.sample(torch.as_tensor(nobs[None], dtype=torch.float32))
                q2n = float(
                    torch.stack(
                        [
                            c(torch.as_tensor(nobs[None], dtype=torch.float32), a2)
                            for c in critics
                        ]
                    )
                    .amin(0)
                    .item()
                )
            q1s.append(q1)
            q2s.append(q2)
            tds.append(min(q1, q2) - (r + gamma * q2n))
            obs = nobs
            steps += 1
            if term or trunc:
                break
        md.append(m)
        contact_rate.append(int(m < CONTACT_OK))
        near_rate.append(near / max(1, steps))
    md = np.array(md)
    act = np.array(act)
    return {
        "min_d_median": round(float(np.median(md)), 4),
        "min_d_best": round(float(md.min()), 4),
        "success_contact_7_5cm": round(float(np.mean(contact_rate)), 3),
        "success_5cm_impossible": round(float(np.mean(md < 0.05)), 3),
        "frac_within_10cm": round(float(np.mean(md < 0.10)), 3),
        "frac_within_7_5cm": round(float(np.mean(md < 0.075)), 3),
        "frac_within_5cm": round(float(np.mean(md < 0.05)), 3),
        "frac_within_2cm": round(float(np.mean(md < 0.02)), 3),
        "near_object_rate": round(float(np.median(near_rate)), 4),
        "action_std": round(float(act.std()), 4),
        "policy_logstd_mean": round(float(np.mean(logstds)), 3),
        "q1_mean": round(float(np.mean(q1s)), 3),
        "q2_mean": round(float(np.mean(q2s)), 3),
        "q_min": round(float(min(np.min(q1s), np.min(q2s))), 3),
        "q_max": round(float(max(np.max(q1s), np.max(q2s))), 3),
        "td_abs_mean": round(float(np.mean(np.abs(tds))), 3),
        "raw_reward_range": [
            round(float(np.min(rews)), 3),
            round(float(np.max(rews)), 3),
        ],
    }


VARIANTS = {
    "ours_auto_alpha": dict(
        init_alpha=0.2, alpha_mode=AlphaMode.AUTO, reward_norm=True
    ),
    "ours_fixed_alpha_0.1": dict(
        init_alpha=0.1, alpha_mode=AlphaMode.FIXED, reward_norm=True
    ),
    "ours_fixed_alpha_0.2": dict(
        init_alpha=0.2, alpha_mode=AlphaMode.FIXED, reward_norm=True
    ),
    "ours_no_reward_norm": dict(
        init_alpha=0.2, alpha_mode=AlphaMode.AUTO, reward_norm=False
    ),
}


def run_variant(name, kw, mean, std, rv, seeds=(0, 1)):
    per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = s1_env.make_s1_env(mean, std, rv)
        actor, critics = build_sac(
            "mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=256
        )
        cfg = SACConfig(
            total_steps=STEPS,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            start_steps=1000,
            capacity=200_000,
            eval_every=8000,
            log_every=8000,
            target_entropy=-4.0,
            **kw,
        )
        log = []
        eenv = s1_env.make_s1_env(mean, std, rv)
        train_sac(
            actor,
            critics,
            env,
            cfg,
            eval_fn=lambda _e, a, _l=log, _ee=eenv, _c=critics: (
                _l.append(eval_metrics(_ee, a, _c, cfg.gamma))
                or _l[-1]["success_contact_7_5cm"]
            ),
        )
        per_seed.append(
            {
                "seed": seed,
                "curve": log,
                "final_contact_success": log[-1]["success_contact_7_5cm"],
                "best_min_d": min(m["min_d_best"] for m in log),
            }
        )
        print(
            f"[ours] {name} seed{seed}: contact_succ {[m['success_contact_7_5cm'] for m in log]} | "
            f"min_d {[m['min_d_median'] for m in log]} | 5cm {[m['frac_within_5cm'] for m in log]} | "
            f"alpha_mode={kw['alpha_mode'].value} rnorm={kw['reward_norm']}",
            flush=True,
        )
    cs = np.median([p["final_contact_success"] for p in per_seed])
    return {
        "impl": name,
        "final_contact_success_median": round(float(cs), 3),
        "SOLVES_reach_contact": bool(cs >= 0.5),
        "per_seed": per_seed,
    }


def main():
    mean, std, rv = s1_env.load_setup(str(HERE / "s1_setup.npz"))
    results = [run_variant(n, kw, mean, std, rv) for n, kw in VARIANTS.items()]
    (HERE / "s1_ours_result.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    print("\n[ours] SUMMARY", flush=True)
    for r in results:
        print(
            f"  {r['impl']:24s} contact_success={r['final_contact_success_median']} SOLVES={r['SOLVES_reach_contact']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
