"""Our train_sac (augmentor=None — the exact "plain SAC" code path) on Pendulum-v1, matched to the SB3 reference.

Two configs: (a) 'matched' = SB3-like (init_alpha 1.0 AUTO, reward_norm OFF); (b) 'coffeepush' = the config the
coffee-push plain baseline actually used (--stable: init_alpha 0.2, reward_norm ON). NO CIP: augmentor=None, no
reverse policy, no causal weights, no counterfactual augmentation. Same seed/steps/gamma/batch/lr/eval as SB3.
Runs in the PROJECT .venv (hymeko_rl + torch 2.12).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from hymeko_rl.train.sac import SACConfig, build_sac, train_sac, AlphaMode

OUT = Path(__file__).resolve().parent
SEED = 42
STEPS = 50_000


def eval_return(actor: object, n: int, seed0: int) -> "tuple[float, float]":
    """Deterministic (greedy action_mean) return over n Pendulum episodes."""
    env = gym.make("Pendulum-v1")
    rets = []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        total = 0.0
        for _ in range(200):
            with torch.no_grad():
                a = actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32))  # type: ignore[attr-defined]
            obs, r, term, trunc, _ = env.step(a.squeeze(0).numpy())
            total += float(r)
            if term or trunc:
                break
        rets.append(total)
    return float(np.mean(rets)), float(np.std(rets))


def run(name: str, cfg_kwargs: dict) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    env = gym.make("Pendulum-v1")
    actor, critics = build_sac("mlp", obs_dim=3, flat_dim=3, action_dim=1, action_scale=2.0, hidden=256)
    mb, sb = eval_return(actor, 10, 1000)
    cfg = SACConfig(total_steps=STEPS, seed=SEED, batch_size=256, gamma=0.99, tau=0.005,
                    actor_lr=3e-4, critic_lr=3e-4, start_steps=100, capacity=1_000_000,
                    eval_every=5000, log_every=5000, **cfg_kwargs)
    curve = train_sac(actor, critics, env, cfg, eval_fn=lambda _e, a: eval_return(a, 10, 20000)[0], augmentor=None)
    ma, sa = eval_return(actor, 20, 30000)
    obs_batch = np.stack([gym.make("Pendulum-v1").reset(seed=s)[0] for s in range(64)])
    with torch.no_grad():
        a, _ = actor.sample(torch.as_tensor(obs_batch, dtype=torch.float32))
    act_std = float(a.std().item())
    res = {
        "impl": "hymeko_rl.train.sac.train_sac (augmentor=None, no CIP)", "config": name,
        "cfg": {k: (v.value if isinstance(v, AlphaMode) else v) for k, v in cfg_kwargs.items()},
        "env": "Pendulum-v1", "seed": SEED, "steps": STEPS,
        "return_before_mean": round(mb, 2), "return_before_std": round(sb, 2),
        "return_after_mean": round(ma, 2), "return_after_std": round(sa, 2),
        "improvement": round(ma - mb, 2), "eval_curve": [round(c, 1) for c in curve],
        "final_action_std_sampled": round(act_std, 4),
        "PASS": bool((ma - mb) >= 500 and ma > -500 and np.isfinite(ma) and act_std > 1e-3),
    }
    print(json.dumps(res, indent=2), flush=True)
    return res


def main() -> int:
    results = [
        run("matched_sb3like", dict(init_alpha=1.0, alpha_mode=AlphaMode.AUTO, reward_norm=False)),
        run("coffeepush_config", dict(init_alpha=0.2, alpha_mode=AlphaMode.AUTO, reward_norm=True)),
    ]
    (OUT / "our_sac_result.json").write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"[our-sac] {r['config']}: before={r['return_before_mean']} after={r['return_after_mean']} "
              f"improvement={r['improvement']} PASS={r['PASS']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
