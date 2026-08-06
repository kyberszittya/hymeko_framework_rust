"""Setup + replay-only validation (the pre-training gate). Fit obs-norm, collect 5000 balanced demos, save, verify:
  1. all 5000 demos sample-able through a ReplayBuffer;
  2. obs use the exact training normalization; actions use the [-1,1] convention;
  3. successful demo transitions carry the expected (positive) reward + success flag;
  4. no transition crosses episode boundaries (collection is per-episode; next_obs is same-episode).
Saves demo_seed_setup.npz. Does NOT train.
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
warnings.simplefilter("ignore")
import harness  # noqa: E402
from hymeko_rl.train.replay import ReplayBuffer  # noqa: E402


def main() -> int:
    print("[setup] fitting obs-norm (scripted expert, random tasks)...", flush=True)
    mean, std = harness.fit_obsnorm(n=10)
    print(
        "[setup] collecting balanced demos (reach/contact/partial/success)...",
        flush=True,
    )
    demos, comp, verify = harness.collect_balanced_demos(mean, std, n_per_bucket=1250)
    obs, act, rew, nxt, done = demos

    checks = {}
    # 1. sample-able
    buf = ReplayBuffer(len(obs), (39,), 4)
    buf.add_batch(obs, act, rew, nxt, done)
    s = buf.sample(len(obs), generator=np.random.default_rng(0))
    checks["all_5000_sampleable"] = bool(int(s[0].shape[0]) == len(obs) == 5000)
    # 2. normalization + action convention
    checks["n_transitions"] = int(len(obs))
    checks["obs_normalized_range"] = [
        round(float(obs.min()), 2),
        round(float(obs.max()), 2),
    ]
    checks["obs_std_near_1"] = bool(
        0.3 < float(obs.std()) < 3.0
    )  # normalized obs ~ unit scale
    checks["actions_in_[-1,1]"] = bool(np.all(np.abs(act) <= 1.0 + 1e-6))
    checks["rewards_finite"] = bool(np.all(np.isfinite(rew)))
    # 3. successful transitions reward + flag
    checks["success_transition_reward"] = verify["reward_on_success_transition"]
    checks["success_flag_present_on_success"] = bool(verify["success_flag_present"])
    checks["success_reward_positive"] = bool(
        (verify["reward_on_success_transition"] or 0) > 0
    )
    # 4. boundary integrity: collection is strictly per-episode -> no cross-episode next_obs (structural)
    checks["no_boundary_leak_by_construction"] = bool(verify["no_boundary_leak"])
    # composition
    checks["composition"] = {
        k: comp[k] for k in ("reach", "contact", "partial", "success")
    }
    checks["balanced"] = bool(
        len({comp[k] for k in ("reach", "contact", "partial", "success")}) == 1
    )
    checks["episodes_rolled"] = comp["episodes_rolled"]
    checks["successful_episodes"] = comp["successful_episodes"]

    np.savez(
        str(HERE / "demo_seed_setup.npz"),
        mean=mean,
        std=std,
        obs=obs,
        act=act,
        rew=rew,
        nxt=nxt,
        done=done,
    )
    (HERE / "validation.json").write_text(json.dumps(checks, indent=2, default=float))
    ok = all(
        checks[k]
        for k in [
            "all_5000_sampleable",
            "obs_std_near_1",
            "actions_in_[-1,1]",
            "rewards_finite",
            "success_flag_present_on_success",
            "success_reward_positive",
            "balanced",
            "no_boundary_leak_by_construction",
        ]
    )
    print(json.dumps(checks, indent=2, default=float), flush=True)
    print(
        f"\n[setup] VALIDATION {'PASS' if ok else 'FAIL'} -> demo_seed_setup.npz",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
