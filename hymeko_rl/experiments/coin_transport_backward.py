"""§10 contingency — bounded short-horizon BACKWARD transport expansion (the live oracle found NO earlier frontier).

The frozen transport policy has a small near-goal basin; no state meaningfully farther from goal is deploy-solvable.
This fine-tunes a COPY of the frozen policy (the original stays frozen) on EARLIER bilateral-contact states — the
CAPTURE-produced states transport currently fails — using the env's native delivery-v2b reward, with strong RETENTION
rehearsal on the original basin. Accept the expanded copy only if the original state 04870b0e stays ≥8/10 AND the
earlier states become solvable. This is a bounded competence expansion, NOT a monolithic clear-start campaign; it adds
no critic / replay variant / n-step / factorial / dwell (§ constraints).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.provenance.state_identity import snapshot_hash
from hymeko_rl.env.planar_snapshot import snapshot_planar
from hymeko_rl.experiments.coin_bridge_relay import _restore_generated, build_basin, greedy_fn, load_transport_policy
from hymeko_rl.experiments.coin_clearance_curriculum import _CURDIR, _clearance
from hymeko_rl.experiments.coin_generator_exp import direct_env
from hymeko_rl.experiments.coin_option_chain import _dtz, option_banks
from hymeko_rl.experiments.coin_problem_generator import load_configs
from hymeko_rl.experiments.coin_two_arm_sac import policy_strict
from hymeko_rl.train.coin_delivery_actor import rollout
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

_STRONG = "04870b0e0357ecb5"
_TP = "experiments/2026_07_21_coin_clearance_curriculum/run_s0/actor_best.pt"


def _strict_pool(env, actor, snaps) -> int:
    return sum(int(bool(policy_strict(rollout(env, greedy_fn(actor), max_steps=60))))
               for s in snaps for _ in [_restore_generated(env, s)])


def run(steps: int, seed: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = direct_env()
    env._base_override = lambda _i, _t: np.zeros(6, np.float32)
    env._delta_override = 1.0
    tp = load_transport_policy()
    held1 = load_configs(_CURDIR / "STAGE1_held.pkl")
    strong = next(c for c in held1 if (_restore_generated(env, c.snapshot),
                  snapshot_hash(snapshot_planar(env.inner)).startswith(_STRONG[:12]))[1])
    labels, _det = build_basin(env, tp, [strong.snapshot] + [c.snapshot for c in held1], stride=2, n_robust=3)
    banks = option_banks(labels)
    basin = banks["T0_ready"]                                       # original near-goal basin (rehearsal)
    earlier = banks["C1_bilateral"]                                # earlier bilateral-contact states (expansion target)
    print(f"[backward] basin={len(basin)} earlier(C1)={len(earlier)}", flush=True)

    # baselines under the FROZEN transport policy
    frozen_strong = _strict_pool(env, tp, [strong.snapshot] * 10)
    frozen_earlier = _strict_pool(env, tp, earlier)
    print(f"[frozen] 04870b0e={frozen_strong}/10  earlier(C1) solvable={frozen_earlier}/{len(earlier)}", flush=True)

    # fine-tune a COPY on earlier states + >=30% original-basin rehearsal, env's native delivery-v2b reward
    copy_actor, critics = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    copy_actor.load_state_dict(torch.load(_TP, map_location="cpu"))
    rng = np.random.default_rng(seed)
    _orig = env.reset

    def _reset(*, seed=None):
        if seed is not None:
            return _orig(seed=seed)
        u = rng.random()                                           # 35% original-basin rehearsal (>=30%), else earlier
        snap = basin[rng.integers(len(basin))] if u < 0.35 else earlier[rng.integers(len(earlier))]
        _restore_generated(env, snap)
        return env._last_obs, {}
    env.reset = _reset
    print(f"[train] fine-tune transport COPY {steps} steps (delivery-v2b reward, 35% basin rehearsal)", flush=True)
    train_sac(copy_actor, critics, env, SACConfig.stable(total_steps=steps, seed=seed, bc_coef=0.0,
              log_every=max(steps, 1), eval_every=max(steps, 1) + 1))
    env.reset = _orig

    exp_strong = _strict_pool(env, copy_actor, [strong.snapshot] * 10)
    exp_earlier = _strict_pool(env, copy_actor, earlier)
    print(f"[expanded] 04870b0e={exp_strong}/10  earlier(C1) solvable={exp_earlier}/{len(earlier)}", flush=True)
    retained = exp_strong >= 8
    gained = exp_earlier > frozen_earlier and exp_earlier >= max(1, int(0.5 * len(earlier)))
    accept = retained and gained
    if accept:
        torch.save(copy_actor.state_dict(), out / "transport_expanded.pt")
    cls = "TRANSPORT_BACKWARD_POSITIVE" if accept else "NO_EARLY_FRONTIER"
    result = dict(seed=seed, steps=steps, basin=len(basin), earlier=len(earlier),
                  frozen_strong=frozen_strong, frozen_earlier_solvable=frozen_earlier,
                  expanded_strong=exp_strong, expanded_earlier_solvable=exp_earlier,
                  retained_original=retained, gained_earlier=gained, accepted=accept, classification=cls)
    (out / "backward_result.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"[result] retained={retained} gained={gained} accept={accept}\n=== CLASSIFICATION: {cls}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/2026_07_21_coin_live_frontier/backward")
    a = ap.parse_args()
    run(a.steps, a.seed, Path(a.out))


if __name__ == "__main__":
    main()
