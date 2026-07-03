"""Discriminating diagnostic for the Galambos planar grasp "0 goals" symptom.

Rolls the trained policy (deterministic mean action) for several episodes and measures
*where the causal chain breaks*:
  - min fingertip->coin distance reached  (do the arms ever approach the coin?)
  - coin displacement from its placed spot (does the coin ever move?)
  - both-contact / in-zone frequency
  - reward decomposition: pull (disk->zone) vs contact vs zone vs action-cost

Hypothesis under test: the reward shapes only disk->zone (zero-gradient until contact) and
has no dense fingertip->coin approach term, so PPO collapses to minimising action cost — the
arms stop moving and the coin is never touched. Read-only; mutates no persistent state.

    uv run python -m hymeko_rl.experiments.diagnose_planar_grasp
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.agents.policy import build_policy

# Default to the baseline checkpoint; pass a path arg to diagnose a different run.
_CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/galambos/ppo.pt")


def _min_arm_dist_to_coin(env: PlanarGraspEnv, coin_xy: np.ndarray) -> float:
    arm_bodies = sorted(env._left_bodies | env._right_bodies)
    return min(
        float(np.hypot(env.data.xpos[b][0] - coin_xy[0], env.data.xpos[b][1] - coin_xy[1]))
        for b in arm_bodies
    )


def main() -> int:
    env = PlanarGraspEnv(max_steps=160)
    feat = int(env.observation_space.shape[-1])
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg,
                      hidden=64)
    loaded = False
    if _CKPT.exists():
        try:
            ac.load_state_dict(torch.load(_CKPT, map_location="cpu"))
            loaded = True
        except Exception as e:  # noqa: BLE001 - diagnostic, report and continue untrained
            print(f"[warn] checkpoint load failed ({e}); using fresh policy")
    ac.eval()
    print(f"policy: hsikan, params={ac.n_parameters()}, checkpoint_loaded={loaded}\n")

    n_ep = 8
    agg = {"min_tip": [], "disp": [], "init_dz": [], "final_dz": [],
           "contact_steps": [], "zone_steps": [], "moved_steps": []}
    for ep in range(n_ep):
        obs, _ = env.reset(seed=1000 + ep)
        coin0 = env._planar_metrics.disk_pos.copy()
        init_dz = env._planar_metrics.disk_to_zone
        min_tip = _min_arm_dist_to_coin(env, coin0)
        sums = {"pull": 0.0, "contact": 0.0, "zone": 0.0, "cost": 0.0}
        contact_steps = zone_steps = moved_steps = 0
        prev = coin0.copy()
        for _ in range(env.max_steps):
            with torch.no_grad():
                a = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
            act = a.squeeze(0).numpy()
            obs, _r, term, trunc, _info = env.step(act)
            m = env._planar_metrics
            min_tip = min(min_tip, _min_arm_dist_to_coin(env, m.disk_pos))
            sums["pull"] += -m.disk_to_zone
            both = m.left_contact and m.right_contact
            sums["contact"] += 0.5 * both
            sums["zone"] += 10.0 * m.in_zone
            sums["cost"] += 0.01 * -float(act @ act)
            contact_steps += int(both)
            zone_steps += int(m.in_zone)
            if float(np.hypot(*(m.disk_pos - prev))) > 1e-4:
                moved_steps += 1
            prev = m.disk_pos.copy()
            if term or trunc:
                break
        disp = float(np.hypot(*(env._planar_metrics.disk_pos - coin0)))
        agg["min_tip"].append(min_tip)
        agg["disp"].append(disp)
        agg["init_dz"].append(init_dz)
        agg["final_dz"].append(env._planar_metrics.disk_to_zone)
        agg["contact_steps"].append(contact_steps)
        agg["zone_steps"].append(zone_steps)
        agg["moved_steps"].append(moved_steps)
        total = sum(sums.values())
        print(f"ep{ep}: min_tip->coin={min_tip:.3f}m  coin_disp={disp:.3f}m  "
              f"dz {init_dz:.3f}->{env._planar_metrics.disk_to_zone:.3f}  "
              f"contact_steps={contact_steps}  zone_steps={zone_steps}  moved_steps={moved_steps}")
        print(f"      reward sum={total:.2f}  [pull={sums['pull']:.2f} contact={sums['contact']:.2f} "
              f"zone={sums['zone']:.2f} cost={sums['cost']:.3f}]")

    def med(k: str) -> float:
        return float(np.median(agg[k]))
    print("\n=== medians over episodes ===")
    print(f"min fingertip->coin : {med('min_tip'):.3f} m   (coin radius 0.035; touch needs <=0.05)")
    print(f"coin displacement  : {med('disp'):.3f} m   (init dist-to-zone {med('init_dz'):.3f})")
    print(f"contact steps      : {med('contact_steps'):.0f} / {env.max_steps}")
    print(f"zone steps         : {med('zone_steps'):.0f} / {env.max_steps}")
    print(f"coin-moved steps   : {med('moved_steps'):.0f} / {env.max_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
