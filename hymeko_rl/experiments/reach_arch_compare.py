"""Architecture comparison: HSiKAN vs MLP on the canonical hymeko-emitted 6-DOF arm.

The result that goes back to Kato. Behaviour cloning from the DLS-IK expert — same task,
same demonstrations, same node-feature observation; **only the backbone differs**. The HSiKAN
policy message-passes over the kinematic hypergraph; the MLP baseline reads the identical obs
flattened. We report the learned policy's final end-effector error (median / IQR / worst over
seeds) and the parameter count — the structural-prior and param-efficiency claims.

    python -m hymeko_rl.experiments.reach_arch_compare --mode smoke   # 1 seed — sanity + cost
    python -m hymeko_rl.experiments.reach_arch_compare --mode full    # 5 seeds — the result

Fully reproducible: each (backbone, seed) seeds the policy init, the BC training, and the env
RNG, so both backbones face identical reaching targets at a given seed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from hymeko_rl.train.bc import run_bc
from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.env.arm_world import emit_arm_mjcf
from hymeko_rl.env.observation import ObservationSpec
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.evaluate import now_stamp

_REPO = Path(__file__).resolve().parents[2]
# The tractable canonical arm: a 4-DOF .hymeko robot, position-controlled, sized so BC clones
# reliably (arm_world-equivalent kinematics, but emitted from .hymeko). The 6-DOF
# anthropomorphic arm under torque is not BC-learnable (compounding) — see the report.
_REACH = _REPO / "data" / "robotics" / "reach_arm.hymeko"
_ROBOT = _REPO / "data" / "robotics" / "anthropomorphic_arm.hymeko"
_OBS = _REPO / "data" / "robotics" / "arm_reach_observation.hymeko"
_TASK = _REPO / "data" / "robotics" / "arm_reach_task.hymeko"


def emitted_arm_factory(
    robot: str | Path = _REACH, obs_profile: str | Path = _OBS,
    task_profile: str | Path = _TASK, *, control_mode: str = "position",
    ee_body: str = "flange", **env_kw: Any,
) -> Callable[[], ArmReachEnv]:
    """A factory of fresh envs for a hymeko-emitted arm — emit + parse the profiles ONCE, then
    build identical envs on demand (the comparison needs fresh envs per seed/eval). Defaults to
    the tractable canonical reach arm under position control.

    # Preconditions the robot / profile ``.hymeko`` exist and the CLI is built.
    # Postconditions each call returns a fresh env on the emitted (canonical) geometry.
    """
    mjcf = emit_arm_mjcf(robot, control_mode=control_mode)
    obs_spec = ObservationSpec.from_hymeko(obs_profile)
    reward_spec = RewardSpec.from_hymeko(task_profile)

    def make() -> ArmReachEnv:
        return ArmReachEnv(mjcf=mjcf, obs_spec=obs_spec, reward_spec=reward_spec,
                           control_mode=control_mode, ee_body=ee_body, **env_kw)

    return make


def _iqr(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    q1, _q2, q3 = statistics.quantiles(xs, n=4)
    return q3 - q1


def compare_backbones(
    env_factory: Callable[[], ArmReachEnv], *, seeds: Sequence[int],
    backbones: Sequence[str] = ("hsikan", "mlp"), n_demos: int = 48,
    n_epochs: int = 150, hidden: int = 64,
) -> dict[str, dict[str, Any]]:
    """Run BC for each backbone × seed; aggregate the learned final EE error.

    # Postconditions one entry per backbone with ``reach_{median,iqr,worst}`` (m),
    ``floor_median`` (m), ``n_params``, and the per-seed errors.
    """
    out: dict[str, dict[str, Any]] = {}
    for kind in backbones:
        runs = [run_bc(kind, env_factory=env_factory, seed=s, n_demos=n_demos,
                       n_epochs=n_epochs, hidden=hidden) for s in seeds]
        reach = sorted(float(r["reach_err_m"]) for r in runs)
        floor = statistics.median(float(r["untrained_floor_m"]) for r in runs)
        out[kind] = dict(
            n_params=int(runs[0]["n_params"]),
            reach_median=round(statistics.median(reach), 4),
            reach_iqr=round(_iqr(reach), 4),
            reach_worst=round(max(reach), 4),
            floor_median=round(floor, 4),
            per_seed=[round(x, 4) for x in reach],
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", default=None, help="write the JSON report to this path")
    a = ap.parse_args(argv)

    seeds = [0] if a.mode == "smoke" else list(range(5))
    factory = emitted_arm_factory()
    start = time.perf_counter()
    result = compare_backbones(factory, seeds=seeds, n_epochs=a.epochs, hidden=a.hidden)
    wall = time.perf_counter() - start
    try:
        import psutil
        rss_mb: float | None = round(psutil.Process().memory_info().rss / 1e6, 1)
    except ImportError:
        rss_mb = None

    report: dict[str, Any] = dict(
        timestamp=now_stamp(), mode=a.mode, arm="reach_arm (emitted, 4-DOF, position-controlled)",
        seeds=seeds, n_epochs=a.epochs, hidden=a.hidden, wall_s=round(wall, 1), rss_mb=rss_mb,
        result=result)
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
