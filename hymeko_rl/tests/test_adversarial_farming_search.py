"""Option C — adversarial farming search: pure verdict logic + reach-oracle + (real-env) run."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hymeko_rl.eval.reward_repair.adversarial_farming_search import _hover, _verdict


def _metaworld_missing() -> bool:
    return importlib.util.find_spec("metaworld") is None


def test_hover_points_toward_object_gripper_open() -> None:
    obs = np.zeros(39, np.float32)
    obs[:3] = [0.0, 0.6, 0.2]
    obs[4:7] = [0.2, 0.6, 0.2]                 # object to +x
    a = _hover(obs, 0)
    assert a[0] > 0.5 and a[3] == -1.0         # move toward object, gripper OPEN (never grasp)


def test_verdict_flags_farming_plateau_and_removal() -> None:
    """A failed grasp-hold worth 77% of expert reward under original / 0.1% under monitor_aligned is flagged."""
    agg = {
        "expert_deliver": {"original_reward": 830.0, "monitor_aligned_reward": 3800.0, "success": 1.0},
        "grasp_hold": {"original_reward": 640.0, "monitor_aligned_reward": 5.0, "success": 0.0},
        "hover": {"original_reward": -30.0, "monitor_aligned_reward": -115.0},
        "cem_max_original": {"original_reward": 860.0, "success": 1.0},
        "cem_max_monitor_aligned": {"success": 1.0},
    }
    v = _verdict(agg)
    assert v["original_credits_failed_trajectory"] is True          # 640 > 0 with success 0
    assert v["monitor_aligned_penalizes_failed_trajectory"] is True # 5 ≤ 0.02·3800
    assert v["original_globally_hackable_by_cem"] is False          # CEM-max delivers
    assert v["monitor_aligned_max_delivers"] is True


@pytest.mark.skipif(_metaworld_missing(), reason="metaworld not installed")
def test_search_runs_and_flags_plateau(tmp_path) -> None:
    from hymeko_rl.eval.reward_repair.adversarial_farming_search import run_adversarial_search
    s = run_adversarial_search(layout_seeds=(0,), out_dir=tmp_path)
    a = s["aggregate"]
    # grasp-hold FAILS but collects most original reward; monitor_aligned gives it far less (relative to expert)
    assert a["grasp_hold"]["success"] == 0 and a["expert_deliver"]["success"] == 1
    gh = s["verdict"]["grasp_hold_reward_ratio_vs_expert"]
    assert gh["original"] > 2 * (gh["monitor_aligned"] if gh["monitor_aligned"] else 0.01)
    assert (tmp_path / "adversarial_farming_search.json").exists()
