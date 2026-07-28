"""Agile-embodiment semantics — widen_stance transform + the propulsion/turning tradeoff it exposes.

Certifies the semantic transform (the four hip-abduction offsets widen; non-positive rejected; the
committed variant emits) and the honest measured tradeoff: the wider-stance AIBO **turns better but
walks worse** — a single geometric parameter does not yield a model that both walks and turns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from scenarios.aibo.agile_embodiment import measure_forward_propulsion, widen_stance
from scenarios.aibo.locomotion_gait import SteeredTrotGait
from scenarios.aibo.motion_contract import JointVelocityGovernor
from scenarios.aibo.turn_authority import AgileTurnGait, stable_turn_ceiling

_BASE = Path("data/robotics/quadruped.hymeko")
_AGILE = Path("data/robotics/quadruped_agile.hymeko")


def test_widen_stance_sets_all_four_hip_offsets() -> None:
    out = widen_stance(_BASE.read_text(), 0.11)
    assert out.count("0.11") >= 4                       # the four hip-abduction lateral offsets
    assert "0.062" not in "".join(ln for ln in out.splitlines() if "@hip_abduct" in ln)


def test_widen_stance_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        widen_stance(_BASE.read_text(), 0.0)


def test_agile_variant_emits_and_builds() -> None:
    assert _AGILE.exists()
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                           max_steps=200, hymeko_path=str(_AGILE))
    assert env.model.nu == 12                            # same 12 leg actuators (only the stance widened)


def test_agile_turns_more_but_walks_less() -> None:
    # the honest tradeoff: the wide stance raises turn authority (fixes the wall) but cripples the
    # forward walk — regression-locking that a one-parameter stance edit is not a clean agility win.
    gait, turn_gait, gov = SteeredTrotGait(), AgileTurnGait(), JointVelocityGovernor(v_max=8.0)

    def env(path):
        return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                                max_steps=1200, hymeko_path=str(path))

    base_turn = stable_turn_ceiling(env(_BASE), turn_gait, gov,
                                    turns=(0.5, 0.7, 0.9))["fastest_stable_deg_per_1000"]
    agile_turn = stable_turn_ceiling(env(_AGILE), turn_gait, gov,
                                     turns=(0.5, 0.7, 0.9))["fastest_stable_deg_per_1000"]
    base_fwd = measure_forward_propulsion(env(_BASE), gait, gov)
    agile_fwd = measure_forward_propulsion(env(_AGILE), gait, gov)
    assert agile_turn > 1.5 * base_turn                  # much stronger stable turn (measured ~4.6×)
    assert agile_fwd < 0.7 * base_fwd                    # but weaker forward propulsion (~2.7× slower)
