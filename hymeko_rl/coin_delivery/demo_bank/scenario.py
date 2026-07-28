"""Coin/target scenarios for the R11.3 demonstration bank, with disjoint train/dev/test splits.

A scenario is a coin pose + an optional target pose (``None`` = the model's canonical delivery zone) with a stable ID and a
split assignment. Splits are disjoint *by construction* — a scenario belongs to exactly one of TRAIN/DEV/TEST (or the
separate INVALID rejection panel) — so there is no interpolation leakage between them. The pilot set is small
(~12 admissible + an invalid panel) and spans the staged generalization axes: canonical, coin+target translated together
(G0), shifted coin with the canonical relative vector, fixed coin with a changed target direction (G1), a diagonal case,
and an invalid initial condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

CANONICAL_COIN = np.array([0.07578387, 0.14279214], dtype=np.float64)


class ScenarioSplit(Enum):
    """Disjoint scenario partitions. INVALID is the rejection panel — never in a success denominator."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    INVALID = "invalid"


@dataclass(frozen=True)
class CoinTargetScenario:
    """One demonstration scenario. ``target_xy is None`` means the canonical model zone (no relocation)."""

    scenario_id: str
    coin_xy: NDArray[np.float64]
    target_xy: "NDArray[np.float64] | None"
    split: ScenarioSplit
    kind: str

    def __post_init__(self) -> None:
        assert self.coin_xy.shape == (2,), "coin_xy must be a 2-vector"
        assert self.target_xy is None or self.target_xy.shape == (2,), "target_xy must be None or a 2-vector"

    def relative_goal(self, zone_xy: NDArray[np.float64]) -> NDArray[np.float64]:
        """The relative goal vector target - coin (target = ``target_xy`` if set, else the passed canonical ``zone_xy``).
        Postcondition: a 2-vector."""
        target = zone_xy if self.target_xy is None else self.target_xy
        return np.asarray(target, np.float64) - self.coin_xy


# Measured canonical delivery zone (rl.inner._zone_x/_y) and the canonical relative delivery vector (zone - coin).
CANONICAL_ZONE = np.array([0.006126, 0.174274], dtype=np.float64)
CANONICAL_RELATIVE = CANONICAL_ZONE - CANONICAL_COIN          # ~[-0.0697, +0.0315] m, |.|=76.4 mm, angle ~156 deg
CANONICAL_ANGLE = float(np.arctan2(CANONICAL_RELATIVE[1], CANONICAL_RELATIVE[0]))


def _c(dx: float, dy: float) -> NDArray[np.float64]:
    return CANONICAL_COIN + np.array([dx, dy], dtype=np.float64)


def _translate(dx: float, dy: float) -> "NDArray[np.float64]":
    """C1/C2 target that preserves the canonical relative delivery vector: target = canonical_zone + (dx, dy)."""
    return CANONICAL_ZONE + np.array([dx, dy], dtype=np.float64)


def _dir_target(r: float, d_angle_deg: float) -> "NDArray[np.float64]":
    """C3 target at distance ``r`` and direction (canonical angle + offset) from the canonical coin."""
    a = CANONICAL_ANGLE + np.radians(d_angle_deg)
    return CANONICAL_COIN + r * np.array([np.cos(a), np.sin(a)], dtype=np.float64)


def build_pilot_scenarios() -> list[CoinTargetScenario]:
    """The R11.3 pilot set: 12 admissible coin/target scenarios across disjoint splits + a 2-case invalid panel.

    Curriculum: C0 canonical; C1 coin+target translated together (relative vector preserved); C2 varied coin, canonical
    relative vector; C3 fixed coin, varied target direction/distance. Postcondition: scenario IDs are unique; each
    admissible scenario is assigned exactly one of TRAIN/DEV/TEST; INVALID entries are the only INVALID-split rows.
    """
    S = ScenarioSplit  # noqa: N806 (local alias for brevity in the literal table below)
    rows: list[tuple[str, "NDArray[np.float64]", "NDArray[np.float64] | None", ScenarioSplit, str]] = [
        # C0 — canonical coin + canonical target
        ("c0_canonical", CANONICAL_COIN, None, S.TRAIN, "canonical"),
        # C1 — coin + target translated together (relative delivery vector preserved)
        ("c1_horiz", _c(-0.03, 0.0), _translate(-0.03, 0.0), S.TRAIN, "translated_together"),
        ("c1_vert", _c(0.0, 0.02), _translate(0.0, 0.02), S.DEV, "translated_together"),
        ("c1_diag", _c(0.03, 0.03), _translate(0.03, 0.03), S.TEST, "translated_together"),
        # C2 — varied coin, canonical relative delivery vector (target = canonical_zone + coin offset)
        ("c2_left", _c(-0.02, 0.0), _translate(-0.02, 0.0), S.TRAIN, "shifted_coin"),
        ("c2_down", _c(0.0, -0.02), _translate(0.0, -0.02), S.DEV, "shifted_coin"),
        ("c2_right", _c(0.03, 0.0), _translate(0.03, 0.0), S.TEST, "shifted_coin"),
        ("c2_upleft", _c(-0.02, 0.02), _translate(-0.02, 0.02), S.TRAIN, "shifted_coin"),
        # C3 — fixed canonical coin, varied target direction (+-45 deg) and distance (5-9 cm)
        ("c3_near", CANONICAL_COIN, _dir_target(0.06, 0.0), S.TRAIN, "changed_target"),
        ("c3_ccw45", CANONICAL_COIN, _dir_target(0.07, 45.0), S.DEV, "changed_target"),
        ("c3_cw45", CANONICAL_COIN, _dir_target(0.05, -45.0), S.TEST, "changed_target"),
        ("c3_far", CANONICAL_COIN, _dir_target(0.09, 20.0), S.TRAIN, "changed_target"),
    ]
    scenarios = [CoinTargetScenario(sid, coin, tgt, split, kind) for sid, coin, tgt, split, kind in rows]
    scenarios += [
        CoinTargetScenario("invalid_right_5cm", _c(0.05, 0.0), None, S.INVALID, "invalid_initial_condition"),
        CoinTargetScenario("invalid_far", _c(0.20, 0.20), None, S.INVALID, "invalid_initial_condition"),
    ]
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids)), "scenario IDs must be unique"
    return scenarios


_BankRow = tuple[str, NDArray[np.float64], "NDArray[np.float64] | None", str]


def _c0_bank() -> list[_BankRow]:
    """4 near-canonical controls (canonical zone)."""
    offs = [(0.0, 0.0), (0.005, 0.0), (0.0, 0.005), (-0.005, -0.005)]
    rows: list[_BankRow] = [(f"bank_c0_{i}", _c(dx, dy), None, "canonical") for i, (dx, dy) in enumerate(offs)]
    return rows


def _c1_bank() -> list[_BankRow]:
    """16 translate-together cells (coin+target shift together; relative vector preserved)."""
    rows: list[_BankRow] = []
    for dx in (-0.03, -0.01, 0.01, 0.03):
        for dy in (-0.02, 0.0, 0.02, 0.03):
            rows.append((f"bank_c1_{dx:+.2f}_{dy:+.2f}", _c(dx, dy), _translate(dx, dy), "translated_together"))
    return rows


def _c2_bank() -> list[_BankRow]:
    """20 varied-coin cells on a half-grid (disjoint from C1), canonical relative vector (target = zone + coin offset)."""
    rows: list[_BankRow] = []
    for dx in (-0.025, -0.015, 0.015, 0.025):
        for dy in (-0.025, -0.015, 0.0, 0.015, 0.025):
            rows.append((f"bank_c2_{dx:+.3f}_{dy:+.3f}", _c(dx, dy), _translate(dx, dy), "shifted_coin"))
    return rows


def _c3_bank() -> list[_BankRow]:
    """24 fixed-coin cells, varied target direction (+-45 deg) and distance (5-9 cm)."""
    rows: list[_BankRow] = []
    for r in (0.05, 0.06, 0.07, 0.09):
        for da in (-45.0, -30.0, -15.0, 15.0, 30.0, 45.0):
            rows.append((f"bank_c3_r{int(r*100)}_a{int(da):+d}", CANONICAL_COIN, _dir_target(r, da), "changed_target"))
    return rows


def _assign_geometric_splits(rows: list[_BankRow]) -> list[CoinTargetScenario]:
    """Assign a contiguous 70/15/15 train/dev/test split within a stage, sorted by a geometric key so dev/test are
    held-out coordinate regions (extrapolation, not interpolation between train cells)."""
    ordered = sorted(rows, key=lambda r: (round(float(r[1][0]), 4), round(float(r[1][1]), 4)))
    n = len(ordered)
    n_train, n_dev = round(0.70 * n), round(0.15 * n)
    out = []
    for i, (sid, coin, tgt, kind) in enumerate(ordered):
        split = ScenarioSplit.TRAIN if i < n_train else (ScenarioSplit.DEV if i < n_train + n_dev else ScenarioSplit.TEST)
        out.append(CoinTargetScenario(sid, coin, tgt, split, kind))
    return out


def build_bank_scenarios() -> list[CoinTargetScenario]:
    """The bounded overnight bank: 64 candidate coin/target scenarios across C0(4)/C1(16)/C2(20)/C3(24), each stage split
    70/15/15 by geometric cells. (Admissibility is checked at run time; inadmissible candidates become rejection-panel
    entries, never counted in the success denominator.) Postcondition: 64 unique IDs."""
    scenarios: list[CoinTargetScenario] = []
    for stage in (_c0_bank(), _c1_bank(), _c2_bank(), _c3_bank()):
        scenarios.extend(_assign_geometric_splits(stage))
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids)) == 64, "bank must have 64 unique scenario IDs"
    return scenarios


_STAGE_BY_KIND = {
    "canonical": "C0", "translated_together": "C1", "diagonal": "C1",
    "shifted_coin": "C2", "edge": "C2", "changed_target": "C3", "invalid_initial_condition": "INVALID",
}


def curriculum_stage(kind: str) -> str:
    """Map a scenario ``kind`` to its curriculum stage C0/C1/C2/C3 (or INVALID). Postcondition: a known stage label."""
    return _STAGE_BY_KIND.get(kind, "C_UNKNOWN")


def split_ids(scenarios: list[CoinTargetScenario]) -> dict[ScenarioSplit, set[str]]:
    """Group scenario IDs by split. Postcondition: the returned sets are pairwise disjoint."""
    out: dict[ScenarioSplit, set[str]] = {s: set() for s in ScenarioSplit}
    for sc in scenarios:
        out[sc.split].add(sc.scenario_id)
    return out
