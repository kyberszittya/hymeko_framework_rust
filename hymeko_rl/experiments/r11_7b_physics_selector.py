"""R11.7B — physics-matched θ selector on the dense box bank (SELECTION_LIMIT branch).

The dense-bank audit showed a delivering θ EXISTS for 4/6 dev snapshots but the descriptor-nearest / k3 selectors
miss it (it comes from a different train scenario). This selector scores each of the 66 stored θ by how well its
SOURCE handoff's *physics* features match the dev snapshot's — required transport (target−coin), coin velocity,
contact asymmetry, and handoff velocity — then picks the argmax and executes it UNCHANGED (no blend). It is compared,
on the SAME reached snapshot, against the descriptor-nearest deploy baseline and the pick-best-θ oracle ceiling.

Retrieval gate (R11_7B_BOX_RETRIEVAL_STABILIZATION): conditional strict-K6 ≥ 50% over delivery-reaching dev
snapshots, ≥3 seeds with K6, 0 safety regression, top-1 full θ, no runtime oracle.

Run:  python -m hymeko_rl.experiments.r11_7b_physics_selector [n_seeds]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer, clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals, reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_7a_u6b_box_pilot import BOX_DEV, _VARIANT

_OUT = Path("reports/2026-08-06-r11-7a-u6b-box-pilot")
_BANK = _OUT / "bank_dense.json"

# Physics features for the score (indices into the 30-D descriptor, see delivery_bc.dataset.descriptor):
#   required transport (target-coin) [14:16] · coin velocity [10:12] · handoff torque (prev_tau) [16:20] ·
#   contact asymmetry (left_right_contact_delay) [25] · handoff velocity (first/second_contact_relvel) [26:28].
_PHYS_IDX = np.array([14, 15, 10, 11, 16, 17, 18, 19, 25, 26, 27])
# required-transport is the dominant physics axis — up-weight it relative to the rest.
_PHYS_W = np.array([2.0, 2.0, 1.0, 1.0, 0.6, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0])


class PhysicsMatchedSelector:
    """Score stored θ by weighted, standardized distance on physics features of their source handoff; pick argmax.
    No blend, no training, no runtime oracle — a deployable top-1 selector."""

    def __init__(self, X: np.ndarray, thetas: np.ndarray) -> None:
        self._thetas = thetas
        self._std = Standardizer.fit(X[:, _PHYS_IDX])
        self._Z = self._std.transform(X[:, _PHYS_IDX]) * _PHYS_W

    def pick(self, x: np.ndarray) -> np.ndarray:
        z = self._std.transform(np.asarray(x, np.float64)[None, _PHYS_IDX])[0] * _PHYS_W
        return self._thetas[int(np.argmin(np.linalg.norm(self._Z - z, axis=1)))]


class NearestSelector:
    """The deploy baseline: standardized full-descriptor top-1 nearest."""

    def __init__(self, X: np.ndarray, thetas: np.ndarray) -> None:
        self._thetas, self._std = thetas, Standardizer.fit(X)
        self._Z = self._std.transform(X)

    def pick(self, x: np.ndarray) -> np.ndarray:
        z = self._std.transform(np.asarray(x, np.float64)[None, :])[0]
        return self._thetas[int(np.argmin(np.linalg.norm(self._Z - z, axis=1)))]


def _k6(snap: Any, theta: np.ndarray) -> "tuple[bool, float, bool]":
    s = _delivery_signals(snap, clip_theta(theta))
    return bool(s.k6), s.dtz_mm, bool(s.safe)


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    bank = json.loads(_BANK.read_text())
    X = np.asarray([s["x"] for s in bank["samples"]], np.float64)
    thetas = np.asarray([s["theta"] for s in bank["samples"]], np.float64)
    physics, nearest = PhysicsMatchedSelector(X, thetas), NearestSelector(X, thetas)
    # oracle ceiling per (scenario,seed) from the dense θ×handoff matrix (deterministic same snapshots) — avoids
    # re-rolling all 66 θ here.
    mat = json.loads((_OUT / "theta_handoff_matrix_dense.json").read_text())
    oracle = {(r["scenario_id"], r["seed"]): (r.get("n_delivering_theta", 0) > 0)
              for r in mat["rows"] if r.get("reached_delivery")}
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)

    rows: list[dict[str, Any]] = []
    for sid in BOX_DEV:
        for seed in range(n_seeds):
            h = reach_capture_descriptor(rig, scenario_by_id(sid), seed, cfg, conf, obj)
            if h.record is not None:
                rows.append({"scenario_id": sid, "seed": seed, "reached": False, "pre": h.record.outcome_class})
                print(f"[{sid:22s} s{seed}] PRE-DELIVERY {h.record.outcome_class}", flush=True)
                continue
            x = np.asarray(h.x, np.float64)
            pk6, pdtz, psafe = _k6(h.snap, physics.pick(x))
            nk6, ndtz, _ns = _k6(h.snap, nearest.pick(x))
            any_k6 = bool(oracle.get((sid, seed), False))    # ceiling from the dense matrix (same snapshot)
            rows.append({"scenario_id": sid, "seed": seed, "reached": True, "physics_k6": pk6, "physics_dtz": pdtz,
                         "physics_safe": psafe, "nearest_k6": nk6, "nearest_dtz": ndtz, "oracle_k6": any_k6})
            print(f"[{sid:22s} s{seed}] physics={pk6}({pdtz}) nearest={nk6}({ndtz}) oracle={any_k6}", flush=True)

    res = _summarize(rows)
    (_OUT / "physics_selector.json").write_text(json.dumps({"rows": rows, **res}, indent=1, default=str))
    print(f"\n{res['verdict']} | reached={res['n_reached']} | conditional K6: physics={res['physics_k6']}/{res['n_reached']} "
          f"nearest={res['nearest_k6']}/{res['n_reached']} oracle={res['oracle_k6']}/{res['n_reached']} | "
          f"physics seeds w/ K6={res['physics_seeds_with_k6']}")
    return 0 if res["gate_pass"] else 1


def _summarize(rows: list[dict]) -> dict[str, Any]:
    reached = [r for r in rows if r["reached"]]
    n = len(reached)
    pk6 = sum(1 for r in reached if r["physics_k6"])
    nk6 = sum(1 for r in reached if r["nearest_k6"])
    ok6 = sum(1 for r in reached if r["oracle_k6"])
    seeds_k6 = len({(r["scenario_id"], r["seed"]) for r in reached if r["physics_k6"]})
    unsafe = [r for r in reached if not r.get("physics_safe", True)]
    # gate: conditional K6 >= 50%, >=3 seeds with K6, 0 safety regression
    gate = bool(n > 0 and pk6 / n >= 0.5 and seeds_k6 >= 3 and not unsafe)
    return {"verdict": "R11_7B_BOX_RETRIEVAL_STABILIZATION_PASS" if gate else "R11_7B_BOX_RETRIEVAL_STABILIZATION_FAIL",
            "gate_pass": gate, "n_reached": n, "physics_k6": pk6, "nearest_k6": nk6, "oracle_k6": ok6,
            "physics_seeds_with_k6": seeds_k6, "n_safety_regressions": len(unsafe),
            "physics_cond_k6": round(pk6 / n, 3) if n else None}


if __name__ == "__main__":
    sys.exit(main())
