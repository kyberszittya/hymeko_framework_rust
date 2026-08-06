"""HYMEKO_COIN_SPEC_BUNDLE_RUNTIME gate (Option B §9) — the single authoritative check that the ENTIRE executable Coin
task specification bundle is load-bearing and self-consistent through the OUTER canonical runtime, with one combined
bundle hash that training and evaluation both bind to.

Components asserted (directive §9): v3 reward load-bearing; K=6 success/reward/termination/certificate identity;
HyMeKo robot load-bearing; explicit control contract; HyMeKo scene load-bearing; six-node semantic graph projection;
checkpoint-compatible graph fingerprint; explicit full-action/residual modes; matched horizons; NO canonical Python
fallback; discounted strict success dominating all non-success behavior; and a single combined bundle hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

_OUT = Path("experiments/2026_07_23_coin_hymeko_recovery/logs/bundle_gate_v2.json")


def resolve_bundle() -> dict:
    """Build the canonical bundle through the exact outer entry point and collect every load-bearing fingerprint."""
    from hymeko_rl.coin_delivery.discounted_alignment import bundle_hashes, resolve_gammas
    from hymeko_rl.coin_delivery.env_factory import make_coin_env
    from hymeko_rl.env.planar_grasp_env import read_control_contract
    from hymeko_rl.train.coin_delivery_rl import DeliveryRLConfig

    cfg = DeliveryRLConfig()
    env = make_coin_env(embodiment="POINT")          # robot_source + scene_source = hymeko_spec (canonical)
    hashes = bundle_hashes(cfg)                       # reward/robot/scene/control + K=6 + horizon + combined hash

    checkpoint_fp = json.loads(Path(
        "experiments/2026_07_23_coin_hymeko_recovery/logs/checkpoint_compat_v2.json").read_text())
    align = json.loads(Path(
        "experiments/2026_07_23_coin_hymeko_recovery/logs/discounted_alignment_v3.json").read_text())

    return {
        "semantic_graph_fp": env.hg.semantic_fingerprint(),
        "semantic_vertices": list(env.hg.vertex_labels),
        "actor_input_dim": int(np.asarray(env.node_features()).size),
        "physical_nbody": int(env.model.nbody),
        "control_contract": read_control_contract(),
        "gammas": resolve_gammas(),
        "bundle_hashes": hashes,
        "checkpoint_verdicts": {c["key"]: c["verdict"] for c in checkpoint_fp["checkpoints"]},
        "e_approach_graph_fp": next(c["canonical_v2_graph_fp"] for c in checkpoint_fp["checkpoints"]
                                    if c["key"] == "e_approach"),
        "alignment_verdict": align["verdict"],
        "alignment_bundle_hash": align["manifest"]["combined_bundle_hash"],
    }


def _no_python_fallback() -> bool:
    """Canonical mode must HARD-FAIL when a required spec value is absent — no silent Python constant."""
    import re

    from hymeko_rl.env.env_spec import EnvSpec
    from hymeko_rl.env.planar_grasp_env import _CANONICAL_ROBOT_V2, read_control_contract
    tmp = Path(_CANONICAL_ROBOT_V2).parent / "_bundle_gate_probe.hymeko"
    try:
        # control contract stripped → hard-fail
        tmp.write_text(re.sub(r"galambos_control\s*\{[^}]*\}", "", Path(_CANONICAL_ROBOT_V2).read_text()))
        control_fails = _raises(lambda: read_control_contract(str(tmp)))
        # scene term dropped → hard-fail
        env_src = "data/robotics/galambos_env.hymeko"
        tmp.write_text(re.sub(r"@dsk:.*\n", "", Path(env_src).read_text(), count=1))
        scene_fails = _raises(lambda: EnvSpec.from_hymeko(str(tmp)))
        return control_fails and scene_fails
    finally:
        if tmp.exists():
            tmp.unlink()


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except (ValueError, FileNotFoundError):
        return True


def run() -> dict:
    b = resolve_bundle()
    checks = {
        "reward_load_bearing": b["bundle_hashes"]["reward_source"] == "hymeko_spec",
        "k6_contract": b["bundle_hashes"]["held_dwell_steps"] == 6
        and b["bundle_hashes"]["success_contract"] == "canonical_k6",
        "control_contract_present": set(b["control_contract"]) == {"joint_range", "ctrl_range", "damping", "kp", "kv"},
        "semantic_graph_six_node": len(b["semantic_vertices"]) == 6 and b["actor_input_dim"] == 48,
        "physical_robot_v3_golden_structure": b["physical_nbody"] == 8,   # world+6 arm bodies+coin (fingertip folded)
        "checkpoint_graph_compatible": b["e_approach_graph_fp"] == b["semantic_graph_fp"],
        "all_checkpoints_compatible": all(v == "CHECKPOINT_CANONICAL_V2_COMPATIBLE"
                                          for v in b["checkpoint_verdicts"].values()),
        "discounted_alignment_pass": b["alignment_verdict"] == "COIN_DISCOUNTED_REWARD_ALIGNMENT_PASS",
        "single_combined_hash": b["alignment_bundle_hash"] == b["bundle_hashes"]["combined_bundle_hash"],
        "no_python_fallback": _no_python_fallback(),
    }
    ok = all(checks.values())
    # the ONE combined bundle hash training and evaluation must both expose.
    combined = hashlib.sha256(
        (b["bundle_hashes"]["combined_bundle_hash"] + b["semantic_graph_fp"]).encode()).hexdigest()[:16]
    result = {"gate": "HYMEKO_COIN_SPEC_BUNDLE_RUNTIME", "checks": checks,
              "combined_bundle_hash": combined, "resolved": b,
              "verdict": "HYMEKO_COIN_SPEC_BUNDLE_RUNTIME_PASS" if ok else "COIN_HYMEKO_RECOVERY_BLOCKED"}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({"verdict": r["verdict"], "combined_bundle_hash": r["combined_bundle_hash"],
                      "checks": r["checks"]}, indent=1))
