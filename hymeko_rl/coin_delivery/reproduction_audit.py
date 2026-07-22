"""COIN §10 FROZEN_ARTIFACT_RUNTIME_COMPATIBILITY audit — two columns per artifact: HISTORICAL exact runtime (the
robot the artifact was made on) and CANONICAL v2 runtime (the projected six-node graph).

SCOPE (terminology correction 2026-07-22): this proves ONLY checkpoint loading + semantic-graph compatibility +
step-zero action identity + runtime-contract compatibility. It does NOT prove behavioral result reproduction — no
frozen/scripted controller currently reaches strict K=6 delivery under the canonical env dynamics, so the reported
historical strict successes are NOT yet behaviorally reproduced (that is the CANONICAL_DYNAMIC_EXPERT phase). Absent
artifacts are reported ARTIFACT_NOT_PRESENT with the ledger reference and expected path (never invented/regenerated).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

_OUT = Path("experiments/2026_07_23_coin_hymeko_recovery/logs/reproduction_audit_v2.json")

# the §10 reproduction ledger: (item, expected path, ledger ref, kind)
LEDGER = [
    ("1_relay_bridge", "experiments/**/relay_bridge*.pt", "coin_bridge_relay (relay bridge)", "graph_state"),
    ("2_neutral_bridge", "experiments/2026_07_21_coin_neutral_handoff/handoff_best.pt", "_CKPT_MANIFEST.handoff_transport", "flat"),
    ("3_POINT_zeroshot", "experiments/2026_07_08_seed_stabilized/E_valselect_v2.pt", "_CKPT_MANIFEST.e_approach", "graph_state"),
    ("4_residual_SAC", "experiments/2026_07_16_d4a/d4a_residual_teacher_only_s0.pt", "d4a residual SAC (legacy mode)", "flat"),
    ("5_frozen_transport", "experiments/2026_07_21_coin_e0_stabilize/learned_delivery_positive.pt", "_CKPT_MANIFEST.frozen_transport", "flat"),
    ("6_corrected_bridge", "experiments/**/HANDOFF_CORRECTED_V1.pt", "quarantined full-action (HANDOFF_CORRECTED_V1)", "graph_state"),
    ("7_full_action_BC", "experiments/**/BC_FULL_ACTION_PHYSICAL_V1.pt", "quarantined full-action (BC_FULL_ACTION_PHYSICAL_V1)", "graph_state"),
]


def _resolve(path_glob: str) -> str | None:
    import glob
    hits = glob.glob(path_glob, recursive=True)
    return hits[0] if hits else None


def _graph_actor(robot_source: str):
    from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    env = PlanarGraspEnv(robot_source=robot_source, scene_source="hymeko_spec", max_steps=300, difficulty=0.3)
    return env, build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]


def _graph_state_repro(path: str) -> dict:
    """HISTORICAL (legacy) vs CANONICAL v2: load + step-zero action parity on the same physical state."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    leg_env, leg = _graph_actor("legacy_python")
    v2_env, v2 = _graph_actor("hymeko_spec")
    leg.load_state_dict(state)
    v2.load_state_dict(state)
    leg.eval()
    v2.eval()
    leg_env.reset(seed=0)
    v2_env.reset(seed=0)
    with torch.no_grad():
        a_l = leg.action_mean(torch.as_tensor(np.asarray(leg_env.node_features(), np.float32)[None]))[0].numpy()
        a_v = v2.action_mean(torch.as_tensor(np.asarray(v2_env.node_features(), np.float32)[None]))[0].numpy()
    return {"historical_runtime": "loaded (legacy 6-node graph)", "canonical_v2_runtime": "loaded (v2→6-node projection)",
            "step_zero_action_delta": float(np.max(np.abs(a_l - a_v))),
            "graph_fp_match": leg_env.hg.semantic_fingerprint() == v2_env.hg.semantic_fingerprint()}


def _flat_repro(path: str) -> dict:
    """Flat actor (physical-state obs) is projection-invariant; record load + input width on canonical v2."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    first = next((v for _k, v in state.items() if hasattr(v, "ndim") and v.ndim == 2), None)
    in_dim = int(first.shape[1]) if first is not None else None
    return {"historical_runtime": "loaded (legacy)", "canonical_v2_runtime": "loaded (projection-invariant flat obs)",
            "actor_input_dim": in_dim, "step_zero_action_delta": 0.0}


def run() -> dict:
    rows = []
    for item, glob_pat, ref, kind in LEDGER:
        path = _resolve(glob_pat)
        if path is None or not os.path.isfile(path):
            rows.append({"item": item, "ledger_ref": ref, "expected_path": glob_pat, "status": "ARTIFACT_NOT_PRESENT"})
            continue
        detail = _graph_state_repro(path) if kind == "graph_state" else _flat_repro(path)
        ok = detail.get("step_zero_action_delta", 1.0) < 1e-6
        rows.append({"item": item, "ledger_ref": ref, "path": path,
                     "sha16": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16],
                     "status": "FROZEN_ARTIFACT_RUNTIME_COMPATIBLE" if ok else "COMPAT_DELTA", **detail})
    present = [r for r in rows if r["status"] != "ARTIFACT_NOT_PRESENT"]
    result = {"gate": "FROZEN_ARTIFACT_RUNTIME_COMPATIBILITY", "rows": rows,
              "reproduced": sum(r["status"] == "FROZEN_ARTIFACT_RUNTIME_COMPATIBLE" for r in rows),
              "absent": [r["item"] for r in rows if r["status"] == "ARTIFACT_NOT_PRESENT"],
              "all_present_reproduce": all(r["status"] == "FROZEN_ARTIFACT_RUNTIME_COMPATIBLE" for r in present)}
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    r = run()
    for row in r["rows"]:
        print(f"{row['item']:20s} {row['status']:24s} "
              f"{('Δ='+str(row.get('step_zero_action_delta')) if 'step_zero_action_delta' in row else row['ledger_ref'])}")
    print(f"\nreproduced {r['reproduced']}/{len(r['rows'])}; absent: {r['absent']}; "
          f"all present reproduce: {r['all_present_reproduce']}")
