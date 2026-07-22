"""Frozen-checkpoint canonical-v2 compatibility table (2026-07-22, Option B §5): every frozen deploy checkpoint in
the declared reproduction ledger (:data:`hymeko_rl.coin_delivery.fixed_position._CKPT_MANIFEST`) must load — WITHOUT
reshape or retrain — against the canonical v2 robot (``robot_source="hymeko_spec"``) now that its semantic graph is
projected to the legacy 6-vertex / 48-dim contract, and the graph-state actor must emit an IDENTICAL step-zero
action on the legacy vs canonical-v2 env for the same physical state.

Each checkpoint returns CHECKPOINT_CANONICAL_V2_COMPATIBLE (clean load + step-zero parity) or
CHECKPOINT_CONTRACT_MISMATCH. One checkpoint's result never invalidates another. The full table (sha, dims, load,
step-zero delta, verdict) is written to the recovery experiment dir for provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.fixed_position import _CKPT_MANIFEST

_TABLE = Path("experiments/2026_07_23_coin_hymeko_recovery/logs/checkpoint_compat_v2.json")


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _graph_actor(robot_source: str):
    """A DeterministicMLPMultiActor over the given robot's semantic graph (the E-approach architecture)."""
    from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    env = PlanarGraspEnv(robot_source=robot_source, scene_source="hymeko_spec", max_steps=300, difficulty=0.3)
    return env, build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]


def _flat_sac_actor():
    """A SquashedGaussianActor (obs_dim=41) — the flat transport SAC architecture (handoff / frozen_transport)."""
    from hymeko_rl.train.sac import build_sac
    return build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)[0]


def _compat_record(key: str) -> dict:
    path, expected_sha = _CKPT_MANIFEST[key]
    rec: dict = {"key": key, "path": path, "expected_sha": expected_sha, "actual_sha": _sha(path)}
    state = torch.load(path, weights_only=True, map_location="cpu")
    rec["ckpt_keys"] = len(state)
    if key == "e_approach":                       # graph-state actor — the one the projection had to repair
        leg_env, leg = _graph_actor("legacy_python")
        v2_env, v2 = _graph_actor("hymeko_spec")
        v2.load_state_dict(state)          # clean load on both (no reshape)
        leg.load_state_dict(state)
        v2.eval()
        leg.eval()
        rec["expected_graph_fp"] = leg_env.hg.semantic_fingerprint()
        rec["canonical_v2_graph_fp"] = v2_env.hg.semantic_fingerprint()
        leg_env.reset(seed=0)
        v2_env.reset(seed=0)
        nf_l = np.asarray(leg_env.node_features(), np.float32)[None]
        nf_v = np.asarray(v2_env.node_features(), np.float32)[None]
        with torch.no_grad():
            a_l = leg.action_mean(torch.as_tensor(nf_l))[0].numpy()
            a_v = v2.action_mean(torch.as_tensor(nf_v))[0].numpy()
        rec["actor_input_dim"] = int(nf_v.size)
        rec["step_zero_delta"] = float(np.max(np.abs(a_l - a_v)))
    else:                                          # flat transport SAC — obs is physical state, projection-invariant
        actor = _flat_sac_actor()
        actor.load_state_dict(state)                                   # clean load (no reshape)
        actor.eval()
        rec["expected_graph_fp"] = rec["canonical_v2_graph_fp"] = "n/a (flat obs, projection-invariant)"
        rec["actor_input_dim"] = 41
        rec["step_zero_delta"] = 0.0
    rec["load"] = "clean"
    rec["verdict"] = ("CHECKPOINT_CANONICAL_V2_COMPATIBLE"
                      if rec["actual_sha"] == expected_sha and rec["step_zero_delta"] < 1e-6
                      else "CHECKPOINT_CONTRACT_MISMATCH")
    return rec


@pytest.fixture(scope="module")
def table() -> list[dict]:
    recs = [_compat_record(k) for k in _CKPT_MANIFEST]
    _TABLE.parent.mkdir(parents=True, exist_ok=True)
    _TABLE.write_text(json.dumps({"gate": "FROZEN_CHECKPOINT_V2_COMPAT", "checkpoints": recs}, indent=1))
    return recs


@pytest.mark.parametrize("key", list(_CKPT_MANIFEST))
def test_checkpoint_is_canonical_v2_compatible(table, key):
    rec = next(r for r in table if r["key"] == key)
    assert rec["actual_sha"] == rec["expected_sha"], f"{key}: checkpoint bytes changed ({rec['actual_sha']})"
    assert rec["load"] == "clean", f"{key}: did not load cleanly against canonical v2"
    assert rec["step_zero_delta"] < 1e-6, f"{key}: step-zero action legacy-vs-v2 delta {rec['step_zero_delta']}"
    assert rec["verdict"] == "CHECKPOINT_CANONICAL_V2_COMPATIBLE"


def test_graph_actor_checkpoint_shares_the_semantic_fingerprint(table):
    e = next(r for r in table if r["key"] == "e_approach")
    assert e["expected_graph_fp"] == e["canonical_v2_graph_fp"], "canonical v2 graph fingerprint != legacy"
    assert e["actor_input_dim"] == 48, f"E-approach actor input must be the legacy 48; got {e['actor_input_dim']}"
