"""Frozen disjoint seed banks for the v3 full-action BC/DAgger competence protocol (§1).

Deterministic seed lists, committed with SHA-256 BEFORE any training. The FINAL_TEST bank is untouched until a
candidate policy is frozen — never searched, queried, trained on, or used for selection. The historical HEADLINE panel
is a regression panel; the searched 4000-4029 panel is CANONICAL_DYNAMIC_EXPERT_COVERAGE_PANEL_V1 (a demonstration
source / recoverability panel, NOT held-out generalization — CEM was optimised on those states).
"""
from __future__ import annotations

import hashlib
import json

# canonical contracts (recorded in every bank/dataset manifest)
BUNDLE_HASH = "6664ac459cca8f62"
SEMANTIC_GRAPH_FP = "sem:469094de1fba54b2ff481706ca2e09ce"
OBS_CONTRACT = "node_features flat 48 (6 vertices x 8)"
ACTION_CONTRACT = "u_expert_executed = inner.data.ctrl[:4] (4 arm actuators, ctrlrange +-4)"

# disjoint banks (by complete scene/seed)
TRAIN_QUERY = tuple(range(6000, 6120))     # 120 — CEM search / demo generation / DAgger rollouts+queries
VALIDATION = tuple(range(7000, 7030))      # 30  — checkpoint selection / early stop / arch compare (NO expert labels)
FINAL_TEST = tuple(range(8000, 8050))      # 50  — UNTOUCHED until the candidate policy is frozen
HEADLINE = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)   # historical regression panel
COVERAGE_PANEL_V1 = tuple(range(4000, 4030))                        # demonstration source / recoverability panel


def _sha(seeds) -> str:
    return hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()[:16]


def manifest() -> dict:
    banks = {"train_query": TRAIN_QUERY, "validation": VALIDATION, "final_test": FINAL_TEST,
             "headline_regression": HEADLINE, "coverage_panel_v1": COVERAGE_PANEL_V1}
    # assert pairwise disjoint (final_test must not overlap anything used upstream)
    used = set(TRAIN_QUERY) | set(VALIDATION) | set(HEADLINE) | set(COVERAGE_PANEL_V1)
    assert not (set(FINAL_TEST) & used), "FINAL_TEST leaks into an upstream bank"
    assert not (set(TRAIN_QUERY) & set(VALIDATION)), "train/validation overlap"
    return {"bundle_hash": BUNDLE_HASH, "semantic_graph_fp": SEMANTIC_GRAPH_FP,
            "obs_contract": OBS_CONTRACT, "action_contract": ACTION_CONTRACT,
            "banks": {k: {"n": len(v), "sha16": _sha(v), "seeds": list(v)} for k, v in banks.items()}}


if __name__ == "__main__":
    print(json.dumps(manifest(), indent=1))
