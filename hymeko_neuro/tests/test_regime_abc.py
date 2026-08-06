"""Tests for the regime A/B/C 5-seed comparison harness.

Unit  — clean-signature parsing, regime-admissible derivation (against
        the authoritative Rust SSG), arch→env cap, aggregation maths.
Integ — dry-run job enumeration (no torch).

The GPU training loop (`run_cell`) is exercised by the standalone
production-scale smoke, not pytest.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hymeko_neuro.experiments.runs.run_hsikan_mixed_composite_smoke import (
    _repo_root,
)
from hymeko_neuro.experiments.runs.run_regime_abc_5seed import (
    REGIMES,
    _clean_signature,
    _mean_sd,
    _paired_delta,
    aggregate,
    arch_env,
    main,
    regime_admissible_archs,
)

SWEEP = Path("data/hsikan/sweep_msg_mixed_protocols.hymeko")


@pytest.fixture(scope="module")
def repo() -> Path:
    return _repo_root()


@pytest.fixture(scope="module")
def dump_built(repo: Path) -> None:
    rel = repo / "target" / "release" / "hymeko_pgraph_dump"
    dbg = repo / "target" / "debug" / "hymeko_pgraph_dump"
    if rel.exists() or dbg.exists():
        return
    try:
        subprocess.run(
            ["cargo", "build", "-p", "hymeko_pgraph", "--bin",
             "hymeko_pgraph_dump", "--release"],
            cwd=str(repo), check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:  # pragma: no cover
        pytest.skip(f"cannot build hymeko_pgraph_dump: {err}")


# ---------------------------------------------------------------- clean sig
def test_clean_signature_complete():
    sig = _clean_signature(
        ["struct_mixed", "attn_dot", "gate_edge_cr", "dm_on", "model_h8",
         "train_short"]
    )
    assert sig == ("dot", True, True)


def test_clean_signature_redundant_axis_is_none():
    assert _clean_signature(
        ["struct_mixed", "attn_none", "attn_dot", "gate_scalar", "dm_off",
         "model_h8", "train_short"]
    ) is None


def test_clean_signature_missing_axis_is_none():
    assert _clean_signature(["struct_mixed", "attn_none", "gate_scalar"]) is None


# ----------------------------------------------------- regime admissibility
def test_regime_admissible_counts(repo, dump_built):
    sets = {r: regime_admissible_archs(repo, repo / SWEEP, r) for r in REGIMES}
    assert len(sets["canonical"]) == 12
    assert len(sets["no-excess"]) == 8
    assert len(sets["cost-dominance"]) == 2
    # No-Excess ⊂ Canonical; quaternion only under Canonical.
    assert sets["no-excess"] < sets["canonical"]
    canon_attn = {a[0] for a in sets["canonical"]}
    ne_attn = {a[0] for a in sets["no-excess"]}
    assert "quaternion" in canon_attn
    assert "quaternion" not in ne_attn


def test_arch_env_attention_caps():
    env = arch_env(("quaternion", False, False))
    assert env["HSIKAN_ATTENTION_M_E"] == "quaternion"
    assert env["HSIKAN_TOPK_K"] == "8"
    assert arch_env(("none", True, True)).get("HSIKAN_TOPK_K") is None


def test_arch_env_topk_override():
    """A custom attention_topk_k drives the attention cap (richer pool)."""
    env = arch_env(("quaternion", False, False), attention_topk_k=32)
    assert env["HSIKAN_TOPK_K"] == "32"
    # non-attention archs ignore the attention cap entirely.
    assert arch_env(("none", False, False), attention_topk_k=32).get("HSIKAN_TOPK_K") is None


# ------------------------------------------------------------- aggregation
def test_mean_sd():
    m, sd = _mean_sd([0.9, 0.8, 0.85])
    assert abs(m - 0.85) < 1e-9
    assert sd > 0


def test_paired_delta_sign_and_wins():
    a = {0: 0.92, 1: 0.94, 2: 0.90}
    b = {0: 0.90, 1: 0.91, 2: 0.89}
    res = _paired_delta(a, b)
    assert res["delta_mean"] > 0
    assert res["wins_a"] == 3
    assert res["n_paired"] == 3


def test_aggregate_regime_best_and_pairing():
    # quaternion arch (canonical-only) is the strongest here.
    q = ["quaternion", False, False]
    d = ["dot", False, False]
    n = ["none", False, False]
    rows = []
    for seed in range(3):
        rows.append({"arch": q, "arch_key": "attn=quaternion|gate=scalar|dm=off",
                     "seed": seed, "auc": 0.95})
        rows.append({"arch": d, "arch_key": "attn=dot|gate=scalar|dm=off",
                     "seed": seed, "auc": 0.90})
        rows.append({"arch": n, "arch_key": "attn=none|gate=scalar|dm=off",
                     "seed": seed, "auc": 0.88})
    regime_sets = {
        "canonical": {("quaternion", False, False), ("dot", False, False),
                      ("none", False, False)},
        "no-excess": {("dot", False, False), ("none", False, False)},
        "cost-dominance": {("none", False, False), ("quaternion", False, False)},
    }
    agg = aggregate(rows, regime_sets)
    assert agg["regime_best"]["canonical"] == "attn=quaternion|gate=scalar|dm=off"
    assert agg["regime_best"]["no-excess"] == "attn=dot|gate=scalar|dm=off"
    cvn = agg["canonical_vs_noexcess"]
    assert cvn["same_architecture"] is False
    assert abs(cvn["paired"]["delta_mean"] - 0.05) < 1e-9
    assert cvn["paired"]["wins_a"] == 3


# --------------------------------------------------------------- dry-run
def test_main_dry_run_job_count(repo, dump_built, capsys):
    rc = main(["--dry-run", "--seeds", "0,1,2,3,4"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_jobs"] == 12 * 5
    assert len(out["union_archs"]) == 12
