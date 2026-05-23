"""Phase 11 (2026-05-20) end-to-end: strict-mode by-product filter
on dominated architecture choices drives the HSIKAN sweep driver
to pick a strictly better architecture than scalar-cost ABB picks.

Pins the Python boundary (`hymeko_pgraph_dump` + the
`hsikan_pgraph_mapping` translation) — the Rust-side structural
behaviour is pinned in
`hymeko_pgraph/tests/byproduct_filter_phase11.rs`.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "data" / "hsikan" / "sweep_msg_byproduct_dominated.hymeko"


def _find_dump() -> Path:
    rel = REPO_ROOT / "target" / "release" / "hymeko_pgraph_dump"
    dbg = REPO_ROOT / "target" / "debug" / "hymeko_pgraph_dump"
    if rel.exists():
        return rel
    if dbg.exists():
        return dbg
    pytest.skip("hymeko_pgraph_dump not built")


def _dump(relaxed: bool = False) -> dict:
    cmd = [str(_find_dump()), str(FIXTURE), "--algorithm", "abb"]
    if relaxed:
        cmd.append("--relaxed-msg")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, f"dump rc={proc.returncode}\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_strict_mode_drops_train_short_and_model_h32():
    j = _dump(relaxed=False)
    msg = set(j["msg_units"])
    assert "train_short" not in msg
    assert "model_h32" not in msg
    assert len(msg) == 6  # 3 cycle × 2 model × 1 train_long


def test_relaxed_mode_keeps_dominated_producers():
    j = _dump(relaxed=True)
    msg = set(j["msg_units"])
    assert "train_short" in msg
    assert "model_h32" in msg
    assert len(msg) == 8


def test_strict_abb_pick_matches_phase8_better_architecture():
    """Strict mode ABB picks `m4+h8+long` — the Phase 8 measured
    AUC 0.491 architecture. Relaxed mode picks `m4+h8+short` —
    the Phase 8 measured AUC 0.430 cost-min."""
    strict = _dump(relaxed=False)
    relaxed = _dump(relaxed=True)
    assert set(strict["abb"]["units"]) == {
        "cycle_topk_m4", "model_h8", "train_long",
    }
    assert set(relaxed["abb"]["units"]) == {
        "cycle_topk_m4", "model_h8", "train_short",
    }
    # Cost gap: strict pays 90 more to escape the dominated pick.
    assert strict["abb"]["cost"] == pytest.approx(150.0)
    assert relaxed["abb"]["cost"] == pytest.approx(60.0)


def test_canonical_certificate_is_strict_mode_independent():
    """The canonical Friedler S1..S5 PASS state is independent of
    strict / relaxed engine mode — the schema is the same."""
    strict = _dump(relaxed=False)
    relaxed = _dump(relaxed=True)
    assert strict["canonical_full"]["status"] == "PASS"
    assert relaxed["canonical_full"]["status"] == "PASS"
    # Extension catches both by-products in both modes.
    assert strict["extension_full"]["status"] == "FAIL"
    assert relaxed["extension_full"]["status"] == "FAIL"


def test_hsikan_mapping_resolves_strict_pick_to_correct_kwargs():
    """The HSIKAN unit→config mapping translates the strict-mode
    ABB pick into the Phase 8 known-better training config
    (h=8, n_epochs=60, m_cycles=4)."""
    from signedkan_wip.src.hsikan_pgraph_mapping import (
        merge_structure_knobs, run_one_kwargs,
    )
    merged = merge_structure_knobs(
        ["cycle_topk_m4", "model_h8", "train_long"]
    )
    kw = run_one_kwargs(
        dataset="bitcoin_alpha", seed=0, structure=merged, base={},
    )
    assert kw["hidden"] == 8
    assert kw["n_epochs"] == 60
    assert kw["m_cycles"] == 4
