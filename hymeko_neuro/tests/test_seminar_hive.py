"""Integration test for the seminar ``hive`` demo (Demo 2 — HIVE compilation).

Needs the built ``hymeko`` module; skips otherwise. Run via the venv interpreter
(``uv run --group`` re-syncs and drops the editable install)::

    PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly \
        hymeko_neuro/tests/test_seminar_hive.py
"""
from __future__ import annotations

import json

import pytest

from hymeko_neuro.demos.seminar.base import REPO_ROOT, DemoRunner
from hymeko_neuro.demos.seminar.cli import build_parser
from hymeko_neuro.demos.seminar.demos import DEMOS

try:
    import hymeko  # noqa: F401

    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False

FANO = REPO_ROOT / "data" / "typical_graphs" / "fano_graph.hymeko"
pytestmark = pytest.mark.skipif(
    not (_HAS_ENGINE and FANO.is_file()),
    reason="needs built hymeko + fano fixture",
)


def _run(tmp_path, extra: list[str]):
    args = build_parser().parse_args(
        ["hive", "--src", str(FANO), "--device", "cpu", *extra]
    )
    return DemoRunner(DEMOS["hive"], out_root=tmp_path).run(args)


def test_hive_is_registered() -> None:
    assert "hive" in DEMOS and DEMOS["hive"].name == "hive"


def test_hive_compiles_fano_counts(tmp_path) -> None:
    m = _run(tmp_path, ["--no-write"]).metrics
    # Fano: 7 vertices + 1 container = 8 nodes; 7 lines. Star/clique COO nnz 42
    # each (sign-0 symmetric push; clique stores directed entries).
    assert m["n_nodes"] == 8
    assert m["n_edges"] == 7
    assert m["star_coo_nnz"] == 42
    assert m["clique_nnz"] == 42
    assert m["canonical_hash"].startswith("blake3:")
    assert len(m["canonical_hash"]) == len("blake3:") + 64


def test_hive_canonicalisation_passes(tmp_path) -> None:
    result = _run(tmp_path, ["--no-write"])
    assert result.metrics["canonicalisation"] == "PASS"
    assert any("PASS" in n for n in result.notes)


def test_hive_writes_report(tmp_path) -> None:
    _run(tmp_path, [])
    report = tmp_path / "hive" / "hive_report.json"
    assert report.is_file()
    doc = json.loads(report.read_text(encoding="utf-8"))
    # Surface text, IR snapshot, both encodings, and the proof are all present.
    assert "surface_text" in doc and "ir_snapshot" in doc
    assert doc["star_coo"]["nnz"] == 42 and doc["clique"]["nnz"] == 42
    proof = doc["canonicalisation_proof"]
    assert proof["order_invariant"] is True
    assert proof["structural_sensitive"] is True
    assert proof["hash_base"] != proof["hash_structural_change"]
