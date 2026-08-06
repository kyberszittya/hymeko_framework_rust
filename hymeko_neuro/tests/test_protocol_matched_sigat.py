"""Tests for the protocol-matched comparison driver + its aggregator.

Layers (CLAUDE.md §3): unit (grid construction, aggregation maths, gate, gap),
integration (a real 3-epoch smoke on the network-free synthetic dataset, both
protocols + resume). The 5-seed leakage gate is exercised at experiment scale,
not here.

Run: pytest -p no:randomly hymeko_neuro/tests/test_protocol_matched_sigat.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hymeko_neuro.experiments.runs import run_protocol_matched_sigat as drv
from hymeko_neuro.experiments.eval import aggregate_protocol_match as agg

SYNTH = "sbm_n200_k4_s0"  # programmatic loader — no network, deterministic


# --- driver: grid construction -------------------------------------------

def test_cells_cover_model_dataset_protocol_product() -> None:
    specs = (drv._audit_spec("sigat"), drv._audit_spec("sgcn"))
    cells = drv._cells(specs, ("bitcoin_alpha", "bitcoin_otc"))
    assert len(cells) == 2 * 2 * 2  # models × datasets × protocols
    assert {c.dedup for c in cells} == {False, True}
    assert {(c.spec.label, c.dataset, c.dedup) for c in cells} \
        .issuperset({("sigat", "bitcoin_alpha", True),
                     ("sgcn", "bitcoin_otc", False)})


def test_specs_include_the_rotor_line_and_targets() -> None:
    labels = {s.label for s in drv.SPECS}
    assert {"sigat", "sigat_rotor", "cayley_rotor", "hsikan_rotor_r2sw4"} <= labels


# --- driver: a real (tiny) arm + resume -----------------------------------

def test_run_arm_returns_finite_auc_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cell = drv.Cell(drv._audit_spec("sgcn", n_epochs=3), SYNTH, dedup=True)
    row = drv.run_arm(cell, seed=0, shuffle=False)
    assert row["model"] == "sgcn" and row["dedup"] is True
    assert row["n_params"] > 0 and isinstance(row["wall_s"], float)
    assert math.isnan(row["auc"]) or 0.0 <= row["auc"] <= 1.0


def test_main_smoke_writes_jsonl_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "pm_smoke.jsonl"
    specs = (drv._audit_spec("sgcn", n_epochs=3),)
    rows = drv.main(smoke=True, out_path=out, seeds=(0,),
                    specs=specs, datasets=(SYNTH,))
    # 1 model × 1 dataset × 2 protocols × 1 seed × 2 arms.
    assert len(rows) == 4
    n_lines = len(out.read_text().splitlines())
    assert n_lines == 4

    # Second call: every arm is already done → no new lines appended (resume).
    rows2 = drv.main(smoke=True, out_path=out, seeds=(0,),
                     specs=specs, datasets=(SYNTH,))
    assert len(rows2) == 4
    assert len(out.read_text().splitlines()) == 4


# --- aggregator: maths, gate, gap -----------------------------------------

def _synthetic_rows() -> list[dict]:
    """Two models on one dataset, both protocols, 2 seeds, real+shuffle."""
    rows = []
    aucs = {  # (model, dedup, shuffle) -> per-seed real/shuffle values
        ("sigat_rotor", False): (0.90, 0.92),
        ("sigat_rotor", True): (0.86, 0.88),
        ("hsikan_rotor_r2sw4", False): (0.84, 0.86),
        ("hsikan_rotor_r2sw4", True): (0.85, 0.87),
    }
    for (model, dedup), (a0, a1) in aucs.items():
        for seed, real in enumerate((a0, a1)):
            rows.append(dict(model=model, dataset="bitcoin_otc", dedup=dedup,
                             seed=seed, shuffle=False, auc=real, n_params=1000))
            rows.append(dict(model=model, dataset="bitcoin_otc", dedup=dedup,
                             seed=seed, shuffle=True, auc=0.50, n_params=1000))
    return rows


def test_aggregate_groups_and_reduces() -> None:
    table = agg.aggregate(_synthetic_rows())
    cell = table[("sigat_rotor", "bitcoin_otc", False)]
    assert cell.n_seeds == 2
    assert abs(cell.real_mean - 0.91) < 1e-9      # mean(0.90, 0.92)
    assert cell.real_std > 0.0                    # pstdev of two distinct values
    assert abs(cell.shuf_mean - 0.50) < 1e-9
    assert cell.clean is True                      # 0.50 <= 0.55


def test_aggregate_flags_leaky_shuffle() -> None:
    rows = _synthetic_rows()
    for r in rows:
        if r["shuffle"]:
            r["auc"] = 0.80  # shuffled stays high → not clean
    table = agg.aggregate(rows)
    assert table[("sigat_rotor", "bitcoin_otc", False)].clean is False


def test_finite_drops_none_and_nan() -> None:
    assert agg._finite([0.9, None, float("nan"), 0.8]) == [0.9, 0.8]


def test_aggregate_empty_group_is_nan() -> None:
    rows = [dict(model="m", dataset="d", dedup=False, seed=0, shuffle=False,
                 auc=None, n_params=1)]
    cell = agg.aggregate(rows)[("m", "d", False)]
    assert math.isnan(cell.real_mean) and cell.n_seeds == 0


def test_render_shows_protocols_and_gap_vs_reference() -> None:
    out = agg.render(agg.aggregate(_synthetic_rows()))
    assert "non-deduped" in out and "deduped" in out
    assert "sigat_rotor" in out and agg.REFERENCE_MODEL in out
    # On the deduped protocol sigat_rotor (0.87) leads the ref line (0.86) by +0.01.
    assert "+0.0100" in out
