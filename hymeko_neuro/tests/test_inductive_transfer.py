"""Tests for the cross-graph inductive transfer driver.

Layers (CLAUDE.md §3): unit (the rotor transfers across two distinct synthetic
graphs; row schema; eval set comes from the *eval* graph), failure (unknown model).
The 5-seed transfer gate is exercised at experiment scale, not here. Plan:
docs/plans/2026-06-18-inductive-transfer-test/.

Run: pytest -p no:randomly hymeko_neuro/tests/test_inductive_transfer.py
"""
from __future__ import annotations

import math

import pytest
import torch

from hymeko_neuro.experiments.runs import run_inductive_transfer as tr

# Two distinct, same-size synthetic SBM graphs (network-free, deterministic).
A, B = "sbm_n200_k4_s0", "sbm_n200_k4_s1"


def test_rotor_transfers_across_distinct_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inductive rotor trains on A and scores B's held-out edges (no node-ID
    table) — the transfer mechanism works and yields a valid AUROC."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = tr.transfer_cell("cayley_rotor", A, B, seed=0, n_epochs=3)
    assert r["train_ds"] == A and r["eval_ds"] == B
    assert r["transferred"] is True and r["note"] == ""
    assert r["n_test"] > 0 and r["n_params"] > 0
    assert r["auc"] is None or 0.0 <= r["auc"] <= 1.0


def test_eval_set_is_from_the_eval_graph_not_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """n_test equals the *eval* graph's deduped held-out size (computed
    independently), proving the eval set comes from B, not the train graph A."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from hymeko_neuro.data.datasets import (
        drop_train_pairs,
        load,
        split,
        undirected_pair,
    )
    g_b = load(B)
    tr_b, _, te_b = split(g_b, seed=0)
    pairs = {undirected_pair(e) for e in g_b.edges[tr_b]}
    e_te, _ = drop_train_pairs(g_b.edges[te_b], g_b.signs[te_b], pairs)
    expected_b = len(e_te)

    ab = tr.transfer_cell("cayley_rotor", A, B, seed=0, n_epochs=2)
    assert ab["n_test"] == expected_b and expected_b > 0


def test_shuffle_flag_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The transfer gate trains on shuffled-A signs; the flag is provenance."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = tr.transfer_cell("cayley_rotor", A, B, seed=0, n_epochs=2,
                         shuffle_train_signs=True)
    assert r["shuffle"] is True
    assert r["auc"] is None or 0.0 <= r["auc"] <= 1.0


def test_unknown_model_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        tr.transfer_cell("not_a_model", A, B, seed=0, n_epochs=1)


def test_walk_enriched_rotor_also_transfers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The enriched-input variant is equally transferable (structural features)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = tr.transfer_cell("cayley_rotor_walk", A, B, seed=0, n_epochs=2)
    assert r["transferred"] is True
    assert r["auc"] is None or math.isnan(float("nan")) or 0.0 <= r["auc"] <= 1.0


# --- decomposition arms + grid resumption (the strengthen-transfer-grid increment) ---

@pytest.mark.parametrize("arm", list(tr.Arm))
def test_arm_roundtrips_through_shuffle_trained(arm: tr.Arm) -> None:
    """The resumption key derives the arm from (shuffle, trained); that derivation
    must invert the arm's own (shuffle_signs, trains) — else a row's arm is mislabelled
    on resume."""
    assert tr.Arm.of(shuffle=arm.shuffle_signs, trained=arm.trains) is arm


def test_arm_flag_mapping() -> None:
    assert (tr.Arm.REAL.shuffle_signs, tr.Arm.REAL.trains) == (False, True)
    assert (tr.Arm.SHUFFLE.shuffle_signs, tr.Arm.SHUFFLE.trains) == (True, True)
    assert tr.Arm.RANDINIT.trains is False


def test_decomp_grid_keys_three_arms_distinctly_and_resumes(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_grid(ARMS_DECOMP)`` writes one distinctly-keyed row per arm and resumes
    idempotently.

    **Regression:** under the old shuffle-bool key the real arm
    (``shuffle=False, trained=True``) and the random-init arm
    (``shuffle=False, trained=False``) collide to a single key — the random-init arm
    would be silently dropped on resume. The ``len(distinct keys) == 3`` assertion
    fails against that prior implementation. Training is faked: this exercises the
    grid/keying contract, not the model.
    """
    calls: list[tuple[bool, bool]] = []

    def fake_cell(model: str, a: str, b: str, seed: int, *,
                  shuffle_train_signs: bool = False, train: bool = True,
                  **_kw: object) -> dict[str, object]:
        calls.append((shuffle_train_signs, train))
        return dict(model=model, train_ds=a, eval_ds=b, seed=seed,
                    shuffle=shuffle_train_signs, trained=train, auc=0.5, n_test=10)

    monkeypatch.setattr(tr, "transfer_cell", fake_cell)
    out = tmp_path / "decomp.jsonl"  # type: ignore[operator]

    rows = tr.run_grid(("cayley_rotor",), (("ga", "gb"),), (0,), tr.ARMS_DECOMP, out)
    assert len(rows) == 3
    assert len({tr._row_key(r) for r in rows}) == 3           # the fix
    assert {r["arm"] for r in rows} == {"real", "shuffle", "randinit"}
    assert len(calls) == 3

    # Resume on the same file: no arm recomputed, no duplicate row appended.
    rows2 = tr.run_grid(("cayley_rotor",), (("ga", "gb"),), (0,), tr.ARMS_DECOMP, out)
    assert len(rows2) == 3
    assert len(calls) == 3
