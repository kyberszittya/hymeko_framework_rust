"""Tests for the no-leakage E1 driver ``run_no_leak_benchmark``.

The 2026-06-11 resume step added the narrow-deep depth/width knob: Gömb cells
must run the determinability-per-parameter Pareto point (h=16, L=8;
``reports/2026-05-30-hsikan-depth-narrow-pareto.md``) instead of the shallow
default the smoke first ran. These tests pin:

1. ``Cell.make`` maps each model family to its pinned architecture
   (Gömb -> narrow-deep, SGCN -> phase8 panel width, depth inert).
2. The Gömb subprocess command actually carries ``--d-middle`` /
   ``--middle-n-layers`` with the narrow-deep values — the regression: the
   prior ``_run_subprocess`` emitted neither flag, so Gömb silently ran the
   shallow single-tier middle.
3. The SGCN in-process call receives the panel width via ``hidden=``.
4. ``audit_gate`` boundary behaviour (clean / signal thresholds).
"""
from __future__ import annotations

import pytest

from signedkan_wip.experiments.runs import run_no_leak_benchmark as nlb


def test_cell_make_pins_narrow_deep_for_gomb() -> None:
    c = nlb.Cell.make("gomb", "bitcoin_alpha", n_epochs=60)
    assert (c.width, c.depth) == nlb.GOMB_NARROW_DEEP == (16, 8)


def test_cell_make_pins_panel_width_for_sgcn() -> None:
    c = nlb.Cell.make("SGCN", "bitcoin_alpha", n_epochs=120)
    assert (c.width, c.depth) == nlb.SGCN_PANEL == (32, 1)


def test_subprocess_command_carries_narrow_deep_flags(monkeypatch) -> None:
    """Regression: the Gömb cmd must include the depth/width flags.

    Against the prior implementation (no ``--d-middle`` / ``--middle-n-layers``)
    this assertion fails — that build ran the shallow middle.
    """
    captured: dict[str, list[str]] = {}

    class _Proc:
        stdout = '{"val_auroc": 0.9, "n_params": 12345}'
        stderr = ""
        returncode = 0

    def _fake_run(cmd, capture_output, text):  # noqa: ANN001 - test stub
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(nlb.subprocess, "run", _fake_run)
    cell = nlb.Cell.make("gomb", "bitcoin_alpha", n_epochs=7)
    auc, n_params = nlb._run_subprocess(cell, seed=0, shuffle=False)

    cmd = captured["cmd"]
    assert "--d-middle" in cmd and cmd[cmd.index("--d-middle") + 1] == "16"
    assert "--middle-n-layers" in cmd and cmd[cmd.index("--middle-n-layers") + 1] == "8"
    assert "--shuffle-train-signs" not in cmd
    assert (auc, n_params) == (0.9, 12345)


def test_subprocess_command_adds_shuffle_flag(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    class _Proc:
        stdout = '{"val_auroc": 0.5}'
        stderr = ""
        returncode = 0

    def _fake_run(cmd, capture_output, text):  # noqa: ANN001 - test stub
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(nlb.subprocess, "run", _fake_run)
    cell = nlb.Cell.make("gomb", "bitcoin_alpha", n_epochs=7)
    nlb._run_subprocess(cell, seed=3, shuffle=True)
    assert "--shuffle-train-signs" in captured["cmd"]


def test_inprocess_passes_panel_width_as_hidden(monkeypatch) -> None:
    """SGCN in-process call must receive ``hidden=cell.width`` (the panel 32)."""
    captured: dict[str, object] = {}

    def _fake_cell(dataset, model_name, *, hidden, n_epochs, max_k4,  # noqa: ANN001
                   device, seed, shuffle_train_signs):
        captured.update(hidden=hidden, shuffle=shuffle_train_signs)
        return {"auc": 0.88, "n_params": 999}

    # cell_signed_graph is imported lazily inside _run_inprocess; patch at source.
    import signedkan_wip.experiments.runs.run_final_cell as rfc
    monkeypatch.setattr(rfc, "cell_signed_graph", _fake_cell)

    cell = nlb.Cell.make("SGCN", "bitcoin_alpha", n_epochs=11)
    auc, n_params = nlb._run_inprocess(cell, seed=1, shuffle=True)
    assert captured["hidden"] == 32
    assert captured["shuffle"] is True
    assert (auc, n_params) == (0.88, 999)


def test_result_key_tolerates_legacy_rows_without_seed() -> None:
    legacy = {"dataset": "bitcoin_alpha", "model": "gomb", "shuffle": False}
    assert nlb._result_key(legacy) == ("bitcoin_alpha", "gomb", 0, False)


def test_load_done_roundtrips_appended_rows(tmp_path) -> None:
    out = tmp_path / "r.jsonl"
    row = {"dataset": "bitcoin_otc", "model": "SGCN", "seed": 2,
           "shuffle": True, "auc": 0.5}
    nlb._append_row(out, row)
    rows, done = nlb._load_done(out)
    assert rows == [row]
    assert done == {("bitcoin_otc", "SGCN", 2, True)}


def test_run_cell_seed_skips_completed_arm(tmp_path, monkeypatch) -> None:
    """A (cell, seed, shuffle) arm already in ``done`` is not re-run."""
    calls: list[bool] = []

    def _fake_run_cell(cell, seed, shuffle):  # noqa: ANN001 - test stub
        calls.append(shuffle)
        return {"dataset": cell.dataset, "model": cell.model, "seed": seed,
                "shuffle": shuffle, "auc": 0.9 if not shuffle else 0.5}

    monkeypatch.setattr(nlb, "run_cell", _fake_run_cell)
    out = tmp_path / "r.jsonl"
    cell = nlb.Cell.make("gomb", "bitcoin_alpha", n_epochs=1)
    # Pre-seed the real arm as done; only the shuffle arm should run.
    done_row = {"dataset": "bitcoin_alpha", "model": "gomb", "seed": 0,
                "shuffle": False, "auc": 0.91}
    rows = [done_row]
    done = {nlb._result_key(done_row)}
    real, shuf = nlb._run_cell_seed(cell, 0, out, rows, done)
    assert calls == [True]          # only the shuffle arm ran
    assert real is done_row         # real reused from the checkpoint
    assert shuf["shuffle"] is True
    # The freshly-run shuffle arm was checkpointed (appended) to disk.
    _, done_after = nlb._load_done(out)
    assert ("bitcoin_alpha", "gomb", 0, True) in done_after


@pytest.mark.parametrize(
    "real, shuf, clean, signal",
    [
        (0.89, 0.50, True, True),    # the smoke result: clean + signal
        (0.99, 0.70, False, True),   # leaks: shuffle stays high, still a margin
        (0.55, 0.54, True, False),   # clean but no learned signal
        (0.80, None, False, False),  # runner failure: neither holds
    ],
)
def test_audit_gate_boundaries(real, shuf, clean, signal) -> None:
    assert nlb.audit_gate(real, shuf) == (clean, signal)
