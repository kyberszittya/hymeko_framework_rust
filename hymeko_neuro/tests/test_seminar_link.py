"""Integration test for the seminar ``link`` demo (Demo 3).

Heavy: loads the committed Bitcoin OTC HSiKAN checkpoint and runs a real
forward pass. Skips cleanly when torch or the checkpoint is unavailable (e.g.
the default, non-``ml`` environment). Run with::

    uv run --group ml --group dev pytest -p no:randomly \
        hymeko_neuro/tests/test_seminar_link.py
"""
from __future__ import annotations

import pytest

from hymeko_neuro.demos.seminar.base import REPO_ROOT, DemoRunner
from hymeko_neuro.demos.seminar.cli import build_parser
from hymeko_neuro.demos.seminar.compat import register_legacy_checkpoint_aliases
from hymeko_neuro.demos.seminar.demos import DEMOS

try:  # the demo loads an HSiKAN checkpoint — torch is mandatory to run it.
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

CKPT = REPO_ROOT / "checkpoints" / "hsikan" / "bitcoin_otc_optuna_best.pt"
AUC_TOL = 0.002

pytestmark = pytest.mark.skipif(
    not (_HAS_TORCH and CKPT.is_file()),
    reason=f"needs torch + checkpoint at {CKPT}",
)


def test_legacy_alias_is_idempotent() -> None:
    import sys

    first = register_legacy_checkpoint_aliases()
    second = register_legacy_checkpoint_aliases()
    assert "hymeko_neuro.signedkan" in sys.modules
    assert second == []  # already installed on the second call
    # first may be empty if a prior import already registered it; either way
    # the legacy module must now resolve to the live one.
    assert first == [] or "hymeko_neuro.signedkan" in first


def _run_link(tmp_path, extra: list[str]) -> object:
    args = build_parser().parse_args(
        ["link", "--checkpoint", str(CKPT), "--device", "cpu", *extra]
    )
    return DemoRunner(DEMOS["link"], out_root=tmp_path).run(args)


def test_link_reproduces_checkpoint_auc(tmp_path) -> None:
    result = _run_link(tmp_path, ["--no-figures"])
    m = result.metrics
    # Acceptance (SEMINAR_DEMOS §3): measured AUC reproduces the checkpoint's
    # own recorded AUC within ±0.002 on the fixed split.
    assert m["dataset"] == "bitcoin_otc"
    assert m["n_test"] == 3560
    assert "reference_AUC" in m
    assert m["AUC_delta"] <= AUC_TOL
    assert any("PASS" in n for n in result.notes)
    # 16 GB cap was measurable and respected.
    assert result.peak_rss_bytes is not None
    assert result.peak_rss_bytes < 16 * 1024 ** 3


def test_link_renders_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    result = _run_link(tmp_path, [])
    pngs = list((tmp_path / "link").glob("*.png"))
    assert {p.name for p in pngs} == {
        "ROC_bitcoin_otc.png", "alpha_k_bitcoin_otc.png",
    }
    assert all(p.stat().st_size > 0 for p in pngs)
    assert len(result.artifacts) == 2
