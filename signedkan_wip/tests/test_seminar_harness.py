"""Unit tests for the seminar demo harness (no checkpoint / torch-free path).

Covers ``resources``, ``base`` (device/seed/runner/cap), the demo registry,
and the CLI parser. The heavy link-inference integration lives in
``test_seminar_link.py``.
"""
from __future__ import annotations

import argparse

import pytest

from signedkan_wip.demos.seminar import base
from signedkan_wip.demos.seminar.base import (
    DemoContext, DemoResult, DemoRunner, resolve_device, set_global_seed,
)
from signedkan_wip.demos.seminar.cli import build_parser
from signedkan_wip.demos.seminar.demos import DEMOS
from signedkan_wip.demos.seminar.resources import fmt_bytes, peak_rss_bytes


# ── resources ────────────────────────────────────────────────────────────
def test_peak_rss_is_positive_on_this_platform() -> None:
    peak = peak_rss_bytes()
    # Windows/Linux/macOS all expose a peak counter; None would mean the cap
    # cannot be enforced, which the runner must warn about (tested below).
    assert peak is not None
    assert peak > 0


@pytest.mark.parametrize(
    "n,expected_unit",
    [(None, "unknown"), (512 * 1024 ** 2, "MB"), (3 * 1024 ** 3, "GB")],
)
def test_fmt_bytes_units(n, expected_unit) -> None:
    assert expected_unit in fmt_bytes(n)


# ── device / seed ────────────────────────────────────────────────────────
def test_resolve_device_cpu_is_identity() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_returns_concrete() -> None:
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_resolve_device_rejects_garbage() -> None:
    with pytest.raises(AssertionError):
        resolve_device("tpu")


def test_set_global_seed_is_deterministic() -> None:
    import random

    set_global_seed(123)
    a = [random.random() for _ in range(5)]
    set_global_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


# ── DemoRunner contract ──────────────────────────────────────────────────
class _DummyDemo:
    """Minimal SeminarDemo: records the ctx it was handed."""

    name = "dummy"
    help = "test-only demo"

    def __init__(self) -> None:
        self.seen_ctx: DemoContext | None = None

    def add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--foo", default="bar")

    def run(self, args: argparse.Namespace, ctx: DemoContext) -> DemoResult:
        self.seen_ctx = ctx
        return DemoResult(name=self.name, metrics={"foo": args.foo})


def _args(**kw) -> argparse.Namespace:
    base_kw = dict(device="cpu", seed=0, quick=False, foo="bar")
    base_kw.update(kw)
    return argparse.Namespace(**base_kw)


def test_runner_populates_resources_and_outdir(tmp_path) -> None:
    demo = _DummyDemo()
    runner = DemoRunner(demo, out_root=tmp_path)
    result = runner.run(_args())
    assert result.wall_s is not None and result.wall_s >= 0
    assert result.peak_rss_bytes is not None and result.peak_rss_bytes > 0
    assert (tmp_path / "dummy").is_dir()
    assert demo.seen_ctx is not None and demo.seen_ctx.device == "cpu"


def test_runner_enforces_rss_cap(tmp_path) -> None:
    # An absurdly low cap must trip on any real process footprint.
    runner = DemoRunner(_DummyDemo(), out_root=tmp_path, rss_cap_gb=1e-9)
    with pytest.raises(MemoryError, match="exceeded"):
        runner.run(_args())


def test_runner_warns_when_peak_unavailable(tmp_path, monkeypatch, capsys) -> None:
    # Simulate a platform with no peak counter: cap is not enforced, but the
    # runner must say so out loud rather than silently passing.
    monkeypatch.setattr(base, "peak_rss_bytes", lambda: None)
    result = DemoRunner(_DummyDemo(), out_root=tmp_path, rss_cap_gb=1e-9).run(_args())
    assert result.peak_rss_bytes is None
    assert "NOT enforced" in capsys.readouterr().out


# ── registry + CLI ───────────────────────────────────────────────────────
def test_link_demo_is_registered() -> None:
    assert "link" in DEMOS
    assert DEMOS["link"].name == "link"


def test_build_parser_exposes_every_demo() -> None:
    parser = build_parser()
    for name in DEMOS:
        ns = parser.parse_args([name])
        assert ns.demo == name
        assert ns.device == "auto" and ns.seed == 0
