"""Tests for the hero demo. Run: pytest -p no:randomly demos/hero/test_hero.py

Unit tests (no binary) cover the pure gate parser + scenario catalog. The
integration test drives the real ``hymeko`` CLI and is skipped when it isn't
built.
"""
from __future__ import annotations

import pytest

from pathlib import Path

from hero_demo import (
    BROKEN_TWIN,
    GateStatus,
    HeroDemo,
    SCENARIOS,
    Target,
    emit_args,
    find_hymeko,
    parse_validate_output,
    verdict_from_run,
)
from learner_parity import parse_hymeko_layers, parse_torch_attrs, structural_parity


# ── unit: gate parser ────────────────────────────────────────────────
def test_gate_clean() -> None:
    v = parse_validate_output("✅ data/robotics/mini_arm.hymeko is valid")
    assert v.status is GateStatus.CLEAN
    assert v.trusted


def test_gate_warnings() -> None:
    v = parse_validate_output(
        "⚠️  arm.hymeko compiled with 1 warnings:\n  - Joint references unknown parent 'world'"
    )
    assert v.status is GateStatus.WARNINGS
    assert v.trusted  # warnings still emit, but are surfaced


def test_gate_rejected_on_failure() -> None:
    for text in (
        "❌ broken.hymeko failed: ResolveError(Undefined(ghost_link))",
        "Compilation failed: Io(...)",
        "x failed to resolve",
    ):
        v = parse_validate_output(text)
        assert v.status is GateStatus.REJECTED, text
        assert not v.trusted


def test_gate_rejects_unrecognised_output() -> None:
    # Fail closed: never trust output we cannot interpret.
    v = parse_validate_output("???")
    assert v.status is GateStatus.REJECTED
    assert not v.trusted


# ── unit: exit-code-authoritative verdict ────────────────────────────
def test_verdict_from_run_uses_exit_code() -> None:
    # non-zero exit ⇒ rejected, regardless of message
    assert verdict_from_run("✅ looks fine", 1).status is GateStatus.REJECTED
    assert verdict_from_run("", 2).status is GateStatus.REJECTED
    # exit 0 ⇒ message only splits clean vs warnings (never rejected)
    assert verdict_from_run("✅ x is valid", 0).status is GateStatus.CLEAN
    assert verdict_from_run("compiled with 1 warnings", 0).status is GateStatus.WARNINGS
    assert verdict_from_run("anything odd", 0).status is GateStatus.CLEAN
    assert verdict_from_run("✅ x is valid", 0).trusted
    assert not verdict_from_run("boom", 3).trusted


# ── unit: scenario catalog ───────────────────────────────────────────
def test_scenarios_well_formed() -> None:
    ids = [s.scenario_id for s in SCENARIOS]
    assert len(set(ids)) == len(ids), "duplicate scenario id"
    for s in SCENARIOS:
        assert s.label and s.model_name
        assert s.source.exists(), f"{s.scenario_id}: missing source {s.source}"
        assert s.targets, f"{s.scenario_id}: no targets"
        exts = [t.ext for t in s.targets]
        assert len(set(exts)) == len(exts), f"{s.scenario_id}: duplicate target ext"


# ── unit: CLI arg construction (emit vs transform) ───────────────────
def test_emit_args_emit_vs_transform() -> None:
    src = Path("data/nn/simple_net.hymeko")
    assert emit_args(src, "Arm", Target("urdf", "urdf", "<robot")) == [
        "emit", "-f", "urdf", str(src), "-n", "Arm",
    ]
    assert emit_args(src, "Net", Target("torch_dataflow", "py", "import torch", via="transform")) == [
        "transform", "-t", "torch_dataflow", str(src), "--transforms-dir", "transforms",
    ]


def test_target_via_defaults_to_emit() -> None:
    assert Target("dot", "dot", "digraph").via == "emit"


def test_catalog_has_robot_and_learner_scenarios() -> None:
    kinds = {s.kind for s in SCENARIOS}
    assert {"robot", "learner"} <= kinds
    learners = [s for s in SCENARIOS if s.kind == "learner"]
    assert any(
        any(t.fmt == "torch_dataflow" for t in s.targets) for s in learners
    ), "a learner scenario should emit torch_dataflow"


# ── unit: structural parity (learner IR ↔ emitted module) ────────────
_HYMEKO_SAMPLE = """
m {
    x:    ten.t_input { shape [4]; }
    a:    lyr.signedkan_layer { hidden 8; arity 3; spline_kind "x"; grid 5; }
    mix:  lyr.arity_mixer { hidden 8; mix_K 1; }
    head: lyr.signed_classifier { d_in 8; d_out 1; }
    @flow: lyr.dataflow { (+ x, ~ a, - y); }
}
"""
_TORCH_SAMPLE = """
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = SignedKANLayer(hidden=8, arity=3)
        self.mix = ArityMixer(hidden=8, mix_K=1)
        self.head = SignedClassifier(d_in=8, d_out=1)
    def forward(self, x):
        y = self.a(x)
        return y
"""


def test_parse_layers_excludes_tensors_and_edges() -> None:
    layers = parse_hymeko_layers(_HYMEKO_SAMPLE)
    assert layers == {"a": "signedkan_layer", "mix": "arity_mixer", "head": "signed_classifier"}
    assert parse_torch_attrs(_TORCH_SAMPLE) == {"a", "mix", "head"}


def test_structural_parity_faithful_and_missing() -> None:
    ok = structural_parity(_HYMEKO_SAMPLE, _TORCH_SAMPLE)
    assert ok.faithful and not ok.missing and ok.n_layers == 3
    # drop one sub-module from the emit → that layer is reported missing
    broken = _TORCH_SAMPLE.replace("        self.mix = ArityMixer(hidden=8, mix_K=1)\n", "")
    rep = structural_parity(_HYMEKO_SAMPLE, broken)
    assert not rep.faithful
    assert rep.missing == ("mix",)


# ── integration: real CLI ────────────────────────────────────────────
_BIN = find_hymeko()
_needs_bin = pytest.mark.skipif(_BIN is None, reason="hymeko CLI not built (cargo build -p hymeko_cli)")


@_needs_bin
def test_scenario_emits_all_targets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _BIN is not None
    scenario = SCENARIOS[0]
    demo = HeroDemo(_BIN, tmp_path)
    report = demo.run(scenario)
    assert report.gate.trusted, f"gate rejected a shipped model: {report.gate.detail}"
    assert len(report.emits) == len(scenario.targets)
    for e in report.emits:
        assert e.ok, f"{e.target.fmt} failed: {e.detail}"
        assert e.path is not None and e.path.exists()
        assert e.n_bytes > 0
        assert e.target.root_token in e.path.read_text(encoding="utf-8")


@_needs_bin
def test_broken_twin_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _BIN is not None
    demo = HeroDemo(_BIN, tmp_path)
    verdict = demo.validate_text(BROKEN_TWIN, "broken_twin")
    assert verdict.status is GateStatus.REJECTED, verdict.detail
    assert not verdict.trusted


@_needs_bin
def test_learner_emits_torch_module(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert _BIN is not None
    learner = next(s for s in SCENARIOS if s.kind == "learner")
    demo = HeroDemo(_BIN, tmp_path)
    report = demo.run(learner)
    assert report.gate.trusted, report.gate.detail
    torch = next(e for e in report.emits if e.target.fmt == "torch_dataflow")
    assert torch.ok and torch.path is not None
    assert "import torch" in torch.path.read_text(encoding="utf-8")


@_needs_bin
def test_gomb_and_soma_emit_faithful_modules(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The Gömb cascade + the Soma vision net: gated clean, and every layer the
    # .hymeko declares is realised in the emitted torch module (structural parity).
    assert _BIN is not None
    for sid in ("gomb_hsikan", "soma_vision"):
        sc = next(s for s in SCENARIOS if s.scenario_id == sid)
        demo = HeroDemo(_BIN, tmp_path)
        report = demo.run(sc)
        assert report.gate.trusted, f"{sid}: {report.gate.detail}"
        emit = next(e for e in report.emits if e.target.fmt == "torch_dataflow")
        assert emit.ok and emit.path is not None, f"{sid}: torch emit failed"
        par = structural_parity(
            sc.source.read_text(encoding="utf-8"), emit.path.read_text(encoding="utf-8")
        )
        assert par.faithful, f"{sid}: missing {par.missing}"
        assert par.n_layers >= 3
