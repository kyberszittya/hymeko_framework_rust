"""Tests for the canonical domain-generic command layer + an architecture regression guard."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hymeko_rl.campaign.runner import resolve_adapter, run_one
from hymeko_rl.campaign.spec import CampaignSpec, ExperimentSpec

_ROOT = Path(__file__).resolve().parents[2]


def test_spec_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        ExperimentSpec.from_dict({"domain": "cip", "typo_field": 1})


def test_unknown_domain_fails_loud() -> None:
    with pytest.raises(KeyError, match="no domain adapter"):
        resolve_adapter("does_not_exist")


def test_adapters_reject_bad_options() -> None:
    for dom, opts, msg in [("coin_delivery", {"policy": "BOGUS", "strategy": "S1_FAST_HANDOFF"}, "policy"),
                           ("cip", {"prioritization_model": "xgboost"}, "unsupported"),
                           ("hypersignedlingam", {"model_variant": "plain_lingam"}, "unsupported")]:
        with pytest.raises(ValueError, match=msg):
            resolve_adapter(dom).validate(ExperimentSpec(domain=dom, experiment_name="x", domain_options=opts))


def test_core_has_no_domain_imports() -> None:
    """The orchestration core (spec, runner) must not import any domain algorithm at module top (§2.1)."""
    for f in ("spec.py", "runner.py"):
        tree = ast.parse((_ROOT / "hymeko_rl" / "campaign" / f).read_text())
        for n in ast.walk(tree):
            mod = (n.module if isinstance(n, ast.ImportFrom) else "") or ""
            assert not any(d in mod for d in ("coin_delivery", "eval.cip", "eval.causal", "coin_neutral",
                                              "signed_hyper")), f"{f} imports domain code: {mod}"


def test_adapters_domain_imports_are_lazy() -> None:
    """Adapters must import their domain lazily (inside methods) so no adapter's domain leaks at module top."""
    tree = ast.parse((_ROOT / "hymeko_rl" / "campaign" / "adapters.py").read_text())
    for n in tree.body:                                            # module-top only
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = (n.module if isinstance(n, ast.ImportFrom) else "") or ""
            assert not any(d in mod for d in ("coin", "cip", "causal", "signed")), f"non-lazy domain import: {mod}"


def test_cip_and_hsl_smokes_produce_contract(tmp_path: Path) -> None:
    """Two different domains run through the SAME runner and both emit the full artifact contract (integration)."""
    contract = {"manifest.json", "resolved_config.json", "provenance.json", "metrics.jsonl", "result.json",
                "stdout.log", "artifact_index.json"}
    for dom, opts in [("cip", {"prioritization_model": "directlingam", "n_samples": 120, "d_state": 4, "d_action": 1}),
                      ("hypersignedlingam", {"n_vars": 4, "n_samples": 200, "bootstrap": 3})]:
        r = run_one(ExperimentSpec(domain=dom, experiment_name="smoke", seed=0,
                                   artifact_root=str(tmp_path), domain_options=opts))
        assert r["status"] == "ok"
        assert contract <= {f.name for f in Path(r["out"]).iterdir()}


def test_campaign_manifests_load() -> None:
    for m in ("cip_smoke", "hypersignedlingam_smoke", "coin_delivery_final_video"):
        c = CampaignSpec.from_manifest(_ROOT / "configs" / "campaigns" / f"{m}.json")
        assert c.runs and all(isinstance(s, ExperimentSpec) for s in c.runs)


def test_architecture_guard_production_import_debt_does_not_grow() -> None:
    """Regression guard (§1.7 #7): production (non-experiment, non-test) modules importing hymeko_rl.experiments.*
    must not exceed the audited baseline. Lowering this number as debt is paid is fine; raising it fails."""
    import os
    baseline = 62                                                 # audited 2026-07-21 raw ImportFrom nodes (56 deduped)
    count = 0
    for dp, _, fs in os.walk(_ROOT / "hymeko_rl"):
        if "__pycache__" in dp or "/experiments" in dp or "/tests" in dp:
            continue
        for f in fs:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse((Path(dp) / f).read_text())
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom) and n.module and (".experiments." in n.module
                                                                   or n.module.endswith(".experiments")):
                    count += 1
    assert count <= baseline, f"production→experiments import debt grew to {count} (baseline {baseline}); consolidate"
