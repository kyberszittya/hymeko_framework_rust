"""AIBO-0 conformance: schema + adapter conformance for CIP-AIBO-01.

Reuses hymeko_control.conformance.battery. SIMULATION (requires mujoco + hymeko CLI).
Genuine ceiling is AIBO-2 (forward+yaw authority measured); this suite verifies AIBO-0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

import hymeko_control
from hymeko_control.cip.protocol import CIP0Adapter
from hymeko_control.conformance import battery
from hymeko_control.language.validator import ValidationError, validate

from scenarios.aibo import SPEC_PATH, load_model
from scenarios.aibo.adapter import AIBOCIPAdapter


def test_aibo0_schema_validates() -> None:
    model = load_model()
    assert model.name.startswith("cip_aibo_01")
    for m in ("STAND", "WALK", "ALIGN", "STOP", "HOLD"):
        assert m in model.mode_names()


def test_aibo0_schema_rejects_malformed() -> None:
    import yaml

    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    raw["task"]["intents"][0]["lower"] = 9.0  # inverted bounds
    with pytest.raises(ValidationError):
        validate(raw)


def test_aibo0_adapter_is_cip0() -> None:
    assert isinstance(AIBOCIPAdapter(model=load_model()), CIP0Adapter)


def test_aibo0_positive_lifecycle_conformance() -> None:
    model = load_model()
    adapter = AIBOCIPAdapter(model=model)
    records = battery.run_positive_lifecycle(model, adapter, max_ticks=12)
    assert records
    for rec in records:
        assert rec.intent.is_bounded()
        assert rec.trace.references(rec.option)


def test_aibo0_core_import_isolation_preserved() -> None:
    core_dir = Path(hymeko_control.__file__).resolve().parent
    assert battery.import_isolation_violations(core_dir) == []
