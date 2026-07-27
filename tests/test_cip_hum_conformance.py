"""HUM-0 conformance: schema + adapter conformance for CIP-HUM-01.

Reuses hymeko_control.conformance.battery. Requires mujoco + the hymeko CLI.
Genuine ceiling is HUM-1 (balance untestable on the fixed-base humanoid); this
suite verifies HUM-0 only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

import hymeko_control
from hymeko_control.cip.protocol import CIP0Adapter
from hymeko_control.conformance import battery
from hymeko_control.language.validator import ValidationError, validate

from scenarios.humanoid import SPEC_PATH, load_model
from scenarios.humanoid.adapter import HumanoidCIPAdapter


def test_hum0_schema_validates() -> None:
    model = load_model()
    assert model.name.startswith("cip_hum_01")
    for m in ("STAND", "REACH", "TOUCH", "RETRACT", "RECOVER"):
        assert m in model.mode_names()


def test_hum0_schema_rejects_malformed() -> None:
    import yaml

    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    raw["hybrid"]["initial_mode"] = "FLYING"  # undeclared mode
    with pytest.raises(ValidationError):
        validate(raw)


def test_hum0_adapter_is_cip0() -> None:
    assert isinstance(HumanoidCIPAdapter(model=load_model()), CIP0Adapter)


def test_hum0_positive_lifecycle_conformance() -> None:
    model = load_model()
    adapter = HumanoidCIPAdapter(model=model)
    records = battery.run_positive_lifecycle(model, adapter, max_ticks=12)
    assert records
    for rec in records:
        assert rec.intent.is_bounded()
        assert rec.trace.references(rec.option)


def test_hum0_core_import_isolation_preserved() -> None:
    core_dir = Path(hymeko_control.__file__).resolve().parent
    assert battery.import_isolation_violations(core_dir) == []
