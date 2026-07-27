"""PNP-0 conformance: schema + adapter conformance for CIP-PNP-01.

Reuses the shared ``hymeko_control.conformance.battery`` so the scenario is held
to the SAME ten CIP-0 guarantees as every other embodiment (no re-implementation).
Requires the physics stack (mujoco/torch) and the hymeko CLI to be available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

import hymeko_control
from hymeko_control.cip.protocol import CIP0Adapter
from hymeko_control.conformance import battery
from hymeko_control.language.validator import ValidationError

from scenarios.pick_place import load_model
from scenarios.pick_place.adapter import PickPlaceCIPAdapter


def test_pnp0_schema_validates() -> None:
    model = load_model()  # raises ValidationError if the .hymeko.yaml is malformed
    assert model.name.startswith("cip_pnp_01")
    assert "APPROACH" in model.mode_names() and "SETTLE" in model.mode_names()
    assert model.is_terminal_mode("SETTLE")


def test_pnp0_schema_rejects_malformed() -> None:
    import yaml
    from scenarios.pick_place import SPEC_PATH
    from hymeko_control.language.validator import validate

    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    del raw["physics"]
    with pytest.raises(ValidationError):
        validate(raw)


def test_pnp0_adapter_is_cip0() -> None:
    adapter = PickPlaceCIPAdapter(model=load_model(), seed=1)
    assert isinstance(adapter, CIP0Adapter)


def test_pnp0_positive_lifecycle_conformance() -> None:
    model = load_model()
    adapter = PickPlaceCIPAdapter(model=model, seed=1)
    records = battery.run_positive_lifecycle(model, adapter, max_ticks=40)
    assert records
    # each tick honoured the CIP-0 contract (bounds / provenance / legal mode)
    for rec in records:
        assert rec.intent.is_bounded()
        assert rec.trace.references(rec.option)


def test_pnp0_core_import_isolation_preserved() -> None:
    core_dir = Path(hymeko_control.__file__).resolve().parent
    assert battery.import_isolation_violations(core_dir) == []
