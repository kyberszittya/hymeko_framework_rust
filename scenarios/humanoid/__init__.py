"""CIP-HUM-01 humanoid scenario adapter (depends on hymeko_control core)."""

from __future__ import annotations

from pathlib import Path

from hymeko_control.language.validator import validate

SPEC_PATH = Path(__file__).with_name("cip_hum_01.hymeko.yaml")


def load_model():
    """Parse + validate the CIP-HUM-01 scenario contract into a ControlModel."""
    import yaml

    raw = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    return validate(raw)


__all__ = ["SPEC_PATH", "load_model"]
