from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "verification" / "epigene16"))

import capacity_sympy  # noqa: E402
import invariants_z3  # noqa: E402


def test_capacity_sympy_theorems() -> None:
    assert capacity_sympy.theorem_raw_capacity_uniform()
    assert capacity_sympy.theorem_monotonic_capacity()
    assert capacity_sympy.theorem_compression_ratio_monotonic()
    assert capacity_sympy.theorem_epigene16_witnesses()


def test_z3_positive_invariants() -> None:
    assert invariants_z3.theorem_authority_cannot_bypass_transaction()
    assert invariants_z3.theorem_contact_critical_implies_contact_guards()
    assert invariants_z3.theorem_high_expression_requires_evidence()


def test_z3_negative_guards_are_load_bearing() -> None:
    assert invariants_z3.negative_contact_guard_is_load_bearing()
    assert invariants_z3.negative_authority_guard_is_load_bearing()

