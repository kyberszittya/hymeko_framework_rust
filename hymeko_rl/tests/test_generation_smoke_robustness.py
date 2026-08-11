"""R11.7A U6A generation smoke — robustness of the summary gate to a *generates-but-does-not-certify* object.

When a variant's MODEL builds and passes the static contracts (handle / collision / mass) but the object fails to
ACQUIRE a certified straddle at S1_SEED (e.g. the round family: a smooth rim gives a contact normal below the
coin-tuned certification threshold), the smoke must record it (``rig_acquisition_failed``) and keep the GENERATION
gate about model health — not crash, and not conflate certification with generation. These tests drive ``_summarize``
directly (no physics) so the regression is fast and deterministic.
"""
from __future__ import annotations

from hymeko_rl.experiments.r11_7a_u6a_generation_smoke import RolloutRecord, _summarize


def _static(mass: float, geom: int, radius: float, *, rig_acquired: bool,
            zero: bool | None) -> dict[str, object]:
    return {"mass": mass, "geom_type": geom, "radius": radius, "handle_ok": True,
            "collision_ok": True, "rig_acquired": rig_acquired, "exact_zero_reset_ok": zero}


def test_generation_gate_passes_and_records_a_non_certifying_object() -> None:
    static = {
        "O0": _static(0.05, 5, 0.02, rig_acquired=True, zero=True),         # reference coin, certifies
        "OX": _static(0.05, 7, 0.025, rig_acquired=False, zero=None),       # a mesh prism that does NOT certify
    }
    rollouts = [RolloutRecord("OX", "center", 0, "RIG_ACQUISITION_FAILED",
                              "rig_no_certified_straddle_at_s1", True, False, False)]
    res = _summarize(static, rollouts)
    # Generation is healthy (model builds, contracts OK, mass differs) even though OX never certifies.
    assert res["gate_pass"] is True
    assert res["rig_acquisition_failed"] == ["OX"]
    assert res["n_model_contract_failures"] == 0
    assert res["certified_capture_by_variant"].get("OX", 0) == 0


def test_generation_gate_still_fails_on_a_real_model_contract_break() -> None:
    # A non-certifying object must NOT mask a genuine model/contract failure elsewhere.
    static = {
        "O0": _static(0.05, 5, 0.02, rig_acquired=True, zero=True),
        "OX": _static(0.05, 7, 0.025, rig_acquired=False, zero=None),
    }
    rollouts = [RolloutRecord("O0", "center", 0, "EXCEPTION:boom", "MODEL_OR_CONTRACT_FAILURE", False, False, False)]
    res = _summarize(static, rollouts)
    assert res["n_model_contract_failures"] == 1
    assert res["gate_pass"] is False
