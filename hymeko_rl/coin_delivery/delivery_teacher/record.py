"""Full-provenance record for a target-conditioned delivery+settle attempt (R11.4A).

Per the R11.4A contract, every attempt stores the *context* a later BC needs, not just the actions: coin/target geometry,
relative goal vector, the capture outcome it started from, the delivery+settle parameters + which dimensions were
searched, the phase energy diagnostics, the strict-K6 result, the baseline (frozen-R2) outcome it is compared against, the
failure-class transition, the teacher seed, and a reproducible provenance hash (wall-clock excluded).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

from hymeko_rl.coin_delivery.delivery_teacher.phase_energy import PhaseEnergyLedger, energy_certificate
from hymeko_rl.coin_delivery.delivery_teacher.solver import DeliveryResult

SCHEMA_VERSION = "r11.4a-v1"
SUCCESS = "SUCCESS"


@dataclass(frozen=True)
class DeliveryRecord:
    schema_version: str
    scenario_id: str
    curriculum_stage: str
    split: str
    phase: str
    coin_pose: tuple[float, float]
    target_pose: Optional[tuple[float, float]]
    zone_pose: tuple[float, float]
    relative_goal: tuple[float, float]
    entry_dtz_mm: float
    original_failure_class: str
    baseline_k6: bool
    baseline_min_dtz_mm: float
    teacher_identity: str
    teacher_seed: int
    search_dims: tuple[int, ...]
    theta: tuple[float, ...]
    resolved_k6: bool
    resolved_min_dtz_mm: float
    resolved_safe: bool
    reached_target: bool
    t_coin_entry: Optional[float]
    overshoot_mm: Optional[float]
    dwell_min_dtz_mm: Optional[float]
    energy: dict[str, Optional[float]]
    energy_verdicts: tuple[str, ...]
    energy_complete: bool
    recovered: bool
    outcome_transition: str
    git_sha: str
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = json.loads(json.dumps(asdict(self)))
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeliveryRecord":
        tup = {"coin_pose", "relative_goal", "zone_pose", "search_dims", "theta", "energy_verdicts"}
        kw = {f.name: data[f.name] for f in fields(cls)}
        for name in tup:
            kw[name] = tuple(kw[name])
        if kw["target_pose"] is not None:
            kw["target_pose"] = tuple(kw["target_pose"])
        return cls(**kw)

    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def is_success(self) -> bool:
        return self.resolved_k6


def _energy_dict(led: PhaseEnergyLedger) -> dict[str, Optional[float]]:
    return {"w_actuator_pos": led.w_actuator_pos, "w_actuator_neg": led.w_actuator_neg,
            "peak_coin_ke": led.peak_coin_ke, "e_after_capture": led.e_after_capture,
            "e_braking_onset": led.e_braking_onset, "e_target_entry": led.e_target_entry, "e_release": led.e_release,
            "e_settle_start": led.e_settle_start, "e_terminal": led.e_terminal, "t_coin_entry": led.t_coin_entry,
            "dH_capture_to_entry": led.dH_capture_to_entry, "dH_entry_to_settle": led.dH_entry_to_settle,
            "target_directed_energy_ratio": led.target_directed_energy_ratio, "overshoot_mm": led.overshoot_mm,
            "dwell_min_dtz_mm": led.dwell_min_dtz_mm}


def build_delivery_record(*, scenario_id: str, curriculum_stage: str, split: str, phase: str,
                          coin_pose: tuple[float, float], target_pose: Optional[tuple[float, float]],
                          zone_pose: tuple[float, float], relative_goal: tuple[float, float], entry_dtz_mm: float,
                          original_failure_class: str, baseline_k6: bool, baseline_min_dtz_mm: float,
                          search_dims: tuple[int, ...], result: DeliveryResult, git_sha: str) -> DeliveryRecord:
    """Assemble a delivery record from a solved :class:`DeliveryResult` + its scenario/baseline context."""
    led = result.energy
    cert = energy_certificate(led)
    resolved_label = SUCCESS if result.k6 else original_failure_class
    transition = f"{original_failure_class}->{resolved_label}"
    payload = {
        "schema_version": SCHEMA_VERSION, "scenario_id": scenario_id, "curriculum_stage": curriculum_stage,
        "split": split, "phase": phase, "coin_pose": coin_pose, "target_pose": target_pose, "zone_pose": zone_pose,
        "relative_goal": relative_goal, "entry_dtz_mm": round(entry_dtz_mm, 3),
        "original_failure_class": original_failure_class, "baseline_k6": bool(baseline_k6),
        "baseline_min_dtz_mm": round(baseline_min_dtz_mm, 3), "teacher_identity": "rollout_primitive_delivery_settle_CEM",
        "teacher_seed": int(result.seed), "search_dims": tuple(search_dims), "theta": result.theta,
        "resolved_k6": bool(result.k6), "resolved_min_dtz_mm": round(result.min_dtz_mm, 3),
        "resolved_safe": bool(result.safe), "reached_target": bool(led.reached_target),
        "t_coin_entry": led.t_coin_entry, "overshoot_mm": led.overshoot_mm, "dwell_min_dtz_mm": led.dwell_min_dtz_mm,
        "energy": _energy_dict(led), "energy_verdicts": cert.verdicts, "energy_complete": led.is_complete(),
        "recovered": bool(result.k6 and not baseline_k6), "outcome_transition": transition, "git_sha": git_sha,
    }
    provenance_hash = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "git_sha"}, sort_keys=True,
                   default=list).encode("utf-8")).hexdigest()
    return DeliveryRecord(**payload, provenance_hash=provenance_hash)  # type: ignore[arg-type]
