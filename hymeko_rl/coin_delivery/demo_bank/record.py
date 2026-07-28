"""The R11.3 demonstration record: one JSONL-serializable row per rollout, with a content hash and replay comparison.

A :class:`DemonstrationRecord` captures everything the pilot must store: the initial-condition certificate result, the
distribution sample ID, admissibility + rejection reason, the RRT reach + reach certificate, coin/target/zone poses and the
relative goal vector, the precontact handoff completeness, the mode trace M0..M7, the teacher's structured capture
parameters + entry observation + contact sequence (the option-level actions/observations), the strict-K6 outcome, the
failure classification, the measured energy ledger + its verdicts, and the rollout provenance + content hash. The record is
built from primitives (already extracted from the IR objects by the pipeline) so serialization is exact and lossless;
:meth:`content_hash` is a stable digest and :func:`replay_matches` checks a re-run reproduces the recorded result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

SUCCESS_LABEL = "SUCCESS"
NON_DETERMINISTIC_FIELDS = frozenset({"planning_time_s"})  # wall-clock — excluded from the reproducible content hash


@dataclass(frozen=True)
class DemonstrationRecord:
    schema_version: str
    scenario_id: str
    split: str
    kind: str
    curriculum_stage: str
    seed: int
    sample_id: str
    ic_valid: bool
    ic_violations: tuple[str, ...]
    admissible: bool
    rejection_reason: str
    coin_pose: tuple[float, float]
    target_pose: Optional[tuple[float, float]]
    relative_goal: tuple[float, float]
    zone_pose: tuple[float, float]
    target_relocation_honored: bool
    reach_found: bool
    goal_set_empty: bool
    n_goals: int
    planning_time_s: float
    reach: Optional[dict[str, float]]
    min_link_clr_mm: Optional[float]
    premature_contacts: int
    handoff_complete: bool
    handoff_admissible: bool
    mode_trace: tuple[int, ...]
    teacher_identity: str
    teacher_seed: int
    teacher_params: dict[str, float]
    entry_obs: dict[str, Optional[float]]
    contact_sequence: tuple[int, ...]
    contacts: int
    cos_dir: Optional[float]
    vel_scale: Optional[float]
    dtau: Optional[float]
    k6: bool
    entry_dtz_mm: Optional[float]
    min_dtz_mm: Optional[float]  # None when no capture ran (keeps the bank valid JSON and equality stable)
    coin_progress_mm: Optional[float]
    safe: bool
    outcome_label: str
    energy_ledger: dict[str, float]
    energy_verdicts: tuple[str, ...]
    energy_measurement_complete: bool
    git_sha: str
    provenance_hash: str
    k6_provenance_link: bool

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict (tuples -> lists). Postcondition: round-trips through :meth:`from_dict` to an equal record."""
        result: dict[str, Any] = json.loads(json.dumps(asdict(self)))
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DemonstrationRecord":
        """Reconstruct a record, restoring the tuple-typed fields JSON flattened to lists."""
        tuple_fields = {"ic_violations", "coin_pose", "relative_goal", "zone_pose", "mode_trace",
                        "contact_sequence", "energy_verdicts"}
        kw = {f.name: data[f.name] for f in fields(cls)}
        for name in tuple_fields:
            kw[name] = tuple(kw[name])
        if kw["target_pose"] is not None:
            kw["target_pose"] = tuple(kw["target_pose"])
        return cls(**kw)

    def content_hash(self) -> str:
        """Stable SHA-256 over the *reproducible* content — wall-clock timings (``planning_time_s``) are excluded so a
        deterministic re-run of the same scenario hashes identically. Postcondition: a 64-char hex digest."""
        payload = {k: v for k, v in self.to_dict().items() if k not in NON_DETERMINISTIC_FIELDS}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def is_success(self) -> bool:
        return self.outcome_label == SUCCESS_LABEL


def replay_matches(recorded: DemonstrationRecord, rerun: DemonstrationRecord) -> bool:
    """A replay reproduces the scenario and result: same provenance hash, same outcome label, same strict-K6 verdict, and
    an identical record content hash. Postcondition: True iff all four agree."""
    return (recorded.provenance_hash == rerun.provenance_hash
            and recorded.outcome_label == rerun.outcome_label
            and recorded.k6 == rerun.k6
            and recorded.content_hash() == rerun.content_hash())
