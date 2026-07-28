"""On-disk demonstration bank (JSONL) + success/rejection accounting.

Records are appended one JSON object per line, so a long generation run is checkpointed incrementally and a partial run is
recoverable. Reading reconstructs :class:`DemonstrationRecord` objects. The bank separates the *success denominator*
(admissible rollouts) from the *rejection panel* (INVALID_INITIAL_CONDITION), so a success rate is never inflated or
deflated by invalid starts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hymeko_rl.coin_delivery.demo_bank.record import DemonstrationRecord
from hymeko_rl.coin_delivery.demo_bank.scenario import ScenarioSplit


@dataclass
class BankSummary:
    """Counts over admissible rollouts (the denominator) and the separate rejection panel."""

    admissible: int
    successes: int
    failures_by_class: dict[str, int]
    rejected: int
    rejection_reasons: dict[str, int]

    @property
    def success_rate(self) -> float:
        """Postcondition: successes / admissible in [0, 1]; 0.0 when no admissible rollouts (rejections excluded)."""
        return 0.0 if self.admissible == 0 else self.successes / self.admissible


class DemonstrationBank:
    """A JSONL demonstration store. Append-only writes; full-file reads."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: DemonstrationRecord) -> None:
        """Append one record as a JSON line (creates parent dirs). Postcondition: the file has one more line."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def read(self) -> list[DemonstrationRecord]:
        """Load every record. Postcondition: [] if the file does not exist."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            return [DemonstrationRecord.from_dict(json.loads(line)) for line in fh if line.strip()]

    @staticmethod
    def _tally(labels: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for label in labels:
            out[label] = out.get(label, 0) + 1
        return out

    @staticmethod
    def _in_denominator(r: DemonstrationRecord) -> bool:
        """A rollout counts toward the success denominator iff it is admissible and not an INVALID-panel entry."""
        return r.admissible and r.split != ScenarioSplit.INVALID.value

    def _success_and_failures(self, admissible: list[DemonstrationRecord]) -> tuple[int, dict[str, int]]:
        successes = sum(1 for r in admissible if r.is_success)
        failures = self._tally([r.outcome_label for r in admissible if not r.is_success])
        return successes, failures

    def summarize(self) -> BankSummary:
        """Aggregate the bank into the success denominator + rejection panel."""
        records = self.read()
        admissible = [r for r in records if self._in_denominator(r)]
        rejected = [r for r in records if not r.admissible]
        successes, failures = self._success_and_failures(admissible)
        reasons = self._tally([r.rejection_reason for r in rejected])
        return BankSummary(admissible=len(admissible), successes=successes, failures_by_class=failures,
                           rejected=len(rejected), rejection_reasons=reasons)
