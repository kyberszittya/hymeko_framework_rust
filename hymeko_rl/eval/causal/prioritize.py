"""CipPrioritizer — rank failures by reward–monitor disagreement (the reason the monitor variables exist).

A rollout where reward and monitor *agree* (both good or both bad) is low value for diagnosis — the reward is a
faithful proxy there. A rollout where they *disagree* is where the learning signal is lying about the task, and
that is where the experiment budget should go (`docs/plans/2026-07-04-hymeko-code-agent/hymeko_to_cip.md` §3).

This module **reads** the disagreement the monitor already computed (via the frame's continuous columns); it does
not re-derive it (§6.1). Each surfaced candidate cause is a **hypothesis to test**, never a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .frame import RolloutFrame

# Disagreement variables → the candidate cause CIP surfaces (hypothesis, isolated later by ablation).
# Maps a frame column name to (higher-is-more-disagreement direction, candidate-cause label).
_DISAGREEMENT: dict[str, str] = {
    "reward_progress_disagreement": "reward_farming_candidate",       # reward ↑ but monitor progress ↛
    "expert_vs_policy_monitor_gap": "covariate_shift_candidate",      # expert succeeds, learner does not
    "critic_vs_monitor_disagreement": "critic_misalignment_candidate",  # critic Q ↑ but monitor success ↓
}


@dataclass(frozen=True)
class PrioritizedFinding:
    """A ranked disagreement channel: which variable, how large, the hypothesised cause, and the worst rollouts."""

    variable: str
    mean_disagreement: float
    max_disagreement: float
    candidate_cause: str
    top_rollouts: list[int]      # indices of the highest-disagreement rollouts (where to look first)

    def as_dict(self) -> dict[str, Any]:
        return {"variable": self.variable, "mean_disagreement": round(self.mean_disagreement, 6),
                "max_disagreement": round(self.max_disagreement, 6), "candidate_cause": self.candidate_cause,
                "top_rollouts": list(self.top_rollouts)}


class CipPrioritizer:
    """Rank disagreement channels and rollouts. Higher disagreement ⇒ higher diagnostic value."""

    def __init__(self, top_k_rollouts: int = 5) -> None:
        self.top_k_rollouts = top_k_rollouts

    def rank(self, frame: "RolloutFrame") -> list[PrioritizedFinding]:
        """Rank the disagreement channels present in ``frame`` by mean magnitude (descending).

        # Postconditions
        * Only channels present as continuous columns are returned; a channel that was entirely defaulted
          (all-missing) is skipped — a fabricated zero is not a real agreement.
        """
        findings: list[PrioritizedFinding] = []
        for name, cause in _DISAGREEMENT.items():
            if name not in frame.continuous:
                continue
            miss = frame.missing.get(name)
            if miss is not None and bool(miss.all()):
                continue
            col = np.abs(frame.continuous[name])
            order = np.argsort(-col)
            top = [int(i) for i in order[: self.top_k_rollouts] if col[i] > 0.0]
            findings.append(PrioritizedFinding(
                variable=name, mean_disagreement=float(col.mean()), max_disagreement=float(col.max()),
                candidate_cause=cause, top_rollouts=top))
        findings.sort(key=lambda f: -f.mean_disagreement)
        return findings

    def rank_rollouts(self, frame: "RolloutFrame") -> list[tuple[int, float]]:
        """Rank individual rollouts by total disagreement across channels (descending). ``(index, score)``."""
        total = np.zeros(frame.n, dtype=np.float64)
        for name in _DISAGREEMENT:
            if name in frame.continuous:
                total += np.abs(frame.continuous[name])
        order = np.argsort(-total)
        return [(int(i), float(total[i])) for i in order]
