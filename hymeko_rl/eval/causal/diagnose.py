"""CausalDiagnosis — orchestrate frame → DirectLiNGAM (continuous only) + CipPrioritizer → a proposed diagnosis.

The single entry point of the CIP diagnostic layer. It enforces the load-bearing contract — **only continuous
columns enter the linear model; categoricals stratify** — as a raise, not a hope, and packages the output as a
:class:`DiagnosisReport` in which every edge, ordering, and intervention is explicitly a **proposal to test by
controlled ablation**, never a verdict (framework doctrine).

Intervention templates encode the two on-record failure chains:

* **Imitation chain** — expert monitor-success ≫ learner ⇒ covariate shift ⇒ close the imitation gap
  (more demos / DAgger) *before* off-policy RL.
* **RL / value-drift chain** — critic Q rises while monitor success falls ⇒ freeze RL, strengthen BC-reg,
  DAgger / phase-gated residual only (matches the TD3+BC value-drift collapse; DAgger avoids it by staying in
  imitation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .lingam import DirectLiNGAM, LingamResult
from .prioritize import CipPrioritizer, PrioritizedFinding

if TYPE_CHECKING:
    from .frame import RolloutFrame

# candidate cause → (next-intervention headline, the controlled test that would decide it).
_INTERVENTION_TEMPLATES: dict[str, tuple[str, str]] = {
    "reward_farming_candidate": (
        "Reward rises without monitor progress — audit reward vs task BEFORE spending RL budget "
        "(action scaling, reward timing, anchor, add the monitor term).",
        "A/B the suspected reward term on/off; the monitor-success delta decides whether it is farming."),
    "covariate_shift_candidate": (
        "Expert monitor-success ≫ learner — imitation gap (covariate shift); close it with DAgger / more demos "
        "before off-policy RL.",
        "DAgger round vs. BC-only, graded on monitor success; a rising learner gap under DAgger confirms shift."),
    "critic_misalignment_candidate": (
        "Critic Q rises while monitor success falls — value drift; freeze RL, strengthen BC-reg, "
        "DAgger / phase-gated residual only.",
        "Freeze the actor, retrain the critic with stronger BC anchor; monitor-vs-Q rank inversion must vanish."),
    "none": (
        "Reward and monitor agree — no dominant disagreement; prioritize the causal ordering's outcome leaf "
        "and its strongest proposed parent.",
        "Ablate the strongest proposed parent edge and measure the outcome leaf (controlled A/B)."),
}


@dataclass(frozen=True)
class DiagnosisReport:
    """A **proposed** causal diagnosis. Nothing here is proof; the ablation plan is what would decide it."""

    causal_order: list[str]
    strongest_edges: list[tuple[str, str, float]]
    failure_ranking: list[PrioritizedFinding]
    next_intervention: str
    ablation_plan: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "causal_order": list(self.causal_order),
            "strongest_edges": [[c, e, round(w, 6)] for c, e, w in self.strongest_edges],
            "failure_ranking": [f.as_dict() for f in self.failure_ranking],
            "next_intervention": self.next_intervention,
            "ablation_plan": list(self.ablation_plan),
            "provenance": dict(self.provenance),
            "_disclaimer": "DirectLiNGAM PROPOSES structure; controlled ablations DECIDE. Not proof of causality.",
        }


class CausalDiagnosis:
    """Run the CIP diagnostic layer over a :class:`RolloutFrame`. Holds no mutable state (config-only, §6.5 #11)."""

    def __init__(self, lingam: DirectLiNGAM | None = None, prioritizer: CipPrioritizer | None = None,
                 n_strongest_edges: int = 5) -> None:
        self.lingam = lingam or DirectLiNGAM()
        self.prioritizer = prioritizer or CipPrioritizer()
        self.n_strongest_edges = n_strongest_edges

    def run(self, frame: "RolloutFrame", continuous_names: list[str] | None = None) -> DiagnosisReport:
        """Diagnose one (already-stratified) frame.

        # Preconditions
        * ``frame`` is a :class:`RolloutFrame`; any name in ``continuous_names`` must be a continuous column
          (a categorical name raises — categoricals never reach the linear model).
        # Postconditions
        * Returns a :class:`DiagnosisReport`; if too few usable continuous columns / samples exist to fit
          DirectLiNGAM, the causal fields are empty and ``provenance['lingam']`` states why (the prioritizer
          ranking is still produced).
        """
        matrix, kept, dropped = frame.continuous_matrix(continuous_names)   # raises on a categorical name
        findings = self.prioritizer.rank(frame)

        lingam_result, lingam_note = self._fit_lingam(matrix, kept)
        order = lingam_result.ordered_names() if lingam_result else []
        strongest = lingam_result.strongest_edges(self.n_strongest_edges) if lingam_result else []

        headline, controlled_test = self._select_intervention(findings)
        ablation_plan = self._ablation_plan(strongest, controlled_test)

        provenance = {
            "n_rollouts": frame.n,
            "continuous_used": kept,
            "continuous_dropped": dropped,
            "categorical_strata": list(frame.categorical),
            "lingam": lingam_note,
        }
        return DiagnosisReport(order, strongest, findings, headline, ablation_plan, provenance)

    def run_stratified(self, frame: "RolloutFrame", stratify_by: list[str],
                       continuous_names: list[str] | None = None) -> dict[tuple[Any, ...], DiagnosisReport]:
        """Run :meth:`run` independently per categorical stratum — categoricals stratify, never mix into the model."""
        groups = frame.group_by(stratify_by)
        return {label: self.run(frame.subset(idx), continuous_names) for label, idx in groups.items()}

    # -- internals --------------------------------------------------------------------------------------------
    def _fit_lingam(self, matrix: np.ndarray, names: list[str]) -> tuple[LingamResult | None, str]:
        """Fit DirectLiNGAM if enough columns/samples exist; else return ``(None, reason)`` — never fabricate."""
        n, d = matrix.shape
        if d < 2:
            return None, f"skipped: need >= 2 usable continuous columns, have {d}"
        if n <= d:
            return None, f"skipped: need n_samples > n_vars, have n={n}, d={d}"
        return self.lingam.fit(matrix, names), f"fit: n={n}, d={d}"

    @staticmethod
    def _select_intervention(findings: list[PrioritizedFinding]) -> tuple[str, str]:
        """Pick the intervention template from the dominant disagreement finding (``none`` if all agree)."""
        cause = findings[0].candidate_cause if findings and findings[0].mean_disagreement > 0.0 else "none"
        return _INTERVENTION_TEMPLATES.get(cause, _INTERVENTION_TEMPLATES["none"])

    def _ablation_plan(self, strongest: list[tuple[str, str, float]], controlled_test: str) -> list[str]:
        """Concrete controlled tests: the intervention's decider, then an ablation per top proposed edge."""
        plan = [f"Decider: {controlled_test}"]
        for cause, effect, weight in strongest:
            sign = "increase" if weight > 0 else "decrease"
            plan.append(f"Ablate/intervene on {cause!r}; LiNGAM proposes it {sign}s {effect!r} "
                        f"(w={weight:+.3f}) — controlled A/B decides if the edge is real.")
        return plan
