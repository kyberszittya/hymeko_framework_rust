"""Critic ranking / repair benchmark — critic-only diagnostics over fixed policy/action classes.

The guarded RL sanity sequence localized the failure to the CRITIC: it ranks the body-shove exploit above the
DAgger policy while the formal monitor ranks DAgger above exploit, so the actor follows a semantically wrong
gradient. This module is the measurement apparatus that decides whether a (repaired) critic is trustworthy BEFORE
any actor is trained again — the pure, reusable pieces:

* rank correlations (Spearman / Kendall) — no scipy dependency;
* manipulation-phase labelling (APPROACH / CONTACT / PUSH / DELIVERY) from monitor signals;
* the five required diagnostics (Q-vs-MC, Q-vs-monitor, OOD overestimation, action-perturbation, phase-wise Q);
* the 7-criterion acceptance gate for a repaired critic.

The env rollouts + critic training that feed these live in the run harness; everything here is deterministic and
env-free so it is unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# manipulation phases (the critic is suspected wrong specifically around CONTACT/PUSH)
PHASES = ("APPROACH", "CONTACT", "PUSH", "DELIVERY")


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared) — the basis of Spearman."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x, y) -> float:
    """Spearman rank correlation (Pearson on average ranks). 0.0 for a degenerate constant input."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return 0.0
    rx, ry = _avg_rank(x), _avg_rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall(x, y) -> float:
    """Kendall tau-a (concordant − discordant) / total pairs."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 2:
        return 0.0
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            if s > 0:
                c += 1
            elif s < 0:
                d += 1
    return float((c - d) / (0.5 * n * (n - 1)))


def phase_label(min_tip: float, both_contact: bool, toward: float, in_zone: bool,
                *, near_coin: float, progress_eps: float) -> str:
    """Label one step into a manipulation phase from monitor-level signals (no policy internals)."""
    if in_zone:
        return "DELIVERY"
    if both_contact and toward > progress_eps:
        return "PUSH"
    if both_contact:
        return "CONTACT"
    return "APPROACH"


@dataclass
class ClassMetrics:
    """Per policy/action class: empirical return + reward + monitor + the critic's on-policy Q."""

    name: str
    mc_return: float
    reward: float
    ft_dom: float
    monitor_pass: float
    monitor_score: float
    violation: str
    q_onpolicy: float


def q_vs_return_calibration(classes: list[ClassMetrics]) -> dict:
    """Diagnostic 1 + 2 (class level): does Q rank-correlate with empirical MC return and with monitor score?"""
    q = [c.q_onpolicy for c in classes]
    mc = [c.mc_return for c in classes]
    mon = [c.monitor_score for c in classes]
    return {
        "spearman_q_mc": round(spearman(q, mc), 3), "kendall_q_mc": round(kendall(q, mc), 3),
        "spearman_q_monitor": round(spearman(q, mon), 3), "kendall_q_monitor": round(kendall(q, mon), 3),
    }


def counterfactual_ranking(q_by_action: dict[str, float], monitor_by_action: dict[str, float]) -> dict:
    """Diagnostic 2 (counterfactual, same states): the critical Q(DAgger) > Q(exploit) / Q(one_fingertip) checks."""
    keys = [k for k in q_by_action if k in monitor_by_action]
    order = sorted(keys, key=lambda k: q_by_action[k], reverse=True)
    dag = "mlp_dagger_selected"
    return {
        "Q_by_action": {k: round(q_by_action[k], 3) for k in q_by_action},
        "ranking_best_to_worst": order,
        "Q_dagger_gt_exploit": bool(q_by_action.get(dag, -1e9) > q_by_action.get("body_shove_exploit", 1e9)),
        "Q_dagger_gt_one_fingertip": bool(q_by_action.get(dag, -1e9) > q_by_action.get("one_fingertip", 1e9)),
        "spearman_q_monitor_counterfactual": round(
            spearman([q_by_action[k] for k in keys], [monitor_by_action[k] for k in keys]), 3),
    }


def ood_overestimation(q_dagger: float, q_random: float, q_perturbed_by_mag: dict[float, float]) -> dict:
    """Diagnostics 3 + 4: Q must not be high on random/OOD actions, and should decrease as the action is perturbed
    away from the DAgger action (monotone-ish)."""
    mags = sorted(q_perturbed_by_mag)
    qs = [q_perturbed_by_mag[m] for m in mags]
    # monotone non-increasing in perturbation magnitude (allow a small tolerance)
    decreasing = all(qs[i + 1] <= qs[i] + 1e-3 for i in range(len(qs) - 1)) if len(qs) > 1 else True
    return {
        "Q_dagger": round(q_dagger, 3), "Q_random": round(q_random, 3),
        "Q_random_below_dagger": bool(q_random < q_dagger),
        "Q_by_perturbation": {round(m, 3): round(q_perturbed_by_mag[m], 3) for m in mags},
        "Q_decreases_with_perturbation": bool(decreasing),
        "Q_largest_perturb_below_dagger": bool(qs[-1] < q_dagger) if qs else True,
    }


def phasewise_ranking(q_dagger_by_phase: dict[str, float], q_exploit_by_phase: dict[str, float]) -> dict:
    """Diagnostic 5: per phase, is Q(DAgger) > Q(exploit)? Flags the phases (expected CONTACT/PUSH) where the
    critic mis-ranks the exploit above the demonstrator."""
    out = {}
    misranked = []
    for ph in PHASES:
        qd = q_dagger_by_phase.get(ph)
        qe = q_exploit_by_phase.get(ph)
        if qd is None or qe is None:
            continue
        ok = qd > qe
        out[ph] = {"Q_dagger": round(qd, 3), "Q_exploit": round(qe, 3), "dagger_gt_exploit": bool(ok)}
        if not ok:
            misranked.append(ph)
    return {"by_phase": out, "misranked_phases": misranked}


@dataclass
class CriticAcceptance:
    q_dagger_gt_exploit: bool
    q_dagger_gt_one_fingertip: bool
    q_not_high_on_ood: bool
    spearman_q_monitor_positive: bool
    spearman_q_mc_positive: bool
    tensor_contract_pass: bool
    policy_provenance_pass: bool
    passed: bool = field(init=False)

    def __post_init__(self):
        self.passed = bool(self.q_dagger_gt_exploit and self.q_dagger_gt_one_fingertip and self.q_not_high_on_ood
                           and self.spearman_q_monitor_positive and self.spearman_q_mc_positive
                           and self.tensor_contract_pass and self.policy_provenance_pass)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("q_dagger_gt_exploit", "q_dagger_gt_one_fingertip",
                "q_not_high_on_ood", "spearman_q_monitor_positive", "spearman_q_mc_positive",
                "tensor_contract_pass", "policy_provenance_pass", "passed")}


def acceptance(counterfactual: dict, ood: dict, calibration: dict, *,
               tensor_contract_pass: bool, policy_provenance_pass: bool) -> CriticAcceptance:
    """Assemble the 7-criterion repaired-critic acceptance from the diagnostic blocks."""
    return CriticAcceptance(
        q_dagger_gt_exploit=counterfactual["Q_dagger_gt_exploit"],
        q_dagger_gt_one_fingertip=counterfactual["Q_dagger_gt_one_fingertip"],
        q_not_high_on_ood=bool(ood["Q_random_below_dagger"] and ood["Q_largest_perturb_below_dagger"]),
        spearman_q_monitor_positive=calibration["spearman_q_monitor"] > 0.0,
        spearman_q_mc_positive=calibration["spearman_q_mc"] > 0.0,
        tensor_contract_pass=tensor_contract_pass, policy_provenance_pass=policy_provenance_pass)


# STRONG-pass margin thresholds: a knife-edge ranking (baseline +0.05) technically passes the boolean gate but
# flips under actor drift and collapses the smoke — only a decisive margin is ACTOR-SAFE (report 2026-07-07-*).
STRONG_MARGIN_EXPLOIT = 3.0
STRONG_MARGIN_ONE_FINGER = 3.0
STRONG_OOD_GAP = 5.0


@dataclass
class CriticClassification:
    """Three-tier margin-aware critic verdict: FAIL / WEAK_PASS (ranks right but fragile) / STRONG_PASS (actor-safe)."""

    tier: str                     # "FAIL" | "WEAK_PASS" | "STRONG_PASS"
    margin_exploit: float         # Q(DAgger) − Q(exploit) on DAgger states
    margin_one_fingertip: float   # Q(DAgger) − Q(one_fingertip)
    ood_gap: float                # Q(DAgger) − Q(random)
    spearman_q_monitor: float
    spearman_q_mc: float
    perturbation_ok: bool
    tensor_contract_pass: bool
    policy_provenance_pass: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("tier", "margin_exploit", "margin_one_fingertip", "ood_gap",
                "spearman_q_monitor", "spearman_q_mc", "perturbation_ok", "tensor_contract_pass",
                "policy_provenance_pass", "reasons")}


def classify_critic(counterfactual: dict, ood: dict, calibration: dict, *,
                    tensor_contract_pass: bool, policy_provenance_pass: bool,
                    margin_exploit: float = STRONG_MARGIN_EXPLOIT,
                    margin_one_fingertip: float = STRONG_MARGIN_ONE_FINGER,
                    ood_gap: float = STRONG_OOD_GAP) -> CriticClassification:
    """Margin-aware 3-tier classification. STRONG_PASS = the basic ranking holds AND the exploit/one-fingertip
    margins ≥ thresholds AND random/OOD is suppressed ≥ ``ood_gap`` below the DAgger action (actor-safe). WEAK_PASS
    = ranks correctly but on a fragile margin (not actor-safe). FAIL = ranking or a guard fails."""
    q = counterfactual["Q_by_action"]
    qd, qe, q1 = q["mlp_dagger_selected"], q["body_shove_exploit"], q["one_fingertip"]
    m_e, m_1, gap = qd - qe, qd - q1, qd - ood["Q_random"]
    spm, spc = calibration["spearman_q_monitor"], calibration["spearman_q_mc"]
    pert_ok = bool(ood["Q_decreases_with_perturbation"] and ood["Q_largest_perturb_below_dagger"])
    guards = bool(tensor_contract_pass and policy_provenance_pass)
    basic = bool(m_e > 0 and m_1 > 0 and ood["Q_random_below_dagger"] and spm > 0 and spc > 0 and pert_ok and guards)
    strong = bool(basic and m_e >= margin_exploit and m_1 >= margin_one_fingertip and gap >= ood_gap)
    tier = "STRONG_PASS" if strong else ("WEAK_PASS" if basic else "FAIL")
    reasons: list[str] = []
    if not guards:
        reasons.append("guard_failed")
    if m_e <= 0:
        reasons.append("Q_exploit>=Q_dagger")
    elif m_e < margin_exploit:
        reasons.append(f"margin_exploit {m_e:.2f}<{margin_exploit}")
    if m_1 < margin_one_fingertip:
        reasons.append(f"margin_one_finger {m_1:.2f}<{margin_one_fingertip}")
    if gap < ood_gap:
        reasons.append(f"ood_gap {gap:.2f}<{ood_gap}")
    if spm <= 0:
        reasons.append("spearman_q_monitor<=0")
    if spc <= 0:
        reasons.append("spearman_q_mc<=0")
    if not pert_ok:
        reasons.append("perturbation_rewards_drift")
    return CriticClassification(tier, round(m_e, 3), round(m_1, 3), round(gap, 3), round(spm, 3), round(spc, 3),
                                pert_ok, tensor_contract_pass, policy_provenance_pass, reasons)
