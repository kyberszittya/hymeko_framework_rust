---
title: Beneficial-support audit V1 (Stage A) + Stage-B gate decision
date: 2026-07-23
branch: recovery/coin-hymeko-bundle-and-results
terminal: HOLD_SIGNAL_DOMINATED_BY_HARM
tags: [coin, residual, beneficial-support, harm-domination, preregistered, negative-for-improvement]
---

# STAGE A — HOLD_SIGNAL_DOMINATED_BY_HARM  →  STAGE B not run

**Label-only. No rollouts collected, no critic trained, no actor, no SAC, gate/reward/residual-bound unchanged, sealed
panels unopened.** Thresholds frozen before analysis (in `beneficial_support_audit_v1.json:frozen`).

## Provenance of the labels
The completed hold-sweep (config `8d85923`, results `5023cf1`) serialized only aggregates, so the per-candidate labels
were **re-materialized from the frozen preregistration**: recaptured panel state-manifest SHA `c9ea1bea` **matches**,
and the re-materialized median|ΔG| per K/family **matches the completed sweep aggregates exactly**
(`matches_completed_aggregates: true`; determinism ×2 already certified in the sweep). The completed sweep's config and
results are unmodified. State identity = the captured obs / base action / causal critic-state / gate state stored in the
labels; nothing is paired against a post-restore recomputed observation (that observation is non-reproducible — see the
sweep report).

## Frozen definitions
beneficial ⟺ ΔG > 1 (also reported at >5, >10); harmful ⟺ ΔG < −1 **or** contact-break **or** target-exit;
contact-preserving ⟺ `contact_persist`; non-harmful ⟺ contact-preserving ∧ ¬target-exit ∧ ΔG ≥ −1; separability margin
= best − second-best ≥ 1.0; support_min = 0.30; K chosen by beneficial support + separability, **not** by max |ΔG|.

## Result — the leverage increase is harm-dominated
- **More groups gain *some* beneficial candidate as K grows** (fraction of groups with ≥1 ΔG>1 candidate: transport
  0.60→1.00, entry 0.50→1.00, contact_retention 0.50→0.90; "only neutral/harmful" groups drop 0.45→0.125). This is the
  benign face of the K-hold leverage increase.
- **But those beneficial candidates break contact.** Beneficial **contact-preserving** support is ~absent: transport
  0/10, contact_retention 0/10, settling 0–1/10, entry 2/10 (flat across K). Pooled beneficial-contact-preserving:
  0.05 → 0.075.
- **Median best NON-harmful ΔG = 0.00 at every K and every family** — for the median state, no contact-preserving
  residual beats the zero residual. The safe gain is nonexistent.
- **Safe + separable ("usable") support = 0.0 at every K.** No group has a beneficial, contact-preserving, non-exit
  candidate that also clears the separability margin. **No eligible K.**
- Beneficial-candidate identity is unstable across K (Jaccard 0.40–0.64), so even the (contact-breaking) beneficial set
  is not consistent.

## Verdict and K-selection
`HOLD_SIGNAL_DOMINATED_BY_HARM`. Pooled usable support {K2..K16} = 0.0 < support_min 0.30 → **eligible_K = ∅,
selected_K = None**. The nuance (reported, not hidden): beneficial contact-preserving support is not *literally* zero —
~2–3 of 40 groups (concentrated in `entry`) — but it is sparse, non-separable, and does not rise with K, while harm
(contact-break ≈ 0.94→0.97, harmful>beneficial per candidate) grows. The frozen safe+separable criterion a critic would
need to act on is met by zero groups.

## Stage-B gate decision
`HARM_GATED_ADVANTAGE_CRITIC_V3` **is not run.** Its run condition — "sufficient beneficial support at an eligible K" —
is not met (no eligible K; median best-non-harmful ΔG = 0). Building the twin harm/advantage critic would give the
advantage head an essentially empty beneficial set to rank; the harm head would have plenty to learn (harm is large and
structured), but that is a **safety filter**, and the gate to build even that here is beneficial support, which is
absent. Consistent with the sweep report's flagged risk: "if only harm-rejection survives, the residual line is a safety
filter, not a policy-improvement lever." This audit resolves it to the harm-dominated side.

## Claims / non-claims
**Claims:** (1) The K-hold leverage increase (established in the sweep) is **not** backed by usable beneficial support:
safe+separable support = 0 at every K; median best-non-harmful ΔG = 0 at every K/family. (2) The increase is
harm-dominated (beneficial candidates predominantly break contact; harm fraction and contact-break rise with K).
(3) No eligible K for a policy-improving residual critic under the frozen safety criterion.
**Non-claims:** NOT that harmful residuals are unlearnable (the harm structure is large — a pure safety filter is a
separate, untested question the Stage-B gate did not authorize here). NOT that beneficial support is *exactly* zero
(entry has 2 sparse groups). NOT a critic/actor/sealed result. 10 groups/family — adequate to reject a 30% support
hypothesis, not to resolve a 5% one tightly.

## Next narrow experiment (if the residual line is pursued as safety-only)
A separate objective, explicitly authorized: a **harm-classifier-only** head (drop the advantage head) trained on the
K-hold labels to reject contact-breaking / target-exit residuals — evaluated purely on harm recall and
false-negative rate. That would answer "is the residual line usable as a safety filter?" without pretending it improves
delivery. It is out of scope here because Stage A's beneficial-support gate for `..._CRITIC_V3` failed.

## Files
- impl/results (this commit): `experiments/…/hold_labels_materialize.py`, `beneficial_support_audit_v1.py`,
  `make_support_audit_figure.py`, `hold_sweep_v1_labels.json` (re-materialized raw labels),
  `beneficial_support_audit_v1.json`, this report, `reports/figures/2026-07-23-beneficial-support-audit.png`
- upstream: sweep (`62f0412` impl, `8d85923` frozen prereg, `5023cf1` results).

## Provenance
Branch `recovery/coin-hymeko-bundle-and-results`; pi_0 `1902454c`; Mac, torch 2.12.0, mujoco 3.10.0. Labels
re-materialized deterministically from frozen prereg (state-manifest sha `c9ea1bea`, aggregates verified identical).
