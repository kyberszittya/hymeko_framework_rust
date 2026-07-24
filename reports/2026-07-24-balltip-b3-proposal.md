---
title: BALLTIP_COLLISION_ON_V1 — Stage B3: ball-tip teacher bank + refit proposal
date: 2026-07-24
branch: feat/balltip-interarm-filtered-v1
baseline: executable-hymeko-option-rl-v1 @ 772a11a4
status: BALLTIP_PROPOSAL_REFIT_SUFFICIENT (beats clamp zero-shot) but UPDATE-0 IS WEAK — B5/SAC NOT yet justified
---

# BALLTIP_COLLISION_ON_V1 — Stage B3 (2026-07-24)

Case-A action from B2: generate a disjoint ball-tip teacher bank and refit a proposal, keeping the option language and
the frozen clamp `pi_0` settling (B1 showed both transfer). No SAC (B5 is gated on a *strong* update-0). Frozen baseline
untouched; ball labels kept separate from the clamp bank (no silent mixing).

## Result (`reports/2026-07-24-balltip-b3-proposal/b3_proposal.json`, ckpt `carry_proposal_balltip_v1.pt`)
- **Teacher bank:** 99 confident (K6-delivering) ball labels from 160 held-out TRAIN states (seeds 9000–10800, disjoint
  from the 14000–15200 eval panel), 128-shot strong expert, collision-on, matched by transplant. ~62% confident rate.
- **Proposal:** template+residual, K=6 (cluster sizes [10,12,22,17,19,19]); fit clf_ce 0.28, **res_mse 0.40**.
- **Eval (24 disjoint ball states):**

| controller | K6 |
|---|---:|
| ball proposal **b=0** (direct) | **0/24** |
| ball proposal **b=8** (proposal+search) | **5/24** |
| clamp proposal b=8 **zero-shot on ball** | **0/24** |
| full structured expert (192-shot) ceiling | **16/24** |

- Ball-proposal solved {3,10,15,16,17} ⊂ expert solved (16 states) — a **subset**; the refit recovers **31% of the
  achievable ceiling**.

## Reading it
- **`BALLTIP_PROPOSAL_REFIT_SUFFICIENT` — narrowly.** The refit ball proposal + b=8 (5/24) beats the frozen clamp proposal
  zero-shot (0/24) under identical eval. So a ball-specific proposal *does* localise the search better than the clamp's.
- **But the update-0 is WEAK.** b=0 = 0/24 — the proposal's direct θ never delivers; it helps only as a search prior. And
  b=8 (5) leaves an 11-state gap to the 16/24 ceiling. `res_mse 0.40` says the residual does not pin the ball's θ* tightly.
- **Diagnosis (option space, not asserted).** The weakness has several candidate causes to distinguish before trusting it:
  (a) **label noise** — 128-shot single-best labels include lucky finds (bank `robust_k6` varied 0.0–1.0); training on
  non-robust labels blurs the residual; (b) **round-0 only** — the clamp `carry_proposal_refined.pt` was *refined* over
  DAgger rounds, this ball proposal is a single fit; (c) **the ball's direct-delivery is genuinely harder** (a sphere
  contact is less forgiving than the clamp's cupping — consistent with b=0 = 0). These are separable by the B3-iteration
  below, not by assertion.

## B5 gate: NOT PASSED
The plan authorises option-level SAC "only after a strong update-0 exists." 5/24 (31% of ceiling, b=0 = 0) is **not**
strong. Proceeding to SAC now would compare a policy against a weak baseline and likely chase search variance. **B5 is
withheld.**

## Recommended next (B3-iteration, before any B5) — GATED on your go-ahead
1. **Robust-label filter:** keep only labels with `robust_k6 ≥ 0.67` (2/3 jitter re-rolls hold K6) — de-noises the residual.
2. **Refinement rounds:** apply the existing `coin_carry_option_refine.py` loop to the ball bank (DAgger-style), as was
   done for the clamp proposal, and re-measure b=0 / b=8 per round.
3. If b=0 climbs and b=8 approaches the ceiling → a strong ball update-0 → *then* B5.

## Note — expert ceiling variance (measured vs inferred)
B3's expert ceiling is 16/24 (search-best outcome directly); B1's was 13/24 (a separate committed re-run of the best θ).
The 3-state gap is eval-methodology variance (search-best vs re-run), not a contradiction — both show a high ceiling
(≈13–16/24) far above any proposal. B3's 16 is the cleaner number (no re-run env mismatch).

## Files
- **NEW** `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_balltip_proposal.py` (B3 orchestration) — committed 1f105b92
- **EDIT** `coin_carry_option_teacher_bank.py` (+reusable `generate_bank(...,transplant=)`, DRY; clamp path unchanged)
- **NEW artifacts** `carry_proposal_balltip_v1.pt` (ball update-0), `carry_option_balltip_bank_v1.npz` (99 labels)
- **CORE.YAML items touched:** none.
