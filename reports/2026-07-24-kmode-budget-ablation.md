---
title: EQUAL_BUDGET_KMODE_ABLATION_V1 — does multimodal proposal+search physically beat single-head at equal budget?
date: 2026-07-24
branch: feat/architectural-assimilation-v1
status: PROVISIONAL — KMODE_NO_DEPLOY_ADVANTAGE_AT_EQUAL_BUDGET (O2 boxes, B=12, prob-weighted allocation); triangle + budget-sweep pending
contract: EQUAL_BUDGET_KMODE_ABLATION_V1
---

# EQUAL_BUDGET_KMODE_ABLATION_V1

The decisive **Track B** control-performance test of the just-frozen `OPTION_RL_STRUCTURED_TEMPORAL_RUNTIME_V1`. The
O2 diagnosis (`MULTIMODAL_OPTION_STRUCTURE_DOMINANT`) said near-identical states have distant winning options that one
MSE centre averages. The architecture's answer was a K-mode proposal + `MultimodalBudgetSearch`. The honest question is
not "does the mixture fit the teacher better" (it does, weakly — 1.34×) but **does it produce a physically better deploy
at the SAME compute** — the user's exact framing: *single-head + total budget B* vs *K-mode + the same total B*, with
the K-mode arms getting **no extra candidates**.

## Setup (what makes this a fair, faithful test)
- **Frozen runtime, no new search code.** The coin binds to `MultimodalBudgetSearch` unchanged. The O2 box proposal's
  K=6 residual-adjusted templates become K strategy modes (`TemplateKModeProposal`); K=1 returns ONLY the classifier-
  argmax template (≡ `proposal.theta(obs)` — the single-head baseline).
- **One shared generator + one shared scorer for every K.** `CoinJitterGenerator` reproduces `structured_random_around`
  exactly (amp std 0.6, dur std 2.0, clipped); `CoinCarryScorer` rolls the committed θ (fresh deepcopy of rl AND gate —
  the gate-contamination fix) and scores by the canonical lexicographic `structured_score` (via a `LexScore` adapter so
  the coin's tuple plugs into the runtime's float contract). So the ONLY thing varying across arms is the number of modes.
- **Equal total budget B=12**, split by `allocate_budget` (≥1/mode, remainder ∝ classifier prob). K=1→[12]; K=6→[7,1,1,1,1,1].
- **Physical grade.** The K6 delivery certificate (6-step dwell), not a fit loss. Fresh-reconstruct O2 box states
  (pi_0 replayed on the ball+box), orientation-stratified, 24 states × 2 search-seeds per shape (2 seeds ⇒ PILOT cap).
- **Object matrix:** square 1:1, rectangle 2:1, rectangle 3:1 (equal area to the r0.020 cylinder). Collision-on ball-tip
  (`BALLTIP_COIN_BASELINE_V1`), frozen reward.

## Result — no K-mode advantage at equal budget
Aggregate over all 144 paired draws (3 shapes × 24 states × 2 seeds):

| arm | K6 rate (all shapes) | Δ vs single-head (K1) | boot95 |
|---|---|---|---|
| **K=1 single-head** | **0.208** | — | — |
| K=2 | 0.194 | −0.014 | [−0.063, +0.035] |
| K=4 | 0.167 | −0.042 | [−0.104, +0.014] |
| K=6 | 0.188 | −0.021 | [−0.076, +0.028] |

Per shape, every Δ(K6−K1) boot95 CI spans zero; on the **most asymmetric** shape (3:1, where multimodality should be
strongest) single-head K1 is nominally the **highest** (0.25 vs K6 0.188, Δ −0.0625 [−0.167, +0.021]):

| shape | K1 | K2 | K4 | K6 | Δ(K6−K1) boot95 |
|---|---|---|---|---|---|
| square 1:1 | 0.167 | 0.125 | 0.208 | 0.188 | [−0.042, +0.104] |
| rect 2:1 | 0.208 | 0.250 | 0.188 | 0.188 | [−0.125, +0.083] |
| rect 3:1 | 0.250 | 0.208 | 0.104 | 0.188 | [−0.167, +0.021] |

**Single-head is at least as good as every K-mode arm at equal budget.** Figure: `kmode_budget.png`.

## Why (the mechanism — this is NOT "K-mode collapsed to K=1")
The selected-mode histogram proves the multimodal exploration genuinely happened — non-argmax modes DO win (rect 2:1,
K=6: mode 0 won only 21/48; modes 2/4/5 won 10/5/5). So the null is not "the mixture never explored." The cause is the
**depth-for-coverage trade at fixed budget**:
- The classifier is argmax-dominated, so `allocate_budget` gives mode 0 the lion's share and each non-argmax mode exactly
  **1** candidate (its bare centre, **no local search**). K=6 = 7 refined samples on mode 0 + 5 unrefined mode-probes.
- Single-head K=1 spends all 12 on refining the (usually-correct) argmax mode.
- Occasionally a non-argmax mode's single probe is the best candidate — but that gain does not offset the lost local-
  search depth on the dominant mode. Net: neutral-to-slightly-negative.

This **corroborates the prior `POLICY_SEARCH_IS_LOAD_BEARING`** finding at the physical-deploy layer: when you already
pay for a budget-B local search, the search — not the proposal's modality — does the work, and splitting the budget
across modes only starves it.

## Verdict (provisional, per the no-first-pass-verdict rule)
`KMODE_NO_DEPLOY_ADVANTAGE_AT_EQUAL_BUDGET` — measured on the O2 boxes at B=12 with prob-weighted allocation. This is
**this arc's measured result**, not a closed verdict on the architecture: three specific follow-ups could still flip it,
and each is a different variable (open them one at a time):
1. **Stronger multimodality — the triangular prism.** Vertices/edges create genuinely distant push strategies the box's
   near-symmetry lacks. This is the O3 geometry proper (mesh manipuland pending) and the strongest remaining case FOR K-mode.
2. **Uniform (not prob-weighted) allocation.** [B/K each] gives every mode real local search instead of a single probe —
   a different `allocate_budget` policy; the current one may be starving the very modes the mixture exists to cover.
3. **Budget sweep.** The K-mode benefit, if any, should appear at LOW budget where single-head local search cannot reach
   a distant mode at all (B ≈ K), and vanish at high budget. B=12 is one point; B∈{6,12,24}×K is the full picture.

What is **measured**: no equal-budget K-mode advantage on boxes. What is **inferred**: search depth is load-bearing, the
prob-weighted split starves non-argmax modes. What is **still hypothesis**: whether stronger geometry / uniform allocation /
low budget would show a benefit.

## Files touched
- `experiments/2026_07_22_coin_v3_learning/rl_entry/coin_kmode_budget_ablation.py` (new, +175) — the ablation; binds the
  frozen runtime to the coin (LexScore, CoinJitterGenerator, CoinCarryScorer, TemplateKModeProposal, run_arm, paired bootstrap).
- `experiments/2026_07_22_coin_v3_learning/rl_entry/plot_kmode_budget.py` (new, +70) — figure + aggregate bootstrap.
- `hymeko_rl/tests/test_kmode_budget_ablation.py` (new, +90) — 8 unit tests (LexScore order + sentinel, jitter bounds+
  determinism, K=1 argmax, top-K ordering/index-keying, distinct centres).
- `reports/2026-07-24-kmode-budget-ablation/{kmode_budget.json, kmode_budget.png, run.log}` — artifacts.

## Tests / provenance
- Unit: 8/8 green (`test_kmode_budget_ablation.py`, 0.83 s). Runtime consumer tests unregressed (25 task-independent green).
- CORE.YAML: none touched (verified — CORE.YAML lists no `hymeko_rl` env/option_rl paths).
- Run: PID 81633, log `reports/2026-07-24-kmode-budget-ablation/run.log`; wall ≈ 6 min single-thread; `torch.set_num_threads(1)`.
- Seeds: fresh-reconstruct panel seeds 14000–15600; search seeds {0,1}. Proposal = `carry_proposal_o2_box_v1.pt` (K=6), reused (not refit).
- §6.5 anti-patterns: none introduced — extends the O2 harness (`fresh_o2_bank`, `SHAPES`, `_ball_tf`) and the frozen
  runtime; no new search/LSTM/eval stack; string-free (enum-free) config; adapter classes, not flags.

## Follow-ups (open one variable at a time)
- [ ] Triangular-prism mesh manipuland → rerun this ablation on it (the O3 geometry proper).
- [ ] `allocate_budget` uniform-vs-prob-weighted A/B at fixed B.
- [ ] Budget sweep B∈{6,12,24} × K.
- [ ] (Track C, parallel) 6D-0 SE(3) pose reach on the frozen runtime — the integration test.
