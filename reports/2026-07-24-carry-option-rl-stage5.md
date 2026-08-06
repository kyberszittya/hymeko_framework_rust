---
title: CARRY_OPTION_RL — Stage 4c refinement + Stage 5 search-in-the-loop semi-MDP option RL
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
status: OPTION_PROPOSAL_RL_IMPLEMENTED_FIRST_PASS_INCONCLUSIVE_5B_CONSISTENT_POSITIVE_LEAN
---

# Search-in-the-loop option RL — Stage 4c + Stage 5 (2026-07-24)

Deployable controller: `state → learned proposal center θ₀ → fixed b=8 structured local search → committed push/brake/release
option → frozen settling pi_0 → K6`. This report covers the bounded proposal refinement (Stage 4c) and the genuine
reward-driven semi-MDP RL over the proposal center (Stage 5). It follows the completed distillation/amortization arc
(`reports/2026-07-24-carry-option-actor-v1.md`): naive one-shot θ distillation is blocked (multimodal + knife-edge basins),
but a learned proposal **localizes** a small search (0.5× the per-state expert at 1/8 budget). Stage 5 asks whether
reward-driven learning can move that proposal distribution to a better place while the fixed b=8 search keeps the
state-specific refinement.

## 1. Stage 4c — bounded refinement of the validated head (paired b=0/4/8)

The mode-averaging wall is at the *head*, not the idea: a single-vector MSE proposal averages the multimodal θ target.
Fix (spec-mandated): **template classifier + template-conditioned residual** — keep multimodality in the discrete template
choice, regress only the unimodal within-mode residual (train θ-MSE 4.77 template-only → 1.70 template+residual). Bounded
refinement (≤2 rounds; at each TRAIN state θ₀=proposal(s) → fixed b=8 search → selected θ* → relabel if K6/handoff, else
strong fallback, else ABSTAIN; re-fit), paired eval with **fixed search seeds** on the disjoint 30-state panel:

| round | b=0 K6 (exit) | b=4 K6 (exit) | b=8 K6 (exit) |
|---|---|---|---|
| 0 (template+residual) | 0.000 (0.00) | 0.167 (0.07) | 0.100 (0.00) |
| 1 | 0.067 (0.17) | 0.167 (0.07) | **0.200 (0.00)** |
| 2 | 0.067 (0.07) | 0.167 (0.00) | 0.167 (0.03) |

Best **safe** proposal = round 1 (b=8 0.200 > round-0 0.100, exit not worse) → **RL init = `carry_proposal_refined.pt`**;
`carry_proposal_round0.pt` also kept. Verdict `STAGE4C_REFINEMENT_IMPROVES_SAFE_PROPOSAL`. The 6/30-vs-3/30 b=8 gap is
thin — reported as such; the checkpoint is safe and paired.

**Scoped control (documented negative):** refining the *deterministic global-MSE* continuous BC proposal does NOT tighten
b=0 (0.033 → 0.0 across 3 rounds; MSE stays ~2.3) — mode-averaging persists →
`DETERMINISTIC_BC_PROPOSAL_DOES_NOT_ABSORB_SEARCH`, motivating the residual head. This control does not gate Stage 5.

## 2. Chosen update-0 checkpoint

RL init = `carry_proposal_refined.pt` (Stage-4c best safe). Update-0 proposal baseline on the disjoint **final** panel
(seeds 12000–13000): **b=0 K6 0.083** (exit 0.042), **b=8 K6 0.167** (exit 0.000). The Stage 5 claim is judged against the
b=8 = 0.167 baseline (the deployed budget).

## 3. SAC and TD3 implementation contracts

Action = θ_center (normalised [-1,1]^15). The **fixed** environment wrapper (`SearchWrapperEnv`, unchanged across all
checkpoints/branches): sample exactly b=8 structured candidates around θ_center → select the best by the frozen canonical
`structured_score` → execute ONE committed push/brake/release option → frozen pi_0 continuation to a terminal K6 decision.
Critic **Q(s_k, θ_center)** (twin critics; the stationary search wrapper is part of the env response to the proposal). The
actor is trained through the critic only — **never** backpropagated through the black-box search, and **never** against
θ_selected (that is provenance). SAC = squashed-Gaussian actor + fixed entropy; TD3 = deterministic actor + target-policy
smoothing + delayed actor updates. Both branches share: identical proposal init (distilled, MSE ~0.03), identical
train/dev/final panels, identical fixed b=8 wrapper, identical frozen pi_0, comparable option budgets, disjoint panels.
Reward: certificate-aligned option reward (the raw v3 per-step reward is anti-aligned, audited 7dc46f24) with a documented
fixed `REWARD_SCALE`; model selection on held-out K6, then containment.

## 4. Semi-MDP target verification

Target = `R_option + (1−terminal)·γ^τ·Q_target(s', π_target(s'))`, where **R_option = Σ_{j<τ} γ^j r_{t+j}** is the
discounted sum of per-step certificate-aligned rewards through the *entire* carry→frozen-settle→K6 continuation (the K6
terminal bonus and handoff bonus are inside this sum, discounted to their step — not truncated at handoff). A dedicated
unit test asserts the target uses **γ^τ**, not one-step γ, for both terminal and non-terminal options (and the tensor path).
Reward certified before launch: R_option(K6) mean **9.62** ≫ R_option(non-K6) mean **−0.22** → `delivers=True`.
Option-return distribution (contract smoke, full wrapper): R_med 0.31, **success R_med 10.5 vs fail −0.0** (cleanly
separated), τ ∈ [1,160] (median 9), terminal_frac 0.95 — a healthy, well-separated signal; critic Q stable.

Six pre-RL contract tests pass: γ^τ target (terminal+non-terminal+tensor); option-return determinism/equivalence;
action↔θ round-trip + bounds; env determinism (same state+center+seed → same selected option + transition); Bellman action =
θ_center (θ_selected provenance-separate); terminal-K6 reward visibility.

## 5. Checkpoint-wise physical results

Each branch (SAC/TD3 × 2 seeds), K6 @ **b=8** on the disjoint final panel (24 states). Update-0 proposal baseline = 0.167.
600 options/branch, γ=0.99, fixed b=8, model selection on dev K6 (20 states). Figure
`reports/figures/2026-07-24-carry-option-rl-stage5.png`.

| branch | update0 | early | mid | best_val (selected) | final |
|---|---|---|---|---|---|
| sac_seed0 | 0.167 | 0.083 | 0.125 | 0.125 | 0.167 |
| sac_seed1 | 0.125 | 0.125 | 0.208 | 0.083 | 0.125 |
| td3_seed0 | 0.083 | 0.167 | 0.083 | 0.083 | **0.25** |
| td3_seed1 | 0.167 | 0.000 | 0.042 | **0.167** | 0.208 |

Learning signals present: critic Q rises across training (−0.03 → 0.21 SAC, −0.02 → 0.37 TD3); dev K6 moves (SAC seed0
peaked 0.20 at it 360). SAC dev curves are more stable than TD3's (TD3 seed1 dev collapsed to 0.0 mid-run) — consistent with
the a-priori expectation that SAC handles the noisy/piecewise Q(s, θ_center) landscape better while TD3 is the clean
baseline. The per-branch `update0` varies (0.083–0.167) because each actor is independently distilled (MSE ~0.026–0.028) —
the *shared* proposal baseline used for the claim is 0.167.

## 6. Update-0 vs RL paired comparison

Required claim: **RL (best_val) proposal + fixed b=8 > update-0 proposal + fixed b=8** in held-out K6, no exit growth.
Best dev-selected `best_val` across all four branches = **0.167** (td3_seed1) = update-0 **0.167** → the claim is **NOT
met** (tie, not a strict improvement); containment-exit not worse (`exit_ok=True`). Two *final* (not dev-selected)
checkpoints exceed the baseline (td3_seed0 0.25, td3_seed1 0.208), but on a 24-state panel that is 6/24 vs 4/24 — within
noise and not selected by any principled criterion, so it is **not** claimed as improvement.

## 7. Exact claims and non-claims

**Claims (supported):** (1) the search-in-the-loop semi-MDP option-RL pipeline is implemented to contract — fixed b=8
wrapper, Q(s,θ_center), γ^τ target (unit-tested), option return through the full carry→settle→K6 continuation, actor never
trained against θ_selected; (2) the option reward is certified to rank delivery above non-delivery (K6 9.62 ≫ −0.22) and the
option-return distribution separates cleanly (success 10.5 vs fail −0.0); (3) the critic learns (Q rises) and the policy
moves (dev K6 up to 0.20). (4) The end-to-end controller **does bring the coin to target**: the deployed proposal + b=8
search + committed option + frozen pi_0 delivers K6 on ~0.167 of held-out carry states (0 for frozen pi_0 alone).

**Non-claims (explicit):** (a) RL does **not** beat the update-0 proposal at first pass (best_val ties 0.167); (b) the
2/4 final checkpoints above baseline are within 24-state noise and not dev-selected — no improvement is claimed from them;
(c) the Stage-4c (6/30) and amortized (3/30) signals remain thin and are not leaned on; (d) K6/certificate tolerances,
frozen pi_0, and b=8 were **never** changed; the reward is the certificate-aligned option reward (the raw v3 per-step reward
is anti-aligned, audited 7dc46f24) with a documented fixed scale; (e) this is a *first-pass, modest-budget* result — per the
binding no-verdict-from-first-pass rule it is **not** a verdict that reward-driven option RL cannot beat the proposal.

## 8. Commits and tests

- `9141484c` Stage 4c: proposal library (`coin_carry_proposal.py`) + refinement entry + global-MSE control + 5 tests.
- `763734c0` Stage 5 infra: `coin_carry_option_rl.py` (env, replay, semi-MDP γ^τ, SAC+TD3, distill, eval) + 6 contract tests.
- Stage-5 results commit: _this report + JSON + figure + checkpoints_.
- Tests: 21 pass across `test_coin_carry_option.py` (10), `test_coin_carry_proposal.py` (5), `test_coin_carry_option_rl.py`
  (6). Lint `ruff --select F` clean; E702 compact-semicolon style consistent with arc modules.

## 9. Blockers

No hard-stop mechanism triggered: the semi-MDP target and replay provenance are correct (tested), the search budget was
fixed at b=8 throughout, eval states did not leak into training/refinement (disjoint 9000-10800 / 11000-12000 /
12000-13000), proposals stayed in legal bounds, K6 did not collapse while exit grew, and the critics/actors were
numerically stable (no stabilization needed). The limiting factor is not a contract failure but **first-pass eval
variance**: 20/24-state dev/final panels + a noisy Q(s, θ_center) landscape make dev-based model selection unreliable at
this budget — the selected checkpoint ties the baseline while unselected ones straddle it.

## 10. Next recommended lever

**Variance-reduced model selection before a longer run.** The policy space demonstrably reaches ≥ 0.208–0.25 on the final
panel (td3), but small-panel dev selection did not capture it. Concretely, in priority order: (1) enlarge the dev panel
(≥ 60 states) and select on a bootstrap-CI-lower of dev K6, not the point estimate; (2) 4–6 seeds and report median/IQR of
best_val (per the §3 benchmark discipline); (3) longer budget (≥ 2000 options) now that the smoke/first-pass showed stable
learning signals — the spec's precondition for a longer run is met; (4) reduce the Q(s,θ_center) variance by averaging the
option return over a few fixed search seeds per transition (a lower-variance wrapper response) — keeping b=8 fixed. Prefer
SAC as the primary branch (more stable here), TD3 as the comparison baseline.

---

**Status: `OPTION_PROPOSAL_RL_IMPLEMENTED_FIRST_PASS_INCONCLUSIVE`.** The search-in-the-loop semi-MDP option-RL pipeline
(SAC + TD3) is implemented to contract, the reward is certified, and learning signals are present, but a modest 600-option
first pass with 20/24-state panels ties the strong update-0 proposal (best_val 0.167 = baseline 0.167) and does not
demonstrate improvement; two unselected final checkpoints straddle above baseline within noise. Per the binding
no-verdict-from-first-pass rule this is inconclusive, gated on variance-reduced selection + a longer run — not a verdict
that RL cannot beat the proposal. (Auto-verdict `OPTION_PROPOSAL_RL_NO_PHYSICAL_IMPROVEMENT` retained in the JSON as the
strict-on-best_val reading.)

## 11. Stage 5b — variance-reduced campaign (the follow-up landed)

The §10 lever was executed (`coin_carry_option_rl_stage5b.py`, commit `4077dd51`): larger disjoint panels (train 90 / dev 54
/ untouched final 36), SAC primary (4 seeds) + TD3 control (1), 2000 options, Bellman-safe eval-time multi-search-seed
averaging, and a **pre-registered** checkpoint selection (paired ΔK6 vs the checkpoint's own update-0 → bootstrap-CI-lower →
−any_exit → return). Figure `reports/figures/2026-07-24-carry-option-rl-stage5b.png`.

| branch | RL best-val b8 | own update-0 b8 | ΔK6 | boot-CI |
|---|---|---|---|---|
| sac seed0 | 0.130 | 0.102 | **+0.028** | [−0.056, +0.111] |
| sac seed1 | 0.148 | 0.102 | **+0.046** | [−0.037, +0.120] |
| sac seed2 | 0.139 | 0.102 | **+0.037** | [−0.065, +0.148] |
| sac seed3 | 0.167 | 0.102 | **+0.065** | [−0.019, +0.157] |
| td3 seed0 (control) | 0.083 | 0.102 | −0.019 | [−0.102, +0.065] |

**SAC across-seed: median ΔK6 +0.042, IQR [+0.035, +0.051], 4/4 seeds positive** — the variance reduction turned the
first-pass tie into a consistent positive lean. **But 0/4 per-seed bootstrap CI-lower > 0** (each CI still spans 0 on the
36-state final panel), so it is **not per-seed CI-significant**. TD3 (control) is negative (−0.019), confirming SAC > TD3 for
the noisy Q(s, θ_center) landscape. Verdict stays honest: `OPTION_PROPOSAL_RL_IMPLEMENTED_FIRST_PASS_INCONCLUSIVE` — a
consistent, reproducible-in-direction positive (4/4 SAC seeds, tight IQR), short of a clean statistical win. The 4/4 sign
result is p≈0.06; the next lever to cross significance is more seeds (≥6) and/or a larger final panel to tighten per-seed
CIs — no contract/plumbing change needed.

## 8. Commits and tests

- `9141484c` Stage 4c · `763734c0` Stage 5 infra · `ad4db7e7` Stage 5 results · `4077dd51` Stage 5b campaign.
- Tests: coin RL/proposal/option/fsm/monitor + framework option_rl — full suite green (35 at last count).

---

**Final status: `OPTION_PROPOSAL_RL_IMPLEMENTED_FIRST_PASS_INCONCLUSIVE` (variance-reduced follow-up: consistent +0.042
SAC lean, 4/4 seeds, not yet CI-significant).** The pipeline is implemented to contract, reward certified, and — after
variance reduction — reward-driven proposal-SAC beats its own update-0 at fixed b=8 on all 4 seeds (median +0.042), TD3
control negative. Not yet a clean statistical win (per-seed CIs span 0); the honest reading is a reproducible positive
direction, gated on more seeds/larger panel for significance — not a verdict that RL cannot beat the proposal.

---

**Status: _pending RL completion_.**
