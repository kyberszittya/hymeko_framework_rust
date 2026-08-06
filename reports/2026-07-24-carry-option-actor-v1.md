---
title: CARRY_OPTION_ACTOR_V1 — semi-MDP carry option, autonomous overnight ledger
date: 2026-07-24
branch: recovery/coin-hymeko-bundle-and-results
status: OPTION_ACTOR_DISTILLATION_BLOCKED_AMORTIZED_SEARCH_PROPOSAL_LOCALIZES
---

# CARRY_OPTION_ACTOR_V1 — autonomous overnight ledger (2026-07-24 02:36 JST)

**One-line status: `OPTION_ACTOR_DISTILLATION_BLOCKED`.** The carry-option advantage is *real and large* (per-state
structured expert **0.533** K6 vs frozen `pi_0` **0.000** on a disjoint 30-state held-out carry panel), but **no
open-loop θ distillation transfers** it (best learned/retrieval variant **0.100** = 0.19× the expert; BC and DAgger option
actors **0.000**). Because the update-0 policy does not clear `pi_0`, **Stage 4 semi-MDP RL over θ was correctly NOT
launched** — that stage is gated on update-0 passing, and it did not. This is a hard-stop *with* a measured mechanism, not
a first-pass verdict on the option idea, which stays open (§10).

Commits (each stage separate, tree clean apart from two pre-existing untracked JSONs from earlier tasks + the new figure):
`cb31beaf` Stage 1 · `ff2bf6fb` Stage 2 · `0fcd2a6d` Stage 3. HEAD `0fcd2a6d`.

---

## 1. Objective and the gate that governs it

Build the deployable carry controller as a **temporally-extended option** — `strict-0 carry state → option actor emits θ
(push/brake/release macro params) → committed stateful executor (physical phase transitions, durations as upper bounds,
safety abort) → robust handoff (strict≥1) → FROZEN settling pi_0 → K6` — and only escalate to genuine semi-MDP RL over θ
**if** an update-0 (BC/DAgger, no RL) option actor first beats `pi_0` at the physical K6 gate. The update-0 gate exists
precisely to stop a wasted RL launch on a policy that cannot even be distilled. It fired.

---

## 2. Frozen contracts honored (verbatim to the mission)

- `pi_0` = `ClipDeterministicActor`, SHA `1902454ca7a74c27` (`frozen/pi0_shared_clip_actor.pt`), loaded `freeze=True`, never
  retrained. The settling policy after handoff is this frozen `pi_0` (+ its `base` late-actor augmentation), unchanged.
- Certificate unchanged: K6 = 6-step held dwell; strict counter = `dtz ≤ CENTER_TOL(0.02) ∧ speed < SETTLE_VEL(0.06)`;
  ENTRY_TOL 0.05; full-containment exit on CENTER_TOL. `k6 = (max_dwell ≥ 6 ∧ touched)`.
- Reward/termination physical meaning untouched; the option executor calls the same `step_ablation` physics + the same
  `stable_engagement_signals` gate update as every prior arc stage.
- Carry→settling hierarchy preserved: the option only governs the strict-0 carry segment; `pi_0` owns settling.
- Seed separation: teacher bank + DAgger-train scan seeds **9000–10800**; the update-0 and diagnostic eval panels are
  **disjoint** seeds **11000–13000**; `pi_0`'s own late_train/late_dev seeds are in the `forbidden` set for every panel.
- "Full open-loop plans are not low-level feedback labels" — respected: the option is a single consistent θ per decision;
  DAgger relabels only the *option-initiation state the student actually reaches* (`recovery_state_theta`), never the
  open-loop suffix of an s0 plan.

---

## 3. Stage 1 — option representation + committed executor (`cb31beaf`)

`hymeko_rl/coin_delivery/coin_carry_option.py`: `OptionActor` (obs48 → θ = 12 amplitudes in ±A_BOUND via tanh, 3 durations
in [T_MIN,T_MAX] via sigmoid — bounds enforced by construction); `option_controller_rollout` (semi-MDP: pick θ, execute
push→brake→release with *physical* phase transitions and durations as upper bounds, re-decide only at option
boundaries / handoff / safety-abort, frozen `pi_0` after handoff); `_safety_abort` (irreversible contact loss after having
had contact, or NaN/gross divergence). 6 focused tests: parameter bounds, deterministic execution + no template mutation,
push-first phase progression, safety-abort predicate, reproducible replay from a saved option-initiation state,
actor-driven re-decision. All pass.

---

## 4. Stage 2 — strong structured teacher bank (`ff2bf6fb`)

Teacher = the **validated uniform structured random-shooting expert** (not CEM — random shooting outperformed it earlier
in the arc). One canonical θ* per state, K6-primary lexicographic (K6 ≻ dwell ≻ fewer exits ≻ contact ≻ less effort ≻
faster). A label is **CONFIDENT only when the option actually delivers K6** through the frozen `pi_0` continuation — never a
merely-least-bad candidate. Scanned **180** held-out strict-0 TRAIN carry states (seeds 9000–10800):

| outcome | count |
|---|---|
| CONFIDENT K6 labels | **105** (contact_retention 82, transport 16, braking 7) |
| near-miss (handoff-only, no K6) | 8 |
| ABSTAIN (no handoff) | 67 |

Robustness recorded per label (small θ-amplitude jitter, re-roll K6): **mean robust_k6 = 0.235** — the winning θ delivers
K6 exactly on its own state but only holds ~24% of the time under a small perturbation. This is the first quantitative
signal that the θ basins are *narrow*. Bank saved: `carry_option_teacher_bank_v1.npz` (obs/θ, sha `8a6d612d3fc515bd`) +
`carry_option_teacher_bank_v1.json` (full per-state provenance). 105 labels ≫ the "toy 20–30" floor the mission forbade.

---

## 5. Stage 3 — option BC + option-level DAgger, update-0 gate (`0fcd2a6d`)

**Pre-checks fired before trusting any full training** (`coin_carry_option_actor.py`):

- **θ normalization round-trip** — invertible (asserted). ✓
- **Tiny-set overfit** — 4 clean labels, MSE **0.008** → the representation *can* fit clean option labels. ✓ (rules out a
  network/optimizer defect; the failure below is not "the actor can't represent θ").
- **Full-bank BC MSE 1.54** ≫ 0.008 tiny floor, and **θ-multimodality cross-seed L2 = [0.0, 1.19, 0.0, 1.47]** (on 2 of 4
  probe states the teacher's canonical θ* differs *wildly* across search seeds) → **a single-mode MSE regressor
  mode-averages a multimodal target**.

**Update-0 evaluation** (disjoint 30-state held-out carry panel, seeds 11000–13000; same executor, only the policy differs):

| policy | K6 | handoff | any_exit |
|---|---|---|---|
| `pi_0` (frozen) | **0.000** | 0.000 | 0.000 |
| BC_option | 0.000 | 0.000 | 0.200 |
| DAgger_option (2 iters) | 0.000 | 0.000 | 0.100 |
| structured_expert (per-state search) | **0.533** | 0.533 | 0.033 |

Gate: `DAgger K6 (0.000) > pi_0 (0.000) + 0.05` is **false** → **update-0 FAILS**. Ceiling (0.533) ≫ clone (0.000): this is a
**distillation gap, not a task ceiling** — the exact pattern the operating contract's evaluation-metric-integrity block
names ("a low score is a gap to explain, not automatically a failure").

---

## 6. Corrective diagnostic — the mechanism (`coin_carry_option_diagnostic.py`)

"One bounded corrective pass before any conclusion." Four θ-selection strategies on the **same** 30-state panel through the
**same** committed executor — isolating *which θ-selection transfers*, so only selection differs:

| θ-selection | K6 | note |
|---|---|---|
| `pi_0` reference | 0.000 | |
| GLOBAL_ROBUST_THETA (one most-robust θ everywhere) | 0.000 | one macro does **not** rule them all |
| RANDOM_BANK_THETA (control) | 0.033 | any confident θ, transplanted |
| NEAREST_RETRIEVAL (nearest bank obs → its θ) | 0.033 | obs-nearest θ does **not** transfer |
| TEMPLATE_CLASSIFY (k-means θ→8 templates, MLP obs→template, execute medoid) | **0.100** | multimodality-respecting |
| structured_expert (per-state search) | **0.533** | the ceiling |

**Precise verdict (three scoped labels, per the user's mid-run correction — the single auto-label `TEMPLATE_TRANSFERS`
was too optimistic):**

- `WEAK_TEMPLATE_LEVEL_TRANSFER_DEMONSTRATED` — classification (0.100, **3/30**) > nearest/random (0.033, 1/30) >
  deterministic BC (0.000). Direction matches theory (classification handles a multimodal target MSE regression averages
  away), but the evidence is **thin and not to be leaned on**: 3 events, no paired uncertainty interval computed, and its
  `any_exit` is **0.133 vs the expert's 0.033** — it exits containment more often. It keeps only ~**19%** of expert coverage.
- `LIVE_PER_STATE_SEARCH_REMAINS_LOAD_BEARING` — the per-state structured search (0.533, 16/30) is the only thing that
  extracts strong performance from the option space. The large, unambiguous effect is the 16/30-vs-3/30 gap.
- `NAIVE_FIXED_THETA_AND_RETRIEVAL_DO_NOT_AMORTIZE_THE_EXPERT` — global single θ (0.000), nearest retrieval (0.033), random
  bank pick (0.033), coarse template classification (0.100), and the earlier deterministic MSE regression (0.000) are **all
  insufficient** one-shot amortizations.

**What is NOT proven (kept explicitly open):** it does *not* follow that no parametric actor can learn θ(s), nor that
per-state search is permanently indispensable. Only the five naive amortizations above are shown insufficient. The open
failure mechanisms remain: (1) multimodality (several distant θ succeed; single-vector regression averages), (2) fine
state-dependence not locally smooth in the current feature space, (3) too little / poorly-covered data (the classifier saw
only sparse modes), (4) knife-edge execution (good θ shatters under small error; robust_k6 0.235), (5) templates too coarse
(classification finds the mode but the within-mode residual is missing).

**Safe claims at this point:** the structured option space is valid; per-state search extracts strong performance from it;
simple one-shot amortization retains only a small fraction (~0.19×) of that performance. **This is not a dead end** — it
reframes the goal from "a net immediately emits the perfect θ" to "the net makes search *cheap and targeted*, and RL then
improves that proposal policy" (§10, §11).

Figure: `reports/figures/2026-07-24-carry-option-distillation.png` (3 panels: update-0, distillation-selection, robustness
histogram).

---

## 7. Why Stage 4 (semi-MDP RL over θ) was NOT launched

The mission gates Stage 4 explicitly: *"genuine semi-MDP SAC/TD3 over θ … if update-0 passes."* It did not pass — the
distilled option actor does not clear `pi_0` (0.000 vs 0.000), and even the strongest non-parametric distillation is
0.19× the ceiling. Launching a γ^τ-bootstrapped macro-critic to *improve* a policy that starts at 0.000 and whose action
target (θ) sits on knife-edge basins would be optimizing on top of an un-distillable representation — the same class of
mistake the contract's §3 forbids (queueing RL before the reward/anchor is certified). RL is deferred pending a
distillable or search-amortized option (§10), not abandoned.

---

## 8. Measured vs inferred vs hypothesis (contract requirement)

- **Measured:** expert 0.533 vs pi_0 0.000 (30 states); BC/DAgger 0.000; distillation family ≤0.100; tiny-overfit 0.008 vs
  full-bank BC MSE 1.54; multimodality L2 [0,1.19,0,1.47]; mean robust_k6 0.235; 105 confident labels / 67 abstains / 180
  scanned. Two independent panels (10-state smoke, 30-state full) show the same expert≫distillation ordering.
- **Inferred:** the distillation failure is caused by (a) multimodality of the obs→θ map and (b) narrow state-specific θ
  basins; (b) dominates. Inference is supported by the robust_k6 measurement and the classification>regression ordering,
  not asserted.
- **Hypothesis (untested tonight, deliberately not run — would be a new campaign):** an MDN / conditional-latent actor, or a
  predicted-θ + cheap online correction (amortized search) at deploy, could close the gap; a closed-loop *receding* option
  (re-search θ at a coarse cadence) is the expert itself and would deploy the advantage at higher inference cost.

---

## 9. Tests, lint, provenance, performance

- **Tests:** `hymeko_rl/tests/test_coin_carry_option.py` — 10 pass in 2.3 s (6 Stage-1 + 4 added: jitter bounds/durations,
  `train_option_bc` tiny-overfit, teacher-label confident-iff-K6 + provenance, `recovery_state_theta`). Coverage gap from
  the Stage-2 commit (new `option_teacher_label`/`_jitter_theta`) closed in the Stage-3 commit.
- **Lint:** `ruff --select F` clean on all three touched files. The only remaining ruff class is **E702** (compact
  semicolons), consistent with the arc modules (`coin_carry_structured.py` has 65) and declared here as a style waiver, not
  a correctness/complexity/F issue. No new `#[allow]`/`# noqa`/`# type: ignore`. No §6.5 anti-patterns introduced (the
  executor is one entry with config, not a per-variant Cartesian dump; algorithm logic stays in the library, not the
  entry).
- **Provenance:** branch `recovery/coin-hymeko-bundle-and-results`, HEAD `0fcd2a6d`; `pi_0` SHA `1902454ca7a74c27`; bank npz
  SHA `8a6d612d3fc515bd`; single-thread torch; seeds — bank/DAgger per-state `rng(1000+i)` / `rng(2000+…)`, eval expert
  `rng(500+i)`, all explicit. Host: Apple-Silicon Mac, CPU (MuJoCo CPU-bound).
- **Performance:** all within budget — teacher bank ~9 min wall (180×64-shot search + robustness), update-0 ~8 min, full
  diagnostic ~5 min; peak RSS well under the 16 GB cap (single-env CPU rollouts, no torch CUDA). No long/overnight run was
  launched (correctly — the gate blocked it), so no multi-hour budget was spent.

---

## 10. The right next architecture — amortized search, not search-free distillation

Per the user's mid-run design note: do **not** try to eliminate the online search immediately. The classifier already
shows a faint pulse (0.100 > 0.000), so it is usable as a *proposal* mechanism. The target architecture is:

```
state → proposal actor / template classifier → K state-specific θ proposals
      → small-budget structured search AROUND the proposals (structured_random_around)
      → committed option execution → frozen settling pi_0
```

i.e. **amortized search**: the net learns *where to search*, the structured planner keeps the robust temporal commitment,
and a small online searcher only *refines* rather than exploring the whole space from zero. Candidate proposal head:
template logits + template-conditioned continuous residual θ + multiple proposal heads.

**Where RL enters (both branches are now legitimately open at the option level):**

1. **Pure option-RL** — template/residual proposal actor → θ execution → option return → semi-MDP SAC/TD3 (target
   `R_option + γ^τ Q_target(s_next, π_target)`). Tests whether reward-driven learning can lift the search-free actor above
   its 0.10 update-0 level.
2. **Search-in-the-loop RL** — the actor emits a proposal *distribution*; the small structured search selects the executed
   θ; RL improves the actor's proposals so that (a) more good candidates appear, (b) a smaller search budget suffices,
   (c) eventual K6 rises, (d) exits fall. Clean, honest RL claim: *RL-trained proposal actor + fixed search budget >
   update-0 proposal actor + the same fixed budget*. This keeps the mechanism that actually delivers the coin.

**Guardrails carried forward:** do not label the 3/30 classifier result a successful option-actor distillation; do not
assert search is permanently indispensable; every claim rests on the paired/fixed-budget comparison above.

## 11. Amortized-search discriminating test (run tonight — the §10 lever, first branch)

`coin_carry_option_amortized.py` implements exactly the amortized-search question: at a fixed small budget *b*, does a
learned-proposal-centred search (`structured_random_around` around the template / BC prediction) recover more of the 0.533
expert than a budget-matched **uniform** search (no proposal) or a **random-bank-centred** search? Sweep *b ∈ {0,4,8,16}*,
30-state disjoint eval panel, `std_amp 0.6 / std_dur 2.0` (local refinement).

**Result** (K6 vs online budget *b*; pi_0 0.000, per-state expert@64 0.533):

| θ-selection | b=0 | b=4 | b=8 | b=16 |
|---|---|---|---|---|
| amortized **TEMPLATE** proposal + search | 0.067 | **0.200** | **0.267** | 0.233 |
| amortized BC proposal + search | 0.033 | 0.100 | 0.100 | 0.100 |
| control: random-bank-center + search | 0.000 | 0.000 | 0.133 | 0.167 |
| control: UNIFORM search (no proposal) | — | 0.100 | 0.033 | 0.200 |

**Verdict `PROPOSAL_LOCALIZES_SEARCH_PARTIAL_RECOVERY`** (`proposal_localizes=True`; best amortized 0.267 = **0.5× the
expert at 1/8 the budget**). The template (classification) proposal **localizes** the search: at small budgets it beats both
the uniform (no-proposal) and random-center controls (b=4: 0.200 vs 0.100/0.000; b=8: 0.267 vs 0.033/0.133), and it beats
the BC (regression) proposal at every budget — coherent with the multimodality finding (classification gives a better
proposal center than mode-averaging regression). This is the measured pulse the §10 architecture predicted: **the net makes
search cheap and targeted.** Figure `reports/figures/2026-07-24-carry-option-amortized.png`.

**Honest hedge (per the no-first-pass-verdict rule):** each (method, b) cell is a single rng draw over 30 states (0.267 =
8/30, 0.200 = 6/30, uniform 0.033 = 1/30); uniform's non-monotone column (0.100→0.033→0.200) shows the magnitude carries
seed/state noise. The **signal** (proposal localizes; template > uniform/random at small b; template > BC) is consistent and
directionally clear; the **exact fraction** (~0.5×) is soft and wants a multi-seed, paired-CI confirmation. I do not stamp
"amortized search solves carry" — I stamp "the proposal localizes the search, partially recovering the expert," which is
what the data support.

---

**Final status: `OPTION_ACTOR_DISTILLATION_BLOCKED` → `AMORTIZED_SEARCH_PROPOSAL_LOCALIZES_PARTIAL_RECOVERY`.**

Naive one-shot distillation of the carry option is blocked (update-0 gate failed; measured compound mechanism = multimodal
+ knife-edge state-specific θ basins; Stage-4 monolithic RL correctly not launched) — sharpened to
`WEAK_TEMPLATE_LEVEL_TRANSFER_DEMONSTRATED` + `LIVE_PER_STATE_SEARCH_REMAINS_LOAD_BEARING` +
`NAIVE_FIXED_THETA_AND_RETRIEVAL_DO_NOT_AMORTIZE_THE_EXPERT`. But the §10 correction is measured to work: a learned
**template proposal localizes a small online search**, recovering 0.5× the per-state expert at 1/8 the budget and beating
both no-proposal and random-center controls (§11). The option space is valid, the advantage is real (0.533 ≫ 0.000), and
the path forward — proposal + budgeted search, then search-in-the-loop RL that improves the proposal at fixed budget — has a
measured pulse. Not a dead end; a redirected one. The clean next RL claim (RL-trained proposal + fixed budget b > update-0
proposal + same b) has its update-0 baseline already measured (template 0.200@b4 / 0.267@b8).
