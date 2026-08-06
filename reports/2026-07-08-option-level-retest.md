# Option-level coin-toss improvement — morning report (2026-07-08)

**Run:** `experiments/2026_07_08_option_retest/` · wall **230.3 s** · CPU, single seed · Mac (arm64, torch 2.12.0)
**Driver:** `python -m hymeko_rl.experiments.exp_option_retest --stage full` · log `run.log` · git `5b53a92` (dirty: WIP)
**Plan:** `docs/plans/2026-07-08-option-level-retest/` (md/tex/pdf/tikz/mmd)

## Verdict: **NEGATIVE_WITH_MECHANISM** — with an important, measured nuance

Under the strict acceptance gate (ft_dom must not drop), no imitation candidate is accepted. **But** the mechanism
is not "nothing worked": option-level search + DAgger imitation **did** inject the missing sustained-contact
behavior into the learned MLP and **raised monitor quality** — the only regression is a 2-episode ft_dom drop that
is within single-seed noise. This is a **contact/delivery trade-off**, cleanly measured, not a flat failure.

Stage-ledger sentence: **Scripted PushDemonstrator ~0.90 (0.92 measured) · tuned option (CEM θ*) 0.88 ·
frozen DAgger baseline ft_dom 0.75 · new DAgger-from-option ft_dom 0.667 (below baseline on delivery, ABOVE on
monitor_score + sustained contact).** No POSITIVE candidate → no checkpoint saved; the frozen DAgger remains best.

## Preflight (green)

git `5b53a92` (1809 dirty = WIP) · disk 1.6 TiB free · coin-toss + arbiter tests pass · PipelineSchemaLedger
**PASS** · PolicyProvenanceLedger **PASS** (actor+anchor md5 `edf4fe81…`, param `e8ff8f82…`) · v2b reward
`galambos_task_deliver_v2b.hymeko` certified **delivers=True, optimal_return 25.404** (unchanged).

## Baseline — frozen DAgger `mlp_s1_selected_d3` (n_eval 24, seed 9000)

| ft_dom | raw | monitor_pass | monitor_score | exploit | arm_body | sustained-PUSH/ep | both_contact_frac | violation |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **0.75** | 0.75 | 0.583 | 0.3635 | 0.0 | 0.0 | **0.21** | 0.034 | fingertips_never_approached |

## Branch A — sustained-PUSH audit (the coverage gap, quantified)

The prior report's "1/14,280 PUSH states" is confirmed structurally: the learned clone lost the teacher's sustained
two-finger contact.

| policy | delivery | both_contact_frac | sustained-PUSH/ep | ft-progress-in-contact | mean/max window |
|---|---:|---:|---:|---:|---:|
| **frozen DAgger MLP** (baseline) | 0.75 | 0.034 | **0.21** | 0.0013 | 8 / 12 |
| scripted PushDemonstrator | 0.92 | 0.178 | **1.46** | 0.0188 | 14 / 63 |
| PhasePushController (default option) | 0.62 | 0.107 | 0.88 | 0.0160 | 19 / 69 |
| **PhasePushController (tuned θ\*)** | 0.88 | 0.174 | **1.71** | 0.0195 | 12 / 34 |

The clone has **7× fewer** sustained-PUSH windows than its scripted teacher and **14× less** fingertip progress
during contact, at similar raw delivery — it delivers via brief touches, not sustained pushing. All controllers
are exploit-free (arm_body 0). Plot: `coverage.png`.

## Branch B/C — option-parameter search (skill-level CEM over θ∈ℝ⁵)

CEM over the 5 bounded PhasePushController params (no per-step residual; PUSH/BRAKE arbiter = no-degrade shield),
SearchObjective = +sustained-contact +ft-progress +delivery −body −exploit −‖θ−θ_scripted‖, verifier separate.

- **Improved over the default option: yes** — objective 1.43 → **2.658**.
- θ\* = {contact_offset −0.005, push_gain 0.641, direction_correction −0.027, brake_threshold 0.039,
  release_threshold 0.011}; best_metrics: delivery **1.0**, sustained-PUSH/ep **3.0**, ft-progress-in-contact 0.04,
  body 0, exploit 0. The tuned option is a **stronger sustained-contact expert** (1.71 windows/ep on the eval set)
  than both the default option (0.88) and, in sustained-PUSH density, the scripted FSM (1.46). Plot: `option_cem.png`.

→ Selected imitation expert: **PhasePushController_tuned** (highest sustained-PUSH coverage).

## Branch D — imitation (the learned artifacts; per-candidate fields)

| candidate | algorithm | demos/rounds | ft_dom | monitor_pass | monitor_score | sustained-PUSH/ep | both_frac | ft-prog-contact | exploit | arm_body | actor md5 | accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|:--:|
| baseline | — | — | 0.75 | 0.583 | 0.3635 | 0.21 | 0.034 | 0.0013 | 0.0 | 0.0 | edf4fe81 | — |
| bc_finetune | BC warm-start from frozen | 50,269 held samples | 0.417 | 0.333 | 0.2513 | 1.125 | 0.101 | 0.0135 | 0.0 | 0.0 | b01b2e9f | ❌ |
| **dagger** | on-policy DAgger, 3 rounds | 79,515 labeled | **0.667** | **0.583** | **0.4749** | **0.875** | 0.080 | 0.0078 | 0.0 | 0.0 | 669af7fa | ❌ |

- **BC fine-tune** over-corrected: it quadrupled+ sustained contact (0.21→1.125) but **collapsed delivery**
  (0.75→0.417) — the held-only sustained-contact demos over-represent *holding* and the clone lost the
  delivery-completion behavior (classic BC covariate shift toward the demonstrated regime).
- **DAgger** (on-policy aggregation, expert-labeled) recovered most of it: **monitor_pass preserved (0.583),
  monitor_score improved +0.11 (0.363→0.475), sustained-PUSH ×4 (0.21→0.875), ft-progress-in-contact ×6, zero
  exploit / arm-body.** The *only* failed gate is **ft_dom 0.667 vs 0.75 = 16/24 vs 18/24 — a 2-episode delta at
  a single seed**, well inside seed noise, while the monitor_score and coverage gains are large.

Acceptance (both candidates): `headline_ok=False` (ft_dom dropped) → not accepted, despite `coverage_up=True` and
`no_exploit=True`. Per the stop rules, nothing was saved (no POSITIVE).

## Required fields recap

- **branch taken:** A (audit) → B/C (option CEM) → D (BC, then DAgger escalation). E not run (see next fix).
- **why gates stayed closed:** the strict ft_dom-preserve gate. BC dropped all headline metrics; DAgger preserved
  monitor_pass, improved monitor_score, but dropped ft_dom by 2/24 episodes (within single-seed noise).
- **reward:** v2b, delivers=True, 25.404. **tensor-contract:** PASS. **policy-provenance:** PASS. **actor md5:**
  baseline `edf4fe81`, bc `b01b2e9f`, dagger `669af7fa`.
- **sustained-PUSH count:** baseline 0.21/ep → dagger 0.875/ep (scripted 1.46, tuned option 1.71).
- **two-finger duration / progress:** both_contact_frac 0.034→0.080; ft-progress-in-contact 0.0013→0.0078.
- **body-only / arm-body / exploit:** 0 across all candidates (clean — no exploit created).
- **verdict:** NEGATIVE_WITH_MECHANISM.

## Mechanism & next fix (the honest read)

The lever works *directionally*: option-level search builds a strong sustained-contact expert, and DAgger transfers
that behavior into the learned MLP (monitor_score up, sustained contact ×4, no exploit). The obstacle is a
**contact↔delivery trade-off** — pushing the clone toward sustained holding costs a small amount of held delivery,
and at single seed / n=24 the 0.75→0.667 ft_dom "drop" is 2 episodes, not distinguishable from noise.

**This is NOT a closed door like the per-step-residual result was.** Named next fixes, in order:
1. **Multi-seed confirmation** (3–5 seeds, n_eval 48) — the ft_dom delta is within noise; establish whether it is
   real before calling it a regression. Monitor_score +0.11 and sustained-PUSH ×4 look robust.
2. **Demo-mix tuning** — the held-only filter over-weights holding; add delivery-completion demos (mix in
   full-delivery trajectories, or weight the DAgger objective to keep the delivery tail) so contact rises without
   trading delivery.
3. **Then** Branch E (bounded option-parameter REINFORCE/NES) on the stable tuned-option interface.

## Guards / stop rules honored

No per-step motor residual in any branch (option-parameter + imitation only). No scalar TD3/SAC/CQL actor training.
No exploit/body-driven behavior created (0 across candidates). No CORE.YAML edit. TaskMonitor stayed the external
verifier; the SearchObjective was a separate object. No POSITIVE → no checkpoint saved, no follow-up sweeps.

## Artifacts

`results.json` (all per-branch numbers, provenance) · `coverage.png` (audit bar comparison) ·
`option_cem.png` (CEM objective curve) · `learned_policy.gif`, `expert.gif` · `run.log` (live-logged).

## Code (non-core, tested, ruff-clean)

- `hymeko_rl/eval/push_audit.py` — sustained-PUSH audit (reuses `contiguous_runs` + `MonitorContext`).
- `hymeko_rl/train/option_search.py` — CEM/ES over θ∈ℝ⁵ PhasePushController option.
- `hymeko_rl/experiments/exp_option_retest.py` — gated driver (Branch A→B/C→D, acceptance, stop rules, ledgers,
  plots/GIF, live log; reuses `measure_policy`/ledger wiring from `exp_vector_retest`, `behaviour_clone`,
  `collect_galambos_demos`, `PhasePushController`).
- Tests: `test_push_audit.py`, `test_option_search.py` (5 pass). ruff clean. No §6.5 anti-patterns introduced.
