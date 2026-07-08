# Fair vector-critic retest — morning report (2026-07-08)

**Run:** `experiments/2026_07_08_vector_retest/` · wall **90.5 s** · CPU, single seed · Mac (arm64, torch 2.12.0 CPU/MPS)
**Driver:** `python -m hymeko_rl.experiments.exp_vector_retest --stage full` · log `run.log` · git `5b53a92` (dirty: pre-existing WIP)
**Plan:** `docs/plans/2026-07-08-fair-vector-critic-retest/` (md/tex/pdf/tikz/mmd)

## Verdict: **NEGATIVE_WITH_MECHANISM**

The fair test was run with the prior obstruction removed, and it produced a clean negative with a **measured**
mechanism — not an ambiguous failure. Two deliverables at once:

1. **Strong diagnostic (the fair test's own question):** with action diversity + measured Monte-Carlo targets, the
   component critics are now **calibrated** (the prior run's blocker is fixed) — yet the vector-projected
   direction is **still not monitor-aligned**. The vector hypothesis is now fairly testable and it **fails**.
2. **Clean negative with mechanism:** every local action move off the demonstrator — scalar ∇Q, vector projected
   ∇, *and* the best of 8 random samples — degrades two-finger contact. The frozen DAgger action is **locally
   contact-optimal**; there is no bounded local improvement for any gradient to find. CEM confirms it.

Stage-ledger sentence: **Scripted controller ~0.84 · BC/DAgger clone (frozen `mlp_s1_selected_d3`) ft_dom 0.75 ·
RL-refined below baseline (this run: projected gradient cuts two-finger contact 0.50→0.20). Best saved checkpoint
remains `bc_clone`/DAgger — no RL refinement improved it.**

## Preflight (all green, recorded)

| gate | result |
|---|---|
| Mac native stack + coin-toss tests | PASS (81/2 quadruped-only fails, unrelated) |
| scripted parity | 0.840 (matches on-record) |
| disk free | 1.6 TiB |
| PipelineSchemaLedger | **PASS** (stage-schema hashes recorded) |
| PolicyProvenanceLedger | **PASS** |
| v2b reward (frozen) `galambos_task_deliver_v2b.hymeko` | oracle-certified **delivers=True, optimal_return=25.404** |
| actor checkpoint | md5 `edf4fe81f04bbda26393ca9f230828b9` (= anchor, stage d3, seed 1) |

## Baseline — frozen DAgger `mlp_s1_selected_d3` (n_eval=24, seed 9000, re-measured on Mac)

| ft_dom | raw_delivery | monitor_pass | monitor_score | body_driven_exploit | arm_body_rate | violation_reason |
|---:|---:|---:|---:|---:|---:|---|
| **0.75** | 0.75 | 0.583 | 0.3635 | 0.0 | 0.0 | fingertips_never_approached |

ft_dom **0.75 matches the on-record** number exactly (physics parity confirmed). The monitor pass/score are the
Mac re-measure and are the grading reference for the smoke/fallback (self-consistent, per plan).

## Stage 1 — action-diverse replay (the fix)

9,000 measured (state, action) rows (1,800 OOD random-action branches) from 14,280 visited states; **1 diverged
episode** (qacc-guarded, dropped + counted). Phase histogram of visited states:

| APPROACH | CONTACT | PUSH | DELIVERY |
|---:|---:|---:|---:|
| 13,863 | 320 | **1** | 96 |

**Structural finding:** sustained two-finger PUSH states are essentially absent (1 of 14,280). The DAgger policy
delivers via APPROACH + brief CONTACT, not sustained pushing — so the contact regime available to optimize
within is thin. This bounds any contact-manifold method, not just the vector critic.

## Stage 2–3 — MC component critics + calibration (the measurement problem is FIXED)

Critics fit by supervised regression to **measured** MC component returns (frozen-policy branch continuations),
OOD shape from the random-action branches. Contrast the prior run, where critics had *negative* Q on
non-negative returns (uncalibrated). Now (`calibration.png`):

| component | spearman(Q,MC) | kendall | within-state action rank | OOD gap | target std | **calibrated** |
|---|---:|---:|---:|---:|---:|:--:|
| approach | +0.780 | +0.517 | 0.77 | −0.010 | 0.0424 | ✅ |
| contact | +0.493 | +0.175 | **0.88** | **+0.083** | 0.2881 | ✅ |
| progress | +0.168 | +0.119 | 0.53 | −0.004 | 0.0038 | ✅ |
| delivery | +0.422 | +0.120 | **1.00** | −0.032 | 0.2438 | ✅ |
| antiexploit | +0.274 | +0.168 | 0.52 | −0.003 | 0.0038 | ✅ |
| body_progress | +0.000 | +0.000 | 0.00 | +0.001 | **0.0000** | ⛔ degenerate |

All projected-relevant critics (delivery, progress, contact, antiexploit) pass within-state action ranking > 0.5
and positive Spearman → **`critics_calibrated=True`**. `body_progress` is degenerate: the frozen policy **never
body-shoves** in-distribution (target identically 0), so that constraint is vacuous here — a clean fact, reported.

## Stage 4–5 — gradient-alignment probe + gate (the decisive measurement)

40 fixed CONTACT/PUSH states, horizon 200, five candidate first-actions branch-rolled under the frozen policy
(`probe_candidates.png`):

| candidate | two-finger rate | ft_progress | monitor_score | arm_body | delivered |
|---|---:|---:|---:|---:|---:|
| **dagger** (baseline) | **0.500** | 0.000 | **−0.186** | 0.0 | 0.225 |
| scalar `+η∇Q_total` | 0.200 | 0.001 | −0.256 | 0.0 | 0.225 |
| **projected** (vector) | 0.200 | 0.001 | −0.253 | 0.0 | 0.225 |
| random | 0.000 | 0.006 | −0.333 | 0.0 | 0.250 |
| best_sampled (best of 8) | 0.325 | 0.004 | −0.217 | 0.0 | 0.225 |

**GATE: `VECTOR_PROJECTED_PROMISING = False`.** projected − dagger: two-finger **−0.30**, monitor_score **−0.068**
(does not preserve). projected ≈ scalar (Δ ≈ 0) — with a near-vacuous body_progress constraint and aligned
contact/anti-exploit, the projection barely changes the scalar direction here.

**The mechanism, measured:** `best_sampled` (0.325) < `dagger` (0.500). Even the best of eight random local
perturbations cannot hold two-finger contact as well as the demonstrator action itself. So this is **not** a
critic-quality or gradient-direction failure — the DAgger action sits on a contact-holding ridge, and *any*
bounded local move (scalar ∇, vector projected ∇, or sampled) falls off it. The local action neighborhood at
engaged states contains no monitor-improving action.

## Fallback — monitor-directed CEM (gate shut; user-authorized, monitor stays verifier)

Bounded phase-gated per-joint residual (θ∈ℝ⁴), objective = SearchObjective components, frozen TaskMonitor as
verifier. Search objective 0.0842 → 0.0870 (marginal). Eval after: ft_dom **0.75** (= baseline), monitor_pass
**0.583** (= baseline), monitor_score **0.352** (baseline 0.3635 — slightly lower), exploit 0.0, arm_body 0.0.
→ **no strict improvement**; the frozen DAgger is a local optimum for this bounded residual class.

## Required fields (morning-report contract)

- **branch taken:** `monitor_directed_cem_fallback` (gate stayed shut).
- **why the gate stayed closed:** projected direction cut two-finger contact 0.50→0.20 and monitor_score −0.068 vs
  baseline. **Not** a measurement artifact — critics were calibrated; the local landscape has no improving move.
- **reward:** v2b, certified delivers=True, optimal_return 25.404 (unchanged).
- **ft_dom:** 0.75 (baseline; CEM after 0.75). **monitor_pass:** 0.583. **monitor_score:** 0.3635 → 0.352 (CEM).
- **violation_reason:** fingertips_never_approached (baseline top violation).
- **engagement / two-finger contact:** probe two-finger dagger 0.500, projected 0.200, best_sampled 0.325.
- **arm-body contact:** 0.0 throughout (no exploit). **body-only progress:** degenerate (0) — policy never shoves.
- **residual norm:** n/a (gate shut → no residual smoke ran); CEM θ ≈ [−0.09, 0, 0.19, −0.06].
- **tensor-contract:** PASS. **policy-provenance:** PASS. **actor/anchor md5:** `edf4fe81…` (param hash `e8ff8f82…`).
- **calibration tables:** above. **verdict:** **NEGATIVE_WITH_MECHANISM**.

## What this settles, and the next fix

- The prior "inconclusive" is **resolved**: with action diversity + measured MC targets, the critics calibrate,
  and the vector-projected gradient is measured to be **not** monitor-aligned. The vector hypothesis, fairly
  tested, does not open the actor-smoke gate. Step 6 (vector actor smoke) **remains correctly withheld**.
- Root cause is now one level deeper than "critics can't learn": **the demonstrator action is locally
  contact-optimal at engaged states** — scalar, vector, and sampled local moves all degrade contact identically.
  Local action refinement (of any critic flavor) is the wrong lever for this task.
- **Next fix (not local):** the historically-proven lever remains **better imitation / broader sustained-contact
  demonstration coverage** (the PUSH regime is nearly unvisited — 1/14,280), or a **temporally-extended**
  sub-policy rather than a per-step residual. Both are non-local and outside what this diagnostic tests.

## Provenance & health

Single seed, CPU, MuJoCo. Peak RSS well under budget (small MLPs + one env). Disk 1.6 TiB free. Ledgers PASS.
No CORE.YAML edit. TaskMonitor stayed the external verifier; the learning objective (SearchObjective) was a
separate object throughout. No scalar TD3/SAC/CQL actor update ran; no multi-seed; v2b reward unchanged.

## Artifacts

- `experiments/2026_07_08_vector_retest/results.json` (full numbers, 14-field provenance)
- `calibration.png` · `probe_candidates.png` (plotted)
- `dagger_baseline.gif` · `best_candidate.gif` (animated, 640px)
- `run.log` (live-logged stages)

## Code (all non-core, tested, ruff-clean)

- `hymeko_rl/env/planar_snapshot.py` — state snapshot/restore + exact branch-from-state component returns.
- `hymeko_rl/train/action_diverse_replay.py` — phase-aware ε-perturbed replay + measured MC targets.
- `hymeko_rl/train/vector_critic.py` — `train_vector_critics_mc` (MC fit) + scale-aware `projected_gradient`.
- `hymeko_rl/eval/component_calibration.py` — per-component Spearman/Kendall/OOD/within-state-rank/sensitivity.
- `hymeko_rl/eval/gradient_probe.py` — long-horizon candidate probe + projected-direction gate.
- `hymeko_rl/train/cem_residual.py` — monitor-directed CEM fallback.
- `hymeko_rl/experiments/exp_vector_retest.py` — the gated driver (one file, `--stage smoke|full`).
- `hymeko_rl/experiments/exp_galambos_coord_ab.py` — additive `action_fn` param on `_coordination_metrics`.
- Tests: `test_{planar_snapshot,action_diverse_replay,component_calibration,cem_residual,vector_retest}.py` +
  extended `test_vector_critic.py` (82 pass total). ruff clean. No §6.5 anti-patterns introduced.
