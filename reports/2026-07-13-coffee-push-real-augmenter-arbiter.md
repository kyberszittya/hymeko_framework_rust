# Real coffee-push — the LLM → HyMeKo augmenter+arbiter loop on MetaWorld data

**Date:** 2026-07-13 · Aiko · branch `hymeko-neuro-migration` · rollouts on **kato15** (metaworld), pipeline on the
Mac (local ollama + OpenAI + our `hymeko_pgraph`). Closes the synthetic arc onto real data: does the
LLM→augment→arbitrate stack produce a faithful coffee-push success spec from real MetaWorld rollouts?

## Setup

200 coffee-push episodes rolled on kato15 (`SawyerCoffeePushV3Policy` + action noise ∈ {0,0.4,0.8,1.2} for both
classes), per-step signal traces (`near_object, grasp_success, in_place, obj_to_target`) + native success saved to a
4 MB JSON, rsync'd to the Mac (`metaworld_rollouts.roll_coffee_push` / `save`/`load`). Split 100 verif / 100 test
(70% positive). Same LLM→gate pipeline as synthetic (repair + calibration + `hymeko_pgraph` SSG conjunct-pruning),
graded by F1 vs native success.

**The real task is cleanly specifiable** and confirms the setup: `F(obj_to_target <= 0.071)` → **F1 1.0**; and
`near_object`/`grasp_success` are **dead constants** (coffee-push is a push task, no grasp) — real noise-signal
traps, exactly the kind the arbiter must prune.

## Result (F1 vs native success, real test split)

| model | raw F1 | gate F1 | note |
|---|--:|--:|---|
| formal ceiling `F(obj_to_target<=0.071)` | — | **1.000** | expert spec |
| gpt-4o-mini (API) | **0.993** | **1.000** | nearly solves it unaided; gate perfects |
| gemma2:2b (1.6 GB, local) | 0.000 | **1.000** | over-constrains with the two noise signals; **pgraph pruning drops them → ceiling** |
| phi3:3.8b (local) | 0.000 | 0.000 | too weak — malformed HTL (`= TRUE`, concatenated formulas) the repair can't rescue |

The gemma rescue, on real data: `F(near_object>=0.5 AND grasp_success>=0.8 AND in_place>=0.9 AND obj_to_target<=0.1)`
(F1 0 — the dead signals force it false) → the `hymeko_pgraph` SSG pruning + calibration →
`F(in_place >= 0.6 AND obj_to_target <= 0.071)` → **F1 1.0**, matching the strong model and the expert.

## Honest findings

1. **The stack works end-to-end on real coffee-push.** A local 1.6 GB model's dead-on-arrival spec (0.0) is turned
   into the expert ceiling (1.0) by the augmenter (repair) + arbiter (calibration + P-graph pruning of the real
   noise signals). This is the synthetic result, reproduced on real MetaWorld data.
2. **A strong model nearly solves it unaided** (gpt-4o-mini raw 0.993) — coffee-push success is a simple
   `F(obj_to_target ≤ ε)`, so the augmentation head-room over a capable model is small here (gate 0.993→1.0).
3. **Prompt-sensitivity is real and load-bearing for small models.** With the first prompt, gemma *truncated* and
   wrote bare signal names as booleans (`F(near_object AND grasp_success AND …)`) → parse-rate 0 → the gate had
   nothing to work with. A tighter prompt (“every predicate MUST be SIGNAL CMP NUMBER; never a bare name”) + more
   tokens got it to a *repairable/prunable* proposal. The gate can only refine a parse-valid candidate.
4. **phi3:3.8b is below the bar** on this prompt — it emits `= TRUE`/concatenated formulas the repair can't fix.

## Non-claims

- **Not** "HyMeKo beats LLMs" — the strong model nearly solves it raw; the augmentation's value is turning a *weak
  local* model's noise-spec into the ceiling (and only when it emits a parseable candidate).
- Single task (coffee-push), 200 rollouts, ~70% positive (mild imbalance); one prompt family per model.
- No RL — the arbitrated spec is graded against native success; it is not yet *driving* a controller (that is the
  remaining loop-closure: arbitrated spec → reward/monitor → CIP-RL-LiNGAM execution).

## Changed / new files

`hymeko_rl/eval/spec_bench/metaworld_rollouts.py` (new — roll/save/load/run_bench) ·
`hymeko_rl/tests/test_metaworld_rollouts.py` · `reports/figures/2026_07_13_coffee_push/coffee_push_rollouts.json`
(real traces). Rollouts on kato15; pipeline + `hymeko_pgraph` on the Mac. Tests green; ruff + mypy clean.
**CORE.YAML untouched; no new deps.**

## Next

- **Close the loop**: turn the arbitrated success spec into a monitor/reward and feed CIP-RL-LiNGAM (the causal
  audit + HSiKAN mechanism), so the spec *drives* execution, not just grades it.
- **Small-model robustness**: a proposal normaliser for bare-signal booleans + a token-budget floor, so the gate is
  reached more reliably from a weak local model.
