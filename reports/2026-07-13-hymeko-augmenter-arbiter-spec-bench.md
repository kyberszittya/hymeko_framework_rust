# HyMeKo as augmenter + arbiter over LLM-proposed task specs — spec_bench (synthetic Stage 0/1)

**Date:** 2026-07-13 · Aiko · branch `hymeko-neuro-migration` · Mac (local ollama + OpenAI HTTP) · **synthetic task
only; no metaworld, no RL.** Tests whether HyMeKo formalization + monitoring can *augment* an LLM at producing a
task's success spec, on a controlled synthetic task, across a model-size ladder + a strong API reference.
Plan: `docs/plans/2026-07-13-hymeko-vs-llm-task-spec/`. Code: `hymeko_rl/eval/spec_bench/`.

> **Scope/claim discipline.** This is *not* "HyMeKo beats LLMs." It is "HyMeKo *gating* makes LLM task-specs
> usable" — and even a strong model needs it. Synthetic task; the mechanism is proven here, the real target
> (coffee-push, CIP-RL-LiNGAM) is the next stage.

## Setup

Each arm emits an **HTL success formula** over signals `{near_object, grasp_success, in_place, obj_to_target}`;
the (unified) HTL engine grades it by F1 of `robustness>0` vs native success on balanced synthetic rollouts (ground
truth: `F(in_place >= 0.9)`; `grasp_success` is a pure-noise distractor; `near_object` a correlated distractor).

- **raw arm:** the LLM's first parse-valid formula (unaided).
- **gate arm (HyMeKo augmenter + arbiter):** (1) *augment* — syntax **repair** (bool-op case, `=`→`==`, stray
  `[ ]`→`( )`) + parse-gate error-loop; (2) *arbiter/refine* — **threshold calibration** (keep the LLM's structure,
  fit its numeric constants to a held-out verification split) + faithfulness-select.
- **formal ceiling:** the expert HTL. **strength ablation:** gpt-4o-mini (OpenAI, HTTP).

## Result (F1 vs native success; test split)

| model | size | raw F1 | **gate F1** | Δ (gate−raw) | raw parse-rate | rtt ms | tok/s |
|---|--:|--:|--:|--:|--:|--:|--:|
| llama3.2:1b | 1.3 GB | 0.000 | **0.868** | +0.868 | 0.00 | 398 | 164 |
| gemma2:2b | 1.6 GB | 0.000 | 0.000 | 0.000 | 1.00 | 517 | 119 |
| phi3:3.8b | 2.2 GB | 0.000 | **0.909** | +0.909 | 0.00 | 451 | 94 |
| qwen2.5:3b | 1.9 GB | 0.000 | 0.643 | +0.643 | 0.00 | 288 | 109 |
| **gpt-4o-mini** (API) | — | 0.000 | **0.941** | +0.941 | 1.00 | 914 | 15 |
| formal ceiling | — | — | 0.941 | — | — | — | — |

## Findings

1. **Raw is unusable for every model — including the strong one.** All raw F1 = 0.000. The failure is not
   capability and not harness plumbing: the LLM gets the **structure** right (correct signals + `F(...)` + logic —
   gpt-4o-mini proposed `F(in_place >= 1 AND obj_to_target <= 0)`) but **blind-guesses the numeric thresholds**,
   because no model is shown the data distribution.
2. **The augmenter+arbiter is decisively load-bearing.** The gate lifts a **1.3 GB local llama to 0.868**, phi3 to
   0.909, gpt-4o-mini to the **0.941 ceiling** — from 0.0. The mechanism: **LLM proposes structure; HyMeKo repairs
   syntax and calibrates the constants against the rollouts.** Calibrating a mis-guessed structure recovers the
   ceiling exactly (`F(in_place>=1 AND obj_to_target<=0)` → `F(in_place>=0.9 AND obj_to_target<=0.59)`, F1 0→0.94).
3. **Cheap-local payoff.** A 1.3 GB local model (398 ms, 164 tok/s) + HyMeKo reaches 0.868, and phi3 (2.2 GB) 0.909
   — approaching gpt-4o-mini's 0.941, which is remote, paid, and ~10× slower per token (15 tok/s). A small local
   model + HyMeKo ≈ a strong API model, for this task.
4. **Honest limit — calibration fixes numbers, not signal choice.** gemma2:2b stays at 0.0: its structure insists on
   the noise signal `grasp_success == 1`, and threshold calibration cannot remove a bad *conjunct*. This names the
   next refinement: the arbiter should also **prune** unfaithful conjuncts (structure refinement), not only calibrate
   thresholds.

## Where this sits in the architecture (the directive)

`spec_bench` is the **LLM → task-spec → HyMeKo-augmenter+arbiter** layer of the target loop
(*LLM produces tasks → CIP-RL-LiNGAM executes coffee-push → HyMeKo augments + arbitrates + refines*). This report
establishes the augmenter (repair) and the first arbiter/refiner (calibration) empirically. The remaining pieces:
conjunct-pruning refinement; extending from a *success predicate* to the full task (reward + monitors); and wiring
the arbitrated spec into the CIP-RL-LiNGAM pipeline on real coffee-push (kato15).

## Non-claims

- **Not** "HyMeKo beats LLMs" — raw is 0 for gpt-4o-mini too; the claim is that gating makes LLM output usable.
- **No** real-MetaWorld validation (synthetic task; the mechanism, not a coffee-push number).
- **No** RL; **no** general-LLM-capability claim (a single task, fixed prompts).
- The synthetic task (single-signal truth + a pure-noise distractor) is mildly adversarial by design — it is a
  mechanism probe, not a difficulty benchmark.

## Changed / new files

`hymeko_rl/eval/spec_bench/{__init__,spec_bench,ollama_model,openai_model,run_model_sweep}.py` (new) ·
`hymeko_rl/tests/test_spec_bench.py` (12 tests) · `docs/plans/2026-07-13-hymeko-vs-llm-task-spec/` (plan) ·
`reports/figures/2026_07_13_spec_bench_model_sweep/` (JSON + PNG). `.keys/` gitignored (secret protection).
**CORE.YAML: none. New deps: none** (both LLMs over raw HTTP). 12 tests green; ruff + mypy clean.

## Next

1. **Conjunct-pruning refinement** (the gemma limit) — arbiter drops unfaithful sub-predicates, not just calibrates.
2. **Full task spec** (reward + monitors, not only success) + wire into **CIP-RL-LiNGAM on real coffee-push** (kato15).
