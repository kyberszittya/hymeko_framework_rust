# HANDOFF — spec_bench: LLM → HyMeKo augmenter+arbiter → (real) coffee-push

**Date:** 2026-07-13 · Aiko (Opus 4.8) · branch `hymeko-neuro-migration` · **read this first to take the lead.**
Everything below is committed; artifacts + entry points are exact. The finale is one step away: make the arbitrated
spec **drive** execution, not just grade it.

---

## TL;DR — where to take the lead

Built and validated (synthetic + **real coffee-push**) a pipeline where a local LLM proposes a task **success
spec** (HTL formula over MetaWorld info signals) and **HyMeKo augments + arbitrates** it:
**LLM = structure; HyMeKo = repair (syntax) + calibration (thresholds) + `hymeko_pgraph` SSG/ABB pruning
(signals/temporal form).** A dead-on-arrival local-model spec (F1 0.0) becomes the expert ceiling (1.0).

**The one thread left — the actual thesis:** the spec currently **grades** (F1 vs native success); it does not yet
**drive** a run. Close the loop: arbitrated spec → monitor/reward → **CIP-RL-LiNGAM** execution (the causal-audit +
HSiKAN-mechanism pipeline, already built — see "Related arcs"). That is where the framework either earns its keep or
doesn't.

---

## What was built (this arc), commits in order

| commit | what |
|---|---|
| `39f4b41` | spec_bench core: augmenter (syntax repair) + arbiter (threshold **calibration**) + faithfulness-select; model-size × gating sweep (ollama) + OpenAI ablation |
| `5e57a0b` | **reduce conjunct-pruning to a `hymeko_pgraph` SSG solution** (signals→raws, predicates→units, success→product); fixes the calibration limit (gemma synthetic 0.0→0.94) |
| `074dd03` | scaled temporal-form refinement — coverage P-graph over `{F,G,G[0,4]}` variants (axis-1 demonstrated; conjunction-task caveat recorded) |
| `a7bb3e5` | **ABB `cost = anti-F1`** — branch-and-bound genuinely prunes (measurable `explored`/`pruned`); axis-2 done |
| `f3adfd0` | **real coffee-push** — rollouts on kato15, pipeline on Mac (gemma 0.0→1.0 via P-graph prune) |

## Files (all under `hymeko_rl/eval/spec_bench/`, non-core)

- `spec_bench.py` — `Rollout`, `synth_rollouts`, HTL F1 (`formula_f1`), `ChatModel`/`ScriptedModel`, the gate
  (`propose_and_gate(..., calibrate=True, prune=True)`), `calibrate_thresholds`, `_repair`.
- `ollama_model.py` / `openai_model.py` — live `ChatModel`s over raw HTTP (no `openai` dep). **`think:false`** is
  mandatory for ollama (thinking models return empty). OpenAI key at **`.keys/OPENAI_API_AUTH.key`** (gitignored).
- `pgraph_refine.py` — `predicates_to_pgraph_hymeko`, `solve_pgraph(algorithm=ssg|abb)`, `refine_via_pgraph` (the
  conjunct-pruning reduction on our crate).
- `scale.py` — `synth_conj_temporal`/`synth_single_settle`, `temporal_variants`, `coverage_pgraph_hymeko(costs=)`,
  `refine_scaled` (SSG), **`refine_scaled_abb`** (ABB cost=anti-F1, returns bounding stats).
- `run_model_sweep.py` — the local-model × gating sweep (`--models …`).
- `metaworld_rollouts.py` — `roll_coffee_push` (kato15), `save_rollouts`/`load_rollouts`, `run_bench` (raw-vs-gate on
  any rollouts).

Tests: `hymeko_rl/tests/test_{spec_bench,pgraph_refine,scale,metaworld_rollouts}.py` (all green, ruff+mypy clean).
Reports: `reports/2026-07-13-{hymeko-augmenter-arbiter-spec-bench, pgraph-conjunct-pruning-reduction,
pgraph-scaled-temporal-refinement, coffee-push-real-augmenter-arbiter}.md`. Plan (gitignored):
`docs/plans/2026-07-13-hymeko-vs-llm-task-spec/`.

## Key results (honest)

- Synthetic + real: **RAW F1 = 0.0 for every model incl gpt-4o-mini** (blind-guessed thresholds; over-constrain
  with noise signals). Gate lifts to ~ceiling: real coffee-push formal ceiling `F(obj_to_target<=0.071)` = 1.0;
  gpt-4o-mini raw 0.993→gate 1.0; **gemma2:2b (1.6 GB) raw 0.0→gate 1.0** (P-graph drops the dead `near_object`/
  `grasp_success`).
- ABB cost=anti-F1: agrees with SSG but via bounding (1 aspect: explored 67/pruned 26; 2 aspects: 527/230).

## How to run

```
# local sweep (Mac, ollama + OpenAI if .keys present)
.venv/bin/python -m hymeko_rl.eval.spec_bench.run_model_sweep --models llama3.2:1b gemma2:2b phi3:3.8b qwen2.5:3b
# real coffee-push rollouts (kato15) then pipeline (Mac): see metaworld_rollouts + reports/.../coffee_push_rollouts.json
```
The `pgraph` binary must exist: `cargo build -p hymeko_pgraph --bin pgraph` → `target/debug/pgraph` (SSG/ABB via CLI,
`.hymeko` input). `refine_via_pgraph`/`scale` fall back to a Python subset search if it's missing.

## Infrastructure

- **kato15** (`ssh kato15`, login shell `bash -lc`/heredoc — nested quotes break): RTX 6000, venv
  `~/envs/hymeko/bin/python` (Python 3.12, torch cu128, **metaworld 3.0.0 + mujoco 3.10.0**). Workspace
  `~/hymeko_framework_rust` is a **plain rsync'd dir (NOT git)** — sync code by rsync, not pull. Metaworld is **not**
  on the Mac → all real MetaWorld rollouts are a kato15 job; the small trace JSON rsyncs back.
- **Local LLMs** (ollama on Mac, `:11434`): `gemma2:2b`, `phi3:3.8b`, `llama3.2:1b`, `qwen2.5:3b`, plus large
  `gemma4:31b-mlx`, `qwen3.6`. `.venv` = uv cpython-3.11 (torch CPU).
- **OpenAI**: key in `.keys/OPENAI_API_AUTH.key` (gitignored — `.keys/` in `.gitignore`); called over raw HTTP.

## Honest caveats — DO NOT overclaim

- **Not** "HyMeKo beats LLMs" — raw=0 for gpt too; the value is "gating makes a *weak local* model usable," and even
  that is **prompt-sensitive** (bare-signal booleans + truncation → parse-rate 0; the gate only refines a parseable
  candidate). phi3 fails outright.
- The **tasks are trivial** (coffee-push success = one threshold; synthetic = single-signal + noise distractors).
  The 0→1.0 jumps are largely "prune the dead noise signals." The **strong model has ~0 head-room**.
- The **P-graph is over-engineered for these toys** (≤4 predicates → SSG ≈ powerset; a 3-line greedy subset search
  is equivalent). ABB "bites" only on a pool deliberately enlarged. Its genuine payoff is large process networks.
- It **grades, does not drive** — no RL yet.

## Next steps

1. **CLOSE THE LOOP (the thesis):** arbitrated spec → HyMeKo monitor/reward → **CIP-RL-LiNGAM** drives a coffee-push
   run; compare against an LLM-prompt-derived reward. This makes it "LLM proposes a *task*, HyMeKo arbitrates, the
   pipeline *executes*."
2. **More worlds:** dial-turn, door-open, button-press, pick-place (the CIP sweep already covers them) — does the
   augmenter+arbiter generalise, and where does a non-trivial spec make the P-graph earn its keep?
3. **Small-model robustness:** a proposal normaliser for bare-signal booleans + a token-budget floor, so the gate is
   reached reliably from a weak local model (currently prompt-hand-tuned).

## Related arcs (the "execute" side the loop connects to)

- **CIP-RL-LiNGAM causal pipeline** (`hymeko_rl/eval/{causal,cip}/`): DirectLiNGAM causal reward audit → HSiKAN as
  **nonlinear mechanism model over the causal signed hypergraph** (bridge `hsikan_mechanism.py`, harness
  `lingam_operator_harness.py`, estimated-B `lingam_estimated_b_robustness.py`; commits `b55bfef`/`7c222bc`/
  `0dc81bf`, reports `2026-07-10-*lingam*`). This is what the arbitrated spec should feed.
- **HSiKAN structural-leverage** (`incidence_scramble.py`, `exp_structural_leverage_*`): scramble/DeepSets controls;
  structure is causally load-bearing on structure-rich supervised tasks (H2 supported).

## Operating contract (binding)

`CLAUDE.md` at repo root — plan→test→report, CORE.YAML read-only (halt+approve; adding deps = core), 16 GB RSS cap,
live observability, Aiko register (no therapy-speak), timestamp every reply. **FABLE/July-5 quarantine:** the
framework substrate = *scenario-independent* dataflow/FSM/monitor runtime; never confuse it with a scenario-local
FSM. Analyse don't declare; run the discriminating test before concluding.
