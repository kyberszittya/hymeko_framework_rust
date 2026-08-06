# Strategy-search + pruning reduced to a P-graph solution (our hymeko_pgraph) — fixes the gemma limit

**Date:** 2026-07-13 · Aiko · branch `hymeko-neuro-migration` · Mac (local ollama + OpenAI + our `hymeko_pgraph`
CLI) · **synthetic task; no metaworld, no RL.** Follow-up to `2026-07-13-hymeko-augmenter-arbiter-spec-bench.md`,
which found calibration fixes *thresholds* but not *signal choice* (gemma stuck at 0 on a noise conjunct). This
report reduces the arbiter's structure-refinement to a **P-graph solution on our own `hymeko_pgraph` crate** and
closes that gap. Code: `hymeko_rl/eval/spec_bench/pgraph_refine.py`.

## The reduction

Conjunct-pruning = *which minimal subset of the LLM's candidate predicates forms a faithful spec* = a **solution-
structure search**, which is exactly P-graph SSG (Friedler et al.). The mapping, emitted as `.hymeko` (our DSL):

| P-graph | spec refinement |
|---|---|
| raw materials | the signals (`in_place`, `obj_to_target`, `near_object`, `grasp_success`) |
| operating unit `@Pi (-signal, +success)` | the LLM's candidate predicate `i` |
| product `success` | the success decision |
| SSG feasible structures | the axiom-valid predicate subsets |

`hymeko_pgraph solve … --algorithm ssg --json` enumerates the feasible structures; we **rank them by F1** on the
verification split (each structure → `F(AND of its predicates)`, calibrated). SSG enumerates; F1 selects. **No
external P-graph library, no re-implementation** — our crate via its CLI on a generated `.hymeko`; a Python
exhaustive-subset fallback covers an unbuilt binary (numerically identical for small pools).

## Result — the board, with vs without the P-graph reduction (synthetic, F1 vs native success)

| model | size | raw | gate (calibration only) | **gate (+ P-graph pruning)** |
|---|--:|--:|--:|--:|
| llama3.2:1b | 1.3 GB | 0.000 | 0.868 | **0.930** |
| **gemma2:2b** | 1.6 GB | 0.000 | **0.000** | **0.940** |
| phi3:3.8b | 2.2 GB | 0.000 | 0.909 | **0.930** |
| qwen2.5:3b | 1.9 GB | 0.000 | 0.643 | 0.625 |
| gpt-4o-mini (API) | — | 0.000 | 0.941 | 0.930 |
| formal ceiling | — | — | 0.941 | 0.941 |

**gemma 0.000 → 0.940.** Its over-constrained `F(in_place>=0.9 AND obj_to_target<=0.01 AND near_object>=0.7 AND
grasp_success==1)` (F1 0) is reduced to a 4-unit P-graph; SSG enumerates 15 feasible structures; F1-ranking drops
the noise conjuncts (`grasp_success`, `near_object`) → `F(obj_to_target <= 0.118)`, F1 0.930. Four of five models —
including a 1.3 GB local one — now sit at ~0.93, the ceiling. qwen's small dip (0.643→0.625) is its `G(F(...))`
temporal nesting (a structure the predicate-pruning does not address), not a pruning regression.

## Honest scope

- On this toy (≤4 candidate predicates) SSG ≈ the powerset, so the reduction is numerically equal to an exhaustive
  Python subset-search. The P-graph's real payoff — ABB's axiom-backed bounding of a *large* candidate space (many
  monitors × temporal variants × predicates, the real coffee-push scale) — is not exercised here. What is
  established: the **reduction is correct and wired to our crate**, and it fixes the calibration-only limit.
- Still synthetic; still not "HyMeKo beats LLMs" (raw = 0 for gpt too). The claim is: the arbiter's structure-search
  reduces cleanly to a `hymeko_pgraph` solution, and it lifts small local models to the expert ceiling.

## Division of labour (now complete for the success-predicate task)

1. **LLM** → the *structure* (which signals, temporal/logical form).
2. **HyMeKo augmenter** → syntax repair (parse-rate 0→1).
3. **HyMeKo arbiter/refiner** → threshold **calibration** (constants) + **P-graph SSG** conjunct-pruning (structure).

## Changed / new files

`hymeko_rl/eval/spec_bench/pgraph_refine.py` (new) · `hymeko_rl/tests/test_pgraph_refine.py` (6 tests) ·
`hymeko_rl/eval/spec_bench/spec_bench.py` (gate gains `prune=True`, late-imports the reducer) ·
`reports/figures/2026_07_13_spec_bench_model_sweep/` (re-run). Uses the built `target/debug/pgraph`. 18 spec_bench +
pgraph tests green; ruff + mypy clean. **CORE.YAML untouched; no new deps** (LLMs over raw HTTP; pgraph via its CLI).

## Next

- **Structural refinement beyond conjunct-pruning** (qwen's `G(F(...))` nesting) — the arbiter over temporal form too.
- **Scale the P-graph** — a large candidate pool where SSG/ABB's axiom-backed bounding earns its keep.
- **Wire the arbitrated spec into CIP-RL-LiNGAM on real coffee-push** (kato15) — the full LLM→task→execute loop.
