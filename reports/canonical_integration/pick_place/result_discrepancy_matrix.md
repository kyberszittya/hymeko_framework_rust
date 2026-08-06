---
title: Pick-place result discrepancy matrix — 0.875 vs 0.167
date: 2026-07-16
scope: reconcile the earlier strong pick-place results with the recent weak TD3+BC number
status: reconciled (outcome B — different protocols) + one integration gap (outcome C)
---

# Pick-place result discrepancy matrix (0.875 vs 0.167)

## The two figures, exact provenance

| | earlier STRONG | recent WEAK |
|---|---|---|
| value | **0.875** | **0.167** |
| policy | FF **hybrid-DAgger** base (`hybrid_dagger_gif/.../hsikan_s0_best.pt`) | **TD3+BC** (`2026_07_13_02_55_fanuc_pick_td3bc_hsikan/.../s0.pt`) |
| training | imitation → DAgger (clones scripted v3 expert) | BC warm-start → TD3 refinement (BC anchor, best-ckpt-on-place) |
| metric | `place` = `info["reached"]` = **`placed_stable`** (settle, `require_settle=True`) | far-spawn **grasp∧place** (strict skill) |
| distribution | **full** spawn annulus (r∈[0.28,0.40]) | **far-spawn only** (spawn > place_radius+0.05 from target) |
| seeds | multi-seed: `2026_07_12_21_58` = [0.833, 0.875, 0.917] | single-seed s0 |
| source | `experiments/2026_07_06_19_03` / `2026_07_12_21_58/results.json` | measured 2026-07-16 (this session) |

## Every protocol axis (explicit)

| axis | earlier 0.875 | recent 0.167 | same? |
|---|---|---|---|
| **success metric** | `placed_stable` (box at target, at rest, released — NOT lift-requiring) | `placed_stable ∧ ever-grasped` (real pick-place) | **NO** |
| **distribution** | full (≈42% spawn already at target → free) | far-spawn only (removes the free episodes) | **NO** |
| **inflation floor** | ~0.458 idle floor included | floor removed | **NO** |
| environment | `fanuc_pick_env(expert_version=3, require_settle=True)` | same | yes |
| initial-state gen | `env.reset(seed)`, full annulus | same reset, far subset | metric-level diff |
| reward | procedural dense (train only) | n/a (eval) | yes |
| observation | 9×10 node features | same | yes |
| action space | 7 = 6 joint targets + grip | same | yes (see loader gap) |
| learner | DAgger (imitation) | TD3+BC (off-policy RL) | **different policy** |
| teacher | scripted v3 expert (DAgger relabel) | scripted demos (BC warm-start) | both use the scripted expert as teacher |
| scripted fallback | **NONE** (agent-verified: `build_scene`→`load_pick_policy` fails loud, no silent expert swap) | NONE | yes |
| checkpoint loader | `load_pick_actor` (raw FF) ✅ | **`load_pick_policy` CANNOT load a DeterministicActor → PickPolicyIncompatible** ❌ | **NO — integration gap** |
| normalization | none (raw obs) | none | yes |
| seed cohort | training-eval cohort (`_EVAL_SEED`) | seed0=20000, n=48 | **NO** |
| eval horizon | max_steps=1000 | same | yes |
| GUI routing | listed + loadable | **NOT listed / not loadable by the GUI** | **NO — integration gap** |

## Reconciliation — all key policies under ONE evaluator (seed0=20000, N=48)

| policy | reached FULL | reached FAR | grasp FAR | **grasp∧place FAR** |
|---|---|---|---|---|
| scripted expert | 0.938 | 0.917 | 1.000 | **0.917** |
| FF-DAgger base (the "0.875") | 0.854 | 0.500 | 1.000 | **0.500** |
| TD3+BC 07-13 s0 (the "0.167") | 0.667 | 0.167 | 1.000 | **0.167** |
| TD3+BC 07-06 s1 (its results.json said 0.75) | 0.417 | 0.000 | 1.000 | **0.000** |

**Reading:** under one evaluator the ranking is *consistent on every metric* — expert > DAgger base > TD3+BC-s0 > TD3+BC-06-s1. The "0.875" is `reached FULL` for the DAgger base; the "0.167" is `grasp∧place FAR` for TD3+BC-s0. They are **the same policies measured under different protocols**, not a regression. Note `TD3+BC 07-06 s1` reported **0.75** in its own `results.json` (a *third* protocol — training-eval cohort) yet is **0.417 full / 0.000 far** on this cohort: that number was cohort-specific and mostly the idle floor.

## Verdict (per §9)

- **Primary: OUTCOME B — compatible, different protocols.** 0.875 (DAgger, `reached`, full) and 0.167 (TD3+BC, `grasp∧place`, far-spawn) are not in conflict; on any single protocol the DAgger base beats TD3+BC. **0.167 does NOT supersede 0.875.** My earlier narrative headlined the strictest metric (0.167) against the loosest (0.875) — the inversion.
- **Secondary: OUTCOME C — one component not integrated.** The canonical loader `load_pick_policy` and the GUI cannot load a TD3+BC `DeterministicActor` checkpoint (fail-loud, no fallback). So the *learned-RL* frontier is real but **unreachable through the canonical path** — an assimilation gap to fix (see `missing_assimilation.md`).
- **NOT a regression, NOT stale-source, NOT invalid** (§9-A/D ruled out: no eval bug, no fallback leak, no wrong-checkpoint; the numbers reproduce under a clean evaluator).

## Canonical numbers to carry forward (this cohort)
- **Strongest deployable**: scripted v3 expert — 0.917 grasp∧place far / 0.938 reached full (but off-limits for the learned demo).
- **Strongest LEARNED (imitation)**: FF-DAgger base — 0.500 far / 0.854 full.
- **Strongest LEARNED RL**: TD3+BC 07-13 s0 — 0.167 far / 0.667 full (fragile; s1/s2 ≈ 0).
- **Negative control**: plain SAC — collapses to the idle floor (F-PP-009).
