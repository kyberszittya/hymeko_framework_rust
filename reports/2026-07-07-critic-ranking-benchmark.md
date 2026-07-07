# Critic ranking / repair benchmark — critic-only, no actor training

**Date:** 2026-07-07 · Git SHA `4320202` (dirty). Non-core. Seed 1, CPU. **No actor was trained.** Guards active
throughout: provenance PASS (DAgger = frozen selected md5 `edf4fe81…`), tensor-contract PASS (replay schema =
canonical). Replay = 22 427 pure-DAgger transitions.

## Headline

Built a critic-only ranking/repair benchmark (8 policy/action classes, 5 diagnostics, 7-criterion acceptance) and
trained the twin critics **alone** under four losses. **A (baseline), B (behavior-support), and C (CQL) all PASS;
E (expectile) FAILS.** But the pass/fail booleans hide the real signal — **the margin by which the critic
separates the exploit is the discriminator, and it grows A ≪ B < C**:

| variant | Q(DAgger)−Q(exploit) on DAgger states | Q(random) vs Q(DAgger) | verdict |
|---|---:|---|---|
| A baseline | **+0.05** (knife-edge) | −13.1 vs −10.9 (−2.3) | PASS (fragile) |
| B behavior-support | +5.83 | −20.2 vs −11.1 (−9.1) | PASS |
| **C CQL** | **+11.57** | **−29.1 vs −12.3 (−16.8)** | **PASS (most robust)** |
| E expectile (τ=0.3) | −0.49 (inverted) | −12.1 vs −11.7 (−0.4) | FAIL |

Figure: `reports/figures/critic_benchmark/critic_ranking.png`. Data: `experiments/v2_critic_benchmark/results.json`.

## Class metrics (critic-independent, monitor + MC return)

| class | MC return | reward | ft_dom | mon_pass | mon_score | violation |
|---|---:|---:|---:|---:|---:|---|
| scripted_v2b | −60.0 | −106.9 | 0.75 | 0.50 | 0.273 | fingertips_never_approached |
| mlp_dagger_selected | **−57.7** | −108.3 | 0.75 | 0.417 | **0.294** | no_right_fingertip_contact |
| mlp_bc0 | −75.5 | −190.0 | 0.333 | 0.25 | 0.136 | coin_pushed_away_during_approach |
| body_shove_exploit | −64.5 | −122.3 | 0.083 | 0.083 | −0.209 | fingertips_never_approached |
| one_fingertip | −118.1 | −335.9 | 0.0 | 0.0 | −0.274 | no_left_fingertip_contact |
| failed_ctde_td3bc | −127.1 | −369.1 | 0.0 | 0.0 | −0.41 | fingertips_never_approached |

MC return and monitor agree on the ordering (DAgger best). This is the external ground truth the critic must match.

## The five diagnostics (per variant)

- **1. Q-vs-MC calibration** (on-policy, across classes): Spearman(Q, MC) = A/B 0.371, **C/E 0.543** (all positive).
- **2. Q-vs-monitor calibration**: Spearman(Q, monitor) = A/B 0.657, **C/E 0.771** (all positive). Counterfactual
  `Q(DAgger) > Q(exploit)` and `> Q(one_fingertip)`: **true for A/B/C, false for E** (E ranks exploit top).
- **3. OOD overestimation**: Q(random) below Q(DAgger) for A/B/C (decisively so for C: −29 vs −12); for E only
  marginally (−12.1 vs −11.7).
- **4. Action-perturbation**: Q decreases monotonically away from the DAgger action for A/B/C; **E INCREASES**
  with perturbation (−11.3 → −10.5) — the opposite of what a sound critic does → fail.
- **5. Phase-wise**: A/B/C rank DAgger > exploit in every observed phase; **E mis-ranks APPROACH** (and blows
  DELIVERY Q positive, +16.8 — unstable). **PUSH never occurred** in the DAgger counterfactual rollout — the
  DAgger policy rarely achieves simultaneous two-fingertip push (consistent with its dominant `no_right_fingertip_contact`
  monitor violation); phases seen were APPROACH / CONTACT / DELIVERY.

## The key insight — static pass is necessary, not sufficient

**Baseline A passes the benchmark as literally specified, yet the guarded actor smoke collapsed
(2026-07-07-guarded-rl-sanity).** These are consistent once you read the margin: baseline A separates the exploit
by **+0.05** — a knife-edge. This also **reconciles the apparent contradiction** with the guarded step-2, where the
baseline critic ranked `exploit −5.70 > dagger −6.55` (margin −0.85): the baseline dagger-vs-exploit ranking sits
within ≈±1 of zero and **flips between runs** with replay composition and seed. It is not a stable property.

So the benchmark's real discriminating power is **margin + OOD suppression**, not the boolean. As the actor drifts
off the DAgger distribution during training, a knife-edge critic flips and the exploit wins the gradient — exactly
the collapse mechanism. **CQL's +11.6 margin and its −16.8 random-action suppression are the properties most likely
to survive actor drift.** CQL is the recommended repair.

**Recommended benchmark tightening** (follow-up, not applied here to preserve the specified criteria): add a
*margin* criterion — `Q(DAgger) − Q(exploit) ≥ m` and `Q(random) ≤ Q(DAgger) − m` for a threshold `m` (e.g. 2–3) —
so the benchmark rejects the fragile baseline and admits only decisively-conservative critics.

## UPDATE — three-tier margin gate (added per directive)

The acceptance gate is now margin-aware: `classify_critic` → **FAIL / WEAK_PASS / STRONG_PASS**. STRONG_PASS =
basic ranking holds AND `Q(DAgger)−Q(exploit) ≥ 3.0` AND `Q(DAgger)−Q(one_fingertip) ≥ 3.0` AND
`Q(random) ≤ Q(DAgger) − 5.0` AND both Spearmans > 0 AND perturbation-safe AND both guards. WEAK_PASS = ranks
right but fails a margin (fragile, **not actor-safe**). Re-classifying the cached results
(`experiments/v2_critic_benchmark/classification.json`):

| variant | tier | Q(dag)−Q(exploit) | Q(dag)−Q(1f) | Q(dag)−Q(random) | reasons |
|---|---|---:|---:|---:|---|
| A baseline | **WEAK_PASS** | 0.05 | 1.28 | 2.27 | all margins < threshold — knife-edge, not actor-safe |
| B behav-support | **WEAK_PASS** | 5.83 | 2.88 | 9.04 | one-finger margin 2.88 < 3.0 — backup |
| **C CQL** | **STRONG_PASS** | **11.57** | **7.31** | **16.77** | — (first actor-safe critic) |
| E expectile | **FAIL** | −0.49 | 1.57 | 0.39 | exploit ≥ dagger; perturbation rewards drift |

This matches the intended reading exactly: baseline is WEAK (not actor-safe), behavior-support is a backup, **CQL
is the first STRONG_PASS / actor-safe critic**, expectile fails. Only a STRONG_PASS critic may seed an actor smoke.

## E (expectile) is rejected

Expectile τ=0.3 was too aggressive: it inverted the APPROACH ranking, produced a non-monotone (increasing)
perturbation response, and pushed DELIVERY Q positive. A milder τ or a proper IQL with a separate value network is
the correct form; the cheap asymmetric-Bellman shortcut is not.

## Files

| file | LOC | note |
|---|---:|---|
| `hymeko_rl/eval/critic_benchmark.py` | 183 | rank corrs, phase labels, 5 diagnostics, 7-criterion acceptance |
| `hymeko_rl/train/critic_repair.py` | 158 | critic-only trainer + A/B/C/E loss Strategies |
| `hymeko_rl/tests/test_critic_benchmark.py` | 161 | 13 unit tests |
| `scratchpad/critic_benchmark_run.py` | — | run harness (8 classes, 4 variants, guards) |

**CORE.YAML:** none. No new dependencies (rank correlations implemented without scipy).

## Tests

- **Unit:** `pytest hymeko_rl/tests/test_critic_benchmark.py` → **13 passed**. Rank correlations (monotone /
  anti / degenerate / ties), phase labelling, all diagnostics (aligned vs inverted), acceptance gate, the four
  critic losses (finite scalar), unknown-variant rejection, and `train_critic_only` keeps the actor frozen.
- **Static:** `ruff` clean.
- **Guards:** provenance + tensor-contract PASS (recorded in results.json).

## What this licenses (and what it does not)

Per directive, a critic passing the benchmark **unlocks another one-seed actor smoke** — A/B/C qualify. I did
**not** run it: this task was to build the benchmark, and the margin analysis says the right next actor smoke uses
the **CQL critic** (decisive margin), not the baseline (knife-edge) — a deliberate choice worth your authorization.
No SAC, no residual, no multi-seed. When a gated actor smoke is run, it must init the critic from a CQL-repaired
critic and clear the full 7-criterion actor acceptance under the complete safety stack.

**Status:** critic ranking benchmark built + passing for A/B/C; CQL is the strongest repair; the actor collapse is
now explained as a *dynamic* margin-flip, not a static ranking bug. Actor still frozen.
