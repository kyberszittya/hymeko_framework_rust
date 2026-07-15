---
title: SignedHypergraphLiNGAM causal diagnosis of the FF-DAgger / LSTM / SAC ensemble (role 1)
date: 2026-07-16
scope: causal attribution of the pick-place policies + the metric confound it surfaced
status: experiment (diagnostic) — proposals to test, per the CIP doctrine
core_touched: none
---

# SignedHypergraphLiNGAM role-1 causal diagnosis

**Goal.** Fit the framework's DirectLiNGAM → signed causal hypergraph (`hymeko_rl.eval.causal`, same pipeline as
`cip_lingam_demo`'s coin mode) over rollouts of FF-DAgger / LSTM / residual-SAC, to causally attribute the walls and
test the F-PP-009 mechanism (does acting far from the base *cause* loss of grasp/success?).

## What the diagnostic surfaced first: the metric confound (F-PP-013)

Including `ever-grasped` in the rollout variables exposed that the arc's success metric is inflated:

```text
                     reached (placed_stable, the arc metric)   ever-grasped   SKILL = placed_stable ∧ ever-grasped
FF_DAgger (DAgger'd)          0.85                                 0.56             0.42   ← real skill
LSTM (BC-only)               0.65                                  0.00             0.00
SAC_residual (collapsed)      0.42                                 0.12             0.00
IDLE (zero action)            0.458                                0.00             0.00   ← the floor
```

`placed_stable` fires for a box within `place_radius` (6 cm) of the target at rest — **no grasp/lift required** — and
the object spawns that close ~46 % of the time, so **idle scores 0.458**. On the **skill-isolating** metric
(`placed_stable ∧ ever-grasped`): the DAgger'd base has **0.42** real skill; **the LSTM BC clone and the collapsed
SAC have 0.00**. The cached "LSTM 0.917 reliability win" and the "SAC 0.458" were **inflation** — the LSTM never
grasps (BC covariate shift, the known "BC rolls out 0 %" failure), and SAC ≈ the idle floor.

## Per-policy signed causal hypergraphs (DirectLiNGAM → A⁺/A⁻)

```text
FF_DAgger  kept={approach,grasp,hold,lift,target,success}  |A+|=3.80 |A-|=2.78
   grasp -> target +1.49 · hold -> grasp +0.77 · success -> target -0.97   (a REAL grasp→place chain)
LSTM       kept={approach,lift,target,deviation}           |A+|=0.79 |A-|=1.24
   grasp/hold/success DROPPED (constant 0 — never grasps); only approach→lift, deviation→approach survive
SAC        kept={approach,grasp,hold,lift,target,deviation} |A+|=2.85 |A-|=1.82
   grasp -> hold +0.91 · hold -> lift +1.48; success DROPPED (constant 0 — grasps 0.12 but never completes)
```

The signed causal hypergraph **confirms only the DAgger'd base has a real grasp→success chain**. The BC clone and
collapsed SAC have **degenerate DAGs** — their grasp/success nodes are constant-zero (dropped by LiNGAM), i.e. the
causal machinery formally records "no path to real success" for them.

## The F-PP-009 causal probe — CONFOUNDED (honest)

Pooled fit (all policies), the `deviation → {grasp, lift, success}` edge:

```text
deviation -> grasp   = +0.000     deviation -> lift = +0.000     deviation -> success = +0.000
(reverse edge present: grasp -> deviation = -0.592; causal order puts grasp before deviation)
```

The pooled probe does **not** cleanly confirm "deviation causes grasp loss." It is **confounded by policy
heterogeneity**: FF has deviation=0/grasp=0.56, LSTM/SAC have deviation>0/grasp≈0, so LiNGAM sees a
grasp↔deviation *correlation* and (given its ordering) assigns `grasp → deviation`, not the causal direction we
asked about. A clean causal test of F-PP-009 needs **within-policy** variation of deviation (e.g. SAC checkpoints at
several collapse levels), not cross-policy aggregates. The direct evidence for F-PP-009 (the critic Q-ranking:
Q(collapsed actor)=−0.687 ≫ Q(teacher)=−1.164) stands independently and is not weakened by this — the causal
*re-derivation* over rollout aggregates simply cannot isolate it here.

## Verdict on the SignedHypergraphLiNGAM idea

- **As a diagnostic (role 1): it works and earned its keep** — it surfaced the metric confound (F-PP-013) and its
  signed hypergraph formally shows only the DAgger'd base has a real grasp→success chain. This is the framework's
  "propose structure over real rollouts; ablations decide" pattern applied to control.
- **For a causal *gate/ensemble* (roles 2/3): the finding undercuts it** — on the skill metric only FF-DAgger has
  real skill (0.42); the LSTM and SAC are ~0. A causal gate over {FF, LSTM, SAC} would just select FF. Mixing them
  buys nothing until there is a *second* policy with real, complementary skill.
- **For the F-PP-009 causal claim**: cross-policy LiNGAM is the wrong instrument (confounded); within-policy
  deviation variation is required — a clean, cheap follow-up if wanted.

## Provenance
- Branch `integration/fanuc-pick-place-canonical`. Experiment `scratchpad/signed_hypergraph_lingam_diag.py`
  (reuses `hymeko_rl.eval.causal` + `lingam_to_signed_adjacency`; LSTM 40 demos/300 ep; SAC 20k AUTO-alpha;
  N=48/policy @ seed0=20000). Skill metric = `placed_stable ∧ ever-grasped`. No CORE. No kato15.
