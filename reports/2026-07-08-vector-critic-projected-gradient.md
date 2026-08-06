# Vector-critic + projected-gradient diagnostic — INCONCLUSIVE (measurement-limited); Step 6 NOT authorized

**Date:** 2026-07-08 · Git SHA `03b01c3` (dirty). Non-core. One seed, CPU. Steps 1–5 only (no actor trained).
The vector hypothesis is **not refuted** — this run could not fairly test it, for a concrete and important reason.

## What ran (per the branch spec)

- **Step 1** — `SearchObjective` (new, `hymeko_rl/train/search_objective.py`): per-step monitor-derived component
  signals (approach/contact/progress/delivery/antiexploit/body_progress) + the objective↔constraint split. Kept a
  **separate class** from the frozen `TaskMonitor` verifier, as required.
- **Step 2** — six component critics (`hymeko_rl/train/vector_critic.py`) trained on the DAgger dataset + a scalar
  `Q_total` (frozen STRONG_PASS CQL).
- **Step 3** — gradient-alignment cosines on 150 CONTACT states.
- **Step 4** — PCGrad-style constraint-projected direction `g_proj`.
- **Step 5** — MuJoCo one-step branch: DAgger vs `+ε∇Q_total` vs `+ε·g_proj` vs random.

6 unit tests pass (SearchObjective signals, cosine, projection); ruff clean; CORE.YAML untouched.

## Results

**Step 3 — cosines `cos(∇Q_total, ·)`:** contact **+0.12**, progress +0.05, −body_progress −0.04, delivery
**−0.11**, antiexploit **−0.13**. So the scalar gradient weakly *conflicts* with the delivery and anti-exploit
component gradients and weakly *aligns* with contact/progress — partial, weak support that one scalar does not serve
all components, but nothing decisive.

**Step 5 — one-step branch (both-contact rate / coin progress / arm-body / short-horizon monitor):**

| candidate | two-finger contact | coin progress | arm-body | short monitor_score |
|---|---:|---:|---:|---:|
| DAgger | 0.060 | −0.0168 | 0.007 | −0.314 |
| +ε∇Q_total | 0.073 | −0.0164 | 0.013 | −0.314 |
| +ε·g_proj | 0.013 | −0.0166 | 0.000 | −0.316 |
| random | 0.033 | −0.0165 | 0.027 | — |

`VECTOR_PROJECTED_PROMISING = False`; **`gate_step6_authorized = False`.** The projected direction did *not*
preserve/raise contact (0.013 < 0.060), and the short-horizon monitor is **flat across all candidates** (≈ −0.314).

## Why this is INCONCLUSIVE, not a negative

Three measurement problems undercut any conclusion — and they share one root:

1. **The vector critics are poorly fit.** Their mean Q on non-negative-return components came out *negative*
   (progress −0.32, delivery −0.12) — a critic predicting negative for a provably ≥0 return is not fit.
2. **The one-step contact signal is too sparse** (rates 0.01–0.07 = a handful of the 150 states) → high-variance,
   and it even *flipped sign* vs the ε-sweep probe (there +∇Q_total lowered contact; here it raised it) — noise.
3. **The short-horizon (k=15) monitor is insensitive** to an ε=0.03 one-step perturbation → flat, non-discriminating.

**Root cause (measured/inferred):** the vector critics were trained on the **near-deterministic DAgger replay with
no action diversity** and a plain Bellman loss (no OOD-action term). With every stored action ≈ π_DAgger(s), a
critic has *no information about how Q varies with the action* — so ∇ₐQ (scalar OR vector) is pure extrapolation.
This is the same upstream obstruction that limited the scalar-critic line: **reliable action-gradients require
action diversity in the replay, which this setup does not provide.**

## Decision

**Do not run Step 6 (the vector-projected actor smoke).** The gate is False, and — more importantly — it would be
built on unreliable gradients. The vector-valued hypothesis remains open; it was **not fairly tested** here.

Two honest paths (your call):

- **(A) Fix the diagnostic and re-decide the vector hypothesis** — the fair test. Needs: (i) **action-diverse
  replay** (exploration noise around DAgger, or an explicit perturbed-action dataset) so the critics can learn the
  Q-vs-action *shape*; (ii) **better-fit component critics** — since the actor is frozen, fit them by supervised
  regression to **Monte-Carlo component returns** (exact, no bootstrapping) plus an OOD-action term for off-manifold
  shape; (iii) a **sensitive Step-5 metric** — larger ε, a component-scored horizon, and filtering to genuine
  two-finger states. Only then do the cosines / projection / branch mean anything.
- **(B) Accept the upstream obstruction** — if near-deterministic-DAgger replay cannot support reliable critic
  gradients, then *scalar and vector critic-gradient RL inherit the same problem*, and the lever is gradient-free /
  monitor-directed search or better imitation (the prior four-attempt conclusion), not more critic engineering.

My recommendation: **(A)** — it is the discriminating test that keeps the vector hypothesis alive fairly, and it
directly attacks the measured root cause (no action diversity). I did **not** run it autonomously: it is a design
choice (how to inject action diversity) and the D: disk is effectively full (see below).

## Infra note

The D: drive hit **100% full** mid-session (system-wide, not this session's artifacts — the bulk is unrelated
experiment GIFs + `.venv`). I freed ~4 GB from regenerable caches to continue; **a real cleanup is needed before
the next multi-run job.**

## Files

`hymeko_rl/train/search_objective.py`, `hymeko_rl/train/vector_critic.py`, tests
`hymeko_rl/tests/test_vector_critic.py` (6 pass); harness `scratchpad/v2_vector_critic_probe.py`; result
`experiments/v2_vector_critic/results.json`; log `scratchpad/vector_probe.log`.

**Status:** vector-critic branch stood up + ran Steps 1–5; the result is measurement-limited (unreliable
action-gradients from a near-deterministic-DAgger replay), so Step 6 is withheld and the vector hypothesis is
neither confirmed nor refuted. Next decision: fix-and-rerun (A) vs accept the upstream obstruction (B).
