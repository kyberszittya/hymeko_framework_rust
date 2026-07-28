# R9 authority audit — learning-free reachability of the ≤30 mm corridor: bounded residual authority is insufficient

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · teacher-free · NO RL this turn**

## Summary

Before spending any more RL on the R3-B wall, this is a **learning-free** action-authority audit: from the two healthy R2-champion
frontiers (57.3 mm, 53.84 mm), a bounded CEM reachability search asks whether a *safe, expressible* residual sequence can reach the
strict ≤ 30 mm corridor — and with how much residual authority. Three families were searched, ordered smallest→largest authority:

- **A0** — the current 4-D per-joint residual, α = 0.15 (the R2/R3-B control).
- **A1** — the same 4-D per-joint residual at a *larger bound*, α ∈ {0.20, 0.25, 0.30}.
- **A2** — a minimally-EXPANDED, **structured** coin-following basis (NOT raw torque): left-tip forward-follow, right-tip
  forward-follow, common squeeze, left/right differential, common tangential pursuit — mapped to Δτ through the *live* tip
  Jacobians so the residual can track the two tips alongside the sliding coin.

**Verdict: `CURRENT_RESIDUAL_AUTHORITY_INSUFFICIENT`.** No family reaches the corridor. Every (frontier × family) cell plateaus at
**36–37 mm min_dtz — ~6 mm short of 30 mm** — *cleanly* (0 stall / 0 reversal / 0 clamp), moving (+v_par), light (Fn < 0.15 N),
and safe. Doubling the residual bound (α 0.15 → 0.30) buys **≈ 0.5 mm**; the structured A2 basis buys **≈ 0.8 mm** over A0. The
authority families improve reachability **monotonically but saturating** — neither more bound nor a structured basis crosses 30 mm.
**RL over any of these families is not yet justified**: the bounded-residual-over-frozen-clone reachable set tops out ~6 mm short.

## Reachability table (best clean min_dtz per cell; corridor ≤ 30 mm)

| frontier | A0 α0.15 | A1 α0.20 | A1 α0.25 | A1 α0.30 | A2 α0.25 | Δ(A0→best) | REACH |
|---|---|---|---|---|---|---|---|
| **57.3 mm** | 36.91 | 36.73 | 36.53 | 36.37 | **36.14** | −0.77 mm | ✗ (all) |
| **53.84 mm** | 37.29 | 37.16 | 37.01 | 36.87 | 36.80 | −0.49 mm | ✗ (all) |

Every cell: `stall/reversal/clamp = 0/0/0`, `safe = True`, `v_par > 0`. Exit force stays light (Fn 0.065–0.145 N). From 53.84 mm the
CEM converges to the **same exit state** across families (exit_v 0.324, Fn 0.123) — authority barely perturbs it; from 57.3 mm there
is slightly more room (exit_v 0.334 → 0.357, Fn 0.065 → 0.145 as authority rises), but min_dtz still saturates at 36.1 mm.

## What this refines about the R3-B hypothesis

R3-B closed with the plausible reading *"reaching the 20–30 mm corridor needs residual authority the current action basis does not
supply."* This audit **sharpens and partly corrects** that:

- **It is not a residual-bound gap.** A1 doubles the bound and moves min_dtz by ≈ 0.5 mm — the reachable set is nearly flat in α. So
  `EXPANDED_KINETIC_AUTHORITY_REACHES_CLOSE_MOVING_MANIFOLD` is **falsified**.
- **It is not closed by the structured basis either.** A2's coin-following basis (tip forward-follow + squeeze + differential +
  tangential pursuit) helps monotonically (best cell, 36.1 mm) but still falls ~6 mm short. So
  `EXPANDED_BASIS_REQUIRED_STRUCTURED_COIN_FOLLOWING` is **not yet supported** — the structured basis is *better*, not *sufficient*.
- **The binding constraint is upstream of the residual.** Both levers that a residual-over-a-frozen-clone can pull — bound and basis
  — saturate ~6 mm short. What a bounded residual can reach is anchored to the **frozen clone's own contact-decaying trajectory**;
  perturbing around it, however expressively, tops out at ~36 mm. The corridor is physically reachable (the teacher holds contact to
  23 mm), so the gap is not kinematic — it is that **the scaffold trajectory the residual is bounded around does not itself pass
  close enough to the corridor** for any admissible perturbation to close the last 6 mm cleanly.

## Pre-registered next diagnostic (NOT run this turn)

Per the decision logic ("if none reach → RL not yet justified; inspect which teacher-torque component is not in the basis span"),
the next learning-free step is a **teacher-torque-span projection**: at the frontier states, project the teacher's KINETIC-phase
torque onto the span of the A0/A2 residual basis and report the residual-orthogonal component — i.e. quantify *which* direction of
the teacher's corrective torque the bounded residual cannot express. This turn establishes the reachability ceiling; the projection
would explain it. It is a distinct analysis and is deferred for review, consistent with the campaign's stop-for-review cadence and
tonight's "no RL, no basis expansion beyond the audited A2" scope.

## Method / safety

- **Learning-free.** CEM over the per-step residual sequence only (horizon 14, pop 48, elite 8, 6 iterations, fixed RNG seed
  20260728) — no policy trained, no gradient step. Score `−min_dtz − 40·(stalls+reversals+clamps)`, `−1e9` if unsafe, so the search
  cannot "reach" by stalling or clamping the coin.
- **Segment-local restart** from the healthy frontier reuses the R3-B hybrid-boundary contract: restored clone GRU hidden + prev
  residual + prev_tau, mode forced to KINETIC. Verified by the update-zero identity (below).
- **Frontiers** are the clean R2 champion's healthy interiors (57.3 mm: v_par 0.357, restart 3; 53.84 mm: v_par 0.324, restart 2);
  the 61 mm edge state stays correctly rejected as a boundary/terminal-risk state (R3-B).
- **Safety gate**: peak_qdot ≤ 3 rad/s, peak coin speed ≤ 1.5 m/s — all cells pass.

## `AUTHORITY_REACHABILITY_PASS` gate

`min_dtz ≤ 30 mm ∧ exit_v_par > 0 ∧ exit_Fn < 2 N ∧ stalls = 0 ∧ reversals = 0 ∧ clamps = 0 ∧ safe`. Extracted as a pure
`reachability_pass(m)` and unit-tested clause-by-clause. **0 / 10 cells pass** → smallest passing family = **None** →
`CURRENT_RESIDUAL_AUTHORITY_INSUFFICIENT`.

## Files touched (all new / additive; K0–R3-B modules unchanged)

| file | lines | role |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_authority.py` | +182 | A2 structured basis (`a2_structured_u` via live tip Jacobians), `KineticAuthorityController` (frozen clone + A0/A1/A2 residual, segment-local restart), `authority_cem` bounded reachability search, pure `reachability_pass` gate |
| `hymeko_rl/experiments/coin_kinetic_authority_audit.py` | +88 | driver: regenerate clean R2 champion → capture healthy frontiers → CEM per (frontier, family) → smallest-passing verdict → JSON |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | +61 | +3 tests (reachability-gate predicate; zero-residual-reduces-to-clone for A0 **and** A2 + live-A2 divergence; CEM determinism) |
| `reports/2026-07-28-coin-r9-authority-audit/authority_audit.json` | — | full 10-cell reachability sweep + verdict |

## Tests / static analysis

- **New authority tests — 3 passed** (93 s): `AUTHORITY_REACHABILITY_PASS` predicate (each clause flips the verdict);
  update-zero identity — a zero residual reproduces the clone restart continuation **bit-for-bit for both A0 and A2**, and a live
  A2 coefficient sequence **diverges** (structured mapping is not a no-op); CEM determinism (same seed → identical result).
- **Full `test_coin_kinetic_contract.py`** — see run log (25 tests: 22 prior + 3 authority).
- `ruff check` clean on all three touched files; `radon cc -a` **A (3.625)**, worst `authority_cem` = **B** (under the ceiling).
  No new suppressions; no §6.5 anti-patterns (families are one config-dispatched entry point, not a Cartesian dump; the string
  `family` is bounded to `{A0,A1,A2}` at the one CEM boundary; no globals).

## Provenance

Git `452d4e5c` (R3-B commit; audit files uncommitted at run time — this report's commit adds them). Python 3.11.15 / mujoco 3.10.0
/ numpy 2.4.6 / torch 2.12.0 / macOS-26.5.2-arm64 (Apple Silicon, CPU). Seeds: cradle 14250; R2 regen 0; CEM 20260728. α ∈
{0.15, 0.20, 0.25, 0.30}. **Peak RSS 0.33 GB; wall 128.3 s** (audit) — both far under budget. Determinism: audit JSON reproducible
(fixed RNG); dev s1 only; s4/s7/f1–f4 never touched.

## Status

`CURRENT_RESIDUAL_AUTHORITY_INSUFFICIENT` — the first real audit verdict. Neither a larger residual bound (A1) nor a structured
coin-following basis (A2) reaches the ≤ 30 mm corridor from the healthy R2 frontier; all families saturate ~6 mm short, cleanly.
**No RL was run and none is justified yet.** No strict K6 → no champion freeze, no tag; **R2 stays the champion of record** (contact
50.66 mm, min_dtz 39.58 mm, clean, moving). Committing the audit on its own boundary. The pre-registered next step
(teacher-torque-span projection) is deferred for user review.
