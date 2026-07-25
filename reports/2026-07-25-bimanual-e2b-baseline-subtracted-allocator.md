# Bimanual E2B — grasp allocation is load-bearing where the contact frame affords it; contact-frame side is the gate

**Date:** 2026-07-25 21:36 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4` + coast target. Deterministic, no RL. O3 stays paused.
**Question:** with the passive-release residue subtracted, does A2 grasp-matrix allocation add a real target-directed
impulse over A0 twist — without first assuming the preload can be released cleanly (E3)?
**One-line outcome:** the answer is **contact-frame-conditioned and positive for A2 where feasible**. From an identical
preload snapshot, over a short fixed horizon, incremental = branch − passive: on the one **far-side** frame (s7) A2
delivers **+81% more target-directed velocity than A0** (Δv∥ 0.489 vs 0.270 m/s, near-zero spin, motion-contract-legal);
on the two **zone-side** frames (s1, s5) A2 **correctly returns zero** (no feasible +e∥ wrench — the tips are on the wrong
side of the coin) while A0 produces lateral-heavy incidental motion, not a directed push. The gating variable is the
**contact-frame side**, and A2 is physically honest about feasibility.

---

## Design (baseline-subtracted, per the analysis)

From each of the 3 E1-validated balanced preloads (s1@+0.03, s5@+0.01, s7@−0.03), three **bit-identical** branches over a
**short fixed horizon** (40 steps — before the trajectories diverge into different contact modes, where subtraction stops
being linear): **P** passive hold, **A0** twist, **A2** grasp. Credit the *incremental* (branch − P) target-directed and
lateral coin velocity, spin, motion-contract, saturation — not raw zone-entry/K6 (the passive drift makes threshold
delivery metrics misleading).

## Results

| state | frame side | (p−c)·e∥ L/R | passive v∥ | A0 Δv∥ / Δv⊥ | A2 Δv∥ / Δv⊥ | A2 engaged |
|---|---|---|---|---|---|---|
| s1 | **zone** | +0.036 / +0.023 | 0.00 | 0.131 / **0.184** | 0.00 / 0.00 | no |
| s5 | **zone** | +0.035 / +0.020 | 0.00 | 0.00 / 0.063 | 0.00 / 0.00 | no |
| s7 | **far** | −0.012 / −0.030 | 0.176 | 0.270 / 0.039 | **0.489** / 0.102 | **yes** |

Absolute s7 branch metrics (all motion-contract-legal, joint ≤ 2.25 ≪ 3.45): P v∥ 0.176; **A0 v∥ 0.445**, v⊥ 0.086, ω
0.019, sat 0.15; **A2 v∥ 0.665**, v⊥ 0.149, ω 0.011, sat 0.25.

**Reading it.**
- **Contact-frame side is the gate.** `(p_tip − coin)·e∥ > 0` (zone-facing) means pressing pushes the coin *away* from the
  zone; the grasp least-squares wants negative Fn (pulling), which the friction cone clips to **zero force**. So on s1/s5
  A2 honestly produces no command — a correct refusal, not a failure. A0's twist Jacobian ignores contact feasibility and
  commands a joint motion that moves the coin, but **laterally** (s1: Δv⊥ 0.184 ≫ Δv∥ 0.131; s5: pure lateral) — not a
  directed launch.
- **Where the frame is far-side (s7), A2 wins decisively:** +81% target-directed increment over A0 (0.489 vs 0.270),
  comparable near-zero spin, within the motion contract — at the cost of somewhat more lateral (0.102 vs 0.039) and
  saturation (0.25 vs 0.15). Grasp-matrix wrench allocation **is** load-bearing when the geometry affords a push.

Verdict: `ALLOCATOR_ADVANTAGE_DEPENDS_ON_CONTACT_FRAME_AND_LOCAL_ACTUATION_AUTHORITY` — the third of the three
pre-registered outcomes, and the informative one: the allocator math is correct *and* its value is conditioned on the
contact frame the acquisition selects. Precisely: **`A2 MECHANISTIC ADVANTAGE DEMONSTRATED ON ONE FAR-SIDE CONTACT
FRAME`** — not a general "A2 decisively wins" (n=1 far-side state).

## Feasibility audit — the A2 zero is a proven honest refusal

Before crediting "A2 correctly declines" as a claim, it must be checked against the physical feasibility bound (a zero
output could also be over-regularisation, a scale error, solver clipping, a wrong normal, or the local arm controller
failing to realise the allocated force). The **forward authority** `A∥(s) = max_{f∈cone, Fn≤1} e_par·F_net = Σᵢ max(0,
e_par·nᵢ + μ|e_par·tᵢ|)` (closed form; unit-tested on far / zone / mixed synthetic pairs) settles it:

| state | frame | A∥ (per-contact) | grasp forward force | A2 engaged | honest? |
|---|---|---|---|---|---|
| s1 | zone | **0.00** (0.00, 0.00) | 0.00 | no | ✅ |
| s5 | zone | **0.00** (0.00, 0.00) | 0.00 | no | ✅ |
| s7 | far | **1.56** (0.54, 1.02) | 0.586 | yes | ✅ |

`A∥ ≤ 0 ⟺ A2 disengages` holds on **all three real snapshots** (`honest-refusal verified: True`). So the zero on s1/s5 is
a **physically honest refusal** — pressing those (zone-side) contacts cannot produce any forward net force — not a bug;
and s7's A∥ = 1.56 with **both contacts contributing** (0.54, 1.02) shows the real gate is feasibility, not a hardcoded
"both far-side" sign.

**Snapshot fidelity:** the P/A0/A2 branches are `deepcopy` clones of the one validated preload env — same qpos/qvel, coin
state, contacts, and MuJoCo warm-start — released at the same instant; the controller state (`prev_tau`) starts fresh and
identical per branch. So the incremental is a true same-state comparison, not a re-reconstruction.

## What this unifies

E0 (false-positive removal) → E1 (balance exists, off-midpoint) → E2 (irreducible passive-release residue) → **E2B: the
balanced frame must also be on the coin's far (−e∥) side for any allocator to push toward the zone.** Balance and
launch-direction are **two requirements on the same acquisition**, and the earlier single-tip result already pointed here
(L3 far-side acquisition helped in `TARGET_DIRECTED_LAUNCH_V1`). The two remaining physical gates (per the analysis) are
now precisely:

1. **Far-side, balanced acquisition** — select a contact frame that is both authority-balanced (E1) *and* on the −e∥ side
   (E2B). s7 shows this frame exists and A2 launches strongly from it.
2. **E3 — net-wrench-null release** — drive the *net* residual wrench (not just Fn_L−Fn_R: also net tangential force and
   coin torque from moment arms / contact-point orientation) to zero, so the release is clean.

Then the delivery is largely assembly: clean far-side preload → A2 controlled impulse → calibrated coast → B1 barrier →
settle. Coast, material model, and motion contract are already solved.

## The semantic mismatch this exposes (the real lesson)

E1's acquisition succeeded on Fn-balance; E2B shows **`balanced contact ≠ launch-capable contact`**. A perfectly balanced
but zone-side pair has zero forward authority, so E1's current termination can mark "done" a terminal state from which the
next option (LAUNCH) is provably infeasible — a classic option-semantics mismatch (ACQUIRE success ≠ LAUNCH entry). The
fix is not a second reward term; it is the acquisition **termination semantics**. Fn-balance is only a diagnostic proxy —
the physical preload target is a null net wrench `w_preload = G·f ≈ 0`, and the physical launch target is feasibility
`∃ f ∈ C: G·f ≈ [F∥*; 0; 0]`. So E3's net-wrench-null is part of *correct acquisition*, not a later refinement.

## Claims / non-claims

**Claimed (measured + verified):** contact-frame side gates allocator feasibility; A2's zero on the zone-side frames is a
**proven honest refusal** (A∥ = 0, verified against the closed-form feasibility bound on the real snapshots); on the one
far-side frame A2 delivers +81% more target-directed increment than A0 within the motion contract; A0 produces
lateral-heavy incidental motion on zone-side frames. Downstream *component* capabilities (coast, passive-barrier braking,
impulse authority on subsets) are established.

**NOT claimed:** that A2 > A0 in general (only `A2 MECHANISTIC ADVANTAGE DEMONSTRATED ON ONE FAR-SIDE FRAME`, n=1); that a
far-side *feasible* balanced frame exists on all states (E1 balance on 3/8, only s7 far-side); that the release is clean
(E2 residue stands); that the **full launch → coast → barrier → settle composition** works from a clean bimanual
acquisition distribution — `DOWNSTREAM COMPONENT CAPABILITIES ESTABLISHED, FULL COMPOSITION STILL OPEN`.

## Exact next rung (ordered, before extending the search)

1. ✅ **Freeze contact-frame sign conventions** (`_contact_frames`) and **add the grasp-feasibility oracle + tests**
   (`forward_authority`, `_grasp_solve` diagnostics; 4 unit tests far/zone/mixed/consistency) — **done here**; the A2
   refusal is now proven honest on the real snapshots.
2. **Replace the Fn-balance acquisition termination** with a lexicographic gate: Gate 1 clean contact (dwell, bounded
   penetration, settled qdot, no saturation) → Gate 2 clean preload (small realized net force + coin torque, quiet
   release baseline) → Gate 3 launch feasibility (A∥ > 0, bounded min cross-force + torque, cone-feasible) → Gate 4
   secondary quality. `done=True` only if Gates 1–3 pass. Search score keys on the wrench residual and feasibility margin,
   not Fn-balance.
3. **Then** extend the E1 search with the far-side sign kept only as a **candidate prior** (not a hard rule — the final
   choice is grasp-matrix feasibility + realizable object wrench), and quantify how often a feasible balanced frame exists.
4. **Then** E3 net-wrench-null release. Only after that: demos / proposal / RL. O3 paused.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `measure_release_branch` (short-horizon single-branch measurement,
  passive/A0/A2 via one code path); `_contact_frames` (frozen sign convention), `_grasp_solve` (grasp allocation with
  unclipped/clipped/residual diagnostics), `forward_authority` (closed-form A∥ feasibility bound); `_grasp_allocation`
  now a thin wrapper over `_grasp_solve`.
- `hymeko_rl/experiments/bimanual_curriculum_e2b_benchmark.py` — 3 bit-identical `deepcopy`-clone branches +
  `_contact_frame_side` + `_feasibility_audit` (A∥ + grasp solve → honest-refusal check) + baseline-subtracted metrics.
- `hymeko_rl/tests/test_cooperative_grasp.py` — 4 new feasibility-oracle tests (far / zone / mixed / A∥↔allocation
  consistency).

### Test results
- Unit: `test_cooperative_grasp` **8/8** pass (4 grasp + 4 feasibility); ruff clean.
- Benchmark: 3 states × (search + feasibility audit + 3 branches), ~14 min wall, single-thread, seeds 14000+250·i.
- Honest-refusal verified on all 3 real snapshots (`A∥ ≤ 0 ⟺ A2 disengages`).
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/bimanual_curriculum_e2b.json`. Coast μ 0.179, horizon 40.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
