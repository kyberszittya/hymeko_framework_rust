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
contact frame the acquisition selects.

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

## Claims / non-claims

**Claimed (measured):** contact-frame side gates allocator feasibility; A2 returns zero on zone-side frames (physically
correct) and, on the far-side frame, delivers +81% more target-directed incremental velocity than A0 within the motion
contract; A0 produces lateral-heavy incidental motion on zone-side frames.

**NOT claimed:** that A2 > A0 universally (only where the frame is far-side and feasible — n=1 far-side state here); that a
far-side *balanced* frame exists on all states (E1 found balance on 3/8, only s7 of those is far-side); that the release is
clean (E2 residue stands — E3 pending). The baseline subtraction is only trusted over the short shared-topology horizon.

## Exact next rung

- **Far-side authority-balanced acquisition search** — extend the E1 search to require `(p−c)·e∥ < 0` on both tips (a
  2-objective: balance ∧ far-side), and quantify how often such a frame exists. Where it does, run A2 as the launch.
- Then **E3 net-wrench-null** release for a clean preload→launch transition. Only then demos / proposal / RL. O3 paused.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `measure_release_branch` (short-horizon single-branch measurement,
  passive/A0/A2 via one code path).
- `hymeko_rl/experiments/bimanual_curriculum_e2b_benchmark.py` — 3 bit-identical branches + `_contact_frame_side`
  diagnostic + baseline-subtracted incremental metrics.

### Test results
- Unit: `test_cooperative_grasp` 4/4 pass; ruff clean.
- Benchmark: 3 states × (search + 3 branches), ~14 min wall, single-thread, deterministic seeds 14000+250·i.
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/bimanual_curriculum_e2b.json`. Coast μ 0.179, horizon 40.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
