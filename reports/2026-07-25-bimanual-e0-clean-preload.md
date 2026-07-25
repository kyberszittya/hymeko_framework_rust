# Bimanual E0 — a false-positive removed, and the honest clean-preload entry condition

**Date:** 2026-07-25 20:13 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4` + coast target. Deterministic, no RL. O3 stays paused.
**Walk:** step-by-step — E0 (static bilateral positive control) is the first rung of `BIMANUAL_ACQUISITION_CURRICULUM_V1`.
**One-line outcome:** the earlier **"co-contact 8/8" was a false positive** — the tips buried 18–34 mm into the
hard-pinned coin and the positive launch velocity was a **pin-release spring explosion**, not a clean bimanual impulse.
Rebuilt E0 as three separately-validated stages; the honest state is now: grasp-allocation math **PASS**, a clean bounded
**balanced** bilateral preload is **rare at the raw tip-midpoint (1/8)** because the two arms have **asymmetric contact
authority** there, and even the one clean preload has a small **release residue** (E0b), so the A0-vs-A2 launch is **not
yet evaluable**. This is not a wall — it is the removal of a false capability proof and a correct benchmark entry gate.

---

## Why the earlier co-contact was a false positive

The first E0 drove both tips toward the coin **centre**. The centre is unreachable (the coin body blocks it), so the FD
close marched each arm in until the torque saturated. Measured preload sanity at release:

| state | penetration L / R | qdot at release | torque saturated | fn balance |
|---|---|---|---|---|
| s0 | −0.028 / −0.004 m | 0.74 | **yes** | 0.72 |
| s1 | −0.034 / −0.018 m | 0.50 | **yes** | 0.74 |

Penetration **deeper than the coin radius (0.02 m)** on one side, with saturated torque and un-settled joint velocity. On
release that stored energy flings the coin. The 8/8 "co-contact" and the positive `v_∥` were an artifact. **Downgraded.**

## The rebuilt E0 (three stages, each gated separately)

- **E0a — clean bounded PRELOAD.** Coin at the two-arm-reachable midpoint, held by a **soft damping pin** (a hard
  kinematic clamp is an infinitely-stiff wall that traps the tips in a qdot limit cycle and never settles). Each arm
  **servos its true contact penetration** (`con.dist`, not a geometric proxy) to δ = 5 mm — equal penetration → balanced
  normals. Radii read from the model (`r_coin + r_tip − δ`), never hardcoded. Gate (frozen): both in contact ∧ penetration
  ∈ [1, 10] mm ∧ Fn_L,Fn_R ≥ 0.15 N ∧ balance ≥ 0.30 ∧ qdot ≤ 0.45 (calibrated above the ~0.33 contact limit-cycle floor).
- **E0b — release-only SANITY.** From the clean preload, **release the pin with no launch command**; a genuinely clean
  preload must not fling the coin (peak speed + displacement). This is the physical backstop that no spring residue drives
  the launch.
- **E0c — A0 vs A2 LAUNCH.** Only from a clean, release-sane preload: A0 coin-twist allocation vs A2 grasp-matrix
  resultant-force allocation (friction-cone projected).

## Results (8 states, frozen physics)

| stage | metric | result | verdict |
|---|---|---|---|
| E0a | clean bounded balanced preload | **1/8 (0.125)** | `CLEAN_BOUNDED_PRELOAD_NOT_YET_ESTABLISHED` |
| E0b | release without launch → no jump | **0/2 acquired states** | `RELEASE_ONLY_SANITY_FAIL_SPRING_RESIDUE` |
| E0c | best allocator target-directed | **0/8** | `COOPERATIVE_FORCE_ALLOCATION_NOT_YET_TARGET_DIRECTED` |

The two states that acquire an in-band co-contact:

| state | pen L / R | fn L / R (balance) | E0b speed / disp | A0 cross | A2 cross |
|---|---|---|---|---|---|
| s2 | −7.6 / −1.1 mm | 2.02 / 0.00 (**0.00**) | 0.481 / **0.113 m** | 0.964 | **0.161** |
| s7 | −1.0 / −5.0 mm | 0.86 / 1.24 (**0.69**) | 0.248 / 0.026 m | 0.634 | 1.883 |

**Reading it.** s7 is the one *clean, balanced* preload (fn balance 0.69) — proof the mechanism can form — yet it still
drifts 2.6 cm on release; s2 is *imbalanced* (right tip barely loaded, fn 0.00) and flings the coin **11 cm**. The release
residue tracks the force imbalance exactly. On 6/8 states the right arm cannot even press a light balanced preload at the
midpoint — **asymmetric contact authority** at the geometric midpoint, not the authority-balanced point.

A0 vs A2 is **mixed** (A2 wins s2, A0 wins s7) — and correctly **uncredited**: with the preload contaminated by the
release residue, the allocator's fine effect cannot be isolated. That A2 reaches cross-ratio 0.161 on s2 is an encouraging
but not-yet-load-bearing signal.

## Honest ledger

```
GRASP_ALLOCATION_PURE_MATH .................... PASS   (test_cooperative_grasp: symmetric pair → target wrench)
FRICTION_CONE_CONSTRAINT ...................... PASS   (|Ft| ≤ μ·Fn, Fn ≥ 0)
SYMMETRIC_PAIR_PRODUCES_TARGET_WRENCH ......... PASS   (zero cross, zero torque along e_par)
PINNED_GEOMETRIC_COCONTACT (in-band) .......... 2/8    (right-arm authority limits the rest)
CLEAN_BOUNDED_BALANCED_PRELOAD ................ 1/8    (achievable but rare at the raw midpoint)
RELEASE_ONLY_SANITY ........................... FAIL   (even the clean preload has a small release residue)
COOPERATIVE_LAUNCH (A0 vs A2) ................. NOT YET EVALUABLE
```

## Claims / non-claims

**Claimed (measured):** the earlier co-contact was a pin-spring artifact (downgraded); the grasp-allocation math is
correct and friction-cone-safe; a clean balanced bilateral preload forms on 1/8 states; the release residue is proportional
to the force imbalance; the geometric tip-midpoint is **not** the authority-balanced point (right-arm authority is the
binding constraint on 6/8).

**NOT claimed:** that A2 beats or ties A0 (uncredited until E0a/E0b pass); that bilateral launch is or isn't achievable —
only that its evaluation is blocked on a clean, release-sane preload; that δ = 5 mm is optimal (it is calibrated, not tuned
against a metric).

## Exact next rung

1. **Place the coin at the authority-balanced point**, not the geometric tip-midpoint — the point where both arms press an
   equal normal force (search along the workspace-overlap axis for `fn_balance → 1.0`). This should convert the 6/8
   no-acquire states and push the release residue toward zero.
2. **Or an explicit force-balance (A1) inner loop** that regulates Fn_L ≈ Fn_R directly (the penetration servo balances
   penetration, but asymmetric authority breaks the penetration↔force map on the weak arm).
3. Then **E0b should pass** and **E0c A0/A2** finally becomes a real allocator comparison → teacher demos → RL. **O3 stays
   paused.**

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — grasp allocation + `TwistAllocator`/`GraspAllocator` strategies;
  `acquire_clean_preload` (soft-pin + penetration-balance servo), `release_only_sanity`, `release_pin`, `_preload_sanity`
  (frozen clean gate), `_tip_contacts`, `_grasp_allocation`, `_surface_target`/`_coin_radius` (model-read radii);
  `cooperative_launch_carry` refactored to a pluggable allocator (A0 default, bit-identical math).
- `hymeko_rl/experiments/bimanual_curriculum_e0_benchmark.py` — E0a→E0b→E0c pipeline + contact fingerprint.
- `hymeko_rl/tests/test_cooperative_grasp.py` — 4 grasp-allocation unit tests (all pass; +3 bimanual = 7).

### Test results
- Unit: `test_cooperative_grasp` 4/4, `test_bimanual_launch` 3/3 — **7 passed**. ruff clean.
- Benchmark: 8 states, ~11 min wall, single-thread, deterministic seeds 14000+250·i.

### Provenance
- Contact fingerprint (saved in artifact): coin r 0.02, tip r 0.02, disk solref [0.02, 1.0], margin 0, timestep 5e-4.
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/bimanual_curriculum_e0.json`. Coast μ 0.179, δ 5 mm.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
