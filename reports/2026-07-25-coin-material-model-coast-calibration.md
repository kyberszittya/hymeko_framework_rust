# Coin material model + coast calibration — the transport wall is over-sticky contact physics

**Date:** 2026-07-25
**Branch:** `feat/architectural-assimilation-v1`
**Direction (user):** the transport wall is likely *too-high coin↔floor drag + poor tangential tip↔coin coupling*, not
too-heavy coin. Test a physically-motivated material model (rubberised finger + smooth table) with the two contacts
decoupled; then a coast calibration to check whether the coin drag is even realistic — calibrate the contact physics
before judging the algorithm.
**One-line outcome:** confirmed — the as-loaded coin drag is **~15× too sticky** (a 1.5 m/s coin stops in 3 cm, μ_eff
3.83 vs realistic 0.05–0.25); the legacy "success" was a high-speed *impact* exploiting that stickiness, not clean
transport. Lowering the drag monotonically increases transport. The contact physics must be recalibrated before the coin
transport task is fairly judged.

---

## 1. Mechanism — how the two contacts actually live in the model

Ruling out two wrong tools first:
- **Per-geom friction** alone can't decouple them: MuJoCo combines two equal-priority geoms' friction by the elementwise
  **maximum**, so lowering the coin's friction can't lower coin↔floor while the floor stays high (the earlier sweep's
  confound).
- **Explicit `<pair>`** overrides the whole contact (solref/solimp) → a **150 N normal-force explosion** (changes contact
  *stiffness*, not friction). Wrong tool.

The right handles (both runtime-mutable, no recompile, frozen model untouched):
- **tip↔coin friction** = fingertip `geom_friction` with `geom_priority = 2` (the higher-priority geom's friction wins the
  contact, overriding max-combination).
- **coin↔floor drag** = the coin's planar-slide **`dof_damping`** (≈ 2.5 as-loaded). The disk's slide resistance is
  *viscous DOF damping*, **not** floor contact friction (`dof_frictionloss` = 0). This is the "smooth table" knob.

(Bug found en route, same class as the earlier `pd_step` bypass: `_planar_metrics.disk_pos` is **cached** and only
recomputes inside `step_ablation`; a raw `mj_step` leaves it stale — the coast readout must use raw `qpos`.)

## 2. Coast calibration — the coin is unrealistically sticky

Inject a fixed coin speed into the free scene (arm held), measure stopping distance → Coulomb-equivalent
μ_eff = v0² / (2 g d).

| coin slide-damping | 1.5 m/s coast dist | μ_eff @ 1.5 m/s | reading |
|---|---|---|---|
| **2.5 (as-loaded)** | **0.03 m** | **3.83** | ~15× too sticky |
| 1.0 | 0.075 m | 1.53 | still sticky |
| 0.5 | 0.10 m | 1.12 | still sticky |
| 0.1 | (low-speed μ_eff ≈ 0.13) | realistic-ish | high-speed coast contaminated |

```
VERDICT: AS_LOADED_COIN_DRAG_UNREALISTICALLY_STICKY
```

A hard coin on a smooth table should have μ_eff ≈ 0.05–0.25 (a 1.5 m/s coin coasts ~0.5–1 m). As-loaded it coasts **3 cm**
(μ_eff 3.83). Realistic drag is around `dof_damping ≈ 0.1` (~25× lower); the exact value needs an **unobstructed coast
lane** — at low damping the high-speed free-coast is contaminated by the coin leaving the workspace / touching the arm
(truncation-flagged in the artifact), so only the low-speed μ_eff there is clean.

**This explains the legacy result.** The legacy scripted push worked in fast physics because it was a **high-speed impact**
(27 rad/s arm → coin launched ~1.5 m/s → excess drag stops it within ~10 cm → lands in the 10 cm zone), not a controlled
transport. Under realistic arm speed the impact energy is gone and the same over-sticky coin barely moves — which is
exactly why it collapsed. Corroborated by the V4 force decomposition (Fn 17 N but Ft only 3.6 N, ~21 % useful shear).

## 3. RUBBER_TIP_LOW_DRAG sweep (mass fixed) — direction confirmed, range too conservative

tip↔coin friction {1, 1.5, 2, 3}× × coin↔floor drag {1, 0.75, 0.5, 0.25}×, frozen V4, intermittent controller.

- **Lower drag monotonically raises transport** (0.012 → 0.033 m); **higher tip friction raises Ft/Fn** (0.2 → 2.35).
- All cells physically sensible (coin not launched: peak coin ≤ 0.17 m/s; terminal ~0; arm motion-legal).
- But within the conservative **0.25× drag floor** (= damping 0.625, still ~25× too sticky per §2) transport stays short
  of zone entry → `MATERIAL_CHANGE_DOES_NOT_UNLOCK_TRANSPORT_WITHIN_SENSIBLE_REGION`. The floor was not low enough; the
  coast calibration says the realistic drag is ~6× below even the 0.25× cell.

## 4. Claims / non-claims

**Claimed (measured):** the as-loaded coin drag is ~15× too sticky (μ_eff 3.83 at 1.5 m/s); lowering slide-damping
monotonically increases transport; higher tip friction increases the useful tangential ratio; the mechanism for
decoupling the two contacts (priority + slide-damping) is verified.

**NOT claimed / provisional:** the exact realistic damping value (the low-damping high-speed free-coast is contaminated —
needs an unobstructed lane); whether realistic drag + high tip friction *fully* unlocks delivery (the sweep didn't reach
realistic drag). The legacy-impact explanation is *inferred* from converging evidence (coast μ_eff, force decomposition,
fast-vs-realistic collapse), not a single isolating experiment.

## 5. Exact next gate

- **Recalibrate the coin drag to realistic** (`dof_damping ≈ 0.1`, or Coulomb `dof_frictionloss` with μ ≈ 0.15 + low
  viscous), validated with an **unobstructed coast lane** at 1.5 m/s → freeze `RUBBER_TIP_LOW_DRAG_COIN_V2` (mass fixed).
- **Re-run the transport sweep + C2** on the recalibrated coin with high tip friction — only then judge how much of the
  gap is the controller vs the (now-realistic) physics.
- Keep `SINGLE_TIP_LOW_FRICTION_COIN_V1` (the coupling-limited negative) intact for comparison.

---

### Commits
- `d693e0f3` — V4 intermittent + mass/friction + C2.
- `090a06bc` — coast calibration (drag ~15× too sticky) + rubber-tip sweep + material-decoupling helper (`b?`).
- this report — final.
