# Humanoid anthropomorphic redesign — AIBO-level cosmetic detail, balance stack preserved

**Date:** 2026-07-29
**Worktree:** `hymeko_humanoid` (branch `research/humanoid-com-lyapunov`)
**Scope:** redesign `data/robotics/humanoid.hymeko` from a bare 18-body stick figure to an
anthropomorphic humanoid at the AIBO's cosmetic-detail level, **without** disturbing the validated
balance / ZMP / capturability stack.

## Summary

The prior humanoid looked like a blocky stick figure (18 bodies) next to the AIBO's 33-body ERS-1000.
The redesign follows the AIBO's exact idiom — **FIXED, non-colliding cosmetic links** (`fixed_joint` +
`collision { contype 0; conaffinity 0; }`) — to add a neck (collar), two shoulder ball-joints, two hip
ball-joints, two hands, and two eyes, plus an anthropomorphic palette (blue suit, tan limbs, skin
head+hands, dark-metallic joints, dark shoes, dark eyes). Because every added part is a fixed,
visual-only link, the **kinematic tree, the 16-motor action space, the leg lengths, and the foot
support geometry are byte-for-byte unchanged** (nq/nv/nu = 23/22/16, identical to the prior model).

**Result: 39/39 humanoid tests pass with ZERO controller re-tuning and ZERO test-threshold changes.**
The only code artifact touched is the one data file (`humanoid.hymeko`); every controller and scenario
module is unchanged. The cosmetic mass is *measured* to be balance-basin-neutral.

## What was measured (the one real risk, and how it was resolved)

The first draft also baked a 12° elbow pre-bend into the arm mount rpy (to avoid rigid-pole arms). That
broke 3 razor's-edge controller-basin tests. A controlled A/B isolated the cause precisely:

| Plant | `descent@0.2` (shaped energy) | PMP certifies @0.3 |
|---|---|---|
| OLD (prior model, 34.60 kg) | 0.916 | ✅ |
| NEW: light cosmetics **+ elbow bend** (35.16 kg) | 0.711 | ❌ |
| NEW: light cosmetics, **straight arms** (35.16 kg) | **0.916** | ✅ |

The regression was **entirely** the elbow bend (bent forearms+hands swing during pitch recovery →
non-monotone shaped-energy descent + PMP overshoot), **not** the cosmetic mass. Straight arms restore
the OLD basins bit-for-bit. So the elbow bend was reverted (measured constraint, documented in the model
comment); the cosmetic mass (+0.56 kg, +1.6%) is proven basin-neutral. This is the honest fix — no gain
was massaged to paper over a plant change; the one offending geometric choice was removed after
measurement.

Two other hypotheses were tested and rejected before landing on the elbow bend: (a) re-tuning the
descent test to the true mass-weighted kinetic energy ½q̇ᵀMq̇ instead of the unit-mass proxy — *worse*
descent (0.58); (b) bumping PMP state cost q_pos/r — did not restore @0.3 (the plant stayed upright but
V did not converge). Both discarded; the elbow-bend removal is strictly cleaner.

## Files touched

- `data/robotics/humanoid.hymeko` — **+54 / −16 lines**. Added 9 cosmetic links (collar, shoulder_pad_l/r,
  hip_cap_l/r, hand_l/r, eye_l/r) + 9 `fixed_joint` mounts, expanded palette, darkened feet (shoe color),
  head recolored skin. Load-bearing links, all 16 rev_joints, couplers, base, and foot geometry unchanged.
- Regenerated video artifacts (model refresh, no code change):
  `reports/2026-07-29-humanoid-zmp-multiembodiment/humanoid_zmp_balance.{mp4,gif}`,
  `reports/2026-07-27-humanoid-sac-residual/humanoid_balance_{compare,frontier}.{mp4,gif}`.

**CORE.YAML items touched:** none (`humanoid.hymeko` is scenario data; `hymeko_control`/scenario code
untouched). No dependencies added.

## Verification

- **Structure (bit-identity of the control surface):** nq=23, nv=22, nu=16 — identical to the prior model.
  Total mass 34.60 → 35.16 kg (+1.6%, thin cosmetic shells). 9 emitted geoms carry `contype=0 conaffinity=0`
  (visual-only, zero contacts).
- **Standing equilibrium preserved:** settled COM height under the a=0 PD-hold controller = 0.6479 m
  (prior h_ref = 0.645; Δ = 3 mm, Lyapunov contribution ≈ 2.6e-5 — h_ref left unchanged). Stands 400 steps
  under zero control at uprightness = 1.000.
- **Tests:** `pytest tests/ -p no:randomly` → **39 passed** (unit + integration + Lyapunov/ZMP/PMP/
  energy-shaping certificate layers). No controller gain or test threshold was modified.
- **Look:** renders read clearly as a humanoid (skin head + eyes, neck, shoulder/hip ball-joints, hands,
  shoe feet, blue suit) — a decisive step up from the stick figure. Side/3-quarter views in `/tmp/hum_look/`.
- **Videos:** the validated Vukobratović-ZMP balance certificate video re-rendered on the new model —
  small push → *ZMP in support, CERTIFIED, 0-step*; big push → *ZMP out, MUST step (1-step region)*.

## §6.5 anti-patterns

None introduced — a data-model edit; no new code paths, no Cartesian API, no globals.

## Provenance

- Git SHA: working tree dirty — changed: `data/robotics/humanoid.hymeko` + 4 regenerated video files
  (listed above). Control/scenario Python unchanged.
- Env: shared venv `hymeko_framework_rust/.venv` (Apple-Silicon, mujoco, numpy, imageio); macOS 25.5.0.
- Emitter: `target/release/hymeko emit -f mjcf` (release build).
- Seeds: tests + renders seed 0 (deterministic).

## Follow-up: parametric rewrite (no magic numbers)

Per the operating principle "not pre-baked values and magic numbers — that's partly why we make HyMeKo",
the model was then rewritten as a **parametric anthropometric template** using HyMeKo's compile-time
numeric-expression resolver (`const NAME = expr;`, arithmetic + forward refs, `hymeko_core` Tier-B —
already in the language, non-core to use; confirmed on `data/minimal_examples/constants/`).

- A **43-line `const` block** of named anthropometric primitives (segment lengths / radii / masses +
  the joint-mount frame). Every link dimension, mass, origin, and joint offset now **derives** from these
  via arithmetic. The scattered literals in the link/joint bodies are gone (only structural `0.0`/`1.0`
  and the `contype/conaffinity 0` collision mask remain).
- This removes the real **duplications** a hand-authored model leaks (§6.1): each segment length appeared
  3× (geometry, mid-link origin `−len/2`, child joint offset `−len`) → now `THIGH_LEN` etc. stated once
  and derived (`THIGH_ORIGIN_Z = −THIGH_LEN/2.0`, `KNEE_Z = −THIGH_LEN`); each lateral mount appeared 4×
  (L joint, R joint, L cosmetic, R cosmetic) → now `HIP_Y` / `SHOULDER_Y` stated once, mirrored via
  `−HIP_Y`. Change one primitive and the body re-derives coherently.
- **Faithfulness proven:** the parametric model emits **byte-identical MJCF** to the validated
  hand-authored model (`diff` empty — all derived values match exactly, no float-format drift), and
  **39/39 tests still pass**. So this is a pure refactor: the balance/ZMP/PMP stack is provably untouched.

## Open items / follow-ups

- A baked natural arm posture (slight elbow/shoulder rest angle) is *possible* but must be co-tuned with
  the balance basins (the bent-arm swing costs monotonicity). Deferred — straight arms are a normal
  standing pose and keep the certified stack bit-stable.
- Plan hygiene: this was executed as a contained single-data-file change + full re-validation (session's
  iterative-data mode), not a 4-artifact plan doc; flagged here per the operating contract.
