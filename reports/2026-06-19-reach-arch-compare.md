# Report — HSiKAN-vs-MLP comparison harness + a halt on the emitted arm's torque scale

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ⏸ **HALTED at the production-scale smoke (§11).** The comparison **harness** is built,
tested, and reproducible; the **4-DOF arm_world** comparison works; but the canonical **6-DOF
emitted arm** is **not learnable as configured** — the torque controller and the arm model are
mutually inconsistent. The 5-seed run is NOT queued pending a control/model decision.

## What was built
- **`hymeko_rl/reach_arch_compare.py`** — `emitted_arm_factory` (emit + parse the robot/obs/
  reward profiles once, build fresh envs on demand) and `compare_backbones` (BC for each
  backbone × seed; median / IQR / worst final EE error + `n_params`). CLI `--mode smoke|full`.
  Fully reproducible: each `(backbone, seed)` seeds the policy init, the BC training, and the
  env RNG, so both backbones face identical targets.
- **`hymeko_rl/bc.py`** — `run_bc` gained `env_factory` (scene selection) and now seeds torch
  before the policy init (the init was previously entropy-seeded — a §3 reproducibility gap).
- **`hymeko_rl/tests/test_reach_arch_compare.py`** — 3 tests (IQR; factory builds the 6-DOF
  arm; `compare_backbones` returns both backbones). All green; `test_reach_bc` still green.

## The smoke (production-scale, 1 seed, 150 epochs, both backbones — wall 62 s)
| arm | backbone | reach err (m) | untrained floor (m) | BC loss | learns? |
|---|---|---|---|---|---|
| arm_world (4-DOF, hand-authored) | hsikan | **0.369** | 0.433 | 1.92 | ✅ yes |
| emitted (6-DOF, canonical) | hsikan | 0.828 | 0.782 | **17.35** | ❌ no (worse than floor) |
| emitted (6-DOF, canonical) | mlp | 0.741 | 0.773 | — | ❌ marginal |

Both backbones fail equally on the emitted arm → this is **not** an architecture result; the
**task is unlearnable as configured**.

## Root cause (measured, not assumed)
The env clips torque actions to **±25 N·m** (the `_DEFAULT_CTRL` fallback, tuned for the small
arm_world). On the 6-DOF emitted arm the computed-torque expert (kp=3000) **demands** torques
of **median 2608, max 15154 N·m** → **80 % of expert actions saturate** at ±25. The
demonstrations are therefore bang-bang and unclonable (BC loss 17.4 vs 1.9 on arm_world), and
even a perfect clone would not reach. Two compounding facts:
1. **The controller is mistuned for the big arm.** kp=3000 × the DLS-IK step `dq` (larger over
   the 1.2 m workspace) → desired accelerations → `M·a_des` torques in the thousands.
2. **The arm model is physically inconsistent.** The `.hymeko` declares joint `effort 50` N·m,
   but the moving links are heavy (2–5 kg) on a long arm — the declared joints could not drive
   these masses. The 25 kg base + 50 N·m efforts read as **placeholders**, not the intended
   BCN3D-Moveo physics (real Moveo links are ~0.5–2 kg).

## Why halted (§11)
The smoke contradicts the assumption that BC works on the emitted arm. The fix is a control /
model decision (which torque scale, whether the masses/efforts are placeholders, torque vs
position) — a robotics-engineering call for the user, not a gain I should guess (§3/§11: a
guessed "now it works" is a superstition). The harness is ready to run the moment the arm is
learnable.

## Options (for the decision)
1. **Make the arm physically consistent + retune torque** — realistic link masses + joint
   `limit_effort`, derive the env ctrlrange from `limit_effort`, scale the expert kp so the
   computed torque fits. Keeps torque (the stated preference); needs the `.hymeko` masses/limits
   to be real, and `extract_joint_limits` to follow the `limit -> …` ref (a CORE edit).
2. **Retune the controller only** — lower expert kp + an arm-appropriate ctrlrange (e.g. a few
   × the gravity-hold torque), leaving the masses as-is. Faster; leaves the model unphysical.
3. **Run the comparison on arm_world now** — a real HSiKAN-vs-MLP result on the learnable
   4-DOF arm (the "serial-arm gap is expected small" milestone), emitted-arm tuning as follow-up.

## Tests / static analysis
- `test_reach_arch_compare.py` (3) + `test_reach_bc` green; ruff clean on the new/changed files.
- No CORE.YAML items touched (option 1 would need the `extract_joint_limits` CORE edit).

## Performance
Smoke wall 62 s (1 seed, both backbones, 150 epochs). RSS not captured (`psutil` absent); the
networks are tiny (16–29 k params, CPU) — far under the 16 GB cap. Full 5-seed run ≈ 5 min.

## Provenance
- `hymeko_rl/` untracked; new tracked file: none (profiles already committed in prior tasks).
- Platform: Windows 11, MuJoCo 3.9.0, Python 3.12. Seeds: smoke seed 0; saturation probe seeds 0–19.
- Measured: expert raw torque median 2608 / max 15154 N·m vs ±25 clip; 80 % saturation.
