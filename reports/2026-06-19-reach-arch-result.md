# Report — HSiKAN vs MLP behaviour-cloning comparison on the canonical reach arm

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ✅ **Delivered.** 5-seed comparison run, result JSON written, tests green. This is the
Kato deliverable: same task, same demonstrations, same observation — only the policy backbone differs.

## Result

5 seeds, 150 epochs, hidden 64, behaviour cloning from the DLS-IK expert on the
hymeko-emitted **4-DOF position-controlled** canonical arm (`data/robotics/reach_arm.hymeko`).

| backbone | n_params | reach median (m) | reach IQR | reach worst | untrained floor (m) |
|---|---|---|---|---|---|
| HSiKAN | 28 745 | 0.2401 | 0.0199 | 0.2610 | 0.463 |
| MLP    | 13 897 | 0.2262 | 0.0208 | 0.2338 | 0.452 |

Per-seed reach error (m):
- HSiKAN: 0.2296, 0.2346, 0.2401, 0.2429, 0.2610
- MLP:    0.2026, 0.2181, 0.2262, 0.2285, 0.2338

## Reading (measured / inferred, kept separate)

- **Measured:** both backbones learn — final EE error ≈ 0.23 m against an untrained floor ≈ 0.46 m
  (≈ 50 % reduction). Reproducible: each `(backbone, seed)` seeds policy init, BC training, and
  env RNG, so both face identical targets per seed.
- **Measured:** the MLP is marginally better (median 0.2262 vs 0.2401, a 1.4 cm gap) at **half**
  the parameters (13.9 k vs 28.7 k).
- **Inferred:** the 1.4 cm gap sits well inside both IQRs (≈ 2 cm), so on this arm the two
  backbones are **statistically indistinguishable** in reach accuracy; the MLP wins only on
  parameter count.
- **Not claimed:** no HSiKAN advantage on a 4-DOF serial chain. This is the *expected* outcome
  — the structural prior (message-passing over the kinematic hypergraph) is hypothesised to pay
  off on **redundant / branched** morphologies, not a short serial arm where a flat MLP over the
  same observation has enough capacity. The serial-arm case is the control, not the showcase.

## Files touched

- `hymeko_rl/reach_arch_compare.py` — **1 line.** Corrected the report's `arm` label from the
  stale `"anthropomorphic_arm (emitted, 6-DOF)"` to `"reach_arm (emitted, 4-DOF,
  position-controlled)"`, matching the actual factory default (`_REACH = reach_arm.hymeko`). The
  prior label would have mislabelled the deliverable geometry.
- `reports/2026-06-19-reach-arch-result.json` — **new**, the machine-readable result.
- `.gitignore` — `/docs/demo/galambos_scenario/` (collaborator sketch + private notes, local-only).

## Tests

- `pytest hymeko_rl/tests/test_reach_arch_compare.py test_reach_bc.py -p no:randomly` →
  **14 passed in 32.7 s.** The edit changes only an output string; no logic touched.

## Performance

- Smoke (1 seed, both backbones, 150 ep): 64.9 s. Full (5 seeds): **277.3 s** — within the
  ~5 min estimate (§11 reconciled).
- Peak RSS **not captured** — `psutil` is absent and adding it is a dependency change (§1). The
  networks are tiny (14–29 k params, CPU-only); RSS is far under the 16 GB cap. Same call as the
  2026-06-19 harness report; flagged, not silently dropped.

## CORE.YAML

None touched.

## §6.5 anti-patterns

None introduced (1-line label fix + report).

## Provenance

- `hymeko_rl/` untracked. New tracked-eligible file: the result JSON under `reports/`.
- Platform: Windows 11, MuJoCo 3.9.0, Python 3.12. Seeds 0–4. Expert: DLS-IK, position control.
- Robot: `data/robotics/reach_arm.hymeko` (4-DOF, axes Z/Y/Y/Z, ~0.64 m workspace), emitted MJCF
  via `emit_arm_mjcf(control_mode="position")`.

## Follow-up

- The showcase morphology (where the hypergraph prior should beat a flat MLP) is a
  **redundant/branched** arm, not this chain. Next RL track per the user: the **Galambos planar
  two-finger grasping** scenario (`docs/demo/galambos_scenario/`, local-only) — two planar elbow
  manipulators pulling a randomly-spawned disk into a fixed target zone. New MuJoCo env, plan-gated
  (§2) before code.
