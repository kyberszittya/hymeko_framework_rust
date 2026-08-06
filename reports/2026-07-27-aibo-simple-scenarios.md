# AIBO Simple Scenarios (simulation) — SIM-01/02/03, gates A0–A3

**Date:** 2026-07-27 (JST)
**Branch:** `simulation/aibo-simple-scenarios-v1` (from `scenario/cip-aibo-v0` @ 95330d55)
**Worktree:** `../hymeko_aibo`  ·  **Highest gate: A3.**  ·  **SIMULATION-CERTIFIED, not hardware.**

---

## Summary

Continued the AIBO track in **simulation only** on the constructed 22-DOF ERS-1000
sim, validating simple HyMeKo/CIP scenarios and the embodiment adapter — *not*
hardware transfer. The one real gap from AIBO-2 (one-directional yaw) is **fixed
with a deterministic primitive** (no RL): all four gates pass.

## The fix — stable bidirectional yaw (localised, then corrected)

The old `SteeredTrotGait` turned stably only one way. **Localised cause:** it
*amplified the outer legs above nominal stride* (×1.45 at yaw 0.5) plus an
abduction bias — over-driving one diagonal and coupling into roll → tip. The
**reduce-inner** primitive (slow the inner/turn-side legs, never amplify, no
abduction) is symmetric by construction and turns **both** directions upright:

| yaw diagnostic (700 steps) | y+0.6 | y−0.6 | y+0.9 | y−0.9 |
|---|---|---|---|---|
| amplify-outer (old) | +1.6° up **−0.45** (flip) | −26° up 0.93 | −1.1° up **−0.44** | +27° up 0.80 |
| **reduce-inner (new)** | +9° up **1.00** | −15° up **1.00** | +14° up **1.00** | −15° up **1.00** |

## Gates (SIMULATION)

| gate | requirement | result |
|---|---|---|
| **A0** | on-axis FORWARD→DECELERATE→STOP→HOLD reproduced | ✅ dist 0.414 m, halted, held, upright, certificate passed |
| **A1** | stable left AND right yaw authority | ✅ +3.2 °/s (left) and −3.0 °/s (right), both upright |
| **A2** | turn→align→stop passes from both yaw directions | ✅ ±30° both: reached, oriented, halted, held, no fall |
| **A3** | approach→align→stop from ±10°/±20° panel | ✅ all 4: reached (dist ≤0.42 m), oriented (9–16° < 25° tol), upright 1.00, certificate passed |

Per-run logging (initial pose, realised forward speed + yaw rate, target-distance
error, orientation error, stopping steps, final body speed, stability, certificate)
is in `aibo_sim_gates.json`.

## Scenarios

- **AIBO-SIM-01** FORWARD → DECELERATE → STOP → STABLE HOLD (on-axis).
- **AIBO-SIM-02** TURN LEFT / TURN RIGHT → ALIGN → STOP (±30°, both directions).
- **AIBO-SIM-03** APPROACH → ALIGN → DECELERATE → STABLE STOP from ±10°/±20° offsets.

All realized by the **unchanged CIP-AIBO-01 contract** + the CIP-0 runtime; only the
scenario-side yaw primitive and the adapter's pursuit control changed. **Shared CIP
core unchanged.** The external certificate now also requires orientation-within-
tolerance at stop (scenario-side certificate; core untouched).

## Files touched (all NEW/scenario-side, non-core)

```
scenarios/aibo/locomotion_gait.py   (SteeredTrotGait -> reduce-inner bidirectional)
scenarios/aibo/adapter.py           (goal-bearing offset + bidirectional pursuit + orientation)
scenarios/aibo/certificate.py       (success now requires 'oriented')
scenarios/aibo/run_aibo_sim.py      (SIM-01/02/03 harness + A0-A3 gates + plot + gif)
tests/test_aibo_bidirectional_yaw.py (locks both-direction upright turning)
reports/2026-07-27-aibo-simple-scenarios/{aibo_sim_gates.json, aibo_sim_paths.png, aibo_sim_approach.gif}
docs/plans/... (n/a)
```

## Tests + lint

- `pytest tests/ hymeko_control/conformance/tests` — **22 passed** (5 AIBO conformance
  + 2 bidirectional-yaw + 15 core). No regression from the gait change.
- `ruff check scenarios/aibo tests/...` — **all pass.**

## Graphical output (§9)

`aibo_sim_gates.json` (numerical), `aibo_sim_paths.png` (dist vs heading-error per
run), `aibo_sim_approach.gif` (960×720 approach-align-stop, gitignored).

## Final verdict — the four distinct claims

1. **CIP conformance:** ✅ the CIP-AIBO-01 contract + CIP-0 runtime drive all three
   scenarios with reward-independent certificates and no core change.
2. **Simulation embodiment success:** ✅ **A3** — bidirectional yaw + approach-align-stop
   from ±10°/±20° all certified on the 22-DOF ERS-1000 **simulation**.
3. **Hardware status:** **not attempted here** — no physical AIBO / SDK (frozen on
   `hardware/aibo-command-response-audit-v0` = `AIBO_HARDWARE_INTERFACE_UNAVAILABLE`).
4. **Remaining transfer prerequisite:** a physical ERS-1000 + SDK/ROS + operator
   floor to run the frozen H1→H2 hardware audit, and confirm the simulated
   command→motion contract (esp. bidirectional yaw + stopping) holds on the robot.

**Simulation is labeled as simulation throughout.** No RL. CORE.YAML items touched: none.
