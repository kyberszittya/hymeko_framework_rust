# Committed pick-place clearance diagnostics module + v1 smoke (2026-07-07)

## Summary

Promoted the ephemeral `scratchpad/pick_clearance.py` (lost across the day boundary) into a **committed, reusable**
evaluation module `hymeko_rl/eval/pick_clearance.py`, and verified it with a **v1 4-episode smoke** that reproduces
the frozen `v1_dirty` clearance signature. This is the smallest step before the v2 clearance gate; it changes no
trajectory, reward, or training and runs no learning.

## Files touched

| file | change |
|---|---|
| `hymeko_rl/eval/pick_clearance.py` | **new** (+~300 lines) — `ClearanceMetric` (RolloutMetric), `run_clearance`, `aggregate`, `gate_verdict`, `write_outputs`, CLI. Reuses `LiftPlaceMetric`/`eval_metric`/`results_to_csv`/`expert_action_fn`/`fanuc_pick_env` (no re-implementation). |
| `hymeko_rl/tests/test_pick_clearance.py` | **new** (+~95 lines) — 7 unit + 1 bounded integration test (§3 coverage). |
| `reports/figures/pick_place_clean_expert/v1_clearance_smoke.{json,csv,png}` | smoke artifacts (data, not code). |

**No existing implementation code was modified.** `_expert_action_v1/_v2`, reward, BC/DAgger, coin-collab v2b, and
the parallel-track env WIP were **not** touched. A copy of the module was placed on kato15 (new remote file).

## CORE.YAML items touched

None.

## Test results

| gate | result |
|---|---|
| `ruff check` (module + test) | **clean** |
| `mypy --strict` (module) | **clean** — no issues |
| `pytest -p no:randomly` (test file) | **8 passed** in 4.37 s (incl. the v1 integration test) |
| v1 smoke (4 ep, seeds 50000–50003, horizon 620) | reproduces dirty signature; gate **FAIL** (expected) |

### v1 smoke signature (reproduces the frozen forensics)

| seed | lift | place | first finger↔table | first over-object | min transit clearance | forbidden pre-object | transit contact |
|---|---|---|---|---|---|---|---|
| 50000 | 1 | 1 | 51 | 225 | −0.0258 | True | 0.446 |
| 50001 | 1 | 0 | 51 | 149 | −0.0248 | True | 0.333 |
| 50002 | 1 | 1 | 53 | 161 | −0.0252 | True | 0.569 |
| 50003 | 1 | 1 | 52 | 217 | −0.0233 | True | 0.468 |
| **agg** | **1.0** | **0.75** | — | — | **min −0.0258** | **rate 1.0** | **0.454** |

Matches `reports/2026-07-06-pick-place-clearance-forensics.md`: first strike ~step 51 (vs ~275 over-object),
~2.6 cm penetration, ~48 % transit contact, lift 1.0 / place ~0.75–0.875. Gate verdict FAIL on criteria 1–3 —
the correct dirty-baseline reading; **v1_dirty numbers are not overwritten** (this is a fresh diagnostic, quoted
separately).

## Performance

Trivial: 4 episodes × 620 steps ≈ a few seconds wall, well under any budget; no RSS concern (single MuJoCo env).

## kato15

The module **imports and resolves cleanly** on kato15 (`~/envs/hymeko/bin/python`, mujoco 3.10, `MUJOCO_GL=egl`).
The remote run is blocked only because kato15's `hymeko_rl/env/pick_place_env.py` is **behind local** — it lacks the
uncommitted `expert_version` param (a parallel-track WIP file). That file was **not** synced (do-not-touch
parallel-track WIP). The authoritative smoke ran locally.

## Anti-patterns / waivers

No §6.5 anti-patterns introduced. No new `#[allow]`/`# type: ignore`/`# noqa`. Error paths: `run_clearance` raises
`ValueError` on bad `version`/`episodes`; `mj_geomDistance` unavailability degrades gracefully to unmeasured
clearance (documented).

## Next (blocked)

The v2 gate command is ready but **blocked until the v2 expert is implemented and passes**:

```
python -m hymeko_rl.eval.pick_clearance --version 2 --episodes 32 --seed0 50000 \
  --out reports/figures/pick_place_clean_expert/v2_clearance_gate
```
