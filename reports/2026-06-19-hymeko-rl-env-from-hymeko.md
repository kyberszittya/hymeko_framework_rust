# Report — `ArmReachEnv.from_hymeko`: the RL env now runs on the canonical hymeko→mjcf arm

**Date:** 2026-06-19
**Author:** Aiko (agent), for Dr. Csaba Hajdu
**Status:** ✅ complete — the reaching env loads, controls, and is reached on the **emitted**
6-DOF arm built from `.hymeko`, alongside the hand-authored 4-DOF arm. Full `hymeko_rl` suite green.

## Summary
Follow-on to `reports/2026-06-19-hymeko-emit-kinematic-rerouting.md` (B-004/B-005). Now that
`hymeko emit -f mjcf` produces a correct articulated arm, the RL env is wired to consume it. This
realises the Kato "one source → three products" story end to end: a single `.hymeko` robot yields
the **MuJoCo scene**, the **kinematic hypergraph** (`HypergraphState.from_mjcf`), and the
**observation space** — no hand-authored scene in the loop.

The env was already largely DOF-agnostic (reads `nu`/`njnt`/Jacobian from the model). Two real
couplings had to be resolved, both non-CORE and confined to `hymeko_rl`:
1. **Under-specified bounds.** The emitted scene declares neither joint ranges nor actuator
   ctrlranges (`jnt_limited`/`actuator_ctrllimited` all `False`, ranges `[0,0]`). A naive load
   gives a degenerate action space `Box(0,0)` and `reset()` sampling `FK(0)` (target = home). The
   env now supplies sensible **fallback bounds** (torque ±25 N·m, joint ±1.5 rad) for any element a
   scene leaves unlimited — inert for `arm_world`, which specifies both.
2. **Scene/EE identity.** The emitted arm is 6-DOF with EE body `tool` (vs `arm_world`'s 4-DOF
   `flange_link`). Handled by a `mjcf=` scene-source override + an `ee_body` argument, fronted by a
   `from_hymeko(...)` factory that fixes `control_mode="torque"` (the emitted actuator type) and
   `ee_body="tool"`.

## Design
- **Bridge (`arm_world.py`).** `emit_arm_mjcf(hymeko_path, name)` shells out to the built CLI
  (`hymeko emit -f mjcf`) — the canonical path, since the PyO3 import resolver is still B-003-blocked
  and the CLI resolves imports. `_hymeko_cli()` locates `target/{debug,release}/hymeko[.exe]` and
  raises a clear `FileNotFoundError` if unbuilt (tests skip on that).
- **Bounds fallback (`arm_reach_env.py`).** `_bounds_with_fallback(limited, lo, hi, default)` — one
  pure helper reused for both joint ranges and actuator ctrlranges; replaces only the genuinely
  degenerate elements (`~limited | hi<=lo`). Strategy-style scene source via `mjcf=None` (→
  `make_arm_mjcf`) or an explicit string. `from_hymeko` is the typed entry point.
- No new algorithm: the IK/computed-torque expert is unchanged and already DOF-generic (DLS over
  `jacp[:, :nu]`); it handles the redundant 6-DOF arm directly.

## Files touched (all in the untracked `hymeko_rl/` package — no git baseline)
| File | Change |
|---|---|
| `hymeko_rl/env/arm_world.py` | +~55 — `emit_arm_mjcf`, `_hymeko_cli`, `_REPO`, imports |
| `hymeko_rl/env/arm_reach_env.py` | +~55/−6 — `_bounds_with_fallback` + constants, `mjcf=` override, robust ctrl/joint bounds, `from_hymeko` factory, `reset()` uses fallback range, refreshed docstring |
| `hymeko_rl/tests/test_arm_reach_from_hymeko.py` | +105 (new) — 5 tests |

**CORE.YAML items touched:** **none.** All changes are in `hymeko_rl` (not CORE; not in CORE.YAML).

## Tests
`pytest -p no:randomly`:
- **`test_arm_reach_from_hymeko.py` — 5 passed** (9.6 s):
  - `test_bounds_fallback_replaces_only_degenerate_elements` — pure unit (no CLI): only
    unlimited / inverted elements get the default; limited ones are preserved.
  - `test_emitted_mjcf_is_articulated` — bridge emits 6 joints, mixed axes (`1 0 0`,`0 1 0`),
    no `world` body (guards B-004/B-005 at the env layer).
  - `test_emitted_arm_loads_and_is_controllable` — 6 actions, obs `(7, 8)`, action bounds = the
    ±25 torque fallback, target non-trivial (joint-range fallback gives a real EE target).
  - `test_emitted_arm_expert_reaches` — 5 seeds; computed-torque expert closes the gap, **median
    final/initial distance ≈ 0.12** (asserted `< 0.4`, every seed monotone-better).
  - `test_emitted_and_authored_arms_share_the_env` — both arms behind one env class; provenance
    differs (`actuator_ctrllimited`: authored all-True/inert, emitted all-False/active).
- **Regression — 36 passed:** `test_arm_world` + `test_reach_bc` + `test_hypergraph_state`
  (21) and `test_policy` + `test_ppo` (15). The 4-DOF path is byte-for-byte behaviourally unchanged
  (`mjcf=None` default, fallback inert when bounds are specified).

Manual smoke (production-scale, §3): `ArmReachEnv.from_hymeko(anthropomorphic_arm.hymeko)`, 5 seeds ×
150 steps — per-seed final/initial ratios `{0.28, 0.59, 0.10, 0.12, 0.07}`.

## Static analysis
- `ruff check` on the 3 changed files → **clean**.
- `mypy` on the changed modules → only `mujoco` `import-untyped` (library ships no stubs; no
  project mypy config). Identical pre-existing note in the unchanged `hypergraph_state.py`; **not a
  regression** and not on any of my annotations.
- **§6.5 anti-patterns:** none. The scene source is a parametric override (config, not a class
  explosion — #1/#8 respected); the bounds fallback is a single shared helper, not duplicated per
  bound kind (#3).

## Performance
Not a perf change. `from_hymeko` adds one ~50 ms CLI subprocess at env construction (once). No hot
path, no budget. Step/expert cost is unchanged from the 4-DOF env modulo the larger `nv`.

## Dependencies
None added or removed.

## Open issues / follow-ups
1. **Joint limits + ctrlranges from `.hymeko` (the principled fix).** The fallbacks are an env-side
   safety net; the emitted scene *should* carry the limits the source declares (`limit ->
   joint_rev_limit`, `limit_effort`). Propagating them requires `extract_joint_limits` to follow the
   ref — a **CORE** edit in `hymeko_query/src/kinematics/kinematic.rs` (needs approval). Tracked under
   B-005's follow-up in `docs/BUGS.md`.
2. **Control modes for the emitted arm.** `from_hymeko` is torque-only (the emitted actuator type).
   Position/velocity would need the emitter to vary `<motor>`→`<position>`/`<velocity>` (or a
   post-process). Low priority — torque is the headline interface.
3. **BC/PPO at 6 DOF.** This wires + smokes the env; it does not retrain. A BC/PPO run on the
   emitted 6-DOF arm is the next compute step (the policy/networks size off the env spaces, so no
   code change is expected).

## Provenance
- Git: `hymeko_rl/` is untracked; this task added/edited the three files above. CLI built from
  `7d16ad0` + the (non-CORE) emit fixes from the prior report.
- Platform: Windows 11, MuJoCo 3.9.0 (Python), Python 3.12.
- Fixture: `data/robotics/anthropomorphic_arm.hymeko` (6-DOF, axes Z/X/Z/X/Y/Z, EE body `tool`).
- Seeds: env reaching probe used seeds 0–4; expert is deterministic given the seed.
