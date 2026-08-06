# Known issues

**Last updated:** 2026-06-18

A working list of real bugs and gaps we've identified, with reproducers and current understanding. Issues land here when they are concrete enough to act on and not yet fixed; when they're fixed, they move to the changelog (`docs/changelog/`) with the commit reference.

---

## Open

> B-003/004/005 surfaced 2026-06-18 while building the `hymeko_rl` robot-RL line (the
> Kato collaboration). All three are in the **hymeko → mjcf → obs** path — the pipeline
> that turns a `.hymeko` robot into a usable simulation. None block Phase 1/2 (worked
> around via the articulated `arm_world` MJCF), but all three must be fixed for the scene
> and the observation hypergraph to come *canonically* from one `.hymeko` source.

### B-005: MJCF emitter drops the per-joint axis (all joints emitted on Z)

**Component:** `hymeko_formats/src/transforms.rs` — MJCF emitter, joint-axis emission (likely shared with the URDF/SDF emitters — check).
**Severity:** High — emitted robots are kinematically **degenerate**: all joints colinear on Z, so the end-effector has zero workspace and *cannot perform any reaching/manipulation task*.
**First observed:** 2026-06-18, `hymeko_rl` Phase-1 reaching (the emitted arm's EE would not move).

**Symptom.** Every joint in the emitted MJCF has `axis = [0 0 1]` regardless of the source axis; EE position is invariant to joint angles (workspace spread = 0).

**Root cause (PINNED 2026-06-19 — the template path, NOT the model extractor).** There
are *two* emit paths: (a) the **model path** `hymeko_query::kinematics::extract_kinematic_model`
→ `hymeko_formats` `emit_mjcf`/`generate_urdf`/`generate_sdf`, and (b) the **template path**
the CLI uses (`emit`/`compile`/`transform` → `render_from_templates` over `transforms/<name>/`).
The **model path is CORRECT** — it reads each joint's `AXIS_*` arc; the new regression test
`hymeko_query/tests/test_anthropomorphic_generation.rs::per_joint_axes_match_the_source_b005`
**passes** (j0=Z, j1=X, j2=Z, j3=X, j4=Y, jtool=Z). The bug is entirely in the **template
path**: `transforms/urdf/template.urdf.xml`, `transforms/sdf/template.sdf.xml` literally
hardcode `<axis xyz="0 0 1"/>`, and `transforms/mjcf/template.mjcf.xml` is worse — it emits
flat bodies with **no `<joint>` elements at all** (which is also B-004). So the CLI's static
templates never read the joint axis. (The earlier "`*_faithful.mjcf` is all-Z" was the
template path too.)

**Fix (recommended).** Route the CLI `emit -f {mjcf,urdf,sdf}` to the **model-based**
emitters (`hymeko_formats::{emit_mjcf, generate_urdf, generate_sdf}`), which already extract
axes correctly — retiring the static templates for these formats. Alternatively make the
templates dynamic (a per-joint loop binding the axis query), but the model emitters already
exist and are correct, so routing is cleaner. Both are **non-CORE** (CLI + `hymeko_formats`
+ template data). The model extractor (`hymeko_query`, CORE) needs **no change** — confirmed
byte-unchanged + the regression test guards it.

**Reproducer.**
```bash
python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('demos/hero/out/anthropomorphic_arm_faithful.mjcf')
print('emitted axes', m.jnt_axis.round(0).tolist())          # all [0,0,1]
d=mujoco.MjData(m); ee=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'tool'); pts=[]
for s in range(4):
    r=np.random.default_rng(s); d.qpos[:]=r.uniform(m.jnt_range[:,0],m.jnt_range[:,1])
    mujoco.mj_forward(m,d); pts.append(d.xpos[ee].copy())
print('EE spread', (np.array(pts).max(0)-np.array(pts).min(0)).round(3).tolist())  # [0,0,0]
"
grep AXIS_ data/robotics/anthropomorphic_arm.hymeko          # source: AXIS_Z/X/Z/X/Y/Z
```

**Workaround (no longer needed for emit).** `hymeko_rl` Phase-1 used the hand-authored `hymeko_rl/env/arm_world.py` arm (mixed Z/Y/Y/Z, EE spread ~0.64 m).

**✅ RESOLVED 2026-06-19.** The model extractor was *never* wrong — it reads each joint's
`AXIS_*` arc correctly (regression test `per_joint_axes_match_the_source_b005` passes). The bug
was the **CLI dispatch** (non-CORE): `emit`/`compile` routed kinematic formats through the static
`transforms/<fmt>/` templates, which **hardcode** `<axis xyz="0 0 1"/>`. Fix: `emit -f {mjcf,urdf,sdf}`
now routes kinematic formats to the model-based emitter (`hymeko_cli/src/main.rs`). `emit -f urdf|sdf`
now emit the source's mixed axes (Z/X/Z/X/Y/Z). MJCF needed two further `emit_mjcf` fixes to *load*:
the root `world` frame now maps to MuJoCo's implicit `<worldbody>` (was a duplicate-name collision),
and `<inertial>` now carries the required `pos`. Net: `hymeko emit -f mjcf <arm>` yields a loadable,
articulated arm (nq=6, EE workspace ~1.2×0.9×0.8 m). Report:
`reports/2026-06-19-hymeko-emit-kinematic-rerouting.md`.

**Follow-up (minor, open).** The emitted joints are **unlimited** — `extract_joint_limits` looks for
direct `limit_lower`/`limit_upper` children, but the fixture declares `limit -> joint_rev_limit` (a
*ref* to a shared limit node), which the extractor does not follow. The arm articulates fine without
ranges; following the ref would add joint limits to the emitted scene. Low priority.

---

### B-004: default `emit -f mjcf` actuator references a non-existent joint target

**Component:** `hymeko_formats/src/transforms.rs` — MJCF emitter, actuator block (default `emit -f mjcf` path; the `*_faithful` transform path is correct).
**Severity:** High — the emitted MJCF fails to load in MuJoCo (hard error at model compile).
**First observed:** 2026-06-18, `HypergraphState.from_hymeko` (CLI emit → MuJoCo).

**Symptom.**
```bash
python -c "import mujoco; mujoco.MjModel.from_xml_path('demos/hero/out/anthropomorphic_arm.mjcf')"
# ValueError: Error: unknown transmission target 'j0' for actuator id = 0
```
The `*_faithful.mjcf` variant loads cleanly (6-DOF).

**Reproducer.**
```bash
./target/debug/hymeko.exe emit -f mjcf data/robotics/anthropomorphic_arm.hymeko -n robot > /tmp/arm.mjcf
python -c "import mujoco; mujoco.MjModel.from_xml_string(open('/tmp/arm.mjcf').read())"   # same error
```

**Root cause (hypothesis).** The default emitter's `<motor … joint="j0">` actuators reference a joint name not present as a `<joint name="…">` in the emitted body — an actuator↔joint name/binding mismatch, or `j0` maps to a *fixed* joint (no DOF, not a valid transmission target). `data/robotics/anthropomorphic_arm.hymeko` has `@j_fix` (fixed) plus `@j0..@jtool` (rev); the actuator emit likely binds to a name the body section did not emit as a free joint.

**✅ RESOLVED 2026-06-19 (same root cause and fix as B-005).** The mismatched actuators were the
**static MJCF template** `transforms/mjcf/template.mjcf.xml`: it emitted flat bodies with **no
`<joint>` elements** plus motors referencing joint names that were never declared. The CLI now
routes `emit -f mjcf` to the model-based `emit_mjcf` (`hymeko_cli/src/main.rs`), which emits nested
bodies with matched `<joint name>` and `<motor joint=…>` pairs and skips the fixed joint. Guarded by
`mjcf_emit_is_loadable_and_articulable_b004_b005` (asserts the 6 hinge joints are present and the
scene is well-formed). The legacy `transforms/mjcf/` template is now unused by the CLI for robots.

---

### B-003: PyO3 `load_file` import resolver fails on `@"…"` imports

**Component:** `hymeko_py` PyO3 binding — `PyHypergraphEngine.load_file` import resolution (the `hymeko` CLI resolver is unaffected).
**Severity:** Medium — blocks obtaining the canonical `.hymeko` star-expansion (`compile_star_expansion`) from Python for any robot file that uses `@"…";` imports; CLI-based emit (MJCF/URDF/…) is unaffected.
**First observed:** 2026-06-18, wiring `hymeko_rl.HypergraphState` (the RL obs-hypergraph bridge).

**Symptom.** Loading a robot `.hymeko` that imports a sibling meta fails with an IO "file not found" for a file that *exists*.

**Reproducer.**
```bash
python -c "import hymeko; hymeko.PyHypergraphEngine().load_file('data/robotics/anthropomorphic_arm.hymeko')"
# SyntaxError: Compile error: Io(IoDiag { op:"read", path:"data\\robotics\\meta_kinematics.hymeko", err: file not found })
ls data/robotics/meta_kinematics.hymeko                                  # the file IS there
./target/debug/hymeko.exe inspect data/robotics/anthropomorphic_arm.hymeko   # CLI resolves the same import fine
```
`chdir` to the source dir does not help (it then reports the bare name not found), so it is not a simple cwd issue.

**Root cause (hypothesis).** The binding's `load_file` passes the wrong base directory to the import resolver; the CLI `compile`/`inspect` path constructs correct absolute import paths (its `inspect` prints `Imports (namespace -> file)` with absolute `\\?\…` paths). The defect is in the binding's import-base wiring, not the resolver core.

**Workaround.** Route through the CLI (`hymeko emit -f mjcf …`) and derive the structure from the emitted artifact — `hymeko_rl.HypergraphState.from_mjcf` does exactly this — or use self-contained `.hymeko` files with no imports.

**Investigation plan.** Compare the resolver root passed in the PyO3 `load_file` vs the CLI `compile` path; set it to the source file's parent directory in the binding. Add a Python smoke test loading an imported robot file once fixed.

**Why not yet fixed.** The RL bridge is unblocked via the MJCF path; the canonical star-expansion route is a refinement, not a Phase 1/2 blocker.

---

### B-002: `Verdict::t` semantics ambiguous (settled time, not observation time)

**Component:** `hymeko_monitor/src/monitor/mod.rs::Verdict`, doc-only.
**Severity:** Low — no incorrect behaviour, but a foot-gun for first-time users writing test assertions or downstream integrations against the monitor.
**First observed:** 2026-04-21 while writing `tests/stl_kinematic.rs`.

**Symptom.** A first-pass test threshold comparing verdicts to "before / after the violation at t=12.30 s" used `v.t < 12.0` for "before" and `v.t > 12.35` for "after". The "before" assertion failed: at the moment the verdict crossed into negative robustness, `v.t` was 11.30, not 12.30 — so the negative verdict was being *recorded as a pre-violation sample*, then asserted to be positive, which it wasn't.

**Reproducer.** `cargo test -p hymeko_monitor --test stl_kinematic` with the original `< 12.0` threshold (now corrected) fails with:
```
pre-violation robustness should be positive, got -0.86
```

**Root cause.** The `Verdict::t` field is the *settled time* `t* = observation_time − formula.horizon`, not the observation time. For a formula with horizon 1.1 s, a violation observed at `τ = 12.30` first appears in the verdict at observation time 12.30 with `v.t = 11.20` — and stays visible to the outer `Always` at settled times in `[11.20, 12.30]`. The current `Verdict` struct doc says only "the timestamp at which this verdict is reported (i.e., the time whose robustness the verdict describes — typically trailing the most recent sample by the formula horizon)", which is technically correct but easy to misread as "the observation time at which we got this verdict."

**Workaround.** Compute the violation-visible window from the formula's horizon explicitly when writing assertions: for an injected violation at observation time `τ_v` with formula horizon `H`, the verdict's settled time `v.t` falls in the danger zone iff `v.t ∈ [τ_v − H, τ_v]`. Test thresholds for "safely pre" / "safely post" should use `< τ_v − H − ε` and `> τ_v + ε` respectively. The corrected `tests/stl_kinematic.rs` uses this pattern for a three-point pre/during/post assertion.

**Investigation plan.** Two non-mutually-exclusive options:

1. **Doc-only fix (small):** rewrite the `Verdict::t` docstring in `hymeko_monitor/src/monitor/mod.rs` and the corresponding section in `SPEC.md` to spell out the settled-time semantics with a worked example (the kinematic-arm violation). Add a doctest on `Verdict` showing the time-arithmetic. Cross-reference from `Monitor::verdict`.
2. **API fix (larger):** rename `Verdict::t` to `Verdict::settled_t`, optionally adding a `Verdict::observation_t` field set by `observe()` and computed as `settled_t + monitor.horizon()`. This is a breaking API change but eliminates the ambiguity at the type level. Worth doing before v1.0 ships externally; not urgent for v0.1.

**Why not yet fixed.** The integration test now passes; the foot-gun is documented in this entry and in the corrected test code. Doc fix is a 10-minute task scheduled with the next round of `hymeko_monitor` polish; API rename should wait for the v1.0 cut so it doesn't churn the v0.1 RV-paper text mid-stream.

---

### B-001: Resolver stack overflow on dense `highArityFixedPool` fixtures

**Component:** `hymeko_core/src/resolution/{intern_pass,resolve}.rs`
**Severity:** Medium — blocks bench-compiling the asymptote-witness fixtures, but the witness itself is closed-form (no compile required) so the empirical claim still holds.
**First observed:** 2026-04-21 while building the Prop 4 asymptote witness.

**Symptom.**
```
[1/1] bench: hap_n200_m200_d2 (200 V, 200 E, d̄=2.00)
thread 'main' (...) has overflowed its stack
fatal runtime error: stack overflow, aborting
```

**Reproducer.**
```bash
python3 scripts/scaling/generate_fixtures.py --out scripts/scaling/fixtures
./target/release/bench_scaling \
    --fixtures scripts/scaling/fixtures \
    --out /tmp/x.csv --reps 1 --warmup 0 \
    --family highArityFixedPool --max-size 10000
```

**Bisection so far.**
- m = 10, 50, 100, 150 over n_pool = 200 at d = 2: **succeed**.
- m = 200 over n_pool = 200 at d = 2: **crash**.
- The structurally identical `highArity` fixture `ha_m200_d2` (also 200 V, 200 E, arity 2, only different RNG-chosen edge participants) **succeeds**. This rules out raw fixture size as the cause.
- Renaming the root decl on a working `ha_m200_d2.hymeko` to `hap_n200_m200_d2.hymeko` and re-benching: **succeeds**. Confirms the crash is not name-driven.
- Same crash with seed = 0, seed = 42, seed = 999. Not seed-dependent.

**Hypothesis.** A topology-dependent recursive walk in the resolve / lower passes blows the default 8 MB main-thread stack on certain dense edge-graph topologies. Likely candidates:
1. A walk that follows `+`-incidence → `-`-incidence chains and recurses without an explicit cycle-break; on a sufficiently dense random graph these chains can be long.
2. The lalrpop-generated parser shifting on a deeply right-recursive grammar production for the hyperedge-incidence list (less likely, since `gen_high_arity` produces the same shape and parses fine).

**Workaround for the witness.** The Prop 4 asymptote witness is computed from fixture parameters (`n_pool`, `m`, `d̄`) via the closed-form `(n+m)/(m·d̄)`. The witness test does not require the fixtures to compile; the asymptote is a structural property of the fixture family parameters, not of any compiled IR. The figure (`storage_asymptote.pdf`) is built from the same manifest parameters via `scripts/scaling/emit_storage_asymptote.py` — no bench data needed.

**Investigation plan when prioritised.**
1. Spawn `compile_fresh` in a thread with `Builder::new().stack_size(64 << 20).spawn(...)`. If the crash goes away, it's pure stack depth. If it persists, it's infinite recursion.
2. With stack-bump in place, instrument the resolver to count recursion depth per pass (`intern_pass::lower_*`, `resolve::*`, `const_resolve::substitute_in_*`). The pass with depth ∝ |E| or |V| at crash time is the culprit.
3. Add an explicit cycle-break or convert the recursive walk to an iterative work-list once located.

**Why deferred.** The asymptote witness for Proposition 4 was the immediate goal, and it is delivered without needing this bug fixed. The bug should be fixed before the journal submission so the bench can also include the `highArityFixedPool` family in the runtime measurements (currently it only contributes to the storage-overhead figure).

---

## Recently fixed (kept here briefly for reference; full history in `docs/changelog/`)

### F-001 (FIXED 2026-04-20): MJCF emitter O(|J|²) recursion

**Component:** `hymeko_formats/src/transforms.rs::emit_mjcf_body`
**Resolution:** Replaced per-recursion-level `iter().find(...)` and `iter().filter(...)` calls with three pre-built `HashMap` indices (parent→children, child→incoming-joint, name→link). 15-line refactor; output byte-equal on all 175 regression tests.
**Result:** Power-law exponent on the chain/tree sweep dropped from $\hat{b} = 1.25$ (CI [1.15, 1.36], super-linear) to $\hat{b} = 0.97$ (CI [0.87, 1.07], linear). At $|V|=5000$ on the tree family, MJCF wall-clock dropped from 161 ms to 9.6 ms (17× faster).
**Pre-fix CSV preserved at:** `scripts/scaling/scaling_results_pre_mjcf_fix_2026-04-20.csv`.

### F-002 (FIXED 2026-04-21): Paper §VI-F overclaim about ρ on `highArity` family

**Component:** `paper/{smc2026,arxiv_v1}/sections/07_eval_scaling.tex`
**Symptom.** Original text claimed "$\rho$ drops below $1.1$ at $\bar{d}=10$ and approaches unity for $\bar{d} \geq 20$." Actual computed values from the fixture manifest: $\rho \approx 1.60$ at $\bar{d}=10$, $\rho \approx 1.55$ at $\bar{d}=20$, $\rho \approx 1.52$ at $\bar{d}=50$. The claim was numerically false on the actual data.
**Root cause.** The `highArity` generator uses $n_v = \max(d{+}1, md/2)$, growing $n$ linearly with $d$ and keeping the bound $\beta = (n+m)/(m\bar{d})$ at $\approx 0.55$ across the swept range. The asymptote claim in the proposition body is mathematically true but unwitnessed by this fixture family.
**Resolution.** Both paper trees rewritten to honestly describe the plateau; the asymptote claim now backed by the new `highArityFixedPool` family (`docs/storage_overhead_asymptote.md`).
**Caught by.** Writing the witness test for the original fixtures and discovering the asserted thresholds didn't pass.

---

## Out-of-scope: documented competitor limitations

These are not HyMeKo bugs but published findings about the standard tooling that we leveraged in §VI-F:

### MuJoCo: URDF importer recursion-depth limit on long chains

`mujoco.MjSpec.from_file()` raises `RuntimeError: Caught an unknown exception` on serial-chain URDFs with $|V| \geq 2000$ links. Tree variants (branching factor 3) succeed because depth stays at $\log_3(n) \approx 8$.

Logged in: `paper/{smc2026,arxiv_v1}/data/failures.json`.

### `gz sdf -p`: URDF→SDF converter has $\sim O(s^{1.8})$ scaling

Log-log fit on chain $|V| \in [2, 5000]$: exponent $\hat{b} = 0.59$ on the median wall-clock vs $|V|+|E|$ axis (the apparent sub-linearity is subprocess-startup dominance at small $|V|$; the *algorithmic* exponent on large-fixture data alone exceeds 1.5).

At $|V| = 5000$: gz sdf takes $\sim 54$ s; HyMeKo's SDF stage takes $\sim 30$ ms.

---

## How to add an issue

1. Reproduce minimally; record the exact command-line and expected vs actual output.
2. Bisect to narrow the trigger: change one variable at a time, log when the symptom flips.
3. Hypothesis section: state your current best understanding of the root cause, with the evidence you have.
4. Workaround section: state what *does* work, so other work isn't blocked.
5. Investigation plan: concrete steps the next person can take.
