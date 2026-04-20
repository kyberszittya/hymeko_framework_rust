# Known issues

**Last updated:** 2026-04-21

A working list of real bugs and gaps we've identified, with reproducers and current understanding. Issues land here when they are concrete enough to act on and not yet fixed; when they're fixed, they move to the changelog (`docs/changelog/`) with the commit reference.

---

## Open

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
