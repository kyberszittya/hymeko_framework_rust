# Report — P-graph solving regimes (Strategy + Adapter)

**Date:** 2026-05-27
**Plan:** `docs/plans/2026-05-27-pgraph-regime-strategy/` (tex/pdf/tikz/mmd)
**Crate:** `hymeko_pgraph` (non-core) + `signedkan_wip` drivers (non-core)

## Summary

Replaced the `strict_no_excess: bool` flag threaded through MSG/SSG/ABB with a
first-class **`Regime`** Strategy trait (CLAUDE.md §7 Strategy/Adapter, §6.5
#1/#7). The canonical Friedler maximal structure is now the **general substrate**
for all P-graph problems; the no-excess refinement is a named strategy
(`NoExcess`). An **Adapter** (`from_strict_flag`) maps the legacy bool options to
regimes, so the change is **behavior-preserving** — the entire prior test suite
passes unchanged.

This also cleanly resolves the PNS-vs-HSiKAN tension surfaced earlier: PNS gets
the canonical regime by default (correct per Jean/the book), while the HSiKAN
architecture-search drivers now select `NoExcess` **explicitly** (restoring their
deliberate by-product-pruning, the measured +0.061 AUC steering on Bitcoin
Alpha) instead of relying on a global default.

## Design

`src/regime.rs` (new):
- `trait Regime { refine_maximal(p, units) -> units;  structure_admissible(p, sel) -> bool;  name() }`
- `Canonical` — identity refinement, admits all base-feasible structures (the general path).
- `NoExcess` — the no-waste filter (`refine_maximal`) + the strict no-excess leaf
  predicate (`structure_admissible`).
- `static CANONICAL/NO_EXCESS` singletons; `from_strict_flag(bool) -> &'static dyn Regime`.

The no-excess logic that was duplicated across `msg` (a filter) and `ssg` (a leaf
check) is now defined **once** in `NoExcess`. MSG/SSG/ABB gained `*_with_regime`
entry points; the bool APIs (`*_with_options`) delegate through the adapter.

### Composite regime — *mixing* canonical with specific refinements

`Composite` (Composite pattern over `Regime`) stacks several refinements in one
solve. Since the canonical reduction + composition is always the base, a
`Composite` expresses *"canonical + R1 + R2 + …"*:
- `refine_maximal` threads the unit set through every component and **iterates
  the whole stack to a combined fixpoint** (one component's removal can trigger
  another's, so order/one-pass don't matter).
- `structure_admissible` is the **conjunction** of all components.
- An empty composite ≡ `Canonical` (identity).

This is the mechanism for combining the general solver with one or more domain
refinements without a new flag or wrapper — each new refinement is an `impl
Regime` that composes for free. Tested: stacking removes the union of
refinements; empty = identity; admissibility is the AND.

### `CostDominance` — a second concrete refinement

To give `Composite` something real to mix, `CostDominance` (an
optimum-preserving search reduction) prunes a unit `u` when another surviving
unit `v` produces the **same outputs** (`outputs(v) == outputs(u)`) from a
**subset of inputs** at **no greater cost**, with a strict improvement
(cheaper, or strictly fewer inputs). Equal outputs make the `u→v` swap
output-neutral, so the cost-optimum is preserved under **any** regime — which is
why it composes *soundly* with `NoExcess` (a naive "superset of outputs"
dominance would be unsound under no-excess, as the swap could introduce excess).
Domination is transitive + strict, so a single pass is correct.

It preserves the *optimum*, not the full solution-structure set — a reduction for
ABB, not for enumerating every structure (documented on the type). HSiKAN-relevant:
prunes a dearer interchangeable architecture choice directly, by cost.

**Headline mix test** (`composite_cost_dominance_and_no_excess_mix`):
`Composite([CostDominance, NoExcess])` on a graph with a cost-dominated twin
*and* an excess-bearing unit prunes **both**, leaving only the cheap waste-free
unit — each refinement alone prunes only its own target. This is the concrete
demonstration of mixing canonical + two specific refinements in one solve.

## Files touched (all non-core)

- `src/regime.rs` — **new** (trait + Canonical + NoExcess + adapter + 3 unit tests).
- `src/msg.rs` — `maximal_structure_with_regime` (canonical base + `regime.refine_maximal`);
  `maximal_structure_with_options` is now the adapter. The inline no-excess filter
  was removed (moved to `NoExcess::refine_maximal`).
- `src/ssg.rs` — added `is_feasible_base` + `is_feasible_with_regime`; `is_feasible`
  is the adapter. The inline no-excess branch removed (moved to `NoExcess`).
- `src/abb.rs` — `solve_with_regime` (regime governs leaf feasibility);
  `SearchState<'r>` holds `&'r dyn Regime`; `solve_with_options` is the adapter.
- `src/lib.rs` — `pub mod regime`; re-export `Regime, Canonical, NoExcess,
  maximal_structure_with_regime`.
- `signedkan_wip/experiments/runs/{run_gomb_msg_sweep,run_cortical_msg_sweep,
  run_hsikan_msg_sweep}.py` — pass `--strict-no-excess` by default (the `NoExcess`
  regime), restoring HSiKAN's intended behaviour under the canonical-default engine.

No new public symbol removed; the bool APIs remain for back-compat.

## CORE.YAML items touched

**None.** `hymeko_pgraph` and `signedkan_wip` are non-core. No pinned-dependency change.

## Test results

`cargo test -p hymeko_pgraph --no-fail-fast` — **all green, 0 failed** (1
pre-existing ignored; lib has 37 tests including 8 `regime` unit tests covering
`Canonical`/`NoExcess`/`Composite`/`CostDominance` + the `Composite` doctest). The prior 129 tests pass **unchanged** (the
behavior-preserving proof: the bool APIs route through the regime adapter and
return identical results), plus 3 new `regime` unit tests (Canonical identity;
NoExcess drops a by-product producer + rejects excess; `from_strict_flag` mapping).
`book_validation.rs`, `ssg_decision_mapping.rs`, `relaxed_msg.rs` all green
without edits.

`cargo clippy -p hymeko_pgraph --all-targets -- -D warnings` — clean. Changed
files individually `rustfmt`-clean.

## Behaviour-preserving evidence

The two regimes reproduce the prior bool semantics bit-for-bit:
- `Canonical` ≡ prior `strict_no_excess: false` (the canonical fix from the
  earlier MSG report).
- `NoExcess` ≡ prior `strict_no_excess: true`.

Confirmed at the engine level on the HSiKAN witness fixture
`sweep_msg_byproduct_dominated`:
- `--strict-no-excess` (NoExcess) → `{cycle_topk_m4, model_h8, train_long}` cost 150
  (the documented HSiKAN pick that yields +0.061 AUC).
- default (Canonical) → `{cycle_topk_m4, model_h8, train_short}` cost 60.

## Performance

No algorithmic change — one `&dyn Regime` dispatch per MSG call and per ABB leaf
(static singleton, no allocation). MSG stays polynomial. `pgraph_bench`
`msg`/`abb`/`ssg_dm` unchanged within noise (`ssg_dm/example3_3` ≈ 34 ms, within
the 10% gate). $\ll 16$ GB.

## What this gains

- **One substrate, many regimes.** Canonical is the general P-graph solver;
  refinements are pluggable, named strategies — no boolean Cartesian product, and
  the no-excess logic is defined once instead of duplicated in `msg` + `ssg`.
- **HSiKAN is explicit and correct** under the new engine: it selects `NoExcess`
  by name, keeping its by-product-pruning architecture-search edge, fully
  decoupled from the canonical PNS default Jean needs.
- **Extensible:** a future HSiKAN-specific regime (e.g. dominance-aware pruning)
  is a new `impl Regime` with no change to core MSG.

## Update — CLI `--regime` syntax + HSiKAN run status (2026-05-27)

**CLI `--regime SPEC`** (done). Both the dump analysis core and the
`hymeko_pgraph_dump` binary now accept an explicit regime spec:
`canonical | no-excess | cost-dominance`, `'+'`-joined
(e.g. `--regime cost-dominance+no-excess`). Implementation: `dump.rs` gained
`analyze_{source,lowered}_with_regime` (regime-driven core); the bool
`*_with_full_options` are adapters; the bin parses the spec into component
strategies (single → used directly so its `name()`/`strict_no_excess` echo is
correct; multiple → `Composite`). `--strict-no-excess` is retained (≡
`--regime no-excess`); `--relaxed-msg` stays a deprecated no-op. JSON
`strict_no_excess` echo = `regime.name() != "canonical"` (preserves the HSiKAN
driver contract). 17 test binaries green; clippy clean.

Demonstrated on `data/hsikan/sweep_msg_byproduct_dominated.hymeko`:

| `--regime` | MSG units | strict_echo | ABB optimum |
|:--|--:|:--|:--|
| `canonical` | 8 | false | {m4,h8,train_short} = 60 |
| `no-excess` | 6 | true | {m4,h8,train_long} = 150 |
| `cost-dominance` | 5 | true | {m4,h8,train_short} = 60 |
| `cost-dominance+no-excess` (composite) | 3 | true | {m4,h8,train_long} = 150 |

**HSiKAN end-to-end run — blocked on this host (reported, not run).** Two
blockers: (1) the base Python env has no `torch`/`numpy` (CORE-pinned
`torch==2.4.1`, in the uv `ml` group; needs `uv sync --group ml`, ~GB); (2) more
fundamentally, CLAUDE.md §4 mandates the 16 GB RSS cap be enforced with
`systemd-run --user -p MemoryMax=16G` (cgroups v2) and **forbids `ulimit -v`** —
neither is available on Windows, so a *compliant* training run cannot be launched
here (§11 halt: resource cap not enforceable). The **engine-level architecture
selection** under each regime (the part the P-graph engine governs) IS
demonstrated above; the trained-model AUC deltas require training, which must run
on Linux/WSL under the cgroup gate (or with explicit, logged cap-override
authorization). The HSiKAN drivers are already wired to the `NoExcess` regime, so
that run is a launch-only step once on a capable host.

## Open issues / follow-ups

- **Run the HSiKAN sweeps** to observe end-to-end numbers under the explicit
  `NoExcess` wiring — a separate, env-/time-gated step (training mutates state;
  CLAUDE.md §11). The engine-level selection is verified; the trained-model
  deltas are not re-measured here.
- Pre-existing items unchanged: ABB `max_explored` silent-incumbent; crate-wide
  fmt drift in untouched modules.

## Provenance

- **Git SHA:** `9abfc3435f55f7443cb07bde4583a17126ac3fc1` (branch
  `feature/pgraph_engine`); working tree uncommitted, layered on the earlier
  (also uncommitted) MSG-canonical-fix and decision-mapping-SSG work.
- **Toolchain:** rustc/clippy 1.93.0; criterion 0.8.2.
- **Determinism:** regimes are pure; no RNG; assertions exact.
- **§6.5 anti-patterns:** none introduced — this change *removes* a boolean-flag
  anti-pattern and de-duplicates the no-excess logic.
