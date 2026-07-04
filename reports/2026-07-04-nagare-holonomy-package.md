# NAGARE: holonomy package promotion, FD-tested projection kernel, frozen fixture, post-fusion parity re-run

Created-at: 2026-07-04 16:07 JST
Plan: `docs/plans/2026-07-04-nagare-holonomy-package/plan.{tex,pdf,tikz,mmd}` (created 15:40, ETA ≈18:15; finished 16:10 — under estimate)

## Summary

Executed the four-step queue from
`reports/2026-07-02-nagare-holonomy-learning-repo-report.md`, sequentially:

1. **Promoted the holonomy learning stack out of the examples into a stacked
   package.** New layering: `ops/project_alpha_mix.rs` (generic alpha-mixed
   subspace projection kernel, forward + closed-form backward for both
   `grad_x` and `grad_basis`) → `src/holonomy/` subsystem package
   (`datasets` / `features` / `pooling` / `projection` / `metrics` /
   `learner`, mirroring the planned `nagare-holonomy-learn` extraction
   layout) → examples reduced to CLI + JSON orchestration shells. The
   backprop-like `HolonomySetNet` baseline stays example-local by design.
2. **Finite-difference tests** for the projection kernel
   (`tests/project_alpha_mix_fd.rs`): central differences vs the analytic
   `grad_x` and `grad_basis`, including alpha=0 identity, alpha=1 pure
   projection, a zero-row basis, and the production 6×28 holonomy shape.
3. **Frozen seed-53 fixture** (`tests/fixtures/moons_spiral_xor_seed53.txt`):
   FNV-1a-64 content hashes over the exact f32 bit patterns of all six
   benchmark datasets (3 tasks × train/test, compare-example seed offsets),
   plus hex previews. `rand`/generator drift now fails loudly. Deliberate
   regeneration via `cargo test --test holonomy_fixture -- --ignored`.
4. **Post-fusion parity re-run** (both engines, same day, same host): the
   parity example now uses `fused_entropy_update_forward`. While doing this,
   a real defect was found and fixed: **the fused kernel was single-threaded**
   while the materialised path it replaced ran through the rayon-parallel
   `linear_forward` — the fused op was 3× *slower* (~300 µs/sample) until the
   forward was parallelised over rows (bit-identical per row; no cross-row
   reduction in the forward).

## Regression gate (pure-move proof)

`entropy_pool_learning_compare` re-run at the production config
(seed 53, 192/96/32 points, 50 epochs) after the refactor **and again after
the rayon change**: all metric fields of the JSON are **bit-identical** to the
committed `reports/2026-07-02-fitted-projection-gate-holonomy-ablation.json`
(timing fields exempt). The promotion changed no arithmetic.

Bonus determinism check: the PyTorch fixture regenerated today is
**byte-identical** to the frozen
`reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-fixture.txt`
(torch 2.12.0+cu132, seeded).

## Parity results (2026-07-04, same-day both engines)

Fixture: n=96 samples, 48 points, hidden=32; 300 repeats after 20 warm-ups;
logit parity ≤ 1.05e-7 in every run. Figure:
`reports/2026-07-04-nagare-fused-parity.png`.

| engine / variant | moons | rings | xor | alloc/fwd | peak RSS |
|---|---:|---:|---:|---:|---:|
| PyTorch 2026-07-01 (report) | 60.5 | 55.5 | 62.4 | 4.93 MB | 663 MiB tree |
| Nagare unfused 2026-07-01 (report) | 94.6 | 93.2 | 91.5 | 4.81 MB | 11.0 MiB |
| PyTorch 2026-07-04 re-run | 38.4 | 43.2 | 42.6 | 4.93 MB | 636 MiB tree |
| Nagare fused serial (defect) | 300.6 | 319.9 | 301.0 | 2.43 MB | 9.0 MiB |
| Nagare fused+parallel, run A | 83.7 | 81.7 | 79.3 | 2.43 MB | — |
| Nagare fused+parallel, run B | 67.7 | 64.2 | 64.6 | 2.43 MB | 9.2 MiB |

(medians, µs/sample; runs A/B are the same binary — the spread is host
variance, reported honestly rather than cherry-picking B.)

**Measured:** fusion + row-parallelism cuts Nagare's forward ~15–30% vs the
07-01 unfused numbers and halves per-forward allocation (4.81→2.43 MB);
parity is unchanged. **Also measured:** PyTorch on today's quiet host is
faster than its own 07-01 numbers, so the honest same-day verdict is that
**PyTorch remains ~1.5–2× faster on this unbatched example shape**; the
07-01 caveat "smaller but not faster yet" is *narrowed* (was 1.5–1.7× deficit
vs same-day PyTorch, now ~1.6× mid-spread with far lower allocation), **not
closed**. **Hypothesis (not yet isolated):** the remaining gap sits in the
serial `global_pool` loops and per-row scalar (non-SIMD) accumulation vs
MKL's vectorised GEMM; a `cargo flamegraph` profile is the next
discriminating step before any further kernel work (§3: no optimization
without a profile). The positive Chebyshev-deploy result (Nagare 1.7–1.9×
faster, 5 MiB vs 553 MiB) is unaffected by today's work.

## Files touched

New (library + tests, ~980 LOC):

- `hymeko_nagare/src/ops/project_alpha_mix.rs` (+201, kernel + 3 in-module tests)
- `hymeko_nagare/src/holonomy/{mod,datasets,features,pooling,projection,metrics,learner}.rs` (+7 files, ~700)
- `hymeko_nagare/tests/project_alpha_mix_fd.rs` (+140, 4 FD tests)
- `hymeko_nagare/tests/holonomy_fixture.rs` (+130, 1 check + 1 ignored writer)
- `hymeko_nagare/tests/fixtures/moons_spiral_xor_seed53.txt` (+8 lines)

Modified:

- `hymeko_nagare/src/lib.rs`, `src/ops/mod.rs` — module registration + re-exports.
- `hymeko_nagare/src/ops/fused_entropy_update.rs` — forward parallelised over rows (regression fix; per-row order unchanged → bit-identical results).
- `hymeko_nagare/examples/entropy_pool_learning_compare.rs` — −430 LOC, now an orchestration shell.
- `hymeko_nagare/examples/holonomy_entropy_toys.rs` — −250 LOC, imports substrate from the crate.
- `hymeko_nagare/examples/global_pool_entropy_parity.rs` — fused update path; dead `update_input` removed.
- `hymeko_nagare/tests/{entropy_pool_learning_compare,holonomy_entropy_toys}.rs` — repointed at the crate API.
- Format-only (crate-wide `cargo fmt`, edition-2024 import ordering, no semantic change): `benches/parity_gate.rs`, `src/ops/{clifford_fir,fsr_mixer}.rs`, `src/{optimizer,runtime}.rs`, `examples/{global_pool_entropy_toys,synthetic_cheby_compare}.rs`, `tests/{global_pool_entropy_toys,integration_training}.rs`.

**CORE.YAML items touched: none.** No dependency added/changed.
`docs/plans/` is gitignored by policy; the plan stays on disk locally.

## Test results

- `cargo test -p hymeko_nagare`: **54 passed, 0 failed, 1 ignored** (the
  deliberate fixture writer) — 29 lib unit + 25 integration across 8 test
  binaries; wall < 2 s.
- Coverage rule: every promoted public item is driven by a ported or new
  test (datasets/pooling/learner/projection via the compare tests; features +
  Clifford metric via the toys tests; kernel via in-module + FD tests;
  fixture via the round-trip test). Regression tests for behaviour changes:
  the FD suite (new kernel), the fixture test (new), and the bit-identical
  production re-run (move + rayon change).
- Gates: `cargo fmt -p hymeko_nagare -- --check` clean;
  `cargo clippy -p hymeko_nagare --all-targets --no-deps -- -D warnings`
  clean. No new `allow`/suppression introduced; no `unwrap` in non-test code.
- §6.5 sweep: no anti-patterns introduced (the change *removes* ~680 LOC of
  example-side duplication; no new flags, no v2 files, no globals).

## Performance results vs budget

- Rust executables peak RSS: 9.0–9.2 MiB (budget < 16 MiB) ✓
- PyTorch process tree: 636 MiB (budget < 1 GiB) ✓; global 16 GiB cap untouched ✓
- Stress-ablation wall ≈ 75 s (budget < 3 min) ✓
- Local-learner forward after refactor: 4.1–4.3 µs/sample (prior 5.7–5.8;
  no regression — slightly faster) ✓
- Parity target "<70 µs": met in run B (64–68), missed in run A (79–84) —
  reported as a spread; the >10%-regression rule was *triggered and resolved*
  during the work (serial fused defect, root-caused to missing rayon
  parallelism, fixed, re-measured).

## Experiment provenance

- Git: branch `hymeko-neuro-migration`, base `0849918` (NAGARE snapshot
  `0211128`); this change committed as the follow-up commit.
- Host: AMD Ryzen 9 5900HX, 31.4 GB RAM, Windows 11 Pro; rustc 1.93.1;
  torch 2.12.0+cu132 (CORE-pinned), 8 torch threads; no other python/torch
  workload during measurement (checked per §6.5 #17).
- Seeds: 53 (+ documented task/stress offsets) for the compare line; 123 for
  the parity fixture.
- Artifacts: `reports/2026-07-04-nagare-fused-parity-{nagare,pytorch}.json`,
  `...-{nagare,pytorch}-rss.txt`, `...-stdout.txt`,
  `reports/2026-07-04-nagare-fused-parity.png`;
  frozen fixture `hymeko_nagare/tests/fixtures/moons_spiral_xor_seed53.txt`.

## Open issues / follow-ups

1. **Parity gap profile:** run `cargo flamegraph` on the parity example to
   isolate serial `global_pool` vs scalar accumulation before any further
   kernel optimization.
2. **Fused backward is still serial** — fine today (training loops are
   batch-small), flagged for when training-side throughput matters.
3. **Extraction decision** (`nagare-holonomy-learn` sibling repo): the
   prerequisites from the 07-02 consolidated report are now all met
   (library module ✓, FD tests ✓, frozen fixture ✓). Awaiting the user's
   go/no-go.
4. **Next discriminating science test** (unchanged from 07-02): corrupt
   hyperedges / shuffle point order / multi-class, to isolate whether the
   projection gate does learning work or rides on strong fixed holonomy
   features.
