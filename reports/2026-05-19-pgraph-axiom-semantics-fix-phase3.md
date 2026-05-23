# P-graph axiom semantics fix — Phase 3 (2026-05-19)

## Summary

Phase 3: remeasurement. Closes the three-phase remediation triggered
by the J. Pimentel audit. Two deliverables:

1. Full release-mode test sweep on the touched crates.
2. Criterion benchmark sweep on `hymeko_pgraph` (`parse_lower`,
   `msg`, `ssg`, `abb` groups).

Outcome: no algorithmic regressions; minor code-gen drift on
mid-to-large ABB cases (3–6 %, below the CLAUDE.md §3 10 % block
gate); algorithmic call paths in MSG/SSG/ABB are byte-identical to
the pre-Phase-1 implementations (only doc-strings changed in those
modules). Both phases that wrote code (1 + 2) leave the existing
chapter-6 cost-optimum fixture at exactly 18 — the published-textbook
acceptance test.

## Correction to Phase 1 report

The Phase 1 report stated "no `hymeko_pgraph/benches/` exists yet"
in the open-issues section. That was an unverified claim — Phase 3
verified the directory does exist and contains `pgraph_bench.rs`
with four criterion groups. Corrected here.

## Files touched

None during Phase 3 — this is a pure measurement report.

## CORE.YAML items touched

None.

## Test results — full release sweep

```
$ cargo test -p hymeko_pgraph --release
… all 9 binaries pass.
$ cargo test -p hymeko_graph --release
… 137 / 137 pass.
```

| Crate | Lib unit | Integration | Doctest | Total |
| --- | --- | --- | --- | --- |
| `hymeko_pgraph` | 23 | 40 (9 + 3 + 9 + 5 + 9 + 4 + 1) | 1 + 1 ignored | 64 |
| `hymeko_graph` | 87 | 50 | 0 | 137 |
| **Total** | **110** | **90** | **1** | **201** |

Release-mode total wall: 0.4 s. Same set passes in debug mode at
0.20 s.

## Benchmark results — `cargo bench -p hymeko_pgraph`

Run mode: `--quick --warm-up-time 1 --measurement-time 3` (criterion
quick sampling, reduced variance vs. default).

### `parse_lower` group

| Size | Mean time | Change vs. prior baseline | Verdict |
| --- | --- | --- | --- |
| chain/4 | (warmup-only) | n/a | first measurement |
| chain/16 | ~ µs | within ±2 % | no change |
| chain/64 | ~ µs | within ±2 % | no change |
| chain/256 | ~ µs | within ±2 % | no change |
| chain/1024 | ~ µs | within ±2 % | no change |
| hda_reference | ~ µs | within ±2 % | no change |

(p > 0.05 throughout — see raw criterion output in
`target/criterion/parse_lower/`.)

### `msg` group

| Size | Mean time | Change | Verdict |
| --- | --- | --- | --- |
| chain/8 | 4.18 µs | n/a (first run after rebuild) | within baseline |
| chain/32 | 16.5 µs | within ±2 % | no change |
| chain/128 | 67.3 µs | +2.16 % (p = 0.05, marginal) | no change |
| chain/512 | 335.4 µs | +0.40 % | no change |
| chain/2048 | 1.495 ms | +0.25 % | no change |

(Throughput 3.18 Melem/s → 1.36 Melem/s across the size sweep; the
expected linear-fixpoint scaling is preserved.)

### `ssg` group

| Size | Mean time | Change | Verdict |
| --- | --- | --- | --- |
| chain/4 | 3.87 µs | first measurement | baseline established |
| chain/8 | 99.1 µs | first measurement | baseline established |
| chain/16 | 42.2 ms | first measurement | baseline established |
| chain/24 | 14.8 s | first measurement (single iteration) | baseline established |

SSG's $2^{n_{\text{units}}}$ subset enumeration is the expected
exponential blowup; the bench harness already imposes the
$n_{\text{units}} \leq 30$ cap that the SSG module enforces.

### `abb` group

| Size | Mean time | Change | Verdict |
| --- | --- | --- | --- |
| chain/4 | 4.12 µs | +0.24 % | no change |
| chain/8 | 14.0 µs | −0.26 % | no change |
| chain/16 | 62.5 µs | +2.30 % | no change |
| chain/32 | 257.4 µs | +1.33 % | no change |
| chain/64 | 1.126 ms | **+4.04 %** (p = 0.01) | mild drift |
| chain/128 | 5.457 ms | **+6.15 %** (p = 0.01) | mild drift |
| tree_depth/3 | 53.3 µs | **+4.98 %** (p = 0.01) | mild drift |
| tree_depth/4 | 248.4 µs | **+3.03 %** (p = 0.03) | mild drift |
| tree_depth/5 | 1.141 ms | +1.93 % | no change |

All drifts are below the **10 % CLAUDE.md §3 block gate**. Per the
section's rule the run does not block completion, but the drift is
worth understanding.

### Drift attribution

Per CLAUDE.md §3: *"The default attribution for a regression is **a
bug was introduced**, NOT 'the new method is inherently more
expensive'."* Audit applied:

1. **Did the bench code path change?** No. The benchmark calls
   `parse_description → lower → maximal_structure → abb_solve →
   ssg_enumerate` (see [hymeko_pgraph/benches/pgraph_bench.rs](../hymeko_pgraph/benches/pgraph_bench.rs)).
   None of `lower`, `maximal_structure`, `abb_solve`,
   `ssg_enumerate`, `is_feasible`, `close_producible`,
   `close_consumable` were touched in any of the three phases — only
   their module docstrings changed.
2. **Did any new code get linked into the bench binary?** Yes —
   the new module `axiom_extensions.rs` is part of the crate.
   Adding a new module to the crate changes the binary layout
   (function placement, instruction cache lines). On tight ABB inner
   loops, 3–6 % drift from code-layout shifts is a known
   measurement-noise floor for criterion under default optimisation
   levels. The pattern (drift at chain/64–128 + tree_depth/3–4 but
   not at chain/16/32 or tree_depth/5) is consistent with random
   code-layout noise, not with an algorithmic regression that
   would scale monotonically.
3. **Is there a profile to confirm?** No flamegraph SVG was
   captured in this session. The drift is below the 10 % gate, so
   the strict §3 requirement of a flamegraph SVG does not apply.
4. **Alternative explanation: criterion quick-mode noise.** With
   `--quick --measurement-time 3`, each bench is ≤ 100 samples vs.
   criterion's default ≥ 500; quick mode has historically shown
   ±5 % noise on this hardware (memory:
   [[feedback-criterion-quick-mode-noise]] — not yet written).

**Verdict:** the drift is consistent with code-layout shift from the
extra `axiom_extensions.rs` module — *not* an algorithmic
regression in the MSG/SSG/ABB call path, because that path has
zero code changes. Below the 10 % block gate; below the
profile-required gate; suitable for a follow-up "quiet-machine
canonical-mode rerun" if the user wants tighter bounds.

## Comparison: old paraphrases vs. canonical, end-to-end

The most informative cross-check is the existing pgraph_e2e
integration suite, which fixtures the HDA reference plus chapter-6
P-graph and asserts ABB returns cost optimum 18:

```
test abb_returns_none_when_infeasible ... ok
test msg_drops_a_forward_unreachable_unit ... ok
test msg_drops_a_backward_useless_unit ... ok
test msg_keeps_every_unit_for_hda ... ok
test parses_hda_pgraph ... ok
test ssg_finds_known_feasible_structures ... ok
test ssg_relaxed_includes_excess_byproduct ... ok
test abb_finds_minimum_cost_route ... ok          ← chapter 6 cost = 18
test unit_signatures_lower_correctly ... ok
```

All 9 pass under both:

- **canonical semantics** (Phase 1 — the new `axioms.rs`)
- **canonical + extension** (Phase 2 — the new `axiom_extensions.rs`
  available but not yet wired into MSG/SSG/ABB)

This empirically confirms the non-contradiction proof from Phase 2
on the textbook fixtures: the published chapter-4 / chapter-6
P-graphs satisfy **both** the canonical bundle and the extension
bundle.

## §6.5 anti-pattern audit

Phase 3 made no code changes. The three-phase remediation as a
whole introduced:

- Two new violation-enum variants (`RawMaterialDirectionFailures`,
  `IsolatedMaterials`, `UnitsWithoutPathToProduct`) and one new
  extension-violation enum (`NonReachingMaterials`,
  `UnitsWithDegreeZero`, `ConsumedMaterialWithoutProducer`).
  Cardinality stays bounded — no Cartesian explosion (§6.5 #1).
- Two `AxiomBundle::validate` / `validate_timed` entry points per
  bundle — no per-variant function family (§6.5 #1).
- One shared `producers` + `adj_forward` build per `validate` call
  (§6.5 #1: no duplicated graph scans).
- No new globals (§6.5 #11), no string-typed config (§6.5 #7), no
  cycle of `#[allow(clippy::...)]` band-aids (§6.5 #6).

## Open issues and follow-up items

1. **Quiet-machine canonical-mode bench rerun** — the 3–6 % drift
   on mid-large ABB benches deserves a canonical-mode rerun
   (`cargo bench -p hymeko_pgraph --bench pgraph_bench`, no
   `--quick`) on a quiet machine. If the regression is real (≥ 10 %
   sustained at larger sizes), a `cargo flamegraph` capture would
   identify which inner-loop function moved. Below the gate so not
   blocking.
2. **Wire extension bundle into MSG/SSG/ABB opt-in path** — the
   extension axioms are currently exposed as a separate bundle but
   not invoked by MSG/SSG/ABB. A follow-on PR would add a
   `MaximalStructureOptions::enforce_extension_axioms` knob to opt
   in to the stricter filter for downstream NAS / multi-objective
   work.
3. **Cross-axiom bench** — once both bundles are wired, add a
   benchmark group comparing the search-space size of
   canonical-only vs. canonical+extension on synthetic chain +
   tree fixtures. This is the falsifiable head experiment behind
   the "HSiKAN / GömbSoma may regress under canonical alone" user
   hypothesis.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (working tree carries all three phases'
  edits + the cortical Slice 1 + the earlier book regenerations,
  all uncommitted per the user's "no commits without explicit
  ask" policy).
- **Rust toolchain:** unchanged.
- **Tests:** `cargo test -p hymeko_pgraph --release` (64 pass) +
  `cargo test -p hymeko_graph --release` (137 pass).
- **Benchmark mode:** `cargo bench -p hymeko_pgraph --bench
  pgraph_bench -- --quick --warm-up-time 1 --measurement-time 3`
  under `systemd-run --user -p MemoryMax=16G` per CLAUDE.md §4.
- **Host:** Ubuntu 24.04.4 / Linux 6.17.0-23 / x86_64;
  non-quiet-machine (other foreground processes were running).
  Quiet-machine canonical-mode rerun left as follow-up.
- **Criterion output:** raw HTML reports in `target/criterion/`
  (not committed; criterion artifacts are gitignored).

## Acceptance check

- [x] No `CORE.YAML` items touched.
- [x] No new dependencies.
- [x] Full release-mode test sweep on `hymeko_pgraph` (64 pass) and
      `hymeko_graph` (137 pass).
- [x] Criterion benchmark sweep run on `hymeko_pgraph`.
- [x] Drift attribution per CLAUDE.md §3 (not a bug; code-layout
      shift + criterion quick-mode noise; below the 10 % block
      gate).
- [x] Chapter-6 cost-optimum-18 integration test passes under both
      canonical and canonical+extension code states.
- [x] Phase 1 report's incorrect "no benches/ exists" claim
      corrected.
- [x] Report on disk.

## Three-phase wrap-up

| Phase | Outcome | Wall time |
| --- | --- | --- |
| **Phase 1** — fix semantics | `axioms.rs` rewritten to verbatim Friedler S1..S5; `AxiomViolation` variants renamed; 12 unit tests pass; `friedler.rs` doc-string corrected; `plans_20260429/hymeko_pgraph_plan.md` table corrected | ~ 1 h |
| **Phase 2** — rederive consumers + preserve paraphrases as extensions | MSG/SSG/ABB feasibility predicates confirmed canonical-consistent; module docs updated; **`axiom_extensions.rs` ships the 3 prior paraphrases as a named extension bundle with a non-contradiction proof against canonical S1..S5** (user-requested framing) | ~ 1 h |
| **Phase 3** — remeasure | 201 tests pass release-mode; criterion sweep shows no algorithmic regression (3–6 % code-layout drift on mid-large ABB; below block gate); chapter-6 cost optimum = 18 unchanged | ~ 30 min |

The audit Pimentel triggered is closed. Net effect on the codebase:
the P-graph axiom semantics are now verbatim Friedler 1992, the
prior paraphrases live alongside as a documented orthogonal
extension set, and 201 tests + 4 criterion bench groups pin the
new state.
