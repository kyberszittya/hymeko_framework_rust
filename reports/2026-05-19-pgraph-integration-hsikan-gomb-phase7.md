# P-graph integration into HSIKAN + Gömb pipelines — Phase 7 (2026-05-19)

## Summary

Wired the canonical Friedler 1992 axiom layer (Phases 1+2) and the
orthogonal extension bundle (Phase 2) into the existing HSIKAN and
Gömb training pipelines. Three pieces, all in scope per the user's
request:

- **(A)** Every `hymeko_pgraph_dump` invocation now emits canonical
  + extension certificates on the **full schema** and on the
  **ABB-selected sub-schema** (plus an echo of the engine's
  `strict_no_excess` mode).
- **(C)** The `strict_no_excess` knob is reachable from Python via
  the existing `--relaxed-msg` flag and surfaced back through the
  new DTO field — no new CLI flag introduced.
- **(B)** A new HSIKAN sweep driver
  (`run_hsikan_msg_sweep.py` + `hsikan_pgraph_mapping.py`) parallel
  to the existing Gömb driver. Maps ABB-selected operating-unit
  names (`cycle_topk_m4`, `model_h8`, `train_short`, ...) to
  `run_compare.run_one` kwargs and optionally launches training.

The Gömb driver was updated to read the new certificate fields,
print a one-line Friedler summary, and stamp the certificate onto
every JSONL training row.

## Files touched

| File | Status | LOC | Notes |
| --- | --- | --- | --- |
| `docs/plans/2026-05-19-pgraph-integration-hsikan-gomb/plan.{tex,pdf,mmd,tikz}` | new | 4-format plan (3 pp PDF) | Compiled before code |
| `hymeko_pgraph/src/dump.rs` | extended | +200 | `AxiomCertificateJson` DTO + `canonical_full` / `extension_full` / `canonical_abb_subschema` / `extension_abb_subschema` / `strict_no_excess` fields + helpers (`name`, `cert_pass`, `canonical_cert`, `extension_cert`, `project_subschema`, `empty_cert_with`); both early-return paths and `analyze_lowered_with_full_options` populate the new fields |
| `hymeko_pgraph/tests/axiom_witness.rs` | extended | +60 | 3 new tests pinning the DTO behaviour on the by-product fixture under strict / relaxed modes + MSG-only algorithm |
| `signedkan_wip/src/hsikan_pgraph_mapping.py` | **new** | 100 | `HSIKAN_UNIT_TO_KNOBS` table + `merge_structure_knobs` + `run_one_kwargs` |
| `signedkan_wip/experiments/runs/run_hsikan_msg_sweep.py` | **new** | 175 | Parallel structure to the Gömb driver; dry-run by default, `--train` flag for actual training |
| `signedkan_wip/experiments/runs/run_gomb_msg_sweep.py` | extended | +60 | `--relaxed-msg` flag forwarded to the binary, `_cert_brief` / `_print_certificate_summary` / `_certificate_fields` helpers, certificate fields stamped onto every JSONL row |
| `signedkan_wip/tests/test_hsikan_pgraph_mapping.py` | **new** | 80 | 7 unit tests for the mapping table |

Total: ~700 LOC.

## CORE.YAML items touched

None.

## Interface changes

### New Rust DTO type

```rust
pub struct AxiomCertificateJson {
    pub status: String,                          // "PASS" or "FAIL"
    pub violation_tags: Vec<String>,             // ["S1".."S5"] or extension tags
    pub offenders: Vec<(String, Vec<String>)>,   // (tag, [decl_name, ...])
}
```

### `PgraphAnalysisJson` additions (additive — back-compat)

```rust
pub struct PgraphAnalysisJson {
    // ... existing fields unchanged ...
    pub canonical_full: AxiomCertificateJson,
    pub extension_full: AxiomCertificateJson,
    pub canonical_abb_subschema: Option<AxiomCertificateJson>,
    pub extension_abb_subschema: Option<AxiomCertificateJson>,
    pub strict_no_excess: bool,
}
```

### Python driver CLI

```bash
# Gömb (existing driver, now with --relaxed-msg)
python -m signedkan_wip.experiments.runs.run_gomb_msg_sweep \
    --pgraph data/hsikan/sweep_msg_gomb.hymeko \
    --algorithm abb \
    --relaxed-msg

# HSIKAN (new driver)
python -m signedkan_wip.experiments.runs.run_hsikan_msg_sweep \
    --pgraph data/hsikan/sweep_msg.hymeko \
    --algorithm abb \
    --dataset bitcoin_alpha \
    --seeds 0 1 2 \
    --train   # optional: actually run HSIKAN training
```

## Test results

| Suite | Count | Status |
| --- | --- | --- |
| `cargo test -p hymeko_pgraph` (all binaries) | 90 | **90 pass + 1 ignored doctest** (up from 87 pre-Phase-7) |
| `axiom_witness.rs` | 27 | **27 pass** (24 prior + 3 new DTO tests) |
| `test_hsikan_pgraph_mapping.py` | 7 | **7 pass** |
| `test_gomb_pgraph_driver.py::test_run_gomb_msg_sweep_msg_phase_only` | 1 | pass — confirms my driver edits didn't break the existing harness |

### Known pre-existing failure (unrelated to Phase 7)

`test_gomb_pgraph_driver.py::test_run_single_gomb_smoke_subprocess`
fails with `AttributeError: module 'hymeko' has no attribute
'enumerate_cycles_rs'. Did you mean: 'enumerate_k_cycles_rs'?` —
this is a PyO3 binding rename from the 2026-05-11 codebase rehaul
(see [[project_codebase_rehaul_plan_2026_05_11]] in memory) that
hasn't been propagated into `run_gomb_smoke.py:112`. Out of scope for
Phase 7; the test failed before my edits too. Track as a separate
follow-up item.

### Smoke runs (recorded in this session)

**Strict mode on the canonical HSIKAN sweep:**

```
description: HSiKAN_Sweep_MSG
algorithm:   abb
strict_no_excess: True
  canonical (full schema): PASS
  extension (full schema): PASS
  canonical (ABB sub-schema): PASS
  extension (ABB sub-schema): PASS
msg_units (8): [cycle_topk_m{4,16,64}, model_h{8,16,32}, train_{short,long}]
abb units (3): [cycle_topk_m4, model_h8, train_short]  cost=60.0

selection: merged structure = {m_cycles=4, hidden=8, n_epochs=10}
           run_one_kwargs    = {model='highway_signedkan', dataset='bitcoin_alpha',
                                seed=0, hidden=8, n_epochs=10, lr=0.05, m_cycles=4}
```

**Relaxed mode on the by-product fixture:**

```
description: HSiKAN_Sweep_Byproduct
strict_no_excess: False
  canonical (full schema): PASS
  extension (full schema): FAIL [E-NoExcess] — E-NoExcess=[redundancy_byproduct]
  canonical (ABB sub-schema): PASS
  extension (ABB sub-schema): FAIL [E-NoExcess] — E-NoExcess=[redundancy_byproduct]
abb units (3): [cycle_topk_m4, model_h8, train_short]  cost=60.0
```

The relaxed-mode sub-schema certificate correctly flags the by-product
that survived the engine's no-excess filter.

## Quantitative results — 5-seed AUC on Bitcoin Alpha

Driver invoked end-to-end with `--train`-equivalent direct
`run_one(**run_one_kwargs)` for each ABB-feasible architecture
selection on `data/hsikan/sweep_msg.hymeko` (and the by-product
variant). Bitcoin Alpha, basic `signedkan` model, lr = 5e-2,
5 seeds (0..4).

| ABB selection | h | epochs | proxy cost | mean AUC ± std | wall / seed |
| --- | --- | --- | --- | --- | --- |
| `cycle_topk_m4, model_h8, train_short` (cheap, canonical PASS) | 8 | 10 | 60 | **0.576 ± 0.029** | 1.6 s |
| `cycle_topk_m16, model_h8, train_short` (strict + by-product) | 8 | 10 | 90 | 0.576 ± 0.029 | 1.6 s |
| `cycle_topk_m4, model_h16, train_short` | 16 | 10 | 100 | 0.473 ± 0.106 | 2.6 s |
| `cycle_topk_m4, model_h32, train_short` | 32 | 10 | 240 | 0.516 ± 0.042 | 3.8 s |
| `cycle_topk_m4, model_h8, train_long` | 8 | 60 | 150 | **0.817 ± 0.015** | 6.1 s |

### Three substantive findings

1. **The cycle-topk axis currently doesn't propagate.** The first
   two rows have identical AUC and identical wall time because
   `run_compare.run_one` does not consume `m_cycles` — the P-graph
   framework picks `cycle_topk_m4` vs `cycle_topk_m16` (the Phase 6
   strict-vs-relaxed divergence) but that distinction is silently
   dropped at the Python boundary. This is a **plumbing gap**, not
   a wrong axiom: the integration is correct, the consumer is
   incomplete. Fix path: add `m_cycles` as a proper `run_one`
   kwarg + thread through the cycle-enumeration backend, or honor
   `HSIKAN_M_CYCLES` env (the driver already sets it under
   `--train`).
2. **The training-length axis dominates the architecture-cost
   ranking.** `train_long` (n_epochs = 60) at cost 150 reaches
   AUC 0.817 ± 0.015 — beating every other P-graph-feasible
   selection by a wide margin. So the P-graph framework's cost-60
   cheapest pick (h=8, train_short) is NOT the best architecture
   on this dataset; cost 150 is. A multi-objective ABB run
   weighting AUC against cost would correctly prefer
   `train_long`. (This is exactly the kind of finding the
   `2026-05-19-pgraph-multi-objective` plan is designed to surface.)
3. **Wider hidden hurts at short epochs.** h=16 gets AUC 0.473 ± 0.106
   (worse than h=8); h=32 gets 0.516 ± 0.042 (also worse). The
   wider models need more epochs to use their capacity — at
   n_epochs = 10 they under-train. This is the kind of brittle
   architectural choice that strict-no-excess + a "wasted
   parameters" by-product material would let the P-graph framework
   automatically penalise: declare a by-product `unused_capacity`
   that `model_h32` produces but nothing consumes, and strict mode
   filters it out at the architecture-search stage.

### The quantitative answer to "did HSIKAN gain anything?"

- **Functional integration:** yes — the P-graph framework now
  drives HSIKAN architecture selection end-to-end (8 candidate
  architectures × Friedler certificate on each).
- **Search-quality gain on the existing `sweep_msg.hymeko`:** no
  — the search space has no by-products, so canonical and
  extension agree, and the cycle-topk dimension doesn't propagate
  through `run_compare.run_one` anyway. ABB picks the cost-60 path
  and that's what runs.
- **Latent gain via by-product injection:** once the cycle-topk
  axis is wired (gap #1 above), the Phase 6 by-product mechanism
  becomes a usable NAS lever — `unused_capacity` / `wasted_compute`
  by-product materials let the framework reject architectures that
  the AUC numbers above show are dominated (h=16/h=32 at short
  epochs).

## Performance results

- Two axiom-validation passes per dump invocation. Each is
  $O(|V|+|E|)$; sub-millisecond at textbook scale. Confirmed by
  re-running `cargo test -p hymeko_pgraph` (0.20 s total wall —
  unchanged vs. Phase 6 baseline).
- The release dump binary is 22 MB (pre-Phase-7) → 22 MB (post-Phase-7);
  no measurable size impact from the additional code.

## §6.5 anti-pattern audit

No new anti-patterns introduced. Specifically:

- The new DTO variant (`AxiomCertificateJson`) is a single shape used
  for all four certificate slots — no Cartesian explosion.
- The Python mapping module mirrors the existing Gömb mapping
  module's shape (§7 Strategy / Adapter pattern).
- The HSIKAN driver uses the same `run_pgraph_dump` helper pattern
  as the Gömb driver — no duplicated scaffold.

## New / removed dependencies

None. All existing.

## Open issues and follow-up items

1. **PyO3 binding rename propagation.** `run_gomb_smoke.py:112` calls
   `hymeko.enumerate_cycles_rs` which was renamed to
   `enumerate_k_cycles_rs` during the 2026-05-11 codebase rehaul.
   Pre-existing, unrelated to Phase 7. Fix in a separate PR.
2. **HSIKAN `m_cycles` env-var path.** `run_compare.run_one` doesn't
   currently accept `m_cycles` as a kwarg; the HSIKAN driver pops it
   from `run_one_kwargs` and sets `HSIKAN_M_CYCLES` env var when
   `--train` is on. Cleaner long-term: thread it as a proper kwarg.
3. **Sweep-output filtering by Friedler-feasibility.** With the
   certificate now baked into every JSONL row, downstream analysis
   scripts can group / filter sweeps by
   `canonical_abb_status == "PASS"`. No script does this yet; a
   single-line `jq` filter would surface it ad-hoc.
4. **Multi-objective plan crossref.** The morning's
   `2026-05-19-pgraph-multi-objective` plan should declare which
   bundle the multi-objective ABB enforces, now that both are
   available end-to-end.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted: P-graph phases 1-7
  + GömbSoma cortical Slice 1 + earlier book regenerations).
- **Build:** `cargo build --release -p hymeko_pgraph --bin
  hymeko_pgraph_dump` ran cleanly (4.19 s).
- **Tests:** `cargo test -p hymeko_pgraph` (90 pass / 1 ignored
  doctest); `pytest signedkan_wip/tests/test_hsikan_pgraph_mapping.py`
  (7 pass).
- **Smoke runs:** the HSIKAN driver ran successfully on
  `data/hsikan/sweep_msg.hymeko` (strict mode) and
  `data/hsikan/sweep_msg_byproduct.hymeko` (relaxed mode) with
  `--seeds 0 1` — both surfaced the certificate fields correctly.

## Acceptance check

- [x] 4-format plan written + PDF compiled before code (3 pp,
      ~130 KB).
- [x] No `CORE.YAML` items touched.
- [x] No new dependencies.
- [x] DTO additions are back-compat (additive).
- [x] All 90 Rust pgraph tests pass.
- [x] All 7 new Python mapping tests pass.
- [x] The Gömb sweep MSG-phase integration test (the one that
      doesn't depend on the pre-existing PyO3 rename bug) passes.
- [x] HSIKAN driver smoke-tested under strict + relaxed modes; both
      surface the expected Friedler certificate.
- [x] §6.5 anti-pattern audit clean.
- [x] Report on disk.

## Memory entry

Added `project_pgraph_integration_hsikan_gomb_phase7_2026_05_19.md`
pointing to this report; the existing Pimentel audit memory entry
gets its Phase 7 footnote.
