# Leakage-free input enrichment — k-walk profile helps the un-propagated rotor, redundant with slerp propagation

**Date:** 2026-06-18
**Plan:** [docs/plans/2026-06-18-leakage-free-input-enrichment](../docs/plans/2026-06-18-leakage-free-input-enrichment/) (4 artifacts; PDF compiles).
**Status:** ✅ registry + new k-walk feature implemented + tested + gated; 5-seed A/B on two harnesses. **Mixed/honest result:** the new feature lifts the *non-propagated* rotor baseline (otc cyc+walk +0.0168), but is **parity on the slerp-propagated line** — propagation already supplies the same signed-`A^k` signal.

## Summary

The rotor line's ceiling is input-bounded (four prior levers flat). This builds the
**input** lever as an extensible, leakage-free `StructuralFeature` registry and adds
the genuinely-new **exact k-walk signed `A^k` profile** (Dr. Hajdu's idea; cycle
participation already existed and is reused, not rebuilt). Findings, all deduped +
shuffle-gated, 5 seeds:

- **Non-propagated rotor baseline (audit, SGCN message-passing):** input enrichment
  helps on otc — `walk` +0.0118, `cyc+walk` **+0.0168** over degree-only (0.8595 →
  0.8763), gate-clean; the walk profile is the larger contributor and stacks with
  cycle. On alpha it is within noise (+0.004 < σ 0.023).
- **Slerp-propagated rotor-HSiKAN line:** enrichment is **parity** — `degree+walk`
  otc +0.0030, alpha +0.0013 (both < σ), gates clean. A striking **seed-0** otc
  0.9131 (which beat pure SiGAT) **did not replicate** — the 5-seed mean is 0.8820;
  it was seed variance, **not a finding**.
- **Inference (suggestive, not airtight):** signed slerp propagation *is* signed-`A^k`
  aggregation on S³ — exactly the walk-profile feature's content — so the feature is
  largely **redundant** on the propagated line and pays off only where that
  aggregation is absent (the SGCN baseline). A within-harness rounds=0-vs-2 5-seed
  confirmation is the follow-up: the seed-0 attempt was inconclusive (variance too
  large to read a single seed — itself the cautionary note here).
- **Bottom line:** the numbers do **not** push past the 0.879-otc propagated
  operating point; the walk profile reaches a similar point by a simpler,
  propagation-free route. The residual gap to pure SiGAT (deduped 0.895) is
  **expressivity** (learned multi-layer attention), not addressable by parameter-free
  input features.

## Results (5-seed mean ± pstdev, deduped, gate = mean shuffled AUROC ≤ 0.55)

### Non-propagated rotor baseline (audit / SGCN) — `rotor_feature_enrichment_ab.jsonl`
| spec | alpha | otc |
|---|---|---|
| degree | 0.8189±0.023 | 0.8595±0.010 |
| +cycle | 0.8208 (+0.0019) | 0.8632 (+0.0037) |
| +walk | 0.8219 (+0.0030) | 0.8713 (+0.0118) |
| **+cyc+walk** | 0.8229 (+0.0040) | **0.8763 (+0.0168)** |
gates 0.49–0.54 (clean).

### Slerp-propagated rotor-HSiKAN line — `rotorprop_walk_enrichment_ab.jsonl`
| spec | alpha | otc |
|---|---|---|
| degree | 0.8500±0.013 (g0.53) | 0.8790±0.010 (g0.52) |
| degree+walk | 0.8513 (+0.0013, g0.53) | 0.8820 (+0.0030, g0.55) |

**Measured / inferred (CLAUDE.md).** *Measured:* the tables + gates; seed-0 propagated
otc+walk = 0.9131 vs 5-seed 0.8820. *Inferred:* slerp propagation ≈ signed-`A^k`
aggregation ⇒ walk redundant on the propagated line. *Rejected as superstition:* the
seed-0 0.913 "beats pure SiGAT" — did not replicate.

## Files touched

**New (2):**
- `signedkan_wip/src/baselines/structural_features.py` (+250) — leakage-free
  `StructuralFeature` registry (Strategy, §6.5 #1/#9). Extractors: `degree` (moved,
  canonical), `cycle_k3`/`cycle_k34` (moved), **`walk_k3`** (new: exact signed/total
  `A^k` reach + ratio via sparse mat-vecs, no cap), **`ratios`** (new: signed/unsigned
  local clustering via `diag(A^3)=(A∘A²)·1`). `build_node_features(spec, …)` derives
  `STRUCT_DIM` from the spec; `("degree",)` reproduces the legacy features exactly.
- `signedkan_wip/tests/test_structural_features.py` (+150) — 13 tests: default-spec
  parity, exact `A^k` vs brute force, all-positive ⇒ unit ratio, clustering bounds +
  planted triangle, signed-clustering balance flip, registry failure cases, purity.

**Modified (mine):**
- `signedkan_wip/src/baselines/cayley_rotor_baseline.py` — degree/cycle features
  **single-sourced** into the registry (re-exported for existing importers); 3 new
  spec-driven baselines (`cayley_rotor_walk`, `_cyc_walk`, `_full`) via one
  parameterised base (no per-variant `build_context` copy).
- `signedkan_wip/experiments/runs/run_hsikan_rotor.py` — `feature_spec` threaded
  through `run()`/`RotorInjector` (`struct_dim` derived); `--feature-spec` CLI;
  provenance. Default `("degree",)` reproduces prior.
- `signedkan_wip/tests/test_hsikan_rotor.py` — enriched-input smoke + provenance.

**Artifacts:** `rotor_feature_enrichment_ab.jsonl` (80 rows), `rotorprop_walk_enrichment_ab.jsonl` (40 rows).
**CORE.YAML items touched:** none.

## Test results
- `test_structural_features.py` 13 ✓; `test_hsikan_rotor.py` 29 ✓ (2 new);
  `test_baseline_audit.py` 16 ✓; `test_cayley_rotor_baseline.py` ✓. (`pytest -p no:randomly`.)
- `ruff check`: clean on all touched files. `mypy --strict`: clean on my added code,
  with one scoped `# type: ignore[import-untyped]` (scipy ships no `py.typed` stubs;
  scipy is an established runtime dep). Pre-existing: the unrelated `_propagate`
  `Any`-return in `signed_rotor_propagation.py` (untouched).

## Performance
- `walk_k3`: `O(K·nnz)` sparse mat-vecs, exact, no cap — negligible (+~270 params at
  the rotor input). Audit cells ~2.5–3 s; propagated cells ~10–12 s, GPU (cu132).
  Peak RSS ≈ 1.8 GB (≪ 16 GB cap).

## §6.5 anti-patterns
None. Registry is a Strategy (spec-driven, no per-variant duplication, §6.5 #1/#9);
features single-sourced (cayley_rotor_baseline re-exports). scipy `type: ignore` is
line-scoped with a reason (§6.3). No new globals; all extractors are pure train-only
functions (leakage-safe, gated).

## Experiment provenance
- Git SHA `7d16ad0` (tree dirty from the session; touched files above). Datasets:
  cached SNAP `bitcoin_alpha`, `bitcoin_otc`. Seeds 0–4. Device CUDA (torch 2.12.0+cu132).
  Recipe: rotor embed, bilinear head, dedup, tuned BCE (wd 1e-4, grad-clip 1.0,
  class-weighted, early-stop), propagation r2 sw4 where applicable.

## Open issues / follow-ups
- **Within-harness redundancy confirmation:** 5-seed rounds=0 vs rounds=2 × {degree, +walk}
  (seed-0 was inconclusive — variance). Would make "slerp ⊇ walk-profile" airtight.
- **The pure-SiGAT gap is expressivity**, not input features. Options that could move it
  further dilute the rotor line's simplicity (learned per-edge propagation weights →
  toward SiGAT). The line's value remains: leakage-free, inductive, ~270× fewer params,
  and now competitive on otc by *either* slerp propagation *or* the walk-profile input.
- **Ratios** (`cayley_rotor_full`) were neutral-to-slightly-negative; kept registered
  for the registry's extensibility, not recommended in the operating spec.
- Fuzzy-defuzzification head unification ([[project-fuzzy-defuzzification-heads]]) still open.
