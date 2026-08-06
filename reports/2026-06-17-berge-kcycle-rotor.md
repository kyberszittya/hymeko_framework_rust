# k-cycle (Berge) neighbourhoods on the rotor line — flat; the fourth lever to land flat

**Date:** 2026-06-17
**Plan:** [docs/plans/2026-06-17-berge-kcycle-rotor](../docs/plans/2026-06-17-berge-kcycle-rotor/) (4 artifacts; PDF compiles).
**Status:** ✅ implemented (reuse, no new enumerator) + tested + 5-seed A/B + gate. ⚠️ **negative** — pooling closed 4-cycles instead of 3-triads does not move the rotor line. Combined with the three comparison-side negatives, this localises the SiGAT gap to a **common root cause**, not any single architectural lever.

## What was built (minimal — the enumerator already existed)
Discovery (Explore sweep) found the closed-signed-cycle enumerator is **already built, Rust-backed, and tested** — a new `hymeko_hre` Berge enumerator would have duplicated `hymeko_graph::cycle_enum` (and `berge.rs` is an *open* traversal, not a cycle enumerator, nor Python-bound). Reused as-is: `hymeko_neuro/hyperedge/n_tuples.py::construct_k_arrays(g, k, max_cycles, seed)` (→ `enumerate_k_cycles_rs` → `hymeko_graph::cycle_enum`, Davis-balance classify); `build_vertex_triad_incidence` and `encode_triads` are already arity-agnostic. **New work:** one `--cycle-k`/`--max-cycles` branch on `run_hsikan_rotor.py` swapping the triad source — the rotor (leakage-free) × k>3 combination, which the existing transductive k-tuple runners never crossed.

## Files touched
- `hymeko_neuro/experiments/runs/run_hsikan_rotor.py` (+~20 LOC: `--cycle-k`/`--max-cycles` branch via `construct_k_arrays`, balance from `edge_signs.prod`, incidence-head guard, provenance)
- `hymeko_neuro/tests/test_hsikan_rotor.py` (+2 tests: k=4 smoke+provenance, cycle_k-requires-endpoint guard)
- `docs/plans/2026-06-17-berge-kcycle-rotor/plan.{tex,pdf,tikz,mmd}` (new)
- CORE.YAML items touched: **none** (reused Rust already built+bound; no Cargo change).

## Tests
- `ruff check`: **PASS**. `pytest -p no:randomly test_hsikan_rotor.py`: **22 passed**. The k=4 path returns finite AUROC and enumerates cycles; the incidence head correctly refuses `--cycle-k`.

## Results — 5-seed A/B (k=3 vs k=4, matched 20k cap, tuned recipe, head bilinear)
`reports/kcycle_ab_20260617.jsonl` (30 cells) + `reports/kcycle_smoke_20260617.jsonl`.

| dataset | k=3 (via construct_k) | k=4 | Δ | per-seed Δ (k4−k3) | gate (k4+shuffle) |
|---|---:|---:|---:|---|---:|
| bitcoin_alpha | 0.8457 | 0.8450 | **−0.0007** | [−.003, −.003, +.001, −.000, +.001] | 0.519 [0.46–0.58] |
| bitcoin_otc | 0.8689 | 0.8691 | **+0.0002** | [+.001, +.001, −.002, +.001, −.000] | 0.526 [0.50–0.57] |

**Path parity (sanity):** k=3-via-`construct_k` (0.8457 / 0.8689) ≈ the `construct` triad baseline (0.8455 / 0.8685) — the σ-convention difference is immaterial, so the A/B isolates arity.

**Leakage gate:** k=4 under `--shuffle-train-signs` → 0.519 / 0.526 ≈ chance (cycles built from train edges only). Leakage-clean.

## Honest read
k=4 cycles ≈ k=3 triads (|Δ| < 0.001, within seed noise). Richer signed neighbourhoods do **not** help the leakage-free rotor line. This is the **fourth** independent lever to land flat:
1. geom_attn dead score (negative),
2. woken geom_attn score (negative),
3. rotor-relative projection (flat),
4. k=4 Berge cycles (flat).

Stronger readout, sign-aware readout, full-rotation comparison, *and* richer neighbourhoods all leave the rotor line at ~0.846 / ~0.869, ~0.04 short of the SiGAT target — every time, leakage-clean. Four independent architectural levers cannot all be coincidentally dead. **The gap is a common root cause, not a per-lever deficiency** — Dr. Hajdu's standing intuition ("something fundamental is missing").

## Performance
- k=4 enumeration with the 20k cap is fast (full 6-cell seed-0 smoke: 82 s wall). Peak RSS not separately re-measured (`/usr/bin/time` unavailable on this Git Bash); same scale/config as the prior 1724 MB measurement (10.5 % of the 16 GB cap), no graph-sized allocation added by the cap.

## §6.5 anti-patterns
None. The enumerator was reused, not rebuilt (§6.1 — the discovery prevented a duplicate `hymeko_hre` Berge enumerator); `--cycle-k` is a config branch on the existing driver (no new run file, §6.5 #13); default-unset reproduces the triad line.

## Decision / next step — STOP adding levers; diagnose the common root cause
Four flat levers say the bottleneck is upstream of all of them. Reaching for a fifth lever is the wrong move (and against the operating contract: analyse, don't keep guessing). Candidate common causes to discriminate next (cheap diagnostics, not new models):
1. **Is the SiGAT target measured under the same leakage-free / dedup / strict-train protocol?** If 0.884/0.902 came from a transductive (leakier) SiGAT, part of the "gap" is the leakage the rotor line deliberately gave up — i.e. an apples-to-oranges target. **Check provenance first.**
2. **Input-feature ceiling:** every lever feeds the same `_structural_features` → rotor input. If the input representation is saturated, no downstream lever can help — discriminate by swapping the input (e.g. richer structural features, or the transductive table) and seeing if AUROC jumps.
3. **Endpoint-head/training ceiling** independent of features.

Recommend (1) first — it is the cheapest and could reframe the entire chase. Awaiting user direction.

Memory: `project-hsikan-geometric-attention-berge` (updated).
