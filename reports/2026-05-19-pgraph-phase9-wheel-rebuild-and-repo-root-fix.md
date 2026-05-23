# Phase 9: Stale wheel + repo-root bug — 2026-05-19

## Summary

What the audit advertised as a "PyO3 binding rename" (carried in
the Phase 7 + Phase 8 open-issues lists) turned out to be **two
separate, unrelated bugs**:

1. **Stale `hymeko` wheel.** The Rust crate at
   `hymeko_py/src/cycles/per_vertex.rs:77` defines
   `enumerate_cycles_rs` — it always has. The installed
   site-packages wheel was just out of date. Rebuilding with
   `maturin develop --release` exposed it.
2. **`_repo_root()` in `run_gomb_msg_sweep.py` returned `signedkan_wip/`
   instead of the repo root.** A `parents[2]` typo (should be
   `parents[3]`). The bug stayed hidden because the binary-lookup
   helper falls back to `cargo run` if the path is wrong, but
   `parse_training()` has no fallback and FileNotFoundError'd as
   soon as the default training file was needed.

Both fixed; both pre-existing test failures resolved; the Gömb
P-graph sweep now runs end-to-end with the Friedler certificate
on the same JSONL row as the training metrics.

## Files touched

| File | Status | Change |
| --- | --- | --- |
| installed `hymeko` wheel | rebuilt | `maturin develop --release` from `hymeko_py/`; no code change |
| `signedkan_wip/experiments/runs/run_gomb_msg_sweep.py` | minor | `_repo_root()` returns `parents[3]` (repo root), not `parents[2]` (signedkan_wip/). Comment names the bug. |

## CORE.YAML items touched

None.

## Test results

Both pre-existing failures resolved:

| Test | Before Phase 9 | After Phase 9 |
| --- | --- | --- |
| `test_cycle_cache.py::test_triads_route_through_topk_path_when_enabled` | FAIL (stale wheel) | **PASS** |
| `test_gomb_pgraph_driver.py::test_run_single_gomb_smoke_subprocess` | FAIL (stale wheel) | **PASS** |

Full sweep:

| Suite | Result |
| --- | --- |
| `cargo test -p hymeko_pgraph` | 90 / 90 pass + 1 ignored doctest |
| `test_cycle_cache.py` | 13 / 13 pass (was 12 / 13) |
| `test_gomb_pgraph_driver.py` | 5 / 5 pass (was 4 / 5) |
| `test_hsikan_pgraph_mapping.py` | 7 / 7 pass |
| `test_hyperedges_m_per_vertex.py` | 7 / 7 pass |

## Quantitative result — end-to-end Gömb sweep

```
$ python -m signedkan_wip.experiments.runs.run_gomb_msg_sweep \
      --pgraph data/hsikan/sweep_msg_gomb.hymeko \
      --algorithm abb --dataset sbm_n200 --device cpu \
      --seed 0 --max-runs 1
```

JSONL output (single training row, abridged):

```json
{
  "model": "gomb",
  "dataset": "sbm_n200",
  "pgraph_units": ["gomb_fast", "gomb_fit"],
  "n_cycles": 524,
  "n_params": 16845,
  "topk": 12,
  "wall_s": 0.70,
  "test_auroc": 0.4918,
  "val_auroc": 0.4684,
  "canonical_full_status":   "PASS",
  "extension_full_status":   "PASS",
  "canonical_abb_status":    "PASS",
  "extension_abb_status":    "PASS",
  "strict_no_excess":        true
}
```

The substantive Phase 9 deliverable is that this row exists at all
— before today the Gömb side of the integration could only emit
analysis JSON; training was blocked by the stale wheel. Now every
Gömb sweep training run is stamped with the Friedler certificate
in the same line as `test_auroc`, ready to be filtered or grouped
downstream by `canonical_abb_status == "PASS"`.

The AUROC on `sbm_n200` is 0.4918 — near chance, expected for the
toy `sweep_msg_gomb.hymeko` P-graph (only 3 units, ~700 ms of
training). The Gömb training quality is not the Phase 9 result;
the end-to-end-pipeline-works fact is.

## Reflection — why Phase 7/8 mis-diagnosed this

The misleading thing was Python's `AttributeError: module 'hymeko'
has no attribute 'enumerate_cycles_rs'. Did you mean:
'enumerate_k_cycles_rs'?` — the `Did you mean` is a *string-
similarity hint*, not a "this was renamed" claim, and `enumerate_k_cycles_rs`
is the obvious lexical neighbour. Phase 7 read it as a rename
because the rest of the codebase contains `enumerate_k_cycles_rs`
extensively (from the 2026-05-11 cartesian-cycles refactor).
Phase 9 verified the truth by listing `dir(hymeko)` against the
installed wheel — `enumerate_cycles_rs` was absent there, but
present in the Rust source. The fix was a no-code rebuild + a
1-line typo correction.

Lesson worth saving: **before transcribing a `Did you mean`
suggestion as fact, list the installed module's actual surface**.
Memory entry below.

## Open issues and follow-up items

1. **The other 4 `signedkan_wip/data/...` references in the Gömb
   driver.** Search showed only the training-file path in
   `run_gomb_msg_sweep.py:245` hits this; the dataset CSVs DO live
   under `signedkan_wip/data/` (correctly so). No further wiring
   required.
2. **Phase 10 (multi-objective ABB)** — still on disk; ready for
   implementation. The Phase 8 regime-crossover result (m=16 wins
   at long epochs by 8 pp, loses at short by 3 pp) is the natural
   empirical motivation.
3. **Phase 11 (by-product NAS filter on real workloads)** — now
   that Phase 8 plumbing + Phase 9 unblocks Gömb training end-to-
   end, this is a clean half-day task.

## Experiment provenance

- **Git SHA:** `2ccaa4d12fae` (still uncommitted: phases 1–9 + cortical
  Slice 1 + earlier book regenerations).
- **Wheel rebuild:** `env -u VIRTUAL_ENV
  /home/kyberszittya/miniconda3/bin/python -m maturin develop
  --release` from `hymeko_py/`; ran in 17.25 s.
- **End-to-end Gömb sweep:** ran the driver on
  `data/hsikan/sweep_msg_gomb.hymeko` with `--max-runs 1
  --timeout-s 220`; completed in ~10 s wall.

## Acceptance check

- [x] No `CORE.YAML` items touched.
- [x] Two pre-existing test failures resolved.
- [x] All 32 Python tests in the Phase 8/9 surface area pass.
- [x] All 90 Rust pgraph tests pass.
- [x] End-to-end Gömb sweep emits a JSONL row with both training
      metrics AND the Friedler certificate.
- [x] §6.5 anti-pattern audit clean (no new patterns; one path-
      resolution bug fixed).
- [x] Report on disk.

## Memory entry

Adding a feedback memory: `feedback_did_you_mean_is_not_rename.md`
— **before transcribing a Python `Did you mean: X?` suggestion as
"the symbol was renamed", check `dir(module)` and the upstream
Rust source. The hint is string-similarity, not metadata.**
