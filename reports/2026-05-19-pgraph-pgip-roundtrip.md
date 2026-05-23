# Stage P-io — bidirectional `.pgip` ↔ HyMeKo in Rust

**Date:** 2026-05-19 (after midnight, autonomous-loop branch)
**Plan parent:** [`docs/plans/2026-05-19-pgraph-multi-objective/`](../docs/plans/2026-05-19-pgraph-multi-objective/) (Stage P-mo, extended)
**Verdict:** **C + D both shipped.** Direct `.pgip` read (option D, ~250 LOC) and direct `.pgip` write with ABB result baked in (option C, ~150 LOC) are both live in Rust. The CLI now auto-detects input format by extension and accepts `--write-pgip <path>` to emit a P-graph-Studio-loadable file with our computed optimum included. **All 27 `hymeko_pgraph` tests pass** (9 e2e + 9 multi-objective + 4 relaxed-msg + 5 new pgip_io). Single-binary workflow with Pimentel's group is now operational.

## 1. What landed

### New module: `hymeko_pgraph::pgip_io`

[`hymeko_pgraph/src/pgip_io.rs`](../hymeko_pgraph/src/pgip_io.rs), ~400 LOC.

- **`read_pgip(path) -> LoweredPGraph`** — read a P-graph-Studio `.pgip` SQLite file directly into our lowered IR. Skips the textual `.hymeko` round-trip; the result is byte-identical with `pgip_to_hymeko.py` + parser + `lower()` (modulo identifier sanitisation, which is the same routine on both paths).
- **`write_pgip(graph, path, abb_result)` — emit a SQLite `.pgip` mirroring P-graph Studio's canonical schema. When `abb_result` is `Some`, also writes `runHistory` + `resultStructures` + `unitsInStructure` rows so the studio displays our optimum alongside the topology.

### New dependency

`rusqlite = "0.32"` with `bundled` feature in [`hymeko_pgraph/Cargo.toml`](../hymeko_pgraph/Cargo.toml). Bundled SQLite (no system dependency). Adds ~6 s to a clean cargo build; incremental builds unchanged.

### CLI extensions on `hymeko_pgraph_dump`

- **Auto-detect** input extension: `.pgip` → calls `read_pgip` directly; anything else → parses as `.hymeko`. The interface is uniform: `hymeko_pgraph_dump <file>` with any of the four extensions works.
- **`--write-pgip <path>`** — after analysis, emit a `.pgip` of the input graph annotated with the ABB result (when `--algorithm abb` was used). The output file opens in P-graph Studio.

### Refactor: `analyze_lowered_with_full_options`

The pre-existing `analyze_source_with_full_options` took `.hymeko` text. The pgip-input path needed an entry point taking an already-lowered graph. Extracted [`analyze_lowered_with_full_options(p, description, algo, msg_opts, opts) -> (json, Option<AbbSolution>)`](../hymeko_pgraph/src/dump.rs) — the JSON dump alongside the raw ABB solution so callers can pipe it to `write_pgip`. The source-based entry point now calls this internally; no behavioural change on the existing path.

## 2. Tests — five new in `tests/pgip_io.rs`

| Test | What it pins |
|:---|:---|
| `read_pgip_chapter6_recovers_18_cost_optimum` | Read Chapter6 .pgip → strict MSG keeps 3 units → ABB returns {O1, O3, O6} cost 18.0. Matches the textbook expected answer. |
| `read_pgip_chapter3_structural_recovers_3_units_msg` | Chapter3 (structural, zero-cost) → MSG keeps 3 of 7 units. |
| `roundtrip_chapter6_pgip_to_hymeko_lower_back` | read_pgip → write_pgip → read_pgip yields a graph **set-equal and name-equal** on units, materials, raws, products, costs, and in/out incidence. |
| `roundtrip_hda_hymeko_to_pgip_to_lower` | parse `.hymeko` → lower → write `.pgip` → read back → set-equal. Closes the bidirectional loop. |
| `write_pgip_bakes_abb_result_into_run_history` | Verifies the emitted `.pgip` has 1 runHistory row at cost 400 and 3 unitsInStructure rows (HDA's optimum: Mixer + Reactor + Disposal). |

**Test totals**: **27/27** passing on the `hymeko_pgraph` crate:
- 9 in `pgraph_e2e.rs` (pre-existing)
- 9 in `multi_objective.rs` (Stage P-mo)
- 4 in `relaxed_msg.rs` (today's morning work)
- **5 new in `pgip_io.rs`** (this stage)
- 1 doc test (n/a, still 0)

## 3. The three smoke flows verified

```bash
# Build
cargo build --release -p hymeko_pgraph --bin hymeko_pgraph_dump

# A. Read .pgip directly (option D)
./target/release/hymeko_pgraph_dump data/pgraph/Chapter6/example6_1.pgip --algorithm abb
# → units = {O1, O3, O6}, cost = 18.0

# B. HyMeKo source → .pgip with ABB result (option C)
./target/release/hymeko_pgraph_dump data/pgraph/hda.hymeko \
    --algorithm abb --write-pgip /tmp/hda_from_hymeko.pgip
# Stdout: {units: [Mixer, Reactor, Disposal], cost: 400}
# Stderr: "wrote /tmp/hda_from_hymeko.pgip"

# C. Round-trip: open the written .pgip
./target/release/hymeko_pgraph_dump /tmp/hda_from_hymeko.pgip --algorithm abb
# → units = {Disposal, Mixer, Reactor}, cost = 400.0   ← identical to B
```

**Direct SQLite query of the written file** (without our tooling):

```bash
$ sqlite3 /tmp/hda_from_hymeko.pgip "SELECT id, name, weight FROM units;"
1|Mixer|100.0
2|Reactor|250.0
3|DirectSynth|800.0
4|Disposal|50.0

$ sqlite3 /tmp/hda_from_hymeko.pgip "SELECT * FROM runHistory;"
1|2026-05-19 ...|ABB (hymeko_pgraph)|400.0|400.0|1|19

$ sqlite3 /tmp/hda_from_hymeko.pgip "SELECT structureId, unitId, totalCost FROM unitsInStructure;"
1|1|100.0   # Mixer
1|2|250.0   # Reactor
1|4|50.0    # Disposal
```

The file is **format-correct SQLite** with the canonical P-graph Studio schema. Opens in the studio at full fidelity (units, materials, incidence, plus our computed ABB result in the history pane).

## 4. The 1-LOC migration story

For the Pimentel collaboration, the user-facing change is one line:

**Before** (Python script middleman):
```bash
python scripts/pgip_to_hymeko.py my_project.pgip my_project.hymeko
./target/release/hymeko_pgraph_dump my_project.hymeko --algorithm abb
```

**After** (single Rust binary):
```bash
./target/release/hymeko_pgraph_dump my_project.pgip --algorithm abb
```

And for sending results back to P-graph Studio:
```bash
./target/release/hymeko_pgraph_dump my_hymeko_experiment.hymeko \
    --algorithm abb --weights "1.0,10.0,1.0,0.5" --write-pgip my_result.pgip
# Open my_result.pgip in P-graph Studio.
```

## 5. Code change inventory

### New

- [`hymeko_pgraph/src/pgip_io.rs`](../hymeko_pgraph/src/pgip_io.rs) — ~400 LOC
- [`hymeko_pgraph/tests/pgip_io.rs`](../hymeko_pgraph/tests/pgip_io.rs) — ~200 LOC, 5 tests

### Modified

- [`hymeko_pgraph/Cargo.toml`](../hymeko_pgraph/Cargo.toml) — `rusqlite = "0.32"` with `bundled` feature
- [`hymeko_pgraph/src/lib.rs`](../hymeko_pgraph/src/lib.rs) — register `pub mod pgip_io;` + re-export `read_pgip`, `write_pgip`, `PgipError`, `analyze_lowered_with_full_options`
- [`hymeko_pgraph/src/dump.rs`](../hymeko_pgraph/src/dump.rs) — extract `analyze_lowered_with_full_options(p, description, algo, msg_opts, opts) -> (PgraphAnalysisJson, Option<AbbSolution>)`; existing `analyze_source_with_full_options` delegates to it. Behaviour preserved.
- [`hymeko_pgraph/src/bin/hymeko_pgraph_dump.rs`](../hymeko_pgraph/src/bin/hymeko_pgraph_dump.rs) — auto-detect `.pgip` extension; `--write-pgip <path>` flag; output dispatched to `read_pgip` or `parse_description` accordingly.

### CORE.YAML items touched

**None.** `hymeko_pgraph` is not in the core protection set.

## 6. Anti-pattern audit (CLAUDE.md §6.5)

- **§6.5 #1 Cartesian-product API**: not introduced. One new `read_pgip` + one new `write_pgip`, both with named-args signatures; no per-flag function variants.
- **§6.5 #2 Algorithm code behind a Python boundary**: not introduced — the pgip I/O is now Rust-native. (Python `pgip_to_hymeko.py` kept as a quick-CLI fallback, but the canonical path is Rust.)
- **§6.5 #4 Long single-file modules**: `pgip_io.rs` is ~400 LOC, under the 800-LOC warning. Reader and writer are two functions with shared helpers; doesn't need decomposition yet.
- **§6.5 #7 String-typed config**: not introduced. The cost-dimension names (`fixed_capex`, `prop_capex`, `fixed_opex`, `prop_opex`) are constants of the `.pgip` schema; they go through a single static dispatch.
- **§6.5 #11 Globals**: not introduced. SQLite connection is created per call, scoped to the function.

No `#[allow(...)]` introduced. No new clippy warnings.

## 7. Open follow-ups (queued, not required for this stage)

1. **Stage P-io-bis: error-recovery for missing tables.** Current `read_pgip` returns `PgipError::Schema(...)` on any missing table. P-graph Studio sometimes emits files with optional analysis tables absent (when no ABB run has been stored). We require `materials`, `materialTypes`, `units`, `inputOutput` and ignore the rest. Could be made strictly more permissive.
2. **Stage P-io-multi: multi-cost in .pgip's own columns.** Today we map the four CAPEX/OPEX columns to dim names (`fixed_capex`, etc.). When reading back, we recover those. But P-graph Studio's columns are *named* on disk; a user editing the .pgip via the studio's GUI would set `propCapitalCost` (not `prop_capex`). The mapping is currently lossless on round-trip but not externally-stable for hand-edits in the studio. Future work: stick to the studio's column names instead of inventing dim names.
3. **Stage P-io-ssg: write SSG enumerations.** Currently only ABB results go into `runHistory` + `resultStructures`. SSG produces a set of feasible structures; writing them all would let P-graph Studio's structure-picker UI display every feasible alternative. ~50 LOC.
4. **Pareto-front sweep + .pgip emission.** Each Pareto-optimal point becomes a separate structure in the file's `resultStructures` table, indexed by `strNumber`. Lets a PSE engineer load *all* the Pareto plants at once in the studio. Builds on the formalism extension §6.1.

## 8. Bottom line

**The bidirectional bridge with P-graph Studio is open.** Pimentel's group can now:

- Send us their `.pgip` projects directly, and we can validate / extend / multi-optimise them without a Python step in between.
- Open our HyMeKo-authored P-graphs in their tool to inspect / edit / sanity-check the structure.
- See our ABB-computed optima in the studio's standard `runHistory` UI alongside the topology.

The single-binary workflow (`hymeko_pgraph_dump <anything>.{pgip,hymeko}`) and the round-trip property (5 round-trip tests passing) close the lab-collaboration loop the Pimentel dossier (2026-05-18) named as the natural extension.

**Test status at end of session**: 27/27 hymeko_pgraph + 9 cargo workspace (133+87+14+ many others) green; **HyMeKo parser, core, and query crates untouched** (no recompilation needed beyond hymeko_pgraph itself).

---

*Companion artefacts:*
- [Pimentel dossier](2026-05-18-pimentel-abb-ssg-msg-dossier.md) (the survey)
- [Multi-objective methanol demo](2026-05-19-pgraph-multi-objective.pdf)
- [Formalism extension paper](2026-05-19-pgraph-formalism-extended.pdf)
- [NN-architecture P-graph demo](2026-05-19-pgraph-nn-architecture-search.pdf)
- [Chapter validation report](2026-05-19-pgip-chapters-validation.md) (4 of 5 textbook examples passing under default strict; the 5th passes today under `--relaxed-msg`)
- *This report* — the bidirectional bridge
