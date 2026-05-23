# Quality improvement plan — derived from 2026-04-21 baseline

**Baseline:** `docs/quality/20260421.md`
**Goal:** bring all `critical` offenders into `warn` or better without inventing cleanup where none is needed.
**Non-goal:** zero warnings. Warnings that reflect real design constraints (deep trait-bounds on `Real`, test-setup fan-out) should stay.

The plan is ordered by *metric movement per hour of work*. Phase 1 fixes drop four critical bands simultaneously; Phase 3 is policy (no code change) and handles false positives the audit can't avoid. Confirm each item moves the numbers by re-running the audit after each phase, not just after each item — some fixes overlap.

---

## Phase 1 — High-leverage refactors (two functions, ~1 day)

Two non-test functions account for **both** CCN-critical entries (`>20`) and the top NLOC entry. Fixing them collapses the four worst rows of the Summary table in one sweep.

### 1.1 `interactive_console` — `hymeko_cli/src/main.rs:508`

**Current:** CCN=51, NLOC=348, nesting=8, fan-out=41 · *tops four lists simultaneously*.
**Why it got this bad:** the REPL dispatcher grew one big `match cmd.as_str() { "load" => {…}, "compile" => {…}, … }` with 20-odd inline handlers. Every new command adds 10-30 lines to the same `fn`.
**Proposed refactor:**

1. For each command arm, extract the body into `fn handle_<cmd>(session: &mut Session, rl: &mut Editor<…>, parts: &[&str], line: &str) -> CommandResult`. Keep one handler per source file if they grow, or group by family (`commands/filesystem.rs`, `commands/compile.rs`, `commands/query.rs`, etc.).
2. Replace the inline `match` with a two-line table: `match cmd.as_str() { "load" | "open" => handle_load(…), "compile" | "gen" => handle_compile(…), … }`. Or switch to a `HashMap<&str, fn(&mut Session, …)>` if you want command registration to be data-driven.
3. Move `interactive_console`'s prelude (banner print, `Editor` setup, history load/save) to `fn start_repl() -> Result<Session, Error>` and `fn persist_repl_state(session, rl)`.

**Expected delta:** CCN 51 → ~8 (remaining arms are 1-liners). NLOC 348 → ≤60. Nesting 8 → ≤3. Fan-out 41 → ~5. Also drops `hymeko_cli/src/main.rs` file SLOC from 990 (warn) to ~300 if handlers land under `hymeko_cli/src/commands/`.
**Effort:** 3–5 hours, mostly mechanical. No behavioural change if the handlers are pure extractions; run the REPL smoke-test from the earlier step-2 session to confirm parity.

### 1.2 `daemon::service::run` — `hymeko_daemon/src/service.rs:53`

**Current:** CCN=27, NLOC=126, nesting=6, fan-out=17.
**Why it got this bad:** three setup phases (Iceoryx egress, clique/raw publishers, ingress subscription) and the main event-loop `select!` all live in one `fn`.
**Proposed refactor:**

1. Extract each setup phase into its own async helper on `self`:
   - `async fn setup_egress(&self, node: &Node<…>) -> Result<Publisher<…>>`
   - `async fn setup_clique_publisher(&self, node) -> Result<Publisher<…>>`
   - `async fn setup_raw_ir_publisher(&self, node) -> Result<Publisher<…>>`
   - `async fn setup_ingress(&self, tx) -> Result<()>`
2. `run` becomes: build node → call `setup_*` in sequence → enter the event loop. The event loop itself can stay inline or be extracted to `async fn event_loop(&self, rx, publishers, is_running) -> Result<()>`.

**Expected delta:** CCN 27 → 8–12. NLOC 126 → ~30–40. Nesting 6 → ≤3.
**Effort:** 2–3 hours. Risk: getting lifetime/ownership right on the `Arc<Self>` captures in the publishers — write a quick smoke test that starts the daemon, publishes one IR, shuts down cleanly before the refactor so you have a regression guard.

---

## Phase 2 — Nesting fixes in logic code (~half-day, two functions)

Only these touch production code *and* have non-trivial CCN as well. The other "nesting-critical" entries are Python false positives (see §3.4) or small-body functions where nesting depth is structural.

### 2.1 `parse_blocks` — `hymeko_query/src/rewrite/template.rs:81`

**Current:** nesting=8, CCN=19 (warn), NLOC=48.
**Why:** classic template-parser state-machine — `if token == "{" { if next == "if" { if body… } }`.
**Proposed refactor:** flatten into one top-level `match` on token kind, with each arm calling a sub-parser `fn parse_if_block(…)`, `fn parse_for_block(…)`, etc. Each sub-parser takes the lexer and returns a `Block`.
**Expected delta:** nesting 8 → 4–5, CCN 19 → 10–12.
**Effort:** 2–3 hours. Covered by existing template tests — parity is easy to verify.

### 2.2 `emit_sysml` — `hymeko_emitter/src/emit_sysml.rs:81`

**Current:** nesting=9, CCN=5, NLOC=75.
**Why:** writing out nested SysML blocks with `for item in ir.decls { for arc in item.children { if arc.is_edge { … } } }`.
**Proposed refactor:** extract the inner `writeln!` cascade into `fn emit_hyperedge(out, ir, interner, edge_did) -> fmt::Result` (and sibling `emit_part_usage`, `emit_metadata`). SysML's structured blocks are naturally one emitter per block type.
**Expected delta:** nesting 9 → 3–4.
**Effort:** 1–2 hours.

---

## Phase 3 — Threshold / scope adjustments (no code change, ~1 hour)

Several audit findings are genuine false positives — the metric itself is measuring the wrong thing for that context. These are threshold fixes in `scripts/quality/report.py` and the spec in `.claude/commands/quality-audit.md`, *not* refactors.

### 3.1 Treat tests separately for NLOC + identifier length

**Observation:** 3 of the 5 NLOC-critical entries are tests with trivial CCN (≤7) and low nesting (≤3). The top-5 long identifiers are all Rust test names like `lowers_multi_bases_into_ir_and_preserves_tags_and_default_direction` — these follow the *correct* Rust convention (`snake_case_describing_what_is_verified`) and would get worse, not better, if shortened.
**Change:** in `scripts/quality/report.py`:
- Mark a function as `is_test` if its path contains `/tests/` or its name starts with `test_` or `bench_`.
- For NLOC: raise threshold to `(150, 300)` for tests; keep `(50, 100)` for production code.
- For long identifiers: raise threshold to 60 for test functions; keep 30 for everything else.

### 3.2 Fog index: separate `docs/plans/` bucket

**Observation:** top-5 Fog offenders are all in `docs/plans/` — internal research/planning prose is legitimately denser than public API docs. The current threshold (median + 5 = 18.9) catches legitimate technical writing for planning consumers.
**Change:** introduce two Fog buckets — `docs/guides/`, `docs/examples/`, `README.md`, and Rust doc-comments use `median + 3 / median + 5`. Everything under `docs/plans/` uses `median + 8 / median + 12` (informational, rarely warn).

### 3.3 Python nesting heuristic uses AST, not indentation

**Observation:** `scripts/scaling/analyze_scaling.py::fit_all_stages` reports `NLOC=8, CCN=2, depth=8` — impossible unless the heuristic is miscounting. Indentation-based nesting confuses the 4-space baseline when there are dict-comprehensions, multi-line function calls, or triple-quoted strings.
**Change:** in `scripts/quality/audit.py::py_max_nesting`, replace the indentation walk with an `ast` visitor that increments depth on `If`, `For`, `While`, `With`, `Try`, and `FunctionDef` (the real control-flow nesters) and ignores `ListComp`, `DictComp`, long-argument calls, etc. About 30 lines.

---

## Phase 4 — Ongoing hygiene

- **Re-run after each Phase 1 item** — actual deltas are the only proof the refactor worked:
  ```bash
  python3 scripts/quality/audit.py > /tmp/audit.json && python3 scripts/quality/report.py
  ```
  Commit the new report under `docs/quality/<YYYYMMDD>.md` alongside the refactor PR so the Δ column has data.
- **Weekly cadence**: wire `/schedule create "run /quality-audit" --cron "0 9 * * 1"` once Phase 1+3 is in — without Phase 3's threshold fixes the scheduled runs produce noisy Δ and trigger alert fatigue.
- **Budget rule**: if a refactor ticket touches a function currently in the top-10 of any metric, the author reruns the audit before merging. The incremental diff is cheap (3–5 s per function affected); the "did it actually move the metric?" question is always worth answering.

---

## Non-goals — explicitly NOT attacking these

Documenting these here so the next person doesn't relitigate them:

- **Test fan-out (≥28):** `test_anthropomorphic_generation.rs::axis_letter` calls 59 helpers; `test_clique_message_passing_matches_clique_view` calls 28. This is what integration tests *do* — shrinking them by "extract a method" makes the tests harder to read, not easier. Accept critical fan-out on tests until Phase 3.1 reclassifies them.
- **Short-identifier count (561):** most are struct fields (`a`, `b`, `n`, `m`), generic params (`T`, `S`, `R`), tuple binders, or lifetime names. Rust idiom favours short names at local scope. The informational list is useful for spotting individual cases; the aggregate count is not a health signal.
- **`QueryEngine::WMC=37`, `MatchContext::WMC=36`:** well within the `≤50` band. These are the workhorse types; concentrated behaviour is their job.
- **Rust DIT of 1 on `Real`:** a trait with `: Copy + Clone` legitimately has one super-trait step. Not a red flag.
- **Overloaded methods — `LalrpopParser::parse` (6 impls):** six trait impls on one type is exactly how one participates in six consumer traits. Informational, no action.

---

## Expected summary movement

If Phase 1 + Phase 2 + Phase 3 land:

| Metric | Current critical | Projected critical | How |
|---|---:|---:|---|
| Fan-out | 81 | ~40 | Test reclassification (3.1) drops most; `interactive_console` (1.1) removes one production entry |
| Cyclomatic complexity | 2 | 0 | Both (1.1, 1.2) drop into warn |
| Function NLOC | 5 | 1–2 | (1.1, 1.2) drop; tests reclassified (3.1) |
| Nesting depth | 17 | 6–8 | (2.1, 2.2) + Python AST (3.4) fix false positives |
| Long identifiers (>30) | 342 | ~50 | Test reclassification (3.1) drops the bulk |
| Fog index | 23 | ~5 | Plans-bucket split (3.2) reclassifies |

Numbers are estimates. The improvement plan's success criterion isn't hitting them exactly — it's that a Monday-morning audit produces *zero new criticals* sustained across three consecutive weeks.
