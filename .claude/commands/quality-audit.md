---
description: Run a periodic software-quality audit across the Rust + Python codebase. Measures fan-in/fan-out, code length, cyclomatic complexity, identifier length, nesting depth, Gunning fog index on docs, and the OO-family metrics (DIT, method fan-in/fan-out, WMC, overloaded-method count). Produces a dated Markdown report under docs/quality/ and diffs it against the previous baseline.
argument-hint: "[--scope <crate>] [--full] [--since <YYYYMMDD>]"
---

# Software quality audit

Run a reproducible, metric-based quality check on this workspace. Output is one Markdown report at `docs/quality/YYYYMMDD.md` that the user can commit; consecutive runs diff against the most recent previous report so drift is visible without manual bookkeeping.

**Languages in scope:** Rust (primary — all `hymeko_*` crates, `parser/`, etc.), Python (`python/`, `scripts/scaling/`, `transforms/*/template*.py`). Ignore `target/`, `__pycache__/`, `generated/`, `archive.zip`, and any `.gitignore`'d path.

**Arguments (optional):**
- `--scope <crate>`: restrict Rust measurement to one crate (e.g. `hymeko_core`). Python is always full-tree unless combined with `--skip-python`.
- `--full`: emit top-20 offender lists per metric instead of the default top-5.
- `--since <YYYYMMDD>`: use a specific prior report as the baseline (default: most recent report under `docs/quality/`).

---

## Run-order and ground rules

1. **Never modify source code** during an audit. This is read-only: walk the tree, compute metrics, write one report file.
2. **Prefer installed tools** when available (faster, more accurate). Fall back to Grep/Read walks when not. Report which path was used in the "Methodology" section of the output so future runs are reproducible.
3. **Persist raw numbers, not judgments.** Classification into good/warn/critical happens from the thresholds table below — if the user disagrees with a threshold, they can adjust it in this file and the next run reclassifies without recomputing.
4. **Be conservative with shell commands.** `cargo check` and `cargo clippy` are safe; anything that writes to `target/` or modifies files is out of scope. `find`, `rg`, `wc`, `scc`, `tokei`, `lizard`, `radon`, `pylint` are all fine if installed.
5. **Handle failures gracefully.** If `lizard` isn't installed, state that in Methodology and use the fallback walk. If a crate fails to parse, note which and continue with the rest.

## Quick path: existing driver scripts

Two checked-in Python scripts operationalise this spec end-to-end — use them first:

```bash
# Compute raw metrics (writes /tmp/audit.json):
python3 scripts/quality/audit.py > /tmp/audit.json

# Post-process into the dated Markdown report:
python3 scripts/quality/report.py
```

Runtime is ~10 s for the whole workspace on a consumer CPU. The scripts require `lizard` and `textstat` (both installable via `pip install --user lizard textstat`); the rest uses the stdlib. If either is missing, install them before running rather than falling back — the fallbacks are noisier and the cost is a one-time minute.

If you edit the thresholds table below, update `scripts/quality/report.py`'s `T` dict at the top in lockstep so classifications stay consistent with the spec. The Markdown spec is the source of truth; the driver is the executable translation of it.

---

## Metrics

For each metric: (a) define what's measured, (b) state the unit and scope, (c) give the preferred tool and fallback, (d) give the threshold band used to classify findings.

### 1. Fan-in / fan-out (function-level)

- **Fan-out (v):** number of distinct functions that function `v` calls directly.
- **Fan-in (v):** number of distinct functions that call `v` directly.
- **Scope:** every `fn` / `pub fn` / `async fn` in Rust, every `def` / `async def` in Python. Methods inside `impl` blocks and classes count.
- **Tool:** `rust-code-analysis-cli --metrics` (Rust) + `radon raw`/`pyan3` (Python) if available. Fallback: Grep for call-site names across the workspace — cheap and sufficient for reporting top offenders.
- **Thresholds (fan-out):** ≤ 7 good · 8–15 warn · > 15 critical (Miller's 7±2).
- **Thresholds (fan-in):** no upper bound (widely-used helpers are fine); flag only the fan-out outliers. Report raw top-10 fan-in for informational "hot functions" ranking.

### 2. Code length

- **Per-function SLOC:** non-blank, non-comment lines inside the body.
- **Per-file SLOC:** full file excluding blank and pure-comment lines.
- **Per-module total:** summed across the crate/package.
- **Tool:** `scc --no-cocomo` or `tokei` for file/module totals; `lizard` for per-function SLOC. Fallback: Grep + Read.
- **Thresholds (function):** ≤ 50 good · 51–100 warn · > 100 critical.
- **Thresholds (file):** ≤ 500 good · 501–1000 warn · > 1000 critical.

### 3. Cyclomatic complexity (McCabe)

- **Definition:** `1 + count(decision-points)` where decision-points are `if`/`else if`, `match` arms (one per arm minus the default), `while`, `for`, `loop`, `?`-operator short-circuits (Rust), `and`/`or` short-circuits (Python), and exception `catch`/`except`.
- **Scope:** per function/method.
- **Tool:** `lizard` (polyglot — this is the preferred tool), `radon cc -s` for Python, `rust-code-analysis-cli --metrics` for Rust. Fallback: Grep for decision-point tokens inside function bodies.
- **Thresholds:** ≤ 10 good · 11–20 warn · > 20 critical.

### 4. Identifier length

- **Definition:** character length of each declared identifier name (function, method, struct field, variable, class attribute). Whitespace and sigils excluded.
- **Scope:** top-level + in-body declarations across all source files.
- **Tool:** language-aware AST walks are ideal but overkill — Grep with regex (`\b(fn|let|struct|const|static|type|trait|impl|enum|mod|def|class)\s+(\w+)`) is sufficient to build the distribution.
- **Thresholds (short):** length < 3 chars flag *unless* it's one of the conventional loop-var set `{i, j, k, x, y, z, t, n, m}`.
- **Thresholds (long):** length > 30 chars flag unconditionally.
- **Report:** histogram + list of both violation categories separately.

### 5. Embedded-if / nesting depth

- **Definition:** deepest level of nested control-flow blocks in any function. `if { if { … } }` = depth 2; `match { _ => if … }` = depth 2; each `for` / `while` / `match` / `if` / `else if` arm that has its own body block adds one level.
- **Scope:** per function/method.
- **Tool:** `lizard`'s cognitive-complexity gives this almost directly; otherwise count brace-depth increments inside function bodies (Rust uses `{}`, Python uses indentation).
- **Thresholds:** ≤ 4 good · 5–6 warn · > 6 critical.

### 6. Gunning fog index (documentation readability)

- **Definition:** `0.4 · ((words / sentences) + 100 · (complex_words / words))` where *complex_words* = words with 3+ syllables, excluding proper nouns and common compound suffixes.
- **Scope:** **documentation only** — doc comments (`///` for Rust, docstrings for Python), `README.md`, `docs/**/*.md`. Do NOT run on code comments like `//` or `#` inline remarks (they're not meant to be paragraphs). Do NOT run on test fixture strings.
- **Tool:** Python `textstat.gunning_fog` is the reference implementation. Fallback: hand-compute with the formula above on paragraphs of > 60 words (shorter paragraphs aren't statistically meaningful for Fog).
- **Thresholds:** ≤ 12 good (high-school graduate) · 13–16 warn (undergraduate) · > 16 critical (grad-level jargon density).
- **Target audience note:** this is a research codebase with legitimate technical vocabulary — flag outliers against the *repo median*, not against a consumer-docs baseline. If the repo median is 15, a file at 18 is the outlier; a file at 13 is fine.

### 7. Object-oriented / trait-family metrics

Rust has no classical inheritance but has an analogous trait hierarchy. Treat Rust traits + their impls as the OO substrate; treat Python classes normally.

#### 7a. Depth of Inheritance Tree (DIT)

- **Rust:** longest chain `Trait_n: Trait_{n-1}: ... : Trait_0` reachable from any user-defined `impl` block.
- **Python:** longest path in the class's MRO excluding `object`.
- **Tool:** AST walks. For Python, `pylint --load-plugins=pylint.extensions.design_analysis` reports `R0901 too-many-ancestors` which maps to DIT directly.
- **Thresholds:** ≤ 5 good · 6–7 warn · > 7 critical.

#### 7b. Method fan-in / fan-out

- Same as §1 but the bipartite target domain is methods on classes / trait impls. Cross-module method calls count; self-calls do not.
- **Thresholds:** same as §1.

#### 7c. Weighted methods per class / trait-object (WMC)

- **Definition:** `Σ cyclomatic_complexity(method)` across all methods of a class (Python) / all methods inside an `impl` block or trait definition (Rust).
- **Tool:** `lizard --output-format=csv` then group by class/impl. For Python, `radon cc` groups naturally.
- **Thresholds:** ≤ 50 good · 51–100 warn · > 100 critical.

#### 7d. Overloaded-method count

- **Rust:** methods with the same name defined for the same type across multiple `impl` blocks (including trait impls). Trait default methods overridden in an `impl` count as one overload per override.
- **Python:** methods with the same name decorated with `@typing.overload` (counted as a group), plus same-name methods across subclasses in a DIT chain.
- **Report:** list names with overload count ≥ 3, sorted descending. No hard threshold — high overload count is informational, not automatically bad.

---

## Output report format

Write to `docs/quality/<YYYYMMDD>.md` (today's date, UTC). If a file with the same date already exists, append `-2`, `-3`, etc. to avoid clobbering. Produce exactly this structure — machine-readable enough for future diffing, human-readable enough to review directly:

```markdown
# Software quality audit — <YYYY-MM-DD>

**Baseline:** <previous_report_filename_or_"none"> (<days> days prior)
**Commit:** <short git SHA + branch>
**Scope:** <crate filter or "full"> · <languages included>

## Summary

| Metric | Good | Warn | Critical | Δ vs baseline |
|---|---:|---:|---:|---:|
| Fan-out (functions) | … | … | … | +N / -N |
| Cyclomatic complexity | … | … | … | +N / -N |
| Function length (SLOC) | … | … | … | +N / -N |
| File length (SLOC) | … | … | … | +N / -N |
| Nesting depth | … | … | … | +N / -N |
| Short identifiers (<3) | — | — | <count> | +N / -N |
| Long identifiers (>30) | — | — | <count> | +N / -N |
| Fog index (docs) | … | … | … | +N / -N |
| DIT | … | … | … | +N / -N |
| WMC | … | … | … | +N / -N |

"Δ" means *change in offender count*, not change in metric value. A −3 in the Critical column means three functions that were critical last time are no longer critical (fixed, deleted, or moved out of scope). Flag regressions: "⚠ 2 new critical functions since baseline".

## Top offenders

For each metric, list the top 5 (or top 20 with `--full`) offenders as `<file>:<line> <symbol> — <metric> = <value>`. Don't include functions that were already in last baseline's top list — instead, add a dedicated "**Repeat offenders**" sub-table listing entries present on both this and the prior report. These are the ones drifting without attention.

### Fan-out
- <file>:<line> `<symbol>` — fan-out = <n>

### Cyclomatic complexity
- …

### Function length
- …

### File length
- …

### Nesting depth
- …

### Identifier length (short)
- <file>:<line> `<ident>` — length = <n>

### Identifier length (long)
- …

### Fog index (docs)
- <path> — fog = <float>

### DIT
- <file> `<type>` — depth = <n>

### WMC
- <file> `<class_or_impl>` — WMC = <n>

### Overloaded methods (informational)
- `<symbol>` — <n> overloads

## Repeat offenders (present in baseline)

<list per metric; empty list OK>

## New since baseline

<list per metric of offenders that weren't in the prior report; empty list OK>

## Resolved since baseline

<list per metric of offenders that were in the prior report but not this one>

## Methodology

- Tools used: <e.g. "lizard 1.17.15 for CCC and nesting; scc 3.2.0 for SLOC; grep fallback for fan-out (rust-code-analysis-cli not installed)">
- Files scanned: <count> Rust, <count> Python
- Files skipped: <list of skipped paths with reason — parse errors, generated, vendor>
- Elapsed: <seconds>

## Notes

<free-text section for anything notable that doesn't fit the tables — a big refactor landed, a new crate appeared, a false positive from a macro, etc.>
```

---

## Baseline diffing

Read the most recent prior report under `docs/quality/` (or the one specified by `--since`). For each metric's offender lists, compute three sets:

- **Resolved:** in baseline ∧ not in current → celebrate.
- **Repeat:** in baseline ∧ in current → drifting without attention, highlight.
- **New:** not in baseline ∧ in current → recent regression, investigate.

Δ column in the Summary table is the signed difference in offender count per severity band. A −2 in Critical and +3 in Warn means two critical items were fixed but three new warning-level items appeared. Worth noting when both move in opposite directions — that can signal a refactor split a big offender into several smaller ones.

If there is no prior report, say so in the Baseline line and skip the Repeat / Resolved / New sections (they'd be empty by construction).

---

## Skipped surfaces (always)

- `target/`, `**/__pycache__/`, `**/*.pyc`
- `generated/`
- `archive.zip`, `*.zip`, `*.tar.*`
- `paper/*/build/`, `paper/*/out/`, `*.aux`, `*.log`
- Test fixtures that are *data*, not code: `data/`, `scripts/scaling/fixtures/`
- Third-party vendored code if present (none currently in this repo)

---

## Running this periodically

To wire this into an automatic cadence:

- **Every Monday morning:** `/schedule create "run /quality-audit" --cron "0 9 * * 1"` (once the `schedule` skill is wired up, ask the user to confirm the cadence before creating the trigger).
- **After every big PR merge:** run manually — `/quality-audit` in the CLI.
- **Drift watch during a refactor:** `/loop 30m /quality-audit --scope hymeko_monitor` to see the metric move as you work.

Reports accumulate under `docs/quality/`; they're small (a few KB each) and a git-visible trail of code-health over time. Delete old ones only if they outgrow usefulness — don't auto-prune, the history *is* the value.
