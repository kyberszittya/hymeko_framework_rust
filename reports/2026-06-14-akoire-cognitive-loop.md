# Report — AKOIRE as a live cognitive synthesizer over the HyMeKo gatekeeper

**Date:** 2026-06-14
**Slug:** `akoire-cognitive-loop`
**Plan:** `docs/plans/2026-06-14-akoire-cognitive-loop/` (`plan.tex` / `plan.pdf` / `plan.tikz` / `plan.mmd`)
**Author:** Csaba Hajdu

## Summary

Implements option 3 of the AKOIRE discussion: the agent acts as the Layer-2
Cognitive OS, emitting HyQL refinement strings that the existing HyMeKo Layer-1
parser gate-keeps. A new **non-core** crate `akoire` wires the three flows of
`architecture/akoire/overview.sysml`:

- **synthesis** — `CognitiveSynthesizer` (Strategy trait) → `HymekoEngine`;
- **success loop** — accepted source → `Ambience` (generation++) → next context;
- **error loop** — `parser::ParseError` rendered into `ErrorFeedback` → next context.

`CognitiveLoop` drives the loop under a hard `max_rounds` cap. The agent seam is
the `CognitiveSynthesizer` trait: `ScriptedSynthesizer` (a fixed queue) makes the
loop deterministic and testable and stands in for "the agent's pre-decided moves";
an LLM-backed synthesizer is the production binding and drops in with no loop
change (sketched in `synthesize.rs`).

The crate is a **control-structure harness, not a tensor backend**. HyMeKo's
COO→CSR commit is delegated to `hymeko_core` in the real system; here it is
represented by an owned `Ambience` (always-parseable source + monotonic
generation + extracted edge names).

## Files touched

| Path | Action | Lines |
|---|---|---|
| `Cargo.toml` (root) | additive: `"akoire"` added to `[workspace].members` | +1 |
| `akoire/Cargo.toml` | new | 25 |
| `akoire/src/lib.rs` | new — crate doc, re-exports, unit tests | 165 |
| `akoire/src/context.rs` | new — `Intent`/`Objectives`/`Kyosei`/`Ambience`/`Goal` | 154 |
| `akoire/src/synthesize.rs` | new — `CognitiveSynthesizer` + `ScriptedSynthesizer` | 80 |
| `akoire/src/engine.rs` | new — `HymekoEngine` gatekeeper facade | 103 |
| `akoire/src/loop_driver.rs` | new — `CognitiveLoop`, `LoopReport`, `Termination` | 135 |
| `akoire/tests/integration.rs` | new — end-to-end self-correction scenario | 45 |
| `akoire/benches/loop_bench.rs` | new — criterion per-round benchmark | 40 |
| `docs/plans/2026-06-14-akoire-cognitive-loop/plan.{tex,pdf,tikz,mmd}` | new | — |
| `reports/2026-06-14-akoire-cognitive-loop.md` | new (this file) | — |

Total new Rust: ~747 lines incl. tests/bench.

## CORE.YAML items touched

**None.** `parser` and `hymeko_core` are `lockdown: full`. `akoire` depends only
on `parser`'s public API (`parse_description`, `parser::ast`) and edits no locked
file. Root `Cargo.toml` is not a CORE item; the one-line member addition is a
non-core additive edit. No pinned dependency was added, removed, or
version-changed.

## Test results

> **Not executed in the authoring environment.** The sandbox has `pdflatex`
> (the plan PDF built and is committed) but **no `cargo`/`rustc`**. Per CLAUDE.md
> §3 and the in-flight-claims honesty rule, no green result is asserted that was
> not run. The tests are written and statically reviewed against the actual
> grammar (`parser/src/hymeko.lalrpop`) and the parser's own test fixtures
> (`parser/tests/using_alias.rs`).

Tests authored (to be run on the dev host):

| Layer | Test | Asserts |
|---|---|---|
| unit | `engine_accepts_valid` | `Accepted`, generation = 1, `joint` in edge names |
| unit | `engine_rejects_malformed` | `Rejected`, non-empty feedback, state unchanged |
| unit | `scripted_exhausts` | yields queue then `None` |
| unit | `loop_converges_first_try` | `Converged`, rounds=1, accepted=1, rejected=0 |
| unit | `loop_self_corrects` | `[malformed, valid]` ⇒ `Converged`, rejected=1 |
| unit | `loop_respects_cap` | all-malformed ⇒ `CapReached`, rounds=cap (contract regression) |
| unit | `loop_exhausts_when_synthesizer_empties` | `Exhausted` when queue < goal |
| integ | `self_correction_then_incremental_convergence` | broken→feedback→partial→complete, both edges present |

### Commands to run on the dev host

```
cargo test  -p akoire
cargo clippy -p akoire --all-targets -- -D warnings
cargo fmt   -p akoire -- --check
cargo llvm-cov -p akoire --lcov --output-path target/akoire-lcov.info
cargo bench -p akoire          # criterion: median / IQR per round
radon cc -a -nc akoire/src     # n/a for Rust; use rust-code-analysis if desired
```

## Performance results

Not measured (no `cargo` in sandbox). Budget from the plan: pure-CPU, peak RSS
< 64 MB (far under the 16 GB cap — the loop holds a handful of small strings);
per converged 1-round run dominated by `parse_description` on a < 1 KB input,
target median < 200 µs on the dev host. The criterion bench
(`cognitive_loop_converge_1_round`) reports median/IQR over its sample once run.

## New / removed dependencies

- **Added (non-core):** `parser` (in-repo `path` dependency; public-API use only).
- **Added (dev):** `criterion = "0.8"` — matches the major already used by
  `parser` to avoid a duplicate criterion build; permitted by `tools.yaml`
  (criterion `major_version: "0"`).
- No runtime/external dependency added; no pinned dependency changed.

## Plan → implementation deltas

- The plan sketched `CognitiveLoop<S, G: Goal>` with a separate goal object. During
  implementation the `Goal` predicate was folded onto `Objectives` (`impl Goal for
  Objectives`) so the target has a **single source of truth** (the synthesizer's
  objective input and the loop's termination check are the same value), removing a
  duplicate-state hazard (§6.1). `CognitiveLoop` is therefore generic only over
  `S: CognitiveSynthesizer`. The `Goal` trait is retained for future termination
  criteria (node-count, balance, entropy — cf. `hymeko_query::entropy`).
- Feedback is rendered with `{:?}` (not `{}`): `parser`'s `Token`/`LexError` derive
  `Debug`, not `Display`.

## §6.5 anti-patterns

None introduced. Trait-based Strategy (`CognitiveSynthesizer`, `Goal`); sealed
enums-with-data (`EvalOutcome`, `Termination`) instead of string modes (#7);
`HymekoEngine` is a thin Facade over the locked parser, no algorithm code crosses
into it (#2); no Cartesian wrapper family (#1/#5); no globals (#11); no `_v2`
files (#13); no preamble-before-action. One `debug_assert!` precondition guards
`max_rounds >= 1` (§8). No `unwrap`/`expect` in non-test code (§6.4).

## Open issues / follow-ups

1. **Run the suite on the dev host** (commands above) before this is considered
   complete per CLAUDE.md §3. This report does not claim a green suite.
2. **Pre-existing drift (not addressed, out of scope):** `parser/Cargo.toml` pins
   `lalrpop >= 0.22.2` while `CORE.YAML` pins `lalrpop = "=0.20.2"`. Both are
   `lockdown: full`; reconciling them is a CORE decision (§1) and was left
   untouched.
3. **Real Layer-1 commit.** `Ambience` currently stores accepted source + edge
   names, not an actual `TensorCoo`/`TensorCsr`. Wiring the commit to `hymeko_core`
   is the natural next step and would need its own plan (and likely touches no core
   if done through a public builder API; verify first).
4. **LLM synthesizer.** Implement `LlmSynthesizer` against the chat backend; the
   prompt should carry `ambience.source()`, `intent`, `objectives`, and
   `last_error` so the error loop actually drives self-correction.

## Provenance

- Git SHA: not captured (sandbox `git` calls timed out; capture on the dev host
  with `git rev-parse --short HEAD` — working tree is dirty with exactly the files
  listed above).
- Environment: authoring sandbox = Linux mount, TeX Live 2022 (`pdflatex`); **no
  Rust toolchain**. Build/test host = the user's machine (Rust toolchain per
  `tools.yaml`).
- Seeds: tests are deterministic (scripted synthesizer, fixed fixtures); no RNG.
- Datasets: none; fixtures are inline string constants validated against the
  grammar.
