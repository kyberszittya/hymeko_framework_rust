# Binding the shared Rust A* into Python — `hymeko.astar_plan` + a footstep planner

**Date:** 2026-08-06 · **Worktree:** `hymeko_humanoid` · **Branch:** `research/humanoid-com-lyapunov`
· **Git SHA at start:** `c98d8a63`. **Plan:** `docs/plans/2026-08-06-hotaru-pyo3-footstep/` (tex/pdf/tikz/mmd, gitignored).

---

## Summary

Made the "mirror" real: the humanoid footstep planner (Python) now runs the **same** `akoire::astar`
engine HOTARU's HIVE-delta planner uses, not a parallel Python copy. `akoire::astar` is exposed through
the existing `hymeko` PyO3 extension as **`hymeko.astar_plan(start, neighbours, is_goal, heuristic,
max_expansions)`** — the search problem is defined by Python callbacks over integer node ids, the search
loop runs in Rust. `scenarios/humanoid/footstep_planner.py` plans a foothold sequence over a
stepping-stone field on top of it, with a pure-Python fallback (the reference A* in
`hymeko_rl.control.graph_planner`) when the extension lacks the function.

## §1 (approved)

Added `akoire = { path = "../akoire" }` to `hymeko_py/Cargo.toml` — **user-approved §1 change**
(2026-08-06). `hymeko_py` is non-core and its `Cargo.toml` is *not* a CORE.YAML-locked pattern (only
`hymeko_core/Cargo.toml` + `parser/Cargo.toml` are); **no new external/pinned dependency** (pyo3/numpy
already present); **no dependency cycle** (akoire depends only on `parser`). `parser`/`hymeko_core`
untouched.

## §11 discovery — cross-worktree venv (why the build is isolated)

The naive "rebuild into `.venv`" assumption was **halted and reconciled** (CLAUDE.md §11): `hymeko_humanoid`
has **no local venv**; the only venv belongs to the **master** worktree, shared across many worktrees, and
`hymeko_py` has **diverged** between branches (and `akoire` does not exist on master). Rebuilding into that
venv would swap master's `hymeko` for this branch's — a cross-worktree, hard-to-reverse mutation. So the
build targets a **dedicated `hymeko_humanoid/.venv`** (gitignored), leaving every other worktree untouched.

## Design

- **`astar_plan`** (`hymeko_py/src/planner.rs`, thin wrapper — §6.5 #2, algorithm stays in `akoire`):
  nodes are `i64`, the edge label is the next node id so `AstarResult.actions` **is** the id path; the
  three Python callbacks are wrapped in closures the shared `akoire::astar` calls. A callback exception is
  captured in a `RefCell<Option<PyErr>>` and **re-raised** after the search (§6.4 — no silent swallow).
- **`footstep_planner.py`**: `SteppingField` (foothold grid, `id = j·W + i`, 4/8-connectivity, Manhattan /
  octile heuristic) + `plan_footsteps(field, start, goal, backend=…)`; `Backend` is an **enum**
  (auto/rust/python), not a string. The `auto` backend prefers `hymeko.astar_plan`, else the reference A*.
- **Torch-free fallback**: `hymeko_rl/__init__.py` eagerly imports the torch-backed policy stack, so a
  plain `import hymeko_rl.control.graph_planner` drags in torch. The fallback loads the (stdlib+numpy)
  reference module **by file path** (`importlib`, registered in `sys.modules` so its dataclasses resolve)
  — the *same* code (§6.1), kept lightweight.

## Files touched

| File | +/− | notes |
|---|---|---|
| `hymeko_py/Cargo.toml` | +3 / 0 | `akoire` path dependency (§1) |
| `hymeko_py/src/planner.rs` | +103 / new | `astar_plan` `#[pyfunction]` (thin binding) |
| `hymeko_py/src/lib.rs` | +5 / 0 | `pub mod planner;` + register the function (minimal) |
| `scenarios/humanoid/footstep_planner.py` | +177 / new | `SteppingField` + `plan_footsteps` + backends |
| `tests/test_footstep_planner.py` | +86 / new | python-backend + rust-backend + parity + error tests |
| `Cargo.lock` | (dep resolution) | `akoire` added to the hymeko_py subgraph |

## CORE.YAML items touched

None locked. The one dependency addition (`akoire` → `hymeko_py`) is the approved §1 change above.

## Test results

Two environments cover all paths (the rust-backend tests skip where `astar_plan` is absent):

| env | result | which |
|---|---|---|
| **isolated `.venv`** (astar_plan present, torch-free) | **6 passed, 1 skipped** | rust direct (`[1,2,3]`, unreachable→None, **raising callback propagates**), rust↔python **parity** (equal cost), + python-backend detour/unreachable/start=goal/octile |
| **master venv** (torch present, astar_plan absent) | **5 passed, 2 skipped** | python-backend all + the "request rust when absent raises" path; rust tests skip |

The Python footstep planner and the shared Rust engine return **equal-cost** plans on the same field —
the real code-sharing verification (the wall-detour optimum is length 7, matching the Rust `GridProblem`
unit test in `akoire`).

## Gate results

- **Rust:** `cargo test -p akoire` → 23 passed (no regression from the downstream dep). `planner.rs` is
  clippy-clean (`cargo clippy -p hymeko_py --no-deps`) and `rustfmt --check`-clean.
- **Python:** `ruff check` → clean (both files). `mypy` on `footstep_planner.py` → no errors in the file
  (the 5 reported are `types-PyYAML` stub-absence in *other*, unmodified modules — pre-existing/environmental).
- **Build:** `maturin develop` into the isolated venv, ~20 s (deps already compiled).

### Pre-existing issues (not introduced here, documented)

- `hymeko_py` has a **pre-existing** `clippy::too_many_arguments` on `enumerate_top_k_walks_rs`
  (`cycles/unsigned.rs`, 8 args — present at `HEAD`, unrelated to this change). Left unmasked (fixing it =
  a config-struct refactor of unrelated code); `astar_plan` (6 args) does not trip it.
- An accidental crate-wide `cargo fmt` reformatting of ~9 pre-existing `hymeko_py/src` files was
  **reverted** — the committed diff touches only `lib.rs` (+5) and the new `planner.rs`.

## §6.5 anti-patterns

None introduced. Thin binding (algorithm in `akoire`, §6.5 #2); no duplicated A* (the fallback loads the
existing reference module, §6.1); `Backend` is an enum, not a string (§6.5 #7); no globals; the one
indexing/`RefCell` path re-raises errors (§6.4). No new `unwrap`/`expect` on the Python boundary.

## Open issues / follow-up

- **Fallback torch-coupling**: the by-path load is a workaround for `hymeko_rl/__init__.py`'s eager torch
  import; making that `__init__` lazy is the proper (out-of-scope) fix.
- **Release build**: verified with a debug `maturin develop`; production uses `--release` (run_maturin.sh).
- **Callback overhead**: per-node Python calls cost GIL round-trips — fine at footstep scale; a pure-Rust
  concrete grid planner (no callbacks) is the follow-up if a hot loop needs it.
- **Wire to the WBC**: the planned footholds can feed `footstep_env`'s WBC swing executor (a real
  protective-step demo) — the next increment.

## Provenance

Git SHA `c98d8a63` at start. Rust: `rustc 1.96.1`, `cargo`/`clippy`/`rustfmt` (§10). Python: isolated
`.venv` (CPython 3.11.15, numpy 2.4.6, maturin 1.14.1, pytest 9.1.1); master venv (CPython 3.11) for the
cross-check. Host: macOS (darwin 25.5), Apple Silicon. Deterministic: A* seed-free; node id `j·W+i`.
