# hymeko_neuro merge — `signed_kan` + `signedkan_wip` → one package

**Date:** 2026-07-04 · **Branch:** `hymeko-neuro-migration` (off `fix-hsikan`) · **Commit:** `e7e5835`
**Plan:** `docs/plans/2026-07-04-hymeko-neuro-merge/` (4 artifacts, `plan.pdf` compiled)

## Summary

Merged the two forked Python packages into a single `hymeko_neuro` with a real hierarchy. The two
*cores* are kept **distinct**, not fused — this was the whole point of re-reading the record before
acting (§6.1):

- `hymeko_neuro/core/` ← `signed_kan` — the unified **pairwise** signed-GCN core (unify Phase 1–2,
  already the target `hymeko_rl` was migrated onto). Kept **pristine/boundaried**.
- `hymeko_neuro/hyperedge/` ← `signedkan_wip/src/core` — a **different algorithm** (k-uniform
  hyperedge gather, k≥3, per-arc inner spline, per-sign-branch outer spline, transductive). Phase 3
  was correctly **halted**; forcing one core would be a leaky abstraction, so they stay separate.

Library machinery is top-level (`models/ kernels/ graph/ data/ baselines/ eval/ runtime/ paperkit/
hymeko/`); use-case/domain code nests under `experiments/` (incl. `runs/` = the 121 `run_*.py`);
`demos/` and `examples/` are importable packages; non-code assets under `assets/` (datasets stay
**untracked**, `.gitignore` updated `/signedkan_wip/data/` → `/hymeko_neuro/assets/data/`).

## Method (deterministic, gated — scripts in session scratchpad)

1. **`migrate_hymeko_neuro.py`** (dry→apply, exhaustiveness-checked): 55 tree moves; ~970 files
   reference-rewritten via a **two-pass** rewriter (slash-paths then dotted-modules — the no-separator
   tokens `signed_kan`/`signedkan_wip` need `.` in module context but `/` in path context, so they
   cannot share one map); merge-safe moves; `parents[N]` depth-adjusted in 11 files.
2. **`fix_relimports.py`**: 49 cross-level relative imports (`from ..core`, `from .datasets`, …) that
   pointed at the old flat `src` level resolved against each file's *old* location and rewritten to
   absolute. Existence-checked so valid subpackage-internal relatives are left untouched.
3. Extended rewrite over `.rs/.hymeko/.sdf/.sbatch/.js/.env.example` (25 files — the extension list
   had missed Rust doc-comments and the `.hymeko` DSL comments).
4. Hand-fixes: the 4 `htl` sys.path bridges → direct `hymeko_neuro.eval.htl` imports; `mujoco_bridge`
   `MUJOCO_GL=egl` guarded off-Windows; removed wrongly-added `tests/__init__.py` (×2) to restore the
   rootdir/prepend test convention; two mis-prefixed run-script imports.

## Files touched

- **1281 renames** (code + tracked data/docs — blobs reused, no repo bloat), **338 modifications**
  (reference rewrites), **9 additions** (new package `__init__.py` + `test_core_boundary.py`),
  **2 deletions** (trivial `__init__.py` git scored as delete+add). Total **1630** staged.
- Previously-untracked datasets (reddit 368+318 MB, wiki, mnist, …) **excluded** from the commit and
  kept ignored — verified no non-rename add >500 KB is staged.

## CORE.YAML

None touched.

## Test results (gates)

| Gate | What | Result |
|------|------|--------|
| A | import-smoke: `core`, `hyperedge`, `eval.htl`, `hymeko_rl` htl bridge | **pass** |
| B | `pytest --collect-only` | `hymeko_rl` **572** tests / 0 err; `hymeko_neuro` **1973** tests / 0 migration-caused err |
| C | behavioural: `core/tests` (47) + `test_htl` (6) + `hymeko_rl/test_htl_reward` (6) | **59 passed** |
| D | `core/` isolation boundary (new `test_core_boundary.py`, AST audit) | **2 passed** |

`py_compile` over all of `hymeko_neuro`: **0 syntax errors**. Residual old-name refs in tracked
**source** (`.py/.sh/.md/.rs/.hymeko/.sdf`): **0**.

## Open issues / follow-ups

- **Pre-existing, not regressions:** collection still reports `optuna` (optional dep, not installed)
  and `fcntl` (Unix-only, Windows) import errors in a few `hymeko_neuro` tests — same as before the
  merge, carried over verbatim.
- **Binary/generated residuals left as-is:** `.pt` checkpoint pickles embed the old module path in
  their unpickling metadata (loading an old checkpoint needs a shim); `docs/book/book/*.html` is
  generated (regenerate from the rewritten `.md`); `.err/.jsonl` run logs are historical artifacts.
  A few Rust `signedkan_wip::src::…` `::`-path doc-comments were not rewritten (cosmetic).
- The scratchpad migration scripts (`migrate_hymeko_neuro.py`, `fix_relimports.py`) are the
  reproducible record; the in-tree fixes match them except the two forward-fixes applied directly
  (demos/examples lifted out of `assets/`, submodule-import corrections).
- Branch is not merged to `fix-hsikan` — left for review. Rollback: `git checkout fix-hsikan`.

## Provenance

Git SHA `e7e5835` (clean tree at commit). Host: win32, Python 3.12.13, pytest-8.4.2, torch (CUDA).
No seeds (structural refactor, no stochastic runs). Pre-migration state preserved as the WIP-snapshot
commit on `fix-hsikan`.
