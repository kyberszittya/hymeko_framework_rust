# Stage signedkan-wip-organize — Slices A/B/C/D

**Date:** 2026-05-19
**Plan:** [`docs/plans/2026-05-19-signedkan-wip-organize/`](../docs/plans/2026-05-19-signedkan-wip-organize/) (4-format)
**Verdict:** **120 files moved, ${\sim}55$ imports rewritten, zero net test regression.** `hymeko_neuro/` top-level Python file count dropped from **152 → 44**; markdown plans/summaries at `hymeko_neuro/` root from 14 → 2. The `run_*.py` sprawl flagged in CLAUDE.md §6.5 #3 is now grouped under `hymeko_neuro/experiments/runs/` with an `ExperimentBase` ABC shell anticipating the future framework refactor (Slice H). Pre-reorg test state: **16 failed / 827 passed**; post-reorg: **16 failed / 827 passed** (identical — same env-drift failures, no new breakage).

## 1. Summary

`hymeko_neuro/` had accumulated $79\,119$ lines of Python and
$341$ import sites over the project's history; the `src/`
directory had $152$ files at its top level, $101$ of which were
`run_*.py` experiment scripts — the explicit anti-pattern §6.5 #3
that CLAUDE.md had been flagging since 2026-05-11.

This stage executed the **low-import-risk** phase of the reorg
plan: four file-move slices (A markdown archival, B run scripts,
C bench/eval helpers, D top-level demo scripts), each verified
with an import smoke + full pytest sweep before proceeding to the
next. The **high-import-risk** phase (E datasets package, F core
modules package, G misc helpers) is deferred to future sessions
with separate plans, per CLAUDE.md `feedback_one_phase_per_session.md`.

## 2. The four slices

### Slice A — markdown plans/summaries

12 files at `hymeko_neuro/` root → `hymeko_neuro/assets/docs/archive/`:

```
BENCHMARK_PLAN.md, DECISIONS.md, FUTURE_DIRECTIONS.md,
HSIKAN_GAP_CLOSING_PLAN.md, HSIKAN_SOTA_SUMMARY_2026_05_03.md,
OVERNIGHT_PLAN_2026_05_03.md, OVERNIGHT_SUMMARY_2026_05_02.md,
PAPER_DRAFT_2026_05_03.md, PLAN_CYCLE_ACCEL_2026_05_03.md,
PLAN_POST_SOTA_2026_05_03.md, PLAN_SGCN_GAP.md, RESULTS_LEDGER.md
```

`README.md` and `STRUCTURE.md` stayed in place (in-tree navigation
landmarks). One Python path reference updated:
`paperkit/build_ledger.py`'s `LEDGER` constant.

**Zero Python imports affected.**

### Slice B — 101 `run_*.py` experiment scripts

`hymeko_neuro/run_*.py` → `hymeko_neuro/experiments/runs/`,
plus a new `_experiment_base.py` shell anticipating the future
ExperimentBase/observer refactor (Slice H):

```
hymeko_neuro/experiments/runs/
    __init__.py
    _experiment_base.py     ← NEW shell ABC + observer protocol
    run_*.py × 101
```

#### Object-oriented commitment

Per the user's "modular, object-oriented, observer-pattern if
needed" guidance, the new
[`_experiment_base.py`](../hymeko_neuro/experiments/runs/_experiment_base.py)
ships the target shape for the future Slice H refactor:

- `EpochEvent`, `SeedEvent`, `RunEvent` — frozen dataclasses for
  the observer protocol.
- `ExperimentObserver` ABC with `on_run_*`, `on_seed_*`,
  `on_epoch_*` hooks (default no-op so observers only override
  what they care about).
- `ExperimentBase` ABC with abstract `build_dataset`,
  `build_model`, `build_optimizer`, `train_step`, `eval_step`;
  `run(...)` orchestration method `NotImplementedError`-stubbed
  (target shape committed; implementation deferred).
- Three concrete observers ready to use:
  - `StdoutObserver` — minimal per-epoch printing.
  - `JsonlObserver` — append per-seed JSONL (mirrors every
    current `run_*.py`'s manual write).
  - `CallbackObserver` — user-callbacks without subclassing.

The Slice H refactor (101 scripts → 101 thin configs +
ExperimentBase subclasses) is queued; tonight only ships the
*architectural commitment* at the directory layout, not the
migration.

#### Import-rewriting story

| Source pattern | Count | Action |
|:---|---:|:---|
| `python -m hymeko_neuro.run_X` in shell scripts / docs | 43 .sh + 12 docs | `s/hymeko_neuro\.src\.run_/hymeko_neuro.experiments.runs.run_/g` (sed pass) |
| `from hymeko_neuro.run_X` in Python | 39 sites | same sed pattern, applied to .py files |
| `from hymeko_neuro import run_X as Y` | 2 tests | regex pass to convert to `import hymeko_neuro.experiments.runs.run_X as Y` |
| `from ..run_X import` in src/demo + src/paperkit | 2 files | regex pass to absolute |
| `from .X import` (top-level, `X` ∈ remaining src) | 190 lines across 55 files | rewrite to `from hymeko_neuro.X import` |
| `from .X import` (top-level, `X` ∈ moved run scripts) | 91 lines | **kept relative** (both ends now in same package) |
| `from .X import` indented (function-body) | 36 lines across 9 files | second regex pass (missed by first — `^`-anchored) |
| `from .eval_metrics_full` in `run_final_cell.py` | 2 | absolute path to `hymeko_neuro.experiments.eval.eval_metrics_full` |

**Bulk import test post-rewrite**: 97 of 101 moved scripts import
cleanly. 4 fail with `ImportError: cannot import name 'MultiLayerSignedKANConfig'
from 'hymeko_neuro.models.mixed_arity_signedkan'` — confirmed
**pre-existing** by stashing the reorg and reproducing the same
error on the pre-reorg tree (symbol was deleted in the 2026-05-11
mixed_arity_signedkan refactor).

### Slice C — bench / eval helpers

5 files moved:

| From `src/` | To |
|:---|:---|
| `bench_abb_enum_walltime.py`, `bench_vertex_filter.py` | `experiments/bench/` |
| `eval_metrics_full.py`, `aggregate_phase3.py`, `annotate_mujoco_frames.py` | `experiments/eval/` |

Tests touched: `test_eval_metrics_full.py` (passes 5/5 post-rewrite).

### Slice D — top-level demo scripts

3 files:

| From | To |
|:---|:---|
| `src/demo_kinematic_mujoco.py` | `hymeko_neuro/demos/` |
| `src/demo_kinematic_pose.py` | `hymeko_neuro/demos/` |
| `src/DEMO_README.md` | `hymeko_neuro/demos/README.md` |

**`src/demo/` (the package — `cliques.py`, `gui.py`, `inference.py`,
`kinematic_classifier.py`, etc) DELIBERATELY LEFT IN PLACE.** It
has 5+ test/experiment cross-imports; reorganising it is its own
slice (deferred to a future session with a separate plan).

## 3. Pre/post structural diff

```
                                BEFORE        AFTER       Δ
hymeko_neuro/ top *.py     152           44         −108
hymeko_neuro/ root *.md         14            2          −12
                                                          ────
                                                         −120 files
```

New homes:

| Directory | File count |
|:---|---:|
| `hymeko_neuro/assets/docs/archive/` | 12 |
| `hymeko_neuro/experiments/runs/` | 103 (101 run_*.py + __init__ + _experiment_base) |
| `hymeko_neuro/experiments/bench/` | 3 (2 .py + __init__) |
| `hymeko_neuro/experiments/eval/` | 4 (3 .py + __init__) |
| `hymeko_neuro/demos/` | 3 (2 .py + README.md + __init__) |

## 4. Test status before and after

| Stage | Failed | Passed | Skipped | Total |
|:---|---:|---:|---:|---:|
| **Pre-reorg** (2026-05-19 audit, morning) | 16 | 827 | 19 | 862 |
| Post-Slice-A | 16 | 827 | 19 | 862 |
| Post-Slice-B (after first import-rewrite pass) | 17 | 826 | 19 | 862 |
| Post-Slice-B (after second indented-import pass) | 16 | 827 | 19 | 862 |
| Post-Slice-C | 16 | 827 | 19 | 862 |
| **Post-Slice-D** (final) | **16** | **827** | **19** | **862** |

**Zero net regression.** All 16 remaining failures are pre-existing
env-drift unrelated to the reorg:

- `test_hymeko_gomb.py` × 7 (CUDA + Bitcoin data, miniconda3 torch
  drift)
- `test_sota_smoke.py` × 3
- `test_cycle_cache.py`, `test_gomb_pgraph_driver.py`,
  `test_gomb_unrestricted_flag.py`, `test_konect_datasets.py`,
  `test_run_gomb_cycle_abb_compare.py`, `test_triton_kernels.py`
  (1 each)

## 5. Anti-pattern audit (CLAUDE.md §6.5)

- **§6.5 #3 (per-experiment scaffold duplication)**: directly
  targeted. The 101 scripts are now grouped, and the
  `_experiment_base.py` shell commits the architectural intent
  for the future migration.
- **§6.5 #4 (long single-file modules)**: not introduced. New
  `_experiment_base.py` is ${\sim}210$ LOC, under the 400-LOC
  early-warning threshold.
- **§6.5 #11 (globals)**: not introduced. `_experiment_base.py`
  uses an instance-level `_observers` list, not a module-level
  registry.

No `# noqa`, `# type: ignore`, or `#[allow]` introduced.

## 6. CLAUDE.md compliance

- **§1** core: `hymeko_neuro` is not in `CORE.YAML` (verified
  pre-reorg).
- **§2** plans: 4-format plan at
  `docs/plans/2026-05-19-signedkan-wip-organize/`.
- **§3** testing: full pytest after each slice; bulk import test on
  all 101 moved scripts; representative smoke import per slice.
- **§4** 16 GiB: irrelevant (file moves only).
- **§6.5** anti-patterns: see §5 above.

## 7. Object-oriented + observer-pattern architecture (Slice H target)

Today's `_experiment_base.py` ships the target shape that the
future Slice H migration will use. The canonical concrete-subclass
pattern:

```python
from hymeko_neuro.experiments.runs._experiment_base import (
    ExperimentBase, JsonlObserver, StdoutObserver,
)

class HsikanBitcoinExperiment(ExperimentBase):
    def __init__(self, dataset, hidden, n_epochs, **cfg):
        super().__init__()
        self.dataset = dataset
        self.hidden = hidden
        self.n_epochs = n_epochs
        ...

    def build_dataset(self, seed): ...
    def build_model(self, seed):  ...
    def build_optimizer(self, model): ...
    def train_step(self, model, optimizer, batch): ...
    def eval_step(self, model, val_loader): ...

# Usage:
exp = HsikanBitcoinExperiment(dataset="bitcoin_alpha", hidden=16,
                               n_epochs=60)
exp.add_observer(StdoutObserver())
exp.add_observer(JsonlObserver("results/hsikan_ba.jsonl"))
exp.run(label="hsikan_ba_5seed", seeds=range(5), epochs=60)
```

Each of the current 101 `run_*.py` files would shrink from
${\sim}200$ LOC (with duplicated argparse / train loop / JSON
emission) to ${\sim}40$ LOC (config + 5 method bodies). The
`ExperimentBase.run` method holds the shared scaffolding.

**Why I didn't migrate any of the 101 tonight**: per the
operating contract, ``one phase per session''. The directory
layout commits the architectural shape; the migration is the
Slice H phase, which will move scripts one at a time with full
diff review per script.

## 8. Open items

| Item | Why it stayed deferred |
|:---|:---|
| **Slice E** datasets to `src/datasets/` package | ~40 import sites; non-trivial test surface |
| **Slice F** 25+ core modules to `src/core/` package | ~150 import sites; the highest-risk slice |
| **Slice G** misc helpers to topical subdirs | ~50 imports; spread across many files |
| **Slice H** 101 `run_*.py` → ExperimentBase subclasses | The actual §6.5 #3 fix; deferred per one-phase-per-session |
| **`src/demo/` package reorganisation** | 5+ test/experiment cross-imports |
| `pyproject.toml` / setup-aware update | `hymeko_neuro/assets/scripts/*.sh` and CI configs may need a `--invocation` style note |

## 9. Bottom line

`hymeko_neuro` is now noticeably organised at the top level:
**152 → 44** files at `src/` top; **14 → 2** markdown plans at
root. The §6.5 #3 anti-pattern flagged by CLAUDE.md is now
mitigated structurally (scripts grouped, architectural shape for
the future ExperimentBase shipped). The reorg is **zero net test
regression** vs the pre-reorg baseline — every import was
rewritten or audited.

The high-risk parts (datasets package, core modules package,
ExperimentBase migration of 101 scripts) are queued as Slices
E-H. Each is a separate session per the one-phase-per-session
operating-contract rule.

---

**Companion artefacts:**

- Plan (4-format): [`docs/plans/2026-05-19-signedkan-wip-organize/`](../docs/plans/2026-05-19-signedkan-wip-organize/)
- ExperimentBase shell: [`hymeko_neuro/experiments/runs/_experiment_base.py`](../hymeko_neuro/experiments/runs/_experiment_base.py)
- Archived plans: [`hymeko_neuro/assets/docs/archive/`](../hymeko_neuro/assets/docs/archive/)
- Moved run scripts: [`hymeko_neuro/experiments/runs/`](../hymeko_neuro/experiments/runs/)
- Test status: same as pre-reorg morning audit (`reports/2026-05-19-pgip-chapters-validation.md` §3 documents the 16 env-drift failures).
