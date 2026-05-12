# Codebase rehaul — multi-phase plan

**Date:** 2026-05-11 (night)
**Reason:** the codebase has accumulated months of Claude-generated cartesian/duplicative output. Today's session closed the worst offender (cycles.rs 17 pyfunctions → 3 Strategy entries, 545 LOC of algorithm code moved to `hymeko_graph`, all touched files ≤300 LOC). The full audit (`reports/2026-05-11-anti-pattern-audit.md`) lists the remaining work.
**Constraint:** **one phase per session.** No more 4-hour firehose pushes. Each phase has an explicit acceptance gate; if it doesn't pass, the session ends without partial state on master.

---

## State as of 2026-05-11 night

**Done today (DO NOT redo):**

- CLAUDE.md amendments — Operating Principles (user is not a casual programmer; paradigm hierarchy: trait/struct > OO > FP > clean code; flat free-functions are last resort) + §6.5 Anti-Patterns (11 items with concrete repo examples).
- Audit: `reports/2026-05-11-anti-pattern-audit.md` — per-AP severity table, concrete file references.
- AP-1 Cartesian PyO3 surface — RESOLVED for per-vertex + top-K global families. 12 legacy pyfunctions deleted. 3 Strategy entries (`enumerate_cycles_rs`, `enumerate_top_k_cycles_rs`, `enumerate_top_k_cycles_entropy_rs`).
- AP-2 Algorithm code behind PyO3 boundary — RESOLVED for unsigned cycles + walks + color-coding + path-closure. ~1100 LOC moved to `hymeko_graph::{unsigned_cycles/, color_coding, path_closure, walks_unsigned, cycle_sampler, rand_lcg}`.
- AP-4 Long files — `cycles.rs` (2251 LOC monolith) DELETED. Replaced by `hymeko_py/src/cycles/` (5 files, max 269 LOC) and `hymeko_graph/src/unsigned_cycles/` (6 files, max 198 LOC).
- AP-9 Bypassing Strategy traits — `CycleSampler` trait + generic `enumerate_par<S>` driver established. Color-coding and path-closure now `impl CycleSampler`. Scorer/pruner dispatch centralized in `cycles/io.rs::pick_scorer` + the 3 Strategy entries.
- AP-11 Globals — `signedkan_wip/src/runtime_config.py` shipped with frozen `RuntimeConfig` / `TopKConfig` / `CycleCacheConfig` dataclasses. **`n_tuples.py` migrated** (7 reads → 1 config object).

**Open (the plan below addresses these):**

- AP-11 Globals — 9 of 10 offender files still scattered. `run_final_cell.py` (13 env reads), `hymeko_train_walker.py` (7), `hymeko_driver.py` (6), `signedkan.py` (4), `profile_stages.py` (4), `profile_hsikan_memory.py` (4), `cycle_cache.py` (4), `triton_kernels.py` (2), `splines.py` (2).
- AP-3 Per-experiment scaffold duplication — 98 `run_*.py` files, 32 with own AUC eval loops, 4 with own `train_val_split`. No `Experiment` framework exists.
- AP-4 Long Python files — `triton_kernels.py` (1345), `mixed_arity_signedkan.py` (1247), `run_final_cell.py` (780), `run_phase2_mixed_arity.py` (758), `cycle_cache.py` (714), `signedkan.py` (696), `splines.py` (644), plus 6 more in the 400–600 range.
- AP-7 String-typed config that should be enum — 35 `_kind: &str` sites remaining in `hymeko_py/src/cycles/`. The internal `match score_kind` ladders inside the 3 Strategy entries are correct local dispatchers but use `&str` where typed enums (`ScoreKind`, `PrunerKind`, `AbbMode`) would be safer.
- AP-10 `ulimit -v` on CUDA scripts — 1 remaining offender (`run_overnight_abb_validation_2026_05_11.sh`).
- Vision-side audit — `hymeko_nagare/src/ops/clifford_fir.rs` was flagged earlier in the day; never received an architectural review.
- The 30+ uncommitted files in the working tree (see `git status`) — the contract says working-tree hygiene matters; this is a separate cleanup pass.

---

## Phase 2 — RuntimeConfig migration (1 session, ~2 hr)

**Goal:** every `os.environ.get("HSIKAN_*"|"HYMEKO_*", ...)` in `signedkan_wip/src/` reads through `RuntimeConfig`. Zero scattered env-var lookups.

**Files in scope (priority order — touch in this order to minimize merge surface):**

1. `signedkan_wip/src/run_final_cell.py` (13 reads — by far the biggest offender)
2. `signedkan_wip/src/hymeko_train_walker.py` (7)
3. `signedkan_wip/src/hymeko_driver.py` (6)
4. `signedkan_wip/src/signedkan.py` (4 — careful, this is the model module)
5. `signedkan_wip/src/profile_stages.py` (4)
6. `signedkan_wip/src/profile_hsikan_memory.py` (4)
7. `signedkan_wip/src/cycle_cache.py` (4 — `_topk_fingerprint`-relevant; preserve fingerprint coverage)
8. `signedkan_wip/src/triton_kernels.py` (2)
9. `signedkan_wip/src/splines.py` (2)

**What changes:** extend `RuntimeConfig` with the env vars used by these files (likely 10–15 new fields across `TrainingConfig`, `KernelConfig`, `ProfileConfig` sub-dataclasses), then mechanical rewrite of each call site to read from the config.

**Acceptance gate:**

- `grep -rE 'os\.environ.*(HSIKAN|HYMEKO)' signedkan_wip/src/ --include="*.py"` returns **only the `runtime_config.py` itself** (the canonical parse).
- Bitcoin OTC smoke through `run_gomb_smoke.py` reproduces the existing 5-seed mean within ±0.001 AUC.
- `pytest -p no:randomly signedkan_wip/tests/` green.

**Estimated wall:** 2 hours focused work. Mostly mechanical edits.

---

## Phase 3 — `Experiment` framework + 5-script POC (1 session, ~3 hr)

**Goal:** stop the 98 `run_*.py` cartesian explosion. Introduce one `signedkan_wip/src/experiment.py` exposing `train_signed_link_prediction(cfg: ExperimentConfig, model: nn.Module) -> ExperimentResult`. Migrate 5 representative `run_*.py` scripts as proof-of-concept.

**Framework surface (sketch):**

```python
@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    seed: int
    n_epochs: int
    val_frac: float
    device: torch.device
    lr: float
    grad_clip: float
    # ... canonical training knobs

@dataclass(frozen=True)
class ExperimentResult:
    val_auc_best: float
    val_auc_per_epoch: list[float]
    train_loss_per_epoch: list[float]
    wall_s: float
    n_params: int
    extra: dict  # model-specific addons (alpha_k, etc.)

class ExperimentScaffold:
    """Owns: dataset load, train_val_split, train loop, AUC eval,
    JSON output, optional checkpointing. Knows nothing about the
    model — that's parametric via `model: Callable[[GraphCfg], nn.Module]`."""
    def __init__(self, cfg: ExperimentConfig): ...
    def run(self, model_factory: Callable, ...) -> ExperimentResult: ...
```

Each migrated `run_*.py` becomes ≤30 LOC: import scaffold, build config, declare model factory, call `scaffold.run(...)`.

**Migration targets (the 5 largest):**

1. `run_final_cell.py` (780 → ~150)
2. `run_phase2_mixed_arity.py` (758 → ~150)
3. `run_multi_domain_perf_bench.py` (698 → ~80)
4. `run_compare.py` (589 → ~80)
5. `run_gomb_smoke.py` (244 → ~50 — already touched today)

**Acceptance gate:**

- `wc -l signedkan_wip/src/experiment.py` < 400 (no monstrosity at birth).
- The 5 migrated scripts each ≤ 200 LOC (≤30 LOC is the target; 200 is the hard ceiling).
- Bitcoin OTC + Slashdot smokes through migrated `run_gomb_smoke.py` reproduce existing numbers within ±0.001 AUC.
- `pytest signedkan_wip/tests/` green.

**Estimated wall:** 3 hours. The 32 *unmigrated* `run_*.py` files stay as-is until Phase 4 (those are the bench scripts; less urgent than the train scripts).

---

## Phase 4 — Long-file decomposition (1 session, ~3 hr)

**Goal:** every Python source file in `signedkan_wip/src/` ≤300 LOC, decomposed by concern.

**Targets (priority order — biggest LOC win first):**

1. `triton_kernels.py` (1345 LOC) → `triton_kernels/` package: `forward.py`, `backward.py`, `highway.py`, `dispatch.py` (each ≤350 LOC)
2. `mixed_arity_signedkan.py` (1247 LOC) → `mixed_arity/` package: `model.py`, `attention.py`, `routing.py`, `mixin.py`
3. `cycle_cache.py` (714 LOC) → `cycle_cache/` package: `fingerprint.py`, `packer.py`, `lazy_loader.py`, `stats.py`
4. `signedkan.py` (696 LOC) → `signedkan/` package: `config.py`, `layer.py`, `model.py`, `multi.py`
5. `splines.py` (644 LOC) → `splines/` package: `bspline.py`, `catmull_rom.py`, `eval.py`

External imports stay flat via `__init__.py` re-exports (same pattern as today's `hymeko_gomb` split).

**Acceptance gate:**

- `find signedkan_wip/src -name "*.py" -size +400l | wc -l` returns **0**.
- All existing `pytest` green.
- Bitcoin OTC + Slashdot smokes reproduce existing numbers within ±0.001 AUC.

**Estimated wall:** 3 hours.

---

## Phase 5 — Typed enums at the Rust boundary (1 session, ~2 hr)

**Goal:** replace 35 `_kind: &str` sites in `hymeko_py/src/cycles/` with typed `ScoreKind` / `PrunerKind` / `AbbMode` enums living in `hymeko_graph`. Parse strings exactly once at the PyO3 boundary.

**Sketch:**

```rust
// In hymeko_graph/src/strategy_kinds.rs (new):
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreKind { Balance, FractionNegative, SignProductAbs, LowRoot }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrunerKind { None, Balance, Unbalanced, Davis }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbbMode { None, StartLocal, GlobalMin }

impl FromStr for ScoreKind { /* ... */ }
impl FromStr for PrunerKind { /* ... */ }
impl FromStr for AbbMode { /* ... */ }
```

`hymeko_py/src/cycles/io.rs::pick_scorer(&str)` becomes `ScoreKind::from_str(&str)` once at the boundary, then enum threads through. The 5 `match _kind` ladders inside the 3 Strategy entries become 3 ladders (one per shape) and stay typed-safe.

**Acceptance gate:**

- `grep -nE "_kind:\s*&str" hymeko_py/src/cycles/` returns **0** (kwargs at the PyO3 boundary stay `&str`, but they are parsed-then-discarded immediately).
- All existing tests green.
- Smokes reproduce.

**Estimated wall:** 2 hours.

---

## Phase 6 — Vision-side audit (1 session, ~2 hr)

**Goal:** apply the §6.5 anti-pattern audit to `signedkan_wip/src/vision/` and `hymeko_nagare/` (you flagged `clifford_fir.rs` hours ago; never circled back).

**Deliverables:**

- `reports/2026-05-12-vision-audit.md` — same shape as the graph-side audit.
- A Phase-2-style decomposition for any vision module > 400 LOC (`hymeyolo_q_smoke.py`: 587, `train_circles_ricci.py`: 555).
- Specific to clifford_fir.rs: confirm the closed-form Clifford backward is structurally clean OR enqueue a decomposition.

**Acceptance gate:**

- Audit report on disk with per-AP severity.
- Concrete next-action recommended per finding.

**Estimated wall:** 2 hours.

---

## Phase 7 — Working-tree hygiene + commit (1 session, ~1 hr)

**Goal:** the 30+ uncommitted files in `git status` get organized into 4–6 thematic commits (CLAUDE.md amendments, audit, cycles.rs refactor, Gömb model + tests, RuntimeConfig, reports). User-driven; I draft messages, user reviews and runs `git commit` per their `feedback_no_auto_commit.md` rule.

**Acceptance gate:** `git status` clean of green/added files. Branch ready for PR or push.

---

## Rules of engagement for these phases

- **One phase per session.** No combining. No "while I'm in here let me also...".
- **No new code until the audit gate passes** for the previous phase.
- **Numerical regression test on every phase** — Bitcoin OTC smoke + the existing 5-seed numbers reproduce ±0.001. If they don't, phase fails, branch resets.
- **No GPU-affecting changes during overnight runs.** Refactor sessions happen during idle GPU.
- **Reports are mandatory per CLAUDE.md §9.** Each phase ends with `reports/2026-05-1X-phase-N-<name>.md`.
- **Halt at the slightest CLAUDE.md §11 trigger.** Don't push through.

---

## What I will NOT do in any of these phases without explicit user OK

- Touch any model architecture (no "improvements", no "while we're here let me add"). Refactor only preserves behavior.
- Combine phases.
- Add new pyfunctions or new wrappers. The contract says config struct + Strategy from now on.
- Rewrite tests beyond what's needed to reflect renamed imports.
- Commit, push, force-anything.

---

## How this maps to memory

Save after writing this plan:

- `feedback_one_phase_per_session.md` — durable rule that fights my "while I'm in here" impulse.
- `project_codebase_rehaul_plan_2026_05_11.md` — pointer to this plan doc, list of phases + state-as-of.

These make the plan recoverable across sessions even if I forget the structure.
