# signedkan_wip reorganisation — phase 2 (Slices E, G-1, F, H-pilot)

**Date:** 2026-05-19 (evening, after Slices A-D)
**Plan parent:** [`docs/plans/2026-05-19-signedkan-wip-organize/`](../docs/plans/2026-05-19-signedkan-wip-organize/)
**Verdict:** **Three deferred slices shipped + Slice H pilot + 10 new tests.** `signedkan_wip/src/` top-level Python file count dropped from **152 (start of day)** → **12** (92 % reduction). Three new properly-curated packages (`src/datasets/`, `src/kinematic/`, `src/core/`); the `ExperimentBase` ABC plus `SimpleExperiment` adapter and observer protocol are fleshed out; one representative `run_*.py` (`run_early_stop.py`) migrated to demonstrate the OO pattern; 10 unit tests pin the observer machinery. **Zero net test regression**: pre-reorg 16 failed / 827 passed; post all five slices, **16 failed / 837 passed** (+10 from the new test file).

## 1. What landed (cumulative)

| Slice | Move | Files | Where | Imports rewritten |
|:---|:---|---:|:---|---:|
| A | markdown plans/summaries | 12 | `signedkan_wip/docs/archive/` | 0 |
| B | run_*.py experiment scripts | 101 | `signedkan_wip/experiments/runs/` | ~55 |
| C | bench/eval/aggregate/annotate | 5 | `experiments/bench/`, `experiments/eval/` | ~8 |
| D | top-level demo scripts | 3 | `signedkan_wip/demos/` | ~5 |
| **E** | **datasets*.py → datasets/ package** | 5 | `src/datasets/` (legacy + continuous + meshes + small + synth) | ~40 |
| **G-1** | **kinematic + mujoco** | 4 | `src/kinematic/` (fixtures + graph + mujoco_bridge + render) | ~12 |
| **F** | **23 core HSiKAN modules → core/ package** | 23 | `src/core/` | **~190** |
| **H pilot** | flesh out ExperimentBase + migrate run_early_stop.py | 1 script + base | (in place) | n/a |
| **Total** | | **153** | | **~310** |

## 2. The numbers

```
                                  start of day    after A-D    after E+G-1+F
signedkan_wip/src/ top *.py            152            44              12       ← 95 % gone
signedkan_wip/ root *.md                14             2               2
signedkan_wip/ subpackages               17           18              20       (+ datasets, kinematic, core)
```

Each new package has a **curated `__init__.py` with explicit
re-exports + `__all__`** (no `from .X import *` flat dumps). The
`__init__` is documentation: a reader can see at a glance what the
package's public surface is, and a re-export is a deliberate
architectural decision rather than an accident of the `*` operator.

## 3. Object-oriented commitment — what we shipped

### 3.1 `_experiment_base.py` — fleshed out from shell to working

The morning's shell-only ABC now has a working observer dispatch
mechanism + concrete `SimpleExperiment` adapter:

```python
class SimpleExperiment:
    """Thin adapter: subclass overrides run_seed; base orchestrates
    the seed loop and observer dispatch."""

    def add_observer(self, obs) -> "SimpleExperiment": ...  # chainable
    def run_seed(self, seed: int, **cfg) -> dict[str, Any]:  # subclass
        raise NotImplementedError
    def run(self, seeds, **cfg) -> list[dict[str, Any]]:
        # 1. emit on_run_start
        # 2. for each seed:
        #      emit on_seed_start
        #      result = self.run_seed(seed, **cfg)
        #      emit on_seed_end with float-castable metrics
        # 3. aggregate mean+std per numeric key
        # 4. emit on_run_end with summary
```

**Three concrete observers** ready to use:

- `StdoutObserver` — per-seed start/end printing.
- `JsonlObserver` — append one JSONL line per seed; correctly
  truncates at `on_run_start` when `mode="w"` (this was a
  latent bug in the morning's shell that the test
  `test_jsonl_observer_writes_one_line_per_seed` caught).
- `CallbackObserver` — user callbacks without a bespoke subclass.

**The full `ExperimentBase` ABC** (with `build_dataset` /
`build_model` / `build_optimizer` / `train_step` / `eval_step`)
remains as the long-term target. New code should subclass that;
existing code can adopt the thinner `SimpleExperiment` first to
get the observer pattern without rewriting the training loop.

### 3.2 Pilot migration: `run_early_stop.py`

The 60-LOC original was a triple loop (dataset × model × seed) with
inline argparse, per-iteration printing, and a JSON write at the
end. The migration:

```python
class EarlyStopExperiment(SimpleExperiment):
    def __init__(self, datasets, hidden, lr, n_epochs, val_every):
        super().__init__(label="early_stop")
        # ... store config ...

    def run_seed(self, seed: int, **cfg) -> dict:
        return run_one(cfg["model"], cfg["dataset"], self.hidden,
                       seed, self.n_epochs, lr=self.lr,
                       early_stopping=True,
                       val_every=self.val_every)

class _PrintObserver(StdoutObserver):
    """Format the per-seed end line like the original script did."""
    def on_seed_end(self, ev: SeedEvent) -> None: ...

# main():
exp = EarlyStopExperiment(...)
exp.add_observer(_PrintObserver(args.hidden))
exp.add_observer(JsonlObserver(str(out_path.with_suffix(".jsonl"))))
exp.run_grid(args.seeds)
```

The script is now ~95 LOC (slightly longer because the observer
pattern is explicit, but the new structure shows where the
boundaries are: config in the class, output in observers,
orchestration in `SimpleExperiment.run`). When the same pattern is
applied to the other 100 `run_*.py` scripts, the shared boilerplate
(argparse + JSON write + printing) becomes one line of registration
per script.

### 3.3 New test file — `test_experiment_base.py`

**10 tests pinning the observer protocol**:

```
test_simple_experiment_runs_one_seed             ✓
test_simple_experiment_runs_five_seeds           ✓
test_simple_experiment_run_seed_must_be_overridden ✓
test_callback_observer_fires_per_seed            ✓
test_jsonl_observer_writes_one_line_per_seed     ✓ (caught a real bug)
test_run_summary_includes_mean_and_std           ✓
test_multiple_observers_all_fire                 ✓
test_run_event_carries_label_and_seeds           ✓
test_add_observer_is_chainable                   ✓
test_event_dataclasses_are_frozen                ✓
```

## 4. Object-oriented patterns at the **package** level

Each new package commits a small set of patterns:

### `src/datasets/`

| Submodule | Type | Public exports |
|:---|:---|:---|
| `legacy.py` | dataclass + free functions | `SignedGraph`, `load`, `split`, `download` |
| `continuous.py` | dataclass + loader | `WeightedSignedGraph`, `load_continuous` |
| `meshes.py` | builder functions | `build_polyhedron`, etc |
| `small.py` | generator functions | `karate_faction_signed`, `sbm_signed`, ... |
| `synth.py` | generator functions | `make_moon_signed_graph`, ... |

The `SignedGraph` dataclass is the **canonical type** other modules
consume. A future cleanup would unify it with `WeightedSignedGraph`
under a `SignedGraphBase` ABC; noted for a future slice.

### `src/kinematic/`

`KinematicJoint` (dataclass) + `MuJoCoBridge` (class) + free
helpers (`urdf_to_signed_graph`, `parse_urdf`). The two classes
are the OO anchor; everything else operates on them.

### `src/core/` — the big one

23 submodules grouped into 11 conceptual sections in the
`__init__.py`:

```
# ── Activations + spline primitives ───────────────────
# ── Hyperedge + cycle + walk construction ─────────────
# ── Spectral / Laplacian init ─────────────────────────
# ── Core SignedKAN model + layers ─────────────────────
# ── Attention / triad / scene ─────────────────────────
# ── CPML routing / capsule / tier ─────────────────────
# ── Learnable M_e + sigma masking ─────────────────────
# ── Regularisers ──────────────────────────────────────
# ── Pruning + distillation ────────────────────────────
# ── Training loop ─────────────────────────────────────
# ── Profiling / sensitivity ───────────────────────────
```

97 public symbols re-exported with explicit `__all__`. The
canonical OO anchors:

- `SignedKAN` / `MultiLayerSignedKAN` (+ their `Config` dataclasses) — the model classes.
- `HighwaySignedKAN` (+ Config) — the residual-stream variant.
- `CPML` (+ Config) — the CPML routing.
- Activation classes: `CatmullRomActivation`, `BSplineActivation`, batched variants.
- Regulariser classes: `EntropyRegulariser`, `ParticipationRegulariser`, `CrossBranchRegulariser`, `NTupleBalanceLoss`.

## 5. Import-rewrite story (cumulative)

| Pattern | Count |
|:---|---:|
| `signedkan_wip.src.<core_mod>` → `signedkan_wip.src.core.<core_mod>` | 138 |
| `signedkan_wip.src.datasets_X` → `signedkan_wip.src.datasets` (re-export) | ~10 |
| `signedkan_wip.src.kinematic_X` → `signedkan_wip.src.kinematic` (re-export) | ~10 |
| `signedkan_wip.src.run_X` → `signedkan_wip.experiments.runs.run_X` | ~80 |
| `from .<core_mod>` (top-level src/) → `from .core.<core_mod>` | 39 |
| `from ..<X>` from src/sub/ → `from ..core.<X>` or `..datasets.<X>` etc | ~10 |
| `from .. import <core_mod>` → `from ..core import <core_mod>` | 6 |
| Indented (function-body) relative imports | 36 |
| Test monkey-patch targets (`DATA_DIR`, `urllib`) updated | ~12 |
| `from src.X import Y` (bare-`src`) | 2 |
| **Total** | **~340** |

## 6. Final test status

| Snapshot | Failed | Passed | Note |
|:---|---:|---:|:---|
| Pre-reorg (morning audit) | 16 | 827 | env-drift baseline |
| Post-Slice-A | 16 | 827 | identical |
| Post-Slice-B | 16 | 827 | identical (after two iterations) |
| Post-Slice-C | 16 | 827 | identical |
| Post-Slice-D | 16 | 827 | identical |
| Post-Slice-E (datasets) | 16 | 827 | identical (after monkey-patch fixes) |
| Post-Slice-G-1 (kinematic) | 16 | 827 | identical |
| Post-Slice-F (core) | 16 | 827 | identical (after `from .. import X` fixes) |
| **Post-Slice-H-pilot** | **16** | **837** | identical fails + **10 new passing** |

**Zero net regression across all five slices.** All 16 remaining
failures are pre-existing env-drift (CUDA / Bitcoin data /
miniconda3 torch drift / Triton kernel install).

## 7. What's still on the table (deferred)

| Slice | Description | Why deferred |
|:---|:---|:---|
| G-2 | Misc top-level files (`hsikan_device_env`, `runtime_config`, `gomb_jsonl_summarize`, `synthetic_signed_graphs`, `tabular_signed_graph`, `test_batched_encode`, `topk_cycle_demo`, `hymeko_*`) | These are small, self-contained; less benefit per move. Could be done quickly but better to wait for clear semantic homes. |
| Slice H proper | Migrate the other 100 `run_*.py` to `SimpleExperiment` / `ExperimentBase` | 100 separate diffs — one phase per session per CLAUDE.md `feedback_one_phase_per_session`. |
| `src/demo/` reorganisation | The package has 9 modules + 5+ cross-imports | Higher risk than the slices we did tonight; needs its own plan. |
| `signedkan_native` rust crate | Separate concern | Out of scope for the Python-side reorg. |

## 8. Object-oriented + observer design — closing the loop on the user's "if needed" guidance

The user's framing was *"modular, object-oriented approach; if
needed observer or dataflow"*. What this session committed:

- **Modular**: 19 subpackages, each with curated `__init__.py` and
  explicit re-exports. The 12 files left at `src/` top are the ones
  whose semantic home isn't yet obvious.
- **Object-oriented**: every new package's canonical types
  (`SignedGraph`, `KinematicJoint`, `MuJoCoBridge`, `SignedKAN`,
  `CPML`, etc.) are surfaced explicitly. Config dataclasses
  (`SignedKANConfig`, `MultiLayerSignedKANConfig`,
  `HighwaySignedKANConfig`, `CPMLConfig`, ...) accompany the model
  classes — the builder/config pattern of CLAUDE.md §7.
- **Observer**: shipped + tested. `ExperimentObserver` ABC,
  `EpochEvent` / `SeedEvent` / `RunEvent` frozen dataclasses,
  `StdoutObserver` / `JsonlObserver` / `CallbackObserver` concrete
  implementations, `SimpleExperiment.run()` as the orchestrator.
  10 passing unit tests.

The "dataflow approach" the user mentioned as a possible alternative
to the observer pattern is *also* now realisable on top of this base
— each `EpochEvent` / `SeedEvent` could be sent into a stream
(asyncio queue, Kafka topic, etc.) by a custom observer; the base
doesn't care because it just calls `obs.on_*(ev)`. No further work
needed to support it.

## 9. Anti-pattern audit (CLAUDE.md §6.5)

| Pattern | Status |
|:---|:---|
| §6.5 #1 Cartesian-product API | Not introduced. Slice F's 23-module reshuffle uses curated re-export, not 23 per-module dispatch fns. |
| §6.5 #2 Algorithm code behind Python boundary | Not relevant. |
| §6.5 #3 Per-experiment scaffold duplication | **Pilot fix shipped** (SimpleExperiment + run_early_stop migration). Full migration of 100 scripts queued. |
| §6.5 #4 Long single-file modules | Not introduced. `_experiment_base.py` is now ~310 LOC (still under the 400-LOC warning threshold). |
| §6.5 #5 New-name-for-new-axis | Not introduced. |
| §6.5 #7 String-typed config that should be enum | Not introduced. |
| §6.5 #11 Globals | Not introduced. Observer list is instance state. |

## 10. Bottom line

Three high-risk slices (E datasets, G-1 kinematic, F core) shipped
with full curation of each new package's `__init__.py`. The
`ExperimentBase` / `SimpleExperiment` / observer pattern shipped
working, with one representative `run_*.py` migrated, with 10
passing tests. **Net file moves today: 153; net import sites
rewritten: ~340; net test regression: zero**.

`signedkan_wip/src/` is now an actual semantic structure:

```
signedkan_wip/src/
├── __init__.py
├── 12 small top-level files (kept; future slice G-2)
├── adapters/, baselines/, benchmarks/, chicken/
├── core/              ← NEW: 23 modules, 97 public symbols
├── cycle_cache/, demo/, hymeko_gomb/, hypergraph/
├── datasets/          ← NEW: 5 submodules + curated API
├── kinematic/         ← NEW: 4 submodules + curated API
├── mixed_arity_signedkan/, paperkit/
├── rapport/, rapport_ros2/, sequence/, test_harness/
├── triton_kernels/, vision/
```

Slice H proper (100-script migration) is the natural next session;
the pattern is established and tested.

---

**Companion artefacts:**
- Plan (4-format): [`docs/plans/2026-05-19-signedkan-wip-organize/`](../docs/plans/2026-05-19-signedkan-wip-organize/)
- Phase-1 report (Slices A-D, morning): [`reports/2026-05-19-signedkan-wip-organize.md`](2026-05-19-signedkan-wip-organize.md)
- ExperimentBase: [`signedkan_wip/experiments/runs/_experiment_base.py`](../signedkan_wip/experiments/runs/_experiment_base.py)
- Tests: [`signedkan_wip/tests/test_experiment_base.py`](../signedkan_wip/tests/test_experiment_base.py)
- Pilot migration: [`signedkan_wip/experiments/runs/run_early_stop.py`](../signedkan_wip/experiments/runs/run_early_stop.py)
- New packages: [`signedkan_wip/src/datasets/`](../signedkan_wip/src/datasets/), [`signedkan_wip/src/kinematic/`](../signedkan_wip/src/kinematic/), [`signedkan_wip/src/core/`](../signedkan_wip/src/core/)
