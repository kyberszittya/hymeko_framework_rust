# Experiments Runner Architecture

Date: 2026-06-04

## Why

The repo currently has **221 launcher scripts** (106 shell, 115 python)
under `hymeko_neuro/experiments/`, with massive overlap. Each new
experiment got a new file rather than a new config — exact violation
of CLAUDE.md §6.5 #3 ("per-experiment scaffold duplication") which
has been on the books since 2026-05-11 and only grew worse.

This document defines the **single Experiment framework** that
replaces them. New experiments become a **YAML config + zero code**.
Existing scripts get migrated incrementally (this commit ships
3 reference migrations).

## Scope of this commit

- Create `hymeko_neuro/experiments/lib/` package:
  `runner.py`, `config.py`, `monitors.py`, `registry.py`
- Create `hymeko_neuro/experiments/configs/` for YAML configs
- Add CLI entry `python -m hymeko_neuro.experiments.run`
- Migrate 3 representative experiments (Bitcoin Alpha edge_cr 5-seed,
  Slashdot edge_cr 5-seed, Epinions edge_cr 5-seed)
- Delete the 3 corresponding shell launchers
- Smoke-verify the migrated framework reproduces prior benchmark
  numbers (Bitcoin Alpha AUC ~0.9868 ± .0058)

## Architecture

```
                      Experiment (ABC)
                     /       |       \
        SingleCell      Sweep         KomondorArray
        Experiment    Experiment     Experiment
            |               |               |
            └──── ExperimentConfig (composes RuntimeConfigs)
                            |
                  +---------+----------+
                  |                    |
            DatasetConfig         ModelConfig
                  |                    |
                  +─── HSiKANConfig ───+
                  +─── CycleCacheConfig ─+
                  +─── KernelConfig    ─+
                  +─── TopKConfig      ─+
                       (existing dataclasses in runtime_config.py)

                  Monitor protocol (Observer)
                     |
            +────────+────────+──────────+
            |        |        |          |
       WallMonitor MemMonitor AUCMonitor JSONLEmitter

  YAML config ───load──> ExperimentConfig ───run──> Experiment
                                                       │
                                                       ▼
                                                  Monitors emit
                                                  events to JSONL
                                                  + stdout summary
```

See also:
- `runner_class_diagram.mmd` (Mermaid version)
- `runner_class_diagram.tikz` (LaTeX figure version, build via plan.pdf)

## Class contracts

### `Experiment` (ABC)
```python
class Experiment(ABC):
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult: ...

    def add_monitor(self, m: Monitor) -> None: ...
```

### `SingleCellExperiment` (concrete)
One (dataset, mode, seed) cell. Wraps `run_final_cell.main()` for
backward compatibility (the 1295-LOC training body is NOT
rewritten in this commit — only orchestrated). Returns
`ExperimentResult(auc, f1m, wall_s, peak_rss_mb, params, config_echo)`.

### `SweepExperiment` (concrete)
A grid over seeds (+ optionally modes, datasets). Calls
SingleCellExperiment.run() per cell, aggregates results, emits
paired-mean ± std + per-cell JSONL.

### `KomondorArrayExperiment` (concrete)
Generates SLURM array submission via the existing canonical
`docs/komondor_setup/submit_hsikan_edge_cr_array.sh`. Does NOT
re-implement SLURM; just generates the right invocation.

### `Monitor` (Protocol)
```python
class Monitor(Protocol):
    def on_cell_start(self, cell: CellSpec) -> None: ...
    def on_cell_end(self, cell: CellSpec, result: ExperimentResult) -> None: ...
    def on_sweep_end(self, summary: SweepSummary) -> None: ...
```

Concrete: `WallMonitor`, `MemoryMonitor` (peak RSS via
psutil), `AUCMonitor` (paired ± std), `JSONLEmitter` (per-cell
JSONL row).

## YAML config schema

```yaml
# hymeko_neuro/experiments/configs/<name>.yaml
name: bitcoin_alpha_edge_cr_5seed
description: 5-seed paired AUC on Bitcoin Alpha with edge_cr highway.
experiment_class: SweepExperiment       # registry lookup

cells:                                  # explicit cell list OR ranges
  - { dataset: bitcoin_alpha, mode: real,    seeds: [0, 1, 2, 3, 4] }
  - { dataset: bitcoin_alpha, mode: shuffle, seeds: [0, 1, 2, 3, 4] }

model:
  family: HSiKAN
  hidden: 4
  n_epochs: 80
  mixed_tuples: [c2, c3, c4, c5, w2, w3]
  attention_m_e: quaternion
  attention_highway: true
  attention_highway_kind: edge_cr

topk:
  mode: per_vertex
  k: 128
  pruner: balance
  scorer: fraction_negative

kernel:
  triton: true
  triton_backward: true

cycle_cache:
  enabled: true
  dir: .cache/hymeko_cycles

env:
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  MALLOC_ARENA_MAX: 4
  OMP_NUM_THREADS: 4

monitors:                               # which monitors to attach
  - WallMonitor
  - MemoryMonitor
  - AUCMonitor
  - JSONLEmitter

output:
  jsonl: experiments/results/${name}.jsonl
  summary: reports/auto/${name}.md
```

## CLI

```bash
# Run a single experiment by config name
python -m hymeko_neuro.experiments.run \
    --config hymeko_neuro/experiments/configs/bitcoin_alpha_edge_cr_5seed.yaml

# List registered experiments
python -m hymeko_neuro.experiments.run --list

# Compare to a prior measurement
python -m hymeko_neuro.experiments.run \
    --config X.yaml --compare-to results/Y.jsonl
```

## Migration policy (going forward)

CLAUDE.md §6.5 #3 already mandates this. **Reinforcement going forward:**

- **No new `hymeko_neuro/experiments/run_*.{sh,py}`** files unless they fall
  outside the SignedLinkPrediction / Komondor pattern.
- New experiment = new YAML config in `hymeko_neuro/experiments/configs/`.
- The existing 221 scripts get migrated **as they are touched**, not
  in a single batch.
- This commit migrates 3 (BA, Slashdot, Epinions edge_cr 5-seed) as
  reference + benchmark-parity validation.

## Benchmark parity protocol (this commit)

After migration, re-run 1 cell of each migrated experiment via the new
runner and verify:

| dataset | mode | prior AUC | new AUC | match? |
|---|---|---|---|---|
| bitcoin_alpha | real (seed 0) | 0.9868 ± .0058 (chain 5-seed mean) | TBD | TBD |
| (slashdot + epinions: cold-cache too expensive locally, validate via Komondor next session) |

Smoke-only is sufficient for the framework correctness check; full
5-seed reproducibility is a separate test.

## Out of scope (this commit)

- Rewriting the 1295-LOC `run_final_cell.py` training body
- Migrating the other ~108 Python launchers + ~80 shell launchers
- HymeYOLO / VOC / pose experiment families (different domain;
  separate Experiment subclass needed)
- Komondor smoke for Slashdot / Epinions migrated paths (next session)

## Risk

- Wrapping `run_final_cell.main()` rather than rewriting it preserves
  numerical behavior with high confidence; the framework adds only
  orchestration around a pinned core.
- YAML → CLI args translation could mis-pass an argument; verified
  via the benchmark parity smoke.
- Old shell scripts continue to work until explicitly deleted.
  We delete 3 reference migrations; the other 78 stay.
