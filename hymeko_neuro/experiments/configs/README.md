# Experiment Configs

Each YAML in this directory is **one experiment** that the runner
framework can execute via:

```
python -m hymeko_neuro.experiments.run --config <name>.yaml
```

The framework (under `hymeko_neuro/experiments/lib/`) replaces
~221 historical launcher scripts under `hymeko_neuro/experiments/run_*.{sh,py}`.

New experiments = new YAML, **never** a new launcher script. See
CLAUDE.md §6.5 #3 (per-experiment scaffold duplication) + #13 (file
proliferation).

## Index (active configs)

| YAML | scope | replaces | reproduces |
|---|---|---|---|
| `bitcoin_alpha_edge_cr_5seed.yaml` | BA 5-seed paired | partial `run_btc_alpha_otc_sota_gate.sh` | local BA 0.9946 @ per_vertex K=128 + Triton |
| `slashdot_edge_cr_5seed.yaml` | Slashdot 5-seed paired | `run_slashdot_edge_cr_5seed_2026_05_09.sh` | local Slashdot 0.9519 @ per_vertex K=128 + Triton (in flight smoke) |
| `epinions_edge_cr_5seed.yaml` | Epinions 5-seed paired | `run_epinions_edge_cr_5seed_2026_05_09.sh` | Komondor 0.8829 ± 0.0128 (config conversion in flight) |

## Smoke fixtures (prefixed with `_`)

| YAML | purpose |
|---|---|
| `_smoke_ba_real_seed0.yaml` | 1-cell parity check vs prior BA AUC 0.9946 |

## Adding a new experiment

1. **Discovery first** (CLAUDE.md §6.5 #12): grep this dir + the old
   `hymeko_neuro/experiments/run_*.{sh,py}` for the dataset/concept;
   if a YAML or script already covers it, edit, do not add.
2. Copy an existing YAML as template.
3. Edit `cells`, `model`, `topk`, `kernel`, `env`.
4. Set a unique `name`; the JSONL + summary paths default to
   `experiments/results/${name}.jsonl` and `reports/auto/${name}.md`.
5. (Optional) Run with `--explain` first to verify the parse.

## Schema (canonical)

See `docs/architecture/experiments_runner/runner_architecture.md`
for the full schema; the `ExperimentConfig` dataclass in
`hymeko_neuro/experiments/lib/config.py` is the source of truth.

## Migration backlog (out of scope this commit)

~78 shell launchers + ~108 Python launchers under
`hymeko_neuro/experiments/` are NOT yet migrated. They continue to
work but should be replaced by YAML configs as touched. Top
priority candidates (paper-relevant):

- `run_slashdot_*` (~10 variants -> one YAML with `topk` / `kernel` knobs)
- `run_gomb_*` (~10 variants -> Gömb-experiment subclass + YAML)
- `run_phase*` (~35 variants -> mostly historical; archive
  candidates rather than active migration)
