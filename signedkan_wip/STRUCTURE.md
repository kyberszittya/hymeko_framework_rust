# `signedkan_wip` source layout

The `signedkan_wip/src/` tree mixes **importable library modules** with **hundreds of experiment entrypoints** (`run_*.py`). The latter are historical — new work should prefer subpackages when adding large new surfaces.

## Stable library packages (prefer imports from here)

| Path | Role |
|------|------|
| `signedkan.py` | `SignedKANLayer`, `MultiLayerSignedKAN` |
| `mixed_arity_signedkan/` | `MixedAritySignedKAN` (multi-slot cycles/walks) |
| `cpml.py` | CPML tiered stack + factorial smoke |
| `hymeko_gomb/` | HymeKo-Gömb cascade, `MixedArityGomb`, **`JointMixGomb`** (joint_ba c3,c4,w2,w3) |
| `datasets.py`, `walks.py`, `n_tuples.py`, `cycle_cache/` | Graph I/O, tuples, disk cache |
| `baselines/` | SGCN, SiGAT, … |
| `runtime_config.py` | Typed env / training config |

## Experiment entrypoints (`run_*.py`)

~100 modules at `src/run_*.py` remain the **de facto** CLI surface (`python -m signedkan_wip.src.run_…`). Migrating them wholesale would churn every shell script and doc; do it in **batches** behind a `runs/` package when there is review bandwidth.

## Paper / profiling utilities

Moved under **`signedkan_wip/src/paperkit/`** (2026-05-12):

- `build_*.py`, `analyze_*.py`, `profile_*.py`

Invoke with `python -m signedkan_wip.src.paperkit.<module>`.

## Tests

`signedkan_wip/tests/` — run with `PYTHONPATH=<repo root>`.
