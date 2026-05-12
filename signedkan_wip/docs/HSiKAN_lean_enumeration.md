# HSiKAN lean line: spend budget on structure, not width

**Index of all related reports, book chapters, and overnight artifacts:**  
[`docs/HSIKAN_lean_artifacts.md`](../../docs/HSIKAN_lean_artifacts.md) (repo root).

HSiKAN is **in-family** work alongside Gömb. The comparative story we want is:

> **Smaller `hidden_dim` (fewer learnable parameters)** while keeping edge
> prediction quality competitive with wider signed GNNs, by **improving what
> enters the sparse incidence head** — not by inflating embedding width.

## Three levers (as used in this repo)

### 1. k-enumeration (top-K / per-vertex pools)

Controlled by `HSIKAN_TOPK_MODE`, `HSIKAN_TOPK_K`, tiering / adaptive flags in
`runtime_config.TopKConfig` and the Rust-backed enumerator paths in
`cycle_cache`.  Fewer, higher-signal tuple columns reduce **noise** in the
incidence matrix and stabilise training at small `hidden`.

### 2. ABB (adaptive batch / blocking on the enumerator side)

`HSIKAN_USE_PER_VERTEX_ABB`, `HSIKAN_USE_PER_VERTEX_ABB_MODE`,
`HSIKAN_PER_VERTEX_ABB_FULLNESS_GATE` — documented in `runtime_config.py` and
exercised in `signedkan_wip/experiments/run_overnight_abb_validation_2026_05_11.sh`.
ABB trims redundant cycle mass so the same `hidden` sees a **cleaner** tuple
support set.

### 3. SSG — selective structural gating (vertex shell)

In HyMeKo language the driver mentions **MSG / SSG / ABB** axiom sweeps
(`hymeko_driver.py`).  For a **concrete, code-backed** mapping in Python land,
use **vertex pre-filtering**:

- `HSIKAN_VERTEX_FILTER` / `HSIKAN_VERTEX_FILTER_MIN_DEGREE`

This gates which vertices participate in the enumerator’s structural budget
(a degree-selected subgraph shell) **before** tuple materialisation.

## Reproducible harness

`python -m signedkan_wip.src.run_hsikan_lean_profile` runs `run_final_cell` in
**subprocesses** with a **scrubbed** `HSIKAN_*` parent environment, then applies
named **profiles** (`PROFILE_ENV` in that module).  Output is **JSONL** with
`auc`, `n_params`, `f1m`, `fwd_per_call_ms`, plus `lean_*` provenance fields
(including `lean_python`: which interpreter ran each `run_final_cell` child).

### uv workspace (recommended for dev / CI)

From the repo root (see `README.md` — Python (uv)):

```bash
uv sync --group ml --all-packages   # PyTorch + workspace members + dev tools
export PYTHONPATH=.
uv run python -m signedkan_wip.src.run_hsikan_lean_profile --help
PY="$(uv run python -c "import sys, torch; print(sys.executable)")"
PYTHONPATH=. uv run python -m signedkan_wip.src.run_hsikan_lean_profile \
  --python "${PY}" \
  --datasets bitcoin_alpha --seeds 0 --hidden 8 --profiles clean_baseline \
  --n-epochs 1 --timeout-s 3600 --out /tmp/hsikan_smoke.jsonl
```

Lint / tests: `uv run ruff check signedkan_wip/`, `uv run pytest -p no:randomly signedkan_wip/tests/test_hsikan_lean_profile.py`.

If `uv run` complains that `.venv` is invalid, remove the broken `.venv` or run `uv sync --group ml --all-packages` again.

`run_hsikan_lean_profile` supports **`--device cpu|cuda|auto`**, which sets **`HSIKAN_DEVICE`** for each `run_final_cell` child — use **`cpu`** for full overnight grids when the GPU is not exclusive. The Bitcoin shell forwards **`HSIKAN_LEAN_DEVICE`**.

### `systemd-run` / cron: set `--python`

Under a transient user unit, `PATH` is often minimal, so `python3` may resolve
to `/usr/bin/python3` **without** torch while your conda env does. Pass an
explicit torch-capable interpreter:

```bash
PY="${HOME}/miniconda3/bin/python3"
PYTHONPATH=. "${PY}" -m signedkan_wip.src.run_hsikan_lean_profile \
  --python "${PY}" \
  ...
```

For overnight launchers, resolve once inside the repo:  
`PY="$(cd /path/to/hymeko_framework_rust && uv run python -c "import sys, torch; print(sys.executable)")"`.

The shell helper `signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh` sets
`PYTHON` in order: use explicit `$PYTHON` if set; else **`uv run`** when `uv`
is on `PATH` and `import torch` succeeds in the workspace; else
`$HOME/miniconda3/bin/python3` (or a few other common paths); else `python3`.
It always passes `--python "${PYTHON}"` so child cells match the parent.

Quick Bitcoin / OTC grid (3 seeds × three widths × five profiles):

```bash
export PYTHONPATH=.
PY="$(uv run python -c "import sys, torch; print(sys.executable)" 2>/dev/null || true)"
if [[ -z "${PY}" ]]; then PY="${HOME}/miniconda3/bin/python3"; fi
"${PY}" -m signedkan_wip.src.run_hsikan_lean_profile \
  --python "${PY}" \
  --datasets bitcoin_alpha bitcoin_otc \
  --seeds 0 1 2 \
  --hidden 8 12 16 \
  --profiles clean_baseline pv_k128 pv_k128_abb_g10 pv_k64_abb_g10 pv_k64_abb_ssg_deg3 \
  --n-epochs 80 \
  --out reports/hsikan_lean_profile_bitcoin.jsonl
```

Shell wrapper: `signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh`.

## What this does *not* do yet

It does **not** change spline grid size, layer count, or classifier architecture
inside `MixedAritySignedKAN` — those are separate knobs for a follow-up “true
width compression” pass once the enumerator-first curve is measured.
