# HSiKAN lean line — artifact index

One place to find **enumerator-first HSiKAN** work: k-pools, ABB, SSG-as-vertex-gate, harnesses, and measurement outputs.

> **Cursor / agents:** there is no separate “memory” API from the agent; treat this file as the canonical bookmark. Update the “Last checked” line when you add runs or reports.

**Last checked:** 2026-05-12 (`HSIKAN_DEVICE` / `--device` for reliable sweeps; `uv` notes).

---

## Strategy and harness (implementation)

| What | Path |
|------|------|
| Strategy (k-enum, ABB, SSG mapping, `--python` under systemd) | `signedkan_wip/docs/HSiKAN_lean_enumeration.md` |
| Subprocess grid driver (`PROFILE_ENV`, JSONL, `--python`, `--device`) | `signedkan_wip/src/run_hsikan_lean_profile.py` |
| Bitcoin Alpha/OTC shell (`HSIKAN_LEAN_DEVICE`, **UV-first**; override via `PYTHON`) | `signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh` |
| Wait for `systemd --user` units, then lean sweep or custom cmd | `signedkan_wip/experiments/schedule_hsikan_lean_after_units.sh` |
| **Pro scheduler** (UV-enforced, lockfile, CUDA guard, wait timeout) | `signedkan_wip/experiments/schedule_hsikan_lean_pro.sh` |
| After-chain line-count witness | `signedkan_wip/experiments/hsikan_chain_witness.sh` |
| Unit tests (env scrubber + device injection) | `signedkan_wip/tests/test_hsikan_lean_profile.py` , `signedkan_wip/tests/test_run_final_cell_device.py` |

---

## Reports (measurement + acceptance)

| What | Path |
|------|------|
| Harness + overnight queue notes, systemd unit name | `reports/2026-05-12-hsikan-lean-enumeration-harness.md` |
| ABB smoke, TopKBuilder, entropy plan pointer (context for ABB vs HSiKAN) | `reports/2026-05-10-abb-hsikan-smoke-and-builder.md` |
| **Overnight grid — JSONL** (one JSON object per cell; **90** rows when complete) | `reports/hsikan_lean_bitcoin_overnight_20260512_004029.jsonl` |
| **Overnight — stream log** (`[lean] dataset …`) | `reports/hsikan_lean_bitcoin_overnight_20260512_004029.log` |
| **Overnight — provenance** (git SHA, paths, `systemd-run` unit, MainPID) | `reports/hsikan_lean_bitcoin_overnight_20260512_004029.meta.txt` |
| Failed dry run (no `torch` on `/usr/bin/python3`); use `--python` | `reports/hsikan_lean_bitcoin_overnight_20260512_003928.jsonl.aborted_no_torch` (+ matching `.log.aborted_no_torch`, `003928.meta.txt`) |

**Progress:** `wc -l reports/hsikan_lean_bitcoin_overnight_20260512_004029.jsonl` → expect **90** when finished.  
**Status:** `systemctl --user status run-rddecb84693354dcc8fed6ba5efd0fc40.service` (unit name is also in the `.meta.txt`).

### Other result files (repo search)

| Artifact | What it is |
|----------|------------|
| `reports/gomb_tune_20260512_004315.jsonl` | **Gömb** tuning on `bitcoin_alpha` — not HSiKAN (`model`: `gomb`, `mixed_arity_gomb`, …). |
| `reports/2026-05-10-abb-hsikan-smoke-and-builder.md` | ABB vs HSiKAN routing narrative + builder work. |

No additional **`HSiKAN-mixed`** JSONL besides `hsikan_lean_bitcoin_overnight_20260512_004029.jsonl` showed up under `reports/`. That file’s best **Alpha** AUC in committed rows is about **0.962** (`lean_profile=pv_k128`, `lean_hidden=12`); several **ABB** rows failed with **CUDA OOM** when the GPU was shared — rerun with **`HSIKAN_LEAN_DEVICE=cpu`** (or `--device cpu`) to finish the grid, then tune **epochs / lr** on top of the best profile.

### Scheduled follow-up (after current `systemd-run`)

`schedule_hsikan_lean_after_units.sh` polls **`systemctl --user is-active`** until each listed unit is no longer active **in order**, then either runs **`run_hsikan_lean_bitcoin.sh`** (default) or, if **`HSIKAN_LEAN_AFTER_CMD`** is set, **`exec bash -lc "$HSIKAN_LEAN_AFTER_CMD"`** (skips the sweep — use for “wait for all results, then witness / merge”).

**Example queued on this machine (2026-05-12):** meta `reports/hsikan_lean_followup_20260512_005949.meta.txt` — waits for **`run-rddecb84693354dcc8fed6ba5efd0fc40.service`**, scheduler unit **`run-r510f90ddd54749488a43368229eb050f.service`**, outputs **`reports/hsikan_lean_bitcoin_followup_20260512_005949.jsonl`** and log **`reports/hsikan_lean_followup_20260512_005949.log`**.

**Chain witness (waits for overnight + follow-up scheduler, then `wc -l` both JSONLs):** meta `reports/hsikan_chain_wait_both_20260512_010400.meta.txt`, transient **`run-r9a245205509d4c78bd8675fefb095020.service`**, log `reports/hsikan_chain_wait_both_20260512_010400.log`, witness append target inside that meta.

```bash
# Wait for two units, then full grid (CPU)
export HSIKAN_LEAN_DEVICE=cpu HSIKAN_LEAN_OUT=reports/my_followup.jsonl
./signedkan_wip/experiments/schedule_hsikan_lean_after_units.sh run-<a>.service run-<b>.service

# Wait for two units, then only a witness script (no third sweep)
export HSIKAN_LEAN_AFTER_CMD="bash signedkan_wip/experiments/hsikan_chain_witness.sh reports/w.txt reports/a.jsonl reports/b.jsonl"
./signedkan_wip/experiments/schedule_hsikan_lean_after_units.sh run-<a>.service run-<b>.service
```

---

## Book / architecture (reader-facing)

| What | Path |
|------|------|
| HSiKAN architecture | `docs/book/src/quickstart/08-hsikan-architecture.md` |
| HSiKAN training / env | `docs/book/src/quickstart/09-hsikan-training.md` |
| Research chapter | `docs/book/src/research/hsikan.md` |
| HyMeKo ↔ HSiKAN note | `docs/hsikan_hymeko.md` |

---

## Older planning notes (`docs/` root)

These are **planning / roadmap** fragments, not the lean harness itself:

| Topic | Path |
|-------|------|
| Tabular benchmarks | `docs/plans_hsikan_tabular_benchmarks_2026_05_09.md` |
| Time series | `docs/plans_hsikan_time_series_2026_05_09.md` |
| Walker validation | `docs/plans_walk_hsikan_validation_2026_05_07.md` |
| RL / AL | `docs/plans_rl_al_hsikan_2026_05_06.md` |

---

## LaTeX / math (separate from Python harness)

| What | Path |
|------|------|
| HSiKAN brief (TeX source) | `reports/hsikan_hymeko_brief.tex` |
| Math foundation (TeX + aux/log) | `reports/hsikan_math_foundation.tex` (+ `.aux`, `.log`) |

---

## Toolchain: uv (recommended)

The repo root is a **uv workspace** (`README.md` — Python (uv)). One-time:

```bash
uv sync --group ml --all-packages
```

Then dev tools and tests need no manual `activate`:

```bash
export PYTHONPATH=.
uv run ruff check signedkan_wip/src/run_hsikan_lean_profile.py signedkan_wip/tests/test_hsikan_lean_profile.py
uv run pytest -p no:randomly signedkan_wip/tests/test_hsikan_lean_profile.py signedkan_wip/tests/test_run_final_cell_device.py -q
```

If `uv run` fails with an invalid `.venv`, delete `.venv` and re-run `uv sync --group ml --all-packages`.

---

## Quick commands (from repo root)

**Preferred (uv + workspace torch):**

```bash
export PYTHONPATH=.
PY="$(uv run python -c "import sys, torch; print(sys.executable)" 2>/dev/null || true)"
if [[ -z "${PY}" ]]; then PY="${HOME}/miniconda3/bin/python3"; fi
"${PY}" -m signedkan_wip.src.run_hsikan_lean_profile --python "${PY}" --help
./signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh   # auto-picks uv .venv if torch works
```

**Or** invoke the harness entirely through uv (same interpreter for parent + `--python` default):

```bash
export PYTHONPATH=.
PY="$(uv run python -c "import sys, torch; print(sys.executable)")"
uv run python -m signedkan_wip.src.run_hsikan_lean_profile \
  --python "${PY}" \
  --datasets bitcoin_alpha --seeds 0 --hidden 8 --profiles clean_baseline \
  --n-epochs 1 --timeout-s 3600 --out /tmp/hsikan_smoke.jsonl
```

After a JSONL run completes, aggregate however you like; keys commonly include `auc`, `n_params`, `lean_profile`, `lean_hidden`, `lean_seed`, `lean_dataset`, `returncode`, `lean_python`.
