# Report: HSiKAN lean enumeration harness (k-pool + ABB + SSG gate)

## Summary

Added a **reproducible subprocess harness** that scrubs inherited `HSIKAN_*`
variables, applies **named profiles** (per-vertex top-K, ABB, degree-based
vertex filter as SSG mapping), runs `run_final_cell` for HSiKAN, and appends
**JSONL** rows with `lean_*` metadata (including `lean_python`).  A **`--python`**
flag fixes `systemd-run`/cron runs where `PATH` points at a system interpreter
without torch.  A **90-cell Bitcoin Alpha/OTC overnight sweep** was queued under
`systemd-run --user` with **MemoryMax=16G** (see “Overnight run” below).

## Files touched (this task)

| Path | Notes |
|------|--------|
| `signedkan_wip/src/run_hsikan_lean_profile.py` | Harness + `--python` + `lean_python` in JSONL. |
| `signedkan_wip/docs/HSiKAN_lean_enumeration.md` | Strategy + usage. |
| `signedkan_wip/experiments/run_hsikan_lean_bitcoin.sh` | Executable wrapper. |
| `signedkan_wip/tests/test_hsikan_lean_profile.py` | 3 tests. |

## CORE.YAML items touched

None (no edits under core lockdown paths).

## New / removed dependencies

None.

## Test results

- Command: `PYTHONPATH=. pytest -p no:randomly signedkan_wip/tests/test_hsikan_lean_profile.py -q`
- Result: **3 passed** in ~0.02 s (host: session dev machine).

**Note:** Full `signedkan_wip/tests` collection from repo root without
`PYTHONPATH=.` fails with `ModuleNotFoundError: signedkan_wip` in this
environment; the documented invocation uses `PYTHONPATH=.` (see
`docs/topk_cycles.md`).

## Performance / smoke

- **Smoke (not a benchmark):** one subprocess,
  `bitcoin_alpha`, `clean_baseline`, `hidden=8`, `n_epochs=1`, `max_k4=50000`,
  seed `0` — **wall ~6.3 s**, `returncode=0`, `n_params=30484`,
  `auc≈0.381`.  Output: `/tmp/hsikan_lean_smoke.jsonl` (local).

Per CLAUDE.md Section 3, this is **diagnostic smoke only** (not 5× iterations
+ median/IQR); no performance budget regression claim.

## Static analysis

`ruff` was not available on the default `python3` in PATH (`No module named ruff`);
`uv run ruff` failed here because `.venv` has no Python executable.  Re-run
`ruff check` / `mypy` from a properly synced workspace when available.

## Protocol note (planning)

CLAUDE.md Section 2 calls for a `docs/plans/<date>-<slug>/` bundle before
non-trivial implementation.  This harness landed from an in-session continuation;
if strict audit compliance is required, add a dated plan bundle **before** the
next dependent change, or treat this delta as **expedited harness-only** with
explicit maintainer sign-off.

## Overnight run (queued 2026-05-12, user machine)

A first `systemd-run` attempt used bare `python3` on a minimal `PATH`
(`ModuleNotFoundError: torch`).  The harness now supports **`--python`**, and
the Bitcoin shell script prefers `$HOME/miniconda3/bin/python3` when `PYTHON`
is unset.

**Active run (90 cells, 16G RSS cap):**

| Artifact | Path |
|----------|------|
| Meta (unit name, PID, `git_sha`, paths) | `reports/hsikan_lean_bitcoin_overnight_20260512_004029.meta.txt` |
| JSONL (one object per cell, growing) | `reports/hsikan_lean_bitcoin_overnight_20260512_004029.jsonl` |
| Stream log (`[lean] …` lines) | `reports/hsikan_lean_bitcoin_overnight_20260512_004029.log` |
| systemd unit | `run-rddecb84693354dcc8fed6ba5efd0fc40.service` |

Check status: `systemctl --user status run-rddecb84693354dcc8fed6ba5efd0fc40.service`

**Aborted first attempt** (wrong interpreter): partial rows in
`reports/hsikan_lean_bitcoin_overnight_20260512_003928.jsonl.aborted_no_torch`
and matching `.log.aborted_no_torch`; notes in
`reports/hsikan_lean_bitcoin_overnight_20260512_003928.meta.txt`.

**Canonical bookmark for humans/agents:** `docs/HSIKAN_lean_artifacts.md` (paths to everything above plus book chapters).

## Open issues / follow-up

- When the overnight JSONL is complete (90 lines), compare **AUC vs `n_params`**
  across `hidden` and profiles; attach a short summary table to the next report.
- Optional: wire a one-line pointer from `DECISIONS.md` or training docs to
  `signedkan_wip/docs/HSiKAN_lean_enumeration.md`.

## Experiment provenance (smoke)

- **Git:** working tree not clean globally; files above are **untracked** new
  paths at report time.
- **Seed:** `0`
- **Command:**  
  `PYTHONPATH=. python3 -m signedkan_wip.src.run_hsikan_lean_profile ...`
