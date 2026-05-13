# Optuna monitor — 2026-05-13 (snapshot)

## Summary (as of last poll)

Two **independent** serial Optuna drivers are active on the host; the repo **CUDA job flock** (`signedkan_wip/experiments/results/.cuda_job_serial.lock`) ensures **only one** `run_optuna_search` parent holds the GPU-critical section at a time.

1. **Follow-up four-graph queue** (`run_after_experiment` → `run_optuna_core_signed_graphs_serial.sh`)  
   - **Log:** `signedkan_wip/experiments/results/follow_optuna_20260513T003359Z.log`  
   - **Storage:** `signedkan_wip/experiments/results/optuna_serial_20260513T010159Z.db`  
   - **Progress:** Study `bitcoin_otc_20260513T010159Z` — trial **0** **PRUNED** (dot attention **CUDA OOM**, same pattern as earlier smokes); trial **1** **RUNNING** (`attn=none`, mixed `c2,c4,w2–w5`, `h=8`, `cap=100000`).  
   - **Remaining in this shell queue (after OTC finishes):** `bitcoin_alpha`, `slashdot`, `epinions` (30×80 each).

2. **Alpha → Slashdot-only queue** (user launch)  
   - **Log:** `signedkan_wip/experiments/results/optuna_alpha_slashdot_20260513T010509Z.log` (header only at snapshot).  
   - **Intended storage:** `signedkan_wip/experiments/results/optuna_serial_20260513T010510Z.db`  
   - **Status:** Driver shell running; **SQLite file not created yet** on disk while the process is **blocked behind (1)** on the same flock — expected until the four-graph driver’s current `run_optuna_search` invocation **releases** the lock (after its full `n_trials=30` OTC study completes, unless pruned early).

## Implication

The **Alpha/Slashdot** job will **not** advance (and will not create its DB) until the **four-graph** job’s **current** `python -m … run_optuna_search` for **OTC** finishes all **30** trials (or you stop one of the drivers). If that was unintentional, **stop** the lower-priority shell (`run_optuna_serial_datasets` for Alpha/Slashdot, or the follow-up core-graph driver) and re-queue it **after** the other completes.

## Quick commands

```bash
# Processes
pgrep -af 'run_optuna_serial|run_after_experiment|run_optuna_search'

# Logs
tail -f signedkan_wip/experiments/results/follow_optuna_20260513T003359Z.log
tail -f signedkan_wip/experiments/results/optuna_alpha_slashdot_20260513T010509Z.log

# Studies / trials (010159 DB — updates as four-graph progresses)
sqlite3 signedkan_wip/experiments/results/optuna_serial_20260513T010159Z.db \
  "SELECT study_name FROM studies; SELECT number,state FROM trials ORDER BY number;"
```

## Follow-up (engineering)

- Prefer **one** serial driver per machine, or chain with `run_after_experiment.sh` so the second starts only after the first **process** exits (already used for follow-up; the extra Alpha/Slashdot launch overlapped that intent).
- Dot-attention trials on **~8 GiB** GPUs with desktop Chrome still tend to **OOM**; consider excluding dot in search space for this hardware or tightening caps / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Completion

This report is a **point-in-time** monitor, not a wait-for-all-trials completion certificate. Re-run the SQLite / `tail` commands above after wall time (30 trials × 80 epochs × multiple datasets is long).
