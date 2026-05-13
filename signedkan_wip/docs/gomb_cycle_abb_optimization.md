# Gömb cycle ABB: optimization without extra parameters

## What this is

`--cycle-abb-mode` on `run_gomb_smoke` forwards **ABB** (branch-and-bound) modes
into Rust `hymeko.enumerate_cycles_rs` for **train-edge** cycle pools (single-`k`,
mixed `cycle_ks`, and joint-mix **c3/c4** slots). Walk slots (w2/w3) are unchanged.

This is **not** the process-synthesis `hymeko_pgraph` MSG/SSG stack; it is the
**signed-graph** enumerator’s own ABB path (`start_local`, `global_min`).

## Why it is a neural-network optimization

- **`n_params` is invariant** under ABB: the same `GombConfig` / shells are built.
- **Runtime is not**: forward and backward work scales with **cycle row count**
  `n_cycles` (gathers, capsule / hypergraph routing, etc.).
- ABB **reshapes the tuple reservoir** under the same per-vertex budget: you can
  **trade wall time and memory** against **which** cycles the net sees.

## Reproduce a paired table

From the repo root (set `PYTHONPATH` as for other `signedkan_wip` CLIs):

```bash
python -m signedkan_wip.src.benchmarks.run_gomb_cycle_abb_compare \
  --dataset bitcoin_otc \
  --edge-split 80_10_10 \
  --device cpu \
  --seed 0 \
  --n-epochs 8 \
  --topk 48 \
  --d-embed 24 --d-outer 12 --M-outer 4 --d-middle 24 --d-core 24 \
  --modes none start_local
```

Optional JSONL log (one JSON object per mode):

```bash
python -m signedkan_wip.src.benchmarks.run_gomb_cycle_abb_compare \
  ...same flags... \
  --jsonl-out signedkan_wip/experiments/results/gomb_abb_compare.jsonl
```

## Interpreting results

On a **short** budget, **smaller `n_cycles`** can **lower AUROC** if the model
never receives a compensating training budget or a larger `topk`. Treat ABB as
a **Pareto knob**: same capacity class, different **effective batch of
structure** — retune epochs / `topk` / mode (`global_min` vs `start_local`) when
chasing both speed and headline AUC.

## See also

- `run_gomb_smoke.py` module docstring (`--cycle-abb-mode`, `--cycle-abb-fullness-gate`).
- `hymeko_gomb/joint_enumeration.py` (`build_joint_ba_pools` passes the same flags
  into **c3/c4** enumeration).
- **P-graph MSG / SSG / ABB** (outer discrete structures): run
  ``python -m signedkan_wip.src.run_gomb_msg_sweep --pgraph data/hsikan/sweep_msg_gomb.hymeko``
  or ``hymeko_driver --backend gomb`` (see ``signedkan_wip/src/hymeko_driver.py``).
  Rust analysis: ``cargo run -p hymeko_pgraph --bin hymeko_pgraph_dump -- <file> --algorithm ssg``.
- **Key-art image prompt** (neo-Tokyo + repo-backed HUD numbers):  
  `signedkan_wip/docs/gomb_key_art_prompt.md`.
