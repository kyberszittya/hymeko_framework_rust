# Recipe: Debug the pipeline

`signedkan_wip/src/test_harness/` ships a 3-layer integrity-test framework that catches the 5 bug classes that have cost overnight GPU time:

| # | bug class | example | how the framework catches it |
|---|---|---|---|
| 1 | **Partial wiring** | `HSIKAN_KB_INIT_TCB` only patched inner KB activation; outer kept default init | `env_var_audit.py` — positive-control assertion: every env var must change all affected model state |
| 2 | **Tail-filter swallowing JSON** | `python ... \| tail -1` ate JSON when warnings came after | `result_capture.py` — captures full stdout+stderr to a file, doesn't depend on grep heuristics |
| 3 | **Exit-0 on crash** | `timeout`-killed bash pipeline reports exit 0 because `tail` exits 0 | `result_capture.CellResult.ok` requires `returncode==0 AND json_result is not None` |
| 4 | **Numerical regression** | NaN AUC, infinite F1, label leakage | `invariant_check.smoke_test` — 1-epoch run asserts AUC ∈ [0,1], finite, n_params > 0 |
| 5 | **OOM at training time** | 1-epoch smoke runs fine, 30-epoch sweep OOMs | `invariant_check.smoke_test(max_gpu_gb=…)` measures `torch.cuda.max_memory_allocated()` |

## Run the integrity tests

```bash
pytest signedkan_wip/tests/test_pipeline_integrity.py -v -k env_var
```

Expected: 5 audit cases pass in ~1 second. If a wiring bug returns, the offending env var is named in the failure.

## Smoke an unfamiliar config

```bash
HSIKAN_TOPK_K=64 HSIKAN_TOPK_PRUNER=balance \
    python -m signedkan_wip.src.test_harness.invariant_check \
    --dataset bitcoin_alpha --epochs 1
```

Catches NaN, infinite outputs, and (with `--max-gpu-gb 6.0`) likely-OOM configs.

## Use `run_cells()` instead of bash `tail`

For multi-cell sweeps, replace bash boilerplate with the Python `run_cells` API:

```python
from signedkan_wip.src.test_harness import run_cells

specs = [
    ("ba m=64 balance",
     ["python", "-m", "signedkan_wip.src.run_final_cell",
      "--dataset", "bitcoin_alpha", "--model", "HSiKAN",
      "--hidden", "16", "--n-epochs", "30", "--seed", "0"],
     {"HSIKAN_TOPK_MODE": "per_vertex", "HSIKAN_TOPK_K": "64",
      "HSIKAN_TOPK_PRUNER": "balance"}),
    # ... more cells
]
results = run_cells(specs, output_jsonl="/tmp/sweep.jsonl")
# Each result has explicit .ok, .json_result, .error_class.
# No more "[FAILED]" lines from grep heuristics.
```

## Adding a new audit case

Whenever you wire a new env var:

```python
# signedkan_wip/src/test_harness/env_var_audit.py — extend cases()
out.append(AuditCase(
    name="HSIKAN_YOUR_FLAG → cfg.your_new_flag",
    env={"HSIKAN_YOUR_FLAG": "1"},
    setup=_new_setup,
    assertion=lambda model: (
        getattr(model.cfg, "your_new_flag", False),
        "cfg.your_new_flag was not set",
    ),
))
```

## See also

- `docs/test_framework.md` — original framework writeup
- `signedkan_wip/src/test_harness/` — three modules: `env_var_audit`, `result_capture`, `invariant_check`
