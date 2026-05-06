# HSiKAN pipeline integrity test framework

Built 2026-05-06 morning, after a series of silent bugs cost us
hours of overnight GPU time. The framework catches **five bug
classes** by construction.

## The bug classes

| # | bug class | example today | how the framework catches it |
|---|---|---|---|
| 1 | **Partial wiring** | HSIKAN_KB_INIT_TCB only patched inner KB activation; outer silently kept zero init | `env_var_audit.py` — positive-control assertion: each env var must change *all* affected model state |
| 2 | **Tail-filter swallowing JSON** | `python ... 2>/dev/null \| tail -1` ate JSON when warnings came after | `result_capture.py` — captures full stdout+stderr to a file, parses last JSON line, doesn't depend on grep heuristics |
| 3 | **Exit-0 on crash** | timeout-killed bash pipeline reports exit 0 because `tail` exits 0 | `result_capture.py.CellResult.ok` requires returncode==0 AND `json_result is not None` |
| 4 | **Numerical regression** | NaN AUC, infinite F1, label leakage | `invariant_check.smoke_test` — 1-epoch run on bitcoin_alpha asserts AUC ∈ [0,1], finite, n_params > 0 |
| 5 | **OOM at training time** | 1-epoch smoke runs fine, 30-epoch sweep OOMs from accumulated activations | `invariant_check.smoke_test(max_gpu_gb=...)` measures `torch.cuda.max_memory_allocated()` after one forward+backward |

## Layout

```
signedkan_wip/src/test_harness/
├── __init__.py                   # public API
├── env_var_audit.py              # 4 env-var positive-controls, ~190 lines
├── result_capture.py             # CellResult + run_cell + run_cells, ~120 lines
└── invariant_check.py            # 1-epoch smoke + memory probe, ~75 lines

signedkan_wip/tests/
└── test_pipeline_integrity.py    # pytest entry, parametrised over audit cases
```

## Usage

### Before any sweep — env-var audit (1.4 s)

```bash
pytest signedkan_wip/tests/test_pipeline_integrity.py -v -k env_var
```

Expected output:

```
test_env_var_takes_effect[HSIKAN_KB_INIT_TCB → both inner+outer KB tcb] PASSED
test_env_var_takes_effect[HSIKAN_ATTENTION_M_E=dot → _AttentionM_e instance] PASSED
test_env_var_takes_effect[HSIKAN_ATTENTION_M_E=quaternion → _QuaternionAttentionM_e] PASSED
test_env_var_takes_effect[direct_messaging=True → cfg.direct_messaging set] PASSED
============== 4 passed in 1.35s ==============
```

If a wiring bug returns, this fails immediately with the offending env var named.

### Before any new sweep config — invariant smoke (30 s)

```bash
HSIKAN_TOPK_K=64 HSIKAN_TOPK_PRUNER=balance \
  python -m signedkan_wip.src.test_harness.invariant_check \
  --dataset bitcoin_alpha --epochs 1
```

Catches NaN, finite, and (with `--max-gpu-gb 6.0`) OOM-likely configs.

### Inside a sweep script — robust capture

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

When you wire a new env var:

```python
# In env_var_audit.cases() add:
def _new_setup():
    return _build_mixed_arity_model(your_new_flag=True)

def _new_assertion(model) -> tuple[bool, str]:
    if not getattr(model.cfg, "your_new_flag", False):
        return False, "cfg.your_new_flag != True"
    return True, "cfg.your_new_flag=True ✓"

out.append(AuditCase(
    name="HSIKAN_YOUR_FLAG → cfg.your_new_flag",
    env={"HSIKAN_YOUR_FLAG": "1"},
    setup=_new_setup, assertion=_new_assertion,
))
```

Pytest picks it up automatically (parametrised over `cases()`).

## What the framework is NOT

- **Not a unit-test suite** for the HSiKAN math (Catmull-Rom evaluation,
  Hamilton product, etc.). Those have their own pytest tests.
- **Not a replacement for full validation runs**. Smoke tests run
  1 epoch; the actual training behaviour at 30 epochs can still
  surprise. The framework is *necessary, not sufficient*.
- **Not a perf benchmark**. `result_capture` records elapsed time but
  doesn't compare to a baseline — that's a separate concern.

## Today's lessons that drove the design

1. **Single-seed claims are dangerous.** Yesterday's "axiom replaces attention"
   was a single-seed artefact; multi-seed inverted the conclusion. The
   framework adds reproducibility checks but doesn't substitute for
   multi-seed.
2. **Bash heuristic capture is fragile.** Three separate sweeps today
   silently produced empty JSONL files because of escape mangling,
   tail filtering, or OOM crashes. The Python `run_cells` API replaces
   the bash boilerplate.
3. **Partial wiring is the most insidious bug.** The KB init bug was
   "working" — half the model was init correctly, the optimizer
   converged to similar values, and the AUC came back identical to
   4 decimal places — perfectly suspicious but not obviously wrong.
   A positive-control assertion at the model-state layer caught it
   immediately when added.
