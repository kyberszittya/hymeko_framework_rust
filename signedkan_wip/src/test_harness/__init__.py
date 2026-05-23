"""HSiKAN pipeline integrity test framework.

Three layers:

- `env_var_audit`: positive-control tests that every HSIKAN_* env var
  actually changes the model's state in the expected way. Catches the
  "partial wiring" bug class (e.g., outer KB init not patched).
- `result_capture`: robust JSON-or-traceback capture for backgrounded
  training cells. Replaces `cmd ... | grep '"auc"'` heuristics with
  proper exit-code + parsed-error classification.
- `invariant_check`: pre-flight 1-epoch smoke test to verify a config
  produces sensible outputs (no NaN, AUC ∈ [0, 1], etc.) before
  burning 15 minutes per cell on a long sweep.

CLI entry points:

    python -m signedkan_wip.src.test_harness.env_var_audit
    python -m signedkan_wip.src.test_harness.invariant_check --dataset bitcoin_alpha

Pytest entry point: `signedkan_wip/tests/test_pipeline_integrity.py`.
"""

from .env_var_audit import run_audit, AuditCase
from .result_capture import run_cell, run_cells, CellResult
from .invariant_check import smoke_test
