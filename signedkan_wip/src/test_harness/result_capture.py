"""Robust result capture for backgrounded training cells.

Today's pain points the bash-side hadn't been reliably catching:

1. `python3 ... | tail -1` swallows JSON when warnings/progress lines
   come after the result; pipe-exit-0 looks like success even when
   python crashed.
2. OOM crashes that produce *no* stdout but write a traceback to
   stderr; the caller has nothing to grep for.
3. Subtle import failures or wheel-version mismatches that silently
   produce wrong outputs (e.g., HSIKAN_TOPK_PRUNER='balanced' typo
   silently fell back to NoOpPruner).

The wrapper here:

- Always captures full stdout+stderr to a file
- Returns a structured `CellResult` with explicit success/failure
- For success, returns the parsed JSON; for failure, returns the
  full last-50-lines tail and a parsed error class
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CellResult:
    cmd: list[str]
    env: dict[str, str]
    returncode: int
    elapsed_s: float
    stdout_path: Path
    json_result: dict | None = None
    error_class: str | None = None
    last_lines: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.json_result is not None


_ERROR_CLASSES = [
    (r"OutOfMemoryError|CUDA out of memory", "oom"),
    (r"NotImplementedError",                  "not_implemented"),
    (r"ValueError",                           "value_error"),
    (r"AssertionError",                       "assertion"),
    (r"RuntimeError",                         "runtime"),
    (r"Killed|signal: 9",                     "killed"),
    (r"Traceback",                            "traceback"),
]


def run_cell(
    cmd: list[str],
    env: dict[str, str] | None = None,
    timeout_s: int = 1800,
    log_path: Path | str = "/tmp/cell.log",
) -> CellResult:
    """Run one training cell and return a structured result.

    The full stdout+stderr is captured to `log_path`. On success
    (returncode=0 AND a JSON result line is found), `.json_result`
    is populated. On failure, `.error_class` is one of the patterns
    in `_ERROR_CLASSES`, and `.last_lines` is the last 50 lines of
    output (for human inspection).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = {}
    if env:
        full_env.update(env)
    import os
    merged_env = {**os.environ, **full_env}

    import time
    t0 = time.time()
    try:
        with open(log_path, "wb") as f:
            proc = subprocess.run(
                cmd, env=merged_env, stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    elapsed = time.time() - t0

    text = log_path.read_text(errors="replace")
    last_lines = text.splitlines()[-50:]

    json_result = None
    error_class = None

    # Try to parse the LAST JSON line of output.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}") and '"auc"' in line:
            try:
                json_result = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    # Classify the error if no JSON found.
    if json_result is None:
        for pat, cls in _ERROR_CLASSES:
            if re.search(pat, text):
                error_class = cls
                break
        else:
            error_class = "unknown_no_json"

    return CellResult(
        cmd=list(cmd), env=full_env, returncode=rc, elapsed_s=elapsed,
        stdout_path=log_path, json_result=json_result,
        error_class=error_class, last_lines=last_lines,
    )


def run_cells(
    specs: list[tuple[str, list[str], dict[str, str]]],
    output_jsonl: Path | str,
    timeout_s: int = 1800,
    log_dir: Path | str = "/tmp/cells",
) -> dict[str, CellResult]:
    """Run a sequence of (label, cmd, env) cells; emit per-cell JSON
    to `output_jsonl` plus a final summary."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(output_jsonl)
    out_handle = output_jsonl.open("w")

    results = {}
    for label, cmd, env in specs:
        log_path = log_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', label)}.log"
        result = run_cell(cmd, env=env, timeout_s=timeout_s, log_path=log_path)
        results[label] = result
        record = {
            "label": label,
            "ok": result.ok,
            "elapsed_s": result.elapsed_s,
            "error_class": result.error_class if not result.ok else None,
            "json": result.json_result,
            "last_lines": (result.last_lines if not result.ok else None),
        }
        out_handle.write(json.dumps(record) + "\n")
        out_handle.flush()
        marker = "OK" if result.ok else f"FAIL ({result.error_class})"
        print(f"[{label}] {marker} {result.elapsed_s:.1f}s")
        if not result.ok:
            for line in result.last_lines[-5:]:
                print(f"  | {line}")

    out_handle.close()
    return results
