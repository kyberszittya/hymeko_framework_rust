"""Shared, domain-agnostic runner + common artifact contract.

Owns everything that is NOT domain semantics: adapter resolution, seed/resource handling, the on-disk artifact
contract, provenance capture, hashing, logging, and loud failure propagation. Domain adapters (resolved by
:data:`ADAPTERS`) own the actual execution. No domain code is imported here.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from hymeko_rl.campaign.spec import CampaignSpec, ExperimentSpec

# common artifact contract — every run produces exactly these (domain outputs are added by the adapter)
_CONTRACT = ("manifest.json", "resolved_config.json", "provenance.json", "metrics.jsonl", "result.json",
             "stdout.log", "artifact_index.json")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def capture_provenance(spec: ExperimentSpec) -> dict[str, Any]:
    """Machine + code provenance for one run (host, git, interpreter, libs); records the dirty flag, never hides it."""
    import numpy as np
    prov = {"host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
            "git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_files": len([ln for ln in _git("status", "--short").splitlines() if ln]),
            "numpy": np.__version__, "backend": spec.backend, "resource_class": spec.resource_class}
    try:
        import torch
        prov["torch"] = torch.__version__
        prov["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        prov["torch"] = None
    try:
        import mujoco
        prov["mujoco"] = mujoco.__version__
    except Exception:
        prov["mujoco"] = None
    prov.update(spec.provenance)
    return prov


def resolve_adapter(domain: str):
    """Fail loudly if the domain is unknown or routed to the wrong adapter (§1.6)."""
    from hymeko_rl.campaign.adapters import ADAPTERS
    if domain not in ADAPTERS:
        raise KeyError(f"no domain adapter for {domain!r}; registered: {sorted(ADAPTERS)} "
                       f"(a domain command must not silently fall back to another adapter)")
    return ADAPTERS[domain]()


def run_one(spec: ExperimentSpec) -> dict[str, Any]:
    """Execute one spec through its domain adapter and write the full artifact contract. Returns the result dict.

    # Postconditions the run directory contains every file in ``_CONTRACT``; on adapter failure a ``result.json`` with
      ``status='failed'`` and the traceback is still written (loud, not swallowed) and the exception re-raised.
    """
    adapter = resolve_adapter(spec.domain)
    out = Path(spec.artifact_root) / spec.run_id()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(spec.to_dict(), indent=1))
    resolved = adapter.validate(spec)                              # domain-specific validation; loud on bad options
    (out / "resolved_config.json").write_text(json.dumps(resolved, indent=1, default=str))
    (out / "provenance.json").write_text(json.dumps(capture_provenance(spec), indent=1, default=str))
    metrics_fp = (out / "metrics.jsonl").open("w")
    log_fp = (out / "stdout.log").open("w")

    def emit(metric: dict[str, Any]) -> None:
        metrics_fp.write(json.dumps(metric, default=str) + "\n")
        metrics_fp.flush()

    def log(msg: str) -> None:
        log_fp.write(msg + "\n")
        log_fp.flush()
        print(msg, flush=True)

    try:
        result = adapter.execute(spec, out, emit=emit, log=log)
        result = {"status": "ok", **result}
    except Exception as exc:                                        # loud failure: persist + re-raise
        result = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
        (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
        log(f"[FAIL] {spec.run_id()}: {exc!r}")
        metrics_fp.close()
        log_fp.close()
        raise
    finally:
        metrics_fp.close()
        log_fp.close()
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    index = {p: _sha256(out / p) for p in _CONTRACT if (out / p).exists()}
    index.update({f.name: _sha256(f) for f in sorted(out.iterdir())
                  if f.is_file() and f.name not in _CONTRACT})       # domain outputs too
    (out / "artifact_index.json").write_text(json.dumps(index, indent=1))
    return {"run_id": spec.run_id(), "out": str(out), **result}


def run_campaign(campaign: CampaignSpec) -> list[dict[str, Any]]:
    results = []
    for i, spec in enumerate(campaign.runs, 1):
        print(f"[campaign {campaign.name}] {i}/{len(campaign.runs)} → {spec.run_id()}", flush=True)
        results.append(run_one(spec))
    summary = Path(campaign.runs[0].artifact_root) / f"campaign_{campaign.name}_summary.json"
    summary.write_text(json.dumps({"name": campaign.name, "n": len(results), "results": results}, indent=2, default=str))
    print(f"[campaign {campaign.name}] wrote {summary}", flush=True)
    return results
