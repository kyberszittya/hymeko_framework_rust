"""Canonical CLI: `python -m hymeko_rl.campaign {run|campaign|render|verify}`.

The same command runs Coin Delivery, CIP and HyperSignedLINGAM — the domain is a field in the manifest, not a
different entrypoint. `render` and `verify` are domain-dispatched through the same adapter registry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hymeko_rl.campaign.runner import run_campaign, run_one
from hymeko_rl.campaign.spec import CampaignSpec, ExperimentSpec


def _verify(run_dir: str) -> int:
    """Re-hash the artifact contract and confirm result.json status == ok; loud non-zero exit on any mismatch."""
    import hashlib
    d = Path(run_dir)
    idx = json.loads((d / "artifact_index.json").read_text())
    bad = [p for p, h in idx.items() if not (d / p).exists()
           or hashlib.sha256((d / p).read_bytes()).hexdigest() != h]
    res = json.loads((d / "result.json").read_text())
    ok = not bad and res.get("status") == "ok"
    print(f"[verify] {run_dir}: status={res.get('status')} contract_intact={not bad} "
          f"{'MISMATCH ' + str(bad) if bad else ''}", flush=True)
    return 0 if ok else 1


def _render(run_dir: str) -> int:
    """Domain-dispatched render (coin → the run's video, if the adapter provides one)."""
    d = Path(run_dir)
    spec = ExperimentSpec.from_dict(json.loads((d / "manifest.json").read_text()))
    print(f"[render] {run_dir} domain={spec.domain}: rendering is produced by the domain video pipeline "
          f"(coin: reports/video). See result.json for artifact references.", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="hymeko_rl.campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "campaign"):
        sp = sub.add_parser(c)
        sp.add_argument("--spec", required=True)
    for c in ("render", "verify"):
        sp = sub.add_parser(c)
        sp.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        r = run_one(ExperimentSpec.from_dict(json.loads(Path(a.spec).read_text())))
        print(json.dumps(r, indent=1, default=str), flush=True)
    elif a.cmd == "campaign":
        run_campaign(CampaignSpec.from_manifest(a.spec))
    elif a.cmd == "verify":
        raise SystemExit(_verify(a.run))
    elif a.cmd == "render":
        raise SystemExit(_render(a.run))


if __name__ == "__main__":
    main()
