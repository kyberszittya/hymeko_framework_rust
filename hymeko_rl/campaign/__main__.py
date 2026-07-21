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


_DOMAIN_ALIAS = {"coin": "coin_fixed_position"}   # the CLI-friendly domain name → adapter used by the flag form


def _spec_from_flags(a: argparse.Namespace) -> ExperimentSpec:
    """Build a fixed-position Coin replay spec from the flag form (``run --domain coin --seed … / --initial-state …``)."""
    domain = _DOMAIN_ALIAS.get(a.domain, a.domain)
    mode = "exact" if a.initial_state else "seed"
    opts = {"mode": mode, "policy_chain": a.policy_chain, "repetitions": a.repetitions, "embodiment": a.embodiment}
    if mode == "exact":
        opts["initial_state"] = a.initial_state
        name = f"exact_{Path(a.initial_state).stem}"
    else:
        opts["seed"] = a.seed
        name = f"seed_{a.seed}"
    return ExperimentSpec(domain=domain, experiment_name=name, model_variant=a.policy_chain,
                          dataset_or_problem=name, seed=a.seed or 0, domain_options=opts,
                          artifact_root="experiments/coin_fixed_position")


def main() -> None:
    ap = argparse.ArgumentParser(prog="hymeko_rl.campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--spec")                                      # generic JSON spec (optional if flag form is used)
    rp.add_argument("--domain")                                    # flag form: coin fixed-position replay
    rp.add_argument("--seed", type=int)
    rp.add_argument("--embodiment", default="POINT")
    rp.add_argument("--policy-chain", dest="policy_chain", default="e_handoff",
                    choices=["zero", "frozen", "e_handoff", "causal"])
    rp.add_argument("--repetitions", type=int, default=10)
    rp.add_argument("--initial-state", dest="initial_state")
    cp = sub.add_parser("campaign")
    cp.add_argument("--spec", required=True)
    for c in ("render", "verify"):
        sp = sub.add_parser(c)
        sp.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        if a.domain:
            spec = _spec_from_flags(a)
        elif a.spec:
            spec = ExperimentSpec.from_dict(json.loads(Path(a.spec).read_text()))
        else:
            ap.error("run requires either --spec <json> or the flag form (--domain coin --seed … / --initial-state …)")
        r = run_one(spec)
        print(json.dumps(r, indent=1, default=str), flush=True)
    elif a.cmd == "campaign":
        run_campaign(CampaignSpec.from_manifest(a.spec))
    elif a.cmd == "verify":
        raise SystemExit(_verify(a.run))
    elif a.cmd == "render":
        raise SystemExit(_render(a.run))


if __name__ == "__main__":
    main()
