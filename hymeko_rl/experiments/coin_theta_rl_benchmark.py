"""COIN TEACHER-TO-RL benchmark — one harness, mode flags (§6.5 #13: modes, not v-files).

Pipeline over the frozen 6-D torque-θ delivery option (tag coin-physical-feasibility-closed, a3459629):
    frozen CEM teacher bank → structured causal θ dataset → BC proposal → update-0 no-regression →
    matched SAC/TD3 smoke → matched multi-seed.

Modes (each STOPS at its gate; downstream artifacts only when authorised):
    --semantics       emit option_semantics.json (Stage 0)
    --teacher-bank    reproduce the 4 canonical trajectories + dev-only CEM augmentation; freeze teacher_bank.json (Stage 1)
    --dataset         build the structured causal θ dataset + splits; dataset_contract.json (Stage 2)
    --bc              fit B0/B1/B2 proposals; bc_results.json (Stage 3)
    --update0         update-0 deploy on the frozen 4-state panel; update_zero.json (Stage 4)
    --rl-smoke        matched SAC/TD3 one-seed smoke (Stage 6, gated on Stage 4)
    --rl-multiseed    matched multi-seed SAC/TD3 (Stage 7, gated)

Run:  python -m hymeko_rl.experiments.coin_theta_rl_benchmark --<mode> [--smoke]
"""
from __future__ import annotations

import json
import os
import sys

REPORT_DIR = "reports/2026-07-27-coin-teacher-to-rl"


def _dump(obj: dict, name: str) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = f"{REPORT_DIR}/{name}"
    json.dump(obj, open(path, "w"), indent=1, default=float)
    return path


def semantics_main() -> dict:
    """Stage 0 — freeze and emit the 6-D torque-θ option semantics."""
    from hymeko_rl.coin_delivery.theta_option.semantics import option_semantics
    sem = option_semantics()
    path = _dump(sem, "option_semantics.json")
    print(f"OPTION SEMANTICS frozen → {path}\n  dim={sem['dim']} components={[c['name'] for c in sem['components']]}\n"
          f"  K6: CENTER_TOL={sem['termination_and_k6']['CENTER_TOL_m']} SETTLE_VEL="
          f"{sem['termination_and_k6']['SETTLE_VEL_mps']} HELD_DWELL={sem['termination_and_k6']['HELD_DWELL_steps']}\n"
          f"  Bellman action = θ_0 (proposal centre); θ_exec = search provenance only\nSEMANTICS_DONE", flush=True)
    return sem


if __name__ == "__main__":
    if "--semantics" in sys.argv:
        semantics_main()
    else:
        print("specify a mode: --semantics | --teacher-bank | --dataset | --bc | --update0 | --rl-smoke | --rl-multiseed")
        sys.exit(2)
