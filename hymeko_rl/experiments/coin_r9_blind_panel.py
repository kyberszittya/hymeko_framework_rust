"""R9 STAGE 1 — freeze a NEW BLIND final panel (pre-registration).

s4/s7 are now VALIDATION states; this seals a fresh set of UNSEEN cradles for the single STAGE-6 evaluation. The selection
rule uses ONLY cradle-GENERATION quantities — the acquisition seed + the t=0 straddle geometry (initial transport distance,
left/right coin offset, contact axis, initial normals/forces). It NEVER runs the delivery scaffold, the oracle, the R8
policy or the R9 policy, and NEVER reads dtz_end / K6 / reward — those stay unseen until STAGE 6.

Rule (deterministic, fixed before any certification is seen):
  1. candidate seeds = the canonical enumeration 14000+250·si for a FIXED fresh si-list (excludes the 4 frozen panel seeds
     and the base seed) — no seed chosen for its outcome.
  2. structurally certify each (both-contact ∧ internal-force-feasible ∧ straddle) via the read-only scout; drop failures.
  3. read each certified cradle's t=0 geometry from snap.branch() WITHOUT stepping (no rollout).
  4. sort the certified cradles by initial transport distance dtz0 and take rank {0, n//3, 2n//3, n-1} — 4 cradles spanning
     SHORT→LONG transport. Report each pick's L/R offset + contact axis to document the realised asymmetry / contact-geometry
     spread. Object shape + scene are kept IDENTICAL to R8 (cylinder coin, same zone) so the STAGE-6 transfer test isolates
     unseen cradle CONFIGURATIONS, not object-shape generalisation (a separate axis).

Emits `final_panel_manifest.json` with status SEALED_NOT_EVALUATED + the generation rule, seeds, geometry, and content
hashes. This is the FIRST R9 commit; no R9 controller/training code runs before it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option.teacher_bank import FROZEN_SEEDS, acquire_snapshot, load_harness

OUT = "reports/2026-07-27-coin-r9-causal-residual-delivery"
CANDIDATE_SI = (2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)  # FIXED fresh list (excludes base si=0 + panel si {1,3,4,7}); extended to yield >=4 STRUCTURALLY-certified straddles


def _t0_geometry(snap: Any) -> dict:
    """Read the t=0 cradle geometry from the handoff branch WITHOUT stepping (generation quantities only — no delivery)."""
    rl = snap.branch()
    u, dtz0 = rl.inner.direction_to_zone()
    coin = np.asarray(_coin_xy(rl), np.float64)
    return {"dtz0": round(float(dtz0), 5), "coin_x0": round(float(coin[0]), 5), "coin_y0": round(float(coin[1]), 5),
            "zone_dir": [round(float(x), 4) for x in np.asarray(u, np.float64)[:2]],
            "straddle0": round(float(snap.straddle0), 4), "fn0": [round(float(x), 4) for x in snap.fn0]}


def _scout(harness: Any) -> list:
    """Structurally certify each fresh candidate seed + read its t=0 geometry. Live progress. NO delivery/K6/reward."""
    rows = []
    for si in CANDIDATE_SI:
        seed = 14000 + 250 * si
        t0 = time.time()
        snap, meta = acquire_snapshot(harness, seed)
        ok = snap is not None
        row = {"si": si, "seed": seed, "certified": bool(ok), "axis": meta.get("axis"),
               "n_dot": (round(float(meta["n_dot"]), 4) if meta.get("n_dot") is not None else None),
               "post_release_hash": (snap.post_release_hash if ok else None),
               "geometry": (_t0_geometry(snap) if ok else None)}
        rows.append(row)
        print(f"  scout si={si} seed={seed} certified={ok} axis={row['axis']} "
              f"dtz0={row['geometry']['dtz0'] if ok else '-'} coin_x0={row['geometry']['coin_x0'] if ok else '-'} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return rows


def _select_span(certified: list) -> list:
    """Deterministic transport-spanning pick: sort by dtz0, take rank {0, n//3, 2n//3, n-1} (dedup, ordered)."""
    s = sorted(certified, key=lambda r: r["geometry"]["dtz0"])
    n = len(s)
    idx = sorted({0, n // 3, 2 * n // 3, n - 1})
    return [s[i] for i in idx]


def _sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def _gh(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def freeze_blind_panel() -> dict:
    """STAGE 1 — build + SEAL the fresh blind final panel. # Postconditions: writes final_panel_manifest.json with status
    SEALED_NOT_EVALUATED; no delivery outcome is ever computed."""
    import os
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    harness = load_harness()
    rows = _scout(harness)
    certified = [r for r in rows if r["certified"]]
    if len(certified) < 4:
        raise RuntimeError(f"only {len(certified)} fresh cradles certified — extend CANDIDATE_SI before sealing")
    picks = _select_span(certified)
    for k, p in enumerate(picks):                                 # blind tags f1..f4 in transport order (no outcome info)
        p["final_tag"] = f"f{k+1}"
        p["transport_rank"] = k
    env_hash = _sha("hymeko_rl/env/planar_grasp_env.py")
    manifest = {
        "contract": "COIN_R9_FINAL_PANEL", "status": "SEALED_NOT_EVALUATED", "date": "2026-07-27",
        "purpose": "fresh unseen cradles for the single STAGE-6 R9 evaluation; s4/s7 are now validation-only",
        "generation_rule": ("seed=14000+250*si over FIXED fresh si-list; structural straddle certification (no delivery); "
                            "read t=0 geometry only; sort by dtz0, pick rank {0,n//3,2n//3,n-1} spanning SHORT->LONG "
                            "transport; scene+object IDENTICAL to R8 (cylinder coin, same zone) to isolate cradle-config "
                            "transfer; NO policy/oracle/reward/K6 used in selection"),
        "candidate_si": list(CANDIDATE_SI), "frozen_panel_seeds_excluded": list(FROZEN_SEEDS),
        "n_candidates": len(rows), "n_certified": len(certified),
        "scene": "compose_planar_scene defaults from load_harness (cylinder coin) — same physical regime as R8",
        "env_physics_hash16": {"planar_grasp_env": env_hash,
                               "teacher_bank": _sha("hymeko_rl/coin_delivery/theta_option/teacher_bank.py"),
                               "velocity_transport": _sha("hymeko_rl/coin_delivery/theta_option/velocity_transport.py"),
                               "motion_contract": _sha("hymeko_rl/env/motion_contract.py")},
        "source_tag": "coin-r8-bounded-residual-heldout-improvement-v1", "branch": _gh("branch", "--show-current"),
        "head": _gh("rev-parse", "HEAD"),
        "panel": [{"final_tag": p["final_tag"], "seed": p["seed"], "si": p["si"], "axis": p["axis"], "n_dot": p["n_dot"],
                   "transport_rank": p["transport_rank"], "post_release_hash": p["post_release_hash"],
                   "geometry": p["geometry"]} for p in picks],
        "all_certified": [{"si": r["si"], "seed": r["seed"], "axis": r["axis"], "dtz0": r["geometry"]["dtz0"],
                           "coin_x0": r["geometry"]["coin_x0"]} for r in certified],
        "seal": "DO NOT run scaffold/oracle/R8/R9 on this panel; DO NOT read dtz_end/K6/reward until the frozen R9 "
                "checkpoint at STAGE 6.",
        "wall_s": round(time.time() - t0, 1)}
    json.dump(manifest, open(f"{OUT}/final_panel_manifest.json", "w"), indent=1, default=float)
    print(f"\n== R9 STAGE 1 — BLIND PANEL SEALED ==\n  certified {len(certified)}/{len(rows)}; sealed {len(picks)} "
          f"(tags {[p['final_tag'] for p in picks]}, seeds {[p['seed'] for p in picks]}, "
          f"dtz0 {[p['geometry']['dtz0'] for p in picks]}) | status SEALED_NOT_EVALUATED | wall {manifest['wall_s']}s\n"
          f"R9_STAGE1_DONE", flush=True)
    return manifest


if __name__ == "__main__":
    freeze_blind_panel()
