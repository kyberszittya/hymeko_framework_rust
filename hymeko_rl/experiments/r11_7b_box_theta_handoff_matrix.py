"""R11.7B — box θ×handoff transfer matrix (causal audit) + flagship certificate freeze.

The decisive audit before any stabilization work: on the box's DEV handoff snapshots that reached delivery, apply
EVERY stored box-bank θ (not the retrieval policy) → strict-K6 / dtz / safety. This separates two very different
failure modes and stops us grinding on a mis-identified component (the coin-arc mistake):

  * a delivering θ exists in the bank but retrieval doesn't select it ⇒ BOX_RETRIEVAL_SELECTION_LIMIT (fix selector);
  * no bank θ delivers on the snapshot                              ⇒ BOX_RETRIEVAL_BANK_COVERAGE_LIMIT (densify).

It also freezes the flagship: the one dev snapshot+θ that produced the first non-circular exact-zero strict-K6, with
full provenance (HyMeKo-spec hash, generated-model fingerprint, exact-zero IC, capture certificate, bank+selected θ,
strict-K6 signals). Claim stays exact: existence demonstrated; robust deployable box policy not yet established.

Run:  python -m hymeko_rl.experiments.r11_7b_box_theta_handoff_matrix
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-untyped]  # mujoco ships no stubs / py.typed marker
import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.retrieval import RetrievalConfig, RetrievalDeliveryPolicy, SelectRule
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals, reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_7a_u6b_box_pilot import BOX_DEV, _VARIANT


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
        return r.stdout.strip() or "unknown"
    except (OSError, ValueError):
        return "unknown"

_OUT = Path("reports/2026-08-06-r11-7a-u6b-box-pilot")
_BANK = _OUT / "bank.json"
_N_SEEDS = 5
_SCENE = "data/robotics/galambos_env_o4_square.hymeko"
_CENTER_TOL_MM = 20.0            # CENTER_TOL 0.02 m — the strict-K6 zone tolerance


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _model_fingerprint(model: Any) -> str:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "disk")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk")
    parts = [model.nq, model.nv, model.nu, model.ngeom, model.nbody, int(model.geom_type[gid]),
             *[round(float(v), 9) for v in model.geom_size[gid]], round(float(model.body_mass[bid]), 9),
             *[int(v) for v in model.geom_contype], *[int(v) for v in model.geom_conaffinity]]
    return _sha(json.dumps(parts))


def _policies(bank: dict) -> dict[str, RetrievalDeliveryPolicy]:
    X = np.asarray([s["x"] for s in bank["samples"]], np.float64)
    Theta = np.asarray([s["theta"] for s in bank["samples"]], np.float64)
    surv = np.ones(len(bank["samples"]), np.float64)
    out = {"top1_nearest": RetrievalDeliveryPolicy.fit(X, Theta, surv,
                                                       RetrievalConfig(True, 1, SelectRule.NEAREST))}
    if len(bank["samples"]) >= 3:
        out["k3_weighted"] = RetrievalDeliveryPolicy.fit(X, Theta, surv,
                                                         RetrievalConfig(True, 3, SelectRule.DIST_WEIGHTED))
    return out


def run(bank_path: Path = _BANK, out_suffix: str = "") -> int:
    bank = json.loads(bank_path.read_text())
    thetas = [np.asarray(s["theta"], np.float64) for s in bank["samples"]]
    theta_src = [s["scenario_id"] for s in bank["samples"]]
    pols = _policies(bank)
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)
    model_fp = _model_fingerprint(rig["cradle"].branch().inner.model)
    hymeko_hash = _sha(Path(_SCENE).read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    flagship: "dict[str, Any] | None" = None
    for sid in BOX_DEV:
        for seed in range(_N_SEEDS):
            h = reach_capture_descriptor(rig, scenario_by_id(sid), seed, cfg, conf, obj)
            if h.record is not None:
                rows.append({"scenario_id": sid, "seed": seed, "reached_delivery": False,
                             "pre_delivery": h.record.outcome_class})
                print(f"[{sid:22s} s{seed}] PRE-DELIVERY {h.record.outcome_class}", flush=True)
                continue
            snap, x = h.snap, np.asarray(h.x, np.float64)
            matrix = []
            for i, th in enumerate(thetas):
                s = _delivery_signals(snap, th)
                matrix.append({"theta_src": theta_src[i], "k6": bool(s.k6), "dtz_mm": s.dtz_mm, "safe": bool(s.safe)})
            deliver_idx = [i for i, m in enumerate(matrix) if m["k6"]]
            best_dtz = min((m["dtz_mm"] for m in matrix), default=None)
            retr = {p: _delivery_signals(snap, pol.predict(x)) for p, pol in pols.items()}
            row = {"scenario_id": sid, "seed": seed, "reached_delivery": True,
                   "n_delivering_theta": len(deliver_idx), "delivering_theta_src": [theta_src[i] for i in deliver_idx],
                   "best_dtz_any_theta": best_dtz,
                   **{f"{p}_k6": bool(s.k6) for p, s in retr.items()},
                   **{f"{p}_dtz": s.dtz_mm for p, s in retr.items()}, "matrix": matrix}
            rows.append(row)
            print(f"[{sid:22s} s{seed}] reached; delivering θ={len(deliver_idx)}/{len(thetas)} "
                  f"best_dtz={best_dtz} | " + " ".join(f"{p}={s.k6}({s.dtz_mm})" for p, s in retr.items()), flush=True)
            # freeze the flagship: first delivery-reaching snapshot where SOME retrieval policy hits strict K6
            if flagship is None:
                hit = next((p for p, s in retr.items() if s.k6), None)
                if hit is not None:
                    s = retr[hit]
                    theta_is_blend = hit == "k3_weighted"
                    flagship = {
                        "certificate": "R11_7A_FIRST_NONCIRCULAR_EXACT_ZERO_K6_CERTIFICATE",
                        "claim": "Existence demonstrated; robust deployable box policy not yet established.",
                        "theta_provenance": ("k3_weighted distance-blend of the 3 nearest bank theta (an INTERPOLATION, "
                                             "not a stored teacher theta)" if theta_is_blend else "stored teacher theta")
                        + "; the delivery is real physics under the frozen certified K6 monitor.",
                        "deploy_free_of": "runtime teacher, CEM, and oracle (theta from the frozen retrieval of offline "
                                          "teacher samples)",
                        "n_stored_theta_delivering_here": len(deliver_idx),
                        "best_stored_theta_dtz_mm": best_dtz,
                        "variant": _VARIANT, "shape": "box", "scenario_id": sid, "seed": seed, "policy": hit,
                        "hymeko_spec": _SCENE, "hymeko_spec_hash": hymeko_hash, "generated_model_fingerprint": model_fp,
                        "exact_zero_ic": "q=[0,0,0,0]", "coin_xy": [round(float(v), 5) for v in scenario_by_id(sid).coin_xy],
                        "selected_theta": [round(float(t), 6) for t in pols[hit].predict(x)],
                        "strict_k6": {"k6": True, "dtz_mm": s.dtz_mm, "center_tol_mm": _CENTER_TOL_MM,
                                      "valid_delivery": bool(s.valid_delivery), "safe": bool(s.safe),
                                      "gap_closed": s.gap_closed},
                        "bank_thetas_src": theta_src, "n_bank_thetas": len(thetas),
                        "descriptor": [round(float(v), 6) for v in x]}

    res = _summarize(rows, bank, flagship, model_fp, hymeko_hash)
    res["bank"] = str(bank_path)
    (_OUT / f"theta_handoff_matrix{out_suffix}.json").write_text(json.dumps({"rows": rows, **res}, indent=1, default=str))
    if flagship is not None and out_suffix == "":              # freeze the flagship only on the primary (sparse) audit
        (_OUT / "flagship_certificate.json").write_text(json.dumps(flagship, indent=1, default=str))
    print(f"\nAUDIT VERDICT: {res['audit_verdict']} (bank={bank_path.name}, {len(bank['samples'])} θ)")
    print(f"  reached delivery: {res['n_reached']} | with a delivering bank θ: {res['n_coverage_ok']} | "
          f"selection-gap (θ exists, top-1 misses): {res['n_selection_gap']}")
    return 0


def _summarize(rows: list[dict], bank: dict, flagship: "dict | None", model_fp: str, hymeko_hash: str) -> dict:
    reached = [r for r in rows if r["reached_delivery"]]
    coverage_ok = [r for r in reached if r["n_delivering_theta"] > 0]
    no_coverage = [r for r in reached if r["n_delivering_theta"] == 0]
    selection_gap = [r for r in reached if r["n_delivering_theta"] > 0 and not r.get("top1_nearest_k6", False)]
    # Verdict: if delivering θ exist on snapshots the deploy top-1 misses ⇒ SELECTION limit dominates;
    # if reached snapshots have no delivering θ at all ⇒ COVERAGE limit; report the mix honestly.
    if reached and len(no_coverage) >= max(1, len(reached) // 2):
        verdict = "BOX_RETRIEVAL_BANK_COVERAGE_LIMIT"
    elif selection_gap:
        verdict = "BOX_RETRIEVAL_SELECTION_LIMIT"
    elif reached and not no_coverage:
        verdict = "BOX_RETRIEVAL_SELECTION_LIMIT"      # every reached snapshot has a delivering θ
    else:
        verdict = "BOX_RETRIEVAL_INCONCLUSIVE"
    return {"audit_verdict": verdict, "n_reached": len(reached), "n_coverage_ok": len(coverage_ok),
            "n_no_coverage": len(no_coverage), "n_selection_gap": len(selection_gap),
            "model_fingerprint": model_fp, "hymeko_spec_hash": hymeko_hash, "git_sha": _git_sha()}


if __name__ == "__main__":
    import sys

    # optional arg: bank filename under the pilot dir (e.g. bank_dense.json) → suffixed matrix output
    if len(sys.argv) > 1:
        bp = _OUT / sys.argv[1]
        suffix = "_dense" if "dense" in sys.argv[1] else "_" + bp.stem
        sys.exit(run(bp, suffix))
    sys.exit(run())
