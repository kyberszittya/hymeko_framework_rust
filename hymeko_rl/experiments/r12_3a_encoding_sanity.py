"""R12.3A — orientation-encoding sanity check (a control, NOT the rotor test).

For a PLANAR yaw θ the encodings sin-cos = (sinθ,cosθ), z-quaternion = (cos θ/2, sin θ/2) and 2D-rotor coefficients
cos θ/2 + I sin θ/2 all encode the SAME one d.o.f. — so feeding them to the same flat MLP and expecting the "rotor"
to win would test a coordinate reparameterization, not the rotor hypothesis. The honest expectation here is therefore
≈0 difference between raw / sin-cos / quaternion. This harness fixes dataset, split, seeds, optimizer AND parameter
budget (every encoding is zero-padded to the SAME 4 extra input dims, so the MLP param count is identical) and only
varies the encoding.

The one SUBSTANTIVE control is the SYMMETRY-AWARE (sin 2θ, cos 2θ): the O5-R rectangle is 180°-equivalent, so its
natural orientation period is π, not 2π. If this wins it is NOT a "rotor win" — it is evidence that the representation
matching the object's SYMMETRY GROUP matters (the real R12.3 lesson would then be symmetry, deferred to R12.3B's
relative-frame geometry). If a quaternion/rotor suddenly wins big, suspect a periodicity / normalization / double-cover
/ feature-scaling artifact before believing it.

Run:  python -m hymeko_rl.experiments.r12_3a_encoding_sanity [family] [epochs] [n_seeds]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.transportability_critic import MLPNet, build_input_row, count_params
from hymeko_rl.experiments.r12_2b_ranker import _handoff_key, _load, _split, _train_reg
from hymeko_rl.experiments.r12_hsikan1_ablation import _auroc

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_PAD = 4                       # every encoding zero-padded to this many extra dims ⇒ identical MLP param budget
# quat == 2D-rotor coefficients for planar yaw (cos θ/2, sin θ/2) — identical, both included to make the point explicit
_ENCODINGS = ("none", "raw", "sincos", "sincos2", "quat", "rotor")


def _encode(yaw_deg: float, kind: str) -> list[float]:
    t = math.radians(yaw_deg)
    vec = {
        "none":    [],
        "raw":     [t / math.pi],
        "sincos":  [math.sin(t), math.cos(t)],
        "sincos2": [math.sin(2 * t), math.cos(2 * t)],           # symmetry-aware (π-periodic): the substantive control
        "quat":    [math.cos(t / 2), math.sin(t / 2)],           # z-quaternion nonzero components
        "rotor":   [math.cos(t / 2), math.sin(t / 2)],           # 2D rotor coefficients — identical to quat (planar)
    }[kind]
    return vec + [0.0] * (_PAD - len(vec))


def _matrix(rows: list[dict], kind: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    X = np.asarray([build_input_row(r["x"], r["theta"], r["handoff_family"]) + _encode(r["post_grasp_yaw_deg"], kind)
                    for r in rows], np.float32)
    y = np.asarray([float(r["k6"]) for r in rows], np.float32)
    dtz = np.asarray([min(float(r["dtz_mm"]), 200.0) / 200.0 for r in rows], np.float32)
    return X, y, dtz


def _run_seed(rows: list[dict], seed: int, epochs: int, dev: str) -> dict:
    tr, te = _split(rows, seed)
    tk = {_handoff_key(rows[i]) for i in tr}
    ek = {_handoff_key(rows[i]) for i in te}
    assert not (tk & ek), "LEAKAGE"
    out: dict = {}
    for kind in _ENCODINGS:
        X, y, dtz = _matrix(rows, kind)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        torch.manual_seed(seed)
        model = MLPNet(110, 4, input_dim=X.shape[1]).to(dev)     # identical budget: input_dim = 41 + _PAD for all
        _train_reg(model, torch.tensor(Xn[tr], device=dev), torch.tensor(dtz[tr], device=dev), epochs, dev)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xn[te], device=dev)).cpu().numpy()
        out[kind] = {"auroc": _auroc(y[te], -pred), "params": count_params(model)}
    return out


def _ci(v: list[float]) -> float:
    return float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = _load(fam)
    runs = [_run_seed(rows, s, epochs, dev) for s in range(n_seeds)]
    params = {runs[0][k]["params"] for k in _ENCODINGS}
    agg = {k: {"auroc": round(float(np.mean([r[k]["auroc"] for r in runs])), 3),
               "auroc_ci": round(_ci([r[k]["auroc"] for r in runs]), 3)} for k in _ENCODINGS}
    base = agg["none"]["auroc"]

    print(f"\n=== R12.3A encoding sanity [{fam}], {n_seeds} seeds × {epochs}ep, MLP, identical budget "
          f"(params {params}) ===", flush=True)
    print("encoding   AUROC          Δ vs none", flush=True)
    for k in _ENCODINGS:
        print(f"  {k:8s} {agg[k]['auroc']:.3f}±{agg[k]['auroc_ci']:.3f}   {agg[k]['auroc'] - base:+.3f}", flush=True)
    # paired Δ across seeds for the substantive comparison (symmetry-aware vs plain sin-cos)
    d_sym = [r["sincos2"]["auroc"] - r["sincos"]["auroc"] for r in runs]
    d_quat = [r["quat"]["auroc"] - r["sincos"]["auroc"] for r in runs]
    sym_m, sym_c = round(float(np.mean(d_sym)), 3), round(_ci(d_sym), 3)
    quat_m, quat_c = round(float(np.mean(d_quat)), 3), round(_ci(d_quat), 3)
    reparam = "REPARAM-NEUTRAL (quat≈sincos, as expected)" if abs(quat_m) - quat_c <= 0 else \
        "⚠️ quat≠sincos — suspect periodicity/normalization/double-cover/scaling artifact"
    symv = "SYMMETRY MATTERS — sin2θ/cos2θ beats sin-cos (object π-symmetry, NOT a rotor win)" if sym_m - sym_c > 0 else \
        "symmetry-aware ≈ sin-cos (no π-symmetry advantage at this encoding)"
    print(f"\nquat − sincos = {quat_m:+.3f}±{quat_c:.3f}  ⇒ {reparam}", flush=True)
    print(f"sin2θ − sincos = {sym_m:+.3f}±{sym_c:.3f}  ⇒ {symv}", flush=True)

    summary = {"family": fam, "epochs": epochs, "n_seeds": n_seeds, "per_encoding": agg,
               "quat_minus_sincos": quat_m, "quat_minus_sincos_ci": quat_c,
               "sincos2_minus_sincos": sym_m, "sincos2_minus_sincos_ci": sym_c}
    (_OUT / "b_encoding_sanity.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote b_encoding_sanity.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
