"""R12.3B — relative-frame geometry: does the object's orientation RELATIVE to the transport direction carry ranking
signal that ABSOLUTE yaw does not?

R12.2-B showed absolute yaw is redundant beside the handoff descriptor. R12.3A showed the ENCODING of that 1-DOF is
irrelevant. The substantive geometric hypothesis that survives in 2-D (SO(2) relative rotor = angle subtraction; the
genuine non-commutative rotor-composition advantage is SO(3)/3-D, deferred) is: the object orientation relative to the
frame it must move into. Here the target/transport frame is `atan2(req_transport)` (verified = direction of
target−coin, and it VARIES across handoffs), so `rel = yaw − transport_angle` is genuinely distinct from absolute yaw.

Ablation (flat MLP, identical dataset/split/seeds/optimizer AND param budget — every encoding zero-padded to the same
+4 dims): B0 none · B1 absolute sin/cos yaw · B3 relative sin/cos(yaw−transport) · B3-sym symmetry-aware
sin/cos(2·rel) · B1+B3 both. The decisive comparison is REL vs ABS vs NONE: if REL beats ABS/NONE, relative-frame
geometry carries signal absolute orientation does not (the real R12.3 lesson on this substrate).

Run:  python -m hymeko_rl.experiments.r12_3b_relative_frames [family] [epochs] [n_seeds]
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
_PAD = 4
_ENC = ("none", "abs", "rel", "rel_sym", "abs_rel")


def _transport_angle(x: list[float]) -> float:
    return math.atan2(x[15], x[14])                          # req_transport (= target − coin) direction; verified


def _encode(x: list[float], yaw_deg: float, kind: str) -> list[float]:
    t = math.radians(yaw_deg)
    rel = t - _transport_angle(x)                            # object orientation relative to the transport frame
    v = {
        "none":    [],
        "abs":     [math.sin(t), math.cos(t)],
        "rel":     [math.sin(rel), math.cos(rel)],
        "rel_sym": [math.sin(2 * rel), math.cos(2 * rel)],   # π-symmetry-aware relative (rectangle 180°-equivalent)
        "abs_rel": [math.sin(t), math.cos(t), math.sin(rel), math.cos(rel)],
    }[kind]
    return v + [0.0] * (_PAD - len(v))


def _matrix(rows: list[dict], kind: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    X = np.asarray([build_input_row(r["x"], r["theta"], r["handoff_family"])
                    + _encode(r["x"], r["post_grasp_yaw_deg"], kind) for r in rows], np.float32)
    y = np.asarray([float(r["k6"]) for r in rows], np.float32)
    dtz = np.asarray([min(float(r["dtz_mm"]), 200.0) / 200.0 for r in rows], np.float32)
    return X, y, dtz


def _run_seed(rows: list[dict], seed: int, epochs: int, dev: str) -> dict:
    tr, te = _split(rows, seed)
    assert not ({_handoff_key(rows[i]) for i in tr} & {_handoff_key(rows[i]) for i in te}), "LEAKAGE"
    out: dict = {}
    for kind in _ENC:
        X, y, dtz = _matrix(rows, kind)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xn = (X - mu) / sd
        torch.manual_seed(seed)
        model = MLPNet(110, 4, input_dim=X.shape[1]).to(dev)
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
    params = {runs[0][k]["params"] for k in _ENC}
    agg = {k: {"auroc": round(float(np.mean([r[k]["auroc"] for r in runs])), 3),
               "auroc_ci": round(_ci([r[k]["auroc"] for r in runs]), 3)} for k in _ENC}

    def paired(a: str, b: str) -> "tuple[float, float]":
        d = [r[a]["auroc"] - r[b]["auroc"] for r in runs]
        return round(float(np.mean(d)), 3), round(_ci(d), 3)

    vs_none = {k: paired(k, "none") for k in _ENC if k != "none"}       # every encoding − none, PAIRED across seeds
    relsym_abs = paired("rel_sym", "abs")
    # best non-trivial encoding whose paired gain over none is CI-excl-0
    winners = [k for k, (m, c) in vs_none.items() if m - c > 0]
    best = max(vs_none, key=lambda k: vs_none[k][0])
    if winners:
        verdict = (f"GEOMETRIC SIGNAL — '{best}' beats none (CI-excl-0): {vs_none[best][0]:+.3f}±{vs_none[best][1]:.3f}. "
                   + ("SYMMETRY-AWARE RELATIVE wins (object-axis vs transport, mod the rectangle's 180° symmetry) ⇒ the "
                      "representation matching the object's SYMMETRY GROUP + relative frame carries signal absolute yaw "
                      "did not — NOT a rotor/encoding win (A showed encodings are neutral)."
                      if best in ("rel_sym",) else
                      "the relative-frame orientation carries signal absolute yaw did not."))
    else:
        verdict = ("no encoding beats none (all paired CIs include 0) — on this planar substrate neither absolute nor "
                   "relative orientation adds ranking signal; the geometric signal, if any, is in contact frames / 3-D "
                   "rotor composition / dynamic Rotor-Spike (R12.4).")

    print(f"\n=== R12.3B relative-frame geometry [{fam}], {n_seeds} seeds × {epochs}ep, MLP, identical budget "
          f"(params {params}) ===", flush=True)
    print("encoding   AUROC          Δ vs none (paired)", flush=True)
    for k in _ENC:
        d = f"{vs_none[k][0]:+.3f}±{vs_none[k][1]:.3f}" if k != "none" else "  —"
        print(f"  {k:8s} {agg[k]['auroc']:.3f}±{agg[k]['auroc_ci']:.3f}   {d}", flush=True)
    print(f"\nrel_sym − abs = {relsym_abs[0]:+.3f}±{relsym_abs[1]:.3f}", flush=True)
    print(f"\n⇒ {verdict}", flush=True)

    summary = {"family": fam, "epochs": epochs, "n_seeds": n_seeds, "per_encoding": agg,
               "vs_none_paired": vs_none, "rel_sym_minus_abs": relsym_abs,
               "per_seed_auroc": {k: [round(r[k]["auroc"], 4) for r in runs] for k in _ENC},
               "winners": winners, "best": best, "verdict": verdict}
    (_OUT / "b_relative_frames.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote b_relative_frames.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
