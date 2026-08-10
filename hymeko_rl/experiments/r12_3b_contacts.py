"""R12.3B-contacts — object↔contact relative frames (the last cheap, physically-motivated planar control).

R12.3B (transport-relative) gave a small symmetry-aware-relative signal (rel_sym +0.017 over none, not clearly above
absolute). The one relation not yet tested is the object's orientation relative to the CONTACT frames. On the existing
powered dataset (no new acquisition), the left/right contact bearings are computed from the arm joints q[0:4] via
analytic 2R FK (`PlanarArm2R.link_points`), and folded by the rectangle's π-symmetry where they involve object yaw.

Ablation (flat MLP, identical dataset/split/seeds AND budget — every encoding zero-padded to +10 dims):
  B0 none · B1 rel_sym(object,transport) · B2 contact-relative frames · B3 rel_sym + contact-relative.
Contact features: object↔left / object↔right (symmetry-aware sin/cos of 2·(yaw−bearing)); left↔transport /
right↔transport (plain sin/cos of bearing−transport).

CLOSURE RULE (user): if B2/B3 gain over none < ~0.02 OR its CI overlaps the rel_sym result ⇒ CLOSE the planar static
representation rung as `R12_3_PLANAR_RELATIVE_GEOMETRY_SMALL_SIGNAL_ONLY` (no planar HSiKAN-C). Only if a contact
feature jumps unexpectedly (stable > +0.04–0.05) is a single structure test (MLP vs task-HSiKAN, ±contact) warranted.

Run:  python -m hymeko_rl.experiments.r12_3b_contacts [family] [epochs] [n_seeds]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC, build_home_snapshot
from hymeko_rl.coin_delivery.theta_option.planar_geometric_approach import build_arms
from hymeko_rl.coin_delivery.transportability_critic import MLPNet, build_input_row, count_params
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_2b_ranker import _handoff_key, _load, _split, _train_reg
from hymeko_rl.experiments.r12_hsikan1_ablation import _auroc

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_PAD = 10
_ENC = ("none", "rel_sym", "contacts", "rel_sym_contacts")
_REL_SYM_BASELINE = 0.017        # R12.3B rel_sym − none (24 seeds); closure compares contacts against this


def _geom(rows: list[dict], fam: str) -> list[tuple[float, float, float, float]]:
    """Per row (yaw, transport, bearing_left, bearing_right) in rad. Arms calibrated once (analytic FK, coin-free)."""
    rig = _rig(object_spec=variant(fam).object_spec)
    home = build_home_snapshot(rig["cradle"], HOME_STATE_V1_GENERIC)
    arm_l, arm_r = build_arms(home, np.zeros(2))
    out = []
    for r in rows:
        x = r["x"]
        coin = np.asarray(x[8:10])
        yaw = math.radians(r["post_grasp_yaw_deg"])
        transport = math.atan2(x[15], x[14])
        tl = np.asarray(arm_l.link_points(np.asarray([x[0], x[1]]))[2])[:2]
        tr = np.asarray(arm_r.link_points(np.asarray([x[2], x[3]]))[2])[:2]
        bl = math.atan2(tl[1] - coin[1], tl[0] - coin[0])
        br = math.atan2(tr[1] - coin[1], tr[0] - coin[0])
        out.append((yaw, transport, bl, br))
    return out


def _encode(g: tuple[float, float, float, float], kind: str) -> list[float]:
    yaw, tr, bl, br = g
    rel_sym = [math.sin(2 * (yaw - tr)), math.cos(2 * (yaw - tr))]
    contacts = [math.sin(2 * (yaw - bl)), math.cos(2 * (yaw - bl)),    # object↔left (symmetry-aware)
                math.sin(2 * (yaw - br)), math.cos(2 * (yaw - br)),    # object↔right (symmetry-aware)
                math.sin(bl - tr), math.cos(bl - tr),                  # left-contact↔transport
                math.sin(br - tr), math.cos(br - tr)]                  # right-contact↔transport
    v = {"none": [], "rel_sym": rel_sym, "contacts": contacts, "rel_sym_contacts": rel_sym + contacts}[kind]
    return v + [0.0] * (_PAD - len(v))


def _matrix(rows: list[dict], geom: list, kind: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    X = np.asarray([build_input_row(r["x"], r["theta"], r["handoff_family"]) + _encode(geom[i], kind)
                    for i, r in enumerate(rows)], np.float32)
    y = np.asarray([float(r["k6"]) for r in rows], np.float32)
    dtz = np.asarray([min(float(r["dtz_mm"]), 200.0) / 200.0 for r in rows], np.float32)
    return X, y, dtz


def _run_seed(rows: list[dict], geom: list, seed: int, epochs: int, dev: str) -> dict:
    tr, te = _split(rows, seed)
    assert not ({_handoff_key(rows[i]) for i in tr} & {_handoff_key(rows[i]) for i in te}), "LEAKAGE"
    out: dict = {}
    for kind in _ENC:
        X, y, dtz = _matrix(rows, geom, kind)
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
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = _load(fam)
    geom = _geom(rows, fam)
    runs = [_run_seed(rows, geom, s, epochs, dev) for s in range(n_seeds)]
    params = {runs[0][k]["params"] for k in _ENC}
    agg = {k: {"auroc": round(float(np.mean([r[k]["auroc"] for r in runs])), 3),
               "auroc_ci": round(_ci([r[k]["auroc"] for r in runs]), 3)} for k in _ENC}

    def paired(a: str, b: str) -> "tuple[float, float]":
        d = [r[a]["auroc"] - r[b]["auroc"] for r in runs]
        return round(float(np.mean(d)), 3), round(_ci(d), 3)

    vs_none = {k: paired(k, "none") for k in _ENC if k != "none"}
    c_vs_rel = paired("contacts", "rel_sym")
    best_contact = max(("contacts", "rel_sym_contacts"), key=lambda k: vs_none[k][0])
    bm, bc = vs_none[best_contact]
    # CLOSURE per user: contacts add nothing new if gain < 0.02 OR its CI overlaps the rel_sym result (~0.017)
    small = bm < 0.02 or (bm - bc) <= _REL_SYM_BASELINE
    big = (bm - bc) > 0.04
    if big:
        verdict = (f"⚠️ CONTACT-RELATIVE JUMP — '{best_contact}' = {bm:+.3f}±{bc:.3f} > +0.04 stable ⇒ a SINGLE structure "
                   "test (MLP vs task-HSiKAN, ±contact-relative) is warranted (per the closure rule).")
    elif small:
        verdict = (f"R12_3_PLANAR_RELATIVE_GEOMETRY_SMALL_SIGNAL_ONLY — contact-relative adds no more than the "
                   f"transport-relative whisper ('{best_contact}' {bm:+.3f}±{bc:.3f} vs rel_sym baseline ~{_REL_SYM_BASELINE:.3f}; "
                   "contacts−rel_sym overlaps 0). CLOSE the planar static representation rung — no planar HSiKAN-C; the "
                   "geometric hypothesis moves to 3-D/SO(3)/Rotor-Spike (R12.4).")
    else:
        verdict = (f"intermediate — '{best_contact}' {bm:+.3f}±{bc:.3f} (between the small-signal gate and the +0.04 "
                   "structure-test threshold); treat as small-signal (close) unless it firms up.")

    print(f"\n=== R12.3B-contacts [{fam}], {n_seeds} seeds × {epochs}ep, MLP, identical budget (params {params}) ===",
          flush=True)
    print("encoding            AUROC          Δ vs none (paired)", flush=True)
    for k in _ENC:
        d = f"{vs_none[k][0]:+.3f}±{vs_none[k][1]:.3f}" if k != "none" else "  —"
        print(f"  {k:16s} {agg[k]['auroc']:.3f}±{agg[k]['auroc_ci']:.3f}   {d}", flush=True)
    print(f"\ncontacts − rel_sym = {c_vs_rel[0]:+.3f}±{c_vs_rel[1]:.3f}", flush=True)
    print(f"\n⇒ {verdict}", flush=True)

    summary = {"family": fam, "epochs": epochs, "n_seeds": n_seeds, "per_encoding": agg, "vs_none_paired": vs_none,
               "contacts_minus_rel_sym": c_vs_rel, "per_seed_auroc": {k: [round(r[k]["auroc"], 4) for r in runs] for k in _ENC},
               "verdict": verdict}
    (_OUT / "b_contacts.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote b_contacts.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
