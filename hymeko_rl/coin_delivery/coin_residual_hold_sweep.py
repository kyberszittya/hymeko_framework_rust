"""RESIDUAL_HOLD_HORIZON_LEVERAGE_SWEEP_V1 — a LABEL-ONLY temporal-leverage diagnostic (no critic, no actor update).

Question: does the local counterfactual return signal ΔG become measurable when the SAME bounded residual acts for K
consecutive *gate-active* steps before the frozen ``pi_0`` continuation? For each captured gate-active state we restore
the complete controller state (planar MuJoCo buffers + wrapper counters + a deepcopy of the StableEngagementGate FSM),
hold one fixed candidate residual for up to K env steps (gate-off steps stay bit-identical to ``pi_0``), then continue
with frozen ``pi_0`` only, and score the canonical discounted return. Candidate identity and ``state_group_id`` are
IDENTICAL across every K, so all horizon comparisons are paired. Every branch is run twice and required identical.

Residual authority (±0.25), the gate, and the reward are UNCHANGED — only the hold horizon K varies.
"""
from __future__ import annotations

import copy

import numpy as np

from hymeko_rl.coin_delivery.coin_counterfactual_labels import (
    GAMMA,
    MAGNITUDES,
    RESIDUAL_BOUND,
    _restore_rl,
    base_action,
    composite,
)
from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, HELD_DWELL, CoinRL4Dof
from hymeko_rl.coin_delivery.coin_stable_engagement import stable_engagement_signals

K_VALUES = (1, 2, 4, 8, 16)
FROZEN_ISO_SEED = 20260723
NONNEG_THRESHOLDS = (1.0, 5.0, 10.0)
BENEFIT_EPS = 1.0                         # |ΔG| below this is "neutral"
FAMILIES = ("transport", "entry", "settling", "contact_retention")


def frozen_isotropic_dirs(n: int = 3):
    """A FROZEN deterministic set of unit isotropic directions (same for every state and every K)."""
    rng = np.random.default_rng(FROZEN_ISO_SEED)
    return [(v / (np.linalg.norm(v) + 1e-9)).astype(np.float32) for v in rng.standard_normal((n, 4))]


def hold_candidates(n_iso: int = 3):
    """Candidate residual set (state-independent, deterministic): MAGNITUDES × {signed actuator bases, frozen
    isotropic}. delta==0 is the sole magnitude-0 entry. Identical identity across all states and all K."""
    out = [("zero", np.zeros(4, np.float32), {"magnitude": 0.0, "kind": "zero", "dir": "none"})]
    dirs: list[tuple[str, np.ndarray]] = []
    for k in range(4):
        e = np.zeros(4, np.float32); e[k] = 1.0
        dirs.append((f"basis+{k}", e.copy())); dirs.append((f"basis-{k}", -e.copy()))
    for j, v in enumerate(frozen_isotropic_dirs(n_iso)):
        dirs.append((f"iso{j}", v))
    for mag in MAGNITUDES[1:]:
        for name, u in dirs:
            out.append((f"{name}@{mag}", (RESIDUAL_BOUND * mag * u).astype(np.float32),
                        {"magnitude": float(mag), "kind": name.rstrip("0123456789+-"), "dir": name}))
    return out


def residual_hold_return(rl: CoinRL4Dof, pi0, snap, gate_snap, base0, delta, K: int, *, gamma: float = GAMMA):
    """Restore controller state; HOLD ``delta`` for ≤K gate-active env steps; continue with frozen ``pi_0``. Returns
    ``(discounted_return, outcome)``. gate-off steps ⇒ executed action bit-identical to ``pi_0``.

    ``base0`` is the CAPTURED ``pi_0(g.obs)`` (= ``g.base``) used for step 0 — a restore does not perfectly reproduce the
    velocity-history buffer inside ``node_features``, so reading ``rl.obs()`` at t0 gives a subtly wrong (and, compounded
    over the chaotic contact horizon, non-reproducible) base. The buffer is correct after the first ``mj_step``, so every
    later step recomputes the base from the live observation. This makes the counterfactual deterministic (verified ×2).
    """
    _restore_rl(rl, snap)
    gate = copy.deepcopy(gate_snap)
    base = np.asarray(base0, np.float32); tot, disc = 0.0, 1.0
    gate_active = disarm = reacq = 0
    m0 = rl.inner._planar_metrics
    min_dtz = rl._dtz(); ever_in = min_dtz <= CENTER_TOL; exited = False
    max_dwell = rl._strict; contact_persist = bool(m0.left_contact or m0.right_contact); clip_loss = []
    term = trunc = False; o2 = None

    def _track():
        nonlocal min_dtz, ever_in, exited, max_dwell, contact_persist
        dtz = rl._dtz(); min_dtz = min(min_dtz, dtz)
        if dtz <= CENTER_TOL:
            ever_in = True
        elif ever_in:
            exited = True
        max_dwell = max(max_dwell, rl._strict)
        mm = rl.inner._planar_metrics
        contact_persist = contact_persist and bool(mm.left_contact or mm.right_contact)

    for _k in range(K):                                    # HOLD phase (residual gated by the live FSM)
        gmult = gate.gate
        act = composite(base, gmult, delta)
        if gmult == 1.0:
            gate_active += 1
            bounded = np.clip(delta, -RESIDUAL_BOUND, RESIDUAL_BOUND)
            clip_loss.append(float(np.linalg.norm(bounded - (act - np.clip(base, -4, 4)))))
        o2, r, term, trunc, _ = rl.step(act); tot += disc * r; disc *= gamma
        lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
        prev = gate.gate; gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
        disarm += int(prev == 1.0 and gate.gate == 0.0); reacq += int(prev == 0.0 and gate.gate == 1.0)
        _track()
        if term or trunc:
            break
        base = base_action(pi0, o2)                        # live obs is correct after the first mj_step
    while not (term or trunc):                             # CONTINUATION phase (frozen pi_0 only)
        o2, r, term, trunc, _ = rl.step(base_action(pi0, o2)); tot += disc * r; disc *= gamma
        _track()
    outcome = {"K": K, "gate_active_steps": gate_active, "disarm": disarm, "reacq": reacq,
               "contact_persist": bool(contact_persist), "entered_zone": bool(ever_in), "target_exit": bool(exited),
               "overshoot": bool(exited), "min_dtz": float(min_dtz), "max_dwell": int(max_dwell),
               "strict_success": bool(term and max_dwell >= HELD_DWELL),
               "eff_clip_loss": float(np.mean(clip_loss)) if clip_loss else 0.0,
               "terminated": bool(term), "truncated": bool(trunc)}
    return float(tot), outcome


def sweep_group(rl, pi0, g, cands, k_values=K_VALUES):
    """For one state group: every candidate × every K, run TWICE, require deterministic identity. Returns a per-K dict
    {K: {names, delta, G, G0, det_ok, outcomes}}."""
    per_k = {}
    for K in k_values:
        names, G, det_ok, outs = [], [], [], []
        for name, d, _meta in cands:
            g1, o1 = residual_hold_return(rl, pi0, g.snap, g.gate_snap, g.base, d, K)
            g2, _o2 = residual_hold_return(rl, pi0, g.snap, g.gate_snap, g.base, d, K)
            if abs(g1 - g2) > 1e-9:
                raise AssertionError(f"non-deterministic hold: {g1} vs {g2} (group {g.group_id} K={K} {name})")
            names.append(name); G.append(g1); det_ok.append(True); outs.append(o1)
        per_k[K] = {"names": names, "G": G, "G0": G[0], "det_ok": det_ok, "outcomes": outs}
    return per_k


# ───────────────────────── metrics (§9) ─────────────────────────
def _iqr(v):
    v = np.asarray(v, float)
    return float(np.percentile(v, 75) - np.percentile(v, 25)) if len(v) else float("nan")


def group_leverage(per_k_entry):
    """Per-(group,K) robust leverage scalar = median |ΔG| over that state's candidates (paired unit for bootstrap)."""
    G = np.asarray(per_k_entry["G"], float); dg = np.abs(G - per_k_entry["G0"])
    return float(np.median(dg[1:])) if len(dg) > 1 else 0.0


def metrics_by_K_family(groups, results, k_values=K_VALUES):
    out = {}
    for K in k_values:
        out[K] = {}
        for fam in FAMILIES:
            gids = [g.group_id for g in groups if g.family == fam]
            if not gids:
                out[K][fam] = {"n": 0}; continue
            dg_all, gaps, ben, neu, harm, cbreak, texit, dwell_ch, ss_gain, clip, best_stable = ([] for _ in range(11))
            for gid in gids:
                e = results[gid][K]; G = np.asarray(e["G"], float); G0 = e["G0"]; dg = G - G0
                dg_all += dg[1:].tolist(); gaps.append(float(G.max() - G.min()))
                ben.append(float(np.mean(dg[1:] > BENEFIT_EPS))); harm.append(float(np.mean(dg[1:] < -BENEFIT_EPS)))
                neu.append(float(np.mean(np.abs(dg[1:]) <= BENEFIT_EPS)))
                cbreak.append(float(np.mean([not o["contact_persist"] for o in e["outcomes"][1:]])))
                texit.append(float(np.mean([o["target_exit"] for o in e["outcomes"][1:]])))
                dwell_ch.append(float(np.median([o["max_dwell"] - e["outcomes"][0]["max_dwell"] for o in e["outcomes"][1:]])))
                ss_gain.append(float(np.mean([o["strict_success"] for o in e["outcomes"][1:]]) - float(e["outcomes"][0]["strict_success"])))
                clip.append(float(np.mean([o["eff_clip_loss"] for o in e["outcomes"][1:]])))
                best_stable.append(int(all(e["det_ok"])))
            dg_abs = np.abs(dg_all)
            out[K][fam] = {
                "n": len(gids), "median_abs_dG": round(float(np.median(dg_abs)), 3), "iqr_abs_dG": round(_iqr(dg_abs), 3),
                **{f"frac_nonneg_{int(t)}": round(float(np.mean(dg_abs >= t)), 3) for t in NONNEG_THRESHOLDS},
                "median_best_worst_gap": round(float(np.median(gaps)), 3),
                "beneficial_frac": round(float(np.mean(ben)), 3), "neutral_frac": round(float(np.mean(neu)), 3),
                "harmful_frac": round(float(np.mean(harm)), 3),
                "best_action_stable": round(float(np.mean(best_stable)), 3),
                "prob_contact_break": round(float(np.mean(cbreak)), 3), "prob_target_exit": round(float(np.mean(texit)), 3),
                "median_dwell_change": round(float(np.median(dwell_ch)), 3),
                "strict_success_gain": round(float(np.mean(ss_gain)), 3),
                "median_eff_clip_loss": round(float(np.median(clip)), 3),
            }
    return out


# ───────────────────────── §10 paired bootstrap vs K=1 ─────────────────────────
def paired_bootstrap_vs_k1(groups, results, k_values=K_VALUES, n_boot=4000, seed=0):
    """For each K, paired (K − K1) difference of the per-group leverage scalar, bootstrapped by ``state_group_id``."""
    gids = [g.group_id for g in groups]
    lev = {K: np.array([group_leverage(results[gid][K]) for gid in gids]) for K in k_values}
    base = lev[1]; rng = np.random.default_rng(seed); out = {}
    for K in k_values:
        diff = lev[K] - base
        bs = [float(np.mean(diff[rng.integers(0, len(diff), len(diff))])) for _ in range(n_boot)]
        out[K] = {"mean_paired_diff": round(float(np.mean(diff)), 3),
                  "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                  "median_leverage": round(float(np.median(lev[K])), 3)}
    return out


def sweep_verdict(paired):
    """§12 conclusion from the frozen paired-bootstrap leverage gate (K vs K=1).

    signif(K) ≡ ci95_low(K − K1) > 0 (leverage increased). FLAT vs UNDERPOWERED is decided on ABSOLUTE CI resolution
    relative to the leverage scale, not relative to the (near-zero) mean — a CI whose half-width exceeds half the
    baseline leverage cannot rule out a meaningful change, so it is underpowered, not flat."""
    ks = [K for K in paired if K != 1]
    signif = {K: paired[K]["ci95"][0] > 0 for K in ks}
    lev = {K: paired[K]["median_leverage"] for K in paired}
    halfwidth = {K: (paired[K]["ci95"][1] - paired[K]["ci95"][0]) / 2 for K in ks}
    base_lev = max(lev.values()) if lev else 0.0
    if base_lev < 1e-6:                                                # no leverage scale at all to resolve
        return "RESIDUAL_HOLD_SWEEP_UNDERPOWERED"
    if not any(signif.values()):
        if all(halfwidth[K] <= 0.5 * base_lev for K in ks):           # CIs tight enough to rule out a real increase
            return "RESIDUAL_SIGNAL_FLAT_ACROSS_HOLD_HORIZON"
        return "RESIDUAL_HOLD_SWEEP_UNDERPOWERED"
    peak = max(ks, key=lambda K: lev[K])
    if peak != max(ks) and signif.get(peak) and not signif.get(max(ks), False) and lev[max(ks)] < lev[peak]:
        return "RESIDUAL_SIGNAL_HAS_FINITE_TEMPORAL_WINDOW"
    return "RESIDUAL_SIGNAL_INCREASES_WITH_HOLD_HORIZON"
