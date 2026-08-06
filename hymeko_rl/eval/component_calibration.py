"""Per-component critic calibration against MEASURED Monte-Carlo returns — the fair-retest trust gate.

A vector critic is only worth taking a gradient of if it has actually learned how the component return varies with
the *action*. This module answers that per component, over the action-diverse replay:

* **rank calibration** — Spearman / Kendall of Q vs the measured MC return (global);
* **sign correctness** — does Q carry the sign of a provably-signed return;
* **OOD behavior** — is Q on random/off-manifold actions below Q on in-distribution actions;
* **perturbation sensitivity** — does Q actually respond to the action (not a constant in ``a``);
* **within-state action-pair agreement** — the decisive one: holding the *state* fixed and varying only the
  *action*, does ``sign(Q(a_i) − Q(a_j))`` match ``sign(mc(a_i) − mc(a_j))``? This is precisely the property a
  usable ∂Q/∂a needs, and it is what the prior near-deterministic replay could not provide.

Pure/deterministic given the inputs; reuses the scipy-free ``spearman`` / ``kendall`` from ``critic_benchmark``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from hymeko_rl.eval.critic_benchmark import kendall, spearman
from hymeko_rl.train.search_objective import COMPONENTS

_DEGENERATE_STD = 1e-6


@dataclass
class ComponentCalibration:
    name: str
    spearman_q_mc: float
    kendall_q_mc: float
    sign_correct: float
    ood_gap: float
    within_state_rank: float
    within_state_pairs: int
    sensitivity: float
    target_mean: float
    target_std: float
    q_mean: float
    degenerate: bool
    calibrated: bool

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _within_state_agreement(q: np.ndarray, mc: np.ndarray, state_id: np.ndarray) -> tuple[float, int]:
    """Fraction of within-state action pairs whose Q-difference sign matches the measured-return sign
    (ties in the measured return excluded). Returns (agreement, n_pairs)."""
    concord = 0
    total = 0
    for sid in np.unique(state_id):
        idx = np.nonzero(state_id == sid)[0]
        if idx.size < 2:
            continue
        qi, mi = q[idx], mc[idx]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                dm = mi[a] - mi[b]
                if abs(dm) < 1e-9:
                    continue
                total += 1
                if np.sign(qi[a] - qi[b]) == np.sign(dm):
                    concord += 1
    return (concord / total if total else 0.0), total


def calibrate_component(name: str, q: np.ndarray, mc: np.ndarray, is_ood: np.ndarray,
                        state_id: np.ndarray) -> ComponentCalibration:
    """Assemble one component's calibration record.

    # Preconditions: ``q``, ``mc``, ``is_ood``, ``state_id`` are 1-D arrays of equal length.
    # Postconditions: ``calibrated`` iff the target is non-degenerate AND Q rank-correlates positively AND the
      within-state action ranking beats chance (0.5)."""
    tgt_std = float(mc.std())
    degenerate = tgt_std < _DEGENERATE_STD
    sp = spearman(q, mc)
    kd = kendall(q, mc) if len(q) <= 600 else _kendall_sampled(q, mc)
    sign_correct = float(np.mean(np.sign(q) == np.sign(mc))) if not degenerate else float("nan")
    in_mask, ood_mask = is_ood < 0.5, is_ood >= 0.5
    ood_gap = float(q[in_mask].mean() - q[ood_mask].mean()) if ood_mask.any() and in_mask.any() else float("nan")
    within, n_pairs = _within_state_agreement(q, mc, state_id)
    # sensitivity: mean over states of Q's spread across action variants, normalised by the target spread
    sens = _sensitivity(q, mc, state_id)
    calibrated = bool((not degenerate) and sp > 0.0 and within > 0.5)
    return ComponentCalibration(
        name=name, spearman_q_mc=sp, kendall_q_mc=kd,
        sign_correct=sign_correct, ood_gap=ood_gap, within_state_rank=within, within_state_pairs=n_pairs,
        sensitivity=sens, target_mean=float(mc.mean()), target_std=tgt_std, q_mean=float(q.mean()),
        degenerate=degenerate, calibrated=calibrated)


def _kendall_sampled(q: np.ndarray, mc: np.ndarray, n: int = 500, seed: int = 0) -> float:
    """Kendall on a random subsample (the O(n^2) exact form is too slow past a few hundred rows)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(q), size=min(n, len(q)), replace=False)
    return kendall(q[idx], mc[idx])


def _sensitivity(q: np.ndarray, mc: np.ndarray, state_id: np.ndarray) -> float:
    """Mean within-state std(Q) / (mean within-state std(mc) + eps): >0 means Q responds to the action."""
    q_spreads, m_spreads = [], []
    for sid in np.unique(state_id):
        idx = np.nonzero(state_id == sid)[0]
        if idx.size < 2:
            continue
        q_spreads.append(q[idx].std())
        m_spreads.append(mc[idx].std())
    if not q_spreads:
        return 0.0
    return float(np.mean(q_spreads) / (np.mean(m_spreads) + 1e-9))


def calibrate_all(critics: dict[str, Any], data: dict[str, np.ndarray], *,
                  device: str = "cpu") -> dict[str, ComponentCalibration]:
    """Calibrate every component critic against its measured MC target over the whole replay.

    # Preconditions: ``critics`` keyed by ``COMPONENTS``; ``data`` has ``obs``/``action``/``z``, ``mc_r_{c}``,
      ``is_ood``, ``state_id``.
    # Postconditions: dict ``component -> ComponentCalibration`` (no side effects on critics)."""
    dev = torch.device(device)
    obs = torch.as_tensor(data["obs"], device=dev)
    act = torch.as_tensor(data["action"], device=dev)
    z = torch.as_tensor(data["z"], device=dev)
    out: dict[str, ComponentCalibration] = {}
    for name in COMPONENTS:
        c = critics[name]
        c.eval()
        with torch.no_grad():
            q = c(obs, act, z).cpu().numpy()
        out[name] = calibrate_component(name, q, data[f"mc_r_{name}"], data["is_ood"], data["state_id"])
    return out


def calibration_table(cal: dict[str, ComponentCalibration]) -> list[dict]:
    """Flat list-of-rows for JSON / markdown rendering, in canonical component order."""
    return [cal[c].as_dict() for c in COMPONENTS if c in cal]
