"""R10.2 Stage 2 (Boundary 3) — local conditioning of the structured-option torque-path coordinate.

Before any exploration/TD3 the 15-D option space must be shown to be *controllable and well-conditioned*, not "an
elegantly-named swamp": every dimension should move the physics, the directions should not be strongly collinear, and the
episodic exploration must perturb each dimension by a comparable *physical* amount (equal numerical std on physically
different quantities is not equal perturbation). This module provides the reusable pieces:

  * ``axis_sensitivity`` — a central-difference (``+/- eps`` in normalised ``z``-space) local audit: per-dimension response
    of the executed action, terminal actuator torque, terminal ``q``/``qvel``, and downstream ``min_dtz``; a standardised
    terminal-state Jacobian, its SVD, effective/stable rank, dead dimensions, and collinear pairs.
  * ``freeze_normalization`` — one per-dimension scale ``D`` (from the physical response magnitudes) so ``theta = sigma*D*z``
    perturbs each dimension comparably. Frozen ONCE; never retuned between exploration scales.
  * ``sample_theta`` — the episodic exploration sampler ``theta_hat = clip(sigma * D * z, -1, 1)`` (a single draw per
    episode; there is never any per-step noise).

Nothing here trains or rewards. It is diagnostics + the frozen exploration transform, used by the admissibility
experiment and (later, once approved) by the TD3 env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.theta_option.torque_path_option import THETA_DIM, TorquePathCaptureRoll

DEFAULT_EPS = 0.10             # central-difference step in normalised z-space (representative of the sigma scales)
_DEAD_FRAC = 0.05             # a dim whose response norm < 5% of the median is "dead"
_COLLINEAR_COS = 0.98         # |cos| above this between two response directions is "strongly collinear"
_REL_SV = 0.05                # singular values >= 5% of the largest count toward the effective rank
_D_CLAMP = (0.5, 2.0)         # per-dimension normalisation is clamped to avoid amplifying near-dead dims


@dataclass(frozen=True)
class AxisResponse:
    """Central-difference response of one option dimension: physical-effect magnitudes + the terminal-state direction."""

    dim: int
    d_action_rms: float           # RMS over steps of the executed action change (always defined)
    d_tau_term: float             # ||terminal prev_tau change||
    d_q_term: float               # ||terminal q change||
    d_qvel_term: float            # ||terminal qvel change||
    d_min_dtz: float              # downstream min_dtz change (task effectiveness)
    direction: np.ndarray         # 12-D terminal-state response (q, qvel, prev_tau), used for the Jacobian SVD

    @property
    def effect(self) -> float:
        return float(np.linalg.norm(self.direction))


def _terminal_state(res: dict) -> np.ndarray:
    d = res["snapshot"].branch().inner.data
    return np.concatenate([d.qpos[:4].copy(), d.qvel[:4].copy(), np.asarray(res["prev"], float)])


def _axis_response(dim: int, roller: TorquePathCaptureRoll, down: Any, eps: float) -> AxisResponse:
    zp, zm = np.zeros(THETA_DIM), np.zeros(THETA_DIM)
    zp[dim], zm[dim] = eps, -eps
    rp, rm = roller.rollout(zp.astype(np.float32)), roller.rollout(zm.astype(np.float32))
    d_acts = (np.asarray(rp["acts"]) - np.asarray(rm["acts"])) / (2 * eps)
    sp, sm = _terminal_state(rp), _terminal_state(rm)
    direction = (sp - sm) / (2 * eps)
    mdp, mdm = down.deliver(rp["snapshot"])[1], down.deliver(rm["snapshot"])[1]
    return AxisResponse(dim=dim, d_action_rms=float(np.sqrt(np.mean(d_acts ** 2))),
                        d_tau_term=float(np.linalg.norm((sp[8:12] - sm[8:12]) / (2 * eps))),
                        d_q_term=float(np.linalg.norm((sp[0:4] - sm[0:4]) / (2 * eps))),
                        d_qvel_term=float(np.linalg.norm((sp[4:8] - sm[4:8]) / (2 * eps))),
                        d_min_dtz=float((mdp - mdm) / (2 * eps)), direction=direction)


def _standardised_jacobian(responses: "list[AxisResponse]") -> np.ndarray:
    """Columns = per-dim terminal-state responses, each observable-row standardised to unit std (unitless SVD)."""
    j = np.stack([r.direction for r in responses], axis=1)        # (12, 15)
    scale = j.std(axis=1, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return j / scale


def _collinear_pairs(responses: "list[AxisResponse]") -> "list[tuple[int, int, float]]":
    pairs = []
    for i in range(len(responses)):
        for k in range(i + 1, len(responses)):
            a, b = responses[i].direction, responses[k].direction
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-9 and nb > 1e-9:
                cos = float(abs(a @ b) / (na * nb))
                if cos > _COLLINEAR_COS:
                    pairs.append((responses[i].dim, responses[k].dim, round(cos, 4)))
    return pairs


def axis_sensitivity(roller: TorquePathCaptureRoll, down: Any, eps: float = DEFAULT_EPS) -> dict:
    """Per-dimension central-difference sensitivity + a standardised terminal-state Jacobian SVD.

    # Preconditions: ``roller``'s zero-theta roll is the frozen scaffold (identity gates passed). # Postconditions: pure
    #   diagnostics — no state edit beyond the roller's own branched rollouts; deterministic given ``eps``.
    """
    responses = [_axis_response(i, roller, down, eps) for i in range(THETA_DIM)]
    effects = np.array([r.effect for r in responses])
    med = float(np.median(effects[effects > 0])) if np.any(effects > 0) else 0.0
    dead = [r.dim for r in responses if med > 0 and r.effect < _DEAD_FRAC * med]
    sv = np.linalg.svd(_standardised_jacobian(responses), compute_uv=False)
    eff_rank = int(np.sum(sv >= _REL_SV * sv[0])) if sv[0] > 0 else 0
    stable_rank = float((sv ** 2).sum() / (sv[0] ** 2)) if sv[0] > 0 else 0.0
    return {"responses": responses, "effects": effects, "median_effect": med, "dead_dims": dead,
            "singular_values": sv, "effective_rank": eff_rank, "stable_rank": round(stable_rank, 3),
            "collinear_pairs": _collinear_pairs(responses)}


def freeze_normalization(responses: "list[AxisResponse]") -> np.ndarray:
    """One frozen per-dimension scale ``D`` equalising the physical (terminal-state) effect per unit ``z``:
    ``D_i = median_effect / effect_i`` clamped to ``_D_CLAMP`` (near-dead dims are NOT amplified without bound).

    # Postconditions: ``D`` has length 15, all entries in ``_D_CLAMP``; deterministic; frozen — never retuned between
    #   exploration scales."""
    effects = np.array([r.effect for r in responses])
    med = float(np.median(effects[effects > 0])) if np.any(effects > 0) else 1.0
    with np.errstate(divide="ignore"):
        raw = np.where(effects > 1e-9, med / np.maximum(effects, 1e-9), _D_CLAMP[1])
    return np.clip(raw, _D_CLAMP[0], _D_CLAMP[1]).astype(float)


def sample_theta(sigma: float, d_norm: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The episodic exploration transform ``theta_hat = clip(sigma * D * z, -1, 1)`` — a single draw per episode.

    # Preconditions: ``d_norm`` and ``z`` have length 15. # Postconditions: ``sigma=0`` gives exactly the zero option
    #   (bit-exact scaffold); output is bounded to ``[-1,1]`` (the actor's tanh range)."""
    return np.clip(sigma * np.asarray(d_norm, float) * np.asarray(z, float), -1.0, 1.0).astype(np.float32)
