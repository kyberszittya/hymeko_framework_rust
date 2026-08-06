"""Gradient-alignment probe — the decisive, exact measurement of the fair vector-critic retest.

From a fixed CONTACT/PUSH state, roll each candidate *first action* forward under the frozen policy at a horizon
long enough to see contact/progress change (the prior k=15 was insensitive), and read the frozen ``TaskMonitor``
verdict on the resulting branch. Candidates:

* ``dagger``      — the frozen policy action π(s) (the baseline to preserve);
* ``scalar``      — ``clip(π(s) + η·∇_a Q_total)`` (the scalar-critic direction under test);
* ``projected``   — ``clip(π(s) + η·g_proj)`` (the vector constraint-projected direction, unit, scale-normalized);
* ``random``      — a uniform action (OOD floor);
* ``best_sampled``— the best of K local perturbations by *measured* objective return (the local action-landscape
  ceiling — if even this can't beat DAgger, there is no local improvement to chase).

The gate opens only if ``projected`` preserves/improves contact ∧ progress ∧ monitor_score AND beats ``scalar`` on
them. This module measures; the driver decides.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from hymeko_rl.env.planar_snapshot import PlanarSnapshot, restore_planar
from hymeko_rl.train.search_objective import COMPONENTS, SearchObjective
from hymeko_rl.train.vector_critic import action_gradient, projected_gradient

_CANDIDATES = ("dagger", "scalar", "projected", "random", "best_sampled")


@dataclass
class ProbeConfig:
    probe_horizon: int = 200
    probe_eta: float = 0.4                 # action-space step along a unit direction (‖a‖ range is [-4,4])
    n_best_sampled: int = 8
    best_sampled_sigma: float = 0.5
    gamma: float = 0.99
    seed: int = 0
    normalize_projection: bool = True


def _greedy(actor: Any, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return actor(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy().astype(np.float32)


def _branch_measure(env: Any, frozen_actor: Any, snap: PlanarSnapshot, first_action: np.ndarray, *,
                    monitor: Any, objective: SearchObjective, horizon: int) -> dict[str, float]:
    """Roll ``first_action`` then frozen π from ``snap`` for ``horizon`` steps; return branch physical stats +
    the frozen TaskMonitor verdict on the branch trajectory (short-horizon monitor_score)."""
    restore_planar(env, snap)
    traj: list[dict[str, Any]] = []
    two_finger = arm_body = 0
    ft_prog = body_prog = 0.0
    prev_dist = snap.disk_to_zone
    a = np.asarray(first_action, dtype=np.float32)
    steps = 0
    for t in range(horizon):
        _nobs, _r, term, trunc, info = env.step(a)
        m = env._planar_metrics
        lg = m.legality
        traj.append({
            "coin_xy": np.array([float(m.disk_pos[0]), float(m.disk_pos[1])]),
            "coin_vel": np.array([float(m.disk_vel[0]), float(m.disk_vel[1])]),
            "dist_to_zone": float(m.disk_to_zone),
            "left_tip_contact": bool(m.left_contact), "right_tip_contact": bool(m.right_contact),
            "arm_body_contact": bool(lg.arm_body_contact) if lg is not None else False,
            "left_tip_dist": float(m.left_tip_dist), "right_tip_dist": float(m.right_tip_dist),
            "in_zone": bool(info.get("in_zone", m.in_zone)),
        })
        sig = objective.step_signals(
            prev_dist=prev_dist, dist=float(info["disk_to_zone"]),
            min_tip=min(float(m.left_tip_dist), float(m.right_tip_dist)),
            both_contact=bool(info["both_contact"]), fingertip_contact=bool(info["fingertip_contact"]),
            arm_body_contact=bool(info["arm_body_contact_this_step"]), in_zone=bool(info["in_zone"]))
        two_finger += int(info["both_contact"])
        arm_body += int(info["arm_body_contact_this_step"])
        ft_prog += sig["progress"]
        body_prog += sig["body_progress"]
        steps = t + 1
        prev_dist = float(info["disk_to_zone"])
        if term or trunc:
            break
        a = _greedy(frozen_actor, _nobs)
    verdict = monitor.evaluate(traj)
    return {
        "monitor_pass": float(bool(verdict.monitor_pass)),
        "monitor_score": float(verdict.monitor_score),
        "two_finger_rate": two_finger / steps,
        "arm_body_rate": arm_body / steps,
        "ft_progress": ft_prog,
        "body_progress": body_prog,
        "delivered": float(traj[-1]["in_zone"]) if traj else 0.0,
        "branch_len": float(steps),
    }


def _objective_return(measure: dict[str, float]) -> float:
    """Scalar objective used only to rank the best_sampled candidate: fingertip progress + delivery signal."""
    return measure["ft_progress"] + measure["monitor_score"]


def build_candidates(critics: dict[str, Any], q_total: Any, obs: np.ndarray, z: np.ndarray,
                     base_a: np.ndarray, lo: np.ndarray, hi: np.ndarray, cfg: ProbeConfig,
                     rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Construct the five candidate first-actions at one probe state (best_sampled is filled by the caller)."""
    grads = {c: action_gradient(critics[c], obs, base_a, z) for c in COMPONENTS}
    g_scalar = action_gradient(q_total, obs, base_a, z)
    n_scalar = np.linalg.norm(g_scalar)
    scalar_dir = g_scalar / n_scalar if n_scalar > 1e-9 else g_scalar
    proj_dir, _info = projected_gradient(grads, normalize=cfg.normalize_projection)
    return {
        "dagger": base_a.copy(),
        "scalar": np.clip(base_a + cfg.probe_eta * scalar_dir, lo, hi).astype(np.float32),
        "projected": np.clip(base_a + cfg.probe_eta * proj_dir, lo, hi).astype(np.float32),
        "random": rng.uniform(lo, hi).astype(np.float32),
    }


@dataclass
class ProbeResult:
    per_candidate: dict[str, dict[str, float]]     # mean stats across probe states
    n_states: int
    gate: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"n_states": self.n_states, "per_candidate": self.per_candidate, "gate": self.gate}


def gradient_alignment_probe(env: Any, frozen_actor: Any, critics: dict[str, Any], q_total: Any,
                             probe_states: list[tuple[np.ndarray, np.ndarray, PlanarSnapshot]], monitor: Any,
                             cfg: ProbeConfig, *, objective: SearchObjective | None = None,
                             log=print) -> ProbeResult:
    """Run every candidate first-action from every fixed probe state; aggregate branch stats per candidate.

    # Preconditions: ``probe_states`` = list of (obs, z, snapshot) at CONTACT/PUSH states; critics keyed by
      ``COMPONENTS``; ``q_total`` a scalar QCritic.
    # Postconditions: mean per-candidate branch stats over the probe states (+ empty gate, filled by
      :func:`evaluate_gate`); ``env`` left at some terminal branch state (caller owns it)."""
    objective = objective or SearchObjective()
    rng = np.random.default_rng(cfg.seed)
    lo, hi = env._ctrl_lo.astype(np.float32), env._ctrl_hi.astype(np.float32)
    acc: dict[str, list[dict[str, float]]] = {c: [] for c in _CANDIDATES}
    for i, (obs, z, snap) in enumerate(probe_states):
        base_a = _greedy(frozen_actor, obs)
        cands = build_candidates(critics, q_total, obs, z, base_a, lo, hi, cfg, rng)
        # best local sampled perturbation: K samples, keep the one with best measured objective return
        best_m, best_obj = None, -np.inf
        for _ in range(cfg.n_best_sampled):
            a_s = np.clip(base_a + rng.normal(0.0, cfg.best_sampled_sigma, size=base_a.shape), lo, hi).astype(np.float32)
            m_s = _branch_measure(env, frozen_actor, snap, a_s, monitor=monitor, objective=objective, horizon=cfg.probe_horizon)
            if _objective_return(m_s) > best_obj:
                best_obj, best_m = _objective_return(m_s), m_s
        for name, act in cands.items():
            acc[name].append(_branch_measure(env, frozen_actor, snap, act, monitor=monitor,
                                              objective=objective, horizon=cfg.probe_horizon))
        acc["best_sampled"].append(best_m)
        if (i + 1) % max(1, len(probe_states) // 5) == 0:
            log(f"  [probe] state {i + 1}/{len(probe_states)}")
    per_candidate = {name: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} for name, rows in acc.items()}
    return ProbeResult(per_candidate=per_candidate, n_states=len(probe_states))


def evaluate_gate(probe: ProbeResult, *, margin: float = 1e-4) -> dict[str, Any]:
    """VECTOR_PROJECTED_PROMISING iff the projected candidate preserves/improves contact ∧ progress ∧
    monitor_score vs the DAgger baseline AND beats the scalar candidate on those three."""
    pc = probe.per_candidate
    d, s, p = pc["dagger"], pc["scalar"], pc["projected"]
    preserves = {
        "two_finger_rate": p["two_finger_rate"] >= d["two_finger_rate"] - margin,
        "ft_progress": p["ft_progress"] >= d["ft_progress"] - margin,
        "monitor_score": p["monitor_score"] >= d["monitor_score"] - margin,
    }
    beats_scalar = {
        "two_finger_rate": p["two_finger_rate"] >= s["two_finger_rate"] - margin,
        "ft_progress": p["ft_progress"] >= s["ft_progress"] - margin,
        "monitor_score": p["monitor_score"] >= s["monitor_score"] - margin,
    }
    promising = bool(all(preserves.values()) and all(beats_scalar.values()))
    gate = {
        "VECTOR_PROJECTED_PROMISING": promising,
        "preserves_baseline": preserves,
        "beats_scalar": beats_scalar,
        "projected_vs_dagger": {k: round(p[k] - d[k], 5) for k in ("two_finger_rate", "ft_progress", "monitor_score")},
        "projected_vs_scalar": {k: round(p[k] - s[k], 5) for k in ("two_finger_rate", "ft_progress", "monitor_score")},
    }
    probe.gate = gate
    return gate
