"""Teacher-torque-span diagnostic — WHY does the bounded residual saturate ~6 mm short of the 30 mm corridor?

The R9 authority audit showed no residual-authority family (A0 per-joint / A1 larger α / A2 structured) reaches the corridor from
the healthy R2 frontier; all saturate ~36 mm, cleanly. This module asks, WITHOUT learning, which piece is missing, along two axes:

  Part A (bound vs scaffold) — the authority *ceiling*: re-run the audit CEM on the A0 per-joint basis at growing α up to full
    per-step override (α = 2.0 ⇒ the residual can set any admissible action, the frozen clone contributes nothing to the action).
    If full override still misses 30 mm from the frontier, the wall is the SCAFFOLD (the clone's contact-decaying state
    trajectory), not the residual's expressiveness; if it reaches, the wall is the residual BOUND.

  Part B (the span projection) — decompose the DELIVERING teacher's per-step action against the clone's at the same states, in the
    slew-normalised action space both controllers emit (Δτ = a·slew). At each KINETIC step of the teacher's delivering rollout
    from the frozen KINETIC entry, `d = a_teacher − a_clone(counterfactual)` is split into (i) magnitude vs the residual bound α,
    and (ii) the component OUTSIDE the A2 structured basis span (`a2_basis_matrix`). This names the missing correction: too large
    for the bound (magnitude), outside the basis (direction), or within reach (a learnability gap).

Learning-free, read-only: no policy trained, no state edited, no teacher in any deployed loop — the teacher rollout and the clone
query are pure measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, _schedule_increment
from hymeko_rl.coin_delivery.theta_option.kinetic_authority import a2_basis_matrix
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor, KineticClone, NormStats
from hymeko_rl.coin_delivery.theta_option.kinetic_contract import kinetic_observe
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout

ALPHA0 = 0.15                                    # the R2/R3-B residual bound (the audit control)


def project_onto_span(d: np.ndarray, basis: np.ndarray) -> "tuple[np.ndarray, float, float]":
    """Least-squares projection of vector ``d`` onto the column span of ``basis``. # Preconditions: ``basis`` is (n, k),
    ``d`` length n. # Postconditions: returns (coeffs length k, ‖d − basis·coeffs‖ the span-orthogonal residual, ‖d‖); the
    residual is 0 iff ``d`` lies in the span (e.g. a full-rank 4×5 basis spans R⁴ ⇒ every direction is expressible)."""
    d = np.asarray(d, np.float64).ravel()
    coeffs, _res, _rank, _sv = np.linalg.lstsq(np.asarray(basis, np.float64), d, rcond=None)
    ortho = float(np.linalg.norm(d - basis @ coeffs))
    return coeffs, ortho, float(np.linalg.norm(d))


@dataclass
class TeacherStep:
    """One KINETIC step of the delivering teacher's rollout with the clone's counterfactual action at the same state."""
    t: int
    phase: str                                   # push | brake | release
    dtz_mm: float
    v_par: float
    fn_l: float
    fn_r: float
    a_teacher: list
    a_clone: list
    d_inf: float                                 # ‖a_teacher − a_clone‖_∞ (per-joint max correction the clone misses)
    exceeds_alpha0: bool                         # correction beyond the α = 0.15 residual bound
    exceeds_2alpha0: bool                        # correction beyond even the A1 α = 0.30 bound
    a2_ortho: float                              # ‖d − proj_{A2}(d)‖ — the part no A2 coefficient can supply
    d_norm: float


class TeacherThetaController:
    """Roll the proven θ-primitive teacher through the frozen deploy kernel (`velocity_rollout`) while logging, per KINETIC step,
    the teacher's slew-normalised action and the frozen clone's COUNTERFACTUAL action at the same state (queried, never executed).
    Reproduces `rollout_primitive(entry, θ)` bit-for-bit (same kernel, start-fixed e_par, per-step coin). # Invariants: the clone
    query does not affect the teacher's trajectory; no teacher signal enters the clone's input."""

    def __init__(self, snap: Any, theta: Any, clone: CloneActor) -> None:
        self.snap = snap
        self.theta = np.asarray(theta, np.float64)
        self.clone = clone
        self.slew = float(snap.stack.tau_rate * snap.stack.control_dt)
        self.release_boundary_step = int(round(float(self.theta[4])))
        self._ramp = float(self.theta[3])
        self._e_par: "np.ndarray | None" = None
        self._obs_hist: list = []
        self.steps: list[TeacherStep] = []

    def reset(self) -> None:
        self.clone.reset()
        self._e_par = None
        self._obs_hist = []
        self.steps = []

    def before_release(self, t: int) -> bool:
        return t <= self.release_boundary_step

    def _phase(self, t: int) -> str:
        return "push" if t <= self._ramp else ("brake" if t <= self.release_boundary_step else "release")

    def dtau_for_step(self, rl: Any, t: int, prev_tau: np.ndarray) -> np.ndarray:
        if self._e_par is None:                                          # start-fixed e_par (matches rollout_primitive)
            self._e_par = np.asarray(rl.inner.direction_to_zone()[0], np.float64)
        dtau = np.asarray(_schedule_increment(rl, self.theta, t, self._e_par, self.slew, _coin_xy(rl)), np.float64)
        self._record(rl, t, dtau)
        return dtau

    def _record(self, rl: Any, t: int, dtau_teacher: np.ndarray) -> None:
        obs, hframe = kinetic_observe(rl, self._obs_hist)               # same extractor + history the clone deploys with
        self._obs_hist.append(hframe)
        a_teacher = np.clip(dtau_teacher / self.slew, -1.0, 1.0)        # teacher action in the clone's [-1,1] slew space
        a_clone = np.clip(np.asarray(self.clone.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        d = a_teacher - a_clone
        e_par_now = np.asarray(rl.inner.direction_to_zone()[0], np.float64)   # fresh e_par — how A2 is applied in the audit
        _coef, a2_ortho, d_norm = project_onto_span(d, a2_basis_matrix(rl, e_par_now))
        _u, dtz = rl.inner.direction_to_zone()
        con = _fn_pair(rl)
        d_inf = float(np.max(np.abs(d)))
        self.steps.append(TeacherStep(
            t=int(t), phase=self._phase(t), dtz_mm=round(float(dtz) * 1000, 2),
            v_par=round(float(np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2] @ e_par_now), 5),
            fn_l=round(con[0], 4), fn_r=round(con[1], 4),
            a_teacher=[round(float(x), 5) for x in a_teacher], a_clone=[round(float(x), 5) for x in a_clone],
            d_inf=round(d_inf, 5), exceeds_alpha0=bool(d_inf > ALPHA0), exceeds_2alpha0=bool(d_inf > 2 * ALPHA0),
            a2_ortho=round(a2_ortho, 5), d_norm=round(d_norm, 5)))


def _fn_pair(rl: Any) -> "tuple[float, float]":
    from hymeko_rl.coin_delivery.forward_displacement import primary_fingertip_contacts
    con = primary_fingertip_contacts(rl)
    return (float(con["left"]["fn"]) if con["left"] else 0.0, float(con["right"]["fn"]) if con["right"] else 0.0)


def load_clone(ckpt_path: Any) -> "tuple[KineticClone, NormStats]":
    """Load a frozen K2 clone checkpoint into (model, norm). # Postconditions: eval-mode model; norm from the saved stats."""
    import torch
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = NormStats(np.asarray(ckpt["norm"]["mean"]), np.asarray(ckpt["norm"]["std"]))
    return model, norm


def decompose_teacher_vs_clone(entry_snap: Any, theta: Any, model: KineticClone, norm: NormStats,
                               *, cfg: Any = DELIVERY_CFG) -> "tuple[dict, list[TeacherStep]]":
    """Roll the delivering teacher θ from the frozen KINETIC entry and return (rollout metrics, per-step teacher-vs-clone
    decomposition). The teacher DELIVERS (its own trajectory); the decomposition reports, at each step, how far the clone's
    action is from the teacher's and whether that gap is bound-limited or A2-span-orthogonal. Deterministic. No learning."""
    ctrl = TeacherThetaController(entry_snap, theta, CloneActor(model, norm))
    m = velocity_rollout(entry_snap, ctrl, cfg)
    return m, ctrl.steps


def summarize_decomposition(steps: "list[TeacherStep]") -> dict:
    """Aggregate the per-step decomposition into the diagnostic verdict inputs over the TRANSPORT steps (push + brake, where the
    coin-following happens; release relaxes grip). # Postconditions: fractions in [0,1]; max/mean gaps in slew-normalised units."""
    tr = [s for s in steps if s.phase in ("push", "brake")]
    if not tr:
        return {"transport_steps": 0}
    d_infs = np.array([s.d_inf for s in tr])
    a2_orthos = np.array([s.a2_ortho for s in tr])
    d_norms = np.array([s.d_norm for s in tr]) + 1e-12
    return {"transport_steps": len(tr),
            "mean_d_inf": round(float(d_infs.mean()), 5), "max_d_inf": round(float(d_infs.max()), 5),
            "frac_exceeds_alpha0": round(float(np.mean([s.exceeds_alpha0 for s in tr])), 3),
            "frac_exceeds_2alpha0": round(float(np.mean([s.exceeds_2alpha0 for s in tr])), 3),
            "mean_a2_ortho": round(float(a2_orthos.mean()), 5), "max_a2_ortho": round(float(a2_orthos.max()), 5),
            "mean_a2_ortho_frac": round(float(np.mean(a2_orthos / d_norms)), 3)}
