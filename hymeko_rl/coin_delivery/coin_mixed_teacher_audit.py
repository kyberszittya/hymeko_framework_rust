"""Mixed-teacher-averaging audit for the FEEDBACK_CHUNK_WARMSTART_V2 chunk actor (item 8 of CHUNK_SUPERVISED_M1_FEEDBACK_V1).

The M=1 diagnostic showed per-step replanning does NOT recover contact (it is worse than M=2), which rules out the
execution horizon as the limiter and points at the *labels*: a supervised regressor trained on two dissimilar teachers
(exact pi_0 fallback vs planner improvement) whose targets disagree in nearby states will learn the **average** first
action — worse than either teacher. This module quantifies that hypothesis without any TD3/eval rollout:

  1. teacher first-action distance  ‖pi0_first − planner_first‖  (how far apart the two teachers are per state)
  2. local teacher-mode disagreement (k-NN fraction of neighbors with the OTHER provenance)
  3. conditional variance of the first-action label within each state's k-NN (label inconsistency in nearby states)
  4. first-action prediction error stratified by admissibility-boundary state (borderline vs interior)
  5. whether the learned first action lies BETWEEN the two teachers (interpolation) and is pulled off its own label

All inputs are action-unit arrays; this is pure/deterministic (no env). Preconditions asserted below.
"""
from __future__ import annotations

import numpy as np

BETWEEN_T = (0.20, 0.80)         # learned's fractional position on the pi0->planner segment counts as "between"
BETWEEN_PERP = 0.35              # perpendicular deviation ≤ this * teacher-gap to count as on the segment
WORSE_FRAC = 0.40                # learned farther than this * teacher-gap from its OWN label => pulled off target
GAP_EPS = 1e-3                   # teachers considered "agreeing" below this gap (excluded from between-analysis)


def _zscore(X):
    mu = X.mean(0); sd = X.std(0); sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def _knn_indices(Xz, k):
    """Row-wise indices of the k nearest neighbors (excluding self) under Euclidean distance on z-scored states."""
    d2 = ((Xz[:, None, :] - Xz[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.argsort(d2, axis=1)[:, :k]


def _segment_position(learned, a, b):
    """Fractional position t of ``learned`` projected onto segment a->b, and perpendicular distance; per row."""
    u = b - a; d = np.linalg.norm(u, axis=1); dd = np.where(d < GAP_EPS, 1.0, d)
    t = ((learned - a) * u).sum(1) / (dd ** 2)
    perp = np.linalg.norm((learned - a) - t[:, None] * u, axis=1)
    return t, perp, d


def mixed_teacher_metrics(X, pi0_first, planner_first, label_first, prov, flags, learned_first, *, k=8):
    """Quantify mixed-teacher averaging.

    Preconditions: all first-action arrays are (N, ACT_DIM) in the same action units; ``prov`` entries ∈
    {"planner","pi0_fallback"}; N ≥ k+1. Returns a metrics dict + a verdict token.
    Postcondition: verdict ∈ {MIXED_TEACHER_AVERAGING_CONFIRMED, MIXED_TEACHER_AVERAGING_NOT_CONFIRMED}.
    """
    N = len(X)
    assert pi0_first.shape == planner_first.shape == label_first.shape == learned_first.shape, "first-action shape mismatch"
    assert N == len(prov) == len(flags) and N > k, "N must exceed k and match provenance/flags length"
    assert set(prov) <= {"planner", "pi0_fallback"}, "unexpected provenance"

    is_planner = np.array([p == "planner" for p in prov])
    gap = np.linalg.norm(pi0_first - planner_first, axis=1)                 # (1) teacher distance
    err_first = np.linalg.norm(learned_first - label_first, axis=1)         # actor first-action error vs its own label

    Xz = _zscore(X); nn = _knn_indices(Xz, k)                              # (2) local teacher-mode disagreement
    nn_diff_mode = np.array([np.mean(is_planner[nn[i]] != is_planner[i]) for i in range(N)])
    # (3) conditional variance of the label first-action within each k-NN (mean over action components)
    cond_var = np.array([label_first[nn[i]].var(0).mean() for i in range(N)])

    # (4) admissibility-boundary strata from label_chunk flags (safe / improving booleans)
    safe = np.array([bool(f.get("safe", False)) for f in flags]); improv = np.array([bool(f.get("improving", False)) for f in flags])
    boundary = safe ^ improv                                               # exactly one condition met = borderline
    strata = {"interior_both": float(err_first[safe & improv].mean()) if (safe & improv).any() else None,
              "boundary_xor": float(err_first[boundary].mean()) if boundary.any() else None,
              "interior_neither": float(err_first[~safe & ~improv].mean()) if (~safe & ~improv).any() else None}

    # (5) learned first-action between the two teachers, and pulled off its label
    both = gap > GAP_EPS
    t, perp, d = _segment_position(learned_first, pi0_first, planner_first)
    between = both & (t >= BETWEEN_T[0]) & (t <= BETWEEN_T[1]) & (perp <= BETWEEN_PERP * np.where(d < GAP_EPS, 1.0, d))
    off_label = err_first > WORSE_FRAC * np.where(d < GAP_EPS, 1.0, d)
    between_and_off = between & off_label
    # on planner-labeled states, t<0.5 means learned is pulled toward pi_0 (i.e. AWAY from the improving teacher)
    pl_pull_to_pi0 = float(np.mean(t[is_planner & both] < 0.5)) if (is_planner & both).any() else None
    pl_mean_t = float(np.mean(t[is_planner & both])) if (is_planner & both).any() else None

    # crux: does first-action error concentrate where teacher modes are locally mixed?
    corr = float(np.corrcoef(nn_diff_mode, err_first)[0, 1]) if np.std(nn_diff_mode) > 0 else 0.0

    m = {"n": int(N), "n_planner": int(is_planner.sum()), "n_pi0_fallback": int((~is_planner).sum()),
         "teacher_gap": {"mean": float(gap.mean()), "median": float(np.median(gap)), "p90": float(np.percentile(gap, 90)),
                         "mean_planner_states": float(gap[is_planner].mean()) if is_planner.any() else None},
         "first_action_error": {"mean": float(err_first.mean()), "median": float(np.median(err_first))},
         "nn_mode_disagreement_mean": float(nn_diff_mode.mean()),
         "conditional_variance_first_label_mean": float(cond_var.mean()),
         "error_by_admissibility_boundary": strata,
         "error_vs_mode_disagreement_corr": corr,
         "between_teachers_frac": float(between.mean()),
         "between_and_off_label_frac": float(between_and_off.mean()),
         "planner_states_pulled_toward_pi0_frac": pl_pull_to_pi0, "planner_states_mean_segment_pos": pl_mean_t}

    # Structural crux: teachers far apart, locally mixed modes, learned interpolates between them, and on the improving
    # (planner) states the learned action falls materially short of the planner target (on-target position t=1.0; a full
    # average sits at 0.5). The error/disagreement correlation is reported as corroborating evidence but is not part of
    # the gate (a uniformly-averaging actor has near-constant error => undefined correlation).
    confirmed = (m["teacher_gap"]["median"] > 0.5 and m["nn_mode_disagreement_mean"] >= 0.25
                 and m["between_teachers_frac"] >= 0.30
                 and (pl_mean_t is not None and pl_mean_t < 0.75))
    m["verdict"] = "MIXED_TEACHER_AVERAGING_CONFIRMED" if confirmed else "MIXED_TEACHER_AVERAGING_NOT_CONFIRMED"
    return m
