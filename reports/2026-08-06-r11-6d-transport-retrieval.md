# R11.6D Phase 4 — Transportability-aware retrieval (signal positive, selection uncalibrated)

**Date:** 2026-08-06
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA:** `ad9dc70f` (frozen composition baseline `8f2d796f` preserved; test sealed)
**Verdict:** `R11_6D_TRANSPORTABILITY_SIGNAL_POSITIVE_SELECTION_UNCALIBRATED` — an honest negative for the *simple* ranker.

---

## What was built (to the approved Phase-4 spec)

- **Full transfer matrix** — 44 train θ × (44 train + 7 dev) handoffs = **2244 cells, 612 strict-K6** (test sealed). Each
  cell carries the rich transport metrics (projected transport, undershoot/overshoot, contact retention, entry speed,
  lateral drift, trajectory hash).
- **Physical ranker** (not a free-capacity regressor): `S = −α·undershoot − β·overshoot − γ·angle_gap + η·k6_rate +
  ρ·contact_rate`, undershoot/overshoot separate. θ-signatures (typical transport, angle range, K6/contact rate) from
  each θ's matrix column. **Top-1 complete θ, no blending.**
- **Train-only nested calibration** — leave-one-scenario-out (own θ + own handoff removed), weights from a pre-registered
  grid by (top-1 K6, −regret). Selected `α=2, β=1, γ=50, η=30, ρ=20` (undershoot 2× overshoot, as the audit predicted).

## Result

| metric | transport-retrieval | descriptor-nearest baseline |
|---|---|---|
| **dev top-1 K6** | **3/7 (0.429)** | **4/7 (0.571)** |
| c3 far-angle strict-K6 | **0/3** | 1/3 (`r7_a+45`) |
| deployment train-like | **0.682** | 1.0 (R11.6C self-retrieval) |
| dev mean AUROC (ranking) | **0.743** | — |

Per dev scenario: transport-retrieval keeps 3 (often *tighter* — c0_1 7.0mm vs 12.4mm, c1_+0.03_-0.02 9.0mm vs 18.4mm),
but **regresses `r7_a+45`** (picks `c1_+0.01_+0.02` → 4.09mm, so-close but no K6-dwell, instead of `r6_a+15` → 12.8mm K6)
and **recovers neither r9** case. It gains no new scenario.

**Gate: not met** — dev 3/7 < 6/7, c3 0/3 < 2/3, does not beat the baseline, deployment train-like < 44/44, r7_a+45
regressed.

## Diagnosis (why the simple ranker fails — and it is informative)

**The transportability signal is real** (dev AUROC 0.743 — the score ranks deliverable θ above non-deliverable
better than chance). **But the per-θ *scalar* transport signature is too coarse for calibrated top-1**, because
**transport is (θ × handoff)-dependent**:

- The delivering θ for the r9 targets (`bank_c0_3`, `bank_c1_+0.01_+0.02`) transport ~70mm from *their own* handoffs but
  ~100mm from the *r9* handoff (the same torque schedule from a different arm/coin state travels farther). Their per-θ
  median transport signature is therefore ~75mm.
- The score matches `d̂_i` (median ~75mm) to `d_req` (~100mm) and penalises these θ as **undershoot** — deprioritising
  exactly the θ that actually deliver from r9. The scalar signature discards the handoff-conditioning that determines
  delivery.

So the direction is right (signal present) but a **handoff-independent** signature cannot select correctly. This is the
pre-registered `SELECTION_UNCALIBRATED` outcome, and per the plan's own rule (MLP/richer model only once train-only CV
shows a nonlinear gap) it is the *evidence* that a **handoff-conditioned transport predictor** is needed — predict
`transport(θ_i, s)` from the θ-signature × query-handoff interaction, not a per-θ median.

## Honest bottom line

- **The simple physical ranker does not beat descriptor-nearest** (3/7 vs 4/7 dev) and does not recover the r9 far-angle
  cases. Descriptor-nearest (4/7) remains the better deploy retrieval; R11.6C's frozen policy is unchanged.
- **Transportability is not refuted** — AUROC 0.743 says the score carries real information; the r9 delivering θ *exist*
  (feasibility probe) and the score ranks them above chance but not top-1.
- **The identified next lever** (a Phase-4.1 decision, HALT for review): a **handoff-conditioned transport predictor** —
  still interpretable, still train-only, still top-1, no blending, but predicting each θ's transport *from the query
  handoff* rather than a per-θ scalar. This directly targets the diagnosed coarseness.

The **dev panel is now spent as a development signal** (it calibrated + evaluated the ranker); the final generalisation
claim remains on the sealed test, which is untouched. The gate did not clear, so **no test unseal is warranted**.

---

## Files / provenance

| File | Δ |
|---|---|
| `hymeko_rl/coin_delivery/transport_retrieval.py` | matrix cell rollout, signatures, physical score, LOSO helpers (new) |
| `hymeko_rl/experiments/r11_6d_transport_matrix.py` | transfer-matrix build (new) |
| `hymeko_rl/experiments/r11_6d_transport_retrieval.py` | calibration + eval + gate (new) |
| `hymeko_rl/tests/test_r11_6d_transport.py` | 7 tests (score, signatures, top-1, LOSO, AUROC) |
| `reports/2026-08-06-r11-6d-matrix/matrix.json` | 2244-cell transfer matrix (new) |
| `reports/2026-08-06-r11-6d-retrieval/retrieval.json` | calibration + eval + per-scenario (new) |

- **CORE.YAML:** none. **Deps:** none. Reach/capture/descriptor/rollout/frozen-table read-only; no canonicalizer, no
  blending, no runtime oracle/CEM/teacher.
- **Env:** framework `.venv`, torch 2.12.0, macOS, CPU; matrix 8-way fanout ~9 min (2244 rollouts + 51 reconstructs),
  calibration+eval < 1 min pure-compute; RSS ≪ 16 GB. Deterministic. 7 tests, ruff/radon clean (all A/B).
- `8f2d796f` preserved; sealed test untouched.

## Boundary

Phase 4 (per-θ scalar transportability ranker) is a clean negative vs baseline. Next (gated): a handoff-conditioned
transport predictor, or accept descriptor-nearest and revisit the r9 far-angle differently. No test unseal.
