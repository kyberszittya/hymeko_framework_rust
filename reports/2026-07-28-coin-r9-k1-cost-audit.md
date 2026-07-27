# K1 cost-audit — 16 representative relabels over the KINETIC-entry neighbourhood

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · NO bank generated, NO BC started**

## Summary

Per the K1 pre-step (`docs/plans/2026-07-27-…/plan`): before sizing any feedback bank, measure the receding-horizon relabel on
a small, deliberately-diverse neighbourhood of the frozen KINETIC entry. 16 legal perturbed-control branches (4 easy / 8 medium
/ 4 edge), same contract as K0 (warm-start = squeeze≈0 entry-delivering θ; search = box-wide legal CEM; label = first executed
action only; actor feature = 41-D causal observation only). Teacher used **offline** to label; nothing deployed.

**Gate: `K1_RELABEL_COST_AND_DIVERSITY_GATE_PASS`** — all three pre-registered brakes clear. Cost is low and uniform, replanning is stable, and the
first-action labels **genuinely vary with the state** (the bank carries feedback information, not a constant-θ copy).
**Stopped here for review — no BC started**, as instructed.

## Results (`coin_kinetic_k1_cost_audit.py`; physics deterministic; total ≈ 118 s; peak RSS 0.25 GB)

| metric | value | brake | status |
|---|---|---|---|
| wall / label (median · p90 · max) | **4.65 · 4.99 · 5.16 s** | p90 ≤ 2× baseline | ✅ (p90 4.99 vs 2×4.55 = 9.10) |
| calibrated baseline (relabel the entry) | 4.55 s | — | p90/baseline = **1.10×** |
| successful replans | **13 / 16** | ≥ half progress | ✅ |
| K6-delivering replans | **8 / 16** | — | (all 4 easy + 4 medium) |
| rejected (inadmissible) | 0 / 16 | — | generator stayed legal |
| first-action pairwise diversity | **0.312** (scale 0.205) | not ≈ constant | ✅ (collapse threshold 0.010) |
| deterministic repeat max\|Δ first-action\| | **0.0, 0.0** | bit-replay | ✅ |
| peak RSS | 0.25 GB | < 1 GB (cap 16) | ✅ |

**Cost reconciliation (the >2× wall-estimate rule).** The K0 report carried a conservative ~17 s/label estimate; the measured
box-wide warm-started relabel is **~4.7 s/label** (the explicit pop 32 × iters 6 budget is cheaper than K0's full-`DELIVERY_CFG`
search). Reconciled downward, not up — no halt. Implied bank costs: 32 labels ≈ 2.5 min; 128 ≈ 10 min; 512 ≈ 40 min — all
checkpoint-affordable. (Wall-time is a measurement, not deterministic; the physics — 13/16 progress, 8/16 deliver, diversity
0.312 — reproduces bit-for-bit run-to-run.)

**Termination reasons:** delivered 8 · progress_no_delivery 5 · **no_progress 3** · inadmissible 0.

**State spread (measured in the 41-D policy-input space):** v_par [−0.045, 0.275] (std 0.101) · slip [0.073, 0.434] (std
0.078) · fn_min [1.18, 4.92] (std 0.963, near-loss → heavy-clamp) · imbalance [0.13, 1.17] (std 0.226) · dtz [59.7, 70.3] mm.

**First-action label variation:** mean `[−0.246, −0.121, 0.163, −0.243]`, std `[0.170, 0.077, 0.132, 0.095]`, range
`[0.705, 0.322, 0.537, 0.329]`. The label swings from a large forward+grip correction on a near-slip-out state
(`slipout_k6`: `[−0.559, −0.275, 0.121, −0.425]`) to near-zero on the stalled/clamped edge states (`clamp_k8`/`stall_k9`:
`≈[−0.02, −0.03, −0.08, −0.10]`). Different states demand different first actions — the feedback signal is real.

## The 3 `no_progress` cases are the stiction wall, not a harness bug (a K1 design finding)

All three failures are the **stalled / clamped edge states** — `clamp_k8` (v_par −0.018, fn_min 4.92), `asym_hi_k7`
(v_par −0.045), `stall_k9` (v_par −0.024, fn_min 4.92). Once the coin has stopped under a firm grip, even the teacher cannot
restart it (min_dtz stays ~62–65 mm), and it correctly returns a near-zero first action. This is exactly the arc's
**"never let the coin stop"** barrier (a stopped gripped coin is in the R1 stiction regime), reproduced on purpose by the edge
perturbations.

**Implication for K1-A (curation by replan result, NOT a manual v_par threshold).** These states are **valuable negative
information**, not noise, and must be *preserved separately* rather than deleted as "rejected". K1-A partitions each admissible
snapshot purely by the teacher-replan outcome:
- *admissible ∧ successful causal replan ∧ valid first action* → a **feedback label** (in `feedback_labels`);
- *admissible ∧ no progressing/delivering continuation* → a **terminal-failure record** (in `terminal_failure_states`, with the
  oracle outcome + failure reason logged).

The near-zero "do nothing" first actions must **never** enter the actor's action-loss (that would teach the coin to stop). The
terminal-failure set is retained for later use — a stiction-risk guard, an avoidance classifier, DAgger early-stop, and
measuring how far a clone drifts into the unrecoverable region. No `v_par`-threshold is frozen; the replan's terminal result is
used for **dataset curation only** and never enters the actor's 41-D input. The 13 progressing states (8 delivering) are the
positive bank material.

## Files touched (all new; K0 frozen module untouched)

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_bank.py` | K1 neighbourhood generator (perturbed-control branches, admissibility-gated) |
| `hymeko_rl/experiments/coin_kinetic_k1_cost_audit.py` | the cost-audit driver + pre-registered brakes |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | +1 test (generator determinism + diversity + 4/8/4 strata) |
| `reports/2026-07-27-…-k0/k1_cost_audit.json` | the audit artifact (md5 `0dd25296d08cc52e`) |

The frozen K0 module `kinetic_contract.py` (tag `coin-r9-learned-s1-kinetic-k0-positive-control`) is **not modified** — the K1
generator sits on top of it (reuses `roll_until`, `TransportSnapshot`, `kinetic_observe`).

## Tests / static analysis

- `pytest test_coin_kinetic_contract.py` — **9 passed** (8 K0 + `test_kinetic_bank_neighbourhood_deterministic_and_diverse`:
  same seed ⇒ same admissible states + descriptors; ≥12/16 admissible; genuine v_par + fn spread; all 3 strata present).
- `ruff check` clean; `radon cc -a` on `kinetic_bank.py` = **A (1.64)**; no new suppressions (no §6.5 anti-patterns; the
  generator is a pure feed-forward legal-branch controller + dataclass specs).

## Provenance

Git `f0aa1683` (`recovery/coin-r9-causal-residual-delivery`; K1 files uncommitted). Python 3.11.15 / mujoco 3.10.0 / numpy
2.4.6 / macOS-26.5.2-arm64, quiet host (no concurrent load during the timed run). Seeds: cradle 14250; generation seed 20260728
(per-spec sub-seeds); CEM seed 20260727 (frozen). Entry hash `ce343c478d2a0cb7`. Determinism: 2 labels bit-replayed (Δ = 0).

## Recommendation (short stop — awaiting your green)

`K1_RELABEL_COST_AND_DIVERSITY_GATE_PASS`: cost acceptable (~4.7 s/label), replanning stable (13/16 progress, 8/16 deliver),
labels state-dependent (diversity 0.312 ≫ collapse floor). **K1-A (32 accepted feedback labels, ≤ 48 attempts) is justified.**
K1-A curates by teacher-replan result (feedback vs terminal-failure, both preserved), uses no manual `v_par` threshold, keeps
the terminal-failure states as a separate labelled set, and stops with a coverage report if 48 attempts do not yield 32 usable
labels. **No BC started; no bank generated in this audit.**
