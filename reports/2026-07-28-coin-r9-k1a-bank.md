# K1-A — the first KINETIC feedback bank (32 accepted first-action labels)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · NO BC / feedback-clone started**

## Summary

Green-lit by the K1 cost/diversity gate (`K1_RELABEL_COST_AND_DIVERSITY_GATE_PASS`). K1-A builds the first feedback bank: 32
accepted first-action labels over legal perturbed-control branches of the frozen KINETIC entry, **≤ 48 attempts**, curated **by
the teacher-replan result** (not a manual v_par threshold). Same contract as K0/K1: warm-start = entry-delivering θ; search =
box-wide legal CEM; label = first executed action only; actor feature = the canonical 41-D observation only.

**Verdict: `K1A_BANK_READY`.** 32/32 feedback labels reached in **37 attempts** (under the 48 cap — no coverage shortfall, no
admissibility dilution); 5 terminal-failure states preserved separately; 0 inadmissible. The bank is diverse, covers the
light-contact / near-slip / asymmetry regimes, is not dominated by teacher-near states, and is deterministic (bit-replay).
**Stopped here — no BC / feedback-clone started**, as instructed.

## The five review questions, answered

| question | answer | evidence |
|---|---|---|
| 32 genuinely distinct feedback-states? | **yes** | near-duplicate pairs 2 / 496; min pairwise 0.117, mean pairwise 2.889 (standardised) |
| action-diversity preserved? | **yes** | first-action pairwise diversity **0.274** (scale 0.205) ≫ collapse floor 0.010; per-dim std `[0.132, 0.066, 0.123, 0.110]` |
| dominated by a few teacher-near states? | **no** | category mix easy 9 / medium 19 / edge 4; only 2 near-dup pairs |
| terminal-failure mass? | **5** | all edge, all `no_progress` — heavy-clamp (fn≈5, v_par≈0) + one extreme-asymmetry (imbal 1.15); preserved separately |
| cover light-contact / near-slip / asymmetry? | **yes** | in the 32 labels: light-contact (fn<2 N) **4** · near-slip (slip>0.25) **7** · asymmetry (imbal>0.4) **5** |

## The bank (`reports/2026-07-28-coin-r9-k1a-bank/`)

- **`feedback_labels.json`** — 32 records, each `{41-D observation, first_action (raw + slew-normalised), descriptor,
  provenance (gen θ / k / noise / category), delivers_k6, min_dtz_mm, replan θ}`. The observation carries the faithful short
  causal history (frames *before* the state, deploy-time convention); it carries **no** teacher/future/K6 information.
- **`terminal_failure_states.json`** — 5 records, each `{descriptor, oracle_min_dtz_mm, oracle_delivers_k6=false,
  failure_reason=no_progress, first-action provenance (not a training label), gen θ}`. Retained as negative information (a
  stiction-risk guard / avoidance classifier / DAgger early-stop / clone-drift measurement) — the near-zero "do nothing"
  actions are deliberately **kept out of the feedback labels** so BC never learns to stop the coin.
- **`manifest.json`** — the metrics below + `bank_hash` + the bit-replay check.

**State spread (32 feedback labels):** v_par [−0.028, 0.277] (std 0.058) · slip [0.062, 0.476] (std 0.092) · fn_min
[0.50, 3.58] N (std 0.769, light→firm) · imbalance [0.020, 0.473] (std 0.112) · dtz [54.7, 70.4] mm.
**Label quality:** 22 / 32 labels are from delivering replans (K6), 10 from progressing (min_dtz < 40 mm, not yet delivering).
**First-action mean** `[−0.261, −0.131, 0.162, −0.261]`, range `[0.563, 0.263, 0.692, 0.383]`.

## Cost / determinism

| metric | value | note |
|---|---|---|
| cost / label (median · p90 · max) | **4.77 · 5.25 · 5.5 s** | consistent with the K1 audit (~4.7 s) |
| attempts used | **37 / 48** | 32 feedback + 5 terminal |
| total wall | 214.8 s | acquire + 37 relabels + 2 bit-replay |
| peak RSS | **0.27 GB** | < 1 GB (cap 16) |
| deterministic repeat max\|Δ first-action\| | **0.0, 0.0** | 2 labels bit-replayed in the final bank |
| bank hash | `bc36e0521982bd36` | content hash of (obs, first-action) — reproducibility anchor |

## Files touched (all new / additive; K0 frozen module untouched)

| file | role |
|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_bank.py` | +`sample_specs`, `labeled_state`, `_advance_with_history` (K1-A primitives) |
| `hymeko_rl/experiments/coin_kinetic_k1a_bank.py` | the K1-A bank builder (replan-result curation, hash, bit-replay) |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | +2 tests (sampler determinism/strata; labeled-state 41-D obs / no-leak) |
| `reports/2026-07-28-coin-r9-k1a-bank/{feedback_labels,terminal_failure_states,manifest}.json` | the bank |

Additions to `kinetic_bank.py` are additive (the K1 cost-audit's `generate_neighbourhood` path is unchanged); the K0 contract
module `kinetic_contract.py` is not modified.

## Tests / static analysis

- `pytest test_coin_kinetic_contract.py` — **11 passed** (9 prior + `test_sample_specs_deterministic_and_covers_strata`,
  `test_labeled_state_carries_41d_obs_no_leak`).
- `ruff check` clean; `radon cc -a` on `kinetic_bank.py` = **A (1.93)**; no new suppressions; no §6.5 anti-patterns.

## Provenance

Git `c12d4da6` (`recovery/coin-r9-causal-residual-delivery`; K1-A files uncommitted at run time). Python 3.11.15 / mujoco
3.10.0 / numpy 2.4.6 / macOS-26.5.2-arm64, quiet host during the timed run. Seeds: cradle 14250; sampler/generation seed
20260728 (per-attempt sub-seeds); CEM seed 20260727 (frozen). Entry hash `ce343c478d2a0cb7`. Deterministic (physics + labels
reproduce; wall-time is a measurement).

## Recommendation (short stop — awaiting your green)

`K1A_BANK_READY`: 32 distinct, diverse, regime-covering feedback labels + 5 preserved terminal-failure states, at ~4.8 s/label,
deterministic. The bank carries real state-dependent feedback (diversity 0.274, not a teacher-θ copy) and is BC-ready. **No BC /
feedback-clone started.** The next step would be the K2 feedback-clone (GRU over the causal history) with the closed-loop
`LOCAL_KINETIC_FEEDBACK_SKILL_PASS` gate — awaiting your go.
