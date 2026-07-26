# R4 — closed-loop basin-aware intent correction: STAGE-0 CONTRACT (frozen before any controller code)

**Created-at:** 2026-07-27 04:20 JST · **Branch:** `recovery/coin-r4-closed-loop-intent` (base `60801b09`, a linear
descendant of `e3624d14` that carries the frozen R4 design freeze
`reports/2026-07-27-coin-r4-closed-loop-intent-correction-contract.md`). Branching from `60801b09` instead of the
literal `e3624d14` is deliberate: it preserves that design-freeze commit rather than orphaning it.
**Design freeze this implements:** `reports/2026-07-27-coin-r4-closed-loop-intent-correction-contract.md` (`60801b09`).

**ETA:** ≈4–6 h wall. Basis: Stage-1/2/3 code + tests ≈2 h; regression suite + ruff ≈20 min; C0 ≈2 min compute (one
`build_panel` ≈93 s + 4 rollouts); C1 dev tuning ≈15–30 min (predictor build reuses R3 path ≈4 min + a small dev
hyper-grid × 2 dev cradles × budget rollouts, each rollout ≈0.03 s); C2 one frozen panel ≈3 min; report + plots ≈45 min.
`build_panel` (harness load + 4 snapshot acquisitions ≈93 s) dominates every compute stage; each `rollout_primitive` is
≈0.02–0.03 s and a budget-8 closed-loop candidate adds one authority re-identification (≈0.05–0.1 s) at the correction
point.

This document **freezes the R4 experimental contract before any controller code exists**, per Stage 0 of the R4 task. It
resolves the design freeze into concrete, machine-checkable structure and discipline. C1 (development) selects the
remaining numeric hyperparameters **on development data only** and freezes them into `controller_parameters.json`; C2 runs
**exactly once** on the frozen panel. The machine-readable mirror is
`reports/2026-07-27-coin-r4-closed-loop-intent/contract_audit.json`.

---

## 0. Frozen baseline (measured this session, `smoke_env`)

| condition | result |
|---|---|
| teacher θ → frozen option → K6 (all 4 states) | **4/4** (s1,s3,s4,s7 delivery_success = True) |
| s4 (held-out) teacher peak coin speed | **0.263 m/s** — s4 rides the motion limit (the R3 witness) |
| dev (s1,s3,s7) teacher peak coin speed | ≈0.44–0.45 m/s |
| R3 open-loop learned intent → decode → budget-8 | **2/4** (dev 2/2, held-out 0/2), verdict `PHYSICAL_INTENT_FACTORISATION_ALONE_INSUFFICIENT` |
| R1 (flat), R2 (relational) | 2/4, held-out 0/2 — same ceiling |

The bottleneck across R1/R2/R3 is the **open-loop amortisation of a fixed decision**: a single intent predicted from the
static cradle does not cover held-out. The concrete witness is **s4** — the R3 predictor gives a dev-like strong transport
(peak velocity **0.47**) where the cradle tolerates only a gentle one (**0.263**). *That over-speed is not decidable before
the trajectory, but it is measurable within the first few physical steps.*

## 1. The single authorised axis (everything else frozen)

Replace the open-loop fixed intent with a **closed-loop basin-aware correction over the frozen R3 decoder**, on one
continuous physical trajectory:

```
initial intent (R3 predictor, or a conservative prior — dev-selected)
  → R3 decoder → θ₀ → budget-8 search selects θ_exec for the INITIAL segment
  → execute a short PROBE window of the frozen option on ONE continuous rl
  → MEASURE the causal physical response (coin speed/accel, contact fₙ retention, dtz, stopping margin, authority)
  → deterministic correction law:  response_error → Δintent   (clip to the intent box; update the 7 R3 roles, not θ)
  → re-decode θ for the remainder  (deterministic, NO extra search budget)
  → continue PUSH → BRAKE → RELEASE (phase-monotone) → frozen K6
```

**Frozen / reused unchanged (identical to R1/R2/R3):** the physics and `CradleSnapshot`; the 6-D PUSH→BRAKE→RELEASE
velocity-feedback option (`forward_displacement.py`, **not modified**); the per-step Δτ map `_schedule_increment`; the
governed step `step_ablation` + `govern_torque`; the R3 deterministic authority-aware decoder
(`authority_decoder.decode_intent`); the 7 intent roles + intent box (`physical_intent.py`); the authority measurement
(`identify_Bcoin` / `object_authority` / `contact_internal_authority`); the frozen K6 (zone 0.02 m, settle 0.06 m/s, dwell
6, `delivery_success`); the centre-inclusive budget search (`fixed_search_select`, `SEARCH_STD = 0.15`); the 4-state panel
(s1=14250, s3=14750 dev; s4=15000, s7=15750 held-out) and held-out discipline; the R3 NW intent predictor and its 6-dev
training set (`_r2_dev_data`).

**Forbidden:** any new STATE feature for the predictor; any held-out-derived feature or hyperparameter; changed
physics/option semantics; a changed reward/K6; peeking at K6 or future values as a controller input; teacher θ / teacher
intent at deploy; any pin / teleport / coin-state edit / snapshot restoration of the executed trajectory.

## 2. Architecture (no modification to the frozen option)

`rollout_primitive`'s `frame_hook(rl, t)` is **observe-only and post-step** — it cannot swap θ mid-trajectory. Therefore
the closed loop is a **new continuous rollout** `closed_loop_rollout(snap, controller, cfg)` that:

- branches `rl = snap.branch()` **once** (the single continuous executed trajectory) and steps it 1…horizon;
- at each step asks the controller for the effective θ, then reuses **verbatim** `_schedule_increment` (Δτ map),
  `step_ablation` + the `govern_torque` control callback (governed step), and the **identical** K6 / peak / contact
  accounting as `rollout_primitive`;
- returns the **same-shape metrics dict** as `rollout_primitive`, so `delivery_success` is the unchanged judge.

**Anti-divergence guarantee (Stage-3 golden test #10a):** `closed_loop_rollout` under a *constant* controller (returns the
same fixed θ every step, no correction) is **bit-identical** to `rollout_primitive(snap, θ)` on all 4 teacher states
(every metric, incl. `coin_trace`). This proves the closed-loop rollout is a strict generalisation that preserves the
frozen option exactly, and justifies re-expressing the loop (the fixed-θ and controller-driven loops are genuinely
different control flow → §6.5 #8 class-per-structural-variant, not a forward-time flag). `forward_displacement.py` is not
edited.

## 3. Frozen response vector (Stage 1 — `ResponseState`, causal past+present only)

Measured from the live `rl` at the correction point (all already-available causal quantities; the same class
`trace_teacher` / the option's brake already read):

- remaining target distance `dtz`; safe remaining distance `d_safe = max(dtz − CENTER_TOL, 0)`;
- coin velocity in the target frame `v_parallel`, `v_perpendicular` (`directional_authority.target_frame`); coin `speed`;
- coin spin (disk angular velocity if surfaced by the planar metrics, else 0 with a provenance flag);
- probe displacement `Δforward` (coin position now − at probe start, projected on `e_par`) and cross-track `Δperp`;
- velocity gain / mean accel over the probe `a_probe = (speed_now − speed_probe_start) / (Δt_probe)`;
- per-contact normal/tangential velocities `v_n`, `v_t` (`measure_contact_velocities`); friction utilisation slip proxy;
- per-side `Fn`, contact-retention (both-contact latch, `straddle`, min `Fn` in push);
- object authority `B_coin` (`object_authority`: forward_push_reach, brake_opposed_reach, lateral_reach); contact/internal
  authority (`contact_internal_authority`: normal_force_reach, balance);
- positive/negative slew headroom (`admissible_dtau_box`);
- **energy / stopping diagnostic:** `E_kin = ½·m_coin·speed²` (m_coin ≈ 0.05027 kg from the model); braking
  authority → achievable decel `a_brake ≈ brake_opposed_reach / control_dt` (governor-realised, slew-admissible);
  `d_stop ≈ v_parallel² / (2·a_brake)`; `W_brake_available ≈ m_coin·a_brake·d_safe`. The **guard** uses the mass-free,
  numerically-equivalent form `d_stop > d_safe` (⇔ `E_kin > W_brake_available`); tiny/zero authority is floored so the
  guard is conservative (treats un-brakable as over-speed);
- current option phase and elapsed horizon.

**Not used:** future trajectory values; K6 outcome as input; teacher intent/θ; held-out labels; any post-hoc information.

## 4. Frozen correction law (Stage 2 — deterministic, `closed_loop_intent.py`)

Deterministic `response_error → Δintent` over the **7 R3 intent roles** (never θ directly); gains/thresholds are dev-tuned
(C1) and then frozen. Rules (each is monotone in its trigger — Stage-3 tests 2,3):

- **A. over-speed / insufficient stopping margin** (`d_stop > d_safe` or `E_kin > W_brake_available`): ↓ peak_velocity,
  ↓ forward_drive, ↑ braking_demand, move brake_entry **earlier**; **never** increase forward simultaneously.
- **B. under-speed with positive margin** (progress insufficient ∧ contact+braking margins healthy): ↑ forward_drive
  conservatively, preserving a safe stopping margin.
- **C. contact-retention risk** (`Fn`/margin dropping or separation growing): ↑ squeeze, ↓ forward_drive; no aggressive
  lateral.
- **D. cross-track / spin error** (target-frame `v_perpendicular`, `Δperp`, spin): update lateral with **mirror-equivariant
  sign** (balance flips under the physical mirror; forward/braking invariant).
- **E. release** (frozen causal guard on dtz, coin speed, stopping margin, contact state, min dwell-safety margin — **never
  the K6 result**).

## 5. Frozen phase guards (Stage 2)

Monotone phase `PUSH(0) → BRAKE(1) → RELEASE(2)`, non-decreasing. Re-decoding may change θ's ramp/release timings; the
controller **timing-clamps** the effective θ so `_schedule_increment` yields a phase `≥` the monotone floor (to force
`≥ BRAKE` at step t, set `ramp' = min(ramp, t−1)`; to force `≥ RELEASE`, `rel' = min(rel, t−1)`). Corrections may bring the
brake **earlier** (a forward advance PUSH→BRAKE, allowed) but **never** revert BRAKE→PUSH. Hysteresis / monotonicity
prevents chatter. The dynamic release step (first RELEASE entry) is what the contact-retention bookkeeping
(`min_fn_push`, `lost_before_release`, `forward_at_release`) is counted against.

## 6. Frozen search-budget accounting (total ≤ 8)

Per the design freeze §1 (verbatim): *"the budget-8 search applies to the initial segment's θ only; corrections are
deterministic re-decodes (no larger search budget)."* Concretely — allocation **`INITIAL_SEGMENT_BUDGET_8`**: the
centre-inclusive `fixed_search_select` (frozen `SEARCH_STD = 0.15`) selects the executed trajectory around the **initial**
θ₀ = decode(initial_intent); each of the ≤8 candidates is a **full continuous closed-loop trajectory** (probe → measure →
correct → continue) from `snap`; the deterministic mid-trajectory corrections add **no** budget. Total structured-search
budget for the complete trajectory = **8** (0 additional for corrections). Budget 0 = direct execution of θ₀ (pure
closed-loop, no search) — the purest isolation of the correction. Provenance records exact centre/candidates/θ_exec.

**Honesty control (load-bearing test).** Closed-loop is *load-bearing* only if it **strictly beats** the R3 open-loop
baseline at the **same** budget (ideally at budget 0). A large search can deliver regardless of the correction; the R3
open-loop condition (`_r3_deploy_one`, decode → `fixed_search_select`) and an **oracle** condition (teacher θ + the same
search, validates search+physics) are run at the gate budget, exactly as R3's update-0. The search budget is never
enlarged for the closed-loop condition only.

## 7. Development-only tuning policy (C1)

Selected on `{s1, s3}` (panel dev) + the 6-dev predictor set only; **s4/s7 never touched** for tuning:

- initial-intent source ∈ `{r3_predictor, conservative_prior}`;
- probe window `W_probe` and correction cadence `C_corr` (first correction at `t = W_probe`, then every `C_corr` steps);
- correction gains + trigger thresholds (per rule A–E), release-guard margins;
- the frozen gate budget (∈ `{0, 4, 8}`, chosen by the informed-vs-uninformed gap on dev).

Selection criterion: development K6 = 2/2 with the smallest, most conservative parameters; ties → the setting that keeps
the largest safe stopping margin. All selected values are written to `controller_parameters.json` and **frozen** before C2.

## 8. Frozen-panel protocol (C2)

Exactly **one** evaluation on `s1, s3, s4, s7`. Hard gate: **total K6 = 4/4, development = 2/2, held-out = 2/2**, one
continuous physical trajectory per state, total search budget ≤ 8, no motion/collision/contract violation, exact
provenance. No repeated held-out attempts; no held-out-derived allocation. Per-state traces (initial intent, probe
response, stopping-distance/energy traces, contact-retention traces, every intent correction, decoder residuals/saturation,
decoded θ centres, search candidate allocation, θ_exec, phase transitions, final distance, terminal velocity, K6 dwell, K6
result, motion/collision margins) are recorded to `c2_frozen_panel.json` + `response_traces.json`.

## 9. Gates & decision tree

| gate | requirement | on fail |
|---|---|---|
| **C0** teacher no-regression | closed-loop on the 4 teacher intents keeps K6 = 4/4, corrections ≈0 on an already-delivering plan, no illegal phase transition | fix a demonstrated correction-law / integration bug only |
| **C1** development | dev (s1,s3) = 2/2 with dev-frozen params | controller / implementation audit — not yet a scientific negative |
| **C2** frozen panel | 4/4 incl held-out 2/2, budget ≤8, no violation | see verdicts below |

| C2 result | verdict | RL |
|---|---|---|
| 4/4, held-out 2/2 | `CLOSED_LOOP_INTENT_CORRECTION_LOAD_BEARING` + `UPDATE_ZERO_MOBILE_TEACHER_NO_REGRESSION_PASS` + `MATCHED_SAC_TD3_AUTHORISED` | **authorised** (do NOT start it this session) |
| 3/4 or held-out 1/2 | `CLOSED_LOOP_FEEDBACK_IMPROVES_GENERALISATION_BUT_GATE_OPEN` | blocked |
| 2/4, held-out 0/2 | `CURRENT_DETERMINISTIC_FEEDBACK_LAW_INSUFFICIENT` (NOT "not learnable") + localise: probe informativeness / gain-timing / phase timing / response history / nonlinear learned residual | blocked |
| C0 fail | `CORRECTION_LAW_DESTABILISES_TEACHER` | blocked |

## 10. Mandatory tests (Stage 3, before any physical gate)

Zero-residual identity; over-speed monotonicity; contact-risk monotonicity; mirror equivariance; energy/stopping guard;
phase monotonicity; decoder contract (corrected intent → frozen decoder → bounded slew-admissible θ); total search-budget
provenance (≤8, exact centre/candidate/θ_exec); no teacher fallback; determinism (same snapshot+seed → identical
correction/θ/diagnostics). Plus the **golden**: constant-controller closed-loop ≡ `rollout_primitive` (bit-identical).

## 11. Plan-bundle status

`pdflatex` / `lualatex` are **absent** on this host (verified 2026-07-27), so `plan.pdf` cannot be built — the same
limitation R3 recorded and proceeded past. The buildable text formats (`plan.tex`, `plan.tikz`, `plan.mmd`) are written to
`docs/plans/2026-07-27-coin-r4-closed-loop-intent/`; the PDF gap is recorded in `contract_audit.json`. The frozen design
contract (`60801b09`) + this Stage-0 contract are the plan-of-record. **CORE.YAML items touched: none.**
