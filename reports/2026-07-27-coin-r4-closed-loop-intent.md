# R4 — closed-loop basin-aware intent correction: deterministic feedback INSUFFICIENT (RL still blocked)

**Date:** 2026-07-27 (JST) · **Branch:** `recovery/coin-r4-closed-loop-intent` · **Base:** `60801b09` (R4 design freeze,
a linear descendant of `e3624d14` = R3 `PHYSICAL_INTENT_FACTORISATION_ALONE_INSUFFICIENT`). Stage-0 contract:
`reports/2026-07-27-coin-r4-closed-loop-intent-contract.md`.

**Result in one line:** the closed-loop machinery is exact (the GOLDEN passes — the driver reproduces the frozen option
bit-identically) and the deterministic coast-in correction measurably *improves the held-out trajectory* (s7 dtz-end
102→85 mm, s4 57→47 mm), but a **uniform deterministic feedback law does not reach the gate**: it delivers only 1/4 on the frozen panel
and does **not** preserve the teacher (C0), because each cradle's delivering settle-timing is cradle-specific and a single
deterministic parameterisation cannot match all four while also preserving the precisely-tuned deliverable cases. Verdict
**`CURRENT_DETERMINISTIC_FEEDBACK_LAW_INSUFFICIENT`**. **SAC/TD3 remain BLOCKED** (Case A not reached; the R5–R10
continuation does not start).

---

## 1. Frozen baseline & the R4 axis

| axis | result |
|---|---|
| teacher θ → frozen option → K6 | **4/4** (measured this session) |
| R1 flat / R2 relational / R3 physical-intent decoder | all **2/4**, held-out **0/2** |
| **open-loop amortisation of a fixed decision** | the wrong abstraction level (R1–R3) |

R4's authorised axis: replace the open-loop fixed intent with a **closed-loop basin-aware correction over the frozen R3
decoder** on one continuous physical trajectory — measure the early physical response, correct the physical INTENT, re-decode
θ, continue. Everything else frozen (physics, 6-D option, R3 decoder, authority, K6, panel, held-out discipline). **CORE.YAML
items touched: none.** `forward_displacement.py` was **not** modified.

## 2. What the physics actually is (measured per-step traces)

The decisive discovery — the delivering strategy is **PUSH-then-COAST**, not push-then-brake. Both teacher trajectories on
the held-out cradles push to build coin momentum, then **RELEASE** (relax the grip, *no active brake*) and let the coin
coast into the zone under friction and settle:

- **s4 teacher** (delivers, dwell 10): PUSH to t≈10 → RELEASE → coasts dtz 81→14.5 mm over ~50 steps.
- **s7 teacher** (delivers, dwell 18): PUSH to t≈9 → RELEASE → coasts dtz 113→18 mm.

The R3 open-loop predictor fails the held-out cradles by **over-control**, breaking this — each traceable to a specific θ role
(decoded from the predicted intent), **not** to peak-velocity (which is a decoder diagnostic and does *not* map to θ):

- **s7 predictor**: a long velocity-feedback **BRAKE** (ramp 7.6→rel 18.3) kills the momentum 105 mm short (over-braking).
- **s4 predictor**: weak push (fwd θ 0.11 vs teacher 0.20) + over-squeeze (0.23 vs 0.04) → the coin coasts only to 64 mm.

Both held-out failures are **under-shoot from over-control**, not overshoot — correcting the R3 "peak 0.47 witness" would do
nothing (it is inert in the decoder). s3 is a distinct **high-response** regime ("reaches 0.45 m/s with a tiny push") whose
fast coin escapes the grip if pushed, and whose teacher uses a delicate brake-and-settle.

## 3. What was built (Stages 1–3, all committed before the gates)

- `theta_option/closed_loop_state.py` — the causal `ResponseState` (dtz, target-frame v∥/v⊥, coin spin, probe
  displacement/accel, per-contact v_n/v_t + friction util, fₙ, retention, object/internal authority, slew headroom) + the
  energy/stopping algebra (`E_kin`, `a_brake`, `d_stop`, `W_brake`; the mass-free guard `d_stop>d_safe ⇔ E_kin>W_brake`).
- `theta_option/closed_loop_intent.py` — the coast-in trigger (`coast_reach = v²/2·a_friction`; PUSH until the momentum can
  coast to the zone, RELEASE to coast in, BRAKE only on predicted overshoot), the deterministic magnitude correction
  (over-squeeze cut, coast-deficit forward boost, contact-retention squeeze, canonical cross-track), and the monotone
  `PhaseMachine` + `ClosedLoopController`.
- `theta_option/closed_loop_rollout.py` — the **continuous** closed-loop rollout driver (one `snap.branch()`, reuses the
  frozen `_schedule_increment` / governed step / K6 monitor **verbatim**) + the budget-8 centre-inclusive closed-loop search.
- `tests/test_coin_r4_closed_loop.py` — 10 mandatory + the **GOLDEN**: a constant-controller closed-loop rollout is
  **bit-identical** to `rollout_primitive(snap, θ)` on all dev teacher states (every metric incl. `coin_trace`). This proves
  the closed-loop rollout is a strict generalisation that preserves the frozen K6 exactly.

**Two real bugs were found and fixed** during calibration (both would have silently corrupted results): a search-variable
collision (`sc` reused for the SnapContext and the score), and — critically — neutralising θ[3] to the horizon corrupted the
PUSH ramp fraction `min(1,t/ramp)` to `t/60`, massively under-pushing every cradle. After the fix the coin reaches near-zone
for all cradles.

## 4. C0 / C1 / C2 — the frozen gates (`c0_*`, `c1_*`, `c2_frozen_panel.json`)

Frozen dev-selected config `a_friction=0.55, k_forward_deficit=2.5` (best dev K6 in the dev-only sweep; passthrough OFF —
see §5). Budget 8, centre-inclusive, all states; motion within contract.

K6 delivery (✅) / dtz-end mm in parentheses:

| condition | s1 (dev) | s3 (dev) | s4 (held) | s7 (held) | dev | held | total |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **C0 teacher intent → closed-loop** | ✅ (18) | ❌ (30) | ❌ (47) | ❌ (96) | 1/2 | 0/2 | **1/4** |
| **C2 R3-predictor intent → closed-loop (CL)** | ✅ (15) | ❌ (47) | ❌ (47) | ❌ (85) | 1/2 | 0/2 | **1/4** |
| open-loop R3 baseline (same intents, same seed) | ✅ (17) | ❌ (37) | ❌ (57) | ❌ (102) | 1/2 | 0/2 | **1/4** |
| oracle (teacher intent + open-loop search) | ✅ | ✅ | ✅ | ✅ | 2/2 | 2/2 | **4/4** |

*(exact per-state dtz / dwell / θ_exec / corrections in `c2_frozen_panel.json` + `response_traces.json`; the R3 canonical
open-loop baseline is **2/4** — s3's predicted-intent delivery is search-seed-fragile and flips at this panel's uniform seed,
which is itself a signal that the deliverable dev basin is narrow.)*

- **C0 FAILS (1/4):** the coast-in **destabilises** three of the four *delivering* teacher plans — it replaces each cradle's
  precisely-tuned release timing with a generic `coast_reach≥target` rule, and the mismatch misses the zone (s3/s4/s7 land
  30–96 mm short) even though those exact θ deliver open-loop (oracle 4/4). A no-op-on-a-delivering-plan (the C0 requirement)
  is exactly what a timing-replacement is not.
- **C2 (1/4, held-out 0/2), NOT load-bearing:** the closed-loop **ties** the open-loop baseline (both 1/4 at the frozen
  seed) — it does not increase delivery. It *does* measurably improve the held-out **trajectory** — **s7 dtz-end 102→85 mm,
  s4 57→47 mm** — so the closed-loop axis has genuine signal, but it does not cross the 20 mm settle-and-hold threshold on
  either held-out cradle. Motion within contract (peak q̇ ≤ 2.1, peak coin speed ≤ 0.45).
- The **oracle** 4/4 confirms the search + physics + frozen decoder are sound; the sole cause of the miss is the correction,
  not the machinery.

## 5. Why a single deterministic law is insufficient (localised, per the contract)

- **Cradle-specific settle-timing.** The real coast deceleration is not a single constant — the teacher traces imply
  ≈0.42 m/s² (s4) to ≈0.67 m/s² (s7). A single `a_friction` therefore releases too early on one cradle and too late on
  another; the delivering corridor is narrow (the R3 ablation showed even *oracle-directed* single/double-role corrections
  toward the teacher do not deliver held-out).
- **No simple regime split.** The natural discriminator (object `forward_push_reach`) is nearly identical across the four
  cradles (0.072–0.080; s1 is the *highest*, not the fast cradle s3), so an authority-gated "coast vs brake-hold"
  passthrough does not separate the regimes and gives no gain (measured).
- **Overriding a tuned plan is net-harmful.** A C0-safe controller (defer to the decoded θ) reduces exactly to R3 (2/4, held
  0/2); any timing override that could help a *failing* predictor also breaks the *succeeding* teacher, because the two are
  indistinguishable at deploy without a per-cradle reference.

The missing element is therefore **not** the representation (R1/R2), the search (R2 basin audit), the decoder (D0), or the
closed-loop machinery (GOLDEN) — it is that the held-out delivering **decision needs a per-cradle, learned coast/settle model
inferred online**, i.e. a *learned* residual over the physical intent, which is precisely what RL over the intent (the R6–R10
axis) would provide — and which remains gated behind a delivery gate that no open-loop *or* deterministic-closed-loop method
has passed.

## 6. Tests, lint, provenance

- **R4 tests:** 9 fast + 1 slow GOLDEN — **pass** (`test_coin_r4_closed_loop.py`). R3 decoder 6/6 fast, canonical/authority/
  theta/relational 41 fast — **pass** (no regression). **ruff clean.** New functions ≤ CC 14 (`closed_loop_rollout` mirrors
  the frozen `rollout_primitive` loop; < the CC-15 fail bar; refactoring it would risk the bit-identity golden).
- **No §6.5 anti-patterns:** modes on one harness (no v-files); shared `SnapContext` (authority FD computed once);
  Strategy-style controller protocol; no globals. The two calibration bugs were fixed, not worked around.
- **Provenance:** the closed-loop is one continuous `snap.branch()` per state (no reset / teleport / coin edit / teacher
  fallback); budget 8; peak RSS < 0.3 GB; env `.venv` torch 2.12, mujoco, Apple-Silicon CPU. Predictor: R3 NW kernel over 6
  dev cradles, bandwidth 3.0. Plan bundle: `pdflatex` absent → `plan.pdf` not built (recorded in `contract_audit.json`);
  `plan.tex/.tikz/.mmd` written under `docs/plans/…`.

## 7. Decision & exact next action

- **Verdict:** `CURRENT_DETERMINISTIC_FEEDBACK_LAW_INSUFFICIENT` (C0 fail + C2 1/4, held 0/2). *Not* "not learnable" — the
  closed-loop axis measurably improves the held-out trajectory and the machinery is exact.
- **SAC/TD3 authorisation: BLOCKED.** Case A (4/4 incl. held-out 2/2) was not reached, so the conditional R5–R10 continuation
  (tag → residual-intent semi-MDP → matched SAC/TD3) **does not begin** this session.
- **Exact next action (fresh contract, not built here):** a *learned* closed-loop residual over the 7-role physical intent
  that infers the per-cradle coast/settle timing online (online friction/settle estimation, or a learned intent-residual head
  trained under the frozen decoder + K6) — the R6-style residual-intent formulation — gated behind the same 4/4 held-out-2/2
  delivery gate. The deterministic scaffold, `ResponseState`, coast model, and continuous closed-loop rollout built here are
  the substrate that residual would plug into.

## 8. Files touched (no CORE.YAML items)

| file | Δ |
|---|---|
| `theta_option/closed_loop_state.py` | new |
| `theta_option/closed_loop_intent.py` | new |
| `theta_option/closed_loop_rollout.py` | new |
| `tests/test_coin_r4_closed_loop.py` | new |
| `experiments/coin_r4_gates.py` | new (C0/C1/C2 harness; reuses the R3 predictor path + open-loop/oracle baselines) |
| `reports/2026-07-27-coin-r4-closed-loop-intent/` | contract_audit / controller_parameters / c0_teacher_no_regression / c1_development / c2_frozen_panel / response_traces .json |
