# Coin — From Exact-Handoff Authority Recovery to Insertion & SAC/TD3 — Campaign Report

**Started:** 2026-07-26 03:33 JST (overnight 2026-07-26/27 campaign). **Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2`
(μ_coast = 0.179) + frozen V4 motion stack. **CORE.YAML touched:** none (all work in `hymeko_rl`; V2/V4 read-only from
JSON). **RL:** not started in this stage. **O3:** paused.

This report is the running ledger of the staged campaign. Each stage is committed separately; negative and partial
results are preserved.

---

## Stage 0 — Provenance & clean start (PASS)

- HEAD `5f2f0133` is the frozen Session-1 result
  `ONE_STEP_DQ_TARGET_AUTHORITY_NULL_AT_HANDOFF_UNDER_TORQUE_RATE_LIMIT`, committed with its report
  (`reports/2026-07-26-h2-control-to-contact-velocity-identification.md`), artifact
  (`reports/2026-07-26-h2-bv-identification/bv_identification.json`), figure, implementation
  (`contact_velocity.py`), benchmark, and 10 passing unit tests. **Not reopened.** The canonical verdict stands
  verbatim.
- Working tree carried only pre-existing untracked artefacts unrelated to this campaign (`experiment_viewer_web/`,
  other experiments' `.pt`, `reports/2026-07-25-*`, `verification/`) — left untouched.
- No external dependency added.

---

## Stage 1 — Recover a usable control interface

### Frozen development / evaluation split (declared before measurement)

- **development:** s1 (seed 14250), s3 (seed 14750)
- **held-out evaluation:** s4 (seed 15000), s7 (seed 15750) — not tuned on, not silently replaced.

### H2 Session 2 — LIFTED-HORIZON AUTHORITY versus CRADLE COLLAPSE (measurement)

Session 1 proved the *one-step* `Δq_target` interface is null because the torque-rate limiter's per-step slew budget
(`τ̇·dt = 0.3 N·m`) is entirely consumed unwinding the acquisition torque debt (`|raw_pd − prev_tau|` up to 9.4 N·m).
That is a statement about the *interface*, not the *embodiment*. The rate limiter is an **integrator**: held for more
than one step it catches up to the raw PD demand, at which point the `±ε` command difference `kp·ε` survives the clip
and authority appears. Session 2 measured the **race** between authority arrival and cradle collapse.

**Method (frozen).** Primary semantics `HELD_OFFSET` (the `Δq` offset is held on the servo target for all H steps);
control `ONE_STEP_PULSE` (offset on step 1 only). Frozen horizons `H ∈ {1,2,4,8,12,16,24,32,40}`, `H_MAX = 40`, both
`ε ∈ {0.0015, 0.003}`. Every rollout reuses the **exact** Session-1 governed stepping (`pd_governed_torque` +
per-sub-step `govern_torque` callback + `step_ablation` + `prev_tau ← a_h`), so trace-equivalence holds — verified: at
`H = 1` the held rollout reproduces the Session-1 dead zone (`rank 0`) on **every** state. The horizon hierarchy:

| symbol | meaning |
|---|---|
| `H_tau` | first step the `±ε` post-governor torque sequences diverge |
| `H_state` | first step `q`/`q̇` measurably diverge |
| `H_vrel` | first step measurable `‖Δv_rel‖ > 10⁻⁴ m/s`, both branches on the same contact mode |
| `H_collapse` | first step the cradle-alive predicate fails (presence + identity + Fn floor + straddle + unambiguous primary + motion contract) — **not** the one-step FD drift gate |

`H_collapse` deliberately uses the multi-step cradle-survival predicate; the one-step FD `xc_drift ≤ 4 mm` /
`normal ≤ 20°` gates are linearisation gates and would false-trip collapse from benign settling drift.

**Usable-authority gate:** `rank(B_v(H*)) ≥ 2` on **both** ε · reproducible across ε (per-column rel. gap ≤ 0.35) ·
sustains `margin = 3` steps · onset `< H_collapse`.

### Result (`authority_recovery.json`)

| state | split | H_tau | H_state | H_vrel | rank-2 onset | H_collapse | repro | sustains | **usable** | route |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 | dev | 4 | 4 | 4 | 6 | **41 (survives)** | ✓ | ✓ | **yes** | C |
| s3 | dev | 11 | 11 | 11 | 12 | **41 (survives)** | ✓ | ✓ | **yes** | C |
| s4 | held-out | 10 | 10 | 11 | 11 | **12** | ✗ | ✗ | no | A |
| s7 | held-out | 3 | 3 | 3 | 7 | **9** | ✗ | ✗ | no | A |

**Findings (measured / inferred / hypothesis labelled):**

1. **(measured) The one-step null is not a controllability verdict.** On the well-conditioned development cradles
   lifted-horizon `Δq_target` authority is real and usable: rank climbs to **3**, contact-velocity authority reaches
   **0.14–0.26 m/s**, reproducible across both ε, and the cradle survives all 40 steps under the active hold. The
   embodiment is locally controllable at these handoffs — the one-step formulation was the wrong *interface*.
2. **(measured) The authority is the sustained rate-limiter catch-up, not a transient.** The `ONE_STEP_PULSE` control
   returns `rank 0` at **every horizon on all four states** — a clean control-vs-treatment dissociation. A single
   perturbation, released, produces no authority; only the *held* offset does. This rules out a "perturbing-the-state
   transient" artefact.
3. **(measured) Usability is gated by acquisition-handoff quality, not the interface.** The two held-out cradles
   collapse in 9–12 steps: s4 has weak grip (`fn0 = 0.61 / 1.50 N`) and s7 a fragile straddle (collapse at step 9
   despite `fn0 ≈ 5 N`). Authority appears only in the collapse transient (margin 1–2 steps, ε-irreproducible) → not
   usable. Per the frozen decision tree this is the **Route A** signal: `H_vrel` appears within margin of
   `H_collapse` ⇒ a lower-torque-debt / higher-survival acquisition handoff is required *before* a universal H2.
4. **(inferred) The campaign gate is correctly conservative.** All development states are usable and both held-out
   are not; the frozen gate (all-dev-usable ∧ ≥1-held-out-usable ⇒ ROUTE_C) is not met, so the campaign route is the
   dominant required repair: **`ROUTE_A_LOWER_DEBT_HANDOFF`**.

**Route decision:** `ROUTE_A_LOWER_DEBT_HANDOFF` — develop a lower-debt / higher-survival mobile handoff on the
development states (s1, s3), then re-evaluate the horizon-authority gate on the held-out states (s4, s7) *without
tuning on them*. Only if a held-out cradle then becomes usable is the campaign authorised to claim ROUTE_C and proceed
to the H2 active cradle.

### Files touched (Stage 1 measurement)

| file | lines | note |
|---|---|---|
| `hymeko_rl/coin_delivery/horizon_authority.py` | +~330 / 0 | NEW — horizon rollout (`held`/`pulse`), per-column authority, `B_v(H)` assembly + rank, the four `H_*` onsets, per-state + campaign route decision |
| `hymeko_rl/tests/test_horizon_authority.py` | +~180 / 0 | NEW — pure decision/rank/onset tests + the rate-limiter integrator identity on the real `pd_governed_torque` |
| `hymeko_rl/experiments/horizon_authority_benchmark.py` | +~200 / 0 | NEW — reproducible benchmark (reuses the Session-1 acquisition/frozen-load harness), artifact + race/rank/authority figure |
| `docs/plans/2026-07-26-h2-lifted-horizon-authority/` | new | plan.md / plan.tex / **plan.pdf** / plan.tikz / plan.mmd |
| `reports/2026-07-27-coin-solution-to-rl/authority_recovery.json` | new | machine-readable measurement |
| `reports/2026-07-27-coin-solution-to-rl/horizon_authority_race.png` | new | 3-panel figure |

**CORE.YAML items touched:** none.

### Tests

- **Unit** (`pytest -p no:randomly hymeko_rl/tests/test_horizon_authority.py`): **11 passed, 0.4 s** — pure helpers
  (`first_crossing`, `numeric_rank` incl. NaN columns, `cradle_alive` each failure mode, `collapse_step`), synthetic
  column assembly / rank / onset / sustain / reproducibility, the full route decision tree (per-state + campaign), and
  the rate-limiter **integrator identity** exercised directly on `pd_governed_torque` (one-step ±ε bit-identical dead
  zone → multi-step divergence → steady-state authority band `2·kp·ε`).
- **Integration + perf** (the benchmark, production-scale env + acquisition): 4 certified states, both ε, all 9
  horizons, deterministic. **Peak RSS 0.23 GB** (cap 16 GB), **wall 129 s** (4 states, ~32 s/state dominated by env
  reconstruct + straddle acquire). H=1 trace-equivalence to the frozen Session-1 dead zone confirmed on every state.

### §6.5 anti-patterns

None introduced. Reused `pd_governed_torque`, `govern_torque`, `step_ablation`, `measure_contact_velocities`,
`CradleSnapshot`, `acquire_certified_straddle`, `_load_frozen`, `_setup` — no re-implementation (§6.1). Config as frozen
dataclasses; `HorizonConfig` embeds `BvConfig` so the contact-mode floor / ε / motion cap are shared, not duplicated.
Route tags are enum-like literals with single dispatch. `ruff` clean on all three new files.

### Graphical output

`reports/2026-07-27-coin-solution-to-rl/horizon_authority_race.png` — (1) the horizon race (H_tau/H_state/H_vrel vs
H_collapse per state, usable window shaded green), (2) controllable rank vs control step (held vs pulse; the pulse
control flat at 0), (3) contact-velocity authority `‖Δv_rel‖` growth at the frozen horizons.

---

### Stage 1 Route A — mobile low-debt handoff conditioning (attempted; DID NOT PASS the gate)

**Diagnosis first (discriminating).** The baseline `H_collapse` rollouts show the held-out collapse mechanism is
**grip loss** — the tip normal force decays below the 0.05 N floor under the hold (s4: chronic weak left contact
fn≈0.05–0.13 N; s7: progressive preload decay 5→0 N) — **not** velocity (peak q̇ ≤ 0.9 ≪ 3.0) or straddle inversion.
So Route A targets sustaining/restoring preload.

**Repair (`mobile_conditioning.py`).** After the directed straddle acquire, a short MOBILE (free-coin, no pin)
conditioning phase starts from the acquisition hold (base servo target `q_target0`, threaded `prev_tau0` — the standing
grip torque preserved) and adds a **monotone inward squeeze** to any under-loaded side, early-stopping the moment both
tips are balanced and above a physical `fn_target = 1.5 N` (a setpoint developed on the dev cradles, not held-out). Two
diagnosed bugs were fixed en route (recorded as engineering signal): (i) the acquisition's FD `_arm_dir` probe steps
the sim 3× and, on a *free* coin, shoves the coin so the gradient is corrupted → replaced with the **analytic tip
Jacobian** `arm_inward_geom` (`mj_jacGeom`, coin-motion-independent); (ii) re-seeding the hold from `qpos` with a fresh
torque **zeroes the standing grip torque and releases the coin** → base must be the acquisition hold, not the current
pose.

**Result (`authority_recovery_conditioned.json`) — honest negative:**

| state | split | conditioned | vs baseline |
|---|---|---|---|
| s1 | dev | settled 4 steps, balanced, survives 41, but rank-2 **repro=False** → not usable | **degraded** (was usable) |
| s3 | dev | settled, usable ROUTE_C, rank-2 onset 12→8 | preserved/improved |
| s4 | held-out | left contact unrecoverable (fn→0), snapshot invalid | not rescued (geometric weak-contact) |
| s7 | held-out | straddle −0.996→−0.88, collapse **9→5**, not usable | **degraded** |

Conditioning preserves s3 and lowers torque debt (s1 `H_tau` 4→2) but **does not convert a fragile held-out cradle
into a usable one** — it trades survival for lower debt inconsistently, degrades s1's reproducibility and s7's
straddle, and cannot overcome s4's geometric weak left-contact. The number of usable states did not increase
(baseline: s1,s3; conditioned: s3). **Route A does not pass the held-out gate.** The fragile held-out cradles collapse
under passive hold too fast for a lifted-horizon *position*-target interface to be actionable, regardless of
conditioning — exactly the case the frozen tree routes to **Route B** (a slew-admissible Δτ decision variable), which
the Session-1 report independently recommended. Tests: 2 integration tests pass (grip preservation on s1,
`arm_inward_geom` direction).

### Stage 1 Route B — slew-admissible torque-increment interface (PASS — the recovered interface)

**Decision variable:** `Δτ_cmd ∈ R^4`, `|Δτ_cmd_j| ≤ τ̇·dt = 0.3 N·m`; applied step `a = clip(prev_tau + Δτ_cmd, lo, hi)`
then the per-sub-step directional governor (no rate-limiter re-entry — `Δτ_cmd` is admissible by construction, no
governor bypass). Identify `B_τ = ∂v_rel,t+1/∂Δτ_cmd` one step at the exact certified handoff, FD around the nominal
hold increment `Δτ_0 = clip(raw_pd − prev_tau, ±step)`; admissible perturbations are central where `Δτ_0` is interior
and one-sided into the box where it sits on the slew edge (the saturated-joint case). Reuses the Session-1 governed
stepping — the *only* change from B_v is the decision variable.

**Result (`authority_recovery_route_b.json`) — clean PASS on all four states:**

| state | split | Δτ_0 (slew edges) | rank ε=0.05 / 0.10 | active cols | repro gap | usable |
|---|---|---|---|---|---|---|
| s1 | dev | [.3,.3,−.3,.3] | 3 / 3 | 4/4 | 0.029 | **yes** |
| s3 | dev | [.3,−.3,.3,−.3] | 3 / 3 | 4/4 | 0.136 | **yes** |
| s4 | held-out | [−.3,.3,−.3,.3] | 3 / 3 | 4/4 | 0.005 | **yes** |
| s7 | held-out | [.3,−.3,−.3,−.3] | 4 / 3 | 4/4 | 0.112 | **yes** |

Every `Δτ_0` sits at the ±0.3 slew edge — confirming the debt saturates the limiter (the B_v dead-zone root) — yet the
one-sided admissible perturbation reveals **rank 3–4 authority with all four columns active, reproducible across ε, on
every state including both held-out**. `ONE_STEP_DQ_TARGET_AUTHORITY_NULL` is a statement about the *position-target*
interface only; the slew-admissible **torque-increment** interface has immediate full one-step authority, and because
it is a one-step measurement the fragile held-out cradles (alive ≥ 9 steps) qualify — the property Route A's
lifted-horizon route could not use.

**Verdict: `ROUTE_B_SLEW_ADMISSIBLE_DTAU_ESTABLISHED`.** `Δτ_cmd` is the recovered control interface for downstream
control. The interface-recovery gate (Stage 1) is **PASSED**.

### Strategy pivot (2026-07-26, user directive) — dynamic transition, not a quasi-static cradle

With the interface established, the campaign de-escalates the over-rigid `acquire → perfect 40-step cradle → null-wrench
→ CLF/CBF-QP → …` chain. The coin move does not require a persistent quasi-static cradle to be *proved stable first*;
it can be a **dynamic coordinated transition** (bilateral contact → brief squeeze → forward push → controlled release)
that only needs to *pass through* the intermediate state. **Kept** as guardrails: exact snapshot + replay, real
common-contact-point velocities, torque provenance, motion & safety contract, straddle/internal-force certificate, the
external success check. **Dropped** (deferred, may return later): the port-Hamiltonian H2 CLF/CBF-QP, a universal
40-step controlled-invariant cradle, a full B_τ audit per controller step, LSTM, further certificate layers.

**Revised path:** slew-admissible torque **primitives** (4–6 params: squeeze magnitude, forward differential torque,
L/R balance, ramp duration, release time) → short-horizon **CEM/MPC search** (8–16 steps, hard-exclusion objective) →
first **`CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT`** (≥ 5 mm target-directed, > 5× passive drift, bounded
cross-track, no motion-contract breach, no pin) → mobile composition → BC / update-0 on the low-dim primitive params →
matched SAC / TD3 in that option space.

---

## Stage 3 — CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT (PASS — 2/4 incl. held-out)

**Primitive (`forward_displacement.py`).** From the exact free-coin handoff, a 5-param torque primitive
θ = (squeeze_mag, forward_mag, balance, ramp_steps, release_step) maps, per control step, to a slew-admissible
increment `Δτ_cmd_t` (|Δτ_cmd| ≤ τ̇·dt) built from the LIVE geometry: `squeeze_dir` = normalized (inward_L + inward_R)
(grip; analytic tip Jacobian), `forward_dir` = normalized (Jᵀe_par|_L + Jᵀe_par|_R) (push both tips toward the zone,
`e_par` = direction_to_zone), `balance` shifts the push between arms; the schedule ramps the push+grip up, holds
(coast), then relaxes the grip at `release_step`. Applied as `a = clip(prev_tau + Δτ_cmd, lo, hi)` then the governor —
the same governed stack as Route B. Bounded CEM (pop 32, iters 6, deterministic) maximises a physical objective
(forward displacement, hard-excluded on a motion-contract breach / coin-speed blow-up / net-backward, soft-shaped
toward a controlled release), H = 16 steps.

**Success certificate (external, physical — not reward):** forward ≥ max(5 mm, 5× passive) ∧ forward > 5× passive ∧
cross ≤ forward ∧ **dual contact retained through the push phase** (`lost_before_release = 0`, `min_fn_push ≥ 0.05 N`) ∧
**controlled release** (terminal coin speed ≤ 1.0 m/s) ∧ no motion-contract breach ∧ no pin. Passive drift is the honest
**position-hold** baseline (pd servo at q_hold — a torque hold is not stationary, its residual torque pushes the coin).

**Result (`controlled_insertion.json`):**

| state | split | passive (mm) | threshold (mm) | forward (mm) | cross (mm) | fn_push (N) | term v (m/s) | peak q̇ | **success** |
|---|---|---|---|---|---|---|---|---|---|
| s1 | dev | 10.30 | 51.5 | 40.2 | 2.1 | 0.60 | 0.37 | 2.01 | no (below 5× bar) |
| **s3** | **dev** | 4.49 | 22.4 | **42.4** | 10.4 | 2.56 | 0.99 | 1.58 | **YES** |
| **s4** | **held-out** | 0.70 | 5.0 | **38.7** | 4.7 | 0.55 | 0.34 | 1.35 | **YES** |
| s7 | held-out | 24.35 | 121.8 | 81.5 | 13.3 | 0.90 | 0.99 | 1.37 | no (below 5× bar) |

**The coin moves — controlled — on a development AND a held-out state.** s3 (dev, 42 mm) and s4 (held-out, 39 mm) meet
the full certificate: contact retained through the push, controlled release, predominantly forward, no motion breach,
no pin. s1 and s7 produce equally *controlled* pushes (40 / 82 mm, contact retained, controlled release) but their
coins **intrinsically slide forward 10 / 24 mm even under a position hold** (the cradle sits on a slope), so their
5×-passive bars (51 / 122 mm) are very high — the controlled push still adds a real 3.3–3.9× over passive there, just
not 5×. The Stage-3 gate (≥ 2 certified successes ∧ ≥ 1 held-out ∧ forward clearly exceeds passive ∧ no motion breach)
is **met**.

**Milestone: `CONTROLLED_BIMANUAL_FORWARD_COIN_DISPLACEMENT`.** Tests: 3 pure predicate/score tests + 1 integration
(s3 controlled push). Graphical output: `controlled_forward_displacement.png` (forward-vs-threshold bars + best-θ coin
trajectories) and non-static GIFs `forward_push_s3.gif` / `forward_push_s4.gif` (16 frames, verified real coin motion).
Peak RSS 0.31 GB, wall 135 s. No new deps (imageio already present).

---

## Stage 4 — MOBILE K6 DELIVERY (a longer-horizon push-and-coast delivers; genuine K6 on s1)

The Stage-3 push closes ~40 % of the delivery gap in 16 steps. Testing the hypothesis that a **longer-horizon
push-and-coast** (H = 48, single continuous trajectory — **no re-grip, no teleport, no coin-state edit**) can deliver:
the coin is pushed toward the zone, released, and coasts on the low-drag surface (μ = 0.179) while decelerating. The
certificate is the **FROZEN K6 monitor** (`CENTER_TOL = 0.02 m`, `SETTLE_VEL = 0.06 m/s`, `HELD_DWELL = 6` consecutive
in-zone-and-settled steps, touched) — the real monitor, not a proxy. (A first pass used a loose 0.25 m/s settle proxy
and mis-scored s1/s4 as delivered; caught and corrected — the metric-integrity guard.)

**Result (`mobile_teacher.json`):**

| state | split | dtz start→end (mm) | gap closed | terminal coin v (m/s) | K6 held-dwell | **K6 delivered** |
|---|---|---|---|---|---|---|
| **s1** | **dev** | 76 → **16.6** | 78 % | **0.000** | **21/6** | **YES** |
| s3 | dev | 100 → 7.7 | 92 % | 0.402 | 0/6 | no (reaches, not settled) |
| s4 | held-out | 96 → 14.4 | 85 % | 0.215 | 0/6 | no (reaches, not settled) |
| s7 | held-out | 138 → 1.4 | 99 % | 0.564 | 0/6 | no (reaches, not settled) |

**A genuine, single-trajectory K6 delivery exists (s1, dev):** the coin is bimanually pushed, coasts into the zone, and
**settles to rest (0.000 m/s), dwelling 21 steps** in-zone (≥ 6 required), touched, motion contract held (peak q̇ 2.06),
under a single continuous unpinned trajectory — a real delivery under the **realistic** motion contract (the property
the legacy coin solutions lacked; see `project-realistic-motion-contract`). GIF: `k6_delivery_s1.gif` (48 frames,
verified real motion). **All four states reach the zone** (dtz end 1.4–16.6 mm, 78–99 % of the gap closed); the three
non-deliveries **reach but do not settle** — the coin arrives still moving (0.2–0.56 m/s). The remaining gap is
**braking/settling**, the known-hard part of realistic-contract delivery.

**Teacher status:** one complete K6-certified trajectory (s1) + three reach-zone near-misses + the CEM populations — a
**minimal** teacher (one delivery, dev only), not yet a multi-seed / held-out-delivering teacher.

---

## Campaign summary & verdict

| stage | result | commit |
|---|---|---|
| 0 provenance | PASS (frozen result intact, no deps) | — |
| Session 2 measurement | lifted-horizon Δq authority real & usable on dev; held-out collapse fast; ONE_STEP_PULSE control flat (authority is sustained catch-up) | `2640567e` |
| 1·Route A (mobile conditioning) | honest **negative** — does not rescue held-out (grip decay diagnosed; degrades s1/s7; can't fix s4 geometric weak contact) | `58de3dd6` |
| 1·Route B (slew-admissible Δτ) | **PASS** — one-step Δτ authority rank 3–4 on **all four** states incl. held-out; `Δτ_cmd` is the recovered interface | `e7b24d6f` |
| 3 forward displacement | **PASS** — controlled bimanual push, 2/4 incl. held-out (s3 42 mm, s4 39 mm) | `6b16bdb9` |
| 4 mobile K6 delivery | **PARTIAL PASS** — genuine frozen-K6 delivery on s1 (dev, settled 21-step dwell); 4/4 reach the zone (78–99 %); 3 not settled | `b0d6e33e` |
| 5 update-0 / 6 SAC-TD3 | **NOT STARTED** — teacher is minimal (1 dev delivery); starting multi-seed RL here would violate the no-rush / ≥1-held-out discipline | — |

**Honest verdict:** the frozen `ONE_STEP_DQ_TARGET_AUTHORITY_NULL` was an *interface* fact, not an uncontrollability
verdict. A slew-admissible **torque-increment** interface (Route B) recovers full one-step control authority on **every**
certified cradle; a 5-param torque primitive under short-horizon CEM produces a **controlled bimanual forward
displacement** (Stage 3) and, at a longer horizon, a **genuine K6-certified mobile coin delivery on s1** — the coin is
finally moved *and* delivered *and* settled, cleanly, under the realistic motion contract, with no pin / teleport /
hidden force. All four cradles reach the zone (78–99 % of the gap); the open gap is braking/settling the coin.

**Remaining blocker:** lift the reach-zone-but-not-settled states (s3, s4, s7 — incl. both held-out) to K6 by adding a
**brake/settle phase** to the primitive (decelerate the coin as it nears the zone; the Δτ interface has the authority),
then a multi-seed, held-out-delivering teacher. **Exact recommended next action:** (1) add a 6th primitive param — a
near-zone braking increment (opposing coin velocity via the contacts) — and re-search delivery on all four; (2) once
≥ 2 states incl. a held-out deliver K6, build the low-dim (θ) teacher dataset from the CEM demonstrations →
BC/update-0 → matched SAC/TD3 in that θ option space (interface + primitive + option-space machinery all in place).
Do **not** start SAC/TD3 before the multi-seed teacher exists (campaign rule).
