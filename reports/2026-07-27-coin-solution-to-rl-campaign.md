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

## Next: Stage 1 continuation — Route A (lower-debt mobile handoff)

Develop a mobile handoff-conditioning phase on the development states to reduce torque debt and raise cradle survival
so the held-out cradles' rank-2 authority becomes usable, then re-run the horizon-authority gate on held-out.
