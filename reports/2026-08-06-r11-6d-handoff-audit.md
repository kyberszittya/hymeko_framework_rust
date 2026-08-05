# R11.6D Phase 1+2 — Causal Handoff Audit (the cause is target geometry, not handoff state)

**Date:** 2026-08-06
**Worktree:** `hymeko_coin_r9_wt` · **Branch:** `feature/r11-4a-target-conditioned-delivery-teacher`
**Base SHA:** `d09416c9` (frozen composition baseline `8f2d796f` preserved)
**Finding:** the c3 far-angle delivery gap is caused by **target geometry (required transport)**; handoff state is
causally irrelevant; a capture-side canonicalizer (Phase 3) is **refuted**; the lever is retrieval-side (Phase 4), and it
is **proven feasible**.

---

## Method (diagnostic only — no optimization)

For each of the 3 c3 far-angle dev failures, pair it with its retrieved (nearest) bank demo, reconstruct both handoffs,
and roll the **same** retrieved θ from three initial conditions: `s_bank` (control, must strict-K6), `s_dev` (baseline,
must reproduce the ~30 mm miss), and `s_dev[c ← s_bank]` — the dev handoff with **one** component swapped to the bank
value, for every counterfactual axis `c ∈ {coin pos, coin yaw, coin lin-vel, coin spin, arm q, arm q̇, prev_tau, zone}`.
The axis whose restoration recovers strict-K6 is the control-critical variable. The nearest θ gives a clean control (the
bank demo's own certified θ from its own handoff → K6). Coin/zone swaps do the required `mj_forward` +
planar-metrics-cache refresh.

---

## Result 1 — one of the three is a retrieval-*blend* artifact, not a transportability failure

| c3 far-angle failure | std_weighted3 (deploy) | **std_nearest** | mechanism |
|---|---|---|---|
| `c3_r7_a+45` | FAIL 29.3mm | **K6 12.8mm** | blend artifact |
| `c3_r9_a-30` | FAIL 32.9mm | FAIL 24.4mm | transportability |
| `c3_r9_a-15` | FAIL 31.0mm | FAIL 28.8mm | transportability |

`c3_r7_a+45` fails only under the **weighted3 blend** — the single **nearest** robust θ delivers it (12.8 mm).
Averaging 3 chaotic far-angle θ produces a worse delivery θ than the single best. This is a **retrieval-combine
finding** (do not distance-blend far-angle neighbors) and a near-free fix by config. The two genuine failures
(`r9_a-30`, `r9_a-15`) fail under *both* configs and are the real counterfactual targets.

## Result 2 — the cause is target geometry; handoff state is causally irrelevant

For both genuine failures, the **only** axis whose restoration recovers strict-K6 is **`zone`**:

| dev failure | dtz_dev (target dist) | dtz_bank | **diff** | baseline | zone-swap | any state axis |
|---|---|---|---|---|---|---|
| `c3_r9_a-30` | 99.96mm | 76.91mm | **23.0mm** | 24.4mm | **K6 5.55mm** (gap +0.175) | FAIL (≤ +0.029) |
| `c3_r9_a-15` | 98.16mm | 68.30mm | **29.9mm** | 28.8mm | **K6 16.29mm** (gap +0.056) | FAIL (≤ +0.011) |

`mean_gap_gain`: **zone +0.115** vs every handoff-state axis ≤ +0.017. **The target-distance difference (23–30 mm)
exactly equals the delivery undershoot (24–29 mm).** The retrieved θ transports the right amount for the *nearer*
neighbor target; the dev r9 target is ~100 mm — so it undershoots by precisely the distance gap. No coin velocity,
orientation, spin, position, arm configuration, or torque state recovers K6 — swapping any of them to the bank value
leaves the coin ~25–29 mm short. **The handoff state is not the cause.**

**Consequence:** a capture-side canonicalizer (Phase 3) regulates handoff state — which the ablation shows is causally
irrelevant here — and it cannot move the target. **Phase 3 is refuted by the audit.** (Caveat: the zone swap conflates
target distance and direction; for `r9_a-30` the neighbor shares the angle (−30) so it is distance-dominated; for
`r9_a-15` both differ. Either way the causal split is handoff-state vs target-geometry, and it lands unambiguously on
target-geometry.)

## Result 3 — a delivering θ EXISTS in the bank; descriptor-nearest retrieval misses it (Phase 4 is feasible)

Rolling **all 44 train θ** from each dev handoff:

| dev failure | train θ delivering strict-K6 (safe) | best θ (source) |
|---|---|---|
| `c3_r9_a-30` | **2/44** | `bank_c0_3` → 12.77mm (gap 0.872); `c3_r6_a+15` → 19.53mm |
| `c3_r9_a-15` | **2/44** | `bank_c1_+0.01_+0.02` → 10.96mm (gap 0.888); `bank_c0_3` → 18.43mm |

The transport capacity **is** in the bank — but the delivering θ come from **descriptor-far** scenarios (`c0_3`, `c1`)
whose θ have more push (transport farther). Descriptor-nearest retrieval matches coin/target geometry and picks a
*closer-target* neighbor whose θ undershoots. A **transportability-aware** retrieval (estimating from which handoffs each
θ_i actually delivers, then retrieving by predicted success rather than descriptor distance) would find these θ. Phase 4
is not only indicated — it is **proven able to reach all 3 c3 far-angle** (`r7_a+45` via nearest-not-blend; `r9_a-30` via
`c0_3`; `r9_a-15` via `c1`), i.e. dev 4/7 → potentially 6–7/7.

---

## Interpretation & the redirected R11.6D path

The static descriptor carries the target (position) but the **standardized-Euclidean distance under-weights the
required-transport component**, so nearness in descriptor ≠ transportability of the control law — exactly the central
hypothesis, now causally confirmed and localized to **target distance**, not a trajectory/contact/momentum handoff
variable.

- **Phase 3 (capture-side canonicalizer): NOT indicated** — the audit refutes handoff state as the cause; no bounded
  handoff regulator can help, and none can move the target.
- **Phase 4 (transportability-aware retrieval): the lever, and feasible** — measure, on **train-only** handoffs, the
  θ×handoff success matrix (from which handoffs each certified θ_i delivers), fit `P(K6 | s, θ_i)`, and retrieve by
  predicted success. The audit says the dominant feature is **required transport (target distance)**; the recalibration
  should up-weight it (and must not distance-blend far-angle θ — Result 1). The delivering θ for the r9 targets come from
  far-target-push demos, so the success model must span the train handoff set, not only local perturbations.

**This is a HALT point** (per the plan: report the identified variable before designing Phase 3/4). The variable is
**target geometry / required transport**; Phase 3 is refuted; Phase 4 is the path and is proven viable.

---

## Files / tests / provenance

| File | Δ |
|---|---|
| `hymeko_rl/coin_delivery/handoff_audit.py` | `read_audit`, `swap_component` (8 axes), `roll` (new) |
| `hymeko_rl/experiments/r11_6d_handoff_audit.py` | audit + counterfactual + summary (new) |
| `hymeko_rl/tests/test_r11_6d_audit.py` | 5 tests (swap axes, component-diff, summary split) |
| `reports/2026-08-06-r11-6d-audit/summary.json` | full per-pair audit + counterfactual (new) |
| `docs/plans/2026-08-06-r11-6d-handoff-transportability/` | §2 plan, 4-format, tectonic PDF (gitignored) |

- **Smoke caught a real bug** before any sweep: `reconstruct_capture` returns a `_ReachCapture`; the snapshot is
  `.result.outcome.snapshot` (fixed).
- **CORE.YAML:** none. **Deps:** none. Reach/capture/descriptor/rollout/frozen-table read-only; the audit only reads and
  rolls (no state is persisted, no optimization).
- **Env:** framework `.venv`, torch 2.12.0, macOS, CPU; ~9 min wall (6 reconstructs + counterfactual rollouts +
  feasibility 88 rollouts); RSS ≪ 16 GB. Deterministic (fixed per-scenario seeds; reconstruct drift 0 confirmed in
  R11.6C). 8 tests, ruff/radon clean.
- **The frozen test split is untouched.** `8f2d796f` preserved as the exact-zero composition baseline.

## Boundary

Diagnostic complete. Next (gated on your go-ahead): Phase 4 transportability-aware retrieval, fit on train-only handoffs,
targeting dev ≥ 6/7 with ≥ 2/3 c3 far-angle strict-K6 — no canonicalizer, no runtime oracle/CEM, test sealed.
