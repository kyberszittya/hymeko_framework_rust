# R10.2 Stage 2 — structured-option torque-path action coordinate: identity gates PASS (pre-training, no learning claimed)

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · HEAD `4d983d4f` · worktree `hymeko_coin_r9_wt` · dev s1 · downstream + transit + Stage-1B scaffold FROZEN · s4/s7 untouched · f1–f4 SEALED · no learned policy claimed · no tag moved**

## Summary

This is **Boundary 2** of the gated R10.2 plan (`docs/plans/2026-07-28-r10-structured-option-torque-path-td3/`): implement the *corrected* action coordinate the Stage-2 audit (`4d983d4f`) specified, and **prove — before any training — that a zero structured-option output changes nothing and still delivers strict K6.** No TD3, no critic warm-up, no structured exploration, no SAC/PPO, no geometry generalization was run.

The retired coordinate (per-step residual `δ_θ(s)` re-evaluated each step) made vanilla TD3 explore with per-step IID noise that integrates into the preload and removed the delivery at every scale. The corrected coordinate emits **one structured option `θ ∈ ℝ¹⁵ = (Δs, Δpreload_start, Δbmax, k₁∈ℝ⁴, k₂∈ℝ⁴, Δτ_T∈ℝ⁴)` once at READY**, builds a desired-torque path `τ_des(φ) = τ₀^θ(φ) + Σ_k c_k ψ_k(φ) + ψ_T(φ)·Δτ_T`, and tracks it with a deterministic per-step law. Exploration (a single Gaussian on `θ` per episode) is therefore temporally coherent *by construction* — but that is a later boundary; here only the coordinate + its identity gates land.

**The load-bearing realization** (approved): the tracker is computed as `a_t = clip(a_π₀^θ(φ) + o_θ(φ)/slew, −1, 1)`, injecting the correction relative to the scaffold's **executed, slew-governed increment** `a_π₀ = clip(target−prev, −slew, slew)/slew` — *not* the raw pre-slew target (the two diverge under saturation; the executed-increment definition is what makes the θ=0 identity bit-exact and avoids the `/slew·slew` round-trip a literal `(τ_des−prev)` tracker would incur). `scaffold_action`/`apply_step` and the governed `step_ablation` stack are reused unchanged, so every torque/slew/joint-speed cap and "no state edit, no hidden force" hold by reuse.

## Verdicts (all PASS — `reports/2026-07-28-r10-structured-option-torque-path-td3/identity_gates.json`)

| gate | result | evidence |
|---|---|---|
| `TORQUE_PATH_ZERO_DELTA_IDENTITY_PASS` | **PASS** | real 15-D zero-init actor emits θ=0 **exactly**; structured params, physical action trace, executable torque path, and terminal `prev_tau` all bit-exact to a pure-scaffold reference |
| `TORQUE_PATH_BIT_EXACT_SCAFFOLD_PASS` | **PASS** | zero-θ roll `qpos`/`qvel`/`prev_tau`/contacts bit-exact to `PhaseShapeCapture.roll(pi0)`; identical downstream input, event/kind trace, and strict-K6 result (identity, training, and deployment share the *same* `rollout` code path) |
| `OPTION_ZERO_POLICY_K6_PASS` | **PASS** | zero-θ → frozen downstream → **strict K6, min_dtz 2.79 mm** (bit-identical to the scaffold reference and the audit's medoid) |

**Exact identity evidence.** All 12 bit-exact fields `true`; `non_bit_exact_fields: []` (nothing was non-bit-exact). `actor_output_exactly_zero: true`. θ=0 terminal-offset `err_norm = 0.0` (requested and executed offsets both `[0,0,0,0]`). Saturation masks at θ=0: 20 steps, `slew_limited` on 16 step-joints (the scaffold's own slewing shows through), **`action_clipped_any: false`** (the zero policy never clips the composed action — as required), `torque_clamped: 0`.

**Nominal K6:** `k6=True, min_dtz_mm=2.79, safe=True` for both the zero-θ option and the independent scaffold roll — identical.

## Files touched

| file | role | Δ |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/torque_path_option.py` | **new** — the action coordinate: `StructuredOption`+`decode_theta`, phase bases (`transient_basis` pinned-0 endpoints, `terminal_basis` 0→1 ramp), `torque_path_offset`, `TorquePathCaptureRoll` (`a_π₀^θ + o_θ/slew`; per-step `StepMasks`), `record_phase_tube`, `terminal_offset_report` | +278 |
| `hymeko_rl/experiments/coin_kinetic_structured_option_gates.py` | **new** — the 3 identity gates + rich bit-exact evidence + saturation masks; imports no trainer | +194 |
| `hymeko_rl/tests/test_torque_path_option.py` | **new** — 16 unit + integration identity tests | +159 |
| `hymeko_rl/coin_delivery/theta_option/capture_rl.py` | **modified (additive)** — `make_zero_actor(seed, act_dim=ACT_DIM)`: optional dim, default 4 (existing behavior/test preserved), enables the 15-D zero-θ actor by reuse | +5 / −4 |
| `reports/2026-07-28-r10-structured-option-torque-path-td3/identity_gates.json` | **new** — machine-readable gate result | — |
| `docs/plans/2026-07-28-r10-structured-option-torque-path-td3/{plan.tex,plan.pdf,plan.tikz,plan.mmd}` | **new** — Boundary-1 plan (4-format, tectonic; Mermaid validated) | — |

**CORE.YAML items touched: none.** Frozen components (`moving_precapture` scaffold, analytic transit, APPROACH/HANDOFF_RESET/R2/coast, physics/safety/K6) untouched — the only committed-file edit is the additive `make_zero_actor` kwarg. No new/removed dependencies.

## Test results

- **`pytest -p no:randomly` on `test_torque_path_option.py` + `test_capture_rl.py`: 24 passed in ~50 s** (16 new + 8 existing). The 8 existing `capture_rl` tests confirm the additive `make_zero_actor` change is backward-compatible (default `act_dim=4`).
- **Unit** (fast, no physics): decoder (zero→exact-0, scale bounds, length precondition), phase bases (transient endpoints=0 and hits knots, terminal ramp 0→1 monotone), offset vanishing at φ=0 and for zero-θ, saturation-mask inference.
- **Integration** (physics): zero-θ bit-exact scaffold (qpos/qvel/prev), real 15-D zero actor exactly-zero, zero-θ strict K6, phase-tube self-consistency, θ=0 terminal-offset exactly zero, and a **non-zero θ actually moves the terminal preload while staying in [−1,1]** (the coordinate is live, not degenerate).
- **Coverage:** every new function/method is exercised (decoder, both bases, offset, `_step_masks`, `StructuredOption.is_zero`, `TorquePathCaptureRoll.{__init__,structural_params,rollout}`, `record_phase_tube`, `terminal_offset_report`). The bit-exact identity test is the regression test that would fail against a non-bit-exact (raw-target) tracker.

## Performance results

Gate experiment: **wall 25.9 s, peak RSS 0.24 GB** (`/usr/bin/time -l`), vs. the plan budget (gates 1–3 min, RSS < 2 GB) and the 16 GB cap. This boundary contains no performance-critical path and asserts no latency budget; the numbers are recorded for provenance.

## Static analysis

- **`ruff check`: All checks passed** (all 4 files).
- **`radon cc -a -nc`: no blocks at C or worse.** `torque_path_option.py` average **A (3.29)**, all blocks A/B. `coin_kinetic_structured_option_gates.py` `run` refactored from **D (23)** to **B** by extracting `_bit_exact_fields` / `_identity_verdicts` / `_mask_summary` / `_same_delivery` (`all([...])` keeps the verdicts flat); average **A**.
- Module lengths: 278 / 194 / 159 LOC — all under the §6.5 400-LOC heuristic.
- **§6.5 anti-patterns: none introduced.** No Cartesian-product API (one `rollout` entry, config via `StructuredOption`/`ThetaScales`); no algorithm-behind-a-boundary; no string-typed config; no global mutable state; no forward-time flags (θ=0 uses the *same* code path, no `if θ==0` branch). The rig is reused from the audit's `_rig` (no third copy).
- **Error handling:** no new `unwrap`-equivalents; `decode_theta` uses an `assert` for its length precondition (§8, Python contract convention). No new suppressions.

## Provenance

- Parent git SHA `4d983d4f`; this boundary is committed as its own commit introducing the files listed above (3 new modules, 1 additive `capture_rl.py`, the plan dir, the report + `identity_gates.json`). No tag moved. Unrelated untracked worktree artifacts (`.pt` checkpoints, earlier r9 report dirs) are deliberately **not** staged.
- Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch 2.12.0 / macOS-arm64 (CPU). `.venv` at the main tree; `hymeko_rl` imported from the r9 worktree.
- Seeds: zero-init actor `seed=1`; rig cradle `kc.S1_SEED` (dev s1). Deterministic — no RNG-driven exploration in this boundary.
- Fixtures: Stage-1B solutions `reports/2026-07-28-moving-precapture-dynamic-handoff/moving_precapture.json` (3 seeds → medoid `pi0`: s=0.3793, preload_start=0.4573, bmax=0.4098, residual_norm=0.892); R2 checkpoint `reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json`.

## Confirmation of scope (what did NOT happen)

- **No training, no critic warm-up, no structured exploration, no SAC/PPO, no geometry generalization.** The gate experiment imports no trainer.
- No persistent state mutated (no checkpoints/datasets/weights written except the read-only gate JSON + report).
- No frozen component or CORE.YAML item modified.

## Open issues / next boundary (STOP here for review)

1. **`TERMINAL_OFFSET_TRACKING` at non-zero Δτ_T** — verify requested vs physically-executed terminal preload matches under slew/clip/contact (the infra `terminal_offset_report` + saturation masks is in place; the θ=0 sanity is 0.0).
2. **`STRUCTURED_THETA_EXPLORATION_ADMISSIBLE`** — ≤3 pre-registered *episodic* θ-noise scales; freeze the smallest giving preserved nominal K6 + safe informative boundary cases + genuinely distinct trajectories. **Freezing the scale is a decision to make with the user in the loop.**
3. Only then: the reward (K6-dominant + phase-tube shaping, no τ★), `StructuredOptionCaptureEnv`, and the 3-seed TD3 (≤400 option-episodes, eval/25, 3-way frozen-panel compare → `STRUCTURED_OPTION_TD3_IMPROVES_OVER_SCAFFOLD` or `..._NO_IMPROVEMENT_WITHIN_FROZEN_BUDGET`).

**Status:** the corrected action coordinate is implemented and proven a transparent glass tube — zero output changes nothing, bit-exact, and still delivers strict K6 (2.79 mm). The scaffold is transparent before the RL is handed a hammer. **STOP at the identity-gate boundary.**
