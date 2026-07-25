# H2 — Control-to-Contact-Velocity Response Jacobian B_v: Identification & Validation

**Date:** 2026-07-26 (JST) · **Scope:** H2 active-stabilizability, session 1 — *identify and validate B_v only*.
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` (μ_coast = 0.179) · frozen V4 stack. **O3 not resumed. No RL. No H2 QP.**

## Summary

The task was to identify and validate the **governed-servo control-to-contact-velocity response Jacobian**

```
B_v = ∂ v_rel,t+1 / ∂ Δq_target   ∈ R^{4×4}
```

at the exact acquisition handoff of the certified straddle cradles, and decide whether it is locally predictive enough to
support a one-step CLF constraint — **without** building the H2 QP.

**Result: `B_v ≡ 0` at every certified post-release handoff.** The one-step response of the contact-relative velocity to a
joint position-servo target perturbation `Δq_target` is *identically zero* — not from a measurement artefact, but because
the shared governed stack's **torque-rate limiter is already slewing at its full per-step rate** to hold the preload, so a
one-step target perturbation (`kp·ε ≤ 0.36 N·m`) is entirely absorbed by the rate clip (`τ̇·dt = 0.3 N·m/step`). The `+ε`
and `−ε` branches receive **bit-identical true post-governor torque sequences** ⇒ bit-identical one-step trajectories ⇒
`Δv_rel = 0`.

**Verdict: `ONE_STEP_DQ_TARGET_AUTHORITY_NULL_AT_HANDOFF_UNDER_TORQUE_RATE_LIMIT`.** None of the four positive §5 verdicts
holds; the one-step, position-target formulation of B_v has null control authority at these handoffs. The overnight
measurement-rigor gate therefore **FAILS** (0 ACTIVE B_v columns), and per protocol the full multi-seed validation was
**not** run. **The H2 CLF/CBF QP was not built.**

This is a clean negative that *characterises the obstacle precisely* and points to the reformulations (longer prediction
horizon and/or a slew-admissible torque-level decision variable) — pursued separately in H2 session 2.

## Exact definition of B_v (the frozen contract)

- **Decision variable:** `Δq_target ∈ R^4`, a perturbation to the four joint position-servo targets. Nominal command
  `Δq = 0` = the acquisition's exported handoff target `q_handoff_target`. Linearisation point = the hold command.
- **Prediction horizon:** exactly one control step (`control_dt = 0.01 s`, `substeps = 20`; one `step_ablation` =
  one `rl.inner.step`).
- **Output:** the two contacts' frozen-frame relative velocity `[v_n,L, v_t,L, v_n,R, v_t,R]`.
- **B_v ≠ J_tip:** the identified object runs through the full governed stack — PD position servo → torque-rate limit →
  per-sub-step directional velocity governor → actuator saturation → MuJoCo contact solver → FREE-coin motion at the
  contacts. It is *not* the geometric fingertip Jacobian. (Measured contrast: `‖B_v − J_tip‖ / ‖B_v‖` is ~3.8·10¹³ here,
  because `B_v ≡ 0` while the geometric `J_tip` is O(40) — the two are maximally different.)

## Snapshot fidelity (exact-state branching)

- **Operating point:** `EXACT_POST_RELEASE_HANDOFF`. The snapshot is a `copy.deepcopy` of the env (MuJoCo model+data
  incl. `qacc_warmstart`, contacts, pin damping) with the coin pin **released** (free coin held only by the two contacts).
  Every branch is a fresh deepcopy restoring the identical state; the stored env is never stepped in place.
  Branch-origin determinism: `max|g(0) − g(0)| = 0.0` (bit-reproducible).
- **Controller state:** the TRUE handoff state exported from `straddle_directed_acquire` — `prev_tau` (the acquisition's
  last applied, rate-limited torque) and `q_target` (its last servo target). No re-synthesised `−kv·q̇` seed (which off
  the contact limit-cycle velocity spuriously saturates the actuator and *fabricates* dead columns — an early wrong
  operating point, caught and discarded), and no passive settle (which drops a tip within a few frames — the
  `PASSIVE_CRADLE_VIABILITY = FAIL` fact — and destroys the cradle).
- **Hybrid-reset honesty:** pin-release is a damping-parameter change, so the instantaneous state (qpos/qvel) is
  unchanged — `pre_release_hash == post_release_hash` for both states — but the *future dynamics* differ, so the
  post-release state is **re-certified**, not inherited: dual contact ✓, straddle ✓ (n_L·n_R = −0.996 / −0.989),
  internal-force certificate ✓, qdot ≤ 1.2 rad/s ✓ (0.454 / 0.914).
- **Handoff transfer proven (replay):** continuing the pre-release acquisition state one step vs the snapshot's Δq=0
  branch gives bit-identical control-step torque (`max mismatch 4.5·10⁻⁶ / 4.7·10⁻⁶ N·m`, within 5-decimal rounding) —
  the operating point was transferred, not two fields copied.

## Relative-velocity semantics (common contact point)

- `v_rel := v_tip(x_c) − v_coin(x_c)` — the velocity of the TIP relative to the COIN, evaluated at the **common MuJoCo
  contact point** `x_c = contact.pos` (not a geom origin). Both bodies' rigid-body velocities are transported to `x_c`
  (`v_origin + ω·ẑ×(x_c − origin)`) — the 20 mm tip radius makes the ω×r term otherwise ~2× wrong.
- **Sign (frozen):** the projection frame `(n0, t0)` is the ACTUAL MuJoCo contact normal at the snapshot t, oriented
  tip→coin, so **`v_n > 0` = closing/compression**; `t0 = R₊₉₀ n0`. The frame is FROZEN at t; every branch projects its
  live t+1 relative velocity onto that same frame, isolating the velocity response from frame rotation in the FD.

## FD procedure

- Central difference, `B_v[:,j] = [g(+ε e_j) − g(−ε e_j)] / (2ε)`, `g(Δ) = v_rel,t+1(Δ)`; baseline `g0 = g(0)`.
- Two ε scales (`0.0015`, `0.003` rad); forward one-sided compared near constraints. **ε was not enlarged** to escape the
  dead zone.
- **Contact-mode validity** (a column is dropped, no derivative fabricated, unless): both tips present with the same
  fingertip–disk geom-pair identity, both `Fn ≥ 0.05 N`, common-contact-point drift `≤ 4 mm`, contact-normal rotation
  `≤ 20°`, exactly one tip–disk contact per side (unambiguous primary), straddle retained, motion contract held.
- **Column classification** from the TRUE per-sub-step post-governor torque: a genuine one-step dead zone requires the
  `±ε` post-governor ctrl **sequences** to be identical (not merely the rate-limited step torque `a`).

## Prediction-vs-simulation result

Both certified smoke states (s1 seed 14250, s3 seed 14750) — the full run uses s1, s3, s4, s7 — return, at both ε scales:

| state | fn0 (L,R) N | prev_tau (N·m) | actuator sat | recert | replay ok | det | valid cols | ACTIVE cols | column class |
|---|---|---|---|---|---|---|---|---|---|
| s1·14250 | 2.84, 8.75 | [−1.17,−0.93,−0.31,2.19] | none | admissible | ✓ | 0.0 | 4/4 | **0/4** | DEAD_ZONE (all) |
| s3·14750 | 2.37, 2.56 | [−1.03,−0.14,−0.25,1.20] | none | admissible | ✓ | 0.0 | 4/4 | **0/4** | DEAD_ZONE (all) |

`B_v` is the 4×4 zero matrix in both cases; every column classified
`DEAD_ZONE_IDENTICAL_POST_GOVERNOR_TORQUE` (`post_governor_seq_identical = True`, `governor_active_frac = 1.0`, no
actuator clipping). Held-out predictions are trivially exact-in-direction but the actual response norm is `< 10⁻⁹`
(no measurable response), so relative error / cosine are undefined — the honest statement of a null response.

### Why (mechanism)

The nominal raw PD torque is already far past the per-step slew budget: `|raw_pd − prev_tau|` = `[1.77, 9.43, 1.77, 0.66]`
(s1) vs the slew step `τ̇·dt = 0.30 N·m` and the absorption band `step + kp·ε ≈ 0.66 N·m`. Every joint's debt ≥ the band,
so `raw ± kp·ε` stays on the same side of the clip → **identical applied torque for +ε and −ε** → identical trajectory →
`Δv_rel = 0`. See `bv_rate_limiter_mechanism.png` (left: per-joint slew debt vs the band; right: the resulting B_v column
norms, all ≈ 0).

## Validity radius

Not applicable in the usual sense: there is **no** local region in `Δq_target` over which `B_v` is nonzero at these
handoffs. The dead zone is not an epsilon-scale artefact — it holds at both `ε ∈ {0.0015, 0.003}` and is bounded below by
the slew-debt/rate-limit inequality above, independent of ε. The trust region of the *one-step position-target* Jacobian
is empty.

## Failure modes / guards exercised

- **Synthetic saturated seed** (early wrong operating point): a `−kv·q̇` seed of `prev_tau` saturated the loaded arm at
  ±4 N·m and produced *fabricated* dead columns; discarded in favour of the exported acquisition torque.
- **Passive settle destroys the cradle:** an 8-step free-coin hold drops the left tip to `Fn = 0` (passive non-viability);
  removed.
- **Pinned positive control** (`MEASUREMENT_PIPELINE_POSITIVE_CONTROL_ONLY`): the same states with the coin held also show
  0 ACTIVE columns — confirming the dead zone is a property of the **arm servo rate-limiter**, independent of coin
  dynamics, and that the pipeline is not silently zeroing a coin response.

## Honest verdict

**`ONE_STEP_DQ_TARGET_AUTHORITY_NULL_AT_HANDOFF_UNDER_TORQUE_RATE_LIMIT`.** The one-step, position-target B_v is
identically zero at every certified handoff because the torque-rate limiter absorbs the perturbation. The overnight gate
(needs ≥ 2 ACTIVE columns) **FAILS**; the multi-seed validation was not run. Recommended reformulations (H2 session 2, not
this session): (i) lift the prediction horizon beyond one control step and test whether `Δq_target` authority appears
*before* the cradle collapses; (ii) a slew-admissible torque-level decision variable; (iii) an acquisition handoff with
lower torque debt. **The H2 CLF/CBF QP was not built.**

## Files touched

| file | + / − | note |
|---|---|---|
| `hymeko_rl/coin_delivery/contact_velocity.py` | +~430 / 0 | NEW — pure transport, readers, snapshot, FD identification/validation |
| `hymeko_rl/tests/test_contact_velocity.py` | +~140 / 0 | NEW — pure + reader unit tests |
| `hymeko_rl/experiments/bv_identification_benchmark.py` | +~380 / 0 | NEW — reproducible benchmark, gate, figure |
| `hymeko_rl/coin_delivery/cooperative_launch.py` | +5 / −2 | additive — `straddle_directed_acquire` exports `final_tau` / `final_q_target` |
| `docs/plans/2026-07-26-h2-bv-identification/` | new | plan.md / plan.tex / plan.tikz / plan.mmd |
| `reports/2026-07-26-h2-bv-identification/` | new | `bv_identification.json`, `bv_rate_limiter_mechanism.png` |

**CORE.YAML items touched:** none (all work in `hymeko_rl`, non-core; frozen V2/V4 read-only from JSON).

## Test results

- **Unit** (`pytest -p no:randomly hymeko_rl/tests/test_contact_velocity.py`): **10 passed, 0.46 s.** Pure transport
  (stationary / translation / rotation-at-offset / sign / L-R ordering / rigid-body transport) + readers on a synthetic
  planar body (`geom_planar_velocity`, `geom_point_velocity`, `coin_twist`, mj_objectVelocity cross-check).
- **Regression on the edited core** (`hymeko_rl/tests/test_cooperative_grasp.py`): **12 passed** — the additive
  `final_tau`/`final_q_target` export does not perturb existing grasp/QP behaviour.
- **Integration + performance smoke:** the benchmark is the production-scale integration exercise (§3): 2 certified states,
  both ε scales, 3 validation levels, pinned positive control, replay + recert guards — deterministic, artefact + figure
  emitted.

## Performance

- **Peak RSS 0.23 GB** (well under the 16 GB cap); **wall 63 s** for the 2-state smoke (~30 s/state, dominated by the
  10 s env reconstruct + 3 s straddle acquire). No prior baseline (new code path). No regression discipline applies.

## New / removed dependencies

None. numpy 2.4.6 / mujoco 3.10.0 / torch 2.12.0 / matplotlib 3.11.0 / pytest 8.4.2 — all already present.

## §6.5 anti-patterns

None introduced. Reused `_contact_frames` sign convention, `_tip_contacts`, `internal_force_feasibility`, `release_pin`,
`pd_governed_torque` / `govern_torque` / `step_ablation` (no re-implementation, §6.1). Config as frozen dataclasses;
string classes are enum-like literal tags with a single dispatch. Complexity: all functions < CC 15 (§6.2 hard gate).
Lint: `ruff` clean on all changed files.

## Experiment provenance

- **Git SHA:** recorded at commit (this report's commit). Working tree: only the H2-session-1 files staged; pre-existing
  untracked artefacts (other experiments' `.pt`, `reports/2026-07-25-*`, `verification/`, `experiment_viewer_web/`) left
  untouched and **not** committed.
- **Env:** darwin (Apple Silicon), `.venv`, numpy 2.4.6, mujoco 3.10.0, torch 2.12.0 (CPU, `torch.set_num_threads(1)`).
- **Seeds:** certified subset s1=14250, s3=14750, s4=15000, s7=15750 (`_reconstruct` seed search, `tries=3`);
  validation RNG seed 20260726. μ_coast = 0.179 (frozen V2 mean).
- **State hashes (qpos+qvel, pre==post release):** s1 `16778d7df544b9e8`, s3 `ede88da6a0ca5673`.
- **Toolchain gap:** `pdflatex`/`lualatex` absent on this host, so `plan.pdf` could not be built; `plan.tex` (source),
  `plan.tikz`, `plan.mmd`, `plan.md` are provided. (The full engineering contract was supplied by the user; installing a
  LaTeX toolchain to produce the PDF was judged a ceremonial detour, not risk reduction — recorded here per §11.)

## Graphical output

`reports/2026-07-26-h2-bv-identification/bv_rate_limiter_mechanism.png` — the mechanism: per-joint one-step slew debt
`|raw_pd − prev_tau|` towering over the `step + kp·ε` absorption band (all joints absorbed), and the resulting B_v column
norms (all ≈ 0). Machine-readable measurements + all frozen conventions: `bv_identification.json`.

## Explicit statements

- **H2 QP NOT BUILT.** Acquisition NOT tuned. RL NOT started. O3 NOT resumed. V2/V4 physics & motion contracts NOT modified.
- A later positive CLF slack `s_V > 0` must never be called a stabilization PASS — no CLF/QP exists.
