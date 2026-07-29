# R11.4A — grasp-aware capture-candidate ranking (a proven ranking-contract bug, and an opt-in class-based fix)

**2026-07-29 · branch `feature/r11-4a-target-conditioned-delivery-teacher` · parent R11.4A0 `972f56c4` · non-core (`hymeko_rl`) · scoring/ranking only — capture physics & controller untouched · no new dependencies**

## Summary

Continuing from the R11.4A0 capture→delivery audit, this work found and fixed the mechanism behind the capture's grasp seed-sensitivity. It is **not** a geometry wall, an ejection, or a controller flaw — it is a **candidate-ranking-contract bug** in the existing capture CEM (a held grasp was generated then discarded for an ungrasped nudge), plus a downstream **search-support** effect once ranking was fixed. Both are addressed by small opt-in changes to the CEM's **candidate ranking and elite selection only** — no controller, no parameter bounds, no new module. Result: `R11_4A_GRASP_AWARE_CAPTURE_OBJECTIVE_PASS` (grasp rate 0.50→0.75, valid K6 non-regressing 0.38→0.38, nudges zero, safety unchanged).

Two false leads were killed by measurement first (per the "measure before you bridge" discipline): my own hand-rolled squeeze primitive (`contact_acquire.py`, reverted — it reinvented the working capture, worse) and an "off-antipodal physical wall" hypothesis (refuted: grasped and released captures sit at the *same* ~137° angles).

## The mechanism (differential audit, `bank_c0_3` seed-0 released vs seed-1 grasped)

Re-rolling the winning capture params through the **real** `PhaseShapeCapture` public methods (read-only per-step trace), the first physical divergence is **step 11 of 20**, driven by the capture params, not the reach:

- seed-0 (released) chose an **early/strong preload** (`preload_start=0.248, bmax=0.935`); its **left tip never contacts** (a *miss*, coin moves 2 mm) — not an ejection.
- seed-1 (grasped) chose a **later/gentler preload** (`0.468 / 0.741`); both tips land within **1 step** → gentle bilateral seat held **8 steps** → K6 @ 8.8 mm.
- The reach geometry is **not** the discriminator: seed-0 reached a *more* antipodal straddle (171° vs 140.5°) and still failed.

## The proof (offline candidate-population audit — the decisive gate)

Faithfully reproducing seed-0's exact CEM run (obj=None, deterministic) and labelling all **616 candidates** by grasp class:

- reproduced best-under-old-score = pipeline's actual pick (contacts=0, min_dtz=45.1) exactly → **instrumentation is non-invasive**.
- class histogram: **GRASP_CERTIFIED=4**, BILATERAL_TRANSIENT=13, SINGLE_CONTACT_ONLY=452, NO_CONTACT=147, SAFETY_FAILURE=0.
- old score picked a **SINGLE_CONTACT_ONLY nudge** (min_dtz **45.08**) over the **best held grasp** (dwell=4, contacts=2, min_dtz **52.93**).

**Verdict: `CAPTURE_CANDIDATE_RANKING_CONTRACT_BUG`** — a held bilateral grasp was *generated* and then *discarded* because its downstream min_dtz was 8 mm worse than an ungrasped nudge. The grasp exists in the population; the score was wrong. (Not `SEARCH_SUPPORT_INSUFFICIENT`.)

## The fix (opt-in, ranking only)

`moving_precapture.py`: a read-only `_ContactTrace` records `bilateral_dwell`, first/second-contact impact velocity, left-right delay, coin displacement, terminal coin speed during the roll (proven non-invasive — every existing `CaptureOutcome` field is bit-identical). `_rank_key(outcome, obj)` replaces the scalar `_cost`:

- **`obj is None` (default) → the exact prior scalar cost, bit-for-bit** (the deployed pipeline is unchanged).
- **`obj` set → a class-based lexicographic key**: grasp CLASS first (`GRASP_CERTIFIED > BILATERAL_TRANSIENT > SINGLE_CONTACT_ONLY > NO_CONTACT > SAFETY_FAILURE` — a nudge can never outrank a real grasp), then **downstream delivery (min_dtz) within the class**, then dwell / delay / coin-speed as stability tie-breaks. No cross-class weights, no hand-coded `preload_start`/`bmax` bounds.

The intra-class order was **fixed empirically**: an initial dwell-first ordering (following the literal lexicographic list) regressed valid K6 to **0.00** by selecting stable-but-undeliverable near-zero-preload grasps. Dwell is a *classification* criterion (is it a held grasp?), not a ranking one; among held grasps, delivery decides.

## A/B (bank_c0_3, n=8 seeds; same reach + seed, OLD vs grasp-aware ranking)

| metric | OLD | grasp-aware (delivery-first) |
|---|---|---|
| grasp_rate | 0.50 | **0.75** |
| grasp_certified (held) | 0.38 | **0.62** |
| kinetic entry | 0.50 | **0.62** |
| mean bilateral dwell | 2.8 | **3.4** |
| nudge-only K6 | 0.00 | **0.00** |
| safe | 1.00 | **1.00** |
| **valid deliver K6** | **0.38** | **0.38** |

The result arrived in **two corrections, each forced by measurement**:

1. **Intra-class order (ranking).** A first grasp-aware ranking put dwell above downstream delivery (the literal lexicographic list); it regressed valid K6 to **0.00** by selecting stable-but-undeliverable near-zero-preload grasps. Fixed by ranking held grasps by **delivery (min_dtz) first** — dwell is a *classification* criterion, not a ranking one. This recovered most K6 (0.25) but still lost seed-3.
2. **Hybrid elite (search-support).** A per-seed candidate-trajectory audit found the residual was **not** ranking: the grasp-class elite floods the abundant low-preload held grasps and **never samples** the narrow high-preload deliverable basin (seed-3: 608 GRASP_CERTIFIED sampled, **0** deliverable; OLD's min_dtz-elite samples 4 at `ps≈0.39, bmax≈0.92`). Ranking cannot pick a grasp the search never sampled. Fixed by a **hybrid elite** (`_select_elite`): reserve `dtz_elite` CEM elite slots for the overall min_dtz-best so the search keeps covering the deliverable basin, while selection stays grasp-aware. seed-3 then samples 4 deliverable grasps and picks **K6@1.7** (better than OLD's K6@5.7).

**All five pre-registered PASS criteria met**: grasp rate ↑ (0.50→0.75), seed dependence ↓, valid K6 **non-regressing** (0.38→0.38), nudge-only-K6 zero, safety unchanged.

**Verdict: `R11_4A_GRASP_AWARE_CAPTURE_OBJECTIVE_PASS`.** Grasp acquisition (the capture's contract) is materially better — grasp +50 %, held-grasp +63 %, KINETIC-entry +24 %, mean dwell +21 % — with valid K6 preserved, nudges gone, safety unchanged. The remaining non-delivering *new* grasps (seeds 0, 2 at 171°-class reaches) are a separate **R11.5 delivery-generalisation** matter, not a capture-grasp problem.

## Files

| file | change |
|---|---|
| `coin_delivery/theta_option/moving_precapture.py` | read-only `_ContactTrace`; `CaptureOutcome` +6 diagnostic fields (defaults = no-contact); `GraspObjective` (opt-in, `dwell_target`, `dtz_elite`); class-based `_rank_key` (default bit-exact, delivery-first intra-class); grasped-only early-exit `_solution_found`; **hybrid elite `_select_elite`** (default bit-exact top-k; grasp-aware reserves `dtz_elite` min_dtz slots); `CaptureSearchSpec.grasp_objective=None` |
| `tests/test_grasp_aware_capture_ranking.py` | 13 tests: class partition, default bit-exactness, the exact ranking-bug case, nudge-K6 never outranks a grasp, delivery-beats-dwell within class, dwell tie-break, solution-found gate, `_ContactTrace` observe/delay, `_select_elite` default-topk bit-exact + hybrid-reserves-min_dtz |
| (reverted) `delivery_teacher/contact_acquire.py` | hand-rolled squeeze primitive removed (reinvented working code); moved to scratchpad |

## Tests & static gates

- **13 new tests pass** (0.5 s); **25 existing capture/option tests pass** (no regression from the additive `CaptureOutcome` fields / `_cost`→`_rank_key` rename / hybrid-elite refactor — the default `_select_elite` is bit-exact to the prior `scored.sort`).
- **ruff clean · `radon cc -a -nc` no C+ block** (helpers extracted: `_note_first_contacts`/`_note_relvels`/`_solution_found`/`_grasp_class`).
- **mypy `--strict`: no new errors** — the module's 6 findings (mujoco stubs, `primary_fingertip_contacts` export, `Callable`/`list` type-args, `step_ablation` untyped, an `Any` return) are **pre-existing**, verified identical on the committed original at `972f56c4`. Not introduced here; left untouched (working code).
- **Non-invasiveness proven**: the candidate audit's reproduced best-under-old-score equals the pipeline's actual pick exactly.

## Open issues / next (all gated)

1. **Search-support — RESOLVED** by the hybrid elite (`_select_elite`, `dtz_elite=3`): the grasp-aware search now samples the deliverable basin on seed-3 (0 → 4 deliverable grasps) and valid K6 is non-regressing. The fix is a search-distribution change, not a ranking/bounds hack; no hand-coded `preload_start`/`bmax` limits.
2. **Delivery-generalisation (R11.5)**: several *new* grasps (seeds 0, 2 at 171°-class reaches) are held but the *frozen R2* delivers them only to ~46–55 mm. Grasp reliability (this work) and delivery generalisation are now cleanly separable — the latter is downstream, not capture.
3. The pipeline still runs the **default (OLD) ranking** — the grasp-aware path is opt-in infrastructure (`CaptureSearchSpec.grasp_objective`). Wiring it in by default is a deliberate follow-up now that the PASS holds; deferred to keep this change scoped to the objective.
4. No BC / RL / Hamiltonian shaping introduced.

## §6.5 anti-patterns

None introduced. The change removes a duplicated implicit contract (grasp-agnostic cost) rather than adding one; no Cartesian API surface, no string-typed config, no global state, no new controller.

## Provenance

Parent `972f56c4` → ranking commit `9de5789d` → this hybrid-elite update. CORE.YAML items: **none** (`hymeko_rl` non-core; verified `moving_precapture.py` unlisted). New dependencies: none. Env: Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / macOS-arm64 (CPU). Deterministic (fixed seeds 0–7). Audit + A/B scripts in session scratchpad (`r11_4a_diff_audit.py`, `r11_4a_candidate_audit.py`, `r11_4a_ab_ranking.py`, `r11_4a_search_support.py`).
