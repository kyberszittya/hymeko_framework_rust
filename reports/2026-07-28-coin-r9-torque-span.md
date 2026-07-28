# R9 teacher-torque-span diagnostic — the ~36 mm ceiling is a residual-BOUND limit (α ≈ 1.0 reaches), not a scaffold or basis limit

**2026-07-28 · branch `recovery/coin-r9-causal-residual-delivery` · worktree `hymeko_coin_r9_wt` · dev s1 (14250) · s4/s7 untouched · f1–f4 SEALED · teacher-free · NO RL this turn**

## Summary

The authority audit (`9894279b`) found no residual family reaching the ≤ 30 mm corridor at α ≤ 0.30 and *speculated* the wall was
"upstream / the scaffold." This learning-free diagnostic tests that speculation directly and **overturns it**. Two read-only
analyses:

- **Part A — authority ceiling (bound vs scaffold).** The audit CEM, re-run on the A0 per-joint basis at growing α up to full
  per-step override (α = 2.0), shows the reachability is a **threshold in α, not a plateau**: flat below ~0.5, then a sharp drop
  through the corridor at **α ≈ 1.0**. **α = 1.0 reaches ≤ 30 mm cleanly from BOTH healthy frontiers** (27.0 mm, 27.3 mm); α = 2.0
  reaches deeply (11.4 mm, 21.5 mm). The audit sampled α ≤ 0.30 — *below the threshold* — and read the local flatness as a
  ceiling.
- **Part B — the teacher-torque-span projection.** The delivering teacher's per-step action, decomposed against the clone's
  counterfactual action at the same states, lies **entirely in the A2 structured span** (orthogonal residual = 0.0 at every
  transport step) but with **magnitude far beyond the α = 0.15 bound** (mean |d|∞ = 0.72, max 1.07; 94 % of steps exceed α = 0.15,
  88 % exceed even α = 0.30). **A magnitude gap in an already-expressible direction.**

**Verdict: `RESIDUAL_BOUND_LIMITED_ALPHA≈1_REACHES_CORRIDOR` — `part_a = BOUND_LIMITED_LARGER_RESIDUAL_REACHES`, `part_b =
TEACHER_CORRECTION_MAGNITUDE_GAP_IN_A2_SPAN`.** The wall is the **α = 0.15 residual bound (≈ 6.7× too small)** — not the frozen
clone's trajectory (scaffold), not the action basis (direction is already spanned). A clean, safe, teacher-free delivering
residual over the *same* frozen clone at the *same* frontier **exists** at α ≈ 1.0 (the CEM found it). **RL is now justified — with
the current architecture and action basis and a larger residual bound α ≈ 1.0 — not a bigger network, structured basis, or
scaffold redesign.**

## Part A — authority ceiling sweep (family A1 = A0 per-joint basis; α = 2.0 ⇒ full per-step override; corridor ≤ 30 mm)

| frontier | α = 0.15 | α = 0.5 | α = 1.0 | α = 2.0 | α = 2.0 (h22 control) |
|---|---|---|---|---|---|
| **57.3 mm** | 36.9 ✗ | 35.5 ✗ | **27.0 ✓** | **11.4 ✓** | 19.4 ✓ |
| **53.84 mm** | 37.3 ✗ | 36.5 ✗ | **27.3 ✓** | **21.5 ✓** | 18.2 ✓ |

Every cell clean (`stall/reversal/clamp = 0/0/0`), safe, +v_par. The α = 0.15/0.30 numbers reproduce the audit (36.9/36.4). The
**horizon control** (h22 vs h14 at α = 2.0) still reaches — the corridor is not gated by the CEM horizon. Reachability is
sigmoidal in α: `36.9 → 35.5` (α 0.15→0.5, still short) then `35.5 → 27.0` (α 0.5→1.0, crosses). The audit's "saturation" was the
sub-threshold left tail.

## Part B — teacher-vs-clone action decomposition (16 transport steps of the delivering teacher, from the frozen KINETIC entry)

The delivering teacher (θ = entry+full_cem, K6 True, dtz_end 15.67 mm) reproduced **bit-for-bit** through the deploy kernel
(`max|Δcoin_trace| = 0`). At each KINETIC step, `d = a_teacher − a_clone` in the shared slew-normalised action space:

| quantity | value | reading |
|---|---|---|
| mean / max ‖d‖∞ | **0.72 / 1.07** | the per-joint correction the clone omits is ≈ half the full action range, up to the whole half-range |
| frac steps ‖d‖∞ > α (0.15) | **0.94** | almost every step needs a correction beyond the R2/R3-B bound |
| frac steps ‖d‖∞ > 2α (0.30) | **0.88** | beyond even A1's largest audited bound |
| mean / max A2-orthogonal residual | **0.0 / 0.0** | the correction is *fully* inside the structured basis span — no missing direction |

The correction grows monotonically as the teacher closes distance (t1 @75.8 mm: ‖d‖∞ 0.06 → t8 @56.3 mm: 0.83 → t9 @53.5 mm:
0.86), reflecting the teacher's sustained hard push (the squeeze≈0 drop-and-drive strategy from K0) that the α = 0.15 residual,
bounded around the clone's gentler grip-and-decay path, cannot reproduce.

## Reconciliation with the audit (honest correction)

The audit's **reachability table was correct** (0/10 cells at α ≤ 0.30). Its **interpretation was a hypothesis, now falsified**: it
extrapolated "no admissible perturbation over the frozen clone closes the last 6 mm ⇒ the scaffold is the wall." Part A shows a
perturbation over the *same* frozen clone at the *same* frontier **does** close it — at α ≈ 1.0. The scaffold trajectory is
recoverable; α = 0.15 was simply below the authority the teacher's correction requires (Part B quantifies it: ~0.72). This is why
the audit's rule mandated this diagnostic before any RL: the α ≤ 0.30 flatness was not the ceiling.

## Consequence for the next step (NOT run — for review)

RL is justified with the **minimal** change: keep the K2 clone, the per-step temporal residual (`kinetic_residual2`), and the
current 4-D action basis; **raise the residual bound to α ≈ 1.0** (a scalar in `ResidualBounds`). The update-zero identity and the
hybrid-boundary frontier curriculum carry over unchanged. Open risks to weigh: (a) a larger bound enlarges the exploration space —
the stall-aware champion + envelope (R3-B) must still gate cleanliness; (b) α ≈ 1.0 approaches full per-step override, so the
residual-over-clone framing weakens toward direct control — worth checking that the clone's GRU still contributes (ablate the clone
vs a zero baseline at α = 1.0). Both are RL-design questions for the greenlit run, not blockers to it.

## Method / safety

Learning-free, read-only: no policy trained, no state edited, no teacher in any deployed loop. Part A reuses the committed,
tested `authority_cem` (deterministic CEM, fixed RNG). Part B rolls the teacher through the frozen deploy kernel
(`velocity_rollout`) and queries the clone counterfactually (the query never affects the teacher trajectory; no teacher signal
enters the clone input). The A2 basis is now a single source (`a2_basis_matrix`) shared by the control (`a2_structured_u`) and the
projection. Safety: all Part A cells peak_qdot ≤ 3, coin speed ≤ 1.5.

## Files touched

| file | lines | role |
|---|---|---|
| `hymeko_rl/coin_delivery/theta_option/kinetic_torque_span.py` | +162 | `project_onto_span`, `TeacherThetaController` (θ-schedule adapter, bit-faithful to `rollout_primitive`, counterfactual clone logging), `decompose_teacher_vs_clone`, `summarize_decomposition` |
| `hymeko_rl/experiments/coin_kinetic_torque_span_diag.py` | +133 | driver: Part B projection + Part A ceiling sweep + verdict → JSON |
| `hymeko_rl/coin_delivery/theta_option/kinetic_authority.py` | +13/−8 | refactor: extract `a2_basis_matrix` as the single A2-basis source; `a2_structured_u` = `basis · coeffs` (behaviour-identical — verified) |
| `hymeko_rl/tests/test_coin_kinetic_contract.py` | +72 | +4 tests (a2-basis single-source/full-rank; `project_onto_span`; teacher-adapter == `rollout_primitive`; magnitude-gap-in-span) |
| `reports/2026-07-28-coin-r9-torque-span/torque_span.json` | — | full ceiling sweep + 60-step decomposition + verdict |

## Tests / static analysis

- **Full `test_coin_kinetic_contract.py` — 29 passed** (25 prior + 4 new). The 3 committed authority tests still pass → the
  `a2_basis_matrix` refactor is behaviour-identical.
- Key new tests: `TeacherThetaController` reproduces `rollout_primitive` bit-for-bit and the teacher delivers K6;
  `project_onto_span` full-rank → 0 residual / rank-deficient → exact out-of-span norm; the delivering correction is
  A2-orthogonal ≈ 0 with magnitude ≫ 2α.
- `ruff check` clean on all touched files; `radon cc -a` **A** (torque-span 2.76; authority still A). No new suppressions; no
  §6.5 anti-patterns. One flagged item: the 4-line R2-champion/frontier regen is duplicated once from the audit driver
  (`_regen_r2_frontiers`) — extract to a shared R2 helper on the third occurrence.

## Provenance

Git `9894279b` (audit commit; this report's files uncommitted at run time). Python 3.11.15 / mujoco 3.10.0 / numpy 2.4.6 / torch
2.12.0 / macOS-26.5.2-arm64 (Apple Silicon, CPU). Seeds: cradle 14250; R2 regen 0; CEM 20260728. θ_deliver = [0.0, 0.2714,
−0.057, 16.5378, 8.7978, 3.4604] (K0 entry+full_cem). Peak RSS 0.33 GB; wall 121 s. Deterministic (JSON reproducible).

## Status

`RESIDUAL_BOUND_LIMITED_ALPHA≈1_REACHES_CORRIDOR`. The ~36 mm ceiling is a residual-bound limit, not a scaffold or basis limit: a
clean, safe, teacher-free delivering residual over the frozen clone exists at α ≈ 1.0, and the required correction is already in the
current action span (magnitude gap, not direction gap). The audit's scaffold speculation is corrected. **No RL run.** No K6 → no
freeze, no tag; R2 stays champion of record. Committing the diagnostic on its own boundary. The greenlightable next step is RL with
α ≈ 1.0 on the current architecture/basis — deferred for review.
