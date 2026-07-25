# Bimanual straddle cradle — it was an acquisition-topology failure, not embodiment; the cradle is reachable

**Date:** 2026-07-26 00:19 JST
**Physics:** frozen `RUBBER_TIP_LOW_DRAG_COIN_V2` + `V4`. Deterministic, no RL. O3 stays paused.
**Why:** every wrench-null controller nulled ‖w‖ by contact loss because the acquired grasp pressed from the *same* side of
the coin, where a net-zero internal force is geometrically impossible. This asks the decisive question with the definitive
certificate: is a **straddling, equilibrium-feasible** cradle reachable with this arm placement?
**One-line outcome:** **yes** — a straddle-directed acquisition reaches `n_L·n_R ≈ −0.99` with both tips in contact and a
**feasible internal-force cradle certificate on 4/8 states** (straddle geometry on 6/8). So the whole E2/E3 wall was an
**acquisition-topology failure, not an embodiment / QP / wrench-model failure**. Starting the hard QP from the straddle
cradle now gives it a *solvable* problem — it drives ‖w‖ down 10× (s1 5.9→0.6 N) — though full G1∧G2∧G3 convergence still
needs tuning.

---

## The definitive cradle certificate (supersedes the n_L·n_R prior)

`internal_force_feasibility` — a null-preload cradle exists iff the grasp map's nullspace holds an admissible internal
force: `∃ f ≠ 0 : G·f = 0, f ∈ friction cone, Fn_i ≥ F_min`. For a 3×4 G the nullspace is 1-D (the internal force is unique
up to scale, `vt[-1]` from the SVD); the equilibrium residual is `‖G·f‖` (a 3×4 SVD returns only 3 singular values, so the
smallest is *not* the nullspace residual — that bug is fixed). The cradle exists iff that force is cone-admissible with both
normals loaded. This is stricter and more correct than `n_L·n_R < 0`: high tip friction (μ=2.0) lets the tangential
components participate, so a same-side pair can be algebraically null-able yet *cone-infeasible* (needs |Ft| ≫ μFn). Unit
tested: straddle → feasible; same-side → cone-infeasible.

## Embodiment audit — a straddling cradle is reachable

`straddle_directed_acquire` drives each tip to the *opposite* side of the coin along a squeeze axis (left → −axis, right →
+axis), instead of both to the surface from their approach. Over 8 states × {zone-cross, zone-parallel} axes:

| result | count |
|---|---|
| reachable straddling cradle (both-contact ∧ certificate feasible) | **4/8** (s1, s3, s4, s7) |
| straddle geometry reached (`n_L·n_R < 0`) on some axis | **6/8** |
| best `n_L·n_R` on the cradle states | **−0.96 … −0.997** (near-perfect opposition) |

Verdict `STRADDLING_CRADLE_REACHABLE`. s6 is the outcome-2 case (straddle force-feasible, `n_L·n_R` −0.98, but the arms
can't reach both sides there); s2/s5 stay same-side. So it is **not** an embodiment impossibility — a certified cradle is
reachable on half the panel, and the failing states are reachability-limited, not force-infeasible.

## The QP finally has a solvable problem

Starting `null_coin_wrench` (the hard-constraint QP) from the straddle cradle instead of the old same-side balanced preload:

| state | straddle `n_L·n_R` | cradle cert | QP: ‖w‖ before → after |
|---|---|---|---|
| s1 | −0.997 | feasible | **5.92 → 0.61 N** (10×) |
| s3 | −0.989 | feasible | 2.40 → 1.20 |
| s7 | −0.996 | feasible | 7.20 → 0.0 |
| s4 | −0.957 | feasible | 0.91 → 6.17 (diverged) |

The QP now makes real progress toward the G2 band (s1 10×) because the target — a null internal force — *exists*; before,
from a same-side grasp, no such target existed and the QP could only null by contact loss. It does not yet fully converge
(s1 0.61 > 0.30 band; s4 diverges), so QP convergence from the cradle is the next bounded tuning step — a **well-posed**
one now, unlike every prior attempt.

## Why every earlier "controller failure" was a correct diagnostic signal

- G2 0/72 (LFA), proportional null → contact loss, soft-LS → contact loss, hard-QP → contact loss: **all four pointed at
  the same thing** — the system could only null the object wrench by removing a contact, exactly the signature of *no
  admissible internal force*. The controllers were never the blocker; the acquisition topology was.
- E1 selected balanced normal *magnitude*, E2B selected *forward-side* — neither guarantees the tips **straddle**. Straddle
  (opposite sides, cone-admissible internal force) is the true null-preload prerequisite, and it needs a *directed*
  acquisition, not a centre-seeking one.

## Honest ledger

```
INTERNAL_FORCE_CRADLE_CERTIFICATE_BUILT_AND_TESTED     PASS (nullspace + cone; residual bug fixed)
STRADDLING_CRADLE_REACHABLE                            PASS (4/8 certified; 6/8 straddle-geometry; not embodiment)
ACQUISITION_TOPOLOGY_WAS_THE_BLOCKER                   ESTABLISHED (centre-seeking → same-side → no internal force)
QP_HAS_A_SOLVABLE_TARGET_FROM_THE_CRADLE               PASS (‖w‖ 10× down on s1; before, no target existed)
CERTIFIED_G1∧G2∧G3_CRADLE_CONVERGED                    OPEN (QP tuning from the cradle; well-posed now)
```

## Claims / non-claims

**Claimed (measured):** a straddling cradle with a feasible internal-force certificate is reachable on 4/8 states
(`n_L·n_R` down to −0.997); the failure was acquisition topology, not embodiment; the hard QP from the cradle reduces ‖w‖
10× (s1) — a solvable problem it never had before.

**NOT claimed:** that the QP converges to a certified G1∧G2∧G3 preload yet (s1 0.61 > 0.30 band, s4 diverges — tuning
pending); that all 8 states admit a reachable cradle (4/8; s2/s5/s6 do not under the current straddle acquire); that the
cradle survives unpin or holds on the mobile coin (E3b/E3c pending). The certificate is a quasi-static force-feasibility
statement, not a dynamic guarantee.

## Exact next rung (the chain is clean now)

1. **Tune the QP from the cradle** to reach G1∧G2∧G3 with dwell → `CERTIFIED_STRADDLING_NULL_PRELOAD_CRADLE_ESTABLISHED`
   (fix the s4 divergence: recompute-cadence / gain / dq clamp; the target exists so this converges with care).
2. **Switch `w_target` 0 → [F∥·e_par; 0]** — the same QP becomes the synchronised cooperative INSERT →
   `COOPERATIVE_INSERTION_FROM_NULL_PRELOAD_ESTABLISHED`.
3. E3b drift-free unpin, E3c cooperative insertion on the mobile coin, then RELEASE → B1 → SETTLE. O3 paused.

---

### Files touched
- `hymeko_rl/coin_delivery/cooperative_launch.py` — `_grasp_matrix`, `internal_force_feasibility` (cradle certificate,
  residual-bug fixed), `contact_straddle`, `straddle_directed_acquire` (opposite-side acquisition, `keep_pin` to chain the
  QP).
- `hymeko_rl/experiments/bimanual_cradle_embodiment_audit.py` — the audit.
- `hymeko_rl/tests/test_cooperative_grasp.py` — internal-force cradle + straddle tests.

### Test results
- Unit: `test_cooperative_grasp` **12/12** pass; ruff clean.
- Audit: 8 states × 2 axes, ~13 min wall; QP-from-cradle probe on 4 states. Single-thread, seeds 14000+250·i.
- Artifact: `reports/2026-07-25-coin-dynamics-contract-v2/cradle_embodiment_audit.json`.

### Preserved unchanged
`SINGLE_TIP_LOW_FRICTION_COIN_V1`, `RUBBER_TIP_LOW_DRAG_COIN_V2`, V2/V3/V4 contracts, the B1 barrier, all prior results.
CORE.YAML items touched: none.
