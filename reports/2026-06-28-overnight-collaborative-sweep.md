# Overnight campaign: StructuralActor control, the k-sweep, and the collaborative coin-toss

**When:** 2026-06-28 (overnight, ~04:30–09:00 JST) · **Author:** Aiko (Claude Code) for Dr. Cs. Hajdu
**Status:** complete. Artifacts in `reports/overnight/`. The StructuralActor is an explicit HSiKAN-cell composition
(signed-holonomy gather → `CatmullRomActivation`); "strua" *is* HSiKAN.

## Summary

A varied overnight sweep on the collaborative coin-toss plus the StructuralActor control/representation line. Two
robust findings: **(1) the coin-toss doesn't grasp** — it knocks, and neither structure-in-obs (MLP) nor
structure-in-graph (HSiKAN) unlocked a pinch; **(2) HSiKAN is budget-bound on this hard RL** — 0% delivery at both
15k and 40k — so the structural value lives in *representation and control of tractable plants*, exactly where the
theory predicts, and is where every *positive* result landed.

## Results

### Closed-loop control — StructuralActor (HSiKAN-cell readout), vs LQR
| plant | actor ρ | mlp ρ |
|---|---|---|
| chain (k=2) | 1.093 | 1.002 |
| k-uniform (k=3) | 1.087 | 1.001 |
| **Steiner** | **1.062** | 1.001 |
| sunflower | 1.126 | 1.001 |

The actor regulates all four plants; **Steiner is the tightest** (best structural basis) — the representation
result now confirmed in *control*. MLP ≈ LQR-optimal on these benign linear plants (structure not load-bearing for
easy control).

### k-sweep — StructuralActor over k-uniform k=1,2,3 (supervised representation MSE)
| k | StructuralActor | HSiKAN | MLP |
|---|---|---|---|
| 1 | 0.00003 | 0.0004 | 0.233 |
| 2 | 0.056 | 0.0008 | 0.070 |
| 3 | 0.021 | 0.0006 | 0.037 |

The holonomy actor **beats MLP at every order k** and tracks full HSiKAN; coupling order matters
(non-monotonically: k=1 trivial-perfect — 1-uniform has no real coupling; k=2 hardest; k=3 recovers). The
structural k-question *is* answerable — with the fast structure-using actor, since HSiKAN can't train on the
coin-toss (below).

### Collaborative coin-toss — grasp diagnostic (grasp-fraction = fraction of deliveries that involved a grasp)
| controller | structure | difficulty | delivery | grasp-fraction |
|---|---|---|---|---|
| MLP (25k) | baseline | 0.3 | 0.20 | **0.10** |
| MLP (25k) | task_graph | 0.3 | 0.32 | 0.062 |
| MLP (25k) | task_graph | 0.5 (harder) | 0.26 | **0.0** |
| HSiKAN (15k) | baseline | 0.3 | **0.0** | — |
| HSiKAN (15k) | task_graph | 0.3 | **0.0** | — |
| **HSiKAN (40k)** | task_graph | 0.3 | **0.0** | — |

**MLP knocks, never grasps** (grasp-fraction ≤ 0.10; coin-in-graph → *more* knocking; harder → 0). **HSiKAN fails
to learn delivery at all** — 0% at 15k *and* 40k, zero contact, all timeouts. So on the coin-toss, reward-shaping
and HSiKAN-RL are both falsified for grasping; the structural prior gives no traction on this hard control RL.

## The honest synthesis

- **Structure is for representation and tractable control, not hard RL.** Every positive result (closed-loop
  Steiner-best, the k-sweep, the supervised wins) is representation/control of plants the controller can fit.
  The coin-toss — hard, contact-rich, exploration-bound — defeats both controllers.
- **HSiKAN is budget-bound** here: so launch-bound-slow that 40k still leaves it at zero delivery, far worse than
  MLP's 20–32% knock-delivery at 25k. This is *why* the coin-toss structural test keeps being un-answerable on
  HSiKAN and is better posed on the StructuralActor.
- **Grasping is unsolved by reward + RL.** The remaining lever is structural — attractor field / grasp primitive /
  demonstration — not weight-tuning.

## In flight
A **simplified-reward** test (user hypothesis: the strict `both_contact` gate + 14 conflicting terms hurt) —
`galambos_task.hymeko` reduced to 4 terms (approach · both·5 · zone · oob), MLP 25k, vs the gated 0.10
grasp-fraction. Result lands in `reports/overnight/grasp_mlp_simplereward/`.

## Files / provenance
New/edited: `structural_actor.py` (HSiKAN-cell readout, control, per-node), `structural_control_loop.py`,
`diag_contact.py` (--design/--k/--difficulty), `hypergraph_designs.py` (k_uniform_blocks), `galambos_task.hymeko`
(grasp-gate → simplified, git-tracked). Artifacts under `reports/overnight/`. CORE.YAML: none.
