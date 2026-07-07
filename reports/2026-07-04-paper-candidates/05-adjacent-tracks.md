# Paper candidates 5 — adjacent tracks (publishable, not Nature-shaped)

Three lines that are real contributions but belong in domain venues. Kept here so the dossier is
complete and so their gaps are on record.

---

## 5a. Cross-view verified IR (HyMeKo commuting square) — PUBLISH NOW

**Working title:** *Machine-Verified Cross-View Consistency over a Canonical Signed-Hypergraph IR*
**Venue:** IEEE T-SMC (per the 2026-06-29 assessment: "the flag"). Alt: MODELS / SoSyM.
**Status:** the only pillar that is both novel and *proven*. No scientific gaps — packaging only.

**Measured:** commuting square `Xᶠ(εᶠ(H)) = Q(H)` verified against the **real CLI emitters**
(Z3 + sympy) across 16 robotics + 2 systems-engineering fixtures (16×5 views), 22 tests; a
generalized faithful-extraction theorem; a forest/DAG global-invariant closure corollary; a
drift-prevention demo that proves the pairwise baseline drifts
(`reports/2026-06-28-cross-view-consistency.md`; memory: article already updated).

**Delineation (from the bounded 06-29 search):** MDE global model management (EMF/Eclipse-OCL,
Epsilon, megamodels, MLIR multi-dialect) enforces consistency *by convention or against a
metamodel*; this verifies the diagram against the real emitters. RoboChart is a different axis
(controller refinement). Nearest precursor is the user's own Hajdu & Hegyi 2025; HyperGraphOS
(2412.04923) must-cite, doesn't pre-empt.

**Missing to submission:** venue formatting; the T-SMC #5/#6 artifact queue
(memory `project-smc-paper-additions-queue`); a non-robotics witness is already in
(Z3 + systems-engineering fixture). This is author-time, not experiment-time.

---

## 5b. LiNGAM-SH — causal discovery on signed hypergraphs (Kato line)

**Working title:** *LiNGAM on Signed Hypergraphs: ICA Identifies the Star-Expansion Grouping*
**Venue:** NeurIPS / UAI / JMLR-shaped (theorem + estimator + demo). Nature-family only if the
robotics demo produces a genuinely surprising discovered structure.
**Status (2026-07-04, one day old):** idea + pipeline defined, nothing measured
(memory `project-kato-lingam-cip-hymeko`).

**The claim to prove:** B = star-expansion factorization; ICA identifiability extends to the
grouping — *the grouping recovery is the theorem*. Pipeline: SMC_02 → LiNGAM-SH → CIP → Kato arms.
PoC: LiNGAM over k-arm coin-toss rollouts → emitted `.hymeko`.

**Missing (everything):** (1) write the identifiability proof; (2) synthetic validation —
ground-truth signed hypergraph, sampled data, exact grouping recovery rate vs noise/sample-size;
(3) the coin-toss PoC; (4) delineation vs arXiv:2511.03831 (CAM→hypergraph — must-cite, judged
non-pre-empting in the recorded assessment; verify with a fresh read); (5) plan docs per §2 before
implementation.

---

## 5c. Declarative Topological Control (DTC)

**Working title:** *The Controller Is a Topology: Match Between Plant and Controller Hypergraphs*
**Venue:** CDC/L4DC-shaped today; more only if the "hard control" prediction lands.
**Status:** original program, evidence **low** (toy N=9, linear; Phase 3 only predicted).

**Measured:** matching law — `best_controller[plant] == plant` 9/9; coherence does *not* rank
control (memory `project-topology-control-matching-law`); match buys ~100× representation, ~3 %
easy control, 2–7 % constrained MPC (`reports/2026-06-27-topology-performance-map.md`,
`…-structured-control-phase2.md`, `…-structured-mpc-phase2b.md`, `docs/theory/declarative_topological_control.md`).

**Missing:** (1) the Phase-3 hard-control experiment where match is *predicted decisive* — the
only result that would upgrade the venue; (2) nonlinear plants; (3) delineation vs structured
optimal control (Lin–Fardad–Jovanović) and network controllability (Liu–Slotine) — named in the
06-29 search, section not yet written; (4) multi-seed discipline (current grid is deterministic
linear, so this is cheap).

---

## Explicitly not paper-candidates (recorded so they are not re-proposed)

- **FSR-prenorm vs transformer** (2.479 vs 2.570 3-seed TinyShakespeare, 3.1× slower/token):
  real signal, wrong scale for any claim. Becomes a *section* of a future architecture paper only
  after a ≥10× scale-up.
- **Soma-holonomy vision:** falsified on record (0.489 ≤ 0.519 ≪ 0.906); the readout-bound revival
  (position-aware pooling 0.945+) is an engineering note, not a paper.
- **Rotor joint-encoding:** falsified (joints never wrap); on record to prevent re-proposal.
