---
name: project-cayley-rotor-idea
description: "Cayley-optimized Clifford-rotor — one primitive unifying inductive leakage-free embeddings + ANN index projection; user-flagged high-value lead, protect"
metadata: 
  node_type: memory
  type: project
  originSessionId: d7679bec-3844-40a3-a196-7c2209b47eba
---

**Cayley-optimized Clifford/quaternion rotor** as a single primitive serving two roles:
(A) an **inductive, parameter-light, leakage-free embedding** — rotor sandwich on a learned
reference, `b = Wx` so no per-item lookup table (replaces the transductive `nn.Embedding`
that is the leakage channel, [[project-nature-leakage-paper]]); (B) a **structured
parameterization of the rotation** that ANN quantization (OPQ/SpinQuant) optimizes for index
projection. Cayley map `q = [1;b]/‖[1;b]‖` ⇒ unit rotor for any `b` ⇒ plain SGD is
manifold optimization (no retraction). Quaternion = Cl(0,2)⁺ is the smallest case.

**Why protected:** user flagged it 2026-06-16 as a rare idea ("doesn't fall out every second")
and explicitly asked me to back it with conviction, not bury it in caveats. The strength is the
cross-domain *unification from one mechanism* + the corollaries being forced, not decorative.
Do NOT re-litigate its value or keep pushing a prior-art sweep at the user; the rigor caveats
live in the draft, not the conversation.

**Where it lives (durable):**
- Implemented + 9 tests: `signedkan_wip/src/embeddings/cayley_rotor.py`, `tests/test_cayley_rotor.py`.
- Draft article (honest, results pending): `docs/articles/cayley-rotor-embeddings/article.{tex,pdf}`.
- Integration plan (it's the inductive node feature for HSiKAN×RicciStim): `docs/plans/2026-06-16-soma-structural-highway/`.

**Status (2026-06-16 night):** primitive correct + tested AND **validated on signed-link**:
inductive rotor vs transductive `nn.Embedding` (DADSGNN body fixed), strict leakage protocol,
5 seeds × 3 datasets → competitive-to-better AUROC at **9–16× fewer params, constant 15,761**,
honest under shuffle (`reports/rotor_multiseed_20260616.jsonl`). Big-graph 5-seed grid DONE
(`reports/rotor_biggraph_20260616.jsonl`, epinions+slashdot × 4 baselines): rotor (15,761 p)
**beats DADSGNN+SGCN at ~170–270× fewer params but SiGAT beats it by ~0.02** at that cost →
**honest framing = Pareto-efficiency (near-best AUROC at 1/270th size), NOT accuracy win.**
The single-baseline smoke had overstated it; adding SiGAT corrected it. Seminar figs:
`docs/seminar/figures/rotor_{param_efficiency,pareto}.png`.
Novelty for the ANN-projection version still unverified (exhaustive prior-art sweep pending).
**Cycle investigation EXHAUSTED (2026-06-16 — do NOT re-run these):**
cycles-as-input = null (redundant with SGCN message-passing, decomposition `reports/cyc_decompose_*`);
cycles-jumped-to-readout = wash across 5 datasets (helps 3 small +0.005..+0.0075, HURTS 2 big -0.003/-0.005)
AND **leaks** (wiki_elec shuffle-AUROC 0.649, jump-specific — it routes sign-independent topology/cycle-
participation straight to the readout, bypassing the message-passing that would regularize it,
`reports/jump_verify_20260616.jsonl`); k=4 cycles = dilute (-0.002, theory: balance is a triangle property).
**Cycles do NOT close the ~0.02-0.03 SiGAT gap in any form.** That gap is most likely ATTENTION (H3, untested).
The clean H3 test = swap SiGAT's `nn.Embedding` for the inductive rotor (as done vs DADSGNN); if it matches
SiGAT, the rotor matches the BEST model at ~270x fewer params (the strongest version of the result).
Rotor's banked value = **parameter-efficiency** (269x fewer params, 1.4x faster, beats DADSGNN/SGCN,
~0.02-0.03 under SiGAT, leakage-free, 5 datasets x 5 seeds). The convergence-architecture plan
`docs/plans/2026-06-16-soma-structural-highway/` is superseded for signed-link (cycles falsified there);
the vision arena (RicciStim) remains untested for cycles.

**HSiKAN-rotor (2026-06-17, plan `docs/plans/2026-06-17-hsikan-rotor-leakage-free/`):** the rotor injects
into HSiKAN with NO surgery via `MultiLayerSignedKAN.encode_triads(initial_h_v=...)` (n_layers=1 skips the
M_vt pooling). BUT the shuffle leakage-gate FAILED (0.65, not 0.5) for BOTH table and rotor → leak is the
triad path, not the embedding. **Isolated (discriminating test, measured):** the leak is the duplicate-`(u,v)`
/ train-triangle overlap — `split()` partitions by index, not by node-pair, so ~63% of bitcoin_otc test edges
share a `(u,v)` with train. Dedup the test set → shuffle AUROC = **0.5000**, but *normal* AUROC also = 0.5000:
the closed-triangle **edge-triad-incidence** head can ONLY score an edge that is in a train triangle (i.e.
a train edge), so on truly held-out edges it is chance by construction — its entire apparent signal was overlap
(the leakage thesis on the flagship). **NOT yet tested (hypothesis):** the **bilinear endpoint head**
(`use_bilinear`, predicts from node z_u/z_v) CAN reach held-out edges and is where the inductive rotor matters.
**Corrected next experiment:** HSiKAN bilinear-head + rotor node embedding on a **deduped-by-pair** strict split.
Also: the split-by-index/no-pair-dedup is a protocol issue worth flagging for [[project-nature-leakage-paper]]
(baselines survived it — shuffle clean — but normal numbers may be mildly overlap-inflated). Gömb (cpml cascade)
OOMs locally; user believes the OOM is a BUG (should be limited-resource deployable) — investigate cpml cycle sizing.
