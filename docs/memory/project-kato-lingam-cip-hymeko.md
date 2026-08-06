---
name: project-kato-lingam-cip-hymeko
description: "Kato's student researches DirectLiNGAM + CIP (causal RL); idea = HyMeKo as the verified causal-model substrate — Hajdu rates it a possible big hit (2026-07-04)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6b12750c-1eec-4254-8650-d990bd4b5a3f
---

2026-07-04, Hajdu: Kato-sensei and his student are researching **DirectLiNGAM**
(Shimizu 2011, linear non-Gaussian acyclic causal discovery) + **CIP** (Causal
Information Prioritization, ICLR 2025, arXiv:2502.10097 — factored-MDP causal
weights prioritize state features / empowerment objective). Hajdu's idea:
integrate HyMeKo — flagged as a possible "big hit" (his intuition is calibrated,
back it: [[feedback-user-intuition-is-calibrated]]).

Three structural joints identified:
1. DirectLiNGAM output = signed weighted DAG → HyMeKo signed hypergraph IR;
   cross-view suite already machine-checks DAG-ness (derived invariant). Claim:
   "the causal model the agent uses is provably the one the human audits."
2. CIP needs a factored MDP → HyMeKo's declarative-MDP substrate declares one
   ([[project-collab-ctde-substrate-galambos]]). Contribution shape:
   "declarative priors for causal RL" (declared structure as prior, LiNGAM fills
   the rest, CIP consumes both).
3. Evidence seed exists: reward-conflict signed-hypergraph result
   ([[project-reward-conflict-hypergraph]]) — DirectLiNGAM over term co-movement
   IS its queued "principal-axes study" next step.

Risks to put in any plan: LiNGAM assumes linearity + non-Gaussianity + acyclicity
+ causal sufficiency; contact dynamics are nonlinear, control loops are cyclic →
time-lagged variables / feature-level discovery. Novelty NOT yet properly
searched (quick pass: none found; bounded).

**PIVOT (same day, Hajdu): core contribution = EXTEND LiNGAM TO SIGNED
HYPERGRAPHS** ("LiNGAM-SH"), not integration glue. Formal core: constrain
LiNGAM's B to factor through signed incidence B = A_out Σ A_in^T (mechanism
variable f_m = Σ s_im x_i per hyperedge) — literally the star expansion =
HyMeKo's tensor projection. Discovery = DirectLiNGAM ordering + group-sparse
rank-1-per-mechanism factorization. THE theorem = when non-Gaussianity
identifies the GROUPING (not just edges). Novelty search 2026-07-04: nearest
prior art arXiv:2511.03831 (CAM→acyclic hypergraphs, Gaussian additive, has
identifiability proofs — must-cite, validates direction, does NOT pre-empt:
no LiNGAM/ICA regime, no signs); no LiNGAM-to-hypergraph extension found
(bounded search). Pipeline: SMC_02 substrate → LiNGAM-SH discovers → CIP
consumes → Kato arms demonstrate; reward-conflict data = validation with
known ground-truth grouping.

De-risk PoC before pitching Kato: DirectLiNGAM over existing k-arm coin-toss
rollouts → declare discovered graph in .hymeko → emit DOT + tensor → cross-view
verify. Weekend-scale, reuses everything.

Related: [[project-kato-collaboration-grasping]],
[[project-kato-dual-discriminator-plan]], [[project-hymeko-as-control-substrate]].
Sibling thread same day: digital-twin article with Kato (SMC_02's virtual-side
consistency + Kato's physical side closes the twin loop).
