# Structural Kolmogorov–Arnold theorem — 2026-05-09

A pure-theory paper formalising the empirical compounding ladder observed on signed-graph benchmarks (BA, OTC, SBM, Slashdot, Epinions) into a density theorem analogous to the classical Kolmogorov–Arnold representation theorem (KART).  Where KART decomposes any continuous $f : [0,1]^n \to \mathbb{R}$ into nested sums of single-variable continuous functions, the structural-KA analogue decomposes signed-graph prediction functions into a *softmax-mixed combination of graph-theoretic operators* (cycles, walks, attention pools).

## Motivation

Tonight's empirical evidence shows that adding structural primitives to HSiKAN's α-mixer monotonically lifts AUC on every dataset tested, and that the routing $\alpha_\kappa$ converges to a per-dataset signature (walks attend, cycles uniform on Slashdot; cycles dominate on SBM; balanced cycles on BA).  This pattern is begging for a theoretical underpinning — *what function class is HSiKAN's α-mixer + Highway-attention pool actually representing?*

The classical KART says: any continuous $f : [0,1]^n \to \mathbb{R}$ admits

$$f(x_1, \ldots, x_n) = \sum_{q=0}^{2n} \Phi_q\!\Bigl(\sum_{p=1}^{n} \phi_{q,p}(x_p)\Bigr)$$

with $2n+1$ outer functions and $n(2n+1)$ inner single-variable continuous functions.  The $\phi_{q,p}$ are generally pathological (often nowhere differentiable).

The structural-KA conjecture is that *for signed graphs*, an analogous representation holds with **graph-theoretic operators** as the inner functions, **convex-combination weights** as the outer aggregation, and the inner operators are *well-behaved* (continuous, finitely parameterised, even differentiable) — at the cost of an open-ended primitive set rather than fixed cardinality.

## The conjecture

**Definition** (admissible primitive class).  An *admissible signed-graph operator class* $\mathcal{F}$ is a set of maps $\Phi : G \times V \to \mathbb{R}^d$ where $G = (V, E, s)$ is a signed graph, each $\Phi$ is parameterised by finitely many real parameters, and $\Phi$ is continuous in those parameters.  Examples: cycle-arity-$k$ uniform pool $\Phi_{c,k}$, walk-length-$L$ attention pool $\Phi_{w,L}$, sparse Hamilton-product attention head $\Phi_{\rm attn}$.

**Conjecture** (structural Kolmogorov–Arnold density).  For any signed graph $G$ and any continuous task-relevant prediction function $f : E \to \mathbb{R}$ in some natural function class $\mathscr{C}(G)$, there exists a finite primitive subset $\mathcal{K} \subset \mathcal{F}$ and a softmax distribution $\boldsymbol\alpha$ over $\mathcal{K}$ such that

$$\Bigl\| f - \sum_{\Phi_\kappa \in \mathcal{K}} \alpha_\kappa \cdot \Phi_\kappa \Bigr\|_{\mathscr{C}(G)} < \varepsilon$$

for any $\varepsilon > 0$, in the limit of operator-class richness $|\mathcal{F}| \to \infty$.

Equivalently: the closure of $\{\sum_\kappa \alpha_\kappa \Phi_\kappa : \mathcal{K} \subseteq \mathcal{F}, \boldsymbol\alpha \in \Delta\}$ is dense in $\mathscr{C}(G)$.

## What needs to be proved

**Density (the main claim).**  Show that the convex-combination span of an admissible class is dense in the task-function class.  Two-step strategy:
1. Prove that $\mathcal{F}$ contains operators whose pairwise compositions distinguish all signed graphs up to a chosen equivalence (a "structural-completeness" property).
2. Show that finite convex combinations of distinguishing operators are dense in the relevant continuous-function space via a Stone–Weierstrass-type argument.

**Constructivity.**  Show that each $\Phi_\kappa$ is finitely parameterised and that gradient-descent training on $\boldsymbol\alpha$ converges to a near-optimal $\boldsymbol\alpha^*$ in polynomial steps in the worst case (or, more weakly, that the routing pattern observed empirically is reproducible).

**Refinement monotonicity.**  Show that for any $\mathcal{K} \subseteq \mathcal{K}'$, the function class representable over $\mathcal{K}'$ contains the function class over $\mathcal{K}$.  This is trivially true by setting $\alpha_{\kappa} = 0$ for $\kappa \in \mathcal{K}' \setminus \mathcal{K}$, but needs to be stated for the empirical compounding-ladder anchor.

**Constructive primitive set.**  Identify a *minimal* primitive set $\mathcal{F}_{\min}$ such that the density holds.  Empirical evidence: $\{c_2, c_3, c_4, c_5, w_2, w_3\}$ + Highway attention gives the best HSiKAN result on Slashdot at $0.9035 \pm .0044$.  Whether this is *necessary* or whether a smaller set suffices is an open theoretical question.

## Connection to classical KART

| | classical KART | structural-KA conjecture |
|---|---|---|
| Domain | $[0,1]^n$ | signed graphs $G = (V, E, s)$ |
| Atomic ops | single-variable continuous $\phi_{q,p}$ | finitely-parameterised graph operators $\Phi_\kappa$ |
| Outer aggregation | sum (unweighted) | softmax-normalised convex combination |
| Cardinality | fixed $2n+1$ | open-ended; depends on operator-class richness |
| Inner-function regularity | pathological (nowhere-diff) | continuous + differentiable in parameters |
| Constructivity | non-constructive | constructive (gradient descent on $\boldsymbol\alpha$) |
| Empirical realisation | KAN (Liu et al. 2024) | HSiKAN, this work |

The trade is clean: classical KART has bounded cardinality at the cost of pathological inner functions; structural-KA has well-behaved inner operators at the cost of open-ended cardinality.  The latter is the right trade-off when the data has structure (signed graph), because the structure imposes constraints that bound the effective primitive richness needed.

## Outline of the paper

1. **Introduction** — the empirical compounding ladder on HSiKAN, KART motivation, structural-KA framing
2. **Definitions** — admissible primitive class, convex-combination span, task-function class on signed graphs
3. **Conjecture and main results** — density theorem (assuming the Stone–Weierstrass argument goes through), refinement monotonicity (P1), constructivity bound
4. **Proof sketches** — full proofs deferred to appendix
5. **Empirical anchor** — HSiKAN's compounding ladder on BA, OTC, SBM, Slashdot, Epinions; routing patterns as constructive evidence
6. **Discussion** — implications for graph-learning architecture design, KART variants for other structured data (meshes, point clouds, time series)
7. **Conclusion + future work** — proving full density on bounded-degree signed graphs as the next concrete target

## Risk register

| risk | probability | mitigation |
|---|---|---|
| Density proof requires technical conditions (graph regularity, primitive class compactness) that don't hold for general signed graphs | high | scope to bounded-degree graphs initially; relax later |
| The conjecture is *wrong* for general primitive classes | medium | empirical compounding evidence suggests it holds for HSiKAN's primitive set; proof might need a specific operator-class structure |
| Stone–Weierstrass-style density requires a Banach-space structure that's awkward to define on signed graphs | medium | use representation theory of graph algebras (groupoid algebras of signed-incidence) |
| The paper is "interesting but unproven" — JMLR rejects without a real theorem | high | aim for at least one concrete density proof on a restricted class (e.g., trees, bounded-degree graphs); accept narrowness for tractability |
| Three months of math doesn't yield the proof | medium | scope a workshop-shape exposition first; submit to a venue tolerant of conjectural-empirical work |

## Order of operations

1. **Lit survey** — KART proofs (Sprecher; Lorentz), structural-data extensions (sequence transformers as KART; convolutional networks as KART). 1-2 weeks.
2. **Formal definitions** — admissible primitive class, task-function class on signed graphs, convex-combination span. 1 week.
3. **Refinement monotonicity proof** — easy (already constructive). 1 day.
4. **Density proof on a restricted class** — bounded-degree signed graphs, primitive class = cycles up to fixed arity. 1-3 months serious work.
5. **Constructivity bound** — gradient-descent convergence on $\boldsymbol\alpha$. 1-2 months.
6. **Paper draft** — 8 pp + 8 pp appendix. 1 month.

Total: ~4-6 months for a JMLR / NeurIPS theory submission.

## Acceptance for the plan

**Tier 1** (workshop): refinement monotonicity proven cleanly; conjecture stated; HSiKAN compounding ladder empirically anchors it.  Submit to NeurIPS workshop on structured prediction.

**Tier 2** (NeurIPS / ICML theory track): density proven on bounded-degree signed graphs for a constructive primitive class.  Substantial theoretical contribution.

**Tier 3** (JMLR): full conjecture proven in generality, with constructivity bound.  Definitive.

If after 4 months no version of density goes through, write a workshop note + open-problem statement.  Negative results in mathematical proof aren't useless — they sharpen the conjecture.

## What this plan does NOT do

- Doesn't propose new HSiKAN architectures.  This is theory only.
- Doesn't run new experiments.  Tonight's empirical anchor is sufficient.
- Doesn't compete with the SMC paper.  Different track entirely.
- Doesn't promise a proof.  Mathematical risk is real.

## Files this plan will touch when executed

- `paper/structural_ka_theorem/main.tex` — new directory, 8 pp + appendix
- `paper/structural_ka_theorem/refs.bib` — KART literature, structural-data extensions
- `docs/plans_structural_ka_theorem_2026_05_09.md` — this file (close out with proof status)

No code; pure theory.

## Connection to other plans

- This is the theoretical underpinning of every HSiKAN empirical result.  Every other plan (mesh, tabular, predictive coding) depends on the conjecture being plausible.
- The paper would *cite* the SMC paper as the empirical anchor.
- Fits the user's stated framing from earlier: *"this could be a new perspective on Kolmogorov–Arnold approximate universality theorem, but from graph-based approach."*  This plan is the realisation.
