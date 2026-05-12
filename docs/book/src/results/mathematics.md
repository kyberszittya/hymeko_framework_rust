# Mathematics (compact reference)

Definitions used when reading **AUC tables**, **thesis IV entropy** rows, and **cycle-based** models. Not a full textbook — pointers to in-repo derivations where they exist.

## Link prediction

**Graph:** directed signed edges \(E \subset V\times V\) with labels in \(\{-1,+1\}\) (or multi-class variants).  
**Task:** score unseen \((u,v)\) edges; rank by score; measure **ROC–AUC** against ground truth.

**ROC–AUC:** probability a random positive edge ranks above a random negative edge. Reported as **AUC** \(\in [0,1]\) in JSON / JSONL.

**Train / val / test split:** experiments must fix splits and seeds for comparability — see each artifact’s `seed` field.

## Paired comparisons (Thesis IV views suite)

For regulariser arms vs baseline, the aggregate table uses **paired** differences across the **same** seeds:

\[
\Delta = \frac{1}{n}\sum_{i=1}^{n}\bigl(\mathrm{acc}^{\mathrm{treat}}_i - \mathrm{acc}^{\mathrm{base}}_i\bigr)
\]

Units: **percentage points (pp)** of validation accuracy (see `RESULTS_VIEWS_SUITE.md` header).

**\(t\)-statistic:** paired two-sided \(t\) on the \(n\) deltas.  
**W / L / T:** wins / losses / ties of treatment vs baseline per seed.

## Spectral entropy on weights (sketch)

For a layer weight matrix \(W\), build an adjacency / Laplacian spectrum (implementation-specific). Normalised Laplacian eigenvalues \(\lambda_i \ge 0\) yield a discrete distribution \(p_i = \lambda_i / \sum_j \lambda_j\).

**Shannon entropy** of the spectrum:

\[
H = -\sum_i p_i \log p_i
\]

**Path B (normalised):** penalise / reward using \(H / \log_2(\mathrm{rank})\) style scaling (see thesis suite arms in `RESULTS_VIEWS_SUITE.md`).

**Path A (entropy target):** quadratic pull toward a target normalised entropy \(H^\*\) (e.g. \(0.5\)).

These are **single-time marginal** spectral statistics on weights — distinct from activation-side **matrix Rényi** constructions (see cross-layer note below).

## Matrix-based Rényi entropy & cross-layer MI (Path F / I)

For activation Gram matrices \(\hat K\) (trace‑1 PSD), the **collision entropy** (\(\alpha=2\)) satisfies

\[
H_2(\hat K) = -\log \mathrm{tr}(\hat K^2)
\]

**Joint** construction via Hadamard product and normalisation yields a joint kernel; **mutual information** style gaps motivate Path F / Path I regularisers. Full derivation and notation: **`reports/sanchez_giraldo_framework.md`** (also PDF next to it).

Path I **total correlation** multi-information across layers is documented in **`reports/phases_11_12_13_brief.md`** and the negative-result note **`reports/ph12_path_i_negative_result.md`**.

## Cycle enumeration & HSiKAN features

**Signed cycles:** HSiKAN consumes a multiset of **signed** simple cycles (and sometimes walks) per edge neighbourhood, capped by **top‑\(K\)** enumeration with optional **ABB** pruning.

**Arity:** cycle length \(k\) (and walk length for `w*` tuples). **Mixed** models learn convex weights \(\alpha_k\) over arity channels.

## Gömb outer shell (engineering view)

**Clifford algebra \(\mathrm{Cl}(0,1)\)** (or chosen signature) coefficient grids per **FIR bank**; outer shell applies batched **einsum** + scatter to map cycle features into embedding space. Details: `reports/2026-05-12-gomb-outer-perf.md`.

**Naming / design:** what “**orthogonal**” means around Gömb vs the HSiKAN×CPML×FIR **factorial** — see **[HymeKo-Gömb: “orthogonal” meanings](../research/gomb-orthogonal.md)**.

---

**Further reading (entropy programme):** repo root `RESULTS_VIEWS_SUITE.md` + `reports/thesis_iv_views_suite.tex` / PDFs in `reports/`.
