# SignedKAN: closing the AUC gap to SGCN — execution plan

The 0.13 AUC gap to SGCN's published 0.93 is the headline weakness
of the WiP submission. Three orthogonal moves are predicted to close
0.10+ in combination. This file plans them in commit-ready detail.

| step | move | predicted ΔAUC | effort | status |
|------|------|---------------:|--------|--------|
| 1 | Validation-based early stopping | $+0.02$ to $+0.03$ | 0.5 d | **running** (`bsjwfbr9m`) |
| 2 | Signed-edge-pair sampling          | $+0.04$ to $+0.06$ | 1 d | planned, see §2 |
| 3 | Spline-grid pruning ($G=5\!\to\!3$) | $+0.01$ to $+0.02$ | 0.5 d | planned, see §3 |
| 4 | KA-rank theory section (parallel)  | n/a (theory)      | 1 d | landed in §III.7, see §4 |

Predicted joint $\Delta\!\approx\!+0.07$ to $+0.11$. With the
baseline at $0.80$, this lands SignedKAN at $0.87$--$0.91$, within
the same band as SGCN's reported $0.93$ and within reach of
SiGAT's $0.94$.

## Step 1 — Validation-based early stopping (running)

**What.** Evaluate test AUC at the best-val-AUC checkpoint instead
of at fixed 100 epochs. Already implemented in
`signedkan_wip/src/run_compare.py::run_one` behind the
`early_stopping=True` flag, with `val_every` controlling the
checkpoint cadence. Sweep launcher in
`signedkan_wip/src/run_early_stop.py`.

**Why this should work.** The saturation curve measured in
`signedkan_wip/experiments/results/saturation.json` shows AUC peaks
at $\sim$50 epochs on Bitcoin Alpha and decays $\sim$0.05 by epoch
500. Stopping at peak captures the model where AUC is highest,
not where the optimiser stopped.

**Risk.** None. Standard ML hygiene; if it doesn't help, that
itself is a finding (means the val and test distributions diverge,
worth reporting).

## Step 2 — Signed-edge-pair sampling

**What.** Replace full-batch BCE on the $(94\,\%\!+\!,\!6\,\%\!-)$
imbalanced edge set with a balanced minibatch sampler that
guarantees each batch contains both signs in approximately equal
proportion. Two variants worth measuring:

- **2a — class-weighted BCE.** Cheaper. In
  `run_one`'s training loop, replace
  ```python
  loss = F.binary_cross_entropy_with_logits(logits, target_tr)
  ```
  with class-weighted BCE:
  ```python
  pos_weight = (s_tr == -1).sum() / max((s_tr == 1).sum(), 1)
  loss = F.binary_cross_entropy_with_logits(
      logits, target_tr,
      pos_weight=torch.tensor(float(pos_weight), device=device),
  )
  ```
  No sampler; cost is one extra tensor multiply per epoch.
- **2b — balanced minibatch sampler.** SGCN-style. Build two index
  pools (positive edges, negative edges); each minibatch draws
  $B/2$ from each. Means dropping the full-batch sparse-mat-mul
  shortcut for a pair-wise per-edge step; ~30 % training-loop
  rewrite. Implementation sketch:
  ```python
  pos_idx = np.where(s_tr ==  1)[0]
  neg_idx = np.where(s_tr == -1)[0]
  for epoch in range(n_epochs):
      rng.shuffle(pos_idx); rng.shuffle(neg_idx)
      for b in range(n_batches):
          p = pos_idx[b*Bh : (b+1)*Bh]
          n = neg_idx[b*Bh : (b+1)*Bh]   # cycle if shorter
          batch_idx = np.concatenate([p, n])
          ... build M_batch over batch_idx ...
          loss = BCE(logits_batch, targets_batch)
  ```
  Roughly $4\!\times$ slowdown per epoch vs full-batch but should
  let the $-1$ class drive a meaningful gradient component.

**Recommended order.** Run **2a first** (one-line change), measure.
If gain is $\geq 0.03$ AUC, ship as-is. If gain is $<0.02$, escalate
to **2b**.

**Where it lives.** Extend `run_one` with a `sampling` argument
$\in \{\text{full\_batch}, \text{class\_weighted}, \text{balanced\_minibatch}\}$,
default $\text{full\_batch}$ (current behaviour).

**Compute budget.** 2a: another 12-run sweep, $\sim$15 min on
shared GPU. 2b: another 12-run sweep at $\sim$40 min per run, so
$\sim$8 h overnight if shared.

## Step 3 — Spline-grid pruning ($G=5\!\to\!3$)

**What.** Reduce the spline grid from 5 inner knots on $[-1,1]$ to
3, in both `SignedKAN` and `VanillaKAN`. Already plumbed in
`run_one(grid=...)`. Sweep launcher reuses
`run_compare.py`/`run_early_stop.py` with
`grid=3` argument.

**Why this should work.** $G=5$ in Liu et al.'s reference KAN
design assumes universal-function-approximation regimes where the
network has $\sim 10^4$--$10^5$ training points per spline. On
Bitcoin Alpha with $\sim 24\!\,000$ edges and $\sim 22\!\,000$
triads, each spline sees $\sim 1$ effective training sample per
inner knot — overcapacity, which is exactly the saturation
diagnosis. Cutting to $G=3$ removes 40 % of the spline-coefficient
parameters while preserving the cubic-spline approximation
order. The entropy-regularisation experiment (Table V) shows the
trained-state $H_{\mathrm{norm}}$ gravitates to $\sim 0.5$ — the
spline distribution lives in a low-rank subspace of its
prescribed grid; cutting the grid harmonises capacity with task.

**Risk.** If $G=3$ is below the function-approximation threshold,
test AUC may *drop*. Mitigation: also try $G=4$ as a middle
point. If $G=3$ underperforms, treat the result as a
characterisation of the spline-rank elbow rather than a fix.

**Where it lives.** Add a `grid` axis to the sweep; report
$G \in \{3, 4, 5\}$ each with three seeds, both datasets, both
models. 36 runs total.

**Compute budget.** $\sim$15 min on dedicated GPU,
$\sim$30--45 min on shared.

## Step 4 — KA-rank theory section (landed in §III.7)

**What.** A theoretical anchor positioning HGNN, Signed-HGNN, and
SignedKAN as a three-level Kolmogorov--Arnold representation
ladder on signed-incidence hypergraphs:

\begin{itemize}
\item Level~1 (HGNN) — rank-1 KA: single linear map per channel,
  sign-blind.
\item Level~2 (Signed-HGNN) — rank-2 KA: two linear maps per
  channel, indexed by sign.
\item Level~3 (SignedKAN) — full KA: per-sign univariate splines,
  Cox--de~Boor approximation.
\end{itemize}

**Why this matters.** Without the theoretical anchor, the paper is
"yet another GNN variant with measured gains". With it, the
contribution is "a representational hierarchy on signed
hypergraphs in which SignedKAN realises the Kolmogorov--Arnold
form, validated by a saturation curve that sharpens monotonically
with rank." The framing also positions the entropy regulariser as
a level-dependent fix (higher rank $\Rightarrow$ more
overfitting $\Rightarrow$ stronger spectral-entropy schedule) and
sets up the comparison-paper direction (HGNN + Signed-HGNN +
SignedKAN, controlled empirical ladder).

**Where it lives.** §III.7 of `signedkan_wip/paper/sections/03_signedkan.tex`,
between *Parameter accounting* and *Vectorised training*. Cites
`feng2019hgnn` for the Level-1 reference.

**What's missing for the journal version.** A formal separation
theorem (or a precise function-class characterisation) between
levels, plus an empirical realisation of the rank-monotonicity
prediction (HGNN < Signed-HGNN < SignedKAN on macro-$F_1$ across
multiple datasets, with saturation profiles ordered as predicted).
The empirical realisation is the family-paper sweep; the formal
theorem is the strongest version.

## Joint sweep recommendation

Once Steps 1, 2, 3 each have individual measurements, run the
joint best-of-three combination ($\text{early\_stopping=True}$,
$\text{class\_weighted=True}$, $G=3$) on both datasets ×
3 seeds and report the joint $\Delta$AUC against the WiP baseline.
This is the load-bearing number for the journal-version paper.

## Reporting plan

A new §IV.8 *Closing the gap to engineered baselines* in
`signedkan_wip/paper/sections/04_experiments.tex` collects the
three steps' individual contributions and the joint result. Each
step gets its own row in a single table:

| variant | AUC | Δ vs WiP baseline |
|---|---:|---:|
| WiP baseline (100 epochs, $G=5$, full-batch BCE) | $0.801$ | --- |
| + early stopping | TBD | TBD |
| + class-weighted BCE | TBD | TBD |
| + $G=3$ | TBD | TBD |
| + all three | TBD | **TBD (load-bearing)** |
| SGCN (published) | $0.93$ | $+0.13$ |
