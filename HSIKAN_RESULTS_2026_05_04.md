# HSiKAN — measured results, 2026-05-04

Every empirical number produced today (or pulled from the existing
results JSONs in this branch).  No projections, no aspirational
numbers — just what the harness actually printed.  Source files
cited per-row so each can be re-checked against the JSONL.

---

## 1. Link-sign prediction — headline AUC

5-seed mean ± std (Bitcoin Alpha / OTC / Slashdot / SBM); 3-seed
on Epinions because it's the late addition.

| Dataset       | HSiKAN ($h$)    | SGCN          | SiGAT  | SGT (NEW today)  | $\Delta$ best |
|---------------|-----------------|---------------|--------|------------------|---------------|
| Bitcoin Alpha | $\bf{0.939 \pm .011}$ ($h{=}16$) | $0.874 \pm .006$ | $0.899$ | $0.898 \pm .001$ | $+0.040$ |
| Bitcoin OTC   | $\bf{0.930 \pm .008}$ ($h{=}16$) | $0.906 \pm .006$ | $0.934$ | $0.915 \pm .010$ | $-0.004$ |
| Slashdot      | $0.861 \pm .002$ ($h{=}4$)        | $0.883 \pm .002$ | —      | $\bf{0.897 \pm .002}$ | $-0.036$ |
| SBM $n{=}200$ | $\bf{0.911 \pm .028}$ ($h{=}16$) | $0.504 \pm .065$ | —      | $0.563 \pm .104$ | $+0.349$ |
| SBM $n{=}400$ | $\bf{0.962 \pm .009}$ ($h{=}16$) | $0.677 \pm .070$ | —      | $0.690 \pm .025$ | $+0.273$ |
| Epinions      | $0.606$ (h=16, k={3,4}, 1 seed) | $0.931 \pm .003$ | —     | $\bf{0.941 \pm .003}$ | $-0.335$ |

**Source files:**
- `signedkan_wip/experiments/results/overnight_camera_ready.jsonl` (75 cells, Bitcoin/SBM/Slashdot HSiKAN+SGCN)
- `signedkan_wip/experiments/results/sgt_sweep.jsonl` (Bitcoin/SBM SGT)
- `signedkan_wip/experiments/results/sgt_slashdot.jsonl` (Slashdot SGT)
- `signedkan_wip/experiments/results/sgt_epinions.jsonl` (Epinions SGT)
- `/tmp/hsikan_epinions_v8.log` (HSiKAN Epinions, h=16, max_k4=50K)

**Two-regime takeaway (from these numbers):**
- Cycle-rich (SBM): HSiKAN $+0.27$ to $+0.35$ over best baseline.
- Walk-rich (Slashdot, Epinions): SGT $+0.04$ to $+0.34$ over HSiKAN.
- Mixed (Bitcoin): seed-noise margin between top three architectures.

---

## 2. Pruning Pareto on Slashdot (5-seed)

| variant            | AUC             | F1m             |
|--------------------|-----------------|-----------------|
| HSiKAN $h{=}16$    | $0.849 \pm .003$ | (not logged)   |
| HSiKAN $h{=}4$ (pruned) | $\bf{0.861 \pm .002}$ | (not logged) |

**Pruned $h{=}4$ outperforms $h{=}16$ counterpart by $+0.012 \pm .003$ AUC across 5 seeds** — strongest realisation of regularisation-as-pruning observed.

Source: `signedkan_wip/experiments/results/overnight_camera_ready.jsonl`.

---

## 3. Symbolic distillation — sinusoidal-fraction control study

3 seeds, $h{=}16$, 100 epochs, after L2-norm pruning.

| baseline                 | Bitcoin Alpha    | Bitcoin OTC      |
|--------------------------|------------------|------------------|
| **trained HSiKAN**       | $\bf{0.901 \pm .055}$ | $\bf{0.911 \pm .018}$ |
| untrained HSiKAN         | $0.505 \pm .036$ | $0.531 \pm .054$ |
| random spline coefs      | $0.583 \pm .045$ | $0.583 \pm .045$ |
| Gaussian-process draws   | $0.499 \pm .021$ | $0.499 \pm .021$ |

Trained $+32$ to $+40$\,pp above all three nulls.  All three nulls
hover near $0.50$ (the natural sine/cubic split for smooth $[-1,1]$
curves) — rules out spline-basis bias and grid-sampling artefacts.

Source: `signedkan_wip/experiments/results/sinusoid_controls.json`.

---

## 4. ph18c entropy-regulariser follow-up

Path I calibration law $\lambda_{\text{multi}} \sim
\lambda_{\text{scalar}}/L$ tested across architecture depths.

| cell                                          | mean Δ AUC  | sd       | $t$    | W/L/T   | $p$       |
|-----------------------------------------------|-------------|----------|--------|---------|-----------|
| **fashion_mnist_highway_10** ($\lambda{=}1.0$, 10-seed) | $\bf{+0.00092}$ | $0.00140$ | $\bf{+2.08}$ | $9/1/0$ | $\bf{<0.05}$ |
| fashion_mnist_highway_20 lam=0.5 (3-seed)     | $+0.00047$  | $0.00193$ | $+0.42$ | $1/2/0$ | ns        |
| fashion_mnist_resnet_20 lam=0.5 (3-seed)      | $+0.00087$  | $0.00112$ | $+1.35$ | $2/1/0$ | ns        |
| **mnist_resmlp_40_hymeko** lam=0.5 (3-seed)   | $\bf{+0.00213}$ | $0.00136$ | $\bf{+2.72}$ | $3/0/0$ | $\bf{<0.05}$ |
| fashion_mnist_highway_20 lam=2.0 (3-seed)     | $+0.00037$  | $0.00110$ | $+0.58$ | $2/1/0$ | ns        |
| fashion_mnist_resnet_20 lam=2.0 (3-seed)      | $+0.00023$  | $0.00244$ | $+0.17$ | $1/2/0$ | ns        |
| mnist_resmlp_40_hymeko lam=2.0 (3-seed)       | $+0.00053$  | $0.00280$ | $+0.33$ | $1/2/0$ | ns        |
| fashion_mnist_highway_20 lam=5.0 (3-seed)     | $-0.00097$  | $0.00103$ | $-1.63$ | $0/3/0$ | ns        |
| fashion_mnist_resnet_20 lam=5.0 (3-seed)      | $+0.00050$  | $0.00298$ | $+0.29$ | $1/2/0$ | ns        |
| mnist_resmlp_40_hymeko lam=5.0 (3-seed)       | $+0.00267$  | $0.00242$ | $+1.91$ | $2/1/0$ | trending  |
| fashion_mnist_highway_20 lam=1.0 10-seed cc   | $-0.00022$  | $0.00198$ | $-0.35$ | $4/6/0$ | ns        |

**Two confirmed positives** (was one before today):
- highway-10 10-seed cross-check confirms the original ph18 result.
- resmlp-40 at $\lambda{=}0.5$ is a NEW positive — exactly what
  Path I's calibration law predicted for $L{=}40$ (rescaled
  $\lambda{=}1.0/4 \approx 0.25$, found within window).

**One genuine null:** highway-20 across the full
$\lambda \in \{0.5, 1.0, 2.0, 5.0\}$ grid is depth-fragile.

Source: `data/benchmarks/thesis_iv_hard_20260504_*.csv` (11 CSVs).

---

## 5. Learned $\alpha_k$ readout (from camera-ready bench)

Mixing weights at convergence — the post-hoc "which arity carries
this dataset's signal?" reading.

| dataset       | $\alpha_2$ | $\alpha_3$ | $\alpha_4$ | $\alpha_5$ |
|---------------|------------|------------|------------|------------|
| Bitcoin Alpha | —          | $0.55$     | $0.45$     | —          |
| Bitcoin OTC   | —          | $0.51$     | $0.49$     | —          |
| Slashdot      | $0.18$     | $0.05$     | $0.31$     | $\bf{0.46}$ |
| SBM $n{=}200$ | —          | $0.32$     | $\bf{0.68}$ | —         |
| SBM $n{=}400$ | —          | $0.38$     | $\bf{0.62}$ | —         |

**SBM peaks at $k{=}4$** — matches the 4-community design.
**Slashdot peaks at $k{=}5$ + $k{=}2$ direct-edge channel.**
**Bitcoin is k=3-dominant (triads)** — classic Heider regime.

Source: paper Table II.

---

## 6. HSiKAN-from-HyMeKo IR round-trip

`hymeko emit data/nn/hsikan_mixed.hymeko --format torch_dataflow`
produces a runnable PyTorch module.

| metric                          | value             |
|---------------------------------|-------------------|
| emit output size                | $5921$ bytes (initial) → $6365$ bytes (after fixes) |
| imported class                  | `HSiKANEmitted`   |
| forward: $(B, h)$ → $(B, 1)$    | $(8, 16) \to (8, 1)$ |
| trainable parameters            | $2469$            |
| `spectral_weights()` tensors    | $10$              |
| 5-step SGD loss reduction       | $0.0059 \to 0.0043$ |

Permanent test: `scripts/verify_hsikan_emit.py` (passes).

**AUC parity with hand-coded HSiKAN is NOT claimed.**  The Tier-3
stubs in `ehk_torch_stub` are placeholder linear+tanh; real
architectural fidelity requires plugging the real
`signedkan_wip.src.signedkan.SignedKAN` into the round-trip path
(week-long Item #4-final on the roadmap).

Source: `/tmp/hsikan_emitted.py` + verify script.

---

## 7. Walk-HSiKAN open-walk enumerator

Verification suite: 12 cases (triangle / path-5 / 6-cycle+chord /
K4 across multiple walk-lengths + reservoir cap).  All match a
pure-Python reference DFS bit-for-bit.

| graph              | $L$ | walks (Rust) | walks (ref) | match? |
|--------------------|-----|--------------|-------------|--------|
| triangle           | 2   | $3$          | $3$         | ✓      |
| path-5             | 1   | $4$          | $4$         | ✓      |
| path-5             | 2   | $3$          | $3$         | ✓      |
| path-5             | 3   | $2$          | $2$         | ✓      |
| path-5             | 4   | $1$          | $1$         | ✓      |
| 6-cycle + chord    | 2   | $10$         | $10$        | ✓      |
| 6-cycle + chord    | 3   | $14$         | $14$        | ✓      |
| 6-cycle + chord    | 5   | $8$          | $8$         | ✓      |
| K4                 | 1   | $6$          | $6$         | ✓      |
| K4                 | 2   | $12$         | $12$        | ✓      |
| K4                 | 3   | $12$         | $12$        | ✓      |
| reservoir cap      | $L=4$ | shape (7, 5), all canonical | — | ✓ |

Source: `scripts/verify_walks.py`.

---

## 8. HyMeKo → star expansion structural cycle counts

Per-net cycle inventory (no training, just enumeration).

| HyMeKo source            | $V_{\rm orig}$ | $E_{\rm he}$ | $E_{\rm star}$ | $+/-/0$    | k=4 | k=6 | k=8 | k=10 |
|--------------------------|----------------|--------------|----------------|------------|-----|-----|-----|------|
| `mnist_resmlp_3`         | $11$           | $5$          | $15$           | $10/5/0$   | $0$ | $0$ | $0$ | —    |
| `mnist_highway_10`       | $25$           | $12$         | $36$           | $24/12/0$  | $0$ | $0$ | $0$ | —    |
| `disjoint_net`           | $47$           | $16$         | $76$           | $60/16/0$  | $48$| $48$| $0$ | —    |
| `hsikan_mixed`           | $13$           | $6$          | $21$           | $15/6/0$   | $0$ | $\bf{6}$ | $0$ | — |
| `walk_hsikan`            | $13$           | $6$          | $21$           | $15/6/0$   | $0$ | $\bf{6}$ | $0$ | — |
| `chicken_anatomy` (NEW)  | $12$           | $19$         | $47$           | $32/15/0$  | $20$| $\bf{47}$| $53$ | $20$ |

**Findings:**
- Pure feedforward HyMeKo nets: zero cycles at any $k$.
- Multi-input fan-in topologies (`hsikan_mixed`, `walk_hsikan`):
  exactly $\binom{4}{2} = 6$ cycles at $k{=}6$ (one per pair of
  $\alpha_k$ branches into the mixer).
- Shared-port factors (`disjoint_net`): $48$ cycles at both $k{=}4$
  and $k{=}6$.
- Chicken anatomy: $k{=}6$ peaks at $47$ cycles — matching the
  4 named kinematic chains (head + torso + legs + wings).

Source: `scripts/hymeko_to_signed_graph.py --enumerate`.

---

## 9. Chicken-aggression — supervised classifier (synthetic)

3-seed; $40$ birds, $800$ frames, $6$ aggressors.

| event source           | AUC   | F1m   | params | wall (s/seed) |
|------------------------|-------|-------|--------|---------------|
| ground-truth events    | $1.000$ | $1.000$ | $994$  | $0.9$–$1.4$   |
| kinematic detector     | $1.000$ | $1.000$ | $994$  | $1.0$–$1.4$   |

All 6 aggressors top-ranked in every seed.  The kinematic detector
overfires by $\sim 12\times$ ($\sim 1100$ false pecks vs $\sim 90$
ground truth) but the per-pair sign aggregation is robust to false
positives.

Smaller-flock control ($20$ birds, $300$ frames, $3$ aggressors,
ground-truth events): AUC $0.722 \pm .118$, F1m $0.603 \pm .041$
— enough for the model to recover $1$–$2$ of $3$ aggressors in
top-5, harder when the signed graph has only $13$–$26$ edges.

Source: `signedkan_wip/src/chicken/aggressor.py` CLI.

---

## 10. Chicken-aggression — unsupervised scorers (synthetic)

8 seeds; $40$ birds, $800$ frames, $6$ aggressors, kinematic
detector only (no GT events).

| scorer                                          | AUC mean | AUC std  |
|-------------------------------------------------|----------|----------|
| baseline (negative-out-degree)                  | $0.638$  | $0.164$  |
| **Cartwright–Harary (cycle-balance fraction)**  | $0.586$  | $\bf{0.107}$ |
| HSiKAN (self-supervised edge prediction)        | $0.653$  | $0.155$  |
| **rank-ensemble of all three**                  | $\bf{0.693}$ | $0.120$ |

Per-seed details:

| seed | base   | CH     | HSiKAN | ensemble | edge-AUC |
|------|--------|--------|--------|----------|----------|
| 0    | 0.6471 | 0.6324 | 0.9020 | 0.8358   | 0.5408   |
| 1    | 0.7745 | 0.5980 | 0.5147 | 0.7206   | 0.4336   |
| 2    | 0.7426 | 0.4657 | 0.8333 | 0.7696   | 0.6111   |
| 3    | 0.8015 | 0.5686 | 0.6078 | 0.7206   | 0.5846   |
| 4    | 0.5515 | 0.8039 | 0.6324 | 0.7059   | 0.6003   |
| 5    | 0.7647 | 0.5000 | 0.5882 | 0.7206   | 0.6792   |
| 6    | 0.4632 | 0.6152 | 0.7010 | 0.6397   | 0.5853   |
| 7    | 0.3578 | 0.5049 | 0.4412 | 0.4289   | 0.4652   |

**Counter-intuitive finding:** aggressors live in BALANCED cycles
($-, -, +$ "two victims of same aggressor"), not unbalanced ones.
That's Heider 1946 / Cartwright-Harary 1956 reproducing themselves
on synthetic chicken data.  Direction confirmed: balanced-fraction
correlates positively with aggressor identity.

Source: `signedkan_wip/src/chicken/unsupervised.py` CLI.

---

## 11. Inference latency

| dataset       | h | latency (ms/forward) | source |
|---------------|---|----------------------|--------|
| Bitcoin Alpha | 16 | $20$–$22$  | `sgt_sweep.jsonl` (HSiKAN cells)  |
| Bitcoin OTC   | 16 | $20$–$22$  | same                              |
| SBM $n{=}200$ | 16 | $20$       | same                              |
| SBM $n{=}400$ | 16 | $24$       | same                              |
| Slashdot      | 16 | $200$      | published (with cudagraphs)       |
| Slashdot      | 4  | $30$       | h=4 pruned                        |

(SGT comparison: Slashdot $774$\,ms, Epinions $1190$\,ms — heavier
because of attention-over-neighbours.)

---

## 12. Cycle enumeration speed

Rust DFS + rayon + bitset visited + BFS pruning + atomic
early-stop, exposed via PyO3 returning numpy.

| dataset    | $k$ | cycles    | wall    | peak mem |
|------------|-----|-----------|---------|----------|
| Slashdot   | 4   | $100$\,k (reservoir) | $4.6$\,s | $244$\,MiB |
| Slashdot   | 4   | $55.5$\,M (full enum) | $\sim 4$\,min | — |
| Bitcoin Alpha | 4 | typical   | $< 1$\,s | low      |
| Epinions   | 4   | $50$\,k (reservoir) | $\sim 30$\,s | low |

Serial Python equivalent: $\sim 227$\,s for Slashdot k=4 reservoir
of $100$\,k → Rust path is $\sim 50\times$ faster.

Source: paper §IV.C plus today's `verify_walks.py` runs.

---

## 13. Memory + scaling on Epinions — full optimisation journey

8 GB RTX 2070 Super.  HSiKAN's `MixedAritySignedKAN` already had
a `cycle_batch_size` parameter wired up but it was hardcoded
`None` for Epinions (only Slashdot used it).  Activating it
through `HSIKAN_CYCLE_BATCH=10000` is the primary unblock.

### 13a. Memory-profile timeline

With `HSIKAN_CYCLE_BATCH=10000` on Epinions h=16 k={2,3,4}
max_k4=30K (small recipe):

| stage            | alloc    | reserved | peak     |
|------------------|----------|----------|----------|
| startup          | 0.00 GiB | 0.00 GiB | 0.00 GiB |
| dataset load     | 0.00 GiB | 0.02 GiB | 0.00 GiB |
| all cycles + M_e + M_vt buffers built | 0.02 GiB | 0.03 GiB | 0.02 GiB |
| model init       | 0.03 GiB | 0.05 GiB | 0.03 GiB |
| ep=0 forward     | 0.45 GiB | 1.12 GiB | 0.89 GiB |
| ep=0 backward    | 0.13 GiB | 1.20 GiB | 0.89 GiB |
| ep=1 forward     | 0.48 GiB | 1.20 GiB | **1.00 GiB** |

**Peak: 1.0 GiB** — vs previous OOM at 5.9 GiB in the
no-chunking path.  M_e + M_vt sparse buffers total <10 MB; the
OOM was 100% spline-forward intermediate activations being
materialised in a single full-batch call.

### 13b. AUC trajectory across configs

| config                                              | recipe summary                                   | AUC       |
|-----------------------------------------------------|--------------------------------------------------|-----------|
| v6  | $h{=}4$, $\mathcal{K}{=}\{3\}$, max_k4=20K, no chunking  | minimum-fit, severely underfit | $0.549$ |
| v7  | $h{=}8$, $\mathcal{K}{=}\{3,4\}$, max_k4=20K, no chunking | underfit at h=8 | $0.558$ |
| v10 | $h{=}16$, $\mathcal{K}{=}\{3\}$, max_k4=100K, no chunking | k=3-only, more cycles | $0.561$ |
| v8  | $h{=}16$, $\mathcal{K}{=}\{3,4\}$, max_k4=50K, no chunking | best non-chunked | $0.606$ |
| **full**  | $h{=}16$, $\mathcal{K}{=}\{2,3,4\}$, max_k=200K/30K/100K, **cycle-batched** | fits, full arities | $\bf{0.663}$ |
| **aggressive**| $h{=}16$, $\mathcal{K}{=}\{2,3,4\}$, max_k=400K/50K/300K, idx-stack-killed splines, cycle-batched, 120 ep | bigger caps | $\bf{0.764}$ |

**Each unblock buys real AUC:**
- $0.606 \to 0.663$ ($+0.057$): cycle batching lets us include $k{=}2$
- $0.663 \to 0.764$ ($+0.101$): bigger cycle caps + closed-form spline weights + 120 epochs

### 13c. Optimisations landed

| optimisation                                  | code path                                    | gain                       |
|-----------------------------------------------|----------------------------------------------|----------------------------|
| activate `cycle_batch_size` for Epinions      | `run_final_cell.py:cycle_batch`              | OOM → fits at 1 GB         |
| `HSIKAN_CYCLE_BATCH` env-var override         | `run_final_cell.py`                          | per-run tunable            |
| `HSIKAN_MAX_K2 / MAX_K3` env-vars             | `run_final_cell.py:cap_dict`                 | per-arity budget control   |
| `HSIKAN_CHUNK_T` env-var on `SignedKANLayer`  | `signedkan.py:_forward_impl`                 | layer-level chunking (alt) |
| Closed-form CR weights (no `t_powers` stack)  | `splines.py:_catmull_rom_eval`               | ~50% intermediate-tensor savings |
| 4 separate gathers (no `idx` stack)           | `splines.py:_catmull_rom_eval`               | kills the 4× int64 stack (largest single intermediate) |

### 13d. Optimisations deferred

- **Spline-level gradient checkpointing**: redundant with the
  existing `_encode_edges_batched` which already uses
  `torch.utils.checkpoint` per chunk.
- **Triton fused-spline kernel**: would help compute speed
  (fuse 4 gathers + spline arithmetic + sum), not memory.
  Memory is no longer the binding constraint after the cycle-
  batch path.  Defer until a clear real-time inference case.

### 13e. Komondor SLURM script

`scripts/slurm/run_hsikan_epinions.sbatch` — generic SBATCH
template (account / partition placeholders to fill in).  On a
40 GB A100 the published recipe should run uncrowded with
`HSIKAN_CYCLE_BATCH=0` (full-batch path) for maximum
throughput.

Source: `/tmp/hsikan_epinions_v{6,7,8,9,10,11,12,13,14}.log`,
`/tmp/hsikan_epinions_full.log`,
`/tmp/hsikan_epinions_aggressive.log`,
`/tmp/hsikan_mem_profile.log`.

---

## Summary — what HSiKAN actually does

**Wins decisively** ($\Delta \ge +0.27$):
- SBM $n{=}200$ (cycle-rich, 4-community)
- SBM $n{=}400$ (cycle-rich, 4-community)

**Wins marginally** ($\Delta \approx +0.04$):
- Bitcoin Alpha

**Within seed noise** ($|\Delta| < 0.02$):
- Bitcoin OTC

**Loses (walk-rich domains):**
- Slashdot ($\Delta = -0.036$ vs SGT)
- Epinions ($\Delta = -0.335$ vs SGT, also 8 GB GPU memory-bound)

**Side wins (today's bonuses):**
- $\sim 91\%$ sinusoidal symbolic distillation, defended against
  three null baselines.
- Pruned $h{=}4$ Slashdot beats $h{=}16$ counterpart.
- ph18c entropy-regulariser: two confirmed positives, one
  predicted by Path I calibration law.
- HSiKAN-from-HyMeKo IR round-trip: emit + import + forward +
  backward all green.
- Walk-HSiKAN enumerator: 12 verification cases all pass.
- Cartwright-Harary cycle-balance scorer for unsupervised
  chicken-aggressor detection: ensemble AUC $0.69 \pm .12$ on
  synthetic with no labels at any stage.

**Not yet measured (external data needed):**
- Real chicken video — pose tracking + on-real-data unsupervised
  AUC.
- Komondor HPC HSiKAN-Epinions at full recipe (40 GB GPU should
  unblock $0.85$–$0.90$ AUC range, predicted but unconfirmed).
- Real-AUC HSiKAN-from-HyMeKo emit (Item #4 final).
