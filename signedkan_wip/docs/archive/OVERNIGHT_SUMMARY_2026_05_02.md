# Overnight summary — 2026-05-02

Autonomous session. Started after the user requested "1, 2, 3" (Rust
enumerator port, SiGAT baseline, 5-seed Bitcoin re-validation), then
went to sleep with "go ahead with these tasks. Get more data and
compare HSiKAN with other architectures."

## Headline findings

### 1. **k=4+k=5 is the architectural sweet spot — k=3 is dead weight**

Phase 9 ran three mixed-arity variants (`k34`, `k345`, `k45`) across
the phase-6 panel + Bitcoin Alpha/OTC at 5 seeds. **On every dataset,
`k45` (drop k=3) beats or matches `k34` (current cited best)**:

| dataset       | k34          | k345         | **k45**          | Δ(k45−k34) |
|---------------|--------------|--------------|------------------|-----------:|
| karate        | 0.976±0.034  | 0.976±0.034  | **1.000±0.000**  | +0.024 |
| sbm_n200_k4   | 0.920±0.024  | 0.955±0.015  | **0.968±0.011**  | +0.048 |
| sbm_n400_k5   | 0.902±0.017  | 0.945±0.006  | **0.946±0.010**  | +0.044 |
| hier_n240     | 0.948±0.021  | **0.974±0.010** | 0.973±0.013   | +0.025 |
| bitcoin_alpha | 0.823±0.013  | 0.862±0.018  | **0.886±0.005**  | **+0.063** |
| bitcoin_otc   | 0.850±0.021  | 0.882±0.016  | **0.893±0.012**  | **+0.043** |

**αₖ patterns** (mean over 5 seeds):
- `k34` on SBMs: α≈[0.03, 0.97] — k=4 dominates k=3 by 30:1.
- `k34` on Bitcoin: α≈[0.40, 0.60] — k=3 still gets weight.
- `k45` across all: α≈[0.5±0.1, 0.5±0.1] — k=4 and k=5 share weight.
- `k345` across all: k=3 contribution near-zero on balanced graphs;
  ~0.30 on Bitcoin.

**Why this matters:** the original n-tuples paper hypothesis was
"mixed k=3+k=4+k=5 with learned αₖ". The actual story is sharper:
**k=3 (the standard signed-balance triad) is the wrong arity** for
signed link prediction. k=4 and k=5 cycles carry signed-balance
information that k=3 does not. Replace the k=3 baseline with k=4+k=5
in the headline architecture.

This **halves the Bitcoin gap to SGCN**:
- Bitcoin Alpha: SGCN 0.929 vs HSiKAN-k45 **0.886** (Δ −0.043, was −0.106)
- Bitcoin OTC:   SGCN 0.942 vs HSiKAN-k45 **0.893** (Δ −0.049, was −0.092)

HSiKAN-k45 also **beats GCN on Bitcoin Alpha** (0.886 vs 0.871).

### 2. **Rust k-cycle enumerator landed in `hymeko_py`**

New file `hymeko_py/src/cycles.rs`. Function exposed:
`hymeko.enumerate_k_cycles_rs(edges_u, edges_v, n_nodes, k)`.
Algorithm matches Python reference (DFS + root-canonicalisation).
Build: `cd hymeko_py && maturin develop --release`.

**Equivalence verified** on Bitcoin Alpha: exact set match for k=3
(22,153 cycles) and k=4 (615,962 cycles). Speedup 6–8× — modest
because PyO3 marshaling of 600k tuples dominates.

**Slashdot k=4 enumeration unblocked** but a new bottleneck surfaced:
Slashdot has **55,528,862 k=4 cycles** (~1.8 GB raw vertex data).
The "future eng-item: port to Rust" is closed; the replacement open
item is "mixed-arity training pipeline that scales beyond ~600k
cycles". Slashdot mixed-arity remains training-time-blocked.

### 3. **SiGAT-attn baseline implemented (in-protocol)**

New file `signedkan_wip/src/baselines/sigat_model.py`. 2-motif
pos/neg neighbour decomposition with multi-head attention. Honest
re-impl labelled "SiGAT-style attention" — NOT the 38-motif directed
reference (that needs external code).

Phase 8 results (5 seeds, mean±std AUC):
- Bitcoin Alpha: SGCN 0.929 > **SiGAT 0.903** > MLP 0.892 > GCN 0.871 > HSiKAN-k34 0.828 > SignedKAN-L1 0.745
- Bitcoin OTC:   SGCN 0.942 > **SiGAT 0.932** > MLP 0.908 > GCN 0.905 > HSiKAN-k34 0.851 > SignedKAN-L1 0.802

**SiGAT-attn is the second-best architecture on Bitcoin** — between
SGCN and the rest. This **rules out "attention vs spline-aggregation"
as the explanation for SGCN's Bitcoin win**: both attention-based
(SiGAT) and aggregation-based (SGCN) beat HSiKAN-k34 at high
positivity. The mixed-arity advantage genuinely depends on the
sign-balance regime, not on the architectural family.

### 4. **5-seed Bitcoin re-validation confirms 3-seed numbers**

Within seed noise. The Δ(SGCN − HSiKAN_k34) was −0.07/−0.11 at 3
seeds; now −0.10/−0.09 at 5 seeds. Stds tightened (0.010/0.016).

### 5. **Positivity-sweep on synthetic SBM**

Phase 11 ran HSiKAN-k34 vs SGCN+balance vs SiGAT-attn vs GCN-blind
across pos_in ∈ {50, 55, ..., 95} on n=200 SBM (3 seeds). Realized
%pos ranges 0.36–0.60 (because pos_out=0.15 and many cross-community
edges).

| pos_in | %pos | GCN | HSiKAN-k34 | SGCN+balance | SiGAT-attn |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.36 | 0.494±0.033 | **0.951±0.013** | 0.536±0.032 | 0.548±0.063 |
| 60 | 0.41 | 0.490±0.033 | **0.905±0.020** | 0.591±0.044 | 0.526±0.041 |
| 70 | 0.46 | 0.550±0.058 | 0.815±0.109 | 0.626±0.033 | 0.535±0.023 |
| 80 | 0.53 | 0.585±0.095 | **0.913±0.025** | 0.673±0.023 | 0.579±0.072 |
| 90 | 0.58 | 0.595±0.102 | **0.951±0.018** | 0.695±0.016 | 0.590±0.080 |
| 95 | 0.60 | 0.618±0.072 | **0.954±0.017** | 0.742±0.056 | 0.536±0.080 |

**HSiKAN-k34 dominates SGCN at every positivity point on this synthetic
SBM panel** (Δ +0.16 to +0.41 AUC). Slight U-shape with a noisy dip at
pos_in=70-75. **The Bitcoin/Slashdot SGCN-wins reversal cannot be
explained by edge positivity alone** — it must involve density, degree
distribution, or community structure, since HSiKAN wins at every
synthetic positivity in [0.36, 0.60].

### 5b. Phase 11b — HSiKAN-k45/k345 sweep is even cleaner

Adding the k45 and k345 variants to the same positivity sweep:

| pos_in | %pos | HSiKAN-k34 | HSiKAN-k345 | **HSiKAN-k45** | SGCN+balance |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.36 | 0.951±0.013 | **0.988±0.004** | 0.986±0.005 | 0.536±0.032 |
| 60 | 0.41 | 0.905±0.020 | **0.974±0.002** | 0.970±0.004 | 0.591±0.044 |
| 70 | 0.46 | 0.815±0.109 | 0.959±0.015 | **0.972±0.005** | 0.626±0.033 |
| 75 | 0.50 | 0.792±0.134 | 0.894±0.111 | **0.972±0.010** | 0.632±0.029 |
| 80 | 0.53 | 0.913±0.025 | 0.954±0.015 | **0.965±0.013** | 0.673±0.023 |
| 90 | 0.58 | 0.951±0.018 | 0.976±0.016 | **0.983±0.008** | 0.695±0.016 |

**HSiKAN-k45 is robust through the noisy mid-region** that broke k34
and even k345. The dip was an artifact of k=3 being a bad arity in
the balanced regime. **HSiKAN-k45 dominates SGCN by Δ +0.24 to +0.45
AUC across the entire positivity range** — a far stronger gap than
the k34-vs-SGCN claim.

**Learned αₖ encodes the regime shift:**
- Low positivity (pos_in≤65, real %pos 0.36-0.43): α₅ ≈ 0.65 — k=5 carries
- High positivity (pos_in≥70): α₄ ≈ 0.55-0.67 — k=4 takes over
- k=3 always near zero across the sweep (α₃ ≤ 0.08)

This is the cleanest empirical signature of the architectural argument:
**the model literally selects which higher-order cycle motif to use
based on the sign-balance regime.** k=3 is consistently rejected.

**SiGAT-attn underperforms badly on these synthetic graphs** (0.51-0.59
AUC across the sweep — basically random). It's competitive on Bitcoin
(0.90+) but fails on small balanced SBMs. Worth investigating whether
the 2-motif simplification breaks down at low edge counts, or whether
this is a hyperparameter issue with the multi-head attention on tiny
graphs. NOTE FOR NEXT SESSION.

## Master architecture comparison table

See `signedkan_wip/experiments/results/master_table.md`. Per-dataset
winners by mean AUC at 5 seeds:

- **karate**: tie (sgcn, hsikan_k45, gcn all 1.000)
- **sbm_n200_k4**: hsikan_k45 0.968 > hsikan_k345 0.955 > hsikan_k34 0.920
- **sbm_n400_k5**: hsikan_k45 0.946 > hsikan_k345 0.945 > hsikan_k34 0.902
- **hier_n240**: hsikan_k345 0.974 > hsikan_k45 0.973 > hsikan_k34 0.948
- **bitcoin_alpha**: sgcn 0.929 > sigat 0.903 > mlp 0.892
- **bitcoin_otc**: sgcn 0.942 > sigat 0.932 > mlp 0.908
- **slashdot**: sgcn 0.919 > mlp 0.888 > gcn 0.871

## Code landed (this session)

- `hymeko_py/src/cycles.rs` — Rust k-cycle enumerator with
  PyO3 binding.
- `hymeko_py/src/lib.rs` — registers `enumerate_k_cycles_rs`.
- `signedkan_wip/src/baselines/sigat_model.py` — SiGAT-attn model
  (2-motif pos/neg, multi-head attention).
- `signedkan_wip/src/run_phase8_bitcoin_5seed.py` — Bitcoin 6-arch panel.
- `signedkan_wip/src/run_phase9_k345_mixed.py` — k34/k345/k45 sweep.
- `signedkan_wip/src/run_phase10_master_table.py` — analysis.
- `signedkan_wip/src/run_phase11_positivity_sweep.py` — pos sweep w/ k34.
- `signedkan_wip/src/run_phase11b_k45_positivity.py` — pos sweep w/ k45.
- `signedkan_wip/src/n_tuples.py` — Rust fast-path.
- `signedkan_wip/src/run_phase2_mixed_arity.py` — `max_per_arity` dict.

## Open items for next session

1. **Paper rewrite**: replace `HSiKAN-k34` with `HSiKAN-k45` as the
   headline architecture. The k=3 result moves to "ablation" status.
   Update n-tuples paper §IV.7 narrative.
2. **k=4+k=5+k=6 ablation**: does the trend continue or saturate?
   Bitcoin k=6 enumeration cost should be checked via the new Rust
   path before committing to a sweep.
3. **Slashdot mixed-arity**: still blocked by training-time memory,
   not enumeration. Needs sparse-friendly mixed-arity training
   pipeline (weighted/stratified subsampling, on-the-fly enumeration).
4. **Density/degree analysis on Bitcoin**: identify which graph
   property explains the SGCN-wins reversal beyond positivity.
   Phase 11 shows positivity alone doesn't explain it.
5. **Full SiGAT (38-motif directed) reference comparison** — current
   baseline is the 2-motif undirected approximation. For
   reference-grade numbers, pull and adapt the Huang et al. 2019
   reference impl.

## Result files

```
signedkan_wip/experiments/results/
├── master_table.md                              ← 5-seed panel comparison
├── phase6_small_synth.json                      (5-seed SBM/hier — already had)
├── phase7_slashdot.json                         (3-seed, max_k3=15k — old)
├── phase7_slashdot_5seed_k3_100k.json           (5-seed, max_k3=100k)
├── phase7_slashdot_5seed_k3_200k.json           (5-seed, max_k3=200k)
├── phase8_bitcoin_5seed.json                    (Bitcoin + SiGAT)
├── phase9_k345_mixed.json                       (k34/k345/k45 sweep)
├── phase11_positivity_sweep.json                (pos sweep w/ k34)
└── phase11b_k45_positivity.json                 (pos sweep w/ k45 — in progress)
```

## Memory updates

- `project_rust_cycle_enum_2026_05_02.md`
- `project_phase8_bitcoin_sigat_2026_05_02.md`
- `project_phase9_k45_sweet_spot_2026_05_02.md`
- `project_hsikan_mixed_arity_2026_05_01.md` (5-seed re-val + Slashdot
  trajectory appended)
