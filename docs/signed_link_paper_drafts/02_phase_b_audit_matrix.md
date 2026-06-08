# Phase B audit matrix — Table 1 specification

**Status**: DRAFT for coworker review. Cells need numbers; each cell's
provenance is specified below the table.

**Goal**: produce the central evidentiary table of the paper. Every
title claim ("inflates seven years"; "strict protocol restores
comparability") reduces to numbers in this table.

---

## Table 1 — Leakage audit and strict-protocol re-evaluation

Per (method, dataset) cell we report **four numbers**:

| Symbol | Quantity | Expected behaviour |
|---|---|---|
| **A_leak** | AUROC under leaky transductive protocol | the published number |
| **S_leak** | Shuffle-AUROC under leaky protocol | > 0.5 (audit fires) |
| **A_strict** | AUROC under strict training-edge-only protocol | < A_leak |
| **S_strict** | Shuffle-AUROC under strict protocol | ≈ 0.5 (audit clean) |

Each number is **mean ± std over n=3 seeds** (seeds 0, 1, 2).

The **inflation index** for that cell is `ΔA = A_leak − A_strict`.

### Matrix layout (6 baselines × 5 datasets + Gömb-strict)

|                 | Bitcoin-Alpha | Bitcoin-OTC | Slashdot | Epinions | Reddit Hyperlinks |
|-----------------|---|---|---|---|---|
| **SGCN**        | [Phase B] | [Phase B] | [Phase B] | [Phase B] | [Phase B] |
| **SiGAT**       | [Phase B] | [Phase B] | [Phase B] | [Phase B] | [Phase B] |
| **SGCL**        | [Phase B] | [Phase B] | [Phase B] | [Phase B] | [Phase B] |
| **SiGformer**   | [Phase B] | [Phase B] | [Phase B] | [Phase B] | [Phase B] |
| **SE-SGformer** | [Phase B] | [Phase B] | [Phase B] | [Phase B] | [Phase B] |
| **DADSGNN**     | [Phase B] | [Phase B] | [Phase B] | [Phase B] | [Phase B] |
| **HSiKAN (transductive)** | 0.9959 ± [?] | [?] | [?] | [?] | [?] |
| **Gömb-strict** | **0.8972 ± [?]** | **0.9145 ± [?]** | **0.9017 ± [?]** | **0.9425 ± [?]** | **0.7612 ± [?]** |

Cell format (per method, per dataset):

```
A_leak: 0.9897 ± 0.0014
S_leak: 0.8231 ± 0.0042     ← shuffle still > 0.5 → AUDIT FIRES
A_strict: 0.7724 ± 0.0089
S_strict: 0.5012 ± 0.0061   ← shuffle ≈ 0.5 → AUDIT CLEAN
ΔA = A_leak − A_strict = +0.2173 pp
```

### Total compute budget per cell

Each cell needs 4 × 3 = 12 training runs (4 protocol×audit conditions
× 3 seeds). Rough budget per run on the RTX 2070 SUPER:

| Method | Wall per run | Cell wall (12 runs) |
|---|---|---|
| SGCN     | ~15 min | ~3 hr |
| SiGAT    | ~30 min | ~6 hr |
| SGCL     | ~30 min | ~6 hr |
| SiGformer | ~45 min | ~9 hr |
| SE-SGformer | ~60 min | ~12 hr |
| DADSGNN  | ~60 min | ~12 hr |
| Gömb-strict | ~30 min | ~6 hr |

**Total Phase B compute**: 7 methods × 5 datasets × ~7 hr/cell mean ≈ **245 GPU-hours**
≈ **10 days continuous** on the single 2070 SUPER, or **3 days** on
2 GPUs in parallel.

Already in hand (the user's prior runs):
- Gömb-strict at strict protocol, 5/5 datasets, seed 0 (point estimates,
  not yet n=3 std bands). Re-run with seeds 1, 2 → **~10 hr** of compute
  to complete the bottom row.

---

## Per-cell reproduction protocol

For each (method, dataset, protocol, seed) combination, the
exact reproduction command is:

```bash
PYTHONPATH=. python signedkan_wip/src/benchmarks/run_audit_cell.py \
    --method   {sgcn|sigat|sgcl|sigformer|sesgformer|dadsgnn|gomb_strict|hsikan} \
    --dataset  {bitcoin_alpha|bitcoin_otc|slashdot|epinions|reddit_hyperlinks} \
    --protocol {leaky|strict} \
    --shuffle  {no|yes} \
    --seed     {0|1|2} \
    --epochs   <method-specific HPO budget> \
    --lr       <method-specific HPO budget> \
    --out      reports/phase_b/{method}/{dataset}/{protocol}_{shuffle}_seed{seed}.json
```

> **[TODO/CW]** The `run_audit_cell.py` script does not yet exist
> as a single entry point — each method has its own training script.
> Phase B blocker: write a thin dispatcher that takes the args above
> and calls the appropriate per-method runner. ~1 day of engineering.

### HPO parity per cell

For comparability under both leaky and strict protocols, every
method must use the **same HPO budget** within a (method, dataset)
pair. Suggested protocol:

1. Run HPO on the LEAKY protocol only (the published setting),
   matching whatever budget the original paper reported (Optuna 50
   trials by default).
2. Lock the discovered HPs.
3. Run all four conditions (leaky / strict × no-shuffle / shuffle)
   with locked HPs and 3 seeds.

This avoids the criticism "you tuned for the strict protocol."

> **[TODO/CW]** Confirm that each baseline's original paper reports
> the HPO budget. For methods that don't report it, use Optuna 50
> trials as a defensible default and note in Methods.

---

## Inflation summary statistic

For the title claim "inflates seven years," we need an aggregate
inflation number across the audit matrix.

### Recommended summary statistic

Mean ΔA across the 30 (method, dataset) cells, with 95% bootstrap CI:

```
ΔA_mean = mean(A_leak[m, d] − A_strict[m, d])
         over (m ∈ {SGCN, …, DADSGNN}) × (d ∈ 5 datasets)
ΔA_95CI = bootstrap-95 over the 30 cells
```

A defensible Abstract claim is then:

> "Under the strict protocol, AUROC drops by ΔA_mean ± half-CI pp
>  on average across methods and datasets (95% CI: [low, high])."

Per-method breakdowns (how each method's inflation distributes
across datasets) belong in a supplementary figure.

> **[TODO/CW]** Decide whether to also report ΔS = S_leak − S_strict
> (audit-firing magnitude) as a secondary statistic. It's evidence the
> audit works as designed, but might overcomplicate the headline.

---

## Special cells

### HSiKAN row

HSiKAN's 0.9959 transductive Bitcoin-Alpha is **Exhibit A of the
leakage magnitude**. It must be in the table but framed deliberately:

- The HSiKAN row reports A_leak only (deliberately — we don't claim
  strict-protocol HSiKAN as a contribution).
- The corresponding S_leak is the audit signal.

> **[TODO/CW]** Discuss whether to additionally re-train HSiKAN under
> the strict protocol for completeness. Cheap (~5 cells × 12 runs
> ≈ 30 hr) and answers "but is the architecture itself audit-clean?"

### Reddit Hyperlinks column

This is the **rhetorical strongest column**: lower numbers
across all methods + an honest 0.7612 for Gömb-strict makes the
audit credible.

Consider opening § Results with Reddit, not closing.

---

## Open coordination questions for the coworker

> **[CW]** Which baselines are already audit-running, which are
> in-progress, which are not yet started? Need a status check before
> compute budgeting.

> **[CW]** Are HPO logs from the original methods available, or
> do we need to re-run HPO from scratch? Affects the ~245 hr budget.

> **[CW]** What's the Bitcoin-Alpha A_strict number for SE-SGformer?
> The "+19.77 pp" Epinions result is derived; the Bitcoin-Alpha
> equivalent would clarify whether the gap is dataset-specific.

> **[CW]** Confirm dataset versions / hashes. Bitcoin-OTC has at
> least two public versions in circulation (different timestamp
> cuts). Need to commit to one and hash-anchor.
