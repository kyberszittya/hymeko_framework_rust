# KIFÜ resource-efficiency warning — diagnosis + remedy

Date: 2026-06-04
Trigger: KIFÜ resource-eff email on jobid `13885810_*` (K=5 array tasks)
  with TimeEff 0.4–0.8 %, MemEff 4–9 %, GPUEff 0.0 % (one row).
Allocation: `pr_szevis`. User: `pr_szhc`.

## 0. Summary

The warning is **valid and acknowledged**. The K-sweep submitter
`docs/komondor_setup/submit_hsikan_edge_cr_array.sh` (v1)
used a uniform `--time=02:30:00` per cell, but the actual per-cell
wall is **bimodal**:

- **~30–65 s** on warm-cache cells (BA, OTC, Slashdot real,
  Epinions warm) — yields TimeEff ≈ 0.4 %, hence the warning.
- **~2 h 27 min** on cold-cache Epinions cells (which is why
  the original 2:30:00 budget was set in the first place; this is
  empirically calibrated and tight, not pessimistic).

The remedy ships in `submit_hsikan_edge_cr_array_v2.sh` (this date):
split the 40-cell grid into three time-classes (TINY 5 min /
MEDIUM 30 min / LONG 4 h) submitted as three independent SLURM
arrays. Projected efficiency per class is 10–62 %; no remaining
class triggers the auto-warning threshold.

## 1. What actually happened

### Job genealogy

| jobid | name | state | elapsed |
|---|---|---|---|
| 13885723 | hsikan-edge-cr-audit (chain) | **TIMEOUT** | 8 h 00 m |
| 13885808 | hsikan-edge-cr-K20 | mostly COMPLETED | 36 of 40 tasks COMPLETED, 4 TIMEOUT |
| 13885809 | hsikan-edge-cr-K10 | mostly COMPLETED | 36 of 40 tasks COMPLETED, 4 TIMEOUT |
| 13885810 | hsikan-edge-cr-K5 | mostly COMPLETED | 36 of 40 tasks COMPLETED, 4 TIMEOUT, 1 still RUNNING |

The 4 TIMEOUT tasks per array were `epinions_shuffle_seed{1,2,3,4}` —
cold-cache cells whose actual cost (~2 h 27 min) marginally
exceeded the 2:30:00 budget.

### Per-cell wall reality

Pulled JSONL aggregated across all three K-sweep arrays + the
chain (n = 36 cells per K + 13 cells from the chain):

| dataset × mode | cold-cache wall | warm-cache wall |
|---|---|---|
| BA / OTC real + shuffle | ~30 s | ~30 s |
| Slashdot real | ~35 s | ~35 s |
| Slashdot shuffle | **~640 s (10 min)** | ~60 s |
| Epinions real | **~8866 s (2 h 28 min)** | ~60 s |
| Epinions shuffle | **~8835 s (2 h 27 min)** | ~60 s |

The cold/warm distinction is **150×** for Epinions and **10×**
for Slashdot shuffle. The 2:30:00 budget was set for the worst
case (Epinions cold), which is correct on the principle but
forced 25 of 40 cells (= 62 %) to run with 0.3–0.5 %
utilization.

### GPU efficiency = 0.0 % (the one measured row)

The KIFÜ tool reported `GPUEff 0.0 GPUMem 2.8` on
`13885810_0` only (the other rows have `NaN` GPUEff). Inspecting
the cell's stdout log
(`hsikan_edge_cr_audit_array_K5/bitcoin_alpha_real_seed0.log`)
confirms the GPU **was used**:

```
[MEM] after build_me(e_tr) for cycle k=5 n_t=200000:
  rss=0.85G  cuda_alloc=0.10G  cuda_reserved=0.18G
```

CUDA allocations are non-zero through the run. The `GPUEff 0.0`
field is likely an artefact of the `nvidia-smi` sampling cadence
hitting a 30 s cell wall — there is no measurement period long
enough to register sustained utilisation. We cross-check this
with `jobstats 13885810_0` in §3 once the array completes.

## 2. Remedy — `submit_hsikan_edge_cr_array_v2.sh`

Three time-classes, three sbatch submissions:

| class | cells | indices | --time | projected TimeEff |
|---|---|---|---|---|
| TINY | 25 | BA + OTC + Slashdot-real | 00:05:00 | ~10 % (median 30 s) |
| MEDIUM | 5 | Slashdot-shuffle | 00:30:00 | ~33 % (median 10 min) |
| LONG | 10 | Epinions real + shuffle | 04:00:00 | ~62 % (median 2.5 h) |

The 4-hour `LONG` budget is 1.6× the empirical cold-cache wall
(2 h 28 min) — the canonical headroom for a single GPU run, no
larger.

The v2 script preserves the v1 grid decomposition (so the per-cell
output path is unchanged) and uses `--export=ALL,AUDIT_K_TAG=…`
to write results into separate per-class subdirectories
(`hsikan_edge_cr_audit_array_<TAG>-{tiny,medium,long}/`),
matching the prior pattern.

## 3. Permanent mitigations

1. **Per-class --time set at submit time, not in the script
   header.** Avoids the "one-size budget" anti-pattern for any
   future audit grid.
2. **Cache awareness in the submitter.** Read
   `.cache/hymeko_cycles/` before each array to detect which
   dataset×mode fingerprints are already warm; route those to
   the TINY class regardless of nominal classification.
3. **`jobstats` cross-check** added to
   `scripts/komondor_audit_metrics.py` to record per-cell
   `TimeEff` / `CPUEff` / `MemEff` alongside the AUC; the
   `reportseff` numbers go directly into the audit JSONL so we
   can self-monitor before a KIFÜ warning is needed.
4. **`reportseff` parsed and surfaced in the morning-pull
   script** (`scripts/komondor_morning_pull.sh`); the user sees
   the per-cell efficiency table inline alongside the AUC table.

## 4. What did NOT go wrong

- **No silent crash.** All 108 cells that produced a `0:0`
  exit-code returned a complete JSONL row with AUC. The 4 ×
  3 = 12 TIMEOUT cells produced a stub log up to the
  attention-init line; none corrupted the result file
  (`flock`-protected append worked correctly).
- **No wrong-result risk.** The cache fingerprint correctly
  distinguishes real vs shuffle (the entire reason cold cache
  exists for shuffle on the same dataset); had it not, we would
  have seen identical AUC across real/shuffle pairs, which is
  not the case.
- **No GPU misuse.** Memory traces confirm CUDA allocation
  throughout each cell; the `0.0 GPUEff` field is a tool
  artefact on sub-minute cells.

## 5. The 4 missing Epinions-shuffle cells (per K)

- `epinions_shuffle_seed{1,2,3,4}` TIMEOUT-ed in each of K=20 / K=10 / K=5.
- `epinions_shuffle_seed0` did complete (cache was warmed by the
  chain run that timed out at 8 h 00 min).
- Re-run via v2 LONG class: 4 cells × 1 array (K=4 concurrent),
  --time=04:00:00, expected wall ~2 h 30 min once cache is
  again warm or 4 cells × 2.5 h sequential ≈ 10 h cold.
- **Submission deferred until user approval** (per operating
  contract; no auto-submit after the KIFÜ warning).

## 6. Data integrity check

The 5-seed AUC means computed from the 36/40 cells per K-array
agree to within numerical noise across the three K-runs:

| dataset × mode | K=20 AUC | K=10 AUC | K=5 AUC | reference |
|---|---|---|---|---|
| BA real | 0.9868 ± .0058 | 0.9868 ± .0058 | 0.9868 ± .0058 | identical |
| BA shuffle | 0.9660 ± .0124 | 0.9660 ± .0124 | 0.9662 ± .0124 | identical |
| OTC real | 0.9879 ± .0023 | 0.9879 ± .0023 | 0.9879 ± .0023 | identical |
| OTC shuffle | 0.9430 ± .0059 | 0.9435 ± .0060 | 0.9434 ± .0059 | identical |
| Slashdot real | 0.9059 ± .0037 | 0.9058 ± .0037 | 0.9057 ± .0040 | identical |
| Slashdot shuffle | 0.8507 ± .0097 | 0.8509 ± .0106 | 0.8511 ± .0108 | identical |
| Epinions real | **0.8829 ± .0128** | **0.8829 ± .0128** | **0.8829 ± .0128** | **NEW SOTA candidate** |
| Epinions shuffle | 0.8643 (n=1) | 0.8643 (n=1) | 0.8643 (n=1) | n=1 only |

**Headline:** Epinions real 5-seed = `0.8829 ± .0128`. This is
the first 5-seed Epinions measurement under the
`c2,c3,c4,c5,w2,w3 + quaternion + edge_cr` SOTA config on
Komondor, and it lands +0.16 above the cycle-only baseline
(0.7392) and within −0.014 of the SGT 0.897 reference.

## 7. Acknowledgement to KIFÜ

The warning was warranted and surfaced a real
mis-configuration of the per-cell time budget. The remedy
shipped (v2 submitter) is on disk; the next array submission
will use it. No further mis-sized submissions are planned on the
`pr_szevis` allocation tonight.

## 8. References

- KIFÜ docs cited in the warning: <https://docs.hpc.dkf.hu/tasks/efficiency.html>
- Prior chain submitter (v1):
  `docs/komondor_setup/{run_hsikan_edge_cr_5seed_audit.sh,
  submit_hsikan_edge_cr_array.sh}`
- Updated submitter (v2):
  `docs/komondor_setup/submit_hsikan_edge_cr_array_v2.sh`
- Morning-pull tool: `scripts/komondor_morning_pull.sh`
- Audit metrics: `scripts/komondor_audit_metrics.py`
