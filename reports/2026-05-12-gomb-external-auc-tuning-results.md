# Gömb external AUC tuning — results (2026-05-12)

Companion to **`docs/plans/2026-05-12-gomb-external-auc-tuning/`** (LaTeX + Mermaid + TikZ + PDF).

## Reference bars (non-HSiKAN)

| Bar | Bitcoin Alpha | Bitcoin OTC | Source in repo |
|-----|----------------:|------------:|------------------|
| **Published SGCN** | ~**0.91** | ~**0.93** | `PAPER_DRAFT_2026_05_03.md`, `build_paper_tables.py` |
| **In-repo tuned SGCN** | **0.927** | **0.957** | Same draft (matched-protocol table) |

HSiKAN headline rows (~0.94 Alpha) are **not** the success criterion here.

## What was run

Driver: **`signedkan_wip/experiments/run_gomb_external_auc_tuning.sh`**  
Settings: `TRIALS=8`, `NEPOCHS=38`, `DEVICE=cuda`, `80_10_10`, `data_seed=0`.

| Envelope | Search seeds | JSONL |
|----------|----------------|-------|
| **Joint-mix** (`--joint-mix`, wide + Bitcoin VRAM clamps) | joint `11` | `reports/gomb_tune_external_joint.jsonl` |
| **Vanilla / mixed** (wide `sample_params`, optional `cycle-ks`) | vanilla `13` | `reports/gomb_tune_external_vanilla.jsonl` |

Wall clock ~**5.7 min** for the full script (two datasets × two envelopes).

## Best `test_auroc` (single best trial per dataset, this batch)

| Dataset | Joint-mix best | Vanilla best | Published SGCN | Tuned SGCN |
|---------|---------------:|---------------:|---------------:|-----------:|
| **bitcoin_alpha** | **0.9058** | 0.8910 | ~0.91 | 0.927 |
| **bitcoin_otc** | 0.9214 | **0.9228** | ~0.93 | 0.957 |

### Readout vs “break outside HSiKAN territory”

- **Alpha vs published (~0.91):** joint-mix **0.9058** is in the same band (marginal “touch” of the published bar; not a clean +0.03 margin). Vanilla did **not** exceed published this run.
- **OTC vs published (~0.93):** best **0.9228** (vanilla) and **0.9214** (joint) sit **~0.007–0.009 below** the common published 0.93 anchor — **Stage A not cleared** for OTC on this short budget.
- **Vs in-repo tuned SGCN:** both datasets remain **below** tuned SGCN (0.927 / 0.957); that is **Stage B**, expected to need more epochs, seeds, and/or Slashdot-scale compact runs.

## Failures / OOM

Several trials scored `−inf` (`returncode` ≠ 0); joint OTC had multiple CUDA OOMs despite `topk` / walk clamps — wide `d_embed` + four-slot forward still spikes VRAM on **8 GB**. See `stderr_tail` in the JSONL lines.

## Follow-ups (plan-aligned)

1. **Multi-seed** best-of-mean for the winning trial params (joint Alpha, vanilla OTC).
2. **Raise `NEPOCHS`** (e.g. 80–120) on the best few configs only (narrow second stage).
3. **Slashdot** under `--architecture compact` + joint or vanilla (separate JSONL; long enumeration).
4. Optional: **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** between trials to reduce fragmentation.

## Artifacts

- Plan: `docs/plans/2026-05-12-gomb-external-auc-tuning/plan.{tex,pdf,mmd,tikz}`
- Logs: `reports/gomb_tune_external_joint.jsonl`, `reports/gomb_tune_external_vanilla.jsonl`
- Script: `signedkan_wip/experiments/run_gomb_external_auc_tuning.sh`

---

## Extension (same programme): inference timing + Slashdot

### `run_gomb_smoke` JSON fields

Every successful smoke summary line now includes:

- **`infer_wall_s`** — wall seconds for one **batched** forward over all held-out edges (val ∪ test for `80_10_10`, else val only), after an **untimed CUDA warmup** when `device=cuda`, with `torch.cuda.synchronize()` around the timed region.
- **`infer_n_edges`** — edge count in that batch.
- **`infer_edges_per_s`** — throughput.

A console line `[inference] edges=… wall_s=… edges_per_s=…` is printed before the final JSON.

### Slashdot pilot (compact, `TRIALS_SLASH=1`, `NEPOCHS_SLASH=6`, CUDA)

| Envelope | Outcome | `test_auroc` (if ok) | Inference (vanilla success row) |
|----------|---------|----------------------|-----------------------------------|
| Vanilla / mixed | **OK** | **0.5378** (6 ep — not converged; SGCN bar ~0.91 is far) | **~3.27M edges/s**, `infer_wall_s` ≈ **0.034 s** for **109 841** val+test edges (`reports/gomb_tune_external_slashdot_vanilla.jsonl` trial 0). |
| Joint-mix | **CUDA OOM** on RTX 2070 Super 8 GB | — | Even with walk caps **4096** and `topk` **16**, four-slot `JointMixGomb` over full Slashdot pools exceeds VRAM (`reports/gomb_tune_slashdot_joint_retry.jsonl`). |

### Tuner / script changes

- **`for_joint_mix_tuning`:** Slashdot / Epinions joint trials clamp **`max_walks_w2/w3` ≤ 4096**; compact Slashdot also clamps **`topk` ≤ 24**.
- **`run_gomb_external_auc_tuning.sh`:** Slashdot **joint** is **opt-in** (`RUN_SLASHDOT_JOINT=1`); Slashdot **vanilla** remains default (`RUN_SLASHDOT_VANILLA=1`). Use larger GPU or **CPU** for Slashdot joint exploration.

### Extra artifact paths

- `reports/gomb_tune_external_slashdot_vanilla.jsonl`
- `reports/gomb_tune_external_slashdot_joint.jsonl` (failed pilot; kept for stderr)
- `reports/gomb_tune_slashdot_joint_retry.jsonl` (post–walk-clamp retry; still OOM)

