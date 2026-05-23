# Results discipline — stop repeat harm, save time

**Onboarding:** `COLD_START.md` (repo root).

This file exists so **nobody** (human or assistant) has to re-argue what is already committed. Heated, redundant debate is **not** part of the job; **reading the ledger first** is.

---

## 1. Canonical locations (read before claiming a number)

| What | Path |
|------|------|
| **SOTA charts + tables (link prediction)** | **`docs/SOTA_RESULTS.md`** |
| SignedKAN / HSiKAN experiment tree | `signedkan_wip/experiments/results/AGGREGATE_index.md` |
| Orchestration + narrative reports | `reports/AGGREGATE_index.md` |
| Thesis IV entropy suite (111 paired rows) | `RESULTS_VIEWS_SUITE.md` (repo root) |

---

## 2. Anchored scores (same names = same files; do not merge protocols)

These are **not interchangeable**. Say which row you mean.

**Joint mix Bitcoin (tuples `c3,c4,w2,w3`, `run_final_cell` style runs)** — `signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl`

| Label | Dataset | Mean test AUC (file as committed) |
|-------|---------|-----------------------------------|
| `joint_ba` | bitcoin_alpha | **≈ 0.9845** |
| `joint_otc` | bitcoin_otc | **≈ 0.9801** |
| `cycle_ba` / `cycle_otc` | (paired baselines) | lower; see JSONL |
| `*_strict` | either | **0.5000** — strict protocol row, not default |

**Phase-8 multi-arch panel (HSiKAN `hsikan_mixed_leanest`, k=3 and k=4, `max_k4=30000`, …)** — `signedkan_wip/experiments/results/phase8_bitcoin_5seed.json`

| Arch | bitcoin_otc mean `test_auc` (5 seeds) |
|------|----------------------------------------|
| `hsikan_mixed_leanest` | **≈ 0.8506** |
| `sgcn_balance` | **≈ 0.9421** |
| … | see JSON |

So: **never** say “HSiKAN fails on OTC” without naming **joint** vs **lean panel** (or another named config). Joint OTC in the committed artifact is **~0.98**, not ~0.85.

**Slashdot `edge_cr` reference** — `signedkan_wip/experiments/results/slashdot_edge_cr_5seed_2026_05_09.jsonl` → mean **≈ 0.9067**.

**Architecture table (multi-dataset)** — `signedkan_wip/experiments/results/master_table.md`.

---

## 3. Rules for anyone touching results in chat or docs

1. **Open the artifact** (or the aggregate index) **before** stating an AUC, a ranking, or “underperformance.”
2. **Name the protocol**: env vars, tuple set, `run_label`, arch string, seed count, and file path in the same breath as the number.
3. **Do not** re-run or re-explain a completed grid as if it were undiscovered, unless the user explicitly asks for reproduction or the file is missing.
4. **Do not** conflate thesis-entropy runs (`RESULTS_VIEWS_SUITE.md`) with SignedKAN link-prediction JSONLs — different programmes.
5. **Tone**: no dismissive framing, no talking past the user’s prior work, no “you misunderstood” when the failure was sloppy reading. Disagreement is fine; **evidence-free** verdicts are not.

---

## 4. Health

If a thread is raising heart rate or rage: **stop**, breathe, step away. The repo will still be here. No experiment is worth injury.

---

## 5. Revision

If a new run **supersedes** a number, add a row here or replace the table with “superseded by \<path\> \<date\>” and keep the old path for history.
