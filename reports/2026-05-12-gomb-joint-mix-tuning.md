# HymeKo-Gömb joint-mix random tuning — pilot and phase 2

**Date:** 2026-05-12  
**Git:** `0c55fa8` (short SHA at report time; working tree may differ)  
**GPU:** NVIDIA GeForce RTX 2070 SUPER, 8192 MiB  
**Tooling:** `python -m signedkan_wip.src.run_gomb_tune` with `--joint-mix`, `run_gomb_smoke` subprocess, edge split `80_10_10`, objective **max `test_auroc`**.

---

## Summary

| Phase | Datasets | Trials / dataset | Epochs | Best `test_auroc` | Artifact |
|-------|----------|-------------------|--------|-------------------|----------|
| **1 (pilot)** | `bitcoin_otc` | 3 | 25 | **0.9238** | `reports/gomb_tune_joint_run.jsonl` |
| **2** | `bitcoin_otc`, `bitcoin_alpha` | 5 each | 30 | OTC **0.9165**, Alpha **0.6970** | `reports/gomb_tune_joint_phase2.jsonl` |

Pilot trial **1** failed with **CUDA OOM** (`topk=128`, wide widths, large joint pools). Best pilot config: trial **2** (`d_embed=32`, `topk=32`, `pos_weight_auto`, walks 24k/8k, `lr=5e-3`).

Phase 2 saw **many OOMs** on the same 8GB card when `topk` was already capped to 64 but **walk caps remained 50k** and widths stayed large — failures are `torch.OutOfMemoryError` in stderr tails (fragmentation / prior trial pressure on the same GPU session).

---

## Tuner hardening (after phase 2)

`signedkan_wip/src/run_gomb_tune.py` — `for_joint_mix_tuning(..., dataset=...)`:

1. **`topk` ≤ 64** for wide joint on `bitcoin_otc` / `bitcoin_alpha` (already planned after pilot).
2. **`max_walks_w2` / `max_walks_w3` ≤ 32_000** on the same graphs (reduces tuple RAM vs 50k draws).

Slashdot / Epinions joint trials are **unchanged** by these clamps. Unit tests: `test_for_joint_mix_clamps_topk_on_bitcoin_wide`, `test_for_joint_mix_tuning_clears_cycle_ks_and_samples_walks`.

**Recommendation:** Re-run phase 2 (or a longer study) with the hardened tuner; optionally `--architecture compact` or `CUDA_VISIBLE_DEVICES` isolation per trial if OOMs persist.

---

## Phase 1 — `bitcoin_otc` (3 trials, 25 epochs, `--search-seed 0`)

| Trial | `returncode` | `test_auroc` | Notes |
|------:|-------------:|-------------:|--------|
| 0 | 0 | 0.9152 | `topk=32`, walks 50k/50k |
| 1 | 1 | — | CUDA OOM (`topk=128`, wide `d_middle`) |
| 2 | 0 | **0.9238** | Best; `pos_weight_auto`, `topk=32`, walks 24k/8k |

**Best trial params (trial 2):**  
`lr=0.005`, `d_embed=32`, `d_outer=8`, `M_outer=12`, `d_middle=16`, `d_core=32`, `topk=32`, `n_tiers=2`, `weight_decay=1e-6`, `pos_weight_auto=true`, `max_walks_w2=24000`, `max_walks_w3=8000`.

---

## Phase 2 — `bitcoin_otc` + `bitcoin_alpha` (5 trials each, 30 epochs, `--search-seed 1`)

Topk was clamped to 64 for Bitcoin wide joint; walk caps were **not** yet clamped to 32k during this run.

### `bitcoin_otc`

| Trial | `returncode` | `test_auroc` |
|------:|-------------:|-------------:|
| 0 | 0 | 0.9165 |
| 1 | 0 | 0.9070 |
| 2 | 1 | — (OOM) |
| 3 | 1 | — (OOM) |
| 4 | 1 | — (OOM) |

**Best:** 0.9165 (trial 0).

### `bitcoin_alpha`

| Trial | `returncode` | `test_auroc` |
|------:|-------------:|-------------:|
| 0 | 1 | — (OOM) |
| 1 | 1 | — (OOM) |
| 2 | 0 | **0.6970** |
| 3 | 1 | — (OOM) |
| 4 | 1 | — (OOM) |

**Best:** 0.6970 (trial 2).

---

## Commands (repro)

**Phase 1 (as run):**

```bash
PYTHONPATH=. python -m signedkan_wip.src.run_gomb_tune \
  --datasets bitcoin_otc --joint-mix --trials 3 --search-seed 0 --data-seed 0 \
  --edge-split 80_10_10 --n-epochs 25 --device cuda --timeout-s 3600 \
  --out reports/gomb_tune_joint_run.jsonl
```

**Phase 2 (as run):**

```bash
PYTHONPATH=. python -m signedkan_wip.src.run_gomb_tune \
  --datasets bitcoin_otc bitcoin_alpha --joint-mix --trials 5 --search-seed 1 \
  --data-seed 0 --edge-split 80_10_10 --n-epochs 30 --device cuda --timeout-s 3600 \
  --out reports/gomb_tune_joint_phase2.jsonl
```

**Suggested next run (post hardening):**

```bash
PYTHONPATH=. python -m signedkan_wip.src.run_gomb_tune \
  --datasets bitcoin_otc bitcoin_alpha --joint-mix --trials 12 --search-seed 2 \
  --data-seed 0 --edge-split 80_10_10 --n-epochs 40 --device cuda --timeout-s 7200 \
  --out reports/gomb_tune_joint_phase3.jsonl
```

---

## Open issues

- **VRAM:** Joint-mix stacks four pools; even `topk=64` + 50k walks can OOM on 8GB — addressed by **32k walk clamp** for Bitcoin in the tuner.
- **Alpha headline:** Phase 2 best 0.697 on test is modest; more epochs, `pos_weight_auto`, or compact search may help — needs more trials after OOM rate drops.

---

## Addendum — saved results & handoff (same session)

**On-disk experiment logs (append-only JSONL):**

| File | Contents |
|------|----------|
| `reports/gomb_tune_joint_run.jsonl` | Phase 1 pilot (3 trials, `bitcoin_otc`) + phase summary line |
| `reports/gomb_tune_joint_phase2.jsonl` | Phase 2 (5 trials × OTC + Alpha) + two summary lines |

**Code landed after phase 2 (tuner VRAM):** `signedkan_wip/src/run_gomb_tune.py` — `for_joint_mix_tuning(..., dataset=ds)` clamps **wide** joint on `bitcoin_otc` / `bitcoin_alpha` to **`topk` ≤ 64** and **each walk cap ≤ 32_000**; Slashdot joint unchanged.

**Regression tests:** `signedkan_wip/tests/test_hymeko_gomb.py` (tuner + Gömb suite) — **30** tests, including `test_for_joint_mix_clamps_topk_on_bitcoin_wide` and joint-mix `_build_cmd` checks.

**Not run yet:** Phase 3 command in section “Suggested next run” above — run when you are back to continue tuning on a cooler GPU.

**Quick index:** see also `reports/2026-05-12-gomb-joint-tune-artifacts.md` for a one-page artifact list.
