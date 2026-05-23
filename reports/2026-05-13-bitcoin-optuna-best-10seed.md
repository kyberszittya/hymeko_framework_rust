# Bitcoin Optuna-best — 10-seed paired validation

**Date:** 2026-05-13
**Git SHA:** `0c55fa81d0df99ed6a96566e3317ea122553d6ce`
**Storage:** `signedkan_wip/experiments/results/bitcoin_optuna_best_5seed_2026_05_13.jsonl` (20 rows: 10 seeds × 2 datasets)
**Wall:** 16:34 → 17:19 CEST = **45 min** for 20 runs.

## Summary

The Optuna best-trial configs from
[reports/2026-05-13-optuna-handoff-slashdot-continuation.md](2026-05-13-optuna-handoff-slashdot-continuation.md)
were validated at **10 seeds** vs the existing 5-seed `joint_mix` baseline
([signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl](../signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl)).

**Headline:** the 0.997 / 0.996 single-trial maxima from this morning are
**not seed luck** — they survive a 10-seed paired sweep with **6–7σ paired
wins**, **5/5 positive seeds** on the joint_mix-matched cohort, **at half the
params on Alpha and a quarter the params on OTC**.

| Dataset | Config | n | hidden | params | fwd ms | AUC mean ± pstdev |
|---|---|---:|---:|---:|---:|---|
| **bitcoin_alpha** | **optuna_best_alpha** | 10 | 8 | **30 487** | 656.1 | **0.9959 ± 0.0011** |
| bitcoin_alpha | joint_ba (ref) | 5 | 16 | 61 094 | 341.6 | 0.9845 ± 0.0025 |
| **bitcoin_otc** | **optuna_best_otc** | 10 | 4 | **23 815** | **30.5** | **0.9933 ± 0.0023** |
| bitcoin_otc | joint_otc (ref) | 5 | 16 | 94 662 | 342.3 | 0.9801 ± 0.0051 |

**Paired-Δ on the matched joint_mix seeds (0-4):**

| Dataset | mean Δ | pstdev | SEM | σ | win-rate |
|---|---:|---:|---:|---:|---:|
| bitcoin_alpha | **+0.0119** | 0.0022 | 0.0010 | **+11.96σ** | **5/5** |
| bitcoin_otc | **+0.0139** | 0.0044 | 0.0020 | **+7.02σ** | **5/5** |

Both promotions clear the `feedback_n_seed_before_paper_promotion`
gate by a wide margin.

## Per-seed AUC table (new 10-seed)

```
alpha: s0=0.9970, s1=0.9962, s2=0.9957, s3=0.9962, s4=0.9967,
       s5=0.9945, s6=0.9947, s7=0.9974, s8=0.9962, s9=0.9941
otc:   s0=0.9957, s1=0.9947, s2=0.9930, s3=0.9946, s4=0.9922,
       s5=0.9952, s6=0.9931, s7=0.9872, s8=0.9935, s9=0.9938
```

Range Alpha: 0.9941–0.9974 (33 pts). Range OTC: 0.9872–0.9957 (85 pts).
OTC seed 7 (0.9872) is the worst outlier and still 7 points above the
joint_otc 5-seed mean.

## What changed vs joint_mix

| Knob | joint_mix Alpha (h=16) | optuna_best_alpha (h=8) | joint_mix OTC (h=16) | optuna_best_otc (h=4) |
|---|---|---|---|---|
| tuple set | c3,c4,w2,w3 | **c2,c5,w2,w3,w4** | c3,c4,w2,w3 | **c2,c5,w2,w3,w4** |
| hidden | 16 | **8** | 16 | **4** |
| attention M_e | (none in baseline) | none | (none) | **quaternion + highway 0.137** |
| α-entropy λ | 0 | 0.0966 | 0 | 1.48e-5 |
| attn-entropy λ | 0 | 0 | 0 | 1.27e-3 |
| max_k_cap | (default) | 100 000 | (default) | 50 000 |

Both Optuna picks chose **walks-dominant + cycle endpoints (c2 + c5)** and
**dropped the cycle middle (c3, c4)** — consistent with the
`project_walks_epinions_5seed_2026_05_11` cross-dataset pattern (walks
displace cycles at larger / sparser graphs; cycle arities that survive
are those walks can't reproduce — c2's closure + c5's long range).

OTC additionally picked **quaternion attention + highway gate** with
`highway_max ≈ 0.14` (mild, not full-open) and a very weak α-entropy
prior + meaningful attention-entropy prior. Alpha dropped attention
entirely. This **disagrees** with the "TPE prunes attention" reading
of just the Alpha result — attention has its place on the harder
sign-imbalance OTC graph.

## Inference time — surprise on Alpha

Alpha at half the params is **~2× slower** at forward (656 ms vs 342 ms);
OTC at a quarter the params is **11× faster** (30 ms vs 342 ms). The split
is the tuple count + arity, not the hidden:

- Alpha config has 5 tuple types (c2, c5, w2, w3, w4) — `c5` is the
  longest cycle in the rotation and its enumeration + per-edge
  aggregation per forward dominates.
- OTC config has the same tuple set but at `h=4` and `cap=50 000` (vs
  Alpha's `cap=100 000`), so `c5` enumeration is half. Plus quaternion
  attention at `h=4` runs in a very fast inner kernel.

Open follow-up: the Alpha forward slowdown is unexpected at half the
params. A `py-spy record` on a single forward at this config should
show whether it's enumeration-dominated or matmul-dominated. If
enumeration-dominated, the cycle cache will amortize subsequent calls;
if matmul-dominated, c5 is the bottleneck and the c_n optimization plan
(`docs/plans/2026-05-13-cn-optimization-pattern/`) becomes the natural
follow-up: Hard-Concrete αₖ might prune c5 if c2 + walks are sufficient.

## Test results

- 20 runs, 20 successful (10 Alpha + 10 OTC, all wrote a JSON row to the jsonl).
- The inline aggregator at the end of the queue log had a Python `f-string`
  syntax error from escaped quotes in a HEREDOC-less `python -c`
  (script bug, **not a results bug**); per-trial AUC printlines were
  empty in the log but the final 10-seed aggregate ran fine and the
  jsonl is complete.
- Acceptance per `feedback_n_seed_before_paper_promotion`: **both promotions clear**.

## Performance budget

- Wall: 45 min for 20 runs ≈ 2.25 min/run (Alpha ~4 min/run, OTC ~16 s/run).
- Peak RSS not captured per-run; nothing in this sweep approached the
  16 GB cap (memory `feedback_ulimit_vs_cuda` only fires on much larger
  vision workloads).
- GPU: RTX 2070 SUPER, ~1.5–2 GB peak per trial.

## New / removed dependencies

None.

## Open issues

1. **SOTA table needs updating.** [docs/SOTA_RESULTS.md](../docs/SOTA_RESULTS.md)
   should gain a row for `optuna_best_alpha` / `optuna_best_otc` at h=8/h=4
   with the new 10-seed numbers. Recommended placement: above `joint mix`,
   labelled as the *Optuna-validated lean* recipe.
2. **Book page parity.** `docs/book/src/results/bitcoin-optuna-vs-sota.md`
   currently shows single-trial maxima with a protocol-caveat header; it
   should be re-shaped to lead with the 10-seed mean ± σ and demote the
   single-trial values to a footnote (or remove them entirely now that
   the 10-seed numbers exist).
3. **Alpha forward-time anomaly.** ~2× slowdown at half params is
   unexpected; profile via `py-spy record` to attribute.
4. **The journal version Table I** can now cite `optuna_best_alpha` /
   `optuna_best_otc` as the canonical Bitcoin numbers; joint_mix becomes
   a structural ablation row (no walks-vs-cycle endpoint specialization).

## Experiment provenance

- **Git:** working tree dirty (many WIP paths, see opening session
  `git status`); SHA at launch `0c55fa81d0df99ed6a96566e3317ea122553d6ce`.
- **Python env:** `/home/kyberszittya/miniconda3/bin/python` (torch 2.11.0+cu130,
  optuna 4.8.0, numpy 2.4.4 — drifts from CORE.YAML torch==2.4.1 /
  numpy<2 pins; see `reference_python_envs_for_optuna.md` for the open
  reconciliation item).
- **Dataset hashes:** not captured (the dataset loaders are deterministic
  from `--seed`; could be added in a follow-up jsonl emit).
- **GPU:** NVIDIA RTX 2070 SUPER, 8 GiB, driver 580.126.09.
- **Random seeds:** 0–9, deterministic.
- **Script:** [signedkan_wip/experiments/run_bitcoin_optuna_best_5seed_2026_05_13.sh](../signedkan_wip/experiments/run_bitcoin_optuna_best_5seed_2026_05_13.sh)
  (filename retains `_5seed_` for git-history continuity; loop body
  is 10 seeds).
- **Storage:** [signedkan_wip/experiments/results/bitcoin_optuna_best_5seed_2026_05_13.jsonl](../signedkan_wip/experiments/results/bitcoin_optuna_best_5seed_2026_05_13.jsonl)
  (20 rows, one per (config × seed)).
- **Queue log:** [signedkan_wip/experiments/results/bitcoin_optuna_best_10seed_2026_05_13.queue.log](../signedkan_wip/experiments/results/bitcoin_optuna_best_10seed_2026_05_13.queue.log).

## CORE.YAML items touched

None. All work is in `signedkan_wip/experiments/` (allowlist per
CORE.YAML) and the result is a new jsonl + report under `reports/`.

## Cross-references

- Originating plan: [docs/plans/2026-05-13-cn-optimization-pattern/](../docs/plans/2026-05-13-cn-optimization-pattern/) (the `c_n` extension that would test whether c5 is replaceable by even-higher arities or pruned out by Hard-Concrete αₖ — recommended next phase).
- Handoff that generated the trial configs: [reports/2026-05-13-optuna-handoff-slashdot-continuation.md](2026-05-13-optuna-handoff-slashdot-continuation.md).
- Cross-dataset pattern this confirms: memory `project_walks_epinions_5seed_2026_05_11` (walks > cycles at scale).
