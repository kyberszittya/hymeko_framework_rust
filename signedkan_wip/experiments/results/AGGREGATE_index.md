# `signedkan_wip/experiments/results/` aggregate index

Primary on-disk ledger for SignedKAN / HSiKAN / graph experiments (alongside `reports/` orchestration logs). **150 files** in this snapshot — multiple programmes in parallel (Phase 7 Slashdot/Bitcoin grids, Phase 8 overnight, Epinions/SGT/SOTA chases, EC/prune tiers, joint-mix Bitcoin, architecture master table). Treating “one headline number” as the whole story **drops** most of this tree.

Companion index: `reports/AGGREGATE_index.md`.

**Evidence / tone contract (read before arguing scores):** `docs/RESULTS_DISCIPLINE.md` — canonical paths, anchored numbers, joint vs phase-8 distinction, health line.

---

## 1. Volume by extension

| ext | count |
|-----|-------|
| json | 105 |
| jsonl | 39 |
| md | 2 |
| log | 4 |

**Total:** 150 files (`find signedkan_wip/experiments/results -type f`).

---

## 2. Layout

| location | files | role |
|----------|-------|------|
| `.` (root) | 136 | sweeps, phase outputs, SOTA jsonl, EC JSON, `master_table.md`, `positivity_summary.md`, logs |
| `phase3_sweep/` | 13 | Bitcoin alpha/OTC h16/h32 per-seed JSON + sweep aggregates + `.log` |
| `ablation/` | 1 | `ablation_bitcoin_alpha_h32.json` |

---

## 3. Human-readable syntheses (read these first)

| file | contents |
|------|----------|
| **`master_table.md`** | Multi-arch × dataset grid (karate, SBM, hier, bitcoin_*, slashdot): mean±std AUC, F1, best-arch bullets, learned αₖ for HSiKAN variants. |
| **`positivity_summary.md`** | Phase 11 / 11b SBM positivity sweep: `pos_in` grid, per-arch AUC, best-arch per row, αₖ traces. |

---

## 4. Largest JSONL streams (line counts = JSON objects, non-blank)

| file | lines |
|------|------:|
| `phase8_overnight_grid.jsonl` | 612 |
| `overnight_camera_ready.jsonl` | 75 |
| `phase7_slashdot_sweep.jsonl` | 45 |
| `joint_mix_5seed_2026_05_08.jsonl` | 39 |
| `phase9_slashdot_sota.jsonl` | 39 |
| `final_table.jsonl` | 35 |
| `phase7_arity_sweep_karate.jsonl` | 27 |
| `phase7_arity_karate_noleak.jsonl` | 27 |
| `phase7_grid_lr_sweep.jsonl` | 24 |
| `phase7_arity_sbm_noleak.jsonl` | 18 |
| `phase7_arity_sbm_leaky.jsonl` | 18 |
| `phase7_vadj_bitcoin_gpu.jsonl` | 18 |
| (remaining jsonl ≤ 15 lines each) | … |

---

## 5. Programme clusters (by filename stem — not mutually exclusive)

- **Phase 7 — Slashdot / Bitcoin / arity / spectral / vertex-adj / LR grid:** `phase7_slashdot*.json(l)`, `phase7_bitcoin_directed.jsonl`, `phase7_vadj_bitcoin*.jsonl`, `phase7_arity_*.json(l)`, `phase7_grid_lr_sweep.jsonl`, `phase7_slashdot_5seed_k3_*.json`, `k4_speed_benchmark.json`, `phase7_arity_bnb_sbm_n200_k4_s0.json`, etc.
- **Phase 8 — Bitcoin 5-seed panel + overnight grid:** `phase8_bitcoin_5seed.json`, `phase8_bitcoin_5seed_2026-05-12.json`, `phase8_overnight_grid.jsonl`, `phase8_sbm_positivity.json`, `phase8_sota_chase.jsonl`, `.log` beside dated runs.
- **Phase 9 — Slashdot SOTA line:** `phase9_slashdot_sota.jsonl`, `phase9_k345_mixed.json`.
- **Joint / edge / kernel / aux — dated 2026-05-08/09:** `joint_mix_5seed_2026_05_08.jsonl`, `slashdot_*2026_05_0*.jsonl`, `epinions_*2026_05_0*.jsonl`, `slashdot_edge_cr_*`, `slashdot_aux_entropy_*`, `slashdot_d16_joint_mix_*`.
- **SGT baselines / sweeps:** `sgt_epinions.jsonl`, `sgt_slashdot.jsonl`, `sgt_sweep.jsonl`.
- **Entropy / Catmull / highway / EC family:** `entropy_*.json`, `catmull_ec.json`, `highway_catmull*.json`, `spectral_ec.json`, `mixed_ec.json`, `multilayer_*.json`, `tier*_*.json`, `refined_entropy.json`, …
- **Prune / distill / structural:** `prune_*.json`, `structural_prune*.json`, `iter_prune.json`, `hsikan_*prune*.json`, …
- **HSiKAN hyperparameter / genetic / clip / depth:** `hsikan_*.json` (15+ distinct stems).
- **Phase 1 / 2 / 4 / 5 / 6 panels:** `phase1_*.json(l)`, `phase2_mixed.json`, `phase4_apples_to_apples.json`, `phase5_arch_panel.json`, `phase6_small_synth.json`, `phase11*_*.json`.
- **Overnight umbrella:** `overnight.json`, `overnight_camera_ready.jsonl`, `overnight_camera_ready.log`.

---

## 6. Cross-links

| need | path |
|------|------|
| Thesis IV views suite (entropy reg, 111 rows) | `RESULTS_VIEWS_SUITE.md` (repo root) |
| Rust / ABB / Gömb / HSiKAN harness narrative | `reports/*.md`, `reports/AGGREGATE_index.md` |
| Orchestrated overnight JSON+err | `reports/overnight_2026_05_11*/` |

---

## 7. Full manifest (sorted relpaths)

Paths relative to `signedkan_wip/experiments/results/`.

```
ablation/ablation_bitcoin_alpha_h32.json
attention_ec.json
bilinear_ec.json
bitcoin_alpha_h8_seed0.json
catmull_ec.json
compare_h32_alpha_500ep.json
compare_h32_alpha.json
compare.json
cross_branch_ec.json
density_sweep.json
early_stop.json
entropy_on_ec.json
entropy_sweep.json
epinions_balance_5seed_2026_05_10.jsonl
epinions_edge_cr_5seed_2026_05_09.jsonl
epinions_overnight_2026_05_09.jsonl
final_table.jsonl
focused_stack.json
full_recipe_fixed.json
full_recipe.json
gap_sweep.json
highway_catmull_500ep.json
highway_catmull.json
hsikan_clip.json
hsikan_depth_cr_200ep.json
hsikan_depth_cr.json
hsikan_genetic_alpha.json
hsikan_genetic.json
hsikan_hpsweep.json
hsikan_init05_prune.json
hsikan_init_clip.json
hsikan_optim.json
hsikan_prune.json
inference_bench.json
iter_prune.json
joint_mix_5seed_2026_05_08.jsonl
k4_speed_benchmark.json
l1_sparsity.json
master_table.md
minibatch_ec.json
mixed_ec.json
multi_domain_perf_bench.json
multi_domain_perf_deep.json
multilayer_ec.json
multilayer_entropy.json
multilayer_skip.json
multilayer_v2.json
multilayer_v3.json
ntuples_mixed_alpha_k34.json
ntuples_mixed_alpha_k3only.json
overnight_camera_ready.jsonl
overnight_camera_ready.log
overnight.json
phase11b_k45_positivity.json
phase11_positivity_sweep.json
phase1_hymeyolo_circles_ricci_2026_05_11.jsonl
phase1_lean.json
phase2_mixed.json
phase3_redundancy.json
phase3_sweep/bitcoin_alpha_h16_seed0.json
phase3_sweep/bitcoin_alpha_h16_seed1.json
phase3_sweep/bitcoin_alpha_h16_seed2.json
phase3_sweep/bitcoin_alpha_h32_seed0.json
phase3_sweep/bitcoin_alpha_h32_seed1.json
phase3_sweep/bitcoin_alpha_h32_seed2.json
phase3_sweep/bitcoin_otc_h32_seed0.json
phase3_sweep/bitcoin_otc_h32_seed1.json
phase3_sweep/bitcoin_otc_h32_seed2.json
phase3_sweep/sweep_bitcoin_alpha.json
phase3_sweep/sweep_bitcoin_alpha.log
phase3_sweep/sweep_bitcoin_otc.json
phase3_sweep/sweep_bitcoin_otc.log
phase4_apples_to_apples.json
phase5_arch_panel.json
phase6_small_synth.json
phase7_arity_bnb_sbm_n200_k4_s0.json
phase7_arity_karate_noleak.jsonl
phase7_arity_sbm_leaky.jsonl
phase7_arity_sbm_noleak.jsonl
phase7_arity_sweep_karate.jsonl
phase7_bitcoin_directed.jsonl
phase7_grid_lr_sweep.jsonl
phase7_slashdot_5seed_k3_100k.json
phase7_slashdot_5seed_k3_200k.json
phase7_slashdot_5seed_k3_300k.json
phase7_slashdot_5seed_k3_full.json
phase7_slashdot_arity.jsonl
phase7_slashdot_fast_calib2.jsonl
phase7_slashdot_fast_calib.jsonl
phase7_slashdot.json
phase7_slashdot_k34_batched_200k.json
phase7_slashdot_k34_pilot.json
phase7_slashdot_noleak.jsonl
phase7_slashdot_pruning_h16_k4_200k.json
phase7_slashdot_spectral.jsonl
phase7_slashdot_sweep.jsonl
phase7_vadj_bitcoin_gpu.jsonl
phase7_vadj_bitcoin.jsonl
phase8_bitcoin_5seed_2026-05-12.json
phase8_bitcoin_5seed_2026-05-12.log
phase8_bitcoin_5seed.json
phase8_overnight_grid.jsonl
phase8_sbm_positivity.json
phase8_sota_chase.jsonl
phase9_k345_mixed.json
phase9_slashdot_sota.jsonl
positivity_summary.md
protocol2_comparison.json
prune_distill_cr.json
prune_distill_extended.json
prune_distill.json
prune_distill_kb.json
prune_perf_alpha_cr.json
r2_compositions.json
refined_entropy.json
saturation.json
sgcn_baseline.json
sgt_epinions.jsonl
sgt_slashdot.jsonl
sgt_sweep.jsonl
sinusoid_controls.json
sinusoid_controls_smoke.json
skip_heterogeneous.json
slashdot_5seed_2026_05_08.jsonl
slashdot_aux_entropy_2026_05_09.jsonl
slashdot_c5_5seed_2026_05_08.jsonl
slashdot_d16_joint_mix_2026_05_09.jsonl
slashdot_ec.json
slashdot_edge_cr_5seed_2026_05_09.jsonl
slashdot_edge_cr_kernel_on_2026_05_09.jsonl
slashdot_k2345.jsonl
slashdot_long_train.jsonl
slashdot_push_2026_05_08.jsonl
slashdot_sota_chase.jsonl
slashdot_sota_final.jsonl
slashdot_sota_push.jsonl
spectral_ec.json
structural_prune.json
structural_prune_otc.json
structural_prune_slashdot.json
structural_prune_slashdot_sota.json
tier1_probe.json
tier1_wd_sweep.json
tier2_optclip_sweep.json
tier2_stride_probe.json
tier3_coef_entropy.json
tier4_r2norm.json
tier5_kl_sweep.json
tier6_smooth_sweep.json
triad_loss.json
```

---

## 8. Regenerate

```bash
find signedkan_wip/experiments/results -type f | wc -l
find signedkan_wip/experiments/results -type f | sed 's|.*/signedkan_wip/experiments/results/||' | sort
find signedkan_wip/experiments/results -name '*.jsonl' -type f -exec wc -l {} + | sort -n
```
