# Gömb-unrestricted Epinions + HymeYOLO kcycle: dispatch report

**Date:** 2026-05-14
**Status:** runs in flight; numbers to be appended on completion.

---

## 1. Summary

Two GPU runs queued under cgroups v2 (`systemd-run --user -p MemoryMax=16G`) on
RTX 2070 SUPER:

1. `gomb-epinions-unrestricted-2026-05-14.service` (PID 216243)
   — 5-seed Epinions v5_combined config under the new
   `--unrestricted-cycles` flag. Brackets the strict-protocol
   0.9526 ± 0.0018 number from above.
2. `hymeyolo-kcycle-2026-05-14b.service` (PID 216707)
   — 5-seed HymeYOLO `+ricci+kcycle` on Cluttered MNIST.
   Sits in the script's wait-for-Gömb loop; takes the GPU after
   the Epinions run finishes.

Plan: [docs/plans/2026-05-14-gomb-unrestricted/plan.tex](../docs/plans/2026-05-14-gomb-unrestricted/plan.tex).

## 2. Files touched

| File | LOC delta | Notes |
|---|---|---|
| [signedkan_wip/src/run_gomb_smoke.py](../signedkan_wip/src/run_gomb_smoke.py) | +19 / −3 | `--unrestricted-cycles` flag + cycle-edge-set rebinding |
| [signedkan_wip/tests/test_gomb_unrestricted_flag.py](../signedkan_wip/tests/test_gomb_unrestricted_flag.py) | +110 / 0 | 5 unit tests (flag wiring + synthetic-triangle bridge test) |
| [signedkan_wip/experiments/run_gomb_epinions_unrestricted_2026_05_14.sh](../signedkan_wip/experiments/run_gomb_epinions_unrestricted_2026_05_14.sh) | new | 5-seed runner, v5_combined config, 80 epochs |
| [docs/plans/2026-05-14-gomb-unrestricted/](../docs/plans/2026-05-14-gomb-unrestricted/) | 4 files | plan.tex / plan.pdf / plan.tikz / plan.mmd |

## 3. CORE.YAML items touched

None. `signedkan_wip` is not listed in `CORE.YAML`.

## 4. Test results

```
$ python -m pytest signedkan_wip/tests/test_gomb_unrestricted_flag.py -v
test_unrestricted_flag_is_registered             PASSED [ 20%]
test_strict_default_path_uses_train_edges        PASSED [ 40%]
test_unrestricted_branch_uses_full_edges         PASSED [ 60%]
test_cycle_enumeration_uses_e_cyc_not_e_tr       PASSED [ 80%]
test_unrestricted_pool_is_strictly_larger_synthetic  PASSED [100%]
========================= 5 passed in 1.93s =========================
```

## 5. Production-scale smoke

Bitcoin Alpha, seed=0, 20 epochs, `--unrestricted-cycles` ON, joint-mix
v5-combined-shape config:

| Property | Strict (5-seed mean, 80 ep) | Unrestricted (1-seed, 20 ep) |
|---|---|---|
| Cycle pool edges | 19 348 | **24 186** (+25 %) |
| `[protocol]` line | `STRICT` | `UNRESTRICTED (transductive)` |
| Joint-mix tuples (total) | n/a | 156 578 |
| val_AUROC | 0.91+ at 80 ep | 0.7928 at 20 ep |
| test_AUROC | 0.8972 ± 0.0079 | 0.8057 at 20 ep |
| Wall (this seed) | n/a | 6.8 s |
| RSS | n/a | 571 MB peak (cgroups-tracked) |

Smoke confirms (i) the protocol line flips to UNRESTRICTED, (ii) the
cycle pool grows by exactly the train-vs-full edge ratio (5/4), and
(iii) training converges. The 20-epoch number is not the production
endpoint — just the wiring sanity gate.

## 6. Performance results

Cgroups v2 RSS cap: 16 GB MemoryMax per `systemd-run` (within CORE.YAML
§4 budget).

| Service | Active since | Current RSS | Memory cap |
|---|---|---|---|
| `gomb-epinions-unrestricted-2026-05-14` | 16:16 CEST | 405 MB → ~6 GB at peak | 16 GB |
| `hymeyolo-kcycle-2026-05-14b` | 16:18 CEST (waiting) | 1.4 MB | 16 GB |

Wall-time estimate (from strict-protocol baseline):
- Gömb Epinions: 6.5 min/seed × 5 = ~33 min, plus ~25 % cycle-pool overhead.
- HymeYOLO kcycle: 760 s/seed × 5 = ~63 min (from this morning's run).

Total expected: ~100 minutes wall.

## 7. New / removed dependencies

None.

## 8. Open issues and follow-up items

- Headline 5-seed Epinions unrestricted AUROC: pending.
- Audit-sanity check under `--unrestricted-cycles --shuffle-train-signs`:
  pending. Expected behaviour: AUC drops, but not to chance (test-edge
  signs are still real, so the leakage path is still open even with
  shuffled training signs).
- Report update: prepend final numbers to this file once both
  services finish.

## 9. Experiment provenance

- Git SHA (working tree dirty — uncommitted plan + flag): `2ccaa4d12f...`
- OS: Linux 6.17.0-23-generic (kyberszittya@gemeauxrapace)
- Hardware: NVIDIA RTX 2070 SUPER, 8 GB VRAM, 2019
- Python: /home/kyberszittya/miniconda3/bin/python (drift from CORE
  torch 2.4.1; PyTorch in miniconda env is 2.11). Same env used for
  strict 5-seed run, so comparison is iso-env.
- Seeds: 0, 1, 2, 3, 4 (same as strict-protocol Epinions).
- Datasets: Epinions (131 828 vertices / 841 372 edges); Cluttered
  MNIST (5000 images, 50 epochs).
- Random seeds fixed via `torch.manual_seed(args.seed)` and
  `np.random.seed(args.seed)`.

## 10. Reproducibility command

```bash
# Strict (baseline):
python -m signedkan_wip.src.run_gomb_smoke --dataset epinions --seed 0 \
    --n-epochs 80 --edge-split 80_10_10 --joint-mix --device cuda \
    --d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
    --n-tiers 3 --topk 64 --lr 0.003

# Unrestricted (this work):
python -m signedkan_wip.src.run_gomb_smoke --dataset epinions --seed 0 \
    --n-epochs 80 --edge-split 80_10_10 --joint-mix --device cuda \
    --unrestricted-cycles \
    --d-embed 32 --M-outer 8 --d-outer 8 --d-middle 8 --d-core 8 \
    --n-tiers 3 --topk 64 --lr 0.003
```

The only difference between strict and unrestricted is the
`--unrestricted-cycles` flag, which rebinds the cycle-pool edge set
from `(e_tr, s_tr)` to `(g.edges, g.signs)`. Everything downstream
— architecture, training loss, test-edge AUC — is identical.

---

*End of dispatch report. Numbers to be appended on completion.*
