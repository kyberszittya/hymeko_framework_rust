# Gömb joint-mix tune — artifact index (2026-05-12)

One-page pointer so you can pick up after a break without re-reading the full report.

| Kind | Path |
|------|------|
| External AUC programme + Slashdot / inference extension | `reports/2026-05-12-gomb-external-auc-tuning-results.md` |
| Full write-up (joint pilot) | `reports/2026-05-12-gomb-joint-mix-tuning.md` |
| Phase 1 JSONL | `reports/gomb_tune_joint_run.jsonl` |
| Phase 2 JSONL | `reports/gomb_tune_joint_phase2.jsonl` |
| Tuner source | `signedkan_wip/src/run_gomb_tune.py` (`--joint-mix`, `for_joint_mix_tuning`) |
| Smoke runner | `signedkan_wip/src/run_gomb_smoke.py` |
| Tests | `signedkan_wip/tests/test_hymeko_gomb.py` |

**Best scores recorded in JSONL (as of these runs):**

- Phase 1 `bitcoin_otc` **test_auroc ≈ 0.9238** (trial 2 in phase 1 file).
- Phase 2 `bitcoin_otc` **≈ 0.9165**; `bitcoin_alpha` **≈ 0.6970** (phase 2 file; several trials OOM before walk-cap hardening).
- Slashdot compact vanilla pilot: `reports/gomb_tune_external_slashdot_vanilla.jsonl` (includes `infer_*` from `run_gomb_smoke`).
- Slashdot joint: OOM on 8GB in pilots — see `reports/2026-05-12-gomb-external-auc-tuning-results.md`.

**Next command when you resume** (post hardening in `run_gomb_tune`):

```bash
cd /path/to/hymeko_framework_rust && PYTHONPATH=. python -m signedkan_wip.src.run_gomb_tune \
  --datasets bitcoin_otc bitcoin_alpha --joint-mix --trials 12 --search-seed 2 \
  --data-seed 0 --edge-split 80_10_10 --n-epochs 40 --device cuda --timeout-s 7200 \
  --out reports/gomb_tune_joint_phase3.jsonl
```
