# Seminar demo program — build-item 4: forward-latency bench extension

**Date:** 2026-06-11 · **Plan:** `docs/plans/2026-06-10-seminar-demo-program/`
· **Spec:** `docs/INFERENCE_DEMOS_OUTLINE.md` Demo 2

## Summary
Extended the existing forward-latency bench (`run_inference_bench.py`) to measure
the **intra-family lean-vs-joint** gap the slide-18 chart contrasts, instead of
asserting it. Each (dataset, device) cell now times **SGCN**, **HSiKAN-lean**
(hidden 4) and **HSiKAN-joint** (hidden 16). Width is a parametric axis, so it is
a config field (`HSIKAN_VARIANTS`), not a separate model class (CLAUDE.md §6.5
#8); the width-independent cycle pool is built once and reused across variants.

Added the slide-18 renderer `bench_to_png.py` (pure `bars_from_rows` shaping +
matplotlib `render`), re-emitted `inference_bench.json`, and brought the runner
up to contract: median/**IQR/worst** (§3) and a peak-RSS + wall-time exit report
with a 16 GB assert (§4), reusing the seminar `resources.peak_rss_bytes` helper
rather than duplicating it (§6.1).

**⚠ §11 finding — the measured gap contradicts the plan's stated 11×.** See the
dedicated section below. The implementation is honest by construction; the
deck's headline number needs a decision from the user before the slide ships.

## Files touched
**New:**
| LOC | File |
|---:|---|
| 110 | `signedkan_wip/experiments/runs/bench_to_png.py` — slide-18 renderer (pure shaping + draw) |
| 152 | `signedkan_wip/tests/test_inference_bench.py` — 12 tests |

**Modified:**
- `signedkan_wip/experiments/runs/run_inference_bench.py` — split
  `build_hsikan_inputs` → `build_hsikan_data` (width-independent) +
  `build_hsikan_model` (per-width); added `_summarize` (median/IQR/worst),
  `HSIKAN_VARIANTS`, `_bench_sgcn`/`_bench_hsikan`/`_bench_cell`/`_print_summary`
  decomposition, `--datasets` CLI (slashdot opt-in), peak-RSS/wall exit report +
  cap assert. Net behaviour: two HSiKAN rows per cell instead of one; +IQR/worst
  fields on every row.

**Regenerated artifacts:**
- `signedkan_wip/experiments/results/inference_bench.json` — 12 rows (2 graphs ×
  2 devices × {SGCN, HSiKAN-lean, HSiKAN-joint}). Prior version is in git.
- `signedkan_wip/experiments/results/inference_bench_{cpu,cuda}.png`.

## CORE.YAML items touched
**None.** Changes are confined to `experiments/` and `tests/`.

## Measured numbers (seed 0; RTX, CUDA 13.2; this host)
Median forward (ms), two Bitcoin graphs:

| dataset | device | SGCN | HSiKAN-lean (h4) | HSiKAN-joint (h16) | joint/lean |
|---|---|---:|---:|---:|---:|
| bitcoin_alpha | cpu | 15.78 | 107.77 | 404.80 | **3.8×** |
| bitcoin_alpha | cuda | 4.35 | 30.56 | 26.43 | **0.9×** |
| bitcoin_otc | cpu | 20.46 | 96.59 | 331.49 | **3.4×** |
| bitcoin_otc | cuda | 4.88 | 28.24 | 26.81 | **0.9×** |

Param counts: SGCN 135.6k/202.7k; HSiKAN-lean 15.2k/23.6k; HSiKAN-joint
61.1k/94.6k. Cycle pool 25 000 (k4=20k, k5=5k) both graphs.

## ⚠ §11 — measurement contradicts the plan assumption  ·  **RESOLVED**
The spec (INFERENCE_DEMOS Demo 2) and the deck headline state a **≈11×** lean-vs-
joint gap ("30.5 ms lean vs 342 ms joint"). Measured **same-device** with the
spec's own definition (lean = h4, joint = h16), the gap is **3.4–3.8× on CPU**
and **≈1× on CUDA** (at these sizes the GPU is not saturated at h=4, so the wider
h=16 matmul amortises launch overhead and the two are within noise).

**Corrected diagnosis.** My first read called the 11× a *device mismatch*. That
was wrong. The 30.5/342 numbers are a **real same-device** result from
`reports/2026-05-13-bitcoin-optuna-best-10seed.md`:
- `optuna_best_otc` "lean" = **10 layers, h4, 23.8k params, tuple set
  {c2,c5,w2,w3,w4}, cap 50k, quaternion attention** → 30.5 ms.
- `joint_otc` = 5 layers, h16, 94.6k params, {c3,c4,w2,w3} → 342.3 ms.

That report's own analysis (its §"Inference time") states the 11× is **"the tuple
count + arity, not the hidden"** — and it is **OTC-only**: the Alpha optuna
config runs **656 ms, ~2× *slower*** than its joint. So the 11× is an
OTC-specific, tuple-set-confounded artifact, not a general width effect. The
clean **width-only** comparison (h4 vs h16 at a fixed tuple set, same device) is
the ~3.5× CPU / ≈1× CUDA this bench measures — reproducible across both graphs.

**Resolution (user decision, 2026-06-11):** re-label slide 18 / the seminar
talking points to the measured same-device **~3.5× CPU** width gap, keeping the
**accuracy-per-parameter** claim (the real win). The committed SOTA/optuna
records (`docs/SOTA_RESULTS.md`, `sota-snapshot.html`, the 2026-05-13 report)
are left untouched — they correctly document the optuna experiment and its AUC.
Re-labeled: `SEMINAR_DEMO_OUTLINE.md`, `docs/seminar/SEMINAR_SUMMARY.md`,
`signedkan_wip/demos/PRESENTER_RUNBOOK.md`, `docs/INFERENCE_DEMOS_OUTLINE.md`.

## Test results
- `pytest -p no:randomly signedkan_wip/tests/test_inference_bench.py`:
  **12 passed**, 15.1 s. Layers:
  - Unit (pure): `_summarize` median/IQR/worst + invariants + empty/`n_te`
    guards (4); `bars_from_rows` device filter, MODEL_ORDER, unknown-model skip,
    missing-device empty (4); `render`/`main` PNG smoke + empty guard (3).
  - Integration + perf (torch, skip without ml env): lean<joint param count;
    lean forward perf gate (≥5 repeats after warmup, worst≥median, median<2 s).
- Production-scale smoke (the deliverable run): full bench, 2 graphs × 2 devices,
  completed; numbers in the table above.

## Performance results (vs plan budget)
- **Peak RSS 1.49 GB** — budget <3.0 GB ✓ (cap 16 GB ✓).
- **Wall 103.3 s** — budget <120 s ✓ (≥5 repeats: 20 timed + 5 warmup per cell).
- Setup (cycle enumeration, amortised, reported separately): ~15–21 s/cell;
  built **once** per cell and shared by both width variants.

## Static analysis
- `ruff check` (all three files): **All checks passed**.
- `mypy --strict` (`bench_to_png.py`): **Success: no issues found**.
- `run_inference_bench.py` follows the existing untyped `experiments/runs/`
  convention (torch/numpy-heavy script siblings); not newly strict-typed. The
  standalone renderer that does have a reusable API surface is fully typed.

## No §6.5 anti-patterns introduced
- Width variants are a config dict (#1/#5/#8 — parametric → config, not a class
  or a `_h16` function name). Cycle pool built once, no duplicated enumeration
  (#3). One file with a `--datasets` mode arg, no `_v2` proliferation (#13). No
  globals (#11). Reused `resources.peak_rss_bytes` instead of a second RSS
  helper (§6.1). `del`-after-closure pyflakes F821 avoided by function-scoping
  the tensors (`_bench_sgcn`/`_bench_hsikan`) rather than suppressing the lint.

## Provenance
- Git SHA `af803ee` (working tree dirty — this change + prior seminar items).
- Python 3.12; torch 2.12.0+cu132; CUDA available; matplotlib 3.10.9.
- Deterministic: seed 0; fixed caps; no system entropy.
- Datasets: bitcoin_alpha (3 783 nodes), bitcoin_otc.

## Open issues / follow-ups
- **Slide-18 framing: resolved** — re-labeled to the measured ~3.5× CPU width
  gap (see §11 above). Deck `.pptx` slide-18 chart should be regenerated from
  `inference_bench_cpu.png` when the deck is next edited (the talking-point
  source files are corrected; the binary `.pptx` is not in this change).
- Remaining build items: Demo 1 balance (5), Demo 4 mesh + Sinkhorn (6),
  Demo 5 bridge (7).
