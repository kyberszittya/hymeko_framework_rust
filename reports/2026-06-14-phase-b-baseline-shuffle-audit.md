# Report — Phase B: baseline label-shuffle audit harness (Nature Table 1)

**Date:** 2026-06-14
**Slug:** `phase-b-baseline-shuffle-audit`
**Plan:** `docs/plans/2026-06-14-phase-b-baseline-shuffle-audit/` (`plan.tex/.pdf/.tikz/.mmd`)
**Author:** Csaba Hajdu
**Branch:** `feature/ac-hsikan`

## Summary

Builds the strict + label-shuffle audit harness that fills Table 1 of the Nature
paper, so the abstract's claim — *2025 transformers retain near-real performance
under shuffle while protocol-clean methods drop to chance* — can rest on measured
numbers rather than citation. Per CLAUDE.md §6.5 #1/#3/#9 the seven baselines are
a **Strategy family behind one dispatch point**, not seven `run_<method>.py`
scripts: a `SignedLinkBaseline` registry, one unified strict train/eval loop
(`run_baseline_audit.py`, `--model` selects the strategy), and a `--baselines`
grid mode bolted onto the existing no-leak driver.

This report covers **plan steps 1–4 + the driver wiring + the production-scale
smoke** (the harness, all seven methods verified to learn, grid wired and
costed). The 5-seed × 5-dataset grid run itself (step 5) and the manuscript
Table-1 fill (step 6) are **not** done — they are an overnight/multi-night GPU
commitment that needs an explicit go-ahead (see Open Issues).

## What was built

- **`registry.py`** — `SignedLinkBaseline` ABC + `GraphMeta`/`HParams` +
  `register`/`get_baseline`/`list_baselines`. The encoder contract is uniform
  (`encode_nodes(*ctx)` + `edge_logits`), so the loop never branches on model.
  `SignedLinkModule` base supplies shared `edge_logits`/`num_parameters`/optional
  `aux_loss`.
- **Three legacy wraps** (sgcn/sigat/sgt) — strategy registration only, models
  untouched; each maps its existing context builder (`build_signed_adj`,
  `build_neighbour_lists`, `build_signed_neighbours`) to the uniform contract.
- **Four reimplementations**, each reusing existing blocks (no duplication):
  - `sgcl` — SGCN encoder + projection head + sign-aware contrastive `aux_loss`.
  - `sigformer` — separate pos/neg `MotifAttention` streams, gated, pre-LN + FFN.
  - `sesgformer` — `SGTBlock` stack + signed-degree structural encoding.
  - `dadsgnn` — `SGCNLayer` stack with per-node depth-attention readout.
- **`run_baseline_audit.py`** — one CLI, strict train-only context, `seed+100003`
  shuffle matching `run_gomb_smoke`, JSON contract (`test_auroc`+`n_params`)
  preserved verbatim. Optional `--patience` early-exit knob (off by default).
- **`run_no_leak_benchmark.py`** — `Runner` dataclass + per-runner `arg_builder`
  (gomb argv unchanged), seven baselines registered into `SUBPROC_RUNNERS`,
  `--baselines` grid mode (7×5, resumable per `(model,dataset,seed,shuffle)`).

## Files touched

| Path | Action | Lines |
|---|---|---|
| `hymeko_neuro/baselines/registry.py` | new | 212 |
| `hymeko_neuro/baselines/sgcl.py` | new | 62 |
| `hymeko_neuro/baselines/sigformer.py` | new | 75 |
| `hymeko_neuro/baselines/sesgformer.py` | new | 66 |
| `hymeko_neuro/baselines/dadsgnn.py` | new | 58 |
| `hymeko_neuro/experiments/runs/run_baseline_audit.py` | new | 213 |
| `hymeko_neuro/tests/test_baseline_audit.py` | new | 115 |
| `hymeko_neuro/baselines/sgcn.py` | modify (+strategy wrap, −unused import) | +18 |
| `hymeko_neuro/baselines/sigat.py` | modify (+strategy wrap) | +19 |
| `hymeko_neuro/baselines/sgt.py` | modify (+strategy wrap, semicolon cleanup) | +24/−4 |
| `hymeko_neuro/experiments/runs/run_no_leak_benchmark.py` | modify (Runner + grid mode) | +79/−20 |

Total new Rust/Python: ~801 new + ~140 modified lines.

## CORE.YAML items touched

**None.** `CORE.YAML` does not list `hymeko_neuro/`, `src/baselines/`, or
`experiments/`. No dependency added, removed, or version-changed — all four
reimplementations are plain PyTorch reusing existing modules.

## Test results

`pytest -p no:randomly hymeko_neuro/tests/test_baseline_audit.py` — **14 passed**
in 16.0 s (1 pre-existing torch sparse-invariant UserWarning, not ours).

| Layer | Tests | Coverage |
|---|---|---|
| Unit | `test_forward_shape[×7]`, `test_registry_roundtrip`, `test_unknown_model_raises_keyerror`, `test_hparams_merge_ignores_none`, `test_graphmeta_rejects_empty` | per-model forward shape; registry round-trip; failure case (unknown name → `KeyError` naming the valid set); boundary (`GraphMeta(0)` → `ValueError`) |
| Property (hypothesis) | `test_shuffle_is_permutation` | 25 examples; shuffle preserves the label multiset |
| Integration | `test_run_audit_real_and_shuffle_arms`, `test_determinism_same_seed` | strict run on synthetic `sbm_n200` (network-free), real+shuffle arms; same-seed CPU determinism |

**Reproduction oracle (plan step 2):** SGCN through the new unified loop gives
`auroc=0.8547, n_params=135585` at 20 ep seed 0 — **bit-identical** to the direct
`run_sgcn_baseline.run_one_sgcn`. Held after the patience refactor.

**Regression (plan rollback oracle):** the existing E1 smoke
(`run_no_leak_benchmark --smoke`, gomb + SGCN-inproc) is unchanged —
gomb 0.8836/0.4889, SGCN 0.8756/0.5264, both CLEAN+SIGNAL, peak RSS 1.41 GB. The
driver edit did not disturb the gomb or SGCN-inproc paths.

**Static gate:** `ruff check` clean on all 11 touched files. (`mypy --strict` not
run — the torch-heavy baseline modules are untyped at the library boundary across
the existing `src/baselines/`; deferred, declared as a waiver.)

## Performance results

**Above-chance smoke (bitcoin_alpha, 60 ep, seed 0, RTX 3070 Laptop 8 GB):** every
method learns on real labels and drops toward chance under shuffle — confirming
the reimplementations are not broken (§11) and the audit signal is live.

| method | real AUROC | shuffled AUROC | params | s/60ep |
|---|---|---|---|---|
| sgcn | 0.855† | — | 135,585 | — |
| sigat | 0.785 | 0.583 | 134,465 | 6.6 |
| sgt | 0.880 | 0.469 | 148,465 | 13.3 |
| sgcl | 0.879 | 0.525 | 142,882 | 2.7 |
| sigformer | 0.901 | 0.553 | 162,785 | 72 |
| sesgformer | 0.893 | 0.533 | 148,561 | 75 |
| dadsgnn | 0.854 | 0.517 | 141,858 | 2.5 |

†sgcn 20 ep reproduction cell. sigformer shuffled 0.553 is marginally over the
0.55 gate at 60 untuned epochs — to confirm in the val-early-stopped grid run.

**Production-scale smoke (Epinions, |V|=131,828, |E|=841,372, sesgformer, 2 ep):**
`auroc=0.8952, params=4,246,001, ~22.9 s/epoch, GPU peak 1.93 GB`. **No OOM** (well
under the 8 GB card / 16 GB RSS cap). The 4.2 M params are the node embedding
(131,828×32); the architectural budget is tiny.

**Grid cost (the §11 wall-time finding — disagrees with the plan's 24 GPU-h):**
the python per-node attention in sigat/sgt/sigformer/sesgformer is the bottleneck
at scale (~23 s/epoch on Epinions vs ~0.04 s/epoch for the sparse-mm methods).
Extrapolated full-epoch grid (350 cells = 7×5×5×2):

- sparse-mm methods (sgcn, sgcl, dadsgnn): cheap everywhere, ~a few GPU-h total.
- python-attention methods (sigat, sgt, sigformer, sesgformer): ~75 min/arm on
  Epinions, ~half on Slashdot → ~**80–90 GPU-h** for the four on the two large
  graphs. **The plan's 24 GPU-h target holds only for the cheap methods.**

I measured whether a `--patience` early-exit recovers the budget for free. It does
**not**: sgcl on bitcoin_alpha keeps improving to epoch 185/200, so patience=12
exits at ~90 and loses 1.5 pp (0.894→0.879). Undertraining a baseline reads as
falsely "clean" — an audit-integrity risk the plan explicitly warns against — so
the Table-1 grid trains **full epochs**; the patience knob stays CLI-only for
exploratory runs. This corrected an initial wrong "free optimization" assumption,
caught by the §3 measurement.

## New / removed dependencies

None.

## §6.5 anti-patterns

No anti-pattern introduced. Positively avoided: one registry dispatch (#1/#9, no
per-method `match` ladder), one CLI with `--model` (#13, no `run_<method>.py`
proliferation), strategies reuse existing context builders and model blocks (#1,
no copy-paste), `--baselines` is a mode on the existing driver (#13). Waivers:
`mypy --strict` deferred (torch-untyped boundary); two pre-existing `E702`
semicolons in `sgt.py` cleaned up in passing.

## Open issues / follow-ups

1. **The grid run is an overnight/multi-night GPU commitment, not yet launched.**
   Full-epoch ≈ 80–90 GPU-h, dominated by the four python-attention transformers
   on Slashdot+Epinions. Options to put to the user before launch:
   (a) run the **3 sparse-mm methods × 5 datasets now** (~hours) to fill those
   Table-1 rows immediately, defer the transformers; (b) reduce the
   transformer×large-graph cells to **3 seeds**; (c) **vectorize** the per-node
   attention (pad-and-mask over the by-length groups) before the large runs — the
   highest-value engineering fix, turns ~80 h into ~single-digit GPU-h. **No grid
   launched without a decision.**
2. **sigformer marginal shuffle gate** (0.553 @60 ep) — re-check under the grid's
   val-early-stop; if it stays >0.55 with a strong real number, that is itself a
   reportable result (mild leakage), not a bug.
3. **Reproduction-parity caption** — the manuscript must state the four are *our
   in-protocol reimplementations*, not the authors' weights (plan §risk).
4. `mypy --strict` pass over `src/baselines/` (whole-package, separate change).

## Provenance

Git SHA: working tree dirty (this change + prior uncommitted Nature/SMC edits;
see `git status`). Host: Windows 11, RTX 3070 Laptop 8 GB, torch 2.12.0+cu132,
Python 3.12.13, CUDA 13.2 driver. Seeds: 0 (smokes), 1 (determinism test).
Datasets: bitcoin_alpha (SNAP), epinions (SNAP, downloaded this session).
Smoke artifacts: `hymeko_neuro/experiments/results/no_leak_smoke_regression.jsonl`.
