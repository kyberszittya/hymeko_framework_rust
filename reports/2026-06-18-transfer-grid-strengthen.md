# Strengthen the inductive-transfer grid — harder pairs push the learned increment past σ

**Date:** 2026-06-18
**Plan:** [docs/plans/2026-06-18-transfer-grid-strengthen](../docs/plans/2026-06-18-transfer-grid-strengthen/) (4 artifacts; PDF compiles).
**Status:** ✅ driver consolidation (3-arm decomp into the canonical driver, latent resumption-bug fixed) + 5 new tests + the 120-cell harder-pair grid. **Result: a real strengthening** — the learned-from-source increment is **positive in 8/8 cells** and **clears 1σ_shuffle in 7/8**, reaching **+2.9σ** on the cleanest pair (vs the bitcoin-pair's borderline ~1–1.5σ).

## Summary

The inductive-transfer follow-up (BACKLOG `▶ NEXT STEP`): push the *learned-from-A*
transfer increment — real − max(shuffle, random-init) — past the shuffle arm's σ on
**harder, more-distinct graph pairs**. Run the 3-arm decomposition on train-small
(bitcoin) → eval-large (slashdot/epinions), 5 seeds.

Two things were needed and done:

1. **Consolidate the decomposition into the driver (+ fix a latent bug).** The
   original 60-row `inductive_transfer_decomp_ab.jsonl` was produced by an ad-hoc loop
   no longer in the driver; the driver's grid wrote only 2 arms and its resumption key
   `_row_key` *collided the real arm with the random-init arm* (both `shuffle=False`),
   so on resume the random-init arm would be silently dropped. Added an `Arm`
   enum {REAL, SHUFFLE, RANDINIT}, unified the 2-arm and 3-arm grids into one
   `run_grid(arms=…)` loop (no loop dup, §6.5 #3), made `_row_key` arm-aware (derives
   the arm from `(shuffle, trained)` — backward-compatible with 2-arm files), and added
   a `--decomp` mode. A regression test pins the fix.

2. **The result.** On the harder pairs the increment is no longer borderline:

| model | pair | real | shuffle | random-init | learned | σ_sh | significance |
|---|---|---|---|---|---|---|---|
| cayley_rotor | otc→epinions | 0.8927±0.005 | 0.8665±0.009 | 0.470±0.151 | **+0.0261** | 0.009 | **+2.90σ** |
| cayley_rotor | otc→slashdot | 0.8497±0.003 | 0.8282±0.009 | 0.516±0.080 | **+0.0215** | 0.009 | **+2.28σ** |
| cayley_rotor_walk | otc→epinions | 0.8939±0.006 | 0.8418±0.026 | 0.461±0.156 | **+0.0520** | 0.026 | **+2.02σ** |
| cayley_rotor_walk | otc→slashdot | 0.8461±0.005 | 0.8050±0.025 | 0.545±0.077 | +0.0411 | 0.025 | +1.62σ |
| cayley_rotor_walk | alpha→epinions | 0.8894±0.004 | 0.8480±0.030 | 0.461±0.156 | +0.0414 | 0.030 | +1.40σ |
| cayley_rotor | alpha→epinions | 0.8916±0.006 | 0.8254±0.060 | 0.470±0.151 | +0.0663 | 0.060 | +1.11σ |
| cayley_rotor | alpha→slashdot | 0.8467±0.004 | 0.7888±0.053 | 0.516±0.080 | +0.0579 | 0.053 | +1.10σ |
| cayley_rotor_walk | alpha→slashdot | 0.8444±0.003 | 0.8092±0.037 | 0.545±0.077 | +0.0352 | 0.037 | +0.94σ |

5-seed mean ± pstdev, strict deduped held-out on the eval graph, `n_test`=63 365
(epinions) / 38 432 (slashdot). `inductive_transfer_decomp_hard.jsonl` (120 rows).
**learned > 0 in 8/8 cells; ≥ 1σ_shuffle in 7/8; mean learned = +0.0427.**

**Measured / inferred / hypothesis (CLAUDE.md).**
- *Measured:* the 8-cell table; 8/8 positive; 7/8 ≥ 1σ; otc-trained cells +2.0–2.9σ.
- *Inferred:* the **variance, not the effect size, is what separates the train graphs.**
  Training on **bitcoin_otc** yields a tight shuffle distribution (σ_sh ≈ 0.009)
  → its transfer claim is statistically clean (+2.3–2.9σ at a *smaller* raw increment).
  Training on **bitcoin_alpha** gives a noisier shuffle arm (σ_sh ≈ 0.03–0.06) → a
  *larger* raw increment (+0.058–0.066) but softened to ~1.1σ. So the otc-trained pairs
  are the honest "clears σ" evidence; the alpha-trained pairs corroborate in direction.
- *Confirmed (not new):* `max(shuffle, random-init) = shuffle` in every cell — the
  random-init floor is at/below chance (0.46–0.55, high-variance), reproducing the
  bitcoin-pair finding that the untrained readout is pathologically anti-aligned. So
  learned = real − shuffle throughout.
- *Honest ceiling:* one cell (alpha→slashdot walk) sits at +0.94σ, just under. The claim
  is made on the otc-trained pairs and the 7/8 ≥ 1σ tally, not on a uniform 2σ sweep.

**What this buys the line.** The bitcoin-pair report could only call the learned
increment "suggestive (~1–1.5σ)". On harder train-small/eval-large pairs it is
**significant on the clean (otc-trained) pairs (+2.3–2.9σ)** and positive everywhere.
The rotor carries **real transported signed structure** from the source graph — not
merely exposing the eval graph's own prior (which the shuffle arm already captures at
0.79–0.87). This is the distinctiveness claim the transductive baselines structurally
cannot make (they cannot index an unseen graph at all).

## Files touched

**Modified (2, mine — both created earlier today, uncommitted):**
- `hymeko_neuro/experiments/runs/run_inductive_transfer.py` (+~95 / −~40): `Arm`
  enum (maps each arm to its `(shuffle_train_signs, train)` pair; `Arm.of` inverse);
  `ARMS_GATE`/`ARMS_DECOMP`, `HARD_PAIRS`, `DECOMP_MODELS`; unified `run_grid(models,
  pairs, seeds, arms, out, *, label)` with a captured `cell` closure (resume-or-compute
  per arm); arm-aware `_row_key` (fixes the real↔random-init collision); `main` now
  delegates to `run_grid`; new `main_decomp` + `--decomp`/`--pairs`/`--models` CLI;
  `_parse_pairs`. `transfer_cell` **unchanged** (preserves the existing tests).
- `hymeko_neuro/tests/test_inductive_transfer.py` (+~55): `Arm` round-trip
  (parametrized over all arms), arm-flag mapping, and the **resumption regression**
  (`run_grid(ARMS_DECOMP)` keys 3 arms distinctly and resumes idempotently — fails
  under the old shuffle-bool key where real≡random-init).

**New artifacts:**
- `docs/plans/2026-06-18-transfer-grid-strengthen/{plan.tex,plan.pdf,plan.tikz,plan.mmd}`.
- `hymeko_neuro/experiments/results/inductive_transfer_decomp_hard.jsonl` (120 rows).
- `hymeko_neuro/experiments/results/decomp_hard_run.log` (grid stdout).

**CORE.YAML items touched:** none. No new dependency. Datasets cached
(`slashdot.txt`, `epinions.txt` present) → no network, no persistent-state mutation.

## Test results
- `test_inductive_transfer.py` **10 ✓** (5 prior + 5 new); `test_baseline_audit.py`
  **16 ✓** (refactor oracle, untouched). `pytest -p no:randomly`, 9.7 s.
- `ruff check`: clean on both touched files. `mypy --strict` (changed file,
  `--follow-imports=silent`): clean. (`-p`-style whole-tree mypy surfaces only
  pre-existing debt in untouched modules — `meshes.py`, `run_baseline_audit.py` — as
  the inductive-transfer report already noted.)
- `radon cc`: `run_grid` B(10), `main` C(11), `transfer_cell` C(12, pre-existing,
  unchanged). All **warns** (>10), all under the §6.2 fail-ceiling (15); the closure
  refactor cut `run_grid`'s **nesting depth 5 → 3**.

## Performance
- **Production smoke (§3)** before the grid: heaviest cell `bitcoin_alpha→epinions`,
  walk variant (sparse A^k over 131 k nodes), all 3 arms — **peak RSS 1.895 GB**
  (Windows working-set via `K32GetProcessMemoryInfo`, no new dependency), wall **24 s**
  for the 3-arm group (real 13 s, shuffle 5 s, random-init 4 s). Well under the 8 GB
  plan budget and the 16 GB cap (§4).
- Full grid: 120 cells, ~12 min wall, checkpointed/resumable jsonl. No regression
  claim (new experimental path).

## §6.5 anti-patterns
None introduced. The grid loop is single-sourced (`run_grid` serves both the 2-arm gate
and 3-arm decomp, §6.5 #3); the new axis (decomposition arms) is an **enum + mode arg**,
not a Cartesian of wrapper functions (§6.5 #1/#13); no globals (the `cell` closure
captures `rows`/`done`/`out_path` explicitly, §6.5 #11); transductive failure still
caught with a *specific* exception type (§6.4). The latent `_row_key` collision is the
opposite of an anti-pattern fix — it was a silent-skip contract gap, now closed + tested.

## Experiment provenance
- Git SHA `7d16ad0` (tree dirty; the two touched files are uncommitted from today's
  session, listed above). Datasets: cached SNAP `bitcoin_alpha`/`bitcoin_otc` (train),
  `slashdot`/`epinions` (eval). Seeds 0–4. Device CUDA (torch 2.12.0+cu132). Recipe:
  each baseline's `default_hparams`, `n_epochs=120`, strict deduped held-out on the eval
  graph; per-arm controls as in the driver.

## Open issues / follow-ups
- **The alpha-trained shuffle arm is noisy (σ_sh ≈ 0.05–0.06).** A higher seed count
  (10) on the two alpha→* cells would settle whether they too clear 1σ, or whether the
  alpha-source shuffle distribution is intrinsically wider (worth a one-line note in any
  write-up: "significance is train-graph-dependent via the gate's variance").
- **Write-up.** The line now has a defensible distinctiveness claim with a significant
  number (otc→epinions +0.0261, +2.90σ). Folding this into the reframed structural-
  feature signed-link narrative (BACKLOG `(b)`) is the natural next move.
- **Node-holdout (within-graph) inductive** — the harder secondary in the parent plan;
  still unrun.
