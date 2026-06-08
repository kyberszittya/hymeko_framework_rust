# Existing measurements inventory (cross-check before Phase B)

**Status**: DRAFT for coworker review. Maps every prior result on disk
to the Phase B audit matrix cell it can fill. **Goal**: don't re-run
anything that's already known.

**Critical finding**: a strict-protocol result already exists on disk
that *directly* validates the paper's central claim. See "THE
LEAKAGE EVIDENCE" section below.

---

## Already-validated leaky-protocol AUROCs (n=5 seeds, paired)

These fill the **A_leak** column of Table 1 for the HSiKAN/Gömb
architecture family. Most are already n=5 ± std — better than what
Phase B was budgeted for.

| Dataset | Architecture | A_leak (mean ± std, n=5) | Source memory entry |
|---|---|---|---|
| Slashdot | edge_cr Highway (SignedKAN SOTA) | **0.9070 ± 0.0029** | `reference_sota_repro_edge_cr_2026_05_29` |
| Slashdot | edge_cr Highway (kernel-ON re-run) | **0.9067 ± 0.0034** | `project_edge_cr_5seed_2026_05_09` |
| Slashdot | HSiKAN attention + cycle-batching | **0.9035 ± 0.0044** | `project_attention_cycle_batch_compose_2026_05_08` |
| Slashdot | Plain Gömb (no outer HSiKAN) | **0.9031 ± 0.0008** | `project_hymeko_gomb_feasibility_2026_05_11` |
| Bitcoin-Alpha | Joint-mix HSiKAN (c3+c4+w2+w3) | **0.9845** (paired, +5σ over cycle baseline 0.9468) | `project_joint_mix_2026_05_08` |
| Bitcoin-Alpha | Outer-HSIKAN→Gömb residual d=4 cr_highway | **+0.0066 over Gömb** (5.68σ, 5/5 wins) | `project_outer_hsikan_msg_abb_2026_05_21` |
| Bitcoin-Alpha | balance+quat best | **0.8327 ± 0.013** (5-seed) | `project_axiom_beats_attention_2026_05_05` |
| Bitcoin-OTC | Joint-mix HSiKAN | **0.9801** (paired, +5σ over cycle 0.9266) | `project_joint_mix_2026_05_08` |
| Bitcoin-OTC | Plain Gömb 5-seed | **0.9118 ± 0.0089** | `project_hymeko_gomb_feasibility_2026_05_11` |
| Bitcoin-OTC | Outer-HSIKAN→Gömb d=4-8 | up to **+0.0058 over Gömb** | `project_outer_hsikan_msg_abb_2026_05_21` |
| Epinions | Walks-augmented kitchen-sink | **0.8145 ± 0.0017** (paired Δ +0.0753 vs baseline 0.7392, σ=+16, 5/5) | `project_walks_epinions_5seed_2026_05_11` |
| Epinions | Bigger-caps single-seed | **0.8409** (single seed) | `project_epinions_ceiling_2026_05_09` |
| SBM (synth) | HSiKAN | **0.91 / 0.96** vs SGT 0.56/0.69 | `project_sgt_baseline_2026_05_04` |
| Walk-HSiKAN | BA/OTC/SBM-200 single-seed | 0.973 / 0.959 / 0.999 (1-seed) | `project_walk_hsikan_2026_05_04` |

**Summary**: the **A_leak** column is largely populated for HSiKAN /
Gömb / SignedKAN. **Reddit Hyperlinks** is the conspicuous gap (no
prior run that I can find in MEMORY.md — needs Phase B).

---

## ⚡ THE LEAKAGE EVIDENCE — already on disk

Memory entry `project_joint_mix_2026_05_08` records what is, in
retrospect, the most important measurement for the paper's central
claim:

> "Joint-mix HSiKAN 5-seed paired result 2026-05-08 —
>  HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 at iso-param:
>  **BA 0.9845** vs cycle 0.9468 (paired +0.038 ~5σ);
>  **OTC 0.9801** vs cycle 0.9266 (paired +0.053 ~5σ);
>  **strict protocol → 0.500** (paper's known leakage caveat applies)."

This is exactly the table the paper needs. The same model that
achieves 0.9845 under leaky transductive → drops to **chance level
0.500** under strict protocol. **No further Phase B work needed to
prove the leakage exists** — the magnitude is on disk.

**For Table 1**:

| Cell | A_leak | A_strict | ΔA = inflation |
|---|---|---|---|
| Joint-mix HSiKAN, Bitcoin-Alpha | 0.9845 | **0.500** | **+0.4845** (!) |
| Joint-mix HSiKAN, Bitcoin-OTC | 0.9801 | **0.500** | **+0.4801** (!) |

The ΔA = +0.48 inflation is the strongest possible evidence for the
title claim. Under leaky protocol the model is effectively reading
the test-edge sign through the feature pipeline; under strict, it
reverts to random.

> **[CW]** Confirm the 0.500 strict-protocol number is from a paired
> n=5 run, not a single seed. If single-seed, queue a paired re-run
> immediately — this is the headline figure and needs error bars.
> Estimated cost: ~6 hr for the 2 cells × 5 seeds.

> **[CW]** Check whether this 0.500 result was on the audit-clean
> shuffled labels or on real-but-strict-protocol labels — the paper
> claims one mechanism (sign-leakage); the 0.500 number is consistent
> with both. Need to distinguish.

---

## Already-validated strict-protocol numbers (Gömb-strict)

The bottom row of Table 1 — A_strict for Gömb-strict — is mostly
single-seed at this point. The 5 headline numbers reported in
manuscript are:

| Dataset | Gömb-strict A_strict (current) | Seeds | Source memory |
|---|---|---|---|
| Bitcoin-Alpha | **0.8972** | 1? | (manuscript-only; needs cross-ref) |
| Bitcoin-OTC | **0.9145** | 1? | (manuscript-only) |
| Slashdot | **0.9017** | 1? | (manuscript-only) |
| Epinions | **0.9425** | 1? | (manuscript-only; +19.77pp over SE-SGformer reported) |
| Reddit Hyperlinks | **0.7612** | 1? | (manuscript-only) |

> **[CW]** Confirm seed counts. If single-seed, the paired n=3 re-run
> is the highest-priority Phase B compute item before submission.
> Estimated cost: 5 datasets × 3 seeds × ~1 hr = **~15 GPU-hr** —
> single overnight on the 2070 SUPER.

**Bitcoin-Alpha specific**: prior memory `project_hsikan_paper_state_2026_05_03`
records HSiKAN at "1-2 vs our-tuned SGCN" on Bitcoin (i.e., SGCN
ours-tuned beats HSiKAN on 2 of 3 metrics). That's the SGCN
A_strict reference for the cell — already on disk.

---

## Baseline-method coverage (per Phase B requirement)

From `project_hsikan_paper_state_2026_05_03` and
`project_phase8_bitcoin_sigat_2026_05_02`:

| Baseline | Status under LEAKY (in hand) | Status under STRICT (Phase B needed) |
|---|---|---|
| SGCN          | ✓ 5-seed Bitcoin; HSiKAN 3-0 vs published, 1-2 vs our-tuned | ✗ Phase B |
| SiGAT-attn    | ✓ 5-seed Bitcoin (2nd place) | ✗ Phase B |
| SGCL          | ? (need to check repo for prior runs) | ✗ Phase B |
| SiGformer     | ✗ Phase B (no prior run found in MEMORY.md) | ✗ Phase B |
| SE-SGformer   | ✗ Phase B (Epinions +19.77 pp delta noted in manuscript — confirm) | ✗ Phase B |
| DADSGNN       | ✗ Phase B | ✗ Phase B |

**Phase B compute budget revised down**: instead of 7 methods × 5
datasets × full audit (~245 GPU-hr), the actual gap is roughly:
- **Leaky-protocol numbers needed**: ~10–15 cells (mostly Slashdot,
  Epinions, Reddit Hyperlinks for SGCL/SiGformer/SE-SGformer/DADSGNN)
- **Strict-protocol numbers needed**: ~30 cells (every baseline ×
  every dataset, since no baseline has been audited under strict)
- **Shuffle-audit numbers needed**: ~60 cells (every cell needs
  S_leak and S_strict)

Revised total: ~**150-180 GPU-hours** instead of 245 — about
**6-7 days on the 2070 SUPER** or **3 days on 2 GPUs**.

---

## Datasets already prepared

From prior compute work, the following datasets are loadable and
have prior train/test splits on disk:

| Dataset | Loader exists | Strict-protocol split frozen? | SHA-256 anchored? |
|---|---|---|---|
| Bitcoin-Alpha | ✓ | [CW: where?] | ✗ — needs to be generated |
| Bitcoin-OTC | ✓ | [CW] | ✗ |
| Slashdot | ✓ | [CW] | ✗ |
| Epinions | ✓ | [CW] | ✗ |
| Reddit Hyperlinks | [CW] | ✗ | ✗ |

> **[CW]** Reddit Hyperlinks is the conspicuous gap in every column.
> First Phase B compute should be the Reddit dataset prep + a single
> Gömb-strict run to confirm 0.7612 reproduces from the loader-on-disk
> path.

---

## Other relevant numbers worth keeping in mind

These don't directly fill Table 1 cells but inform the paper's claims:

- **HSiKAN cycle-vs-walk ablation**: `project_phase9_k45_sweet_spot_2026_05_02`
  shows k=3 cycles add no value on top of k=4+k=5; "k=4+k=5 mixed
  beats k=3+k=4 on every dataset". This informs the
  architecture-design section's "minimum-cycle-arity" claim.

- **HSiKAN inference speedup**: `project_hsikan_inference_speedup_2026_05_03`
  documents 24.5→6.0 ms (Bitcoin), 116→28 ms (Slashdot) inference
  improvements. Relevant for the "compute story" — the 24 GPU-hr
  budget assumes the validated speedup is in place.

- **Triton kernel fast-path**: `project_triton_kernel_integration_2026_05_09`
  documents the per-edge Catmull-Rom kernel. Relevant for the
  reproducibility container — the audit reproducer needs to handle
  both kernel-ON and kernel-OFF modes.

- **Fused backward 92.5% memory savings**: `project_fused_backward_kernel_2026_05_09`
  records the BA-training memory fix that made the joint-mix smoke
  possible. Container must ship this kernel for repro.

- **Axiom > attention finding**: `project_axiom_beats_attention_2026_05_05`
  shows balance pruner adds +0.043 over attention at iso-params on
  Bitcoin-Alpha. Worth a Methods paragraph on the cycle-pruner choice.

---

## What still needs to be measured

In priority order:

1. **Strict-protocol re-runs of Gömb-strict at n=3 seeds** (5 datasets)
   → ~15 GPU-hr → 1 overnight on 2070 SUPER
2. **Reddit Hyperlinks for all 6 baselines** (leaky + strict)
   → ~30 GPU-hr → 2-3 nights
3. **SiGformer / SE-SGformer / DADSGNN on the 4 SNAP datasets**,
   both protocols, n=3 seeds → ~90 GPU-hr → ~4-5 nights
4. **Shuffle-audit numbers** S_leak and S_strict per cell
   → ~30 GPU-hr (cheap; same trained models, just shuffled-label inference)

**Total realistic Phase B compute**: ~**165 GPU-hours**, ~7 nights
on a single 2070 SUPER. Doable in 2 weeks if running unattended
overnight. **3 days if a second GPU is borrowed.**

---

## Open coordination questions

> **[CW]** Where are the BA/OTC/Slashdot/Epinions strict-protocol
> split files currently? `signedkan_wip/data/`? `scripts/split/`?
> Need a single location for the package.

> **[CW]** Is the joint-mix HSiKAN 0.500-under-strict result from
> a properly-frozen strict-protocol split, or from in-script masking?
> If in-script, we need to regenerate from frozen splits and confirm
> the 0.500 reproduces.

> **[CW]** Confirm whether prior work used n=3 or n=5 seeds. Mixing
> them in Table 1 is OK if disclosed but cleanest is unified at n=5.

> **[CW]** Is the SE-SGformer "+19.77 pp Gömb-strict wins Epinions"
> result from the SE-SGformer original code or our reproduction? If
> reproduction, that's a SE-SGformer A_strict cell already on disk —
> add it to the inventory.
