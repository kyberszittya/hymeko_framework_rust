# Komondor HPC integration + SGCN label-shuffle audit

Date: 2026-06-02
Plans: [`docs/komondor_setup/`](../docs/komondor_setup/),
       [`docs/plans/2026-05-17-nature-comm-leakage-audit/`](../docs/plans/2026-05-17-nature-comm-leakage-audit/)
Authors: HyMeKo / SignedKAN, run on Komondor HPC (KIFÜ HUN-REN) account `pr_szhc` / project `pr_szevis`.

## Summary

Two milestones reached:

1. **Komondor HPC pipeline operational.** Singularity image
   `hymeko_signedkan.sif` (5.0 GB, torch 2.6.0+cu124) built on the
   login node; probe SGCN job ran in 19 s wall on an A100-SXM4-40GB
   (job `13883232`), reproducing local `sgcn_baseline.json` AUC within
   ±0.005. Pipeline validated end-to-end.

2. **SGCN row of NComms Table 2 (label-shuffle audit) populated.**
   24-job array `13883233` (4 datasets × 2 modes × 3 seeds) ran in
   ~5 minutes wall. Every SGCN cell drops to chance under shuffled
   labels, matching the paper's "*chance under shuffle is what a
   protocol-clean signed-link prediction method should look like*"
   claim ([main.tex §4 line 295](../paper/nature_comm_v1/main.tex#L295)).

## Pipeline

### Singularity image build

Built via [`docs/komondor_setup/build_image.sh`](../docs/komondor_setup/build_image.sh)
from [`hymeko_signedkan.def`](../docs/komondor_setup/hymeko_signedkan.def):

| Layer | Choice | Why |
|---|---|---|
| Base | `nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04` | Python 3.12 in apt; CUDA 12.6 runtime libs |
| Python | venv at `/opt/venv` | Sidesteps debian-managed pip + PEP-668 |
| torch | `2.6.0+cu124` | Highest on cu124 wheel index; cluster driver compatible; local is 2.11.0+cu130 (see "Reproducibility note") |
| numpy / scipy / sklearn | 2.4.4 / 1.17.1 / 1.8.0 (tight pins) | Numerically reproducible — affects RNG ordering + AUC computations |
| networkx / pandas / pyyaml / tqdm | loose pins | UI/utility; no numerical impact |
| matplotlib | not installed | Plotting happens on workstation, not HPC; saves ~250 MB |

Build went through 8 rounds (`set -o pipefail` in dash, Python version mismatch,
debian pip RECORD, torch wheel index, matplotlib wheel, venv symlink, fakeroot
write target) — all resolved cleanly. Final image: 5.0 GB.

### Probe job (validation of pipeline)

Job ID `13883232`, partition `gpu`, A100-SXM4-40GB, 19 s wall, AUC 0.8756.
Sanity gates passed:

| Gate | Result |
|---|---|
| JSON line produced | ✓ |
| AUC ∈ [0.85, 0.92] | ✓ (0.8756) |
| Wall < 60 s | ✓ (19 s) |
| `n_params` matches local | ✓ (135,585 ↔ 135,585) |
| `torch.cuda.is_available()` inside container | ✓ (`--nv` flag works) |

### Reproducibility note

Container torch 2.6.0+cu124, local torch 2.11.0+cu130. Cross-checked
on `bitcoin_alpha` seed=0:

| | Local (RTX 2070 SUPER, torch 2.11) | Komondor (A100, torch 2.6) | Δ |
|---|---|---|---|
| AUC | 0.8704 | 0.8756 | +0.005 |
| n_params | 135,585 | 135,585 | exact |

Δ is well within seed-to-seed RNG noise (local seeds spanned ±0.018).
SGCN has no torch-version-sensitive code path between 2.6 and 2.11.
For the NComms reproducibility hook, the container's image hash
(`singularity inspect --labels hymeko_signedkan.sif` → `org.hymeko.*`)
is the canonical citation pin.

## SGCN label-shuffle audit (job array 13883233)

### Setup

| Axis | Values |
|---|---|
| Datasets | `bitcoin_alpha`, `bitcoin_otc`, `slashdot`, `epinions` |
| Modes | real (no flag), shuffled (`--shuffle-train-signs`) |
| Seeds | 0, 1, 2 |
| Total | 24 cells |
| Submit | [`submit_sgcn_audit_array.sh`](../docs/komondor_setup/submit_sgcn_audit_array.sh) |
| Wall | ~5 min total (parallel across A100s) |

SGCN does not use cycle features, so the strict/transductive
distinction collapses — adjacency is always built from training
edges only.  The two modes (real vs shuffled) are sufficient to
test for label-leakage (`A_leak` ↔ `S_leak`).

### Results (mean ± pstdev, n=3 seeds)

| Dataset | A_leak (real) | S_leak (shuffled) | Δ drop | Audit clean? |
|---|---|---|---|---|
| `bitcoin_alpha` | **0.8685 ± 0.0083** | **0.5284 ± 0.0023** | −0.3401 | ✓ |
| `bitcoin_otc` | **0.9037 ± 0.0101** | **0.5169 ± 0.0110** | −0.3868 | ✓ |
| `slashdot` | **0.8792 ± 0.0017** | **0.5095 ± 0.0026** | −0.3697 | ✓ |
| `epinions` | **0.9314 ± 0.0025** | **0.4874 ± 0.0099** | −0.4440 | ✓ |

Every cell drops to chance (AUC ≈ 0.5) under label shuffle.
Cross-check against [`sgcn_baseline.json`](../signedkan_wip/experiments/results/sgcn_baseline.json):
local BA real AUC (3-seed median) was 0.8704; Komondor 3-seed mean is
0.8685. Local OTC median 0.9044; Komondor mean 0.9037. Container ↔
local agreement is within noise for both.

### NComms Table 2 row (drop-in replacement)

Current draft ([main.tex line 272](../paper/nature_comm_v1/main.tex#L272)):

```latex
SGCN (2018)$^{\dagger}$  & 0.929 / 0.550 & TBD / TBD & TBD / TBD & TBD / TBD \\
```

Becomes:

```latex
SGCN (2018)$^{\dagger}$  & 0.869 / 0.528 & 0.904 / 0.517 & TBD / TBD & 0.931 / 0.487 \\
```

(Column order in Table 2: Bitcoin-Alpha, Bitcoin-OTC, Reddit Hyperlinks, Epinions.
Reddit cell remains TBD pending a small loader wire-up; Slashdot result feeds Table 3.)

The headline number 0.929 in the draft was a transcribed reference;
our re-run gives 0.869. The discrepancy is **publishing-relevant** —
the paper should cite our 0.869 ± 0.008 from this audit, not the
unsourced 0.929, since the 0.869 is reproducible end-to-end on a
documented pipeline.

## Wall-time accounting

| Step | Wall |
|---|---|
| Singularity image build (round 8, login node) | ~7 min |
| Probe job (sgcn smoke, 1 cell) | 19 s |
| Array sweep (24 cells, parallel) | ~5 min |
| Reduce step (per-cell JSON aggregation) | ~3 s |
| **Total HPC GPU-hours used** | **~0.05 GPU-hr (3 min of A100 time)** |

Comparison: local 24-cell sweep (planned but not run; sgcn-audit-chain.scope
parked behind pose chain) would have been ~12 min wall serial on a single
RTX 2070 SUPER. Parallel A100 speedup on this trivially-small workload
is modest (~2.4×); for larger HSiKAN sweeps the speedup compounds
substantially.

## Open follow-ups

1. **Reddit loader wire-up.** Add `reddit` to
   `run_final_cell.py`'s `--dataset` choices + dispatch to the
   existing Reddit loader (used by `gomb_strict_benchmark` scripts).
   Estimated ~15 min code + 1 array job (~5 min wall) to fill the
   Reddit column of Table 2 SGCN row.
2. **HSiKAN-Optuna audit.** In flight as job `13883269_[0-39]`
   (40 cells: 2 datasets × 4 conditions × 5 seeds) — fills the
   HSiKAN-Optuna row of Table 2 with the 4-condition matrix
   (A_leak / S_leak / A_strict / S_strict). Validates the central
   99.5% leakage claim.
3. **Phase B baseline port.** SE-SGformer / DADSGNN / SiGformer / SGCL
   are not in the repo. Each needs ~1-2 workstation-days of porting
   from upstream, then ~1 hr of parallel A100 wall for the full
   4-condition × 5-seed audit.

## Provenance

| | |
|---|---|
| Git HEAD (workstation) | `8fd8187c7dc3e9c7bda67c01c10364f416127e54` |
| Container | `hymeko_signedkan.sif` (5.0 GB; torch 2.6.0+cu124; Python 3.12; Ubuntu 24.04) |
| Container labels | `singularity inspect --labels hymeko_signedkan.sif` |
| Hardware | NVIDIA A100-SXM4-40GB (Komondor `gpu` partition; node `x1001c6s2b0n0` for probe) |
| Singularity | `singularity-ce version 4.0.1` |
| Workspace | `/scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust/` |
| Probe slurm log | `slurm_logs/sgcn-smoke-13883232.out` |
| Array slurm logs | `slurm_logs/sgcn-audit-13883233_{00..23}.{out,err}` |

## Memory entries to add

- `project_komondor_sgcn_audit_2026_06_02.md` — this report's headline (24-cell SGCN audit done on Komondor; SGCN clean under shuffle on 4 datasets; HPC pipeline operational; torch 2.6/2.11 cross-validated)
