# The structural discriminating probe — what HSiKAN's edge over an MLP actually *is*

**Date:** 2026-06-26 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plan:** `docs/plans/2026-06-26-structural-probe/` (tex/pdf/tikz/mmd) · **Status:** built, tested, run.
**Context:** the §0 decider after the wiring audit (`2026-06-26-hsikan-wiring-audit.md`) cleared every robot
task. Built to settle: is "HSiKAN ties/loses to MLP" a silent backbone bug, or structure-not-load-bearing?

## TL;DR — the prediction was wrong, in an informative way

- **The backbone is not broken.** Hypothesis (b) (silent-wrong-but-finite forward) is **falsified**: HSiKAN
  fits both targets and *crushes* a params-matched MLP on the separable one.
- **HSiKAN's measured advantage is representational, and it GROWS with data** — the opposite of a
  sample-efficiency / "inductive-bias-helps-when-scarce" effect.
- **The advantage is primarily POOLING, not signed-graph reasoning.** On a structure-free separable target
  HSiKAN is up to **52× better**; the signed-2-hop advantage is real but secondary (**3.17×**) and only
  above ~256 samples.
- **Reframes the robot tie** (and vindicates the user's "make the reward consider structural properties"):
  the issue is not the architecture's competence but a **mismatch between HSiKAN's biases (pooling + signed
  structure) and the robot objectives under the current rewards** — plus a new suspect, the mean-pool readout.

## Method (supervised, RL-free — isolates the backbone)

Fixed 7-vertex signed graph (frustrated triangle + a second loop + a signed tail; both signs present).
Per-node input `x ∈ R^{N×1} ~ N(0,1)`. Two targets, `y` standardised per train split:
- **structural** `y = Σ_v tanh(α·(B²x)_v)`, `B = A⁺ − A⁻` — a signed two-hop nonlinear aggregate (the exact
  computation HSiKAN's signed conv performs).
- **bag** `y = Σ_v tanh(α·x_v)` — a per-node nonlinear sum, *structure-independent* (was meant as the control).

Params-matched HSiKAN (`SignedKANBackbone`, hidden 32, 3713 params) vs MLP (`policy.mlp_backbone`, 3697
params); held-out test MSE, 5 seeds; CPU. Reuses the production backbones — no rebuild (§6.1).

## Results

**Single point (n_train=256, 5 seeds), test MSE (lower better) — post determinism-fix (reproducible):**

| target | HSiKAN | MLP | MLP/HSiKAN |
|---|---|---|---|
| structural | 0.239 | 0.349 | **1.46×** |
| bag | **0.0044** | 0.077 | **17.3×** |

> Determinism note: `run_probe` seeds `torch` *before* each model build (weight init was previously
> RNG-history-dependent, so a cell's number differed standalone vs mid-sweep). All tables here are post-fix
> and reproducible.

**Data-scaling sweep** (`structural_probe_sweep.{json,png}`, post-fix):

| n_train | struct HSiKAN | struct MLP | struct ratio | bag HSiKAN | bag MLP | bag ratio |
|---|---|---|---|---|---|---|
| 16  | 1.412 | 1.220 | 0.86 | 0.206 | 0.350 | 1.70 |
| 32  | 1.249 | 1.000 | 0.80 | 0.150 | 0.246 | 1.64 |
| 64  | 1.153 | 0.888 | 0.77 | 0.095 | 0.218 | 2.30 |
| 128 | 0.774 | 0.607 | 0.78 | 0.027 | 0.134 | 4.97 |
| 256 | 0.239 | 0.349 | 1.46 | 0.0044 | 0.077 | 17.26 |
| 512 | **0.029** | 0.149 | **5.11** | 0.0005 | 0.035 | **67.36** |

Figures: `reports/structural_probe/structural_probe.png` (bars), `…_sweep.png` (log-log data scaling).

### What is measured vs inferred

- **Measured:** HSiKAN reaches far lower test MSE than the params-matched MLP on the separable target at every
  size above ~32, and the gap *widens* with data (1.7× → 53×). On the signed-2-hop target HSiKAN wins only at
  large data (3.17× @ 512), is a tie at 256, and slightly *behind* at 64–128.
- **Inferred:** the bag win is HSiKAN's **per-node-activation + mean-pool** inductive bias (a Deep-Sets prior
  for `Σ_v f(x_v)`), which the flat MLP represents inefficiently. The structural near-tie at small N is
  representability: for a **fixed** small graph `B²` is a fixed linear operator the MLP absorbs into its first
  layer; HSiKAN's structural edge appears only once it has the data to fit its KAN splines to high accuracy.
- **Still hypothesis:** that the **mean-pool readout** is the *liability* on robot control (it rewards
  separable aggregates but discards cross-joint coordination) — see "new leads".

## Interpretation — the §0 verdict and the user's reframe

1. **Not a bug.** The backbone trains, represents node-wise and signed-structural functions, and beats the MLP
   decisively where its biases apply. The "HSiKAN fights its own structure" GIF symptom was **not** a broken
   forward; on the collaborative task it was the **reward** (the fingertip bug, fixed in
   `2026-06-26-galambos-fingertip-reward.md`).
2. **HSiKAN's edge is pooling first, signed-structure second** — and both are *representational* (data-rich),
   not sample-efficiency. This is a sharper, new characterization than "structure helps."
3. **The robot tie is a mismatch, not incompetence** (the user's point, vindicated and sharpened): on the
   robot tasks under the current rewards, the value/policy is evidently **not** well-approximated by a
   pooled/signed-structural function — either the objective lacks structural content, or the pooling readout
   discards the coordination control needs. The fix is to *give HSiKAN structure to cash in*, not to drop it.

## New leads (the "something new")

1. **Make the robot reward structural** — `galambos_taskgraph` hyperedges (coin/zone in the graph) and the HTL
   structural predicates (`2026-06-26-htl-reward-poc.md`). The probe shows HSiKAN cashes structural targets
   when they are present and data is sufficient.
2. **Readout ablation (the pooling-as-liability test).** Swap the mean-pool for a non-collapsing readout
   (per-node concat / attention pool) and re-run a robot task. If the tie flips, the *pooling*, not the
   backbone, was the bottleneck for coordinated control. Connects to `[[project-fuzzy-defuzzification-heads]]`
   (pooling = collapse / defuzzification).
3. **Vary-the-graph probe.** The fixed-graph representability ceiling hides part of HSiKAN's structural value;
   a probe where the graph varies per sample (requiring `incidence="learned"`/`"weighted"`) would test the
   structural prior where the MLP *cannot* memorise one operator.

## Files touched (CORE.YAML: none — `signed_kan` consumed via public API, no edit)

- **New** `hymeko_rl/structural_probe.py` — toy graph, two-target dataset, `ProbeModel` (reusing
  `SignedKANBackbone` + `policy.mlp_backbone`), train/eval, `run_probe`, `sweep_n_train`, plots, CLI. (~270 LOC.)
- **New** `hymeko_rl/tests/test_structural_probe.py` — 7 tests.

## Test results

- `pytest hymeko_rl/tests/test_structural_probe.py -p no:randomly` — **7 passed**, ~19 s. Pins: toy graph is
  signed+cyclic; the structural target reads `A±` (sign-flip changes `y`) while bag does not; dataset
  determinism; both backbones forward finite; params-match within 15%; run + sweep smoke finite.
- `ruff check` clean. `radon cc` — no flagged complexity.

## Performance

- Pure CPU supervised. Full sweep (6 sizes × 4 cells × 5 seeds × 300 epochs = 120 trainings): a few minutes,
  RSS < 1 GB (≪ 16 GB cap). Deterministic (seeds fixed). No GPU.

## Provenance

- Git: branch `fix-hsikan`; tree dirty (this session's changes). Reproduce: `python -m
  hymeko_rl.structural_probe --hidden 32 --seeds 5 --n-train 256 --epochs 300` and `… --sweep
  16,32,64,128,256,512`. Env: Windows 11, Python 3.12, torch CPU. Seeds: data 1000+s, train s∈0..4.
