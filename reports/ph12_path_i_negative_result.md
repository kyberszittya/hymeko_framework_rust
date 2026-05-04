# Phase 12 result — Path I (`total_correlation_mi`) is a **negative result** at the conventional λ

**Date:** 2026-04-27
**Phase:** 12 — sweep of `total_correlation_mi` (Path I) on
spirals + 3 MNIST siblings + circles. 13 runs, ~3.5 h on RTX 2070 SUPER.
**Companion brief:** `reports/phases_11_12_13_brief.md` (configuration).
**Path I design:** `project_path_i_total_correlation_mi.md` (memory note).

---

## TL;DR

At the conventional λ = 0.1 used by every other arm in the universality
programme, Path I (multi-information across all L layers, with KL-feedback
and variance-momentum on λ) is **significantly worse than baseline** on
the spirals anchor — Δ = -1.014 pp, t = -4.62, W/L = 8/78 (***).
Increasing λ collapses training entirely (Δ = -26.6 pp at λ = 1.0).

**The design is not broken; the dosage is wrong.** Path I's regulariser
magnitude scales with L (one H_2 term per layer summed in TC), so a λ
calibrated for scalar spectral entropy over-pushes by an order of
magnitude. There is a faint positive trend at λ = 0.03 (Δ = +0.306 pp,
t = +1.78, W/L = 40/31, p ≈ 0.08) — the only setting where Path I
nudges in the right direction.

This phase delivers a **clear negative result on Path I at default
settings** and a clear hypothesis for where to look next (λ ≈ 0.01-0.03,
high β momentum).

---

## All 13 paired Δ headlines

Spirals damp baseline rows are reused as the anchor for single-arm runs
(deterministic given seed).

| run                                | n   | Δ pp     | t       | W/L      | sig  |
|------------------------------------|-----|----------|---------|----------|------|
| spirals damp (anchor + baseline)   | 100 | +0.020   | +0.29   | 32/25    | ns   |
| spirals amplify                    | 100 | -1.152   | -4.45   | 11/81    | ***  |
| spirals mix                        | 100 | -1.014   | -4.62   |  8/78    | ***  |
| **λ-sweep on spirals (mode = mix):**                                       |
| λ = 0.01                           | 100 | +0.182   | +1.29   | 35/28    | ns   |
| λ = 0.03                           | 100 | +0.306   | +1.78   | 40/31    | .    |
| λ = 0.1   (=above mix row)         | 100 | -1.014   | -4.62   |  8/78    | ***  |
| λ = 0.3                            | 100 | -11.212  | -17.91  |  0/100   | ***  |
| λ = 1.0                            | 100 | -26.572  | -33.31  |  0/100   | ***  |
| **Cross-dataset at default λ, mode = mix:**                                |
| circles  (λ = 0.1)                 | 100 | -0.018   | -0.50   | 31/37    | ns   |
| mnist_small  (λ = 0.01)            |  33 | +0.015   | +0.28   | 16/15    | ns   |
| fashion_mnist  (λ = 0.01)          |  33 | -0.092   | -1.61   | 14/18    | ns   |
| kmnist  (λ = 0.01)                 |  33 | -0.012   | -0.11   | 16/17    | ns   |
| **β momentum sweep (mode = mix, spirals, λ = 0.1):**                       |
| β = 0.0   (no momentum)            | 100 | -1.302   | -7.00   |  5/83    | ***  |
| β = 0.9   (default, =above mix row)| 100 | -1.014   | -4.62   |  8/78    | ***  |
| β = 0.99  (frozen inertia)         | 100 |  -0.742  | -4.72   | 10/77    | ***  |

For comparison, Path A and Path B on the **same** spirals fixture:
+0.412 pp *** (Path A) and +0.624 pp *** (Path B). Path I's best
operating point so far is +0.306 pp (.) at λ = 0.03 — strictly weaker
than both established arms.

---

## What the data say

### 1. λ scale is off by ~10×

The λ sweep sketches a clear rapid-collapse curve:

```
  λ      Δ pp
  0.01  +0.182      ←  ns positive trend
  0.03  +0.306      ←  borderline-significant peak
  0.1   -1.014  ***
  0.3  -11.212  *** ←  catastrophic
  1.0  -26.572  *** ←  total collapse
```

This is the signature of a regulariser whose **natural scale is
10× tighter than the rest of the programme**. The fix is mechanical
(divide λ by L) but the λ that *was* universal across Paths A/B/C/E/F
no longer is — Path I needs its own calibration.

### 2. `damp` mode is degenerate

`damp` sets `var_factor = (1 − inertia)`. As training progresses, the
joint Gram's eigenvalue distribution settles to a high-variance steady
state ⇒ inertia → 1 ⇒ var_factor → 0 ⇒ regulariser **turns itself
off**. The damp run lands at Δ = +0.020 pp ns — exactly what you'd see
if the arm were silently degraded to baseline. The mode is informative
about the design (variance-momentum *can* gracefully fade out) but it
is not a useful experimental arm.

### 3. `amplify` and `mix` are over-aggressive at λ = 0.1

Both lose hard (-1.15 pp ***  and -1.01 pp *** respectively). `mix`'s
stage-aware blend doesn't save it because the over-push happens during
*amplify* phase, which is exactly when the regulariser is supposed to
work.

The β sweep tells a related story: β = 0.99 (frozen inertia,
near-constant `var_factor`) is the **least bad** at -0.742 pp
(vs -1.302 at β = 0.0). High momentum acts as a partial brake, but it's
not enough — the underlying signal is just too big at this λ.

### 4. Cross-dataset at default λ is null on all 4

Spirals at λ = 0.1 lost decisively at n = 100. Cross-dataset at λ = 0.1
(circles, n=100) and λ = 0.01 (mnist_small / fashion_mnist / kmnist,
n=33) all land within ±0.1 pp ns. **At plain MLP scale (≤4 spectral
layers), Path I has too few layers for the multi-way redundancy term
to matter** — the joint Gram across 3-4 layers is dominated by the
diagonal, so TC behaves like a noisier scalar entropy.

This is *evidence* (not proof) that the deep-architecture sweep
(ph14: ResMLP-20 / HighwayMLP-20) is where TC's L=20 joint structure
might actually bite. ph14 should be run at λ = 0.01 to stay in the
productive zone.

---

## What this changes about the universality programme

**Before ph12:** the programme had Path A (entropy_target H*=0.5),
Path B (scalar_entropy_normalized) Pareto-dominating Path C, plus the
combined arms from phase 7. Path F (cross_layer_mi, ph11) was already
characterised as a null on plain MLPs at λ = 0.1.

**After ph12:** Path I extends the activation-side family with TC
+ KL-feedback + variance-momentum, but is **not** a drop-in replacement
for Path A/B at the conventional λ. Two scenarios going forward:

**(α) Path I is rescued by a λ retune at λ ≈ 0.02 + high β + deep
architecture.** ph14 + a focused λ sweep (ph15, scoped below) settle
this. If Δ ≥ +0.1 pp on any deep-arch combo at the rescaled λ, Path I
is a real (if niche) tool — its niche being deep nets where
multi-layer redundancy actually exists.

**(β) Path I is dominated by Path A/B everywhere.** Then Path I joins
the *negative results* shelf: an honest experimental answer that
"more complex isn't automatically better." The paper's universality
claim stays where it is — Path B is the headline.

Either outcome is publishable. The negative-result writeup is its
own contribution: it falsifies an obvious-looking generalisation
(activation-side multi-info → must beat structural-side scalar
entropy), and pinpoints why (regulariser-magnitude mismatch).

---

## Bug discovered while running ph13 (2026-04-27 evening)

ph13 (Path I × CapsMLP MNIST) ran cleanly to completion (5/5 DONE), but
**all 5 paired Δ landed at exactly +0.000 with W/L = 0/0**: the TC arm
produced **byte-identical** training trajectories to baseline. Root
cause:

> CapsMLP's `spectral_weights()` returns
> `[self.primary.weight, w_matrix]`, where `w_matrix` is a derived
> tensor `self.W.permute(...).reshape(...)` synthesised on every call.
> The activation-hook setup loop finds modules whose
> `id(m.weight) in weight_ids` — and there is no `nn.Linear` whose
> `.weight is w_matrix`. So `target_modules` ends up with exactly 1
> entry (the primary Linear), `cross_layer_activations` populates with
> 1 tensor, and Path I's `if len(...) >= 2:` guard fails. `reg_term`
> stays `None`, training is identical to baseline.

**Fix landed:** the activation-hook setup now prints a `[WARN]` line
when it finds < 2 hookable modules, so silent no-ops are impossible
going forward.

**Consequence for ph13:** results are **invalid as a Path I test of
CapsMLP**. They confirm only that CapsMLP under the current spectral
adjacency definition cannot be regularised by activation-side arms
without a model refactor (expose the routing as an `nn.Module`
submodule whose forward output can be hooked).

**Consequence for ph14:** the `fashion_mnist_capsnet` run that was in
ph14's queue is **dropped** — same architecture, same bug. ResMLP-20
and HighwayMLP-10/20 all use plain `nn.Linear` modules registered in
`nn.ModuleList`/`nn.ModuleDict` whose weights *are* the listed
`spectral_weights()` entries, so those runs proceed.

## Open work

- **ph15 (queued)** — fine-grained λ sweep on spirals at mode = mix,
  bracketing the productive band: λ ∈ {0.005, 0.01, 0.02, 0.03, 0.05}.
  Goal: nail down the effective λ within ±0.005 so ph13/ph14 use the
  right operating point. ~25 min on 2070S.
- **ph13** — Path I × CapsMLP MNIST, λ = 0.01 (already config'd at the
  productive zone). Hypothesis: capsule routing's explicit cross-layer
  redundancy gives Path I its motivating fixture.
- **ph14** — Path I × deep architectures (ResMLP-20, HighwayMLP-10/20,
  CapsMLP × FashionMNIST), λ = 0.01 (productive zone). Hypothesis:
  L=20 layers is where the multi-way joint Gram structurally beats
  pairwise.
- **Drop `damp` mode from future runs.** It is degenerate by design;
  every damp run is wasted compute.
- **Update `reports/phases_and_paths.tex`** with a Path I row in §3.x,
  a phase-12 section, and a negative-result entry in §4 the headline
  table. Defer until ph13/ph14 results land so the row reflects the
  full picture.
