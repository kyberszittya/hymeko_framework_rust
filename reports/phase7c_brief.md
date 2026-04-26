# Phase 7c — Universality Breadth Brief

**Generated:** 2026-04-25
**Auto-fires after:** `Views PH7b suite finished` marker in
`/tmp/thesis_iv_views_ph7b.log`
**Estimated wall-clock:** ~17 h
**Output CSVs land in:** `data/benchmarks/thesis_iv_hard_2026042?_*.csv`

## 1. Purpose

Phases 6 and 7 demonstrated that two new spectral-entropy variants —
`scalar_entropy_normalized` (Path B) and `entropy_target H*=0.5` (Path A) —
universally Pareto-dominate the original `scalar_entropy` regulariser
across **four stress datasets** (spirals, circles, MNIST plain MLP,
CapsMLP MNIST). Phase 7c extends this stress test to **six additional
dataset/architecture combinations** that were tested only with the
original `scalar_entropy` in phases 1–5. Goal: settle whether the
universality claim holds on:

- **Deep skip-connection architectures** — ResMLP-20 (residual MLP),
  HighwayMLP-20 (gated)
- **MNIST-shape image siblings** — FashionMNIST, KMNIST
- **Different output-class structure** — EMNIST Letters (26 classes)
- **Different input shape** — SVHN (32×32 colour)

If both new arms preserve sign and magnitude (or amplify) on these six
combinations, the universality claim covers **5 architectures × 9
datasets** with no significant negative outliers.

## 2. Test arms

Two arms — both target `--view dataflow` and `--target-entropy 0.5`.
Original `scalar_entropy` arm runs alongside as the paired baseline.

| Arm | Formula | Hyperparameters |
|---|---|---|
| `scalar_entropy_normalized` | `λ · H(A) / log₂(rank(A))` | none beyond λ |
| `entropy_target` | `λ · (H_norm − H*)²` | `H* = 0.5` |

Code references (`python/benches/thesis_iv_hard/run_benchmark.py`):

- `scalar_entropy_normalized` branch: see lines 1380–1397
- `entropy_target` branch: see lines 1398–1416

---

## 3. Experiment cards

### 3.1 ResMLP-20 (deep residual MLP)

**Dataset / arch:** MNIST handwritten digits (60 k train / 10 k test);
20 residual blocks of width 16 (`mnist_resnet_20`).

**Config:** 33 seeds × 5 epochs, λ = 0.1, reg-every-n = 10, dataflow.

**Original (phase 1, scalar_entropy):**
- Δ = +0.046 pp, t = +1.76 (marginal)
- Δ = +0.070 pp, t = +1.89 (marginal, replication)

**Expected outcome:** Both new arms should keep the marginal positive
but tighten the t-stat. If normalised entropy is the right scale,
deeper architectures should benefit *more* from normalisation than
plain MLPs because their effective rank grows with depth.

**Commands:**

```bash
# scalar_entropy_normalized
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets mnist_resnet_20 \
    --arms baseline scalar_entropy_normalized \
    --seeds 33 --epochs 5 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5

# entropy_target H*=0.5
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets mnist_resnet_20 \
    --arms baseline entropy_target \
    --seeds 33 --epochs 5 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5
```

---

### 3.2 FashionMNIST (image sibling)

**Dataset / arch:** Fashion items (28×28 grayscale, 10 classes); plain
MLP 784→16→8→10 (`fashion_mnist`).

**Config:** 33 seeds × 15 epochs, λ = 0.1, reg-every-n = 10, dataflow.

**Original (phase 1, scalar_entropy):** Δ = +0.039 pp, t = +1.13 (null).

**Expected outcome:** This is the sibling test. If MNIST's anchor result
was MNIST-specific the original null persists; if normalisation is the
real fix, both new arms turn FashionMNIST into a positive of similar
magnitude (≈ +0.10 pp expected).

**Commands:**

```bash
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets fashion_mnist \
    --arms baseline scalar_entropy_normalized \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5

python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets fashion_mnist \
    --arms baseline entropy_target \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5
```

---

### 3.3 KMNIST (Japanese hiragana)

**Dataset / arch:** 70 k handwritten hiragana (28×28 grayscale, 10
classes); plain MLP (`kmnist`).

Backed by the HuggingFace `tanganke/kmnist` parquet mirror, decoded
once and cached in `datasets/kmnist/kmnist_decoded.npz` (the official
`codh.rois.ac.jp` torchvision URL is dead).

**Config:** 33 seeds × 15 epochs, λ = 0.1, reg-every-n = 10, dataflow.

**Original (phase 3, scalar_entropy):** Δ = −0.029 pp, t = −0.33 (null).

**Expected outcome:** Different script family from MNIST → if the new
arms produce a positive, the universality claim covers cross-script
generalisation.

**Commands:**

```bash
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets kmnist \
    --arms baseline scalar_entropy_normalized \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5

python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets kmnist \
    --arms baseline entropy_target \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5
```

---

### 3.4 SVHN (Street View House Numbers, 32×32 colour)

**Dataset / arch:** 73 k digit crops (32×32×3 colour, 10 classes);
SVHNNet 3072→64→32→10 with the 3072→64 input projection skipped from
the regulariser (`svhn`).

**Config:** 15 seeds × 20 epochs, λ = 0.1, reg-every-n = 10, dataflow.

**Original (phase 5, scalar_entropy):** Δ = −0.137 pp, t = −0.55 (null,
directionally negative).

**Expected outcome:** The hardest test in this batch — colour input +
small seed count. If both new arms move SVHN toward zero or positive,
the regulariser's *failure-mode mitigation* claim is supported even on
3-channel inputs.

**Commands:**

```bash
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets svhn \
    --arms baseline scalar_entropy_normalized \
    --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5

python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets svhn \
    --arms baseline entropy_target \
    --seeds 15 --epochs 20 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5
```

---

### 3.5 EMNIST Letters (26-class siblings of MNIST)

**Dataset / arch:** 145 k handwritten letters (28×28, 26 classes,
labels shifted to 0..25 from torchvision's 1..26 convention);
MNISTNetSmallNClass(26) (`emnist_letters`).

**Config:** 33 seeds × 15 epochs, λ = 0.1, reg-every-n = 10, dataflow.

**Original (phase 5, scalar_entropy):**
- Δ = +0.170 pp, t = +1.24 (directional +, not significant)
- σ drops from 0.941 → 0.684 (27 % variance reduction — strongest
  variance-reduction signature of any image dataset)

**Expected outcome:** This is the most likely big-win conversion. The
underlying signal (massive variance drop) is already strong with
unnormalised entropy; both new arms should bring the t-stat into
significance (≥ p < 0.05) given normalisation handles the larger
26-class output dimension better.

**Commands:**

```bash
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets emnist_letters \
    --arms baseline scalar_entropy_normalized \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5

python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets emnist_letters \
    --arms baseline entropy_target \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5
```

---

### 3.6 MNIST HighwayMLP-20 (deep gated MLP)

**Dataset / arch:** MNIST; 20-block Highway MLP (gated skip
connections at width 16) (`mnist_highway_20`).

**Config:** 33 seeds × 15 epochs, λ = 0.1, reg-every-n = 10, dataflow.

**Original (phase 4, scalar_entropy):**
result not yet aggregated to a single line; baseline was significantly
positive in phase 4 but the full table is in `RESULTS_VIEWS_SUITE.md`.
Expectation is in the same neighbourhood as ResMLP-20.

**Expected outcome:** Highway gating differs from ResMLP's additive
skip; the question is whether normalisation + target entropy still
land positive across both deep-MLP architectures, confirming the claim
is *architecture-independent* among skip-connection styles.

**Commands:**

```bash
python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets mnist_highway_20 \
    --arms baseline scalar_entropy_normalized \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5

python3 python/benches/thesis_iv_hard/run_benchmark.py \
    --datasets mnist_highway_20 \
    --arms baseline entropy_target \
    --seeds 33 --epochs 15 --lam 0.1 --reg-every-n 10 \
    --view dataflow --target-entropy 0.5
```

---

## 4. Aggregated expected outcomes

If the universality claim holds across phase 7c:

| Dataset | original Δ | expected B Δ | expected A Δ |
|---|---|---|---|
| ResMLP-20 (MNIST) | +0.046 marginal | +0.05 to +0.10 sig | +0.05 to +0.10 sig |
| FashionMNIST | +0.039 null | +0.08 to +0.12 sig | +0.08 to +0.12 sig |
| KMNIST | −0.029 null | −0.02 to +0.03 null | 0 to +0.05 null |
| SVHN | −0.137 null | −0.05 to +0.05 null | −0.05 to +0.05 null |
| EMNIST Letters | +0.170 directional | +0.20 to +0.25 sig | +0.20 to +0.25 sig |
| HighwayMLP-20 | (re-test) | +0.04 to +0.10 marginal/sig | +0.04 to +0.10 marginal/sig |

If observed Δ falls outside these ranges by more than 50 %, the claim
needs qualification per dataset.

## 5. After phase 7c

Re-run `python3 python/benches/thesis_iv_hard/aggregate_views_suite.py`
to refresh `RESULTS_VIEWS_SUITE.md` with the 12 new rows. Then
regenerate the LaTeX report (`reports/thesis_iv_views_suite.tex`) for a
final 132-experiment summary.
