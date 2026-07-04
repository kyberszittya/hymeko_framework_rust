# Entropy-Pool Hypergraph Learning vs Backprop-Like Nagare

Created-at: 2026-07-02 02:32 JST

## Summary

Implemented a first discriminating experiment for the thesis:

> Hypergraph-structured neural learning can use entropy global pooling and local
> structural updates instead of full reverse-mode credit assignment.

The new comparison harness evaluates:

- `entropy_pool_local`: fixed quaternion holonomy vertex features, global
  hypergraph pooling, entropy-gated local readout update only.
- `backprop_like_holonomy`: the existing Nagare holonomy entropy toy with
  closed-form backward updates through embed, pool, fused entropy update, and
  readout.

The local learner does **not** backpropagate into the feature generator,
pooling, or hypergraph state. Its update is:

```text
W <- W + lr * gate(entropy(p)) * phi * (y - p)
```

where `phi` is the globally pooled hypergraph/holonomy feature vector.

## Files

- `docs/plans/2026-07-02-entropy-pool-vs-backprop-hypergraph/plan.{tex,pdf,tikz,mmd}`
- `hymeko_nagare/examples/entropy_pool_learning_compare.rs`
- `hymeko_nagare/tests/entropy_pool_learning_compare.rs`
- `reports/2026-07-02-entropy-pool-vs-backprop-hypergraph.json`
- `reports/2026-07-02-entropy-pool-vs-backprop-hypergraph-rss.json`

## Results

Shape: `n_train=192`, `n_test=96`, `points=32`, `hidden=24`, `epochs=50`.

| task | local acc | backprop acc | local median | backprop median | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| moons | 1.000 | 1.000 | 5.79 us | 143.31 us | 24.73x |
| spiral | 1.000 | 1.000 | 5.83 us | 138.43 us | 23.73x |
| xor | 1.000 | 1.000 | 6.02 us | 136.83 us | 22.71x |

Diagnostics:

| task | local CE | local entropy | local Clifford error |
| --- | ---: | ---: | ---: |
| moons | 0.001513 | 0.015819 | 0.000007 |
| spiral | 0.004486 | 0.041033 | 0.000044 |
| xor | 0.006714 | 0.056046 | 0.000137 |

Model size:

| learner | params |
| --- | ---: |
| entropy-pool local | 60 |
| backprop-like holonomy | 2,836 |

Sampled comparison executable peak RSS: `7.59 MiB`.

## Interpretation

This is a positive substrate result. On easy generated hypergraph-like point-set
tasks, the entropy-pool local learner matches the backprop-like model's accuracy
with about `47x` fewer parameters and `22x-25x` lower forward latency.

This does **not** prove the biological claim. It supports a narrower engineering
claim:

```text
When quaternion/holonomy structural features are already informative,
global entropy pooling plus a local output update can replace full hidden-state
backpropagation on simple tasks.
```

The important distinction is that the local learner still uses supervised output
error `(y - p)`. What it avoids is exact reverse-mode propagation through the
hypergraph feature generator and pooled structural state.

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
rustfmt --check hymeko_nagare\examples\entropy_pool_learning_compare.rs hymeko_nagare\tests\entropy_pool_learning_compare.rs
cargo test -p hymeko_nagare --test entropy_pool_learning_compare -- --nocapture
cargo clippy -p hymeko_nagare --example entropy_pool_learning_compare --test entropy_pool_learning_compare --no-deps -- -D warnings
cargo run -p hymeko_nagare --release --example entropy_pool_learning_compare -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --hidden 24 --epochs 50 --batch-size 32 --lr 0.05 --seed 53 --out reports\2026-07-02-entropy-pool-vs-backprop-hypergraph.json
```

All checks passed.

## Next Discriminating Test

The next test should make the structural features less trivially sufficient:

- noisy or missing hyperedges,
- shuffled point order versus ordered holonomy,
- fewer training samples,
- harder multi-class synthetic tasks,
- compare local entropy-gated update against the same local rule without entropy
  gating.

That isolates whether entropy feedback is doing learning work, not merely riding
on strong fixed holonomy features.
