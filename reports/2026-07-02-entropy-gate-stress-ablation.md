# Entropy-Gate Stress Ablation

Created-at: 2026-07-02 02:43 JST

## Summary

Ran the hard ablation requested after the `~24x` local-learning speed result.
The test compares:

- entropy-gated local update: `gate = 0.25 + H(p)` and entropy feedback feature,
- constant-gated local update: `gate = 1.0` and constant feedback feature.

Both learners use the same fixed quaternion holonomy features and global pooled
hypergraph statistics. The stress conditions are clean, noisy, missing points,
and few-shot noisy+missing.

## Main Result

This is **not** a positive entropy-gate result. Accuracy saturates on most rows,
and where the stress is harder the constant gate is at least as good.

| task | stress | entropy acc | constant acc | entropy loss | constant loss |
| --- | --- | ---: | ---: | ---: | ---: |
| moons | clean | 1.000 | 1.000 | 0.001513 | 0.000465 |
| moons | noisy | 1.000 | 1.000 | 0.001631 | 0.000500 |
| moons | missing | 1.000 | 1.000 | 0.001656 | 0.000516 |
| moons | few-shot noisy+missing | 1.000 | 1.000 | 0.011536 | 0.005352 |
| spiral | clean | 1.000 | 1.000 | 0.004487 | 0.001581 |
| spiral | noisy | 0.990 | 0.990 | 0.043077 | 0.040756 |
| spiral | missing | 1.000 | 1.000 | 0.015772 | 0.011051 |
| spiral | few-shot noisy+missing | 0.927 | 0.938 | 0.286730 | 0.269849 |
| xor | clean | 1.000 | 1.000 | 0.006731 | 0.002713 |
| xor | noisy | 1.000 | 1.000 | 0.006969 | 0.002783 |
| xor | missing | 1.000 | 1.000 | 0.008426 | 0.003485 |
| xor | few-shot noisy+missing | 1.000 | 1.000 | 0.033405 | 0.022253 |

Constant gate has lower loss on all 12 stress rows. Accuracy is tied on 11 rows;
constant is better on the remaining spiral few-shot noisy+missing row.

## Timing

The local learner remains fast. Median local forward times are generally
`3.5-6.0 us/sample`, still far below the backprop-like holonomy reference
(`~135-139 us/sample` in the same run). Peak sampled RSS for the comparison
executable was `7.75 MiB`.

## Interpretation

The earlier result remains useful:

```text
fixed holonomy features + global pooling + local readout update
```

is a very fast and small substrate for these toy tasks.

But this ablation says the current entropy feedback rule is not yet the reason.
In this implementation, entropy mostly slows confidence growth because the gate
shrinks toward `0.25` as the model becomes certain. The constant gate learns
the same structure with lower loss.

Conclusion:

```text
Holonomy/global-pool local learning: positive.
Current entropy-gated update: not validated.
```

## Files

- `docs/plans/2026-07-02-entropy-gate-stress-ablation/plan.{tex,pdf,tikz,mmd}`
- `hymeko_nagare/examples/entropy_pool_learning_compare.rs`
- `hymeko_nagare/tests/entropy_pool_learning_compare.rs`
- `reports/2026-07-02-entropy-gate-stress-ablation.json`
- `reports/2026-07-02-entropy-gate-stress-ablation-rss.json`

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
rustfmt --check hymeko_nagare\examples\entropy_pool_learning_compare.rs hymeko_nagare\tests\entropy_pool_learning_compare.rs
cargo test -p hymeko_nagare --test entropy_pool_learning_compare -- --nocapture
cargo clippy -p hymeko_nagare --example entropy_pool_learning_compare --test entropy_pool_learning_compare --no-deps -- -D warnings
cargo run -p hymeko_nagare --release --example entropy_pool_learning_compare -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --hidden 24 --epochs 50 --batch-size 32 --lr 0.05 --seed 53 --out reports\2026-07-02-entropy-gate-stress-ablation.json
```

All checks passed.

## Next

The next entropy rule should not simply scale the supervised output update by
uncertainty. More plausible variants:

- entropy as a target/homeostatic term,
- entropy derivative or entropy acceleration feedback,
- class-conditional entropy floor,
- entropy gating on structural feature generation, not just readout update,
- local rule with no labels in the inner update, using labels only for outer
  evaluation.
