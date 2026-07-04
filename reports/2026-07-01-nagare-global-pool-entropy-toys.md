# Nagare Global Pool + Entropy Feedback Toys

Created-at: 2026-07-01 04:43 JST

## Summary

Ported the moons/rings/xor point-cloud toy examples to a Nagare-side Rust harness. The new example uses Nagare `LinearLayer`, `linear_backward`, and `AdamState` for explicit closed-form training, plus local closed-form backward code for ReLU, global `mean/std/max` pooling, and softmax cross entropy. No PyTorch/autograd is used in the Nagare run.

## Files Touched

- `docs/plans/2026-07-01-nagare-global-pool-entropy-toys/plan.tex` (+49)
- `docs/plans/2026-07-01-nagare-global-pool-entropy-toys/plan.pdf`
- `docs/plans/2026-07-01-nagare-global-pool-entropy-toys/plan.tikz` (+19)
- `docs/plans/2026-07-01-nagare-global-pool-entropy-toys/plan.mmd` (+12)
- `hymeko_nagare/examples/global_pool_entropy_toys.rs` (+766)
- `hymeko_nagare/tests/global_pool_entropy_toys.rs` (+44)
- `reports/2026-07-01-nagare-global-pool-entropy-toys.json` (+80)
- `reports/2026-07-01-nagare-global-pool-entropy-toys.svg`
- `reports/2026-07-01-nagare-global-pool-entropy-toys-rss-run.txt`

## CORE.YAML

No CORE.YAML-protected items touched. No dependency changes.

## Results

Run:

```powershell
cargo run --release -p hymeko_nagare --example global_pool_entropy_toys -- --tasks moons,rings,xor --n-train 192 --n-test 96 --n-points 48 --hidden 32 --epochs 80 --batch-size 32 --out reports\2026-07-01-nagare-global-pool-entropy-toys.json
```

| task | model | test acc | test entropy | median forward / sample | params | param bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| moons | baseline | 1.000 | 0.017723 | 15.03 us | 290 | 1,160 |
| moons | entropy feedback | 1.000 | 0.000384 | 91.21 us | 4,644 | 18,576 |
| rings | baseline | 1.000 | 0.064207 | 16.28 us | 290 | 1,160 |
| rings | entropy feedback | 1.000 | 0.000207 | 91.44 us | 4,644 | 18,576 |
| xor | baseline | 1.000 | 0.016248 | 15.97 us | 290 | 1,160 |
| xor | entropy feedback | 1.000 | 0.000590 | 82.66 us | 4,644 | 18,576 |

Plot: `reports/2026-07-01-nagare-global-pool-entropy-toys.svg`.

## PyTorch Comparison

The earlier PyTorch CPU harness used a deeper GELU MLP, so this is not architectural parity. Still, the Nagare CPU harness lands in the same microsecond regime with much lower framework overhead:

| model | PyTorch CPU median | Nagare CPU median |
| --- | ---: | ---: |
| baseline | 19.40-20.71 us | 15.03-16.28 us |
| entropy feedback | 46.77-60.13 us | 82.66-91.44 us |

The baseline is faster in Nagare. Entropy-feedback is slower here because the Rust harness currently materializes and backprops a wide broadcast update tensor `(batch * points, 4h + 1)` without fusion. That identifies the next optimization target: fuse broadcast-context construction, update linear, and pooling backward.

## Memory

Peak working set from a release executable run sampled by PowerShell: 10.27 MiB. Parameter memory is tiny: 1.16 KiB for the baseline and 18.58 KiB for entropy-feedback. As expected, executable/runtime overhead dominates the live model parameters.

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
rustfmt --check hymeko_nagare\examples\global_pool_entropy_toys.rs hymeko_nagare\tests\global_pool_entropy_toys.rs
cargo test -p hymeko_nagare --test global_pool_entropy_toys -- --nocapture
cargo clippy -p hymeko_nagare --no-deps --example global_pool_entropy_toys --test global_pool_entropy_toys -- -D warnings
```

Results:

- plan PDF built successfully.
- `cargo test`: 3 passed.
- targeted `clippy --no-deps`: passed.
- full `cargo clippy -p hymeko_nagare ... -D warnings` is blocked by pre-existing warnings in `hymeko_graph`; no CORE edits were made for that.
- `cargo fmt -p hymeko_nagare --check` is blocked by pre-existing formatting drift outside this change; the new files pass direct `rustfmt --check`.

## Open Issues

The tasks are again saturated, so accuracy is a sanity check rather than a discriminating benchmark. The entropy-feedback Nagare path should be fused before treating it as a performance result for the intended substrate.
