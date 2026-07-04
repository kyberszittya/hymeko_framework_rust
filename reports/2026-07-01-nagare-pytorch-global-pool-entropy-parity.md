# Nagare/PyTorch Global-Pool Entropy Forward Parity

Created-at: 2026-07-01 15:24 JST

## Summary

Implemented exact CPU forward parity for the global-pool plus entropy-feedback toy model. PyTorch exports identical moons/rings/xor test inputs, ReLU model weights, expected logits, first-pass logits, and entropy into a plain text fixture. Nagare/Rust parses that fixture with the standard library, runs the same forward, checks max absolute error, and reports forward time plus allocation estimates.

## Files Touched

- `docs/plans/2026-07-01-nagare-pytorch-forward-parity/plan.tex`
- `docs/plans/2026-07-01-nagare-pytorch-forward-parity/plan.pdf`
- `docs/plans/2026-07-01-nagare-pytorch-forward-parity/plan.tikz`
- `docs/plans/2026-07-01-nagare-pytorch-forward-parity/plan.mmd`
- `scripts/dev/global_pool_entropy_parity_fixture.py`
- `hymeko_nagare/examples/global_pool_entropy_parity.rs`
- `hymeko_nagare/tests/global_pool_entropy_parity.rs`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-fixture.txt`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-pytorch.json`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-pytorch-rss-summary.json`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-pytorch-rss.txt`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-pytorch-rss.err`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-nagare.json`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity-nagare-rss.txt`
- `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity.svg`

## CORE.YAML

No CORE.YAML-protected files touched. No dependency changes.

## Commands

```powershell
uv run --group ml python scripts\dev\global_pool_entropy_parity_fixture.py --tasks moons rings xor --n-train 192 --n-test 96 --n-points 48 --hidden 32 --seed 123 --repeats 300 --fixture reports\2026-07-01-nagare-pytorch-global-pool-entropy-parity-fixture.txt --summary reports\2026-07-01-nagare-pytorch-global-pool-entropy-parity-pytorch.json
cargo run --release -p hymeko_nagare --example global_pool_entropy_parity -- --fixture reports\2026-07-01-nagare-pytorch-global-pool-entropy-parity-fixture.txt --repeats 300 --out reports\2026-07-01-nagare-pytorch-global-pool-entropy-parity-nagare.json
```

## Parity

| task | max abs logits | max abs first logits | max abs entropy |
| --- | ---: | ---: | ---: |
| moons | 8.94e-8 | 1.79e-7 | 1.79e-7 |
| rings | 1.04e-7 | 1.79e-7 | 1.79e-7 |
| xor | 1.04e-7 | 1.49e-7 | 1.79e-7 |

All cases are below the `1e-5` parity budget.

## Performance

Median CPU forward time, microseconds per sample:

| task | PyTorch | Nagare/Rust | Rust/PyTorch |
| --- | ---: | ---: | ---: |
| moons | 60.49 us | 94.61 us | 1.56x |
| rings | 55.51 us | 93.24 us | 1.68x |
| xor | 62.36 us | 91.54 us | 1.47x |

Plot: `reports/2026-07-01-nagare-pytorch-global-pool-entropy-parity.svg`.

## Memory

| engine | allocation measure | allocation bytes/forward | peak RSS |
| --- | --- | ---: | ---: |
| PyTorch CPU | profiler positive CPU memory events | 4,926,736 | 663.05 MiB process tree |
| Nagare/Rust | instrumented Vec allocation estimate | 4,812,672 | 11.00 MiB process |

The per-forward allocation volume is similar because both unfused forwards materialize the broadcast update tensor. Process memory is very different: PyTorch carries the Python/Torch runtime and allocator stack; the Rust executable stays near 11 MiB.

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
uv run ruff check scripts\dev\global_pool_entropy_parity_fixture.py
uv run --group ml python -m py_compile scripts\dev\global_pool_entropy_parity_fixture.py
rustfmt --check hymeko_nagare\examples\global_pool_entropy_parity.rs hymeko_nagare\tests\global_pool_entropy_parity.rs
cargo test -p hymeko_nagare --test global_pool_entropy_parity -- --nocapture
cargo clippy -p hymeko_nagare --no-deps --example global_pool_entropy_parity --test global_pool_entropy_parity -- -D warnings
```

Results: plan PDF built; Ruff passed; py_compile passed; rustfmt passed for new Rust files; parity test passed; targeted clippy passed.

## Readout

This isolates the current bottleneck. Nagare/Rust is numerically correct, uses far less resident memory, but is slower than PyTorch on this exact unfused forward because the implementation still materializes `update_x = [h, pooled, entropy]` for every point and then runs generic linear/pool passes. The next speed step is a fused entropy-feedback op that streams pooled context directly into the update linear and avoids the 4.8 MB temporary allocation.
