# Nagare Holonomy Entropy Toys

Created-at: 2026-07-01 17:18 JST

## Summary

Implemented a Nagare-native holonomy toy path with:

- quaternion periodic feature lift for point-set inputs,
- fused global-pool entropy update without materialising `[h, pool, H]`,
- closed-form supervised updates through Nagare linear/pool/fused kernels,
- Clifford `Cl(2,0)` probability-vector error as an additional diagnostic.

This is not a PyTorch-shaped model. The periodic component is generated with
quaternion transport (`cayley_to_unit_quat`, `quat_mul`, `quat_rotate`), and the
error diagnostic embeds probability residuals as grade-1 vectors in
`Cl(2,0)`.

## Files

- `docs/plans/2026-07-01-nagare-holonomy-fused-toys/plan.{tex,pdf,tikz,mmd}`
- `hymeko_nagare/src/ops/fused_entropy_update.rs`
- `hymeko_nagare/src/ops/mod.rs`
- `hymeko_nagare/src/lib.rs`
- `hymeko_nagare/examples/holonomy_entropy_toys.rs`
- `hymeko_nagare/tests/holonomy_entropy_toys.rs`
- `reports/2026-07-01-nagare-holonomy-entropy-toys.json`
- `reports/2026-07-01-nagare-holonomy-entropy-toys-rss.json`

## Results

Shape: `n_train=192`, `n_test=96`, `points=32`, `hidden=24`, `epochs=50`.

| task | test acc | CE loss | entropy | Clifford error | median forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| moons | 1.000 | 0.000033 | 0.000515 | 0.000000 | 230.71 us/sample |
| spiral | 1.000 | 0.000046 | 0.000697 | 0.000000 | 214.39 us/sample |
| xor | 1.000 | 0.000086 | 0.001237 | 0.000000 | 225.29 us/sample |

Memory/readout:

| metric | value |
| --- | ---: |
| params | 2,836 |
| param bytes | 11,344 |
| materialised update buffer estimate | 1,191,936 bytes |
| fused update buffer estimate | 294,912 bytes |
| update-buffer reduction | 75.26% |
| sampled peak RSS | 7.61 MiB |

## Interpretation

The fused op now expresses the intended Nagare path directly:

```text
h_i, pooled_context, entropy -> update linear
```

rather than:

```text
materialise [h_i, pooled_context, entropy] -> generic linear
```

For this toy shape, the update buffer estimate drops from `1,191,936` bytes to
`294,912` bytes. The model solves the three easy synthetic point-set tasks; the
important result is architectural, not task difficulty.

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
rustfmt --check hymeko_nagare\src\ops\fused_entropy_update.rs hymeko_nagare\src\lib.rs hymeko_nagare\examples\holonomy_entropy_toys.rs hymeko_nagare\tests\holonomy_entropy_toys.rs
cargo test -p hymeko_nagare --test holonomy_entropy_toys -- --nocapture
cargo test -p hymeko_nagare fused_entropy_update -- --nocapture
cargo clippy -p hymeko_nagare --example holonomy_entropy_toys --test holonomy_entropy_toys --no-deps -- -D warnings
cargo run -p hymeko_nagare --release --example holonomy_entropy_toys -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --hidden 24 --epochs 50 --batch-size 32 --lr 0.002 --seed 37 --out reports\2026-07-01-nagare-holonomy-entropy-toys.json
```

All listed checks passed. The RSS run used the release executable with a
PowerShell polling wrapper and wrote
`reports/2026-07-01-nagare-holonomy-entropy-toys-rss.json`.

## Next

The next discriminating test should compare this holonomy feature path against
the older non-holonomy entropy toy at matched hidden size and task difficulty.
The present task establishes the native substrate; the next task should isolate
whether quaternion holonomy features improve accuracy/latency tradeoffs on
harder generated problems.
