# Nagare CR/Chebyshev-CR Small Performance

Created-at: 2026-07-01 15:33 JST

## Summary

Checked native Nagare CR and Chebyshev-CR performance on small descriptor-like tensors and the small point-cloud toy shape. The first pass exposed an inefficient direct Chebyshev deploy implementation that allocated per scalar; I removed that temporary allocation and reran the measurements.

## Files Touched

- `docs/plans/2026-07-01-nagare-cr-cheby-small-perf/plan.tex`
- `docs/plans/2026-07-01-nagare-cr-cheby-small-perf/plan.pdf`
- `docs/plans/2026-07-01-nagare-cr-cheby-small-perf/plan.tikz`
- `docs/plans/2026-07-01-nagare-cr-cheby-small-perf/plan.mmd`
- `hymeko_nagare/examples/cr_chebyshev_small_perf.rs`
- `hymeko_nagare/src/ops/catmull_rom.rs`
- `reports/2026-07-01-nagare-cr-cheby-small-perf.json`
- `reports/2026-07-01-nagare-cr-cheby-small-perf-rss.txt`

## CORE.YAML

No CORE.YAML-protected files touched. No dependency changes.

## Results

Release command:

```powershell
target\release\examples\cr_chebyshev_small_perf.exe --repeats 500 --out reports\2026-07-01-nagare-cr-cheby-small-perf.json
```

Median microseconds per sample:

| case | shape | CR train | Cheb-CR train | Cheb deploy |
| --- | --- | ---: | ---: | ---: |
| tiny descriptor | `n=1, c=16, grid=8, k=5` | 1.1000 | 2.1000 | 0.4000 |
| small descriptions | `n=8, c=32, grid=12, k=5` | 1.0750 | 0.8875 | 0.2375 |
| toy point batch | `n=4608, c=32, grid=12, k=5` | 1.0717 | 1.0013 | 0.4043 |

Peak working set for the release run: 7.80 MiB.

## Readout

Direct Chebyshev deploy is the clear small-forward winner after removing per-scalar temporary allocation: about `2.7x` faster than CR on the tiny descriptor, `4.5x` faster on the small description batch, and `2.65x` faster on the point-cloud toy shape.

Cheb-CR train mode is roughly CR-speed at larger toy shape, but worse for `n=1` because it still builds Chebyshev-derived control points before CR evaluation. For inference/deploy, use direct Chebyshev. For training-quality CR interpolation, use Cheb-CR train and cache the knot basis.

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
rustfmt --check hymeko_nagare\src\ops\catmull_rom.rs hymeko_nagare\examples\cr_chebyshev_small_perf.rs
cargo test -p hymeko_nagare catmull_rom -- --nocapture
cargo clippy -p hymeko_nagare --lib --example cr_chebyshev_small_perf --no-deps -- -D warnings
```

Results: plan PDF built, rustfmt passed, native CR/Cheb tests passed (`5 passed`), targeted clippy passed.

## Next

Wire `chebyshev_deploy_forward` into the global-pool entropy forward as the activation fast path, then rerun the exact PyTorch parity fixture and forward/RSS comparison.
