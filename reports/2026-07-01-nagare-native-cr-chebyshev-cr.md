# Native Nagare CR and Chebyshev-CR

Created-at: 2026-07-01 15:26 JST

## Summary

Added native Nagare/Rust Catmull-Rom and Chebyshev-CR operators. The implementation mirrors the canonical `hymeko_neuro.core.splines` convention: CR clamps inputs to `[-1, 1]`, uses uniform cubic Catmull-Rom with boundary control-point replication, and Chebyshev-CR supports both train mode (`coef -> control points -> CR`) and direct Chebyshev deploy mode.

## Files Touched

- `docs/plans/2026-07-01-nagare-native-cr-chebyshev-cr/plan.tex` (+45)
- `docs/plans/2026-07-01-nagare-native-cr-chebyshev-cr/plan.pdf`
- `docs/plans/2026-07-01-nagare-native-cr-chebyshev-cr/plan.tikz` (+13)
- `docs/plans/2026-07-01-nagare-native-cr-chebyshev-cr/plan.mmd` (+9)
- `hymeko_nagare/src/ops/catmull_rom.rs` (+455)
- `hymeko_nagare/src/ops/mod.rs`
- `hymeko_nagare/src/lib.rs`

## CORE.YAML

No CORE.YAML-protected files touched. No dependency changes.

## Native API

New exported functions/types:

- `catmull_rom_forward`
- `catmull_rom_backward`
- `chebyshev_knot_basis`
- `chebyshev_control_points`
- `chebyshev_cr_forward`
- `chebyshev_cr_backward`
- `chebyshev_deploy_forward`
- `chebyshev_deploy_backward`
- `CatmullRomCache`
- `CatmullRomBackward`
- `ChebyshevCrBackward`

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
rustfmt --check hymeko_nagare\src\ops\catmull_rom.rs hymeko_nagare\src\ops\mod.rs hymeko_nagare\src\lib.rs
cargo test -p hymeko_nagare catmull_rom -- --nocapture
cargo clippy -p hymeko_nagare --lib --no-deps -- -D warnings
```

Results:

- Plan PDF built successfully.
- Touched Rust files pass `rustfmt --check`.
- `cargo test -p hymeko_nagare catmull_rom`: 5 passed.
- Targeted `clippy --lib --no-deps`: passed.

## Test Coverage

The native tests cover:

- CR interpolation exactly at control-point knots.
- CR backward finite-difference parity for both input and control coefficients.
- Chebyshev knot basis/control-point generation.
- Direct Chebyshev deploy backward finite-difference parity for both input and coefficients.
- Chebyshev-CR train backward shape/finite sanity through the CR chain rule.

## Open Issues

These kernels are now native but not yet wired into the global-pool/entropy-feedback toy or the larger HSIKAN/FSR harnesses. The next integration step is to replace ReLU/GELU toy activations with CR or Chebyshev-CR variants and then re-run the PyTorch parity path against the native operators.
