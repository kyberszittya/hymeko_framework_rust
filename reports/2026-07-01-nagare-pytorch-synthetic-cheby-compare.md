# PyTorch vs Nagare on Small Synthetic Problems

Created-at: 2026-07-01 15:51 JST

## Summary

Compared PyTorch CPU and Nagare/Rust CPU on three small synthetic classification datasets: moons, spiral, and XOR. The comparison uses an exact shared fixture: PyTorch generates inputs and weights for a small `2 -> 32 -> 2` classifier with direct Chebyshev deploy activation (`k=5`) and Chebyshev-domain rescale only (`scale=0.5`), then Nagare parses the same plain text fixture and runs the same forward.

This is a forward/runtime comparison, not a training-quality benchmark. Accuracies are from random fixed weights and are only a sanity check that both engines see the same problem.

## Files Touched

- `docs/plans/2026-07-01-nagare-pytorch-synthetic-cheby-compare/plan.tex`
- `docs/plans/2026-07-01-nagare-pytorch-synthetic-cheby-compare/plan.pdf`
- `docs/plans/2026-07-01-nagare-pytorch-synthetic-cheby-compare/plan.tikz`
- `docs/plans/2026-07-01-nagare-pytorch-synthetic-cheby-compare/plan.mmd`
- `scripts/dev/synthetic_cheby_compare.py`
- `hymeko_nagare/examples/synthetic_cheby_compare.rs`
- `reports/2026-07-01-nagare-pytorch-synthetic-cheby-compare-fixture.txt`
- `reports/2026-07-01-nagare-pytorch-synthetic-cheby-compare-pytorch.json`
- `reports/2026-07-01-nagare-pytorch-synthetic-cheby-compare-nagare.json`

## CORE.YAML

No CORE.YAML-protected files touched. No dependency changes.

## Results

Shape: `n=256`, input dim `2`, hidden `32`, Chebyshev `k=5`, output dim `2`. Activation policy: `linear -> 0.5 * z -> Chebyshev deploy`; no `tanh` bound.

| task | max abs logits | acc | PyTorch median | Nagare median | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| moons | 7.45e-8 | 0.500 | 4.05 us | 2.18 us | 1.86x |
| spiral | 8.94e-8 | 0.500 | 5.43 us | 3.10 us | 1.75x |
| xor | 1.04e-7 | 0.477 | 4.27 us | 2.54 us | 1.68x |

RSS samples:

| engine | peak RSS |
| --- | ---: |
| PyTorch process tree | 552.89 MiB |
| Nagare release executable | 5.39 MiB |

## Commands

```powershell
uv run --group ml python scripts\dev\synthetic_cheby_compare.py --tasks moons spiral xor --n-samples 256 --hidden 32 --k 5 --seed 321 --repeats 300 --fixture reports\2026-07-01-nagare-pytorch-synthetic-cheby-compare-fixture.txt --summary reports\2026-07-01-nagare-pytorch-synthetic-cheby-compare-pytorch.json
cargo run --release -p hymeko_nagare --example synthetic_cheby_compare -- --fixture reports\2026-07-01-nagare-pytorch-synthetic-cheby-compare-fixture.txt --repeats 300 --out reports\2026-07-01-nagare-pytorch-synthetic-cheby-compare-nagare.json
```

## Verification

```powershell
pdflatex -interaction=nonstopmode -halt-on-error plan.tex
uv run ruff check scripts\dev\synthetic_cheby_compare.py
uv run --group ml python -m py_compile scripts\dev\synthetic_cheby_compare.py
rustfmt --check hymeko_nagare\examples\synthetic_cheby_compare.rs
cargo clippy -p hymeko_nagare --example synthetic_cheby_compare --no-deps -- -D warnings
```

Results: plan PDF built; Ruff passed; py_compile passed; rustfmt passed; targeted clippy passed.

## Readout

Nagare matches PyTorch numerically at about `1e-7` absolute logit error and is faster on all three small synthetic forwards. Removing the old `tanh` bound also made PyTorch much faster, so these are the more honest Chebyshev-domain-policy numbers. The bigger win is memory footprint: the Rust executable is tiny compared with the PyTorch runtime for this class of small deployed model.
