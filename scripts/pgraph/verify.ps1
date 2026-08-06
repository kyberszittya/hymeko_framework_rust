# Canonical-correctness verification for the HyMeKo P-graph engine (MSG/ABB).
# Runs the Rust conformance test suite + the book-example CLI conformance check.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

Write-Host "== 1/2  Rust canonical-correctness suite (hymeko_pgraph) =="
# book_validation (book values), axiom_witness (S1-S5), ssg_decision_mapping
# (19 / 3465 structure counts), relaxed_msg, pgraph_e2e, regime tests, ...
cargo test -p hymeko_pgraph
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "== 2/2  Book-example CLI conformance (MSG/ABB vs published values) =="
python scripts/pgraph/run_examples.py --regimes
exit $LASTEXITCODE
