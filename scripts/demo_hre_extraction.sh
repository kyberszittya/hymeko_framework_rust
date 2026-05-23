#!/usr/bin/env bash
#
# Demo — verify the hymeko_hre crate extraction (2026-04-18).
#
# Exercises: the new crate compiles standalone, its two integration tests
# pass, hymeko_core no longer ships an `engine` module, and `hymeko_py`
# correctly re-imports HypergraphEngine from hymeko_hre.
#
# Run from the workspace root:
#   bash scripts/demo_hre_extraction.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Workspace members"
grep -A 10 '^\[workspace\]' Cargo.toml | head -12

echo
echo "==> hymeko_hre crate layout"
find hymeko_hre -type f \( -name '*.toml' -o -name '*.rs' \) | sort

echo
echo "==> cargo check — full workspace"
cargo check --workspace --quiet 2>&1 | tail -5

echo
echo "==> cargo test — hymeko_hre (integration: 2 tests)"
cargo test -p hymeko_hre 2>&1 | grep -E "^(running|test result)"

echo
echo "==> Confirm hymeko_core no longer exposes ::engine"
if cargo check -p hymeko_core --quiet 2>&1 && ! grep -q "pub mod engine" hymeko_core/src/lib.rs; then
    echo "    ok — hymeko_core::engine removed"
else
    echo "    FAIL — engine module still in hymeko_core"
    exit 1
fi

echo
echo "==> hymeko_py import (should reference hymeko_hre now)"
grep -n "use hymeko_hre\|use hymeko::engine" hymeko_py/src/interface_python/api.rs || true

echo
echo "==> Done."
