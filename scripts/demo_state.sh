#!/usr/bin/env bash
#
# Demo — print the framework state snapshot.
#
# Combines: workspace crate list, test-per-crate counts, a pointer to
# docs/STATE.md, and the list of dated changelog entries. Useful as a
# "status at a glance" check before picking up new work.
#
# Run from the workspace root:
#   bash scripts/demo_state.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Workspace crates"
sed -n '/^members = \[/,/^\]/p' Cargo.toml | sed 's/^/    /'

echo
echo "==> Test counts per crate (cargo test --workspace)"
cargo test --workspace 2>&1 \
    | awk '/^     Running / {bin=$2} /^test result/ {print "    " bin, $0}' \
    | head -20

echo
echo "==> Recent dated changelogs"
ls docs/changelog/ | sort -r | head -8 | sed 's/^/    /'

echo
echo "==> Pending uncommitted work (top 20 lines)"
git -C "$(pwd)" status --short | head -20

echo
echo "==> See docs/STATE.md for the full snapshot."
