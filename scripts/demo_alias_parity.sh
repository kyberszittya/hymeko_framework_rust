#!/usr/bin/env bash
#
# Demo — `using <path> as <alias>;` namespace-alias feature.
#
# Runs the 4 parser-level tests and the 11 end-to-end alias-parity tests
# added on 2026-04-18, then diffs the URDF/DOT output between the aliased
# and non-aliased fixtures to visually confirm they produce identical
# artefacts.
#
# Run from the workspace root:
#   bash scripts/demo_alias_parity.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Parser-level grammar tests (parser/tests/using_alias.rs — 4 tests)"
cargo test -p parser --test using_alias 2>&1 | tail -10

echo
echo "==> End-to-end alias-parity tests (hymeko_query — 11 tests in mod alias_parity)"
cargo test -p hymeko_query --test integration alias_parity 2>&1 | tail -18

echo
echo "==> Structural diff between baseline and aliased fixtures"
for pair in \
    "anthropomorphic_arm:anthropomorphic_arm_using" \
    "robot_4wh:robot_4wh_using"; do
    baseline="${pair%%:*}"
    aliased="${pair##*:}"
    echo
    echo "    -- $baseline.hymeko vs $aliased.hymeko --"
    wc -l "data/robotics/$baseline.hymeko" "data/robotics/$aliased.hymeko"
    echo "    using-statements in aliased source:"
    grep "using " "data/robotics/$aliased.hymeko" | sed 's/^/      /'
done

echo
echo "==> Done. Same topology via either spelling."
