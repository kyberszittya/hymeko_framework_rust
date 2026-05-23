#!/usr/bin/env bash
#
# Demo — emit every visualization / transform for a robotics fixture.
#
# Uses `hymeko_cli emit` to produce URDF, SDF, MJCF, DOT, and ROS2 launch
# outputs for `mini_arm.hymeko`, then pipes the DOT render through
# `graphviz` (if available) to produce an SVG.
#
# Run from the workspace root:
#   bash scripts/demo_visualizations.sh [optional/path/to/file.hymeko]
#
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT="${1:-data/robotics/mini_arm.hymeko}"
OUT="out/visualizations/$(basename "${INPUT%.hymeko}")"
mkdir -p "$OUT"

echo "==> Emitting all registered formats for $INPUT -> $OUT/"
cargo run --quiet -p hymeko_cli -- emit --all "$INPUT" -o "$OUT" || {
    echo "    emit failed — falling back to per-format calls"
    for fmt in urdf sdf mjcf dot; do
        echo "    -- $fmt --"
        cargo run --quiet -p hymeko_cli -- emit --format "$fmt" "$INPUT" > "$OUT/$(basename ${INPUT%.hymeko}).$fmt" || true
    done
}

echo
echo "==> Generated files"
ls -la "$OUT"

if command -v dot >/dev/null 2>&1; then
    DOTFILE=$(find "$OUT" -name "*.dot" | head -1)
    if [[ -n "$DOTFILE" ]]; then
        SVG="${DOTFILE%.dot}.svg"
        dot -Tsvg "$DOTFILE" -o "$SVG"
        echo
        echo "==> Rendered DOT -> $SVG"
    fi
else
    echo
    echo "==> Install graphviz (\`apt install graphviz\`) to auto-render DOT -> SVG."
fi

echo
echo "==> Docs: docs/examples/visualizations.md has six hand-authored renders."
