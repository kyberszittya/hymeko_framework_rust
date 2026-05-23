#!/usr/bin/env bash
set -euo pipefail

# Build a source-focused zip suitable for Claude/Clause uploads.
# Keeps common source/docs/config files, excludes build artifacts and images.

usage() {
  cat <<'EOF'
Usage:
  ./create_clause_upload_zip.sh [repo_root] [output_zip]

Examples:
  ./create_clause_upload_zip.sh
  ./create_clause_upload_zip.sh /path/to/repo /tmp/hymeko_clause_upload.zip
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: 'zip' is required but not installed." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$SCRIPT_DIR}"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Error: repo root does not exist: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/Cargo.toml" ]]; then
  echo "Error: '$REPO_ROOT' does not look like this workspace root (missing Cargo.toml)." >&2
  exit 1
fi

if [[ -n "${2:-}" ]]; then
  OUTPUT_ZIP="$2"
else
  OUTPUT_ZIP="$REPO_ROOT/clause_upload_$(date +%Y%m%d_%H%M%S).zip"
fi

mkdir -p "$(dirname "$OUTPUT_ZIP")"

INCLUDE_PATTERNS=(
  "*.rs" "*.lalrpop" "*.md"
  "*.toml" "*.lock" "*.yml" "*.yaml" "*.json"
  "*.txt" "*.rst" "*.hymeko"
  "*.py" "*.sh" "*.ps1" "*.bat" "*.mk"
  "Makefile" ".gitignore" ".gitattributes"
)

EXCLUDE_PATTERNS=(
  "target/*" "*/target/*"
  ".git/*" "*/.git/*"
  ".idea/*" "*/.idea/*"
  ".vscode/*" "*/.vscode/*"
  "*.jpg" "*.jpeg" "*.png" "*.gif" "*.bmp" "*.webp" "*.svg" "*.ico"
  "*.zip"
)

pushd "$REPO_ROOT" >/dev/null

rm -f "$OUTPUT_ZIP"
zip -q -r "$OUTPUT_ZIP" . -i "${INCLUDE_PATTERNS[@]}" -x "${EXCLUDE_PATTERNS[@]}"

FILE_COUNT="$(zipinfo -1 "$OUTPUT_ZIP" | wc -l | tr -d ' ')"
if [[ "$FILE_COUNT" == "0" ]]; then
  rm -f "$OUTPUT_ZIP"
  echo "Error: no files matched include patterns; archive not created." >&2
  popd >/dev/null
  exit 1
fi

ARCHIVE_SIZE="$(du -h "$OUTPUT_ZIP" | awk '{print $1}')"

echo "Created: $OUTPUT_ZIP"
echo "Files:   $FILE_COUNT"
echo "Size:    $ARCHIVE_SIZE"

echo
echo "Preview (first 20 entries):"
zipinfo -1 "$OUTPUT_ZIP" | head -20

popd >/dev/null

