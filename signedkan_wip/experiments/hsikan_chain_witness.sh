#!/usr/bin/env bash
# Append UTC stamp + ``wc -l`` for each JSONL that exists (used after HSiKAN
# transient chains complete).
set -euo pipefail
WITNESS="${1:?usage: hsikan_chain_witness.sh <witness.txt> <jsonl> [jsonl ...]}"
shift
{
  echo "witness_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for f in "$@"; do
    if [[ -f "${f}" ]]; then
      wc -l "${f}"
    else
      echo "missing ${f}"
    fi
  done
} >> "${WITNESS}"
