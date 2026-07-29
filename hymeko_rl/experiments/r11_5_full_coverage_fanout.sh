#!/usr/bin/env bash
# R11.5 full-51 coverage fan-out launcher (kato14).
# Frozen protocol: 51 certified-grasp DELIVERY_FAILURE states, transport-only solve (full_transport_spec,
# SAME coordinate/objective as the re-gate), delivery CEM restarts R=11, capture-seeds<=5, NO new param/score change,
# every scenario kept in the shard ledger. Fans 51 scenarios across N workers by --offset/--limit, one shard each,
# then merges and runs the 3-verdict coverage gate over all rows.
#
# Usage (from the worktree root, on kato14):
#   setsid bash hymeko_rl/experiments/r11_5_full_coverage_fanout.sh > reports/2026-07-30-r11-5-coverage/fanout.log 2>&1 &
set -euo pipefail

WT="${WT:-/home/hajdu/hymeko_coin_r9_wt}"
VENV="${VENV:-/home/hajdu/hymeko_framework_rust/.venv}"
OUTDIR="${OUTDIR:-$WT/reports/2026-07-30-r11-5-coverage}"
WORKERS="${WORKERS:-16}"
RESTARTS="${RESTARTS:-11}"
CAPTURE_SEEDS="${CAPTURE_SEEDS:-5}"
N_FAILS="${N_FAILS:-51}"

export PYTHONPATH="$WT"
mkdir -p "$OUTDIR"
cd "$WT"

# Contiguous, balanced chunks over N_FAILS: first (N_FAILS % WORKERS) workers get one extra.
base=$(( N_FAILS / WORKERS )); rem=$(( N_FAILS % WORKERS )); off=0; pids=()
echo "=== R11.5 FULL-51 FAN-OUT === workers=$WORKERS R=$RESTARTS capture_seeds=$CAPTURE_SEEDS fails=$N_FAILS $(date -u +%FT%TZ)"
for (( w=0; w<WORKERS; w++ )); do
  lim=$(( base + (w < rem ? 1 : 0) ))            # first `rem` workers get one extra; set-e-safe (no `&&` guard)
  if [ "$lim" -eq 0 ]; then continue; fi
  shard="$OUTDIR/shard_$(printf '%02d' "$w").jsonl"
  echo "  worker $w: offset=$off limit=$lim -> $shard"
  "$VENV/bin/python" -m hymeko_rl.experiments.r11_5_full_coverage \
      --offset "$off" --limit "$lim" --restarts "$RESTARTS" --capture-seeds "$CAPTURE_SEEDS" \
      --out "$shard" > "$OUTDIR/worker_$(printf '%02d' "$w").log" 2>&1 &
  pids+=($!)
  off=$(( off + lim ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "=== workers done (fail=$fail) $(date -u +%FT%TZ) ==="

# Merge shards + run the 3-verdict coverage gate over all rows.
cat "$OUTDIR"/shard_*.jsonl > "$OUTDIR/coverage.jsonl"
"$VENV/bin/python" - "$OUTDIR/coverage.jsonl" <<'PY'
import json, sys
from hymeko_rl.experiments.r11_5_full_coverage import coverage_gate
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"merged rows: {len(rows)}")
print(json.dumps(coverage_gate(rows), indent=2))
print("R11_5_FULL_COVERAGE_FANOUT_DONE")
PY
