#!/usr/bin/env bash
#
# komondor_morning_pull.sh -- one-shot script for the morning after
# the overnight chain 13885723 (Slashdot + Epinions edge_cr 5-seed)
# plus the BA+OTC chain 13885739.
#
# Pulls JSONL + orchestrator logs + SLURM logs from Komondor into
# the local repo's hsikan_edge_cr_audit/ + slurm_logs/ dirs, then
# runs komondor_audit_metrics.py with no args (it reads the hardcoded
# REPO-rooted paths).
#
# Usage (after `ssh komondor` and EduID 2FA in a separate terminal):
#
#   ./scripts/komondor_morning_pull.sh
#
# Idempotent: re-running just refreshes the local copies. The audit
# summary is written to reports/2026-06-04-overnight-audit-summary.md.

set -uo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO"

REMOTE_REPO=/scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust
SUMMARY="$REPO/reports/2026-06-04-overnight-audit-summary.md"
mkdir -p "$REPO/hsikan_edge_cr_audit" "$REPO/slurm_logs" \
         "$REPO/komondor_results/2026-06-04-overnight"

LOG="$REPO/komondor_results/2026-06-04-overnight/pull.log"
exec > >(tee -a "$LOG") 2>&1

echo "[morning-pull] $(date -Iseconds)  REPO=$REPO"

# 1. SLURM status snapshot.
echo "[morning-pull] querying squeue / sacct for the two chain jobs..."
ssh komondor "
    echo '=== squeue (running) ===';
    squeue -h -o '%i %T %M %L %j' -u \$USER 2>/dev/null || true;
    echo;
    echo '=== sacct (chain jobs) ===';
    sacct -j 13885723,13885739 -X \
        --format=JobID,JobName%30,State,Elapsed,MaxRSS,Start,End -P \
        2>/dev/null || true;
    echo;
    echo '=== sacct (per-step detail, last 20) ===';
    sacct -j 13885723,13885739 \
        --format=JobID,JobName%30,State,Elapsed,MaxRSS -P 2>/dev/null \
        | tail -20 || true
" > "$REPO/komondor_results/2026-06-04-overnight/slurm_status.txt"
cat "$REPO/komondor_results/2026-06-04-overnight/slurm_status.txt" | head -30

# 2. Audit JSONL + per-cell logs.
echo
echo "[morning-pull] rsync hsikan_edge_cr_audit/ ..."
rsync -av --partial --info=stats1 \
    "komondor:$REMOTE_REPO/hsikan_edge_cr_audit/" \
    "$REPO/hsikan_edge_cr_audit/" 2>&1 | tail -8

# 3. SLURM logs (latest 20).
echo
echo "[morning-pull] rsync slurm_logs/ (latest 20)..."
ssh komondor "ls -t $REMOTE_REPO/slurm_logs/ 2>/dev/null | head -20" \
    | xargs -I{} rsync -a --partial \
        "komondor:$REMOTE_REPO/slurm_logs/{}" \
        "$REPO/slurm_logs/" 2>/dev/null || true

# 4. Aggregate audit.
echo
echo "[morning-pull] running komondor_audit_metrics.py ..."
python3 scripts/komondor_audit_metrics.py | tee "$SUMMARY"

# 5. One-screen verdict.
echo
echo "========================================================"
echo "                    MORNING SUMMARY"
echo "========================================================"
echo "  pull log:          $LOG"
echo "  slurm status:      $REPO/komondor_results/2026-06-04-overnight/slurm_status.txt"
echo "  jsonl:             $REPO/hsikan_edge_cr_audit/results.jsonl"
echo "  jsonl (ba+otc):    $REPO/hsikan_edge_cr_audit/results_ba_otc.jsonl"
echo "  audit summary:     $SUMMARY"
echo "========================================================"
