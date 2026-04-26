#!/usr/bin/env bash
# Abort the ph11→ph12 chain watcher set up on 2026-04-26 evening.
#
# The watcher is a bash loop that polls /tmp/thesis_iv_views_ph11.log
# every 60s for the "PH11 finished" line, then auto-launches
# run_overnight_views_ph12.sh. This script kills the watcher *without*
# touching ph11 (which has its own PID and log).
#
# Use cases:
#   - ph11 results come in disappointing → don't burn another 90 min on
#     ph12.
#   - You decided to launch ph12 manually with different parameters.
#   - You want to extend ph12 with extra arms (e.g., add ph13 inline).
#
# After running this, ph11 keeps running to completion. Launch ph12
# manually with: bash run_overnight_views_ph12.sh

set -u

# Match the exact command-line signature of the watcher:
PIDS=$(pgrep -f 'until grep -q "PH11 finished".*run_overnight_views_ph12' || true)

if [ -z "$PIDS" ]; then
    echo "No ph11→ph12 chain watcher found running."
    echo "Either it never started, or ph11 already finished and ph12 is now running."
    echo "If ph12 is in flight and you want to stop it, use:"
    echo "    pkill -f run_overnight_views_ph12.sh"
    exit 0
fi

echo "Found watcher PID(s): $PIDS"
echo "Killing chain watcher (ph11 itself is untouched)…"
# Kill the parent bash and any child sleep.
kill $PIDS 2>/dev/null
sleep 1
# Also clean up any sleeping children that survived parent termination.
pkill -P $PIDS 2>/dev/null || true

echo "Done. ph11 continues; ph12 will NOT auto-launch."
echo "To launch ph12 manually after ph11: bash run_overnight_views_ph12.sh"
