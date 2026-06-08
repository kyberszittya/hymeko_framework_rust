#!/usr/bin/env bash
# Build the hymeko_signedkan.sif image on Komondor's login node.
#
# Run from /scratch/pr_szevis/hajdu/hymeko/hymeko_framework_rust:
#   bash docs/komondor_setup/build_image.sh
#
# ~10-15 min, ~5-7 GB. Logs to build.log. Re-runnable; will overwrite
# any existing hymeko_signedkan.sif.

set -uo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

LOG=build.log
SIF=hymeko_signedkan.sif
DEF=docs/komondor_setup/hymeko_signedkan.def

echo "[build] $(date -Is) starting" | tee "$LOG"
echo "[build] cwd: $(pwd)" | tee -a "$LOG"

# Ensure singularity is on PATH
if ! command -v singularity >/dev/null 2>&1; then
    echo "[build] loading singularity module..." | tee -a "$LOG"
    module load singularity 2>&1 | tee -a "$LOG"
fi
echo "[build] singularity: $(singularity --version)" | tee -a "$LOG"

# Clear any leftover .sif from a failed previous build. Permission-
# denied on overwrite is a known fakeroot/file-ownership gotcha — the
# fakeroot context may write the leftover with root-mapped ownership
# we can't overwrite.
if [ -e "$SIF" ]; then
    echo "[build] removing existing $SIF (size=$(du -h "$SIF" 2>/dev/null | awk '{print $1}'))" | tee -a "$LOG"
    rm -f "$SIF" || {
        echo "[build] WARN: rm failed; try manual: chmod u+w '$SIF' && rm '$SIF'" | tee -a "$LOG"
        ls -la "$SIF" 2>&1 | tee -a "$LOG"
    }
fi

# Build into $TMPDIR first, then mv to final path. Many HPCs have
# fakeroot quirks on shared FS (Lustre/GPFS) at write time; building
# on local node FS sidesteps that, and `mv` doesn't trigger the same
# permission path as `open()` in fakeroot's namespace.
TMP_SIF="${TMPDIR:-/tmp}/${USER}_hymeko_build_$$.sif"
echo "[build] tmp-staging path: $TMP_SIF" | tee -a "$LOG"
mkdir -p "$(dirname "$TMP_SIF")"

# Build (--fakeroot lets unprivileged user run mknod / setuid inside the
# build container; --fix-perms makes the resulting .sif readable post-build).
echo "[build] $(date -Is) singularity build starting" | tee -a "$LOG"
t0=$(date +%s)
singularity build --fakeroot --fix-perms "$TMP_SIF" "$DEF" 2>&1 | tee -a "$LOG"
rc=$?

# If build succeeded, move .sif into place
if [ "$rc" -eq 0 ] && [ -f "$TMP_SIF" ]; then
    echo "[build] moving $TMP_SIF → $SIF" | tee -a "$LOG"
    mv "$TMP_SIF" "$SIF" || {
        echo "[build] ERROR: mv to final path failed. Image is at $TMP_SIF" | tee -a "$LOG"
        rc=1
    }
fi
t1=$(date +%s)
echo "[build] $(date -Is) singularity build rc=$rc wall=$((t1-t0))s" | tee -a "$LOG"

if [ "$rc" -ne 0 ]; then
    echo "[build] FAILED. Check $LOG for the error trace." | tee -a "$LOG"
    exit "$rc"
fi

# Verify
echo "[build] verifying..." | tee -a "$LOG"
ls -lh "$SIF" | tee -a "$LOG"
singularity inspect --labels "$SIF" 2>&1 | tee -a "$LOG"
singularity exec "$SIF" python -c \
    'import torch; print("torch=", torch.__version__, "(cuda runtime requires --nv flag on a GPU node)")' \
    2>&1 | tee -a "$LOG"

echo "[build] DONE. Image: $SIF ($(du -h "$SIF" | awk '{print $1}'))"
