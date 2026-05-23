# Benchmark plan: memory close experiment (sanity vs “math doesn’t add up”)

**Goal:** Before you kill big consumers (rust-analyzer, Cursor helpers, ClickHouse, CVAT, etc.), capture **reproducible counters**. After each close step, re-measure. The plan checks whether **(a)** `free`/`MemAvailable` moves as expected and **(b)** **sum of RSS** drops in the same direction (knowing RSS **over-counts** shared pages).

This is **not** a latency benchmark; it is a **memory accounting** experiment.

---

## 0. Preconditions

- Same machine, same login session if possible (or document session change).
- Note **GPU** in use or idle (`nvidia-smi` once) — affects `Shmem`/driver mappings.
- No intentional heavy compile/train during the window (or log if unavoidable).
- Fix **wall clock** at start/end of each phase (UTC).

---

## 1. Canonical snapshot script (run each phase)

Save as `/tmp/mem_snap.sh` (or keep inline):

```bash
#!/usr/bin/env bash
set -euo pipefail
STAMP="${1:?usage: mem_snap.sh <label>}"
OUT_DIR="${MEM_SNAP_DIR:-/tmp/mem_snaps}"
mkdir -p "$OUT_DIR"
F="$OUT_DIR/${STAMP}.txt"
{
  echo "stamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) label=$STAMP"
  echo "--- free -h"
  free -h
  echo "--- meminfo (subset)"
  grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SReclaimable|Shmem|SwapTotal|SwapFree):' /proc/meminfo
  echo "--- sum RSS all PIDs (KiB) -> GiB note: over-counts shared"
  python3 - <<'PY'
import subprocess
out = subprocess.check_output(["ps", "-eo", "rss="], text=True)
kb = sum(int(x) for x in out.split() if x.strip().lstrip("-").isdigit())
print("rss_sum_kib", kb)
print("rss_sum_gib_naive", kb / (1024.0**2))
PY
  echo "--- top 15 by RSS (KiB)"
  ps -eo pid,user,rss,cmd --sort=-rss | head -16
} | tee "$F"
echo "wrote $F"
```

**Primary metrics per phase**

| Metric | Source | Interpretation |
|--------|--------|----------------|
| `MemAvailable` | `/proc/meminfo` | Best single “headroom” number |
| `used` | `free -h` | Coarse; includes non-cache pressure |
| `sum(RSS)` | `ps -eo rss=` | Must move **with** closes; absolute value **≥** unique RAM (double-count) |
| `Shmem` | `/proc/meminfo` | Often jumps with GPU / shared maps |
| Top PIDs | `ps` | Confirms *what* you think you killed is gone |

---

## 2. Experiment phases (sequential — one knob per phase)

**Phase A — Baseline**  
Run `mem_snap.sh baseline`.

**Phase B — rust-analyzer**  
1. Note PID from baseline top (`rust-analyzer`).  
2. Close/disable RA (or kill that PID if you accept Cursor restarting it).  
3. Wait **30 s** (drop caches stabilise).  
4. Run `mem_snap.sh after_rust_analyzer`.

**Expected:** `MemAvailable` ↑ on the order of **~2–4 GiB** if RA was ~3–4 GiB RSS; `sum(RSS)` ↓ similar magnitude (not 1:1).

**Phase C — Cursor helpers (optional)**  
Close extra Cursor windows / reload window. `mem_snap.sh after_cursor_trim`.

**Phase D — ClickHouse (optional)**  
Stop service only if you own it: e.g. `sudo systemctl stop clickhouse-server` **only if** you use it rarely.  
`mem_snap.sh after_clickhouse_stop`.

**Expected:** ~0.5–1+ GiB depending on load.

**Phase E — CVAT stack (optional)**  
Stop supervisord / docker stack **only** via whatever you normally use to run CVAT (document exact command).  
`mem_snap.sh after_cvat_stop`.

**Expected:** Many × ~250–300 MiB workers → **multi-GiB** if all workers die.

**Phase F — Plasma / Chrome (optional)**  
Only if you want a “clean desktop” upper bound — document what you closed.

---

## 3. Pass / fail criteria (for “idiot check” on accounting)

**PASS if:**

1. After each phase, **`MemAvailable` never decreases** vs previous phase (noise < ~200 MiB acceptable once).
2. **`sum(RSS)` decreases** in the same phase when you killed a large PID that appeared in the prior top list.
3. The PID you targeted **disappears** from `ps` top output.

**INCONCLUSIVE (not “idiot”, but needs more data) if:**

- `MemAvailable` up but `sum(RSS)` flat → likely **cache / Shmem** moved; re-check `Cached`/`Shmem`.
- Both flat → wrong PID killed, or process respawned (supervisord).

**FAIL hypothesis (accounting wrong) only if:**

- You kill a **confirmed** multi-GiB process and **`MemAvailable` and `sum(RSS)`** both unchanged after 60 s — then we dig (cgroup, zram, another host view).

---

## 4. Optional sharper tool (if installed)

```bash
sudo smem -tk | head -40
```

`smem` apportions **shared** memory (PSS) so row sums approximate physical usage better than raw RSS sum.

---

## 5. Deliverable

One directory, e.g. `/tmp/mem_snaps/`, with:

- `baseline.txt`, `after_rust_analyzer.txt`, …
- A 10-line **summary table**: label | MemAvailable (KiB) | sum_RSS (KiB) | Shmem (KiB) | note

Paste that table back — then we can say definitively whether the earlier “top N didn’t add to used” issue was **under-counting small PIDs** vs something else.

---

## 6. Safety

- Do **not** `kill -9` random `python3` without checking cmdline (avoid killing system services you need).
- Prefer **your normal stop** for CVAT/ClickHouse so they don’t respawn under supervisord.
