# Reproduce published Bitcoin HSiKAN numbers (dry checklist)

Target figures (from `docs/plans_entropy_learning_2026_05_08.md` and `joint_mix_5seed_2026_05_08.jsonl`):

| Row | Dataset | Mean AUC | Config label on disk |
|-----|---------|----------|----------------------|
| A | `bitcoin_alpha` | **0.9845 ± 0.0028** | `run_label=joint_ba`, tuples `c3,c4,w2,w3` |
| B | `bitcoin_otc` | **0.9801 ± 0.0057** | `run_label=joint_otc`, tuples `c3,c4,w2,w3` |

These are **not** the Phase‑8 `hsikan_mixed_leanest` row (cycles **k=3,k=4 only**). That is a different harness (`run_phase8_bitcoin_5seed.py`).

---

## 0. Preconditions

- Repo root: `hymeko_framework_rust/`
- Python env: `uv sync --group dev --group ml --all-packages` (or equivalent with `torch==2.4.1`, `numpy<2`, code on `PYTHONPATH`).
- From repo root: `export PYTHONPATH=$PWD`
- Optional: `export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (only affects pytest, not these runs).
- GPU: original overnight used GPU; CPU may work but wall time differs. Same **numerical** result is not guaranteed across devices unless you pin what PyTorch documents for determinism.

---

## 1. Reproduce joint mix 5‑seed (rows A and B)

Canonical orchestrator (truncates result file — **copy path** if you want to keep the committed artifact):

- Script: `signedkan_wip/experiments/run_overnight_joint_mix_2026_05_08.sh`
- Driver: `python -m signedkan_wip.src.run_final_cell`
- Fixed CLI per cell: `--hidden 16 --n-epochs 80 --seed <0..4>`
- Env for joint rows:
  - `HSIKAN_MIXED_TUPLES=c3,c4,w2,w3`
  - `HSIKAN_CYCLE_BATCH=4000`

### 1a. One seed smoke (fast sanity)

```bash
cd /path/to/hymeko_framework_rust
export PYTHONPATH=$PWD
LOG=/tmp/repro_joint_smoke.log
env HSIKAN_MIXED_TUPLES=c3,c4,w2,w3 HSIKAN_CYCLE_BATCH=4000 \
  ./.venv/bin/python -m signedkan_wip.src.run_final_cell \
  --dataset bitcoin_alpha --hidden 16 --n-epochs 80 --seed 0 \
  | tee "$LOG"
grep -E '^\{"dataset"' "$LOG" | tail -1
```

Expect a JSON line with `"auc"` ≈ **0.98x** for Alpha seed 0 (see existing `joint_mix_5seed_2026_05_08.jsonl` line 1).

### 1b. Full 5‑seed joint + paired cycle baselines

```bash
bash signedkan_wip/experiments/run_overnight_joint_mix_2026_05_08.sh
```

Artifacts:

- `signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl` (script **clears** this at start)
- Logs: `/tmp/joint_overnight_2026_05_08/`

### 1c. Verify mean ± spread (matches table)

```bash
./.venv/bin/python - <<'PY'
import json, statistics, pathlib
p = pathlib.Path("signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl")
for label, ds in [("joint_ba", "bitcoin_alpha"), ("joint_otc", "bitcoin_otc")]:
    xs = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("run_label") == label and r.get("dataset") == ds:
            xs.append(r["auc"])
    print(label, ds, "n=", len(xs), "mean=", round(statistics.mean(xs), 4),
          "pstdev=", round(statistics.pstdev(xs), 4) if len(xs) > 1 else "n/a", "seeds=", xs)
PY
```

Compare to **0.9845 ± 0.0028** (Alpha) and **0.9801 ± 0.0057** (OTC). Small differences vs the frozen `2026_05_08` file are acceptable if code/data changed; large gaps mean env / commit / data drift.

---

## 2. Reproduce Phase‑8 Bitcoin panel (separate table)

Not the 0.98 joint row — this is the **multi‑arch** strict panel.

```bash
export PYTHONPATH=$PWD
./.venv/bin/python -m signedkan_wip.src.run_phase8_bitcoin_5seed \
  --out signedkan_wip/experiments/results/phase8_bitcoin_5seed_<DATE>.json
```

Defaults: datasets `bitcoin_alpha` `bitcoin_otc`, seeds `0..4`, `n_epochs=200`, `max_k4=30000`, includes MLP/GCN/SignedKAN/**HSiKAN leanest (k=3,k=4)**/SGCN/SiGAT.

Verify: `summary.hsikan_mixed_leanest|bitcoin_*` median AUC ≈ **0.83 / 0.85** (see existing `phase8_bitcoin_5seed.json`).

---

## 3. Optional: strict‑protocol joint rows (`auc≈0.5` in committed file)

Same script Phase B sets `HSIKAN_STRICT_PROTOCOL=1` for `joint_*_strict` labels. Use the shell script Phase B loop or mirror env in manual runs. Purpose: **endpoint / sigma‑leakage** check — not the 0.98 headline.

---

## 4. What to archive with any claim

- `git rev-parse HEAD` and `git status --short`
- `uv export` or `pip freeze` from the venv used
- Full JSONL / JSON written by the run
- `nvidia-smi` snapshot (if GPU) + host OS

---

## 5. File map (read‑only references)

| Claim | Primary artifact |
|-------|------------------|
| 0.9845 / 0.9801 joint | `signedkan_wip/experiments/results/joint_mix_5seed_2026_05_08.jsonl` (`joint_ba`, `joint_otc`) |
| Table caption | `docs/plans_entropy_learning_2026_05_08.md` |
| Orchestrator | `signedkan_wip/experiments/run_overnight_joint_mix_2026_05_08.sh` |
| Phase‑8 leanest | `signedkan_wip/experiments/results/phase8_bitcoin_5seed.json` |
