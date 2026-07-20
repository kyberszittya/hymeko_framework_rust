"""F11 vs F12 semantic-critic contrast campaign — matched, thread-pinned, multi-seed.

Runs the 16-cell matrix {F11 = 1 actor × task critic, F12 = 1 actor × task + semantic Q_mechanism critic} × seeds
{0..3} × reps {0,1} through the single-variable driver :mod:`coin_nstep_exp` (n_step=1 fixed; only ``critic_mode``
differs), then does the §12 paired analysis: F12−F11 per matched (seed, rep) pair on the primary endpoint (STAGE-2
certified held-out coverage, clearance ≥ +0.030) with STAGE-1 / strong-state / 64102 retention guards and STAGE-1
mechanism-validity deltas (P_bilat / P_clean / P_attr — the Q_mechanism critic's stated job), and classifies
CRITIC_POSITIVE / CRITIC_MECHANISM_POSITIVE / NO_EFFECT / CRITIC_NEGATIVE / BLOCKED via the canonical bootstrap CI.

Fresh matched F11 controls are (re)run here — previous stochastic F11 numbers are NOT reused (§9).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hymeko_rl.eval.paired_stats import paired_stats

_CELLS = {"F11": "TASK_ONLY", "F12": "TASK_AND_MECHANISM"}
_PAIRS = [(s, r) for s in range(4) for r in range(2)]               # 8 matched pairs
_BOOT_SEED = 20_260_721
_PIN = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def _run_dir(root: Path, cell: str, s: int, r: int) -> Path:
    return root / f"{cell}_s{s}r{r}"


def _launch_one(root: Path, cell: str, s: int, r: int, steps: int) -> tuple[str, int, float]:
    """One thread-pinned subprocess run of the single-variable driver. Returns (tag, returncode, wall_s)."""
    out = _run_dir(root, cell, s, r)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cell}_s{s}r{r}"
    env = {**os.environ, **_PIN}
    cmd = [sys.executable, "-m", "hymeko_rl.experiments.coin_nstep_exp", "--nstep", "1",
           "--critic-mode", _CELLS[cell], "--seed", str(s), "--rep", str(r), "--steps", str(steps), "--out", str(out)]
    t0 = time.perf_counter()
    with (out / "run.log").open("w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0
    print(f"  [{tag}] done rc={rc} wall={wall:.0f}s", flush=True)
    return tag, rc, wall


def launch(root: Path, steps: int, max_parallel: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(cell, s, r) for cell in _CELLS for (s, r) in _PAIRS]
    print(f"[campaign] launching {len(jobs)} runs ({steps} steps, ≤{max_parallel} parallel, thread-pinned) → {root}",
          flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        results = list(ex.map(lambda j: _launch_one(root, j[0], j[1], j[2], steps), jobs))
    ok = sum(rc == 0 for _, rc, _ in results)
    manifest = dict(steps=steps, max_parallel=max_parallel, wall_s=round(time.perf_counter() - t0, 1),
                    completed=ok, total=len(jobs),
                    runs=[dict(tag=t, rc=rc, wall_s=round(w, 1)) for t, rc, w in results])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[campaign] {ok}/{len(jobs)} runs ok in {manifest['wall_s']}s", flush=True)


def _best(root: Path, cell: str, s: int, r: int) -> dict | None:
    p = _run_dir(root, cell, s, r) / "run.json"
    return json.loads(p.read_text()).get("best_metrics") if p.exists() else None


def _mc(m: dict) -> float:
    v = m["stage2"]["max_certified_clearance"]
    return v if v > -9 else 0.0


def _pair_deltas(c: dict, t: dict) -> dict:
    """F12 (t) − F11 (c) on the primary endpoint, retention guards, and STAGE-1 mechanism-validity."""
    return dict(
        s2_cov=t["stage2"]["coverage"] - c["stage2"]["coverage"],                 # primary transport endpoint
        s2_loose=t["stage2"]["loose"] - c["stage2"]["loose"],
        s2_maxclr=round(_mc(t) - _mc(c), 4),
        s1_ret=t["stage1"]["coverage"] - c["stage1"]["coverage"],                 # retention guard
        strong_ret=int(t["strong_strict"]) - int(c["strong_strict"]),
        r64102=int(t["s64102_strict"]) - int(c["s64102_strict"]),
        s1_bilat=round(t["stage1"]["P_bilat"] - c["stage1"]["P_bilat"], 3),       # mechanism-validity (Q_mech's job)
        s1_clean=round(t["stage1"]["P_clean"] - c["stage1"]["P_clean"], 3),
        s1_attr=round(t["stage1"]["P_attr"] - c["stage1"]["P_attr"], 3),
        t_s2cov=t["stage2"]["coverage"], c_s2cov=c["stage2"]["coverage"],
        t_s1=t["stage1"]["coverage"], c_s1=c["stage1"]["coverage"])


def _classify(agg: dict, per: list[dict]) -> str:
    """§13 taxonomy. Order: transport win → retention loss → mechanism-quality win → no effect."""
    any_f12_certified = any(p["t_s2cov"] > 0 for p in per)
    s2_lo = agg["s2_cov"]["boot95"][0]
    retention_degrades = ((agg["s1_ret"]["boot95"][1] < 0 or agg["strong_ret"]["boot95"][1] < 0)
                          and agg["s2_cov"]["median"] <= 0)
    mech_up = agg["s1_clean"]["boot95"][0] > 0 or agg["s1_bilat"]["boot95"][0] > 0
    if any_f12_certified and s2_lo > 0:
        return "CRITIC_POSITIVE"
    if retention_degrades:
        return "CRITIC_NEGATIVE"
    if mech_up:
        return "CRITIC_MECHANISM_POSITIVE"
    return "NO_EFFECT"


def analyze(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    keys = ["s2_cov", "s2_loose", "s2_maxclr", "s1_ret", "strong_ret", "r64102", "s1_bilat", "s1_clean", "s1_attr"]
    per = []
    for s, r in _PAIRS:
        c, t = _best(root, "F11", s, r), _best(root, "F12", s, r)
        if c is None or t is None:
            continue
        per.append(dict(seed=s, rep=r, **_pair_deltas(c, t)))
    if not per:
        out = dict(classification="BLOCKED", reason="no completed F11/F12 pairs", n_pairs=0)
        (root / "f11_f12_comparison.json").write_text(json.dumps(out, indent=1, default=float))
        print("=== CLASSIFICATION: BLOCKED (no completed pairs)")
        return out
    agg = {k: paired_stats([p[k] for p in per], _BOOT_SEED + i) for i, k in enumerate(keys)}
    cls = _classify(agg, per)
    out = dict(classification=cls, n_pairs=len(per), bootstrap_seed=_BOOT_SEED, aggregate=agg, pairs=per,
               any_f12_stage2_certified=any(p["t_s2cov"] > 0 for p in per))
    (root / "f11_f12_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for p in per:
        print(f"  s{p['seed']}r{p['rep']}: S2cov {p['c_s2cov']}→{p['t_s2cov']}(Δ{p['s2_cov']:+d}) | "
              f"S1ret {p['c_s1']}→{p['t_s1']}(Δ{p['s1_ret']:+d}) | "
              f"S1 clean Δ{p['s1_clean']:+.2f} bilat Δ{p['s1_bilat']:+.2f} attr Δ{p['s1_attr']:+.2f}", flush=True)
    print(f"--- {len(per)} pairs, paired deltas (F12 − F11) ---")
    for k in keys:
        a = agg[k]
        print(f"  {k}: mean={a['mean']:+.3g} median={a['median']:+.3g} (+{a['pos']}/0={a['zero']}/-{a['neg']}) "
              f"boot95={a['boot95']}")
    print(f"any F12 STAGE2 certified: {out['any_f12_stage2_certified']}")
    print(f"=== CLASSIFICATION: {cls}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="experiments/2026_07_21_coin_f11_f12")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    root = Path(a.out)
    if a.launch:
        launch(root, a.steps, a.parallel)
    if a.analyze or not a.launch:
        analyze(root)


if __name__ == "__main__":
    main()
