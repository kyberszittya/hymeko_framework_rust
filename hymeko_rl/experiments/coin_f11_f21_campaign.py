"""F11 vs F21 contact-actor-bank contrast campaign — matched, thread-pinned, multi-seed.

Isolated structural variable: one actor (F11, DIRECT / TASK_ONLY) vs two contact-mode actors (F21, ACTOR_REPOSITION +
ACTOR_TRANSPORT behind HYMeko_CONTACT_MODE / TASK_ONLY). Everything else — env, delivery-v2b reward, strict predicate,
twin-Q SAC, n_step=1, replay, generator, obs/action schema, BC/competence logic, curriculum source checkpoint, the
frozen STAGE-2 corpus + committed 70/15/15 mix, evaluation — is held identical (F21 warm-starts BOTH heads from the F11
curriculum policy, so both arms share source lineage; the only step-0 difference is the architecture).

Runs {F11, F21} × seeds {0..3} × reps {0,1} through :mod:`coin_nstep_exp` (``--actor-head``), then the §11 paired
analysis: F21−F11 on STAGE-2 certified coverage (primary, clearance ≥ +0.030) + retention guards + contact-
establishment (P_bilat) + mechanism-validity (P_clean/P_attr), classifying ACTOR_POSITIVE / ACTOR_CONTACT_POSITIVE /
NO_EFFECT / ACTOR_NEGATIVE / BLOCKED. Reuses the F11/F12 paired-delta primitives (§6.1) and paired_stats bootstrap.
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
from hymeko_rl.experiments.coin_f11_f12_campaign import _best, _pair_deltas  # reuse the matched-pair extractors (§6.1)

_CELLS = {"F11": "pooled", "F21": "contact_bank"}                   # actor_head per cell (critic_mode TASK_ONLY both)
_PAIRS = [(s, r) for s in range(4) for r in range(2)]
_BOOT_SEED = 20_260_721
_PIN = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1"}
_KEYS = ["s2_cov", "s2_loose", "s2_maxclr", "s1_ret", "strong_ret", "r64102", "s1_bilat", "s1_clean", "s1_attr"]


def _run_dir(root: Path, cell: str, s: int, r: int) -> Path:
    return root / f"{cell}_s{s}r{r}"


def _launch_one(root: Path, cell: str, s: int, r: int, steps: int) -> tuple[str, int, float]:
    out = _run_dir(root, cell, s, r)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{cell}_s{s}r{r}"
    cmd = [sys.executable, "-m", "hymeko_rl.experiments.coin_nstep_exp", "--nstep", "1",
           "--actor-head", _CELLS[cell], "--seed", str(s), "--rep", str(r), "--steps", str(steps), "--out", str(out)]
    t0 = time.perf_counter()
    with (out / "run.log").open("w") as log:
        rc = subprocess.run(cmd, env={**os.environ, **_PIN}, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0
    print(f"  [{tag}] done rc={rc} wall={wall:.0f}s", flush=True)
    return tag, rc, wall


def launch(root: Path, steps: int, max_parallel: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(cell, s, r) for cell in _CELLS for (s, r) in _PAIRS]
    print(f"[campaign] F11 vs F21: {len(jobs)} runs ({steps} steps, ≤{max_parallel} parallel, thread-pinned) → {root}",
          flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        results = list(ex.map(lambda j: _launch_one(root, j[0], j[1], j[2], steps), jobs))
    ok = sum(rc == 0 for _, rc, _ in results)
    (root / "manifest.json").write_text(json.dumps(dict(
        steps=steps, max_parallel=max_parallel, wall_s=round(time.perf_counter() - t0, 1), completed=ok,
        total=len(jobs), runs=[dict(tag=t, rc=rc, wall_s=round(w, 1)) for t, rc, w in results]), indent=1))
    print(f"[campaign] {ok}/{len(jobs)} runs ok in {round(time.perf_counter() - t0, 1)}s", flush=True)


def _mode_occupancy(root: Path, s: int, r: int) -> float | None:
    """Mean TRANSPORT occupancy over the F21 run's eval windows (the head-liveness signal, §9)."""
    p = _run_dir(root, "F21", s, r) / "run.json"
    if not p.exists():
        return None
    hist = json.loads(p.read_text()).get("eval_history", [])
    occ = [h["bank_diag"]["mode_occupancy_transport"] for h in hist if h.get("bank_diag")]
    return round(sum(occ) / len(occ), 4) if occ else None


def _classify_actor(agg: dict, per: list[dict]) -> str:
    """§11 taxonomy. Order: transport win → retention/transport loss → contact-establishment win → no effect."""
    any_f21_certified = any(p["t_s2cov"] > 0 for p in per)
    s2_lo = agg["s2_cov"]["boot95"][0]
    degrades = ((agg["s1_ret"]["boot95"][1] < 0 or agg["strong_ret"]["boot95"][1] < 0
                 or agg["s2_loose"]["boot95"][1] < 0) and agg["s2_cov"]["median"] <= 0)
    contact_up = agg["s1_bilat"]["boot95"][0] > 0                   # bilateral-contact ESTABLISHMENT improved
    if any_f21_certified and s2_lo > 0:
        return "ACTOR_POSITIVE"
    if degrades:
        return "ACTOR_NEGATIVE"
    if contact_up:
        return "ACTOR_CONTACT_POSITIVE"
    return "NO_EFFECT"


def analyze(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    per = []
    for s, r in _PAIRS:
        c, t = _best(root, "F11", s, r), _best(root, "F21", s, r)
        if c is None or t is None:
            continue
        per.append(dict(seed=s, rep=r, occ_transport=_mode_occupancy(root, s, r), **_pair_deltas(c, t)))
    if not per:
        out = dict(classification="BLOCKED", reason="no completed F11/F21 pairs", n_pairs=0)
        (root / "f11_f21_comparison.json").write_text(json.dumps(out, indent=1, default=float))
        print("=== CLASSIFICATION: BLOCKED (no completed pairs)")
        return out
    agg = {k: paired_stats([p[k] for p in per], _BOOT_SEED + i) for i, k in enumerate(_KEYS)}
    cls = _classify_actor(agg, per)
    occs = [p["occ_transport"] for p in per if p["occ_transport"] is not None]
    out = dict(classification=cls, n_pairs=len(per), bootstrap_seed=_BOOT_SEED, aggregate=agg, pairs=per,
               any_f21_stage2_certified=any(p["t_s2cov"] > 0 for p in per),
               mean_transport_occupancy=round(sum(occs) / len(occs), 4) if occs else None)
    (root / "f11_f21_comparison.json").write_text(json.dumps(out, indent=1, default=float))
    for p in per:
        print(f"  s{p['seed']}r{p['rep']}: S2cov {p['c_s2cov']}→{p['t_s2cov']}(Δ{p['s2_cov']:+d}) | "
              f"S1ret {p['c_s1']}→{p['t_s1']}(Δ{p['s1_ret']:+d}) | S1 bilat Δ{p['s1_bilat']:+.2f} "
              f"clean Δ{p['s1_clean']:+.2f} | occT={p['occ_transport']}", flush=True)
    print(f"--- {len(per)} pairs, paired deltas (F21 − F11) ---")
    for k in _KEYS:
        a = agg[k]
        print(f"  {k}: mean={a['mean']:+.3g} median={a['median']:+.3g} (+{a['pos']}/0={a['zero']}/-{a['neg']}) "
              f"boot95={a['boot95']}")
    print(f"any F21 STAGE2 certified: {out['any_f21_stage2_certified']} | mean TRANSPORT occupancy: "
          f"{out['mean_transport_occupancy']}")
    print(f"=== CLASSIFICATION: {cls}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="experiments/2026_07_21_coin_f11_f21")
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
