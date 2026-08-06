"""Aibo reward A/B (2026-07-17): Hamiltonian-momenta + Lyapunov reward vs the CIP-bounce reward, flat SAC from
scratch, seeds paired (cip,ham per seed so a comparison emerges early). Metric = dx + CIP propel/bounce edges of
the LEARNED policy (reward-agnostic → fair A/B). Reuses run_cell; compiled (max-autotune)."""
import json, os, resource, time
try:
    resource.setrlimit(resource.RLIMIT_DATA, (16 * 1024**3, 16 * 1024**3))
except (ValueError, OSError) as e:
    print("[ham-ab] rlimit note:", e, flush=True)
import torch
torch.set_float32_matmul_precision('high')
from hymeko_rl.experiments.exp_sac_walk_campaign import run_cell
from hymeko_rl.experiments.exp_aibo_cip_walk import CipAiboEnv

H = 300
OUT = os.environ.get("HYMEKO_OUT", "experiments/2026_07_17_aibo_hamiltonian_ab")
STEPS = int(os.environ.get("HYMEKO_STEPS", "1200000"))       # ≥1M per the mined "train longer" lever
SEEDS = tuple(int(x) for x in os.environ.get("HYMEKO_SEEDS", "0,1,2").split(","))
REWARDS = {
    "cip_bounce":  lambda: CipAiboEnv(goal_distance=4.0, max_steps=H, bounce=3.0),
    "hamiltonian": lambda: CipAiboEnv(goal_distance=4.0, max_steps=H, hamiltonian=True, target_speed=0.6),
}
os.makedirs(OUT, exist_ok=True)
journal = os.path.join(OUT, "cells.jsonl")
done = set()
if os.path.exists(journal):
    for line in open(journal):
        try:
            r = json.loads(line)
            if "dx" in r:
                done.add((r["reward"], r["seed"]))
        except Exception:
            pass
cells = [(rk, s) for s in SEEDS for rk in ("cip_bounce", "hamiltonian")]   # paired per seed
print(f"[ham-ab] {len(cells)} cells, {len(done)} done, steps={STEPS} seeds={SEEDS} -> {OUT}", flush=True)
t0 = time.time()
with open(journal, "a") as fh:
    for rk, s in cells:
        if (rk, s) in done:
            continue
        try:
            rec = {"reward": rk, **run_cell(REWARDS[rk], "flat", s, steps=STEPS,
                                            compile=True, compile_mode="max-autotune")}
        except Exception as exc:
            rec = {"reward": rk, "actor": "flat", "seed": s, "error": f"{type(exc).__name__}: {exc}"}
        fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
        print(f"[ham-ab] {rk}/s{s} dx={rec.get('dx','ERR')} propel={rec.get('propel_edge','')} "
              f"bounce={rec.get('bounce_edge','')} | {(time.time()-t0)/60:.0f}m", flush=True)
print(f"[ham-ab] DONE peak_rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.2f}GB", flush=True)
