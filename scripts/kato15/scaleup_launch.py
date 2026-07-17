"""Humanoid scale-up (the body the Aibo pilot omits): humanoid × {flat,structural} × {bounce 3,8} × seeds × 800k,
compiled, flat-first. Seeds from HYMEKO_SEEDS env (default 0,1,2,3,4) so it can split across kato15/kato14."""
import json, os, resource, time
try:
    resource.setrlimit(resource.RLIMIT_DATA, (16 * 1024**3, 16 * 1024**3))     # §4 cap (systemd-run --user unavailable)
except (ValueError, OSError) as e:
    print("[scaleup] rlimit note:", e, flush=True)
import torch
torch.set_float32_matmul_precision('high')                                     # TF32 on Ada
from hymeko_rl.experiments.exp_sac_walk_campaign import run
seeds = tuple(int(x) for x in os.environ.get("HYMEKO_SEEDS", "0,1,2,3,4").split(","))
t0 = time.time()
res = run(steps=800_000, seeds=seeds, bounce_weights=(3.0, 8.0),
          bodies=("humanoid_walk",), actors=("flat", "structural"), compile=True, compile_mode="max-autotune")
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
print(json.dumps(res, default=float), flush=True)
print(f"[scaleup] DONE seeds={seeds} wall={(time.time() - t0) / 3600:.2f}h peak_rss={peak:.2f}GB", flush=True)
