# Runnable+visualizable cart-pole sim interface, and T2 (content-addressed blob storage)

**Date:** 2026-06-21 · **Author:** Aiko (Claude Code) for Dr. Csaba Hajdu
**Plans:** [sim-interface](../docs/plans/2026-06-21-sim-interface/) ·
[binary-tensor-storage](../docs/plans/2026-06-21-binary-tensor-storage/) (T2 phase)

## 1. Simulation interface — the loop closes
A policy **stored as a HyMeKo hypergraph is loaded back, run, and watched.** `render_inverted_pendulum.py`:
- `load_policy_from_hymeko(path, env)` — `hymeko_to_policy` → `state_dict`; the **architecture is inferred
  from the tensor shapes** (`hidden`, `n_layers`, per-vertex `feat`, `action_dim`), rebuilt via `build_policy`
  and loaded. Works for an `hsikan` *or* `signedkan` artifact (param-vs-buffer is irrelevant for inference);
  a vertex-count mismatch errors clearly.
- `render_run` — reuses the repo's env-agnostic offscreen `render_episode_gif` (PIL, no `imageio`) with a
  side camera (X–Z plane), plus a trajectory PNG (cart_x, pole_angle vs step). CLI:
  `python -m hymeko_rl.render_inverted_pendulum --policy <.hymeko|.pt> --out <dir>`.

**Demo:** a freshly trained policy (200/200 upright) re-stored to
[data/nn/cartpole_hsikan_policy.hymeko](../data/nn/cartpole_hsikan_policy.hymeko), then **loaded from that
file** and rendered: `reports/gifs/cartpole/cartpole.gif` (the cart-pole balancing) +
`cartpole_trajectory.png` (pole within ±0.1 rad, cart within ±0.5 m, all 200 steps). MuJoCo offscreen render
confirmed working headless here.

## 2. T2 — content-addressed external blob storage (binary plan, phase 3)
The third storage tier from the binary plan, completing T0/T1/T2:
- `policy_to_hymeko(sd, path, tier="t2"|"auto", blob_path=...)` — T2 tensors are written to a sibling
  `.npz` (stdlib `numpy.savez`, **no new dependency**); the `.hymeko` holds only the structure + per-tensor
  `blob "…npz"; npz_key "…"; sha256 "…"; dtype; shape`. `auto` routes `>1M`-element tensors to T2.
- `hymeko_to_policy` — a robust **per-block parser** dispatches on the present field (`data` / `data_b64` /
  `blob`); a T2 tensor is loaded from the (cached) npz, resolved relative to the `.hymeko`, and its
  **sha256 verified** — a missing or stale blob raises `ValueError`.

**Verified:** the cart-pole policy as T2 = **9 KB `.hymeko`** (structure + refs only) + a 110 KB `.npz`; valid
HyMeKo, bit-exact, and **tampering is rejected** (corrupt one tensor → sha256 mismatch). T0/T1/auto remain
bit-exact (regression).

| tier | cart-pole size | self-contained | bit-exact | use |
|---|---|---|---|---|
| T0 decimal | 332 KB | ✅ | ✅ | small/structural, auditable |
| T1 base64 | 142 KB | ✅ | ✅ | medium, compact + inline |
| **auto** | 148 KB | ✅ | ✅ | structural decimal + bulk binary (default) |
| **T2 blob** | **9 KB** + 110 KB npz | ⚠ ref (sha256-verified) | ✅ | large nets, content-addressed |

## Files touched
| File | Δ |
|---|---|
| `hymeko_rl/render_inverted_pendulum.py` | new (+130): load-from-hymeko, side camera, gif + trajectory, CLI |
| `hymeko_rl/policy_store.py` | +50 (T2: `_sha256`, npz write, per-block reader, sha256 verify) |
| `hymeko_rl/tests/test_render_inverted_pendulum.py` | new (+50, 3 tests) |
| `hymeko_rl/tests/test_policy_store.py` | +30 (T2 round-trip, small-hymeko+blob, tamper rejection) |
| `data/nn/cartpole_hsikan_policy.hymeko`, `reports/cartpole_hsikan_policy.pt` | regenerated (200/200 policy) |
| `reports/gifs/cartpole/*` | rendered gif + trajectory |

**CORE.YAML / deps:** none (npz is stdlib-numpy; `safetensors` deliberately *not* adopted — would be a §1
decision; T2 defaults to npz).

## Test results
- `test_policy_store.py` **22 passed** (T0/T1/T2/auto round-trip; T2 small-hymeko+blob; tamper rejection;
  validity per tier; codec; identity), `test_render_inverted_pendulum.py` **3 passed** (load reproduces;
  vertex-mismatch rejected; gif+trajectory written — GL-guarded), `test_policy.py` **13 passed**. Total **38**.
- `ruff` clean; `mypy --strict` clean (one scoped `# type: ignore` on `np.savez` — numpy stub mis-types
  `**kwds`; documented inline).

## Provenance + `--save` — the train→store→render chain (added)
The stored `.hymeko` now records **who/how trained it**, so the render self-labels:
- `policy_to_hymeko(..., meta={"algo","backbone","upright","seed"})` writes a `provenance { … }` block (valid
  HyMeKo); `read_provenance(path)` reads it back. Every trainer gained `--save <path>`
  (`train_inverted_pendulum`, `ddpg`, `sac`) → stores the trained policy with its provenance.
- `load_policy_from_hymeko` was **generalized** to dispatch on the stored keys to the right actor — PPO
  `ActorCritic` (`actor_mean`), SAC `SquashedGaussianActor` (`mu`/`log_std`), DDPG/TD3 `DeterministicActor`
  (`head`) — all reconstructed bit-exact (hsikan-family). `eval_balance`/the render now take a `GreedyPolicy`
  Protocol so one path serves all algorithms.
- The render **auto-labels** from the provenance (HUD title + filename `cartpole_<algo>_<backbone>.gif`), no
  `--algo` needed. So: `python -m hymeko_rl.sac --policy hsikan --save data/nn/p.hymeko` then
  `python -m hymeko_rl.render_inverted_pendulum --policy data/nn/p.hymeko` → `cartpole_sac_hsikan.gif`,
  self-titled. Verified end-to-end (validate + bit-exact reload for all three actor types); the two stored
  cart-pole policies were re-stamped with provenance. **+6 tests** (provenance round-trip; per-actor reload).

## §6.5 anti-patterns
None. The sim interface reuses `render_episode_gif`/`run_episode`/`build_policy`/`policy_store` (no
duplication). T2 is the third branch of the existing tier chooser + one per-block reader (not per-format
wrappers). The per-block parser replaced the growing single regex before it became unmaintainable.

## Provenance
Git SHA `292388b` (dirty). torch 2.12.0+cu132, mujoco 3.9.0 (offscreen render OK headless), matplotlib 3.11.
Demo policy: hsikan, 160 vec-iters, seed 0, 200/200 upright.

## Open issues / follow-ups
1. Nicer render scene (ground/rail, lighting) — the current gif is clear but minimal.
2. `safetensors` as an optional T2 backend (mmap, cross-framework) — a §1 dependency decision.
3. Load a T2-stored policy whose blob is moved — the error message already says "missing or stale"; a
   `--blob` override on the loader would let it find a relocated blob.
