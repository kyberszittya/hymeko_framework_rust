# galambos delivery sweep — 2026-07-02 03:07

SA-HSiKAN actor + TD3+BC + critic LayerNorm + vectorized off-policy (n_envs=8), 1e5 steps, 100-episode eval.
Delivery = coin held in the target zone (DwellMetric). GIF = an episode where the arms deliver.

- `gifs/`   — delivering-episode GIFs, one per (difficulty, seed). (Re-rendered from `policies/` at consolidation;
              the live sweep's GIF names collided on the difficulty's dot — policies are the source of truth.)
- `policies/` — trained actor state_dicts, `diff<d>_s<seed>.pt`.
- `sweep.log` — full live log (per-cell BC / refine delivery + steps/s).
- results table appended below on completion.

## Results (filled on completion)
| difficulty | BC (clone) | refine (median) | delivering GIF |
|---|---|---|---|
| 0.0 | 0.10 | **0.42** (s0; full 3-seed pending) | gifs/diff0.0_s0_deliver_0.42.gif |
| 0.15 | pending | pending | pending |
| 0.3 | pending | pending | pending |
