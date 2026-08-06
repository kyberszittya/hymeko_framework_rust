---
name: feedback-experiment-own-folder
description: "RULE (Hajdu 2026-07-02) — every successful experiment goes into its OWN timestamped folder (gifs + policies + results + log), never scattered/overwriting"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 913c706b-9719-45ca-aa85-e9cfbef27d5d
---

Hajdu, 2026-07-02: **"from now on put every successful experiment to a separate folder."**

**Why:** scattering run outputs across shared dirs (`reports/gifs/`, `checkpoints/`) collides and overwrites — on-record 2026-07-02 a difficulty-sweep rendered every cell's GIF to the same `diff0.gif` (the difficulty float's dot truncated the `render_actor_gif` name), so all but the last were lost; only policies survived. A self-contained folder per run is reproducible, loss-proof, and hand-to-Kato-ready.

**How to apply:**
- Each run writes to a fresh **`experiments/<YYYY_MM_DD_HH_MM>_<name>/`** (use `hymeko_rl.evaluate.experiment_dir`, the existing convention `exp_pernode_actor_ab` uses) with subdirs **`gifs/`**, **`policies/`**, and a **`results.json`** + **`README.md`** table + the run **log**. Do NOT write GIFs/checkpoints to shared `reports/`/`checkpoints/` roots.
- **GIF/file names must be dot-free** (encode difficulty as `diff03` not `diff0.3`; `render_actor_gif`/`Path.with_suffix` treat the first dot as the extension → collision).
- The saved **policy `.pt` is the source of truth** (names survive because `torch.save` keeps them) — re-render GIFs from policies if a live render collided.
- Ties [[project-rl-evaluator-simulator-ecosystem]] (evaluator/sim ecosystem): the experiment folder is the unit of a run's artifacts, like the report is the unit of acceptance (§9).
