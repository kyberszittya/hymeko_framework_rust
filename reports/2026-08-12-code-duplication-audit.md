# Code-duplication & library-integration audit (2026-08-12)

Scope: how integrable the current code is, and how much duplicated code exists — codebase-wide, and self-critically for
this session's additions. Evidence is `grep`-quantified; parity claims are measured.

## TL;DR

- **This session's *library* additions are already well-integrated** (shape families live in `hymeko_rl/env` +
  `coin_delivery`, use a shared helper, and the R12 harnesses *import* their scaffold rather than copy it).
- **The biggest duplication is pre-existing and in the visualization layer** — the coin-delivery video/demo scripts each
  re-declare the camera + renderer instead of using the canonical `viz/` helpers that already exist.
- **A second, verified duplication: the retrieval policy** — a library `RetrievalDeliveryPolicy` (+ `load_frozen`)
  exists, yet ≥6 call sites reimplement its nearest-lookup byte-for-byte.
- **Self-audit: my `delivery_viewer.py` had 2 of these dups; 1 is fixed in this commit** (now wraps
  `RetrievalDeliveryPolicy`, measured Δθ = 0), 1 (the render loop) is folded into the recommendation below.

## 1. Visualization layer — the largest duplication

| Duplicated unit | Copies | Where |
|---|---|---|
| `def _cam()` (MjvCamera setup) | **8** | `video_coin_r10_structured_option`, `video_coin_zero_home`, `video_r11_6c_composition`, `video_coin_variants`, `video_coin_r8_residual`, `horizon_authority_benchmark`, `coin_kinetic_r2_h1_demo`, `coin_rl_demo` |
| `class _Filmer` (frame-hook renderer) | **3** | `video_coin_r10_structured_option`, `video_coin_zero_home`, `video_r11_6c_composition` (r11_7b_flagship *imports* one) |
| `mujoco.Renderer(...)` render loop | **17 files** | 10 experiments + 2 tests + `viz/render_reach`, `viz/locomotion_render`, `eval/evaluate`, `gui/vehicle_qt`, `gui/delivery_viewer` |

The camera params are **not identical** — there are ~3 presets (`0.9/-55/90` ×4, `1.02/-87/90`, `0.60/90/-68`) — so this
is "the same boilerplate, a few constants" duplication, not literal copies. **Canonical helpers already exist and are
ignored:** `viz/render_reach.py` has `CameraView`, `render_rollout`, `encode`/`_encode_gif`/`_encode_mp4`;
`viz/rollout_overlay.py` has `encode_clip`, `overlay_frames`. The coin-delivery video files fork because those helpers
target `ArmReachEnv`, while the coin films over `rl.inner.model/data`.

**Recommendation (medium effort, low risk):** one `viz/rollout_film.py` —
`class Filmer(cam)` (the frame_hook), `CAMS = {"top_down": …, "angled": …}` presets, and
`render_qpos_seq(model, qseq, cam)`. Migrate the 8 video/demo files + fold my `render_gif` into it. Removes ~3 `_Filmer`
+ ~8 `_cam` + the ad-hoc loops. Risk is low (viz is non-behavioural), but verify **frame parity** on one video before/after
(the renders are deterministic). ~10 files, mechanical.

## 2. Retrieval policy — verified byte-exact duplication

`hymeko_rl/coin_delivery/delivery_bc/retrieval.py` provides `RetrievalDeliveryPolicy.fit(...).predict(x)` and a
`load_frozen(spec)` loader. Yet the nearest-lookup (`argmin(np.linalg.norm(Xs - std.transform(x)))`) is reimplemented in:
`r11_6d_transport_retrieval`, `r11_6d_handoff_audit`, `video_r11_6c_composition`, `coin_zero_home_rrt`,
`r11_7b_physics_selector`, and (before this commit) `gui/delivery_viewer`. **Measured parity:** `RetrievalDeliveryPolicy.
predict` vs the manual nearest gives **max|Δθ| = 0.0** across 8 scenarios — a literal reimplementation.

**Recommendation (low effort, low risk):** replace the manual lookups with `load_frozen(...).predict(...)`. Parity is
guaranteed (Δθ = 0), so no behavioural change. **Done for `delivery_viewer` in this commit.**

## 3. This session's code — integrability

Mostly clean; it extends the library rather than sitting beside it:

- **Shape families** (`Shape.NGON/ELLIPSE/CAPSULE`, the prism generators): live in `env/object_spec.py` +
  `env/planar_grasp_env.py`, additive `elif` branches, and share one `equal_area_regular_ngon_circumradius` helper
  (footprint ≡ geometry). **No duplication introduced** (the N-gon unification *removed* a would-be per-polygon dispatch).
- **R12 ranker/encoding harnesses**: `r12_3b_relative_frames` / `r12_3b_contacts` **import** `_split`, `_train_reg`,
  `_load`, `_handoff_key` from `r12_2b_ranker` — the scaffold is shared, not copied. Good.
- **`delivery_viewer.py`**: thin glue over `reconstruct_capture` + retrieval + `rollout_primitive` + `mujoco.Renderer`.
  Two self-inflicted dups found: (a) **retrieval** — fixed (now `RetrievalDeliveryPolicy`); (b) **render loop** — folded
  into the §1 `viz/rollout_film.py` recommendation (its `render_gif` should call the shared `render_qpos_seq`).

Nothing from this session needs to be *moved into* the library — it is already there (non-core `hymeko_rl/**`). The
integration work is the *pre-existing* viz + retrieval consolidation.

## 4. Broader duplication (flagged in memory / CLAUDE.md §6.5, not re-measured here)

CLAUDE.md and memory already record larger clusters outside this session: the 16 `#[pyfunction]` cycle variants
(`hymeko_py/src/cycles.rs`), 98 `run_*.py` in `signedkan_wip/`, 8 `_train_val_split` reimplementations. Those are their
own refactors; this audit stays on the coin-delivery / viz surface this session touched.

## Priority

1. **Retrieval → `RetrievalDeliveryPolicy`** (low effort, Δθ = 0 parity) — 1 fixed, 5 call sites remain.
2. **`viz/rollout_film.py`** consolidating `_cam` (×8) + `_Filmer` (×3) + render loops — medium effort, verify frame parity.
3. Leave the non-coin clusters (Rust cycles, signedkan) to their own planned refactors.

Neither §1 nor §2 is a same-commit change (each touches ~5–10 files and wants a frame/θ-parity gate) — they are scoped
follow-ups with a 4-format plan. This commit does only the verified, safe self-fix (retrieval in `delivery_viewer`).
