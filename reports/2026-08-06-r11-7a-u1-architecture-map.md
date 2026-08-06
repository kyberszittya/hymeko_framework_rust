# R11.7A — U1 Architecture Map & Duplicate-Ownership Audit

**Date:** 2026-08-06 · **Stage:** U1 (pre-implementation, per the unification directive)
**Goal:** consolidate the divergent manipulation model-construction paths so the exact-zero pipeline, the video/object-
variant tools, and future multi-object experiments are all generated from **one canonical HyMeKo scene/object
specification**. This document is the responsibility map + disposition table; **no code changes yet**.

---

## 0. The one physically-authoritative builder (recon result)

Both existing paths already funnel through a **single** MuJoCo builder:

```
galambos_env.hymeko (scene: zone/spawn/bounds/success/@dsk radius)
   └─ EnvSpec.from_hymeko (env/env_spec.py:61)  ─┐
                                                 ▼
   PlanarGraspEnv.from_hymeko / __init__ (planar_grasp_env.py:681)
        └─ compose_planar_scene(...)  (planar_grasp_env.py:488)   ← THE model builder
             └─ with_arm_coin_collision (:446) + apply_collision_contract (collision_contract.py:24)
                mujoco.MjModel.from_xml_string  (planar_grasp_env.py:798)
```

`_load_frozen()` (`bv_identification_benchmark.py:39`) loads **only dynamics + material JSON** (`V3Stack`, rubber-tip
friction) — **no MjModel, no XML, no baked geometry**. So the "frozen stack" freezes *dynamics/policy/checkpoint*, never
the object geometry. The coin is pinned to `cylinder, r=0.020` by **one literal** at `bv_identification_benchmark.py:49`
(`_make_env`). This is why unification is consolidation, not a rewrite: the canonical builder already exists; the
duplication is in the **object-injection + acquisition layers above it**, and in the **hardcoded contracts** copied into
experiment/consumer code.

---

## 1. Responsibility map (who owns what today)

| Responsibility | Current owner(s) | file:line | Duplicated? |
|---|---|---|---|
| Scene/task geometry (zone, spawn, bounds, success, disk radius) | `galambos_env.hymeko` → `EnvSpec.from_hymeko` | `env/env_spec.py:61` | **canonical (HyMeKo)** — but object spec is radius-only |
| MuJoCo model + MJCF assembly | `compose_planar_scene` | `planar_grasp_env.py:488` | **single** (good) |
| Collision contract (ARM_LEGALITY, contype/conaffinity) | `with_arm_coin_collision` + `apply_collision_contract` | `planar_grasp_env.py:446`, `collision_contract.py:24` | single, applied once |
| Object geom/body/site NAMES | `compose_planar_scene` (always `"disk"`/`"target_zone"`) | `planar_grasp_env.py:517-548` | single, **stable across shapes** |
| Object-spec INJECTION (shape/size) | `reconstruct_handoff` kwargs → `CoinRL4Dof` → env | `coin_late_start.py:86-88` | plumbed but **hardcoded at `_make_env:49`** for R11.6C |
| Object density/friction | `compose_planar_scene` accepts; **dropped** by `make_coin_env`/`neutral_env`/`CoinRL4Dof`/`reconstruct_handoff` | `env_factory.py:41`, `coin_neutral_start.py:114`, `coin_rl_env.py:22`, `coin_late_start.py:79` | **plumbing gap** |
| Exact-zero acquisition (reach→certified straddle→snapshot) | `_rig`/`load_harness`/`acquire_snapshot`/`acquire_certified_straddle`/`_make_env` | `audit.py:39`, `teacher_bank.py:32,39`, `bv:48` | R11.6C-only stack over the shared builder |
| Reach straddle geometry | `CoinStraddleTargets` (hardcoded r=0.02 standoff/angles) | `planar_geometric_approach.py:64` | **coin-radius-hardcoded** |
| Snapshot type + delivery rollout | `CradleSnapshot` → `rollout_primitive` | `contact_velocity.py:272`, `forward_displacement.py:131` | R11.6C canonical delivery |
| Variant reconstruction (video) | `video_coin_variants._reconstruct` → `reconstruct_handoff` → `structured_carry_rollout` | `video_coin_variants.py:148` | **parallel consumer** of the shared builder, different rollout |
| Object geom/body lookup | hardcoded name `"disk"` resolved via `mj_name2id` | `contact_velocity.py:39,77`, `planar_grasp_env.py:823` | stable name, but **name literal repeated** |
| Execution contracts (modes/guards/IC/provenance/energy) | `hymeko_rl/ir/*` (R11.2 IR) | `ir/hybrid_mode.py`, `initial_condition.py`, `provenance.py`, `energy.py` | canonical (HyMeKo IR) |

---

## 2. Disposition table (KEEP / MOVE / ADAPT / DEPRECATE / DELETE)

| Item | Disposition | Rationale / target |
|---|---|---|
| `compose_planar_scene` | **KEEP** (canonical builder) | already the single MJCF/collision generator; extend its `coin_shape`/`disk_radius(_y)`/`density`/`frictionloss` kwargs to a typed `ObjectSpec` |
| `galambos_env.hymeko` + `EnvSpec.from_hymeko` + `meta_env.hymeko` object vocab | **ADAPT** (extend) | make the HyMeKo scene carry the **full** object spec (shape, dims, thickness, mass/density, friction, family, collision class, contact semantics) — today only `radius`. This becomes the single source of truth. |
| `PlanarGraspEnv.from_hymeko` | **KEEP + promote** | the canonical `HyMeKo → model` generator; make it the sole constructor path |
| hardcoded object literal at `_make_env` (`bv:49`) | **DELETE** | replace with the `ObjectSpec` read from the HyMeKo scene, threaded from `_rig` |
| `_rig`/`load_harness`/`acquire_snapshot`/`acquire_certified_straddle`/`_make_env` | **ADAPT** (thread `ObjectSpec`) | additive pass-through; default = current coin ⇒ O0 bit-identical |
| density/friction drop in `make_coin_env`/`neutral_env`/`CoinRL4Dof`/`reconstruct_handoff` | **ADAPT** (plumb 2 kwargs) | mechanical; close the mid-chain gap |
| `CoinStraddleTargets` r=0.02 hardcode | **ADAPT** (parameterize by footprint) | the one real reach change for size/box; drive from `ObjectSpec` |
| `reconstruct_handoff` | **KEEP as model-layer** | it already builds via `compose_planar_scene`; keep as the shared env constructor, remove its role as a *separate* object-injection convention |
| `video_coin_variants._reconstruct` (object injection) | **MOVE/DEPRECATE** | its object-injection responsibility moves to the canonical `build_manipulation_rig(scene, object_spec)`; keep only its *demo-selection* heuristic |
| `structured_carry_rollout` (video rollout) | **KEEP (adapter)** | a legacy rollout for the video-vs-expert artifact; narrow adapter, not a second model builder |
| hardcoded `"disk"`/`"target_zone"` name literals in consumers | **ADAPT** (generated registry) | expose a `body/geom/site` registry from the generator (`object_body`/`object_geom`/`object_site`) so consumers never hardcode; names stay stable, ownership moves to the generator |
| `CradleSnapshot` + `rollout_primitive` | **KEEP** (canonical delivery) | R11.6C's certified delivery; unaffected |
| `hymeko_rl/ir/*` (modes/guards/IC/provenance/energy) | **KEEP** (canonical contract IR) | already the execution-contract source; bind certificate/provenance to the generated scene |

**Net:** one thing to KEEP-and-extend (`compose_planar_scene` + the HyMeKo scene), a handful to ADAPT (thread the spec,
close the plumbing gap, parameterize the straddle, expose a name registry), one hardcoded literal to DELETE, and the
video path's object-injection to MOVE behind the canonical API (keeping only its selection heuristic + rollout adapter).
No third pipeline is created; the duplicated *ownership* of object/collision/reconstruction setup is removed.

---

## 3. The canonical API (target seam)

```python
scene = load_hymeko_scene("galambos_env.hymeko")          # HyMeKo = source of truth
rig   = build_manipulation_rig(scene, object_spec)         # ONE generator; both paths consume it
#   -> model (compose_planar_scene), collision contract, body/geom/site registry,
#      obs/action schema, target zone, mode/guard + certificate/provenance bindings
```
`object_spec` is itself a HyMeKo object declaration (OBJ_O0..O4 = data variants, not Python branches). Consumers
(reach/capture, retrieval delivery, object-variant eval, video, certificate generation) take `rig` and use the
**generated** registry + contracts — never a hardcoded name or shape assumption.

---

## 4. Migration stages (compatibility-preserving)

- **U1 (this doc):** architecture map + audit. Authoritative builder = `compose_planar_scene`; frozen = dynamics/policy,
  not geometry; duplication located.
- **U2:** extend the HyMeKo scene/object schema (`meta_env.hymeko` object vocab + `EnvSpec`/a typed `ObjectSpec`). No
  parallel YAML — the `.hymeko` scene already exists.
- **U3:** one canonical `build_manipulation_rig(scene, object_spec)` generator; route both `reconstruct_capture` and the
  video/variant tools through it.
- **U4 — O0 parity gate (blocking):** the original coin through the generated path must match the frozen R11.6C original
  on: same scenario+seed, initial state, reach/capture verdict, retrieved θ, contact/collision trace (within tol),
  strict-K6, provenance + energy ledger. **No O1–O4 result is valid before O0 parity passes.**
- **U5:** migrate callers; deprecate legacy constructors; delete the hardcoded object literal + duplicated collision/
  object setup; keep only narrow, *validated* adapters for historical-artifact reproduction. No silent parallel builders.
  **[User decision 2026-08-06] Deprecate-then-delete-after-variants:** U5 routes all callers through the generator +
  marks legacy deprecated (warn + duplicate-ownership test), but the `bv:49` literal + dead branches are physically
  removed only *after* U6 OBJ_O1–O4 run green — nothing legacy is deleted while variants are still validated against it.
- **U6:** define OBJ_O1–O4 as HyMeKo object declarations and run R11.7A object variants — with the stage-resolving
  failure taxonomy (`REACH_GEOMETRY_FAILURE` / `CAPTURE_PROPOSAL_TRANSFER_FAILURE` / `CONTACT_RETENTION_FAILURE` /
  `DELIVERY_POLICY_TRANSFER_FAILURE` / `TARGET_ENTRY_FAILURE`) so a shape failure names its stage.

---

## 5. Frozen meaning (explicit)

**Frozen = controller / policy / checkpoint / action-coordinate / scientific baselines.** The MuJoCo object geometry is
**not** frozen — it is regenerated every acquisition under an identical contract, so an object variant under the same
contract is faithful by identity. Any compatibility adapter (e.g. reproducing a historical balltip artifact) must be
explicitly validated, not assumed.

## 6. Prior object results = prior hypotheses (not conclusions)

All prior variant work ran through `reconstruct_handoff → structured_carry_rollout`, **never** the exact-zero R11.6C
`rollout_primitive` pipeline: `O1_EXPERT_CEILING_HOLDS_ACROSS_SIZES`, `O2_STRUCTURALLY_SOLVABLE_DEPLOY_GAP` (expert
contact-retention rises with elongation, deploy flat), O3 triangle physics validated. So *physical* solvability +
size-invariance are established **by the expert**; **deployable** shape/scale generalization through the exact-zero
retrieval pipeline is the open R11.7A question — and the reason the failure taxonomy must separate the stages.

---

## Verdict target

`R11_7A_HYMEKO_GENERATED_MANIPULATION_ARCHITECTURE_UNIFIED` — one canonical HyMeKo scene/object source, one shared
generator, R11.6C O0 parity, video + exact-zero eval on the same generated architecture, OBJ_O1–O4 expressible without
new model-building branches, duplicated object/collision/reconstruction code removed or formally deprecated, no
regression in the existing coin-delivery certificates. **The conclusion is not "object overrides are supported" — it is
that the executable manipulation environment, collision semantics, and certificate bindings are generated from a shared
HyMeKo specification, eliminating the divergent reconstruction architectures.**
