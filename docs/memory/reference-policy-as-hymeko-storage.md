---
name: reference-policy-as-hymeko-storage
description: Trained policies round-trip state_dict ⇄ .hymeko (weights = star expansion incidence); HyMeKo number lexer rejects scientific notation
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3060e292-680f-4645-82c1-156ce78e537c
---

`hymeko_rl/policy_store.py` (2026-06-21) stores a trained torch policy AS a valid HyMeKo hypergraph and reads
it back **bit-exact**. Basis identity: a weight matrix `W (m×n)` is the star expansion of a weighted
hypergraph (m vertices, n hyperedges, incidence `B[i,j]=W[i,j]`). `policy_to_hymeko(state_dict, path)` /
`hymeko_to_policy(path)`; `weight_to_hypergraph`/`hypergraph_to_weight` expose the identity. Proven on the
cart-pole HSiKAN policy: 332 KB valid HyMeKo, max|Δ|=0 over 26 259 weights, eval reproduced.
Report `reports/2026-06-21-policy-weight-storage.md`; 6 figures in `docs/figures/2026-06-21-policy-storage/`.

**Two non-obvious gotchas (cost me two iterations — don't repeat):**
1. **HyMeKo's number lexer REJECTS scientific notation** (`1.5e-05` → Parse error `UnrecognizedToken Ident("e")`).
   Format floats positionally: `np.format_float_positional(np.float32(x), unique=True)` — shortest decimal,
   no `e`, still round-trips float32. (Some reader *regexes* accept `eE`, but the grammar/parser does not.)
2. **`@"file.hymeko"` imports resolve relative to the importing file's dir.** A generated artifact that imports
   `meta_nn.hymeko` only validates if co-located with it. Make stored artifacts **self-contained** (inline the
   minimal vocab) so `hymeko validate` passes anywhere.

**`signedkan` variant (learned incidence = star edges):** `build_policy("signedkan", …)` (policy.py
`learn_incidence` flag) makes `a_pos/a_neg` trainable `nn.Parameter`s (init = kinematic); the trained
incidence IS the weighted star edges, round-trips bit-exact. Drifts only ~0.03 on cart-pole and gives no perf
gain (2-vtx, parity with hsikan) — mechanism demo, not a result. `hsikan` keeps the incidence a buffer.

**Binary storage — ALL THREE TIERS BUILT** (`policy_to_hymeko(tier="t0"|"t1"|"t2"|"auto")`; reader is a
per-block parser dispatching on `data`/`data_b64`/`blob`): **T0** decimal (332 KB), **T1** base64-float32
(142 KB, a *string* so no number-lexer issue), **auto** (148 KB: structural decimal + bulk binary, the
default), **T2** content-addressed `.npz` blob via stdlib `np.savez` (9 KB `.hymeko` + sha256-verified blob;
tamper → ValueError). `safetensors` deliberately NOT adopted (§1 dep). All bit-exact + valid HyMeKo.

**Provenance + save chain:** `policy_to_hymeko(..., meta={algo,backbone,upright,seed})` writes a
`provenance {}` block (valid HyMeKo); `read_provenance(path)` reads it. All trainers have `--save <path>`
(train_inverted_pendulum/ddpg/sac). `load_policy_from_hymeko` dispatches on stored keys → reconstructs PPO
`ActorCritic` (actor_mean) / SAC `SquashedGaussianActor` (mu) / DDPG `DeterministicActor` (head), hsikan-family,
bit-exact (`GreedyPolicy` Protocol unifies eval). Render AUTO-LABELS from provenance (HUD title + filename
`cartpole_<algo>_<backbone>.gif`), gif HUD shows step/return/pole/cart/force/status. The 2 stored cart-pole
policies re-stamped with provenance.

**Sim interface (loop closed):** `hymeko_rl/render_inverted_pendulum.py` —
`load_policy_from_hymeko(path, env)` infers architecture from tensor shapes (hidden/n_layers/feat/action_dim)
and reconstructs; reuses `evaluate.render_episode_gif` (PIL, no imageio; MuJoCo offscreen works headless here)
+ a side camera + trajectory PNG. CLI `python -m hymeko_rl.render_inverted_pendulum --policy <.hymeko>`. So a
stored policy is loaded→run→watched. Dense layers map to a vacuous
complete-bipartite hypergraph — the representation earns its keep on signed/sparse incidence (HSiKAN's `M_e`).
Part of [[project-cartpole-hsikan-testbed]]; the editor's [[reference-editor-hyperedge-on-hyperedge]] List-value
serialization is what lets the editor render these stored weights on the HUD.
