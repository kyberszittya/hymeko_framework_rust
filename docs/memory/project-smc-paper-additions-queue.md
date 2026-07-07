---
name: project-smc-paper-additions-queue
description: "Tomorrow's (2026-06-14) queued work — SMC paper additions"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f0358ca-96dd-4433-8a43-96cdde7f4a00
---

Queued by the user 2026-06-13 to start **2026-06-14**. Three tracks, all in the
**active** checkout `d:/hakiko_ai_ws/03_implementation/hymeko_framework_rust/`
(branch `feature/ac-hsikan`) — NOT the stale sibling copy
`…/03_implementation/hymeko/hymeko_framework_rust/` (May-dated), which only holds
the plan doc.

**1. SMC paper additions** — plan at `…/hymeko/hymeko_framework_rust/SMC_PAPER_ADDITIONS_PLAN.md`.
T-SMC: Systems review items, real artifacts, then wired into
`d:/hakiko_ai_ws/01_analysis/articles/superweapon/smc_02` (14-page hard cap → in-body
additions tiny, bulk to supplement/Zenodo).
- **#5 — DONE 2026-06-14** (Phase 1). Report `reports/2026-06-14-sysml-requirements-trace.md`,
  plan `docs/plans/2026-06-14-sysml-requirements-trace/`. Implementation differed from this
  sketch (better): **template-only, no `emit_sysml.rs` change, no core edit** — the template
  path bypasses the core `ModelKind` enum, so it's `data/paper/{meta_sysml_trace,
  traceability_smc}.hymeko` + `transforms/requirements_{sysml,dot}/{queries.hymeko,template.*}`
  + two template-only `DomainTransform`s registered in `hymeko_formats`. Witness validates;
  artifacts `data/paper/traceability_smc.{sysml,dot}` generated; 3 Rust tests + 209-test
  ecosystem regression pass. **Phase 2 (editor SysML lens)** reuses the SAME templates via
  `include_str!`+`execute_transform` in `CompiledDoc::to_sysml` (no FS in WASM) + `views/sysml.js`
  — NOT started. **Paper wiring DONE 2026-06-14**: smc_02 §V subsection `sec:codegen-trace` +
  Listing `lst:trace` + softened §IX limitation; figure → supplement. Compiles clean; **paper is
  now 15pp (was 14)** — user accepted, to be reclaimed by reviewer item 4 (shorten §VI to 0.5pg).
  Graphviz 15.0.0 installed; `data/paper/traceability_smc.{svg,pdf,png}` rendered. Paper edits are
  **uncommitted** in the smc_02 repo. **Phase 2 (editor SysML lens) DONE 2026-06-14** (report
  `reports/2026-06-14-editor-sysml-lens.md`): `CompiledDoc::to_sysml` in `hymeko_wasm/src/compile.rs`
  (include_str! transforms/sysml/{queries.hymeko,template.sysml} + in-process `execute_transform`, FS-free
  for WASM) + wasm.rs forwarder + editor.js/index.html export-menu wiring; native test green (package +
  part def + determinism); **WASM REBUILT** (`wasm-pack build --target web --release --out-dir
  ../docs/editor/pkg`, toolchain present) so to_sysml binding ships in docs/editor/pkg. The browser
  editor already compiled live + exported URDF/SDF/DOT; SysML was one method away. **LIVE LENS ALSO DONE
  2026-06-14:** `docs/editor/views/sysml.js` (createSysmlView + `highlightSysml` pure highlighter, 4
  node --test cases green) + editor.js VIEWS entry + index.html SysML tab/pane + editor.css (.sysml-lens
  reusing hl palette) + `?view=` deep-link. **Browser-verified via Chrome headless screenshot**
  (`docs/editor/sysml_lens_screenshot.png`): SysML tab renders `package robot { part def Link {
  attribute mass:Real } ... }` live, syntax-highlighted, beside the source — confirmed working, good-looking.
  **#6 ablation WIRED INTO smc_02 2026-06-14**: §VII "Honest attribution" — the "ablation sketched as
  future work" replaced with the MEASURED result (C1=0.84ms hypergraph-inproc / C2=0.009ms binary-mock /
  C3≈464ms subprocess; architectural ≈500× = essentially all advantage; representational sub-ms; C2
  mock-lower-bound + C3-consistent-with-head-to-head caveats kept); §IX conclusion updated (coarse split
  now measured, finer per-axis hypergraph ablation remains future). Compiles clean, **still 15pp** (over
  14 cap). Paper edits UNCOMMITTED in smc_02. Remaining SMC: the §VI cut to reclaim the page (not done).
- **#6 — DONE (code+measure) 2026-06-14.** Plan `docs/plans/2026-06-14-smc-inproc-vs-subprocess-ablation/`
  (4 fmts compile); report `reports/2026-06-14-smc-inproc-vs-subprocess-ablation.md`. Built:
  `hymeko_formats/src/binary_graph.rs` (`BinaryGraph`/`BinaryEdge` + clique-expand reusing the
  `binary_vs_hypergraph.rs` arithmetic + mock emitter, 6 unit tests) + `hymeko_bench/src/bin/
  bench_ablation.rs` (C1 real `generate_description`, C2 `mock_emit`, C3 analytic; ≥5-iter
  median/IQR/worst, **NO criterion** — adding it is a §1 dep-add, so used the crate's Instant-bin
  idiom meeting §3). 10 formats tests pass, clippy+fmt(edition2024) clean. NOT a core edit
  (CORE.YAML lists neither crate). **KEY MEASURED FINDING (overturns the #6 sketch's assumption):**
  the gain is **architectural, not representational** — in-process emit C1=0.84ms / C2=0.009ms vs
  subprocess C3=464ms analytic = **~500×**; representational term C1−C2 is +0.83ms i.e. WRONG SIGN
  (the real hypergraph URDF render is slower than the trivial mock, because the mock emits no real
  artifact — it's a lower bound). So the paper must lead with the ARCHITECTURAL ~500× only; a fair
  C1−C2 needs a REAL binary-graph URDF emitter (follow-up). C3 still analytic (xacro/gz/mujoco
  absent on this box). Artifact `hymeko_bench/results/ablation.csv`. Paper wiring into smc_02 not
  yet done.
- **Tool gaps verified absent on this machine 2026-06-13:** `xacro`/`gz`/`mujoco`
  (C3 falls back to analytic ~464 ms with explicit note, per plan guard) and `dot`/graphviz
  (#5 `.dot` generates, but `dot -Tpdf` figure must render elsewhere). Paper wiring deferred
  to a machine with ROS/Gazebo/MuJoCo + graphviz.

**2. Kato seminar demos** — see [[project-seminar-demos-and-hymeyolo-plan]]. Remaining:
`mesh` (only new algorithm — Sinkhorn matcher + signed-vs-unsigned chiral ablation;
discovery-grep `sinkhorn|optimal.transport` FIRST) and `bridge` (`.hymeko`→IR→perception
closer). Package `signedkan_wip/demos/seminar/` on the shared `DemoRunner` harness.

**3. Hymeko web editor** — scaffolding in `demo_web/` (commit `5d243c8 Added editor`),
plus `docs/editor`, `docs/plans/06_wasm_editor`, `docs/plans/2026-06-12-editor-stereotype-views`.
Active thread = the WASM editor (`docs/editor`): SysML lens done (#5 Phase 2 above);
**hypergraph example gallery DONE 2026-06-15** (report
`reports/2026-06-15-editor-hypergraph-examples.md`, plan
`docs/plans/2026-06-15-editor-hypergraph-examples/`): data-driven catalog
`docs/editor/views/examples.js` + `examples.test.mjs` surfacing the EXISTING
`data/typical_graphs/*.hymeko` fixtures (Fano S(2,3,7), sunflower Δ-system, K₄³,
generic) as an `Examples` dropdown that jumps to the Hypergraph 3D view; embed≡fixture
JS test + `hymeko_wasm` compile-smoke (6 tests green); **no WASM rebuild needed** (compile
path unchanged), no CORE edit. Consolidated the old inline `EXAMPLE`/`EXAMPLE_TRACE`
consts into the catalog. **Then "Generate" TAB DONE 2026-06-15** (report
`reports/2026-06-15-editor-hypergraph-generators.md`, plan
`docs/plans/2026-06-15-editor-hypergraph-generators/`): parametric on-demand generators
— `views/generators.js` (pure: Strategy `GENERATORS` registry + `specToHymeko`; STS S(2,3,n)
via Bose for n≡3 mod6 + MRV-backtracking for n≡1, offered n∈{7,9,13,15,19,21,25}; sunflower
k/core/petal; complete Kₙ⁽ʳ⁾) + `views/generator_view.js` (data-driven form + progress bar)
+ editor.js VIEWS `generate` entry w/ `loadGenerated` callback (→recompile→hyper3d) + tab/pane
(v=21). 48 JS tests + 7 Rust (incl generated-source compile-smoke); browser-verified screenshot.
NOTE: naive backtracker was 639ms@STS(19) → MRV fixed. **Still no WASM rebuild / no CORE.**
**Imports + PROFILES DONE 2026-06-15** (user-chosen build; report
`reports/2026-06-15-editor-profiles-imports.md`, plan `docs/plans/2026-06-15-editor-profiles-imports/`):
Rust `compile_sources(root,&[(name,src)])` in hymeko_wasm/src/compile.rs + WASM binding
`parse_and_compile_files(root, files_json)` in wasm.rs (serde_json already dep, NO new dep, NO core
edit — only consumes hymeko_core's public ModuleStore/MemProvider). **WASM REBUILT** into docs/editor/pkg
(wasm-pack 0.15). Editor now keeps a compile "space" (root + aux meta files); `recompile` feature-detects
the multi-file binding (falls back single-file). Data-driven `views/profiles.js` PROFILES registry
(kinematics single-file / HRI imports meta_hri / SysML-trace imports meta_sysml_trace) — profile
`<select>` loads meta files into space + root template + rebuilds palette; old addLink/addJoint →
generic addNodeDecl/addEdgeDecl. New canonical files `data/profiles/{meta_hri,meta_sysml_trace,hri_cell,
sysml_cell}.hymeko` (Rust-compiled + JS embed≡fixture tested). `?profile=` deep-link. 11 Rust + 54 JS
tests green; browser-verified HRI profile (24 nodes, include resolved). The `@"..."`+`using X as y;`
include mechanism works in the multi-file space.
**Arc-value editing DONE 2026-06-15** (report `reports/2026-06-15-editor-arc-values.md`, plan
`docs/plans/2026-06-15-editor-arc-values/`): selecting an edge now shows an inline ARC-REFS editor
(per ref: sign +/-/~, target select, **value** input e.g. joint origin `[[0,0,0.2],[0,0,0]]`) + add/remove
+ Apply; arc-line click opens its edge. Pure `views/arcs.js` (splitTopLevel bracket-aware / parseArcRef /
parseArcTuple / rewriteArcTuple) + arcs.test.mjs (9 tests); string-mutation on edge body (source-as-truth),
NO WASM/snapshot change. `?select=<name>` deep-link added. 63 JS tests green; browser-verified
(?select=spin_joint shows the editable origin value). 
**2D composite layout + 3D pan DONE 2026-06-15** (report `reports/2026-06-15-editor-layout-pan.md`,
plan `docs/plans/2026-06-15-editor-layout-pan/`): View panel "Graph layout" select Force(cose) vs
**Composite (roots centred)** = Cytoscape concentric keyed by scope depth (pure `adapters.scopeDepths`
→ clevel=maxDepth-depth, roots central); 3D hypergraph view gained translate-pan (right/shift-drag moves
look-at `target`; orbit=plain drag, zoom=wheel unchanged). 65 JS tests (scopeDepths +2); browser-verified
both (?layout=concentric, ?view=hyper3d). No Rust/WASM/deps/CORE.
**Editor cache now v=24; deep-links: ?profile= ?view= ?select= ?layout=.**
**Hero demo Phases 1/2/3 + editor robot-arm profile + Phase 3 (Gömb parity + minimal Soma) ALL DONE
2026-06-15** — see [[project-hero-demo]]. **HRI profile FIX DONE** (relations → signed hyperedges;
report `reports/2026-06-15-editor-hri-hyperedges.md`; editor v=26; validates ✅, profiles.test 6/6).
**SEMINAR PPT ADDITIONS DONE 2026-06-15** (report `reports/2026-06-15-seminar-deck-additions.md`):
deck = `docs/seminar/HyMeKo_Seminar.pptx` (34 slides, .pptx). Added star/clique heatmap backgrounds to
slide 12 ("Star vs clique") + a References slide + a Future-work slide → wrote NEW `…with_refs.pptx`
(36 slides; original untouched). `docs/seminar/make_tensor_heatmaps.py` (matplotlib heatmaps) +
`insert_into_deck.py` (python-pptx). **INSTALLED python-pptx==1.0.2 (+lxml,xlsxwriter) — USER-APPROVED
`APPROVED-CORE-EDIT: pptx-seminar-tooling` ("why not install"); into uv venv, NOT added to pyproject.**
Verified structurally (zipfile: 36 slides, slide12 has 2 media refs, refs=slide35/future=slide36); VISUAL
NOT rendered (no LibreOffice/PowerPoint headless — user must open to eyeball faded-bg opacity). Slide
content also in `docs/seminar/REFERENCES_AND_FUTURE_WORK.md`. **2D BIDIRECTIONAL RELATIONS DONE**
(report `reports/2026-06-15-editor-bidirectional-2d.md`): pure `adapters.bidirectionalEdgeIds` (reciprocal
X→Y & Y→X) + renderGraph `bidir` class + cytoscape style (both-end arrows, accent, bowed); 66 JS tests;
editor v=27. Visual not headless-screenshotted (editor has no load-by-URL). 3D pan still hyper3d-only.

**Why:** user explicitly deferred all three to 2026-06-14; no work touched 2026-06-13.
**How to apply:** SMC #5 → #6(C1/C2) per plan §Sequencing; write the plan-protocol docs
(CLAUDE.md §2) before code since these are non-trivial multi-file changes.

- **Refs audited+fixed 2026-06-14** (smc_02 inline thebibliography, no .bib): all 30 bibitems checked; 2 incorrect fixed (web-verified): `ujhelyi2015emfincquery` G.->Z. + full author list (EMF-IncQuery SoCP 2015); `wei2022hypergcl` wrong authors->T.Wei/You/Chen/Shen/He/Wang + full title (NeurIPS 2022 HyperGCL). Cosmetic: `rath2008live` key says 2008 but entry correctly 2012 (left; renaming needs cite-site updates). Recompiles 15pp clean. UNCOMMITTED.

- **§VI/page cut DONE 2026-06-14 → paper now 14pp (was 15, within cap).** Levers: tightened §VI editor catalog→pointer + removed redundant workspace sentence in §V + trimmed lst:trace + compressed #6 ablation prose; DECISIVE: **moved Appendix A (alias-invariance proof, Prop.1) to supplementary** (per IEEE rule appendices go before refs+supp material is uncounted). A1 .tex retained on disk (sections/A1_proof_alias_invariance.tex) for the supplement doc; 4 cross-refs reworded to "supplementary material" (A2, A3x2, §IX conclusion range now app:proof_content--app:proof_storage). Compiles clean, 0 undefined refs. UNCOMMITTED. NOTE: author must add A1 to the actual supplementary material document.

- **ChatGPT review_2 ADDRESSED 2026-06-15** (smc_02/review_2, Hungarian). All 5 points done, paper stays 14pp clean: #1 abstract repositioned correctness-first ("invariants by construction"; speed demoted to "consequence of architecture, not the central claim"); #2 "any instantiation"->qualified + "proof sketches backed by implementation, not machine-checked" (intro+abstract); #3 rho was NON-ISSUE (source already rho>=1 everywhere; reviewer saw old draft/conflated w/ 0.75ms latency or 0.70 scaling exp) + added guard line in eval (rho>=1, overhead=rho-1=O(log n/dbar), distinct from b-hat<1 / latency); #4 NEW measurable non-robotics eval in sec:codegen-trace (traceability witness: 4 req, 4 blocks, 7 signed trace edges [5 satisfies/1 derives/1 allocated_to]; cross-view count-identity by construction via prop:commute; coverage query 3/4 directly satisfied, R2a covered transitively); #5 softened SOTA tables (fig_sota_comparison: humility note re axes/other frameworks lead elsewhere, "Speed-up vs HyMeKo"->"Relative latency", "4 proved"->"4 (sketches)"). Reviewer positioning adopted in spirit: robotics = empirical witness not domain boundary. Paper UNCOMMITTED in smc_02.

- **ChatGPT review_1 (FOLLOW-UP, reviews the post-review_2 version) ADDRESSED 2026-06-15.** Confirms review_2 fixes landed (abstract reposition, proof-sketch framing, non-robotics witness, VII-E rho fix all praised; "submission-close, surgical not restructure, do not over-polish"). Surgical fixes applied: (CRITICAL) Appendix C (A4 storage) self-contradiction "rho<0.7" vs "rho=2.00" -> reframed as overhead rho-1 falling 1.00->0.01 (rho 2.00->1.01), removed wrong <0.7; "hash makes structural equality decidable" -> "collision-resistant content hash for practical structural-equality checks" (01_intro); Table V "Format cost [LoC/fmt]" -> "Approx. impl. surface" + order-of-magnitude/not-controlled caption note; "are proved once" -> "established once". ALSO Fig 7(b) fig_benchmarks semilogx->loglog (y log, ymin1000/ymax30000; panel a already loglog, c stacked-linear left as-is since log-stacking misleading). 14pp clean. Heeding "do not over-polish" -> STOP SMC edits. UNCOMMITTED.
