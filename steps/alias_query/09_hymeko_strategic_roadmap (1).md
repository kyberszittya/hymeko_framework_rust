# HyMeKo Strategic Roadmap — April 2026

## Author: Dr. Csaba Hajdu
## Affiliations: SZTIP (Székesfehérvár), Óbuda University Alba Regia Faculty, Széchenyi István University (Győr)
## Key collaborators: Prof. Dr. Péter Galambos, Dr. habil Károly Széll ("Karesz"), Zoltán Szilágyi, Bálint Farkas
## Last updated: April 1, 2026

---

## 1. Current State of the Framework

### 1.1 Codebase
- **Language**: Rust (5 crates, ~15,400 lines + ~8,000 lines tests)
- **Crates**: parser (SIMD lexer + LALRPOP), hymeko_core (IR, tensors, query engine, transforms), hymeko_daemon (iceoryx2 IPC), hymeko_py (PyO3 bindings), hymeko_client (CLI)
- **Query engine v2**: 1,507 lines, 9 files — QueryMatch/ArcBinding structs, iterator-based API, domain transforms (URDF/SDF)
- **Benchmarks**: 7,500 IPC runs, 300 configs, 100% success, 0 timeouts. Star best 4.3ms, parse 0.55ms for 20-node robot.

### 1.2 Key technical claims (paper-ready)
1. Signed incidence enables native encoding of directional multi-entity relationships
2. Query-as-description unifies data definition and query language in a single grammar
3. Star expansion is 15–20× faster than clique at high density
4. Zero-copy IPC achieves sub-5ms latency for typical robot descriptions
5. 20–35% description conciseness compared to URDF/SDF
6. Deterministic reproducibility via Blake3 Merkle hashing with log-linear scaling
7. Cross-context constraints formalize higher-order dependencies beyond binary graphs

### 1.3 Two parser versions
- **PyLark (v1)**: `<-`/`->` syntax, `<<copy>>`/`<<use>>`/`<<extend>>`, `seq`/`concurrent` edge types. Documented in Acta paper.
- **LALRPOP (v2)**: `+`/`-`/neutral syntax, `: base` inheritance, `@` edge prefix. Used in COINS/SMC papers.
- Both are valid; v2 is the actively developed version.

---

## 2. Publication Strategy

### 2.1 Immediate deadlines (April–May 2026)

| Paper | Venue | Deadline | Content | Status |
|-------|-------|----------|---------|--------|
| IEEE COINS Track 8 | IEEE COINS 2026 (Agentic AI) | Apr 15 | Query-as-description + URDF generation, 6 pages | Ready to write |
| IEEE SMC Regular | IEEE SMC 2026 | Apr 19 | Extended framework: compilation pipeline + tensor expansion + benchmarks, 8 pages | Ready to write |
| IEEE SMC WiP | IEEE SMC 2026 WiP | ~May 3 | Structural entropy OR XR context visualization (Zoltán lead), 4 pages | Planned |
| SISY special session | SISY 2026 | End of May | Cognitive architecture / Global Workspace angle (Karesz organizing) | Topic TBD |
| AD&I journal | Advanced Devices & Instrumentation | End of May | Sensing + tensor pipeline (Galambos contact with guest editor Huijun Gao) | To discuss with team |
| Acta Polytechnica revision | Acta Polytechnica Hungarica (Acta1034) | End of May | HyMeKo Language paper: benchmark section (R2.2), ~10 remaining issues | In progress |

### 2.2 Summer 2026

| Paper | Venue | Deadline | Content |
|-------|-------|----------|---------|
| MDPI Actuators | Actuators special issue | Jun 30 | "Context-Aware Adaptive Robot Control with XR Supervisory Feedback" (Zoltán lead) |
| MDPI Technologies | MDPI Technologies (rolling) | Flexible | 39-page paper, Sections 4–5 done (Szilágyi et al.) |
| CogInfoCom 2026 | CogInfoCom | ~Sep | Global Workspace / Parallel-V cognitive model paper |
| crates.io release | crates.io | Summer | hymeko_core parser crate — open source |

### 2.3 Fall 2026 – Spring 2027

| Paper | Venue | Target |
|-------|-------|--------|
| HyperKAN | NeurIPS 2026 workshop or Neural Networks journal | Fall 2026 |
| IEEE T-SMC:Systems | IEEE Transactions on SMC: Systems | Fall 2026 |
| N-layer motion planning | ICRA 2027 | Jan 2027 submission |
| arXiv preprint | arXiv | After crates.io |
| JOSS | Journal of Open Source Software | After arXiv |

### 2.4 Content separation (avoid self-overlap)

- **COINS**: Query engine + URDF/SDF domain transforms (6 pages, focused)
- **SMC Regular**: Compilation pipeline + tensor expansion + IPC benchmarks (8 pages, different angle)
- **SMC WiP**: Structural entropy or XR context (4 pages, Zoltán)
- **SISY**: Cognitive architecture / Global Workspace (conceptual, Karesz's session)
- **AD&I**: Sensing + control pipeline (Galambos + Huijun Gao)
- **Actuators**: XR supervisory feedback + context-aware control (Zoltán lead)
- **Technologies**: Full multi-context representation (39-page comprehensive paper)
- **CogInfoCom**: Cognitive model formalization (Parallel-V, Baranyi connection)

---

## 3. Research Directions — Rated

### 3.1 Tier 1: Build now (April–June 2026)

| Direction | Novelty | Strategic | Action |
|-----------|---------|-----------|--------|
| **A. Core framework** (query engine, crates.io) | 7/10 | 10/10 | Query engine v2 integration → COINS/SMC papers → crates.io |
| **P. General framework positioning** | 7/10 | 10/10 | arXiv preprint after crates.io, multi-domain examples |
| **M. Structural entropy module** | 7/10 | 7/10 | ~300 lines Rust, feeds SMC WiP paper |

### 3.2 Tier 2: Build summer 2026

| Direction | Novelty | Strategic | Action |
|-----------|---------|-----------|--------|
| **L. HyperKAN** | 9/10 | 9/10 | Signed interaction KAN on hypergraph. NURBS/B-spline activations. No prior art. NeurIPS target |
| **E. Behavior modeling** | 6/10 | 8/10 | State machines + behavior trees as hyperedge declarations. CogInfoCom 2026 |
| **I. Knowledge graph interop** | 5/10 | 7/10 | RDF/OWL import/export, SPARQL-to-Predicate. Positions HyMeKo as KG superset |
| **Graph/Hypergraph convolution** | 8/10 | 9/10 | Three levels: GCN (clique adj), HGNN (incidence B with degree norm), Signed-HGNN (W₊/W₋). Foundation for HyperKAN |

### 3.3 Tier 3: Build fall 2026 – 2027

| Direction | Novelty | Strategic | Action |
|-----------|---------|-----------|--------|
| **O. N-layer motion planning** | 8/10 | 9/10 | Multi-resolution planning with cross-layer constraint hyperedges. ICRA 2027 |
| **B. Runtime monitoring** | 5/10 | 8/10 | Temporal IR extension, streaming query evaluation, cross-context constraint violations |
| **Arrow/DLPack FFI bridge** | — | Critical | Zero-copy path from Rust tensors to PyTorch. Gemini's top recommendation, confirmed correct |
| **D* Lite on CSR** | 6/10 | 7/10 | Incremental replanning for dynamic robot environments. Pairs with N-layer planning |

### 3.4 Tier 4: 2027+

| Direction | Novelty | Strategic | Action |
|-----------|---------|-----------|--------|
| **F. Quantum circuit representation** | 8/10 | 8/10 | Quantum gates as N-adic hyperedges, complex-valued incidence |
| **wGPU compute engine** | — | 7/10 | Cross-platform GPU for tensor ops + message passing |
| **16-bit compositional tensor algebra** | 8/10 | 9/10 | LLM-to-LLM communication protocol (Gemini collab idea). NeurIPS workshop / AAMAS 2027 |
| **Distributed multi-agent** | 7/10 | 8/10 | Agent-scoped .hymeko, cross-agent constraint hyperedges. Architecture already supports it |

---

## 4. Key Design Decisions (Already Made)

### 4.1 Query engine: `Vec<QueryMatch>` with `ArcBinding` (not bare DeclId)
- Each match carries id, name, depth, and captured arc-reference bindings
- Domain transforms read bindings directly — zero re-traversal
- Iterator-based core: `query_iter()` → lazy, `query()` → collect, `query_first()` → early exit

### 4.2 No heap/BTreeMap for query results at current scale
- `Vec<QueryMatch>` with `.sort_by()` beats all alternatives for N < 10,000
- Cache locality dominates at typical robot description sizes (5–200 nodes)
- Revisit only if runtime monitoring hits 100K+ live entities

### 4.3 Distributed architecture: scope by agent prefix, not code changes
- `agent_alpha.robot.base_link` vs `agent_beta.robot.base_link`
- Blake3 Merkle hashing provides content-addressed identity across network
- iceoryx2 pub-sub pattern maps directly to distributed message bus (DDS, ZeroMQ)
- Consensus/conflict resolution is a 2027 research problem, not an April deadline

### 4.4 Graph convolution: design the trait now, implement with HyperKAN
- `HypergraphConv` trait unifying GCN (L1), HGNN (L2), Signed-HGNN (L3)
- Existing `implicit_clique_step` IS hypergraph convolution minus degree normalization and learnable weights
- Signed-HGNN (separate W₊/W₋ for positive/negative incidence) is novel — no prior art

### 4.5 Gemini tactical review — accepted and rejected suggestions

| Suggestion | Verdict | Reason |
|-----------|---------|--------|
| Arrow/DLPack FFI bridge | **Accept** — after query engine | Correct priority, correct engineering |
| E-graph HRE | **Reject** — wrong algorithm | Use DPO rewriting for hypergraph topology, not term equivalence |
| Simplicial Message Passing | **Reject** — overshoot | HyMeKo hypergraphs aren't simplicial complexes. Cell Complex MP is more general |
| D* Lite on CSR | **Accept** — premature now | Correct but target ICRA 2027 |
| "Don't run matrix multiply in Rust" | **Reject** | Sub-5ms IPC requires Rust SIMD. GPU only for batch training |
| ECS pattern | **Reject** — wrong paradigm | Compiler uses indexed arenas, not game engine ECS |
| CSF format | **Reject** — premature | Incidence matrix is 2D. CSF solves a problem HyMeKo doesn't have yet |
| Arena allocation (bumpalo) | **Accept** — later | Good optimization for AST building, not needed for April deadlines |
| Unified `Id<T>` with PhantomData | **Accept** — later | Clean refactor, 30 lines, do after papers ship |

---

## 5. Collaboration Opportunities

### 5.1 Baranyi CPII-Corvinus bilateral project
- **What**: "Enhancing trustworthiness of real-world AI applications" — InnoHK CPII (CUHK Hong Kong) + Corvinus University (Budapest)
- **Led by**: Prof. Péter Baranyi (CogInfoCom founder)
- **Connection**: HyMeKo's deterministic compilation, query-as-description transparency, and formal cross-context constraints align with trustworthy AI goals
- **Action**: Talk to Galambos about connecting via CogInfoCom 2026
- **NOT a competitor** — complementary: HyMeKo is the representation layer, their project is the AI architecture layer

### 5.2 Szilágyi XR project
- "XR-assisted context acquisition and interaction for hypergraph-based adaptive robot control"
- Patent filing flagged as priority before public disclosure
- Architecturally sound; needs MDPI Actuators and Technologies papers for publication

### 5.3 Pannonia scholarship — Nagoya, Japan
- At least 1 month research stay confirmed
- Nagoya has strong robotics community
- Galambos responded: "Congratulations. I'm jealous."

---

## 6. Execution Order

### April 2026
1. ✅ Query engine v2 code complete (1,507 lines)
2. → `cargo build --workspace` on query engine
3. → Write IEEE COINS paper (Apr 15)
4. → Write IEEE SMC Regular paper (Apr 19)

### May 2026
5. → IEEE SMC WiP (structural entropy or XR context)
6. → SISY special session paper (cognitive architecture)
7. → Acta Polytechnica revision (R2.2 benchmarks, remaining issues)
8. → AD&I journal submission (if Galambos confirms)
9. → Structural entropy module (~300 lines Rust)

### June 2026
10. → MDPI Actuators submission (Zoltán lead, Jun 30)
11. → crates.io publication of hymeko_core parser crate
12. → arXiv preprint

### Summer 2026
13. → HypergraphConv trait + Signed-HGNN implementation
14. → HyperKAN sprint (depends on conv module)
15. → Arrow/DLPack FFI bridge
16. → CogInfoCom 2026 paper

### Fall 2026
17. → N-layer motion planning design (ICRA 2027 target)
18. → Runtime monitoring module
19. → IEEE T-SMC:Systems journal submission

### 2027
20. → ICRA 2027 submission
21. → 16-bit tensor algebra protocol (if validated)
22. → Distributed multi-agent architecture
23. → Quantum circuit representation exploration
24. → FTC 2027 (Berlin, with fresh contribution)

---

## 7. Documents Produced in This Session

| # | Document | Lines | Purpose |
|---|----------|-------|---------|
| 1 | hymeko_report_galambos.pdf | 9 pages, 11 tables | Framework overview for Prof. Galambos |
| 2 | hymeko_summary_sister.pdf | 3 pages | Plain-language summary for sister |
| 3 | hymeko_query_engine_v2.zip | 1,507 lines, 9 files | Query engine with QueryMatch/ArcBinding |
| 4 | 08_query_engine_integration_steps.md | 10-step guide | Step-by-step integration instructions |
| 5 | This document | — | Strategic roadmap |

---

## 8. Key URLs and Repos

- Python (PyLark) parser: https://github.com/kyberszittya/himeko_lang
- Rust (LALRPOP) framework: https://github.com/kyberszittya/hymeko_framework_rust
- Acta paper submission ID: Acta1034
- CPII-Corvinus press release: https://www.cpr.cuhk.edu.hk/en/press/innohk-cpii-and-corvinus-university-of-budapest-sign-contract-to-launch-hungarian-funded-bilateral-research-project-enhancing-trustworthiness-of-real-world-ai-applications/

---

*This document serves as a persistent context anchor. Upload it at the start of a new chat session for immediate project continuity.*
