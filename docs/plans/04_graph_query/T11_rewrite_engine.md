# T11 — Query-Driven Rewrite Engine

**Status:** ✅ DONE (integrated 2026-04-16)
**Files:**
- `hymeko_query/src/rewrite/mod.rs` (24 lines)
- `hymeko_query/src/rewrite/match_context.rs` (219 lines)
- `hymeko_query/src/rewrite/template.rs` (~310 lines)
- `hymeko_query/src/interpret.rs` — added `interpret_transform_queries`, `*_opts` variants
- `hymeko_cli/src/main.rs` — `Transform` subcommand + `tf`/`tdir` REPL
- `transforms/{urdf,sdf,dot,mjcf,ros2_launch}/` — external specs

## 1. Motivation

Previously each output format (URDF, SDF, MJCF, DOT) was a hardcoded Rust
module under `hymeko_query::formats` / `hymeko_query::codegen`. Adding a
new format meant writing a new file, recompiling `hymeko_query`, and
shipping the binary. The engine accumulated domain-specific logic.

The rewrite engine generalises this: a transform is now two plain files
on disk — a query file and a template file. The engine is domain-neutral.

```
transforms/<name>/
├── queries.hymeko       "what to find" — HyMeKo grammar
└── template.<ext>       "what to write" — target syntax + {{tags}}
```

## 2. Pipeline

```
┌─────────────────┐
│ robot.hymeko    │          hypergraph description
└────────┬────────┘
         │ ModuleStore::compile()
         ▼
┌─────────────────┐
│ Compiled IR     │          DeclNodes · Edges · Arcs · Interner
└────────┬────────┘
         │
         │       ┌──────────────────────────┐
         │       │ transforms/<name>/       │
         │       │   queries.hymeko         │
         │       │   template.<ext>         │
         │       └────────────┬─────────────┘
         │                    │
         ▼                    ▼
  ╔═══════════════════════════════════════╗
  ║  rewrite::template::execute_transform ║
  ║                                       ║
  ║  1. parser::parse_description(queries)║
  ║  2. interpret_transform_queries(ast)  ║  unwraps `context { … }`,
  ║                                       ║   leading names = labels only
  ║  3. QueryEngine::query_batch(…)       ║  Predicate → Vec<QueryMatch>
  ║  4. parse_template(template_source)   ║  Vec<Block>
  ║  5. render(blocks, RenderContext)     ║  MatchContext pulls fields
  ║                                       ║   & arc bindings from IR
  ╚═══════════════════════════════════════╝
         │
         ▼
┌─────────────────┐
│ output string   │         .urdf / .sdf / .xml / .dot / .launch.py / …
└─────────────────┘
```

## 3. Three-Layer Separation

| Layer | Module | Scope |
|-------|--------|-------|
| 1 — Query engine | `engine`, `predicate`, `interpret` | IR traversal, predicate evaluation |
| 2 — Rewrite engine | `rewrite::template`, `rewrite::match_context` | Template parsing/rendering, field extraction |
| 3 — Transform specs | `transforms/<name>/` (external) | Domain conventions (URDF/SDF/MJCF/DOT/…) |

Layer 1 knows about hypergraphs.
Layer 2 knows about matches and `{{tags}}`.
Layer 3 knows about specific file formats.
Dependencies only point downward — no domain leakage into the engine.

## 4. Data Flow

### 4.1 Query interpretation (transform mode)

`interpret_transform_queries(&Description)`:

1. Walk `ast.items`; if a top-level `HyperItem::Node` is named `context`,
   descend into its body.
2. Call `interpret_items_as_queries_opts(items, use_name_as_filter = false)`.
3. For each `links: link {}` — emit `NamedQuery { label = "links",
   predicate = Kind(Node) ∧ InheritsFrom("link") }`. The leading name is
   the **label only**, not a `Named()` filter (that is the key behavioural
   difference from the standalone `interpret_as_queries`).
4. For each `@fixed_joints: fixed_joint {}` — emit edge predicate labelled
   `fixed_joints`.

### 4.2 Matching

`QueryEngine::query_batch(&queries)` → `Vec<(String, Vec<QueryMatch>)>`.
Each `QueryMatch` carries: `DeclId`, resolved `name`, `kind`, `depth`,
`arc_bindings: Vec<ArcBinding { sign: i8, target_name: String }>`.

### 4.3 Template rendering

`parse_template(&str) -> Vec<Block>`:

| Block | Source | Meaning |
|-------|--------|---------|
| `Literal(s)` | raw text | emit verbatim |
| `Interpolate(expr)` | `{{expr}}` | resolve & emit |
| `Each { label, body }` | `{{#each L}}…{{/each}}` | iterate matches |
| `If { field, body }` | `{{#if expr}}…{{/if}}` | emit when expr non-empty |

Expression resolution (`resolve_expr`):

| Pattern | Source | Behaviour |
|---------|--------|-----------|
| `config:K` | anywhere | look up `config[K]` |
| `name` | inside `Each` | current match's name |
| `kind` | inside `Each` | `"node"` / `"edge"` / `"arc"` |
| `depth`, `id` | inside `Each` | scalar from match |
| `field:a.b.c` | inside `Each` | `MatchContext::get_field` — walk children, follow refs |
| `bind:<sign>:<idx>\|all` | inside `Each` | Nth arc binding target, or space-joined |
| _fallback_ | inside `Each` | treat as bare field path |

### 4.4 Field extraction

`MatchContext::get_field(path)` walks the IR from the matched declaration:

1. Split on `.`.
2. For each segment, search `decl_children` for a child with that name.
3. On leaf, return `FieldValue` derived from the declaration's
   annotation value, or — if the node has exactly one child with a
   scalar — that child's value.
4. If a direct child isn't found, follow edge references:
   any child edge whose base targets a decl named `<segment>` is
   followed transparently. This makes `field:color` work for
   `color -> link_color` indirection.

## 5. Surfaces

### CLI (one-shot)

```
hymeko transform robot.hymeko -t urdf -o robot.urdf --name my_robot
hymeko transform robot.hymeko -t dot --transforms-dir my_transforms/
```

### REPL

```
hymeko [robot]> tf urdf robot.urdf
hymeko [robot]> tf dot
hymeko [robot]> tdir ./my_transforms
```

### Programmatic

```rust
use hymeko_query::rewrite::{execute_transform, TransformSpec};
let spec = TransformSpec {
    name: "urdf".into(),
    query_source: fs::read_to_string("transforms/urdf/queries.hymeko")?,
    template_source: fs::read_to_string("transforms/urdf/template.urdf.xml")?,
};
let mut config = HashMap::new();
config.insert("robot_name".into(), "my_robot".into());
let urdf = execute_transform(&compiled.ir, &ms.it, &spec, &config)?;
```

## 6. Integration Decisions

| Decision | Rationale |
|----------|-----------|
| `interpret_transform_queries` as a separate entry point | Keep existing `interpret_as_queries` semantics (name = filter) unchanged for regular query files; only transforms get label-only behaviour. |
| `context { … }` wrapper convention | Lets the transform query file keep a valid HyMeKo description header (`<name>_transform {}`) while grouping the actual queries. Parser requires a description block. |
| Joint queries require `@` prefix | Joints are edges in HyMeKo. Without `@`, `fixed_joints: fixed_joint {}` parses as a node pattern — zero matches. Shipped query files fixed accordingly. |
| `{e:?}` over `{e}` for parse errors | `parser::ParseError<_, Token, LexError>` only implements `Debug`. Matches existing CLI convention. |
| Feature-gated on `interpret` | Rewrite needs AST→predicate compilation, already gated behind `interpret` feature. Daemon/IPC builds that skip `interpret` also skip rewrite. |

## 7. Current Capabilities

Implemented:
- Flat iteration (`{{#each}}`)
- Dotted field paths (`{{field:a.b.c}}`)
- Signed arc bindings (`{{bind:+:0}}`, `{{bind:-:all}}`)
- Conditional blocks (`{{#if …}}`)
- Comments (`{{#comment}} … {{/comment}}`)
- Config injection (`{{config:…}}`)
- Reference following in field paths

Not yet:
- `{{#else}}` / `{{#switch}}`
- Recursive tree expansion (MJCF body-in-body)
- Expression arithmetic (`{{expr:x * 0.5}}`)
- Cross-query joins (`{{#with label match}}`)
- Indentation control
- Hot reload in daemon mode

## 8. Relationship to Existing Code

The hardcoded `hymeko_query::formats::{urdf, sdf}` and the
`hymeko_query::transforms::{DomainTransform, TransformRegistry}` registry
remain in place as reference implementations. The rewrite engine sits
alongside them; the CLI exposes both paths (`compile` vs `transform`).
Future work can migrate reference formats into the external
`transforms/` directory and eventually retire the hardcoded modules.

## 9. Files at a Glance

```
hymeko_query/src/
├── rewrite/
│   ├── mod.rs                 re-exports execute_transform, TransformSpec
│   ├── match_context.rs       FieldValue, MatchContext<'a, R>
│   └── template.rs            Block, parse_template, render,
│                              RenderContext, execute_transform
└── interpret.rs               + interpret_transform_queries()
                               + *_opts() variants with label-only mode

hymeko_cli/src/main.rs         Commands::Transform, tf/tdir REPL cmds,
                               load_transform_spec, list_available_transforms

transforms/
├── urdf/     queries.hymeko + template.urdf.xml
├── sdf/      queries.hymeko + template.sdf.xml
├── mjcf/     queries.hymeko + template.mjcf.xml
├── dot/      queries.hymeko + template.dot
└── ros2_launch/ queries.hymeko + template.launch.py
```
