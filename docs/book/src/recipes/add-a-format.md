# Recipe: Add a new format

Goal: emit a new output format (e.g. ONNX, GraphML, your own) from any `.hymeko` IR.

In most cases — **no Rust changes needed**. Just two text files:
- `transforms/<name>/queries.hymeko` — what to extract from the IR
- `transforms/<name>/template.<ext>` — how to render it

The template engine handles the rest.

## Step 1: pick a name

E.g. `graphml` for the GraphML XML format. Conventionally, `transforms/<name>/` is the directory.

```bash
mkdir transforms/graphml
```

## Step 2: write `queries.hymeko`

This declares what kinds of decls you want to iterate over. Mirror the structure of an existing simple format like DOT (`transforms/dot/queries.hymeko`):

```hymeko
graphml_transform {}
context
{
    // All node decls — vertices in GraphML
    nodes: meta_node {}

    // All edge decls — edges in GraphML
    edges: meta_edge {}

    // For visiting per-arc:
    @arcs: meta_arc {}
}
```

The query-engine binds these collections so the template can iterate them with `{{#each nodes}}...{{/each}}`.

## Step 3: write `template.<ext>`

Per-collection iteration + per-decl access. Example `transforms/graphml/template.graphml`:

```handlebars
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="{{config:robot_name}}" edgedefault="directed">
    {{#each nodes}}
    <node id="n{{name}}"/>
    {{/each}}
    {{#each edges}}
    {{#each arcs}}
    <edge source="e{{bind:~:0}}" target="{{bind:+:0}}" sign="{{bind:+:sign:0}}"/>
    {{/each}}
    {{/each}}
  </graph>
</graphml>
```

Available template directives (full reference: `hymeko_query/src/rewrite/template.rs`):

| directive | what it does |
|---|---|
| `{{name}}` | The decl's resolved name |
| `{{field:fieldname}}` | A scalar child node's value |
| `{{#each label}}...{{/each}}` | Iterate over the matches of `label` from queries.hymeko |
| `{{#if field:fieldname}}...{{/if}}` | Conditional on field presence |
| `{{#inherits "base"}}...{{/inherits}}` | Conditional on inheritance |
| `{{bind:+:0}}` / `{{bind:-:0}}` / `{{bind:~:0}}` | First positive / negative / op binding name |
| `{{bind:+:all_csv}}` | All positive bindings, comma-separated (multi-input fan-in) |
| `{{config:robot_name}}` | The `robot_name` field from the `TransformConfig` |
| `{{#comment}}...{{/comment}}` | Template comments (not emitted) |

## Step 4: register the format in Rust

In `hymeko_formats/src/transforms.rs`, add:

```rust
pub struct GraphmlTransform;

impl DomainTransform for GraphmlTransform {
    fn name(&self) -> &'static str { "graphml" }
    fn extension(&self) -> &'static str { "graphml" }
    fn accepts(&self) -> ModelKind { ModelKind::Generic }
    fn emit(&self, _model: &ModelView, _config: &TransformConfig) -> Option<String> {
        // Pure template-driven — no Rust string-builder needed
        None
    }
    fn template_dir(&self) -> Option<&'static str> { Some("graphml") }
}
```

In `hymeko_formats/src/lib.rs`, register it in `register_defaults`:

```rust
pub fn register_defaults(reg: &mut TransformRegistry) {
    // ... existing ones ...
    reg.register(Box::new(GraphmlTransform));
}
```

In `hymeko_formats/src/codegen.rs`, add an `OutputFormat::GraphML` enum variant + `transform_name` mapping:

```rust
pub enum OutputFormat {
    // ... existing ...
    GraphML,
}

impl OutputFormat {
    fn transform_name(self) -> &'static str {
        match self {
            // ...
            OutputFormat::GraphML => "graphml",
        }
    }
}
```

In `hymeko_cli/src/main.rs`, register the CLI flag:

```rust
fn parse_format(s: &str) -> OutputFormat {
    match s.to_lowercase().as_str() {
        // ...
        "graphml" => OutputFormat::GraphML,
        // ...
    }
}
```

## Step 5: rebuild + use

```bash
cargo build --release
target/release/hymeko emit some_input.hymeko --format graphml -o out.graphml
```

## What if I need richer extraction than queries can express

Sometimes a format needs typed model extraction (URDF needs `KinematicModel` with parsed origins/axes/limits). For that:

1. Add a typed extractor in `hymeko_query/src/kinematics/` (or a new sibling module) producing your `MyModel` struct
2. Implement `DomainTransform::emit` to call your extractor + run a Rust-side renderer (or use the extractor's output to feed a richer template)

The URDF / SDF / MJCF emitters do this via `extract_kinematic_model` + `generate_urdf_from_model`. See `hymeko_formats/src/urdf.rs` for the pattern.

## Verification

`hymeko_query/tests/test_template_driven.rs` includes `render_from_templates_is_deterministic` — a good template for adding format-specific tests. Add a `tests/test_graphml.rs` file with a known-input → known-output assertion.

## Next

- [Add a new layer kind](./add-a-layer-kind.md) — extends the schema instead of the output
- [Add a new query](./add-a-query.md) — typed predicate API
