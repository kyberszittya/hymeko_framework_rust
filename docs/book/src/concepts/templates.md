# Concept: Templates and codegen

Templates are the **primary codegen path**. No Rust string-builders on the main dispatch path; format-specific output is text files in `transforms/<name>/`.

## The directive set

`hymeko_query/src/rewrite/template.rs` (~500 LOC) implements the template engine. Supported directives:

```handlebars
{{name}}                              the current decl's resolved name
{{field:fieldname}}                   value of a child node statement
{{config:robot_name}}                 a TransformConfig field
{{#each label}}...{{/each}}           iterate matches of named query "label"
{{#if field:fieldname}}...{{/if}}     conditional on field presence
{{#inherits "base"}}...{{/inherits}}  conditional on inheritance
{{#comment}}...{{/comment}}           template comment (not emitted)
{{bind:+:0}}                          first +arc-ref binding
{{bind:-:0}}                          first -arc-ref binding
{{bind:~:0}}                          ~ (op) binding
{{bind:+:all}}                        all + bindings, space-separated
{{bind:+:all_csv}}                    all + bindings, comma-separated
```

## Render flow

`hymeko_formats::codegen::generate_description(ir, resolver, name, format)` → `TransformRegistry::render_from_templates`:

1. Read `transforms/<format>/queries.hymeko` → list of `NamedQuery`
2. Read `transforms/<format>/template.<ext>`
3. For each `{{#each label}}...{{/each}}`, run the named query, bind matches, render body
4. Inside an each-block, `{{name}}`, `{{field:...}}`, `{{bind:...}}` resolve against the current match
5. Concatenate everything and return the string

## When NOT to use templates

If the emit needs typed model extraction (URDF needs parsed origin / axis / limits as floats, MJCF needs a body-tree), implement a typed extractor (see `hymeko_query::kinematics::extract_kinematic_model`) and have `DomainTransform::emit` use it. The result can still feed a template (URDF and SDF do this — the template path delegates to `generate_urdf_from_model` after the May 2026 cleanup).

## See also

- [Add a new format](../recipes/add-a-format.md) — full walkthrough
- `hymeko_query/src/rewrite/template.rs` — engine source
- `transforms/dot/template.dot` — the simplest reference template
