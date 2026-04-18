# Query Variables — `?name` Syntax

The `?` token introduces a **query variable**: a placeholder in a `.hymeko`
pattern that captures whatever concrete declaration the query engine matches
at that position, so later rules can refer to the same match by name.

This document covers what's currently wired, what a minimum-viable example
looks like, and where to extend next. See
`docs/plans/04_graph_query/T10_lalrpop_extension.md` for the design record
and `parser/tests/query_variable.rs` for executable regression tests.

---

## What's wired today (2026-04-18)

| Layer | Status |
|-------|--------|
| Lexer token `Question` | ✅ emitted on `?` (`parser/src/lexer/common.rs:260`) |
| AST `QueryVar` | n/a — rule produces a borrowed `&str` directly |
| Grammar rule `pub QueryVar: &'a str = { "?" <Ident> => … }` | ✅ `parser/src/hymeko.lalrpop:286` |
| Parser entry `parse_query_var(input)` | ✅ `parser/src/lib.rs` |
| Grammar regression tests | ✅ 7 tests in `parser/tests/query_variable.rs` |
| Reachable from `parse_description` (top-level `.hymeko`) | ❌ not yet — `?x` in a full file is still an error |
| Interpreted by the query engine | ❌ future work |

So: today, **`?x` is a standalone syntactic fragment** the parser can recognise,
but it does not yet appear inside the top-level Description grammar. The wiring
is deliberately narrow so the token and rule can be tested now without a
concurrent grammar-semantics change.

---

## Simple example 1 — the rule in isolation

```rust
use parser::parse_query_var;

let name = parse_query_var("?x").unwrap();
assert_eq!(name, "x");
```

Equivalent call from a Rust integration test (as in
`parser/tests/query_variable.rs`):

```rust
#[test]
fn simple_query_variable_parses() {
    assert_eq!(parse_query_var("?link_name").unwrap(), "link_name");
}
```

## Simple example 2 — what rejection looks like

```rust
assert!(parse_query_var("x").is_err());       // missing `?`
assert!(parse_query_var("?").is_err());       // missing name
assert!(parse_query_var("?x ?y").is_err());   // single-shot entry rejects two vars
```

## Simple example 3 — whitespace tolerance

The lexer skips whitespace before every token, so `? name` parses the same as
`?name`:

```rust
assert_eq!(parse_query_var("? spaced").unwrap(), "spaced");
```

## Simple example 4 — intended future query pattern (design sketch)

Once `?` is wired into the pattern grammar, a query for "every revolute joint
between two links, capturing the parent, child, and joint" will look like:

```hymeko
// Pattern file — NOT yet parsed by hymeko_cli.
// Sketch only: shows what the integrated T10 syntax will enable.
@?joint: rev_joint {
    (+ ?parent, - ?child, - _);
}
```

The corresponding match binds `?joint` to the `DeclId` of the hyperedge,
`?parent` to the `+`-signed endpoint, and `?child` to the `-`-signed endpoint
the axis atom is discarded via the `_` wildcard that already works today.

The query engine will then expose these bindings — something like:

```rust
// Forward-looking API — will land when the interpreter consumes ?x.
let batch = engine.query_with_vars(&pattern);
for m in batch {
    println!(
        "joint {} connects parent {} to child {}",
        m.var("joint").name(), m.var("parent").name(), m.var("child").name()
    );
}
```

## Simple example 5 — rewrite rule with captured variables

The `hymeko_query::rewrite` template engine already accepts matched results.
Once `?` is plumbed through `interpret_as_queries`, templates can reference
captures by name:

```text
{{#each matches}}
  parent={{var:parent}}  child={{var:child}}  joint={{var:joint}}
{{/each}}
```

This is the natural handshake between the variable-binding front end and the
existing template back end.

---

## Why it is scoped this narrow today

The T10 planning doc calls out the full extension as "3 one-line edits" at
the parser layer but also notes the harder work of **where** to permit `?x`
in the grammar (it can appear in identifier positions, ref-target positions,
and annotation-tag positions — each with different ambiguity implications)
plus the semantic additions in `hymeko_query::interpret` to thread captures
into `QueryMatch`. Landing that across the stack in one pass risks silent
grammar ambiguities.

So today the deliverable is:

1. The lexer emits `?` correctly.
2. The grammar has a named, **tested** rule for `?name`.
3. A standalone parser entry point lets tools and tests exercise the rule.
4. A regression suite guards the surface.

The next vertical slice lands `?` inside `Ref` (or a new `QueryTerm` rule
that's accepted wherever a `Ref` is) and adds an AST variant
`Ref::Var(Id)` so the query interpreter has something to match against.

## Follow-up (not in this slice)

- Extend `Ref` in `parser/src/ast.rs` with a `Var(Id)` variant.
- Grammar: permit `QueryVar` where `Ref` appears in arc signed-ref tuples
  and in node-base positions.
- Interpreter (`hymeko_query/src/interpret.rs`): lift `Var` into a new
  `Predicate::Capture { name, inner }`.
- Engine (`hymeko_query/src/engine.rs`): maintain a `BTreeMap<SymId, DeclId>`
  on each `QueryMatch` for captures.
- Template engine (`hymeko_query/src/rewrite/template.rs`): expose captures
  via `{{var:<name>}}` placeholders.
