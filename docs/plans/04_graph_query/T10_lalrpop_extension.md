# T10 — LALRPOP Grammar Extension

**Status:** ❌ NOT DONE  
**Files to change:** 3 files, 3 one-line edits  
**Risk:** Low  
**Priority:** Nice-to-have (paper can describe it without implementation)

---

## Current State

The `_` wildcard already works because `_` is a valid identifier in the lexer (`is_ident_start` accepts `b'_'`). The interpreter checks `name == "_"` and skips the `Named` predicate. No grammar change was needed for this.

## What's Missing: `?` Token for Variable Binding

Variable binding (`?x` captures the matched DeclId for later use) requires a `?` token.

### Change 1: `parser/src/lexer/token.rs`

```rust
pub enum Token<'a> {
    // ... existing variants ...
    Question,  // '?' for query variable binding
    EOF,
}
```

### Change 2: `parser/src/lexer/common.rs`

In `next_token`, add before the `other =>` arm:

```rust
b'?' => Token::Question,
```

### Change 3: `parser/src/hymeko.lalrpop`

In the `extern` enum block:

```lalrpop
"?" => crate::lexer::Token::Question,
```

### Optional: Grammar Rule

```lalrpop
QueryVar: &'a str = {
    "?" <name:Ident> => name,
};
```

This rule is additive — no existing production uses `?` in any position where this would conflict.

## Why Not Done

Cannot verify LALRPOP grammar compilation without edition 2024 support (Rust 1.85+). The changes are mechanical and safe, but shipping untested grammar changes for a deadline would be irresponsible.

## Future Extensions (v2)

| Token | Syntax | Purpose |
|-------|--------|---------|
| `?` | `?x` | Variable binding |
| `*` | `*` in ref position | Any ref target |
| `..` | `.. { child }` | Deep containment (any nesting depth) |
| `>=` | `mass >= 10.0` | True comparison syntax (conflicts with `<`/`>` for tags) |
| `!` | `!tagged` | Negation prefix |
| `\|` | `a \| b` | Disjunction between alternatives |

The `>=`/`<=` operators require disambiguating `< gt >` (tag) from `<= 10.0` (comparison) at the lexer level. This is non-trivial and deferred.
