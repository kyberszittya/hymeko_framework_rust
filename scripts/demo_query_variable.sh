#!/usr/bin/env bash
#
# Demo — T10 slice 1: the `?` query-variable token.
#
# Shows the parser-level surface landed on 2026-04-18: the lexer emits the
# `?` token, the grammar has a `pub QueryVar` rule, and `parse_query_var()`
# is exposed as a single-fragment entry point.
#
# Full integration into query/rewrite patterns is the next T10 slice;
# see docs/examples/query_variables.md § "Follow-up" for the punch list.
#
# Run from the workspace root:
#   bash scripts/demo_query_variable.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Build parser (populates OUT_DIR for the LALRPOP-generated module)"
cargo build -p parser --quiet

echo
echo '==> Run the `?`-token regression suite (7 tests)'
cargo test -p parser --test query_variable 2>&1 | tail -15

echo
echo '==> Example 1: parse_query_var("?x") via an inline Rust snippet'
cat <<'RUST' > /tmp/hymeko_qvar_demo.rs
fn main() {
    let tests = [
        "?x",
        "?link_name",
        "?MyVar",
        "? spaced",
    ];
    for src in tests {
        match parser::parse_query_var(src) {
            Ok(name) => println!("  {:<14}  ->  binds `{}`", format!("{:?}", src), name),
            Err(e)   => println!("  {:<14}  ->  ERR {:?}", format!("{:?}", src), e),
        }
    }
    for bad in ["x", "?", "?x ?y"] {
        match parser::parse_query_var(bad) {
            Ok(name) => println!("  {:<14}  ->  unexpectedly parsed as `{}`", format!("{:?}", bad), name),
            Err(_)   => println!("  {:<14}  ->  rejected (expected)", format!("{:?}", bad)),
        }
    }
}
RUST

# Build a throwaway binary that depends on the parser crate.
TMP_BIN_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_BIN_DIR"' EXIT
cat > "$TMP_BIN_DIR/Cargo.toml" <<EOF
[package]
name = "qvar_demo"
version = "0.0.0"
edition = "2024"

[[bin]]
name = "qvar_demo"
path = "main.rs"

[dependencies]
parser = { path = "$(pwd)/parser" }
EOF
cp /tmp/hymeko_qvar_demo.rs "$TMP_BIN_DIR/main.rs"

cargo run --quiet --manifest-path "$TMP_BIN_DIR/Cargo.toml"

echo
echo "==> Done."
