# Reference: CLI

The `hymeko` binary (built from `cargo build --release`) is the universal entry point.

```
hymeko [COMMAND]

Commands:
  compile    Compile a .hymeko file and generate robot description output
  validate   Validate a .hymeko file (parse + resolve + check topology)
  inspect    Pretty-print the compiled IR
  console    Start the interactive console (default with no args)
  query      Run queries from a .hymeko query file against a compiled description
  transform  Run a template-driven transform
  emit       Emit a format by rendering a template
  entropy    Compute per-scope structural entropy on the compiled IR
  rewrite    Propose a k=2 split of a hypervertex body via k-means on incidence-row signatures
  help       Print this message or the help of the given subcommand(s)
```

## `parse` / `validate` / `inspect`

Lex + parse + resolve + validate.

```bash
hymeko validate data/robotics_imported/wam/wam.hymeko
hymeko inspect  data/robotics_imported/wam/wam.hymeko
```

`inspect` prints the IR in a human-readable form — useful for debugging your `.hymeko` files.

## `emit` (recommended)

Template-driven format emit. Used in nearly every quickstart in this book.

```bash
hymeko emit <input.hymeko> --format <urdf|sdf|mjcf|dot|torch_dataflow|gazebo_world|mermaid> \
    [--name <name>] [-o <output_path>]
```

`--format` corresponds to a directory `transforms/<format>/`. Add a new format = add a new directory; see [Add a new format](./recipes/add-a-format.md).

## `compile` (legacy)

Pre-template compile path. Goes through the Rust string-builder URDF / SDF emitters. After the May 2026 cleanup, both `compile` and `emit` converge on the same emission code; prefer `emit` for new use.

## `query`

Run a `.hymeko` query file against a description.

```bash
hymeko query <input.hymeko> '<predicate-string>'
```

E.g.:
```bash
hymeko query foo.hymeko 'INHERITS(link)'
hymeko query foo.hymeko 'INHERITS(rev_joint) AND SCOPEDIN(my_robot)'
```

See [Concepts: Queries](./concepts/queries.md) for the predicate language.

## `transform`

Run a transform with full control over which queries + template to use. Lower-level than `emit`.

## `entropy`

Compute structural entropy of the IR.

```bash
hymeko entropy <input.hymeko>
```

Outputs Shannon entropy of the singular-value spectrum of the clique-tensor expansion; per-scope breakdown if the IR is hierarchical.

## `rewrite`

Suggest a k=2 split of a hypervertex body — Step 3 of the entropy hot-swap plan.

```bash
hymeko rewrite <input.hymeko> <hypervertex_name>
```

Reads the entropy and clusters vertices by which edges they participate in.

## `console`

Interactive REPL. Type a `.hymeko` source, see live IR / DOT / query results.

## See also

- [Quickstart: Parse](./quickstart/01-parse.md) — first contact with the CLI
- [Quickstart: Emit URDF](./quickstart/02-emit-urdf.md) — the most common emit path
