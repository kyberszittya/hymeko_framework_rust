# Extension points

Where to plug new code without rewriting the framework.

## Adding a new emit format

`transforms/<name>/{queries.hymeko, template.<ext>}` plus a 3-line `DomainTransform` impl in `hymeko_formats`. See [Add a new format](../recipes/add-a-format.md).

## Adding a new layer kind

Touch four places: `meta_nn.hymeko`, `transforms/torch_dataflow/queries.hymeko`, `transforms/torch_dataflow/template.py`, `python/ehk_torch_stub/`. See [Add a new layer kind](../recipes/add-a-layer-kind.md).

## Adding a new query

Define in `hymeko_query::predicate` (typed) or call `match_expr` directly with a string. See [Add a new query](../recipes/add-a-query.md).

## Adding a new training-graph op

Register a handler in `signedkan_wip/src/hymeko_train_walker.py` via `@register("kind")`. See [HyMeKo-controlled training](../quickstart/09-hsikan-training.md).

## Adding a new IR analysis

If non-destructive: free function over `&Ir`. If destructive: produce a new IR (don't mutate). See e.g. `hymeko_query::entropy::compute_entropy` for the pattern.

## Adding a new graph-runtime engine

`hymeko_hre` defines `HypergraphEngine` as the runtime entry-point for tensor expansions. New expansion variants (clique, star, tensor) plug in there.
