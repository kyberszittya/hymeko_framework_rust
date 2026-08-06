# hymeko_pgraph_py

PyO3 bindings for the `hymeko_pgraph` crate. Exposes Friedler MSG /
decision-mapping SSG / ABB top-K to Python.

## Build (in-tree dev)

```
cd hymeko_pgraph_py
maturin develop --release
python -c "import hymeko_pgraph_py; print(hymeko_pgraph_py.__name__)"
```

## API

```python
import hymeko_pgraph_py as hp

g   = hp.from_hymeko_text(open("data/pgraph/Chapter6/example6_1.hymeko").read())
msg = hp.maximal_structure_rs(g)            # canonical (strict_no_excess=False)
sss = hp.enumerate_ssg_rs(g, msg)            # all Friedler solution structures
sol = hp.solve_abb_rs(g, msg)                # cost-minimal feasible structure
top = hp.solve_top_k_abb_rs(g, msg, k=3)     # 3 cheapest feasible structures
```

See plan: `docs/plans/2026-06-04-hymeko-pgraph-py/plan.pdf`.
