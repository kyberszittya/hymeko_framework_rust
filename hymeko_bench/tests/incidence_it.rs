//! Integration coverage for `incidence`: compile a generated hypergraph and
//! check the structural counts and adjacency baseline against ground truth.

use hymeko::module_store::module_store::ModuleStore;
use hymeko::module_store::source_provider::StdFsProvider;
use hymeko::util::real_parser::RealParser;

use hymeko_bench::corpus::random_hymeko_source;
use hymeko_bench::incidence::{build_adjacency, build_ir_records, structure_counts};

fn compile_source(
    tag: &str,
    src: &str,
) -> std::sync::Arc<hymeko::module_store::module_store::CompiledProgram> {
    // Unique path per test: the integration binary runs test fns concurrently,
    // so a shared filename would race into a corrupted source.
    let dir = std::env::temp_dir().join("hymeko_bench_incidence_it");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("case_{tag}.hymeko"));
    std::fs::write(&path, src).unwrap();
    let mut store = ModuleStore::new(StdFsProvider::new(), RealParser);
    store.compile(&path).expect("compile generated source")
}

#[test]
fn counts_match_generated_structure() {
    let nodes = 40;
    let edges = 20;
    let src = random_hymeko_source(nodes, edges, 0.15, 99);
    let compiled = compile_source("counts", &src);
    let sc = structure_counts(&compiled.ir);
    // The compiled IR carries one implicit context/root vertex in addition to
    // the `nodes` declared ones (COO suite shows the same 16 -> 17 offset).
    assert_eq!(sc.n, nodes + 1, "vertex count (declared + implicit root)");
    assert_eq!(sc.m, edges, "hyperedge count");
    assert!(sc.nnz >= edges, "each edge contributes >= 1 incidence");
    assert!(sc.mean_arity > 0.0);
    assert!(sc.n_over_m() > 0.0);
    assert!(sc.rho_predicted_unit() > 1.0);
}

#[test]
fn adjacency_entry_total_equals_nnz() {
    let src = random_hymeko_source(24, 12, 0.2, 7);
    let compiled = compile_source("adj", &src);
    let sc = structure_counts(&compiled.ir);
    let adj = build_adjacency(&compiled.ir);
    let total: usize = adj.iter().map(Vec::len).sum();
    assert_eq!(total, sc.nnz, "adjacency stores exactly nnz entries");

    let (digests, name_index) = build_ir_records(sc.n + sc.m);
    assert_eq!(digests.len(), sc.n + sc.m);
    assert_eq!(name_index.len(), sc.n + sc.m);
}
