use criterion::{Criterion, criterion_group, criterion_main};
use hymeko_hive::{HiveDelta, HiveNode, HiveQuery, HiveRelation, HiveStore, HiveTransaction, Sign};

fn build_store(n_nodes: usize, n_relations: usize) -> HiveStore {
    let mut store = HiveStore::new();
    let mut deltas = Vec::with_capacity(n_nodes + n_relations);
    for i in 0..n_nodes {
        let ty = if i % 4 == 0 {
            "trait"
        } else if i % 4 == 1 {
            "behavior"
        } else if i % 4 == 2 {
            "context"
        } else {
            "boundary"
        };
        deltas.push(HiveDelta::AddNode(HiveNode::new(
            format!("node_{i:05}"),
            ty,
        )));
    }
    for i in 0..n_relations {
        let src = format!("node_{:05}", i % n_nodes);
        let dst = format!("node_{:05}", (i * 17 + 3) % n_nodes);
        deltas.push(HiveDelta::AddRelation(HiveRelation::new(
            format!("rel_{i:05}"),
            if i % 2 == 0 {
                "responds_with"
            } else {
                "constrains"
            },
            vec![(Sign::Plus, src.into()), (Sign::Minus, dst.into())],
        )));
    }

    let tx = HiveTransaction::new("bench-bootstrap", store.state_hash(), "bench", 0, deltas);
    store.commit(tx).unwrap();
    store
}

fn query_bench(c: &mut Criterion) {
    let store = build_store(10_000, 25_000);

    c.bench_function("hive query nodes by type", |b| {
        b.iter(|| store.query(&HiveQuery::NodesByType("trait".to_string())))
    });

    c.bench_function("hive query relations by type", |b| {
        b.iter(|| store.query(&HiveQuery::RelationsByType("responds_with".to_string())))
    });

    c.bench_function("hive query relations by endpoint node type", |b| {
        b.iter(|| {
            store.query(&HiveQuery::RelationsByEndpointNodeType {
                sign: Sign::Minus,
                node_type: "behavior".to_string(),
            })
        })
    });
}

criterion_group!(benches, query_bench);
criterion_main!(benches);
