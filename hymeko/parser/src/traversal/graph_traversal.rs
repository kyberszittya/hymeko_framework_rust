use std::collections::{HashSet, VecDeque};
use crate::traversal::graphview::GraphView;

pub fn bfs<V: GraphView>(view: &V, start: V::Node) -> Vec<V::Node> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut q = VecDeque::new();

    seen.insert(start);
    q.push_back(start);

    while let Some(u) = q.pop_front() {
        out.push(u);
        for v in view.neighbors(u) {
            if seen.insert(v) {
                q.push_back(v);
            }
        }
    }
    out
}

pub fn dfs_preorder<V: GraphView>(view: &V, start: V::Node) -> Vec<V::Node> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut st = vec![start];

    while let Some(u) = st.pop() {
        if !seen.insert(u) { continue; }
        out.push(u);

        for v in view.neighbors(u) {
            st.push(v);
        }
    }
    out
}