//! Generic implicit-graph A\* — the planner-framework core HOTARU searches with.
//!
//! Nodes are produced on the fly by a `neighbours` closure, so the search space is never
//! materialised. This mirrors the framework's implicit-graph A\*
//! (`hymeko_rl.control.graph_planner.astar`); the *other* A\* in the workspace
//! (`hymeko_graph::astar`) is specialised to a fixed integer-indexed CSR — it does not fit an
//! on-the-fly node space, and depending on it from `akoire` would add a cross-crate dependency
//! (a `CORE.YAML` §1 change). So HOTARU carries this small `std`-only routine: the same
//! algorithm, a node model that fits the HIVE-delta space, and zero new dependencies.
//!
//! # Contract
//! - **Pre**: `max_expansions >= 1`; every edge `cost >= 0`; `heuristic(n) >= 0` and *admissible*
//!   (never overestimates the remaining cost to a goal) for the returned path to be optimal.
//! - **Post**: `actions == Some(p)` ⇒ following `p` from `start` reaches a node satisfying
//!   `is_goal`, and (with an admissible heuristic) `p` has minimum total cost; `actions == None`
//!   ⇒ no goal was reachable within `max_expansions`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

/// Outcome of an [`astar`] search: the optimal action path (if any) plus effort statistics.
#[derive(Debug, Clone)]
pub struct AstarResult<E> {
    /// The optimal sequence of edges/actions from `start` to a goal, or `None` if unreachable
    /// within the expansion budget.
    pub actions: Option<Vec<E>>,
    /// Nodes expanded (popped and closed) — the deterministic effort budget the perf test asserts.
    pub expansions: usize,
    /// Peak size the open set (frontier) reached — a memory-pressure proxy.
    pub frontier_peak: usize,
}

/// Priority-queue entry ordered as a *min-heap on `f`* (Rust's [`BinaryHeap`] is a max-heap, so the
/// comparison is reversed), tie-broken by an insertion counter so the ordering is total and
/// deterministic and the node payload is never compared.
#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    f: f64,
    seq: u64,
}
impl Eq for HeapEntry {}
impl PartialEq for HeapEntry {
    fn eq(&self, o: &Self) -> bool {
        self.f == o.f && self.seq == o.seq
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        // Smaller f pops first; on ties, the earlier-inserted (smaller seq) pops first.
        o.f.partial_cmp(&self.f)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.seq.cmp(&self.seq))
    }
}

/// A\* over an implicit graph. `neighbours(n)` yields `(edge, next_node, cost)` triples generated on
/// demand; `is_goal` and `heuristic` score nodes. Returns the optimal action path plus effort stats.
///
/// See the module contract for pre/postconditions.
pub fn astar<N, E, FN, IT, FG, FH>(
    start: N,
    mut neighbours: FN,
    mut is_goal: FG,
    mut heuristic: FH,
    max_expansions: usize,
) -> AstarResult<E>
where
    N: Clone + Eq + Hash,
    E: Clone,
    FN: FnMut(&N) -> IT,
    IT: IntoIterator<Item = (E, N, f64)>,
    FG: FnMut(&N) -> bool,
    FH: FnMut(&N) -> f64,
{
    let mut g: HashMap<N, f64> = HashMap::new();
    let mut came: HashMap<N, (N, E)> = HashMap::new();
    let mut closed: HashSet<N> = HashSet::new();
    let mut nodes: Vec<N> = Vec::new(); // seq -> node payload (keeps N out of the heap's Ord)
    let mut open: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut expansions = 0usize;
    let mut frontier_peak = 0usize;

    g.insert(start.clone(), 0.0);
    let f0 = heuristic(&start);
    nodes.push(start);
    open.push(HeapEntry { f: f0, seq: 0 });
    frontier_peak = frontier_peak.max(open.len());

    while let Some(HeapEntry { seq, .. }) = open.pop() {
        // Invariant: every `seq` pushed onto `open` was first pushed onto `nodes`, so the index is
        // always in range — the panic is unreachable (documented per CLAUDE.md §6.4).
        let cur = nodes[seq as usize].clone();
        if closed.contains(&cur) {
            continue;
        }
        if is_goal(&cur) {
            return AstarResult {
                actions: Some(reconstruct(&came, cur)),
                expansions,
                frontier_peak,
            };
        }
        if expansions >= max_expansions {
            break;
        }
        closed.insert(cur.clone());
        expansions += 1;

        let cur_g = *g.get(&cur).unwrap_or(&f64::INFINITY);
        for (edge, nxt, cost) in neighbours(&cur) {
            if closed.contains(&nxt) {
                continue;
            }
            let tentative = cur_g + cost;
            if tentative < *g.get(&nxt).unwrap_or(&f64::INFINITY) {
                g.insert(nxt.clone(), tentative);
                came.insert(nxt.clone(), (cur.clone(), edge));
                let f = tentative + heuristic(&nxt);
                let seq = nodes.len() as u64;
                nodes.push(nxt);
                open.push(HeapEntry { f, seq });
                frontier_peak = frontier_peak.max(open.len());
            }
        }
    }

    AstarResult {
        actions: None,
        expansions,
        frontier_peak,
    }
}

/// Walk the `came_from` chain back from `goal`, collecting the edges in forward order.
fn reconstruct<N, E>(came: &HashMap<N, (N, E)>, goal: N) -> Vec<E>
where
    N: Clone + Eq + Hash,
    E: Clone,
{
    let mut actions = Vec::new();
    let mut cur = goal;
    while let Some((prev, edge)) = came.get(&cur) {
        actions.push(edge.clone());
        cur = prev.clone();
    }
    actions.reverse();
    actions
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny explicit weighted graph as an adjacency map, searched through the implicit API.
    fn grid_neighbours(
        adj: &HashMap<u32, Vec<(u32, f64)>>,
    ) -> impl Fn(&u32) -> Vec<(u32, u32, f64)> + '_ {
        move |n: &u32| {
            adj.get(n)
                .map(|es| es.iter().map(|&(m, c)| (m, m, c)).collect())
                .unwrap_or_default()
        }
    }

    fn line_graph() -> HashMap<u32, Vec<(u32, f64)>> {
        // 0 -1-> 1 -1-> 2 -1-> 3, plus a shortcut 0 -1-> 3 via a longer weighted detour 0 -5-> 3.
        HashMap::from([
            (0, vec![(1, 1.0), (3, 5.0)]),
            (1, vec![(2, 1.0)]),
            (2, vec![(3, 1.0)]),
        ])
    }

    #[test]
    fn astar_finds_shortest_weighted_path() {
        let adj = line_graph();
        let res = astar(
            0u32,
            grid_neighbours(&adj),
            |n| *n == 3,
            |n| (3 - *n) as f64,
            100,
        );
        // 0->1->2->3 (cost 3) beats the direct 0->3 (cost 5).
        assert_eq!(res.actions, Some(vec![1, 2, 3]));
    }

    #[test]
    fn astar_zero_heuristic_still_optimal() {
        let adj = line_graph();
        let res = astar(0u32, grid_neighbours(&adj), |n| *n == 3, |_| 0.0, 100);
        assert_eq!(res.actions, Some(vec![1, 2, 3])); // h≡0 ⇒ Dijkstra, same optimum
    }

    #[test]
    fn astar_returns_none_when_unreachable() {
        let adj: HashMap<u32, Vec<(u32, f64)>> = HashMap::from([(0, vec![(1, 1.0)])]);
        let res = astar(0u32, grid_neighbours(&adj), |n| *n == 9, |_| 0.0, 100);
        assert!(res.actions.is_none());
    }

    #[test]
    fn astar_respects_expansion_budget() {
        // An infinite ray 0->1->2->…; with a tiny budget the goal (far away) is never reached.
        let res = astar(
            0u64,
            |n: &u64| vec![(n + 1, n + 1, 1.0)],
            |n| *n == 1_000,
            |n| (1_000 - *n) as f64,
            5,
        );
        assert!(res.actions.is_none());
        assert!(res.expansions <= 5);
    }

    #[test]
    fn astar_start_is_goal_zero_expansions() {
        let res = astar(
            7u32,
            |_: &u32| Vec::<(u32, u32, f64)>::new(),
            |n| *n == 7,
            |_| 0.0,
            10,
        );
        assert_eq!(res.actions, Some(vec![]));
        assert_eq!(res.expansions, 0);
    }
}
