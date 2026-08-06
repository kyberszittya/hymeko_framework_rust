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

/// A planning problem the [`solve`] framework can search — the one planner interface reused across
/// domains. An implementor supplies a start node, an on-demand successor relation, a goal predicate,
/// and an (ideally admissible) heuristic; the framework contributes the A\* search.
///
/// HOTARU's structure-synthesis search (`hotaru::HiveDeltaProblem`, `Node = HyQL source`,
/// `Edge = HiveDelta`) is one implementor; a motion/grid search (`Node = cell`, `Edge = step`) is
/// another (see the tests). The humanoid footstep planner (Python `scenarios/humanoid`) *mirrors*
/// this same shape on the other side of the FFI boundary.
pub trait SearchProblem {
    /// A search state. Must be hashable so visited states collapse.
    type Node: Clone + Eq + Hash;
    /// An action / edge label recorded along the solution path.
    type Edge: Clone;

    /// The initial state.
    fn start(&self) -> Self::Node;
    /// Successors of `node` as `(edge, next_node, cost)`, generated on demand.
    fn neighbours(&self, node: &Self::Node) -> Vec<(Self::Edge, Self::Node, f64)>;
    /// Whether `node` satisfies the goal.
    fn is_goal(&self, node: &Self::Node) -> bool;
    /// Estimated remaining cost from `node` to a goal (admissible ⇒ optimal path).
    fn heuristic(&self, node: &Self::Node) -> f64;
}

/// Solve a [`SearchProblem`] with the A\* core. The trait-based entry point of the planner framework;
/// it simply adapts the problem's methods to [`astar`]'s closures, so the same search serves every
/// domain that implements [`SearchProblem`].
pub fn solve<P: SearchProblem>(problem: &P, max_expansions: usize) -> AstarResult<P::Edge> {
    astar(
        problem.start(),
        |n| problem.neighbours(n),
        |n| problem.is_goal(n),
        |n| problem.heuristic(n),
        max_expansions,
    )
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

    /// A motion-flavoured `SearchProblem`: a 4-connected occupancy grid with obstacles — the same
    /// shape a footstep planner takes. Proves the `SearchProblem`/`solve` framework spans *motion*,
    /// not just HOTARU's structure-synthesis search (the "reused by both" claim, Rust-side).
    struct GridProblem {
        w: i32,
        h: i32,
        goal: (i32, i32),
        blocked: Vec<(i32, i32)>,
    }

    impl SearchProblem for GridProblem {
        type Node = (i32, i32);
        type Edge = (i32, i32); // the cell stepped into

        fn start(&self) -> Self::Node {
            (0, 0)
        }
        fn neighbours(&self, n: &Self::Node) -> Vec<(Self::Edge, Self::Node, f64)> {
            [(-1, 0), (1, 0), (0, -1), (0, 1)]
                .iter()
                .map(|(di, dj)| (n.0 + di, n.1 + dj))
                .filter(|c| c.0 >= 0 && c.0 < self.w && c.1 >= 0 && c.1 < self.h)
                .filter(|c| !self.blocked.contains(c))
                .map(|c| (c, c, 1.0))
                .collect()
        }
        fn is_goal(&self, n: &Self::Node) -> bool {
            *n == self.goal
        }
        fn heuristic(&self, n: &Self::Node) -> f64 {
            ((self.goal.0 - n.0).abs() + (self.goal.1 - n.1).abs()) as f64 // Manhattan (admissible)
        }
    }

    #[test]
    fn solve_drives_a_motion_grid_problem() {
        // A wall at column x=1 for rows y=0,1 (gap at y=2): a straight run to (3,0) is blocked, so
        // the shortest plan detours up and over.
        let prob = GridProblem {
            w: 4,
            h: 4,
            goal: (3, 0),
            blocked: vec![(1, 0), (1, 1)],
        };
        let res = solve(&prob, 1000);
        let path = res.actions.expect("goal reachable");
        assert_eq!(path.first(), Some(&(0, 1))); // straight is walled, so the first step is forced up
        assert_eq!(path.last(), Some(&(3, 0))); // ends at the goal
        assert!(!path.iter().any(|c| prob.blocked.contains(c))); // never steps into the wall
                                                                 // Manhattan distance 3, plus a +4 detour (up 2 to clear the two-cell wall, back down 2).
        assert_eq!(path.len(), 7);
    }
}
