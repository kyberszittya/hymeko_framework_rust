//! Solved small-game witness for AKOIRE + HyMeKo + HIVE.
//!
//! Run with:
//!     cargo run -p akoire --example solved_grid_game
//!
//! The example solves a deterministic grid game by value iteration, gate-keeps a
//! generated HyMeKo representation, then commits the solved transition system
//! and optimal policy to HIVE as a signed hypergraph.

use std::collections::BTreeMap;

use akoire::{EvalOutcome, HymekoEngine, Refinement};
use hymeko_hive::{
    AttributeValue, HiveDelta, HiveNode, HiveQuery, HiveRelation, HiveStore, HiveTransaction,
    NodeId, Sign,
};

const WIDTH: usize = 4;
const HEIGHT: usize = 4;
const GAMMA: f64 = 0.95;
const EPSILON: f64 = 1e-9;
const MAX_ITERS: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tile {
    Start,
    Empty,
    Wall,
    Goal,
    Pit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    const ALL: [Self; 4] = [Self::Up, Self::Down, Self::Left, Self::Right];

    fn name(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    fn mark(self) -> &'static str {
        match self {
            Self::Up => "U",
            Self::Down => "D",
            Self::Left => "L",
            Self::Right => "R",
        }
    }

    fn delta(self) -> (isize, isize) {
        match self {
            Self::Up => (0, -1),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        }
    }
}

#[derive(Debug, Clone)]
struct SolvedGame {
    values: BTreeMap<(usize, usize), f64>,
    policy: BTreeMap<(usize, usize), Action>,
    iterations: usize,
}

fn main() {
    let solved = solve_game();
    let hymeko_source = render_hymeko_source(&solved);

    let mut engine = HymekoEngine::new();
    match engine.evaluate(&Refinement(hymeko_source)) {
        EvalOutcome::Accepted { generation } => {
            println!("HyMeKo gate accepted solved game at ambience generation {generation}");
        }
        EvalOutcome::Rejected(feedback) => {
            println!("HyMeKo gate rejected generated game source");
            println!("{}", one_line(&feedback.message));
            return;
        }
    }

    let mut hive = HiveStore::new();
    let deltas = hive_deltas(&solved);
    hive.commit(HiveTransaction::new(
        "solved-grid-game",
        hive.state_hash(),
        "akoire-game-solver",
        0,
        deltas,
    ))
    .expect("solved game should commit to HIVE");

    let transitions = hive.query(&HiveQuery::RelationsByType("game_transition".to_string()));
    let policies = hive.query(&HiveQuery::RelationsByType("optimal_policy".to_string()));
    let valued = hive.query(&HiveQuery::NodesWithAttribute("value".to_string()));

    println!("\nSolved 4x4 escape game");
    println!("  iterations: {}", solved.iterations);
    println!("  HIVE generation: {}", hive.generation());
    println!("  HIVE hash: {}", hive.state_hash());
    println!("  transition relations: {}", transitions.relations.len());
    println!("  optimal policy relations: {}", policies.relations.len());
    println!("  valued state nodes: {}", valued.nodes.len());

    println!("\nPolicy map");
    println!("{}", render_policy_map(&solved));

    println!("\nValue map");
    println!("{}", render_value_map(&solved));

    println!("\nOptimal-policy Mermaid graph");
    println!("{}", render_mermaid_policy(&solved));
}

fn solve_game() -> SolvedGame {
    let mut values = playable_positions()
        .into_iter()
        .map(|pos| (pos, 0.0))
        .collect::<BTreeMap<_, _>>();

    let mut iterations = 0;
    for iter in 1..=MAX_ITERS {
        let mut next = values.clone();
        let mut delta: f64 = 0.0;

        for pos in playable_positions() {
            if is_terminal(pos) {
                next.insert(pos, 0.0);
                continue;
            }
            let best = Action::ALL
                .into_iter()
                .map(|action| action_value(pos, action, &values))
                .fold(f64::NEG_INFINITY, f64::max);
            let old = values[&pos];
            next.insert(pos, best);
            delta = delta.max((best - old).abs());
        }

        values = next;
        iterations = iter;
        if delta < EPSILON {
            break;
        }
    }

    let policy = playable_positions()
        .into_iter()
        .filter(|&pos| !is_terminal(pos))
        .map(|pos| {
            let action = Action::ALL
                .into_iter()
                .max_by(|&a, &b| {
                    action_value(pos, a, &values)
                        .partial_cmp(&action_value(pos, b, &values))
                        .unwrap()
                        .then_with(|| b.name().cmp(a.name()))
                })
                .unwrap();
            (pos, action)
        })
        .collect();

    SolvedGame {
        values,
        policy,
        iterations,
    }
}

fn action_value(
    pos: (usize, usize),
    action: Action,
    values: &BTreeMap<(usize, usize), f64>,
) -> f64 {
    let next = transition(pos, action);
    reward(next) + GAMMA * values.get(&next).copied().unwrap_or(0.0)
}

fn transition((x, y): (usize, usize), action: Action) -> (usize, usize) {
    if is_terminal((x, y)) {
        return (x, y);
    }
    let (dx, dy) = action.delta();
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    if nx < 0 || ny < 0 || nx >= WIDTH as isize || ny >= HEIGHT as isize {
        return (x, y);
    }
    let candidate = (nx as usize, ny as usize);
    if tile(candidate) == Tile::Wall {
        (x, y)
    } else {
        candidate
    }
}

fn reward(pos: (usize, usize)) -> f64 {
    match tile(pos) {
        Tile::Goal => 10.0,
        Tile::Pit => -10.0,
        _ => -0.25,
    }
}

fn tile((x, y): (usize, usize)) -> Tile {
    match (x, y) {
        (0, 0) => Tile::Start,
        (1, 1) | (1, 3) => Tile::Wall,
        (2, 1) => Tile::Pit,
        (3, 3) => Tile::Goal,
        _ => Tile::Empty,
    }
}

fn playable_positions() -> Vec<(usize, usize)> {
    (0..HEIGHT)
        .flat_map(|y| (0..WIDTH).map(move |x| (x, y)))
        .filter(|&pos| tile(pos) != Tile::Wall)
        .collect()
}

fn is_terminal(pos: (usize, usize)) -> bool {
    matches!(tile(pos), Tile::Goal | Tile::Pit)
}

fn state_id((x, y): (usize, usize)) -> String {
    format!("c{x}_{y}")
}

fn action_id(action: Action) -> String {
    format!("a_{}", action.name())
}

fn render_hymeko_source(solved: &SolvedGame) -> String {
    let mut out = String::from("SolvedGridGame {\n");
    for pos in playable_positions() {
        out.push_str("  ");
        out.push_str(&state_id(pos));
        out.push_str(";\n");
    }
    for action in Action::ALL {
        out.push_str("  ");
        out.push_str(&action_id(action));
        out.push_str(";\n");
    }
    out.push_str("}\n");

    for pos in playable_positions() {
        if is_terminal(pos) {
            continue;
        }
        for action in Action::ALL {
            let next = transition(pos, action);
            out.push_str(&format!(
                "@t_{}_{} : {}, {}, {} {{ }}\n",
                state_id(pos),
                action.name(),
                state_id(pos),
                action_id(action),
                state_id(next)
            ));
        }
        let policy = solved.policy[&pos];
        out.push_str(&format!(
            "@pi_{} : {}, {} {{ }}\n",
            state_id(pos),
            state_id(pos),
            action_id(policy)
        ));
    }

    out
}

fn hive_deltas(solved: &SolvedGame) -> Vec<HiveDelta> {
    let mut deltas = Vec::new();

    for pos in playable_positions() {
        let mut node = HiveNode::new(state_id(pos), "game_state");
        node.attributes
            .insert("x".to_string(), AttributeValue::Int(pos.0 as i64));
        node.attributes
            .insert("y".to_string(), AttributeValue::Int(pos.1 as i64));
        node.attributes.insert(
            "tile".to_string(),
            AttributeValue::Text(tile_name(tile(pos)).to_string()),
        );
        node.attributes.insert(
            "terminal".to_string(),
            AttributeValue::Bool(is_terminal(pos)),
        );
        node.attributes.insert(
            "value".to_string(),
            AttributeValue::Float(solved.values[&pos]),
        );
        deltas.push(HiveDelta::AddNode(node));
    }

    for action in Action::ALL {
        deltas.push(HiveDelta::AddNode(HiveNode::new(
            action_id(action),
            "game_action",
        )));
    }

    for pos in playable_positions() {
        if is_terminal(pos) {
            continue;
        }
        for action in Action::ALL {
            let next = transition(pos, action);
            let mut relation = HiveRelation::new(
                format!("t_{}_{}", state_id(pos), action.name()),
                "game_transition",
                vec![
                    (Sign::Minus, NodeId(state_id(pos))),
                    (Sign::Neutral, NodeId(action_id(action))),
                    (Sign::Plus, NodeId(state_id(next))),
                ],
            );
            relation
                .attributes
                .insert("reward".to_string(), AttributeValue::Float(reward(next)));
            relation.attributes.insert(
                "q_value".to_string(),
                AttributeValue::Float(action_value(pos, action, &solved.values)),
            );
            deltas.push(HiveDelta::AddRelation(relation));
        }

        let policy = solved.policy[&pos];
        let mut relation = HiveRelation::new(
            format!("pi_{}", state_id(pos)),
            "optimal_policy",
            vec![
                (Sign::Minus, NodeId(state_id(pos))),
                (Sign::Plus, NodeId(action_id(policy))),
            ],
        );
        relation.attributes.insert(
            "state_value".to_string(),
            AttributeValue::Float(solved.values[&pos]),
        );
        deltas.push(HiveDelta::AddRelation(relation));
    }

    deltas
}

fn tile_name(tile: Tile) -> &'static str {
    match tile {
        Tile::Start => "start",
        Tile::Empty => "empty",
        Tile::Wall => "wall",
        Tile::Goal => "goal",
        Tile::Pit => "pit",
    }
}

fn render_policy_map(solved: &SolvedGame) -> String {
    (0..HEIGHT)
        .map(|y| {
            (0..WIDTH)
                .map(|x| match tile((x, y)) {
                    Tile::Wall => " # ".to_string(),
                    Tile::Goal => " G ".to_string(),
                    Tile::Pit => " X ".to_string(),
                    Tile::Start | Tile::Empty => {
                        format!(" {} ", solved.policy[&(x, y)].mark())
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_value_map(solved: &SolvedGame) -> String {
    (0..HEIGHT)
        .map(|y| {
            (0..WIDTH)
                .map(|x| {
                    if tile((x, y)) == Tile::Wall {
                        "  #### ".to_string()
                    } else {
                        format!("{:>6.2} ", solved.values[&(x, y)])
                    }
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_mermaid_policy(solved: &SolvedGame) -> String {
    let mut out = String::from("flowchart LR\n");
    for pos in playable_positions() {
        let label = match tile(pos) {
            Tile::Start => format!("S\\n{:.2}", solved.values[&pos]),
            Tile::Goal => "G".to_string(),
            Tile::Pit => "X".to_string(),
            Tile::Empty => format!("{}\\n{:.2}", state_id(pos), solved.values[&pos]),
            Tile::Wall => unreachable!(),
        };
        out.push_str(&format!("  {}[\"{}\"]\n", state_id(pos), label));
    }
    for (&pos, &action) in &solved.policy {
        let next = transition(pos, action);
        out.push_str(&format!(
            "  {} -->|{} r={:+.2}| {}\n",
            state_id(pos),
            action.name(),
            reward(next),
            state_id(next)
        ));
    }
    out
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
