//! Solved reinforcement-learning witnesses for the browser WASM bundle.
//!
//! This module is deliberately native-testable: it solves deterministic worlds,
//! renders a HyMeKo source representation, gates that source through the same
//! compile path the editor uses, and returns a JSON DTO for JavaScript.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::compile::compile_source;

const GAMMA: f64 = 0.95;
const EPSILON: f64 = 1e-9;
const MAX_ITERS: usize = 10_000;
const MAX_SQUARE_DIM: i32 = 16;
const MAX_HEX_RADIUS: i32 = 6;

type Pos = (i32, i32);

#[derive(Debug, Clone)]
struct SeedRng {
    state: u64,
}

impl SeedRng {
    fn new(seed: u32) -> Self {
        Self {
            state: u64::from(seed).wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 32) as u32
    }

    fn index(&mut self, len: usize) -> usize {
        if len == 0 {
            0
        } else {
            self.next_u32() as usize % len
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Topology {
    Square,
    Hex,
}

impl Topology {
    fn name(self) -> &'static str {
        match self {
            Self::Square => "square",
            Self::Hex => "hex",
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "hex" | "hexworld" | "hexaworld" => Self::Hex,
            _ => Self::Square,
        }
    }
}

#[derive(Debug, Clone)]
struct WorldSpec {
    id: String,
    label: String,
    topology: Topology,
    width: i32,
    height: i32,
    id_prefix: &'static str,
    start: Pos,
    display_positions: Vec<Pos>,
    walls: Vec<Pos>,
    goals: Vec<Pos>,
    pits: Vec<Pos>,
}

impl WorldSpec {
    fn grid4() -> Self {
        let width = 4;
        let height = 4;
        Self {
            id: "grid4".to_string(),
            label: "4x4 square grid".to_string(),
            topology: Topology::Square,
            width,
            height,
            id_prefix: "c",
            start: (0, 0),
            display_positions: square_positions(width, height),
            walls: vec![(1, 1), (1, 3)],
            goals: vec![(3, 3)],
            pits: vec![(2, 1)],
        }
    }

    fn grid6() -> Self {
        let width = 6;
        let height = 6;
        Self {
            id: "grid6".to_string(),
            label: "6x6 larger grid".to_string(),
            topology: Topology::Square,
            width,
            height,
            id_prefix: "c",
            start: (0, 0),
            display_positions: square_positions(width, height),
            walls: vec![(1, 1), (1, 2), (3, 1), (3, 2), (3, 3), (2, 4)],
            goals: vec![(5, 5)],
            pits: vec![(4, 2), (2, 5)],
        }
    }

    fn hex() -> Self {
        let radius = 3;
        Self {
            id: "hex".to_string(),
            label: "hexaworld radius 3".to_string(),
            topology: Topology::Hex,
            width: radius * 2 + 1,
            height: radius * 2 + 1,
            id_prefix: "h",
            start: (-3, 0),
            display_positions: hex_positions(radius),
            walls: vec![(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)],
            goals: vec![(3, 0)],
            pits: vec![(-1, 2), (1, -2), (2, -1)],
        }
    }

    fn generated(topology: &str, width: i32, height: i32) -> Self {
        match Topology::from_name(topology) {
            Topology::Square => Self::generated_square(width, height),
            Topology::Hex => Self::generated_hex(width.max(height)),
        }
    }

    fn randomized_start(mut self, seed: u32) -> Self {
        let goals = self.goals.clone();
        let mut candidates = self
            .playable_positions()
            .into_iter()
            .filter(|&pos| !self.is_terminal(pos))
            .filter(|&pos| goals.iter().all(|&goal| manhattan(pos, goal) >= 2))
            .collect::<Vec<_>>();
        candidates.sort_unstable();
        if !candidates.is_empty() {
            let mut rng = SeedRng::new(seed);
            self.start = candidates[rng.index(candidates.len())];
            self.id = format!("{}_seed{}", self.id, seed % 10_000);
            self.label = format!("{} random start", self.label);
        }
        self
    }

    fn generated_square(width: i32, height: i32) -> Self {
        let width = width.clamp(2, MAX_SQUARE_DIM);
        let height = height.clamp(2, MAX_SQUARE_DIM);
        let start = (0, 0);
        let goal = (width - 1, height - 1);
        let mut walls = Vec::new();
        let mut pits = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let pos = (x, y);
                let protected = pos == start || pos == goal || y == 0 || x == width - 1;
                if protected {
                    continue;
                }
                let hash = (x * 31 + y * 17 + width * 7 + height * 11).rem_euclid(23);
                if hash == 0 || hash == 5 {
                    walls.push(pos);
                } else if hash == 9 {
                    pits.push(pos);
                }
            }
        }

        Self {
            id: format!("custom_grid_{width}x{height}"),
            label: format!("custom {width}x{height} square grid"),
            topology: Topology::Square,
            width,
            height,
            id_prefix: "c",
            start,
            display_positions: square_positions(width, height),
            walls,
            goals: vec![goal],
            pits,
        }
    }

    fn generated_hex(radius: i32) -> Self {
        let radius = radius.clamp(1, MAX_HEX_RADIUS);
        let start = (-radius, 0);
        let goal = (radius, 0);
        let mut walls = Vec::new();
        let mut pits = Vec::new();

        for pos in hex_positions(radius) {
            let (q, r) = pos;
            let protected = pos == start || pos == goal || r == 0;
            if protected {
                continue;
            }
            let hash = (q * 29 + r * 19 + radius * 13).rem_euclid(31);
            if hash == 0 || hash == 7 {
                walls.push(pos);
            } else if hash == 11 {
                pits.push(pos);
            }
        }

        Self {
            id: format!("custom_hex_r{radius}"),
            label: format!("custom hexaworld radius {radius}"),
            topology: Topology::Hex,
            width: radius * 2 + 1,
            height: radius * 2 + 1,
            id_prefix: "h",
            start,
            display_positions: hex_positions(radius),
            walls,
            goals: vec![goal],
            pits,
        }
    }

    fn from_id(id: &str) -> Self {
        match id {
            "grid6" | "large" | "larger" => Self::grid6(),
            "hex" | "hexworld" | "hexaworld" => Self::hex(),
            _ => Self::grid4(),
        }
    }

    fn tile(&self, pos: Pos) -> Tile {
        if self.start == pos {
            Tile::Start
        } else if self.walls.contains(&pos) {
            Tile::Wall
        } else if self.goals.contains(&pos) {
            Tile::Goal
        } else if self.pits.contains(&pos) {
            Tile::Pit
        } else {
            Tile::Empty
        }
    }

    fn contains(&self, pos: Pos) -> bool {
        self.display_positions.contains(&pos)
    }

    fn playable_positions(&self) -> Vec<Pos> {
        self.display_positions
            .iter()
            .copied()
            .filter(|&pos| self.tile(pos) != Tile::Wall)
            .collect()
    }

    fn is_terminal(&self, pos: Pos) -> bool {
        matches!(self.tile(pos), Tile::Goal | Tile::Pit)
    }

    fn actions(&self) -> &'static [Action] {
        match self.topology {
            Topology::Square => &Action::SQUARE,
            Topology::Hex => &Action::HEX,
        }
    }

    fn state_id(&self, pos: Pos) -> String {
        format!(
            "{}{}_{}",
            self.id_prefix,
            coord_token(pos.0),
            coord_token(pos.1)
        )
    }

    fn observation_id(&self, pos: Pos) -> String {
        format!("o_{}", self.state_id(pos))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Tile {
    Start,
    Empty,
    Wall,
    Goal,
    Pit,
    Rival,
    Automaton,
    Virus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum Action {
    Up,
    Down,
    Left,
    Right,
    East,
    West,
    NorthEast,
    NorthWest,
    SouthEast,
    SouthWest,
}

impl Action {
    const SQUARE: [Self; 4] = [Self::Up, Self::Down, Self::Left, Self::Right];
    const HEX: [Self; 6] = [
        Self::East,
        Self::West,
        Self::NorthEast,
        Self::NorthWest,
        Self::SouthEast,
        Self::SouthWest,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Left => "left",
            Self::Right => "right",
            Self::East => "east",
            Self::West => "west",
            Self::NorthEast => "north_east",
            Self::NorthWest => "north_west",
            Self::SouthEast => "south_east",
            Self::SouthWest => "south_west",
        }
    }

    fn glyph(self) -> &'static str {
        match self {
            Self::Up => "U",
            Self::Down => "D",
            Self::Left => "L",
            Self::Right => "R",
            Self::East => "E",
            Self::West => "W",
            Self::NorthEast => "NE",
            Self::NorthWest => "NW",
            Self::SouthEast => "SE",
            Self::SouthWest => "SW",
        }
    }

    fn delta(self) -> Pos {
        match self {
            Self::Up => (0, -1),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
            Self::East => (1, 0),
            Self::West => (-1, 0),
            Self::NorthEast => (1, -1),
            Self::NorthWest => (0, -1),
            Self::SouthEast => (0, 1),
            Self::SouthWest => (-1, 1),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GameDto {
    world_id: String,
    world_label: String,
    topology: &'static str,
    width: i32,
    height: i32,
    start_state: String,
    gamma: f64,
    iterations: usize,
    gate: GateDto,
    counts: CountDto,
    concepts: Vec<ConceptDto>,
    cells: Vec<CellDto>,
    transitions: Vec<TransitionDto>,
    policy_edges: Vec<PolicyEdgeDto>,
    trace: Vec<TraceStepDto>,
    evolution: Vec<EvolutionDto>,
    hymeko_source: String,
    mermaid: String,
}

#[derive(Debug, Clone, Serialize)]
struct GateDto {
    accepted: bool,
    node_count: usize,
    edge_count: usize,
    arc_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CountDto {
    playable_states: usize,
    action_edges: usize,
    transition_edges: usize,
    policy_edges: usize,
    concept_edges: usize,
    simulation_steps: usize,
}

#[derive(Debug, Clone, Serialize)]
struct ConceptDto {
    id: &'static str,
    label: &'static str,
    role: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct CellDto {
    id: String,
    x: i32,
    y: i32,
    q: i32,
    r: i32,
    tile: Tile,
    value: f64,
    policy: Option<ActionDto>,
}

#[derive(Debug, Clone, Serialize)]
struct ActionDto {
    name: &'static str,
    glyph: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct TransitionDto {
    from: String,
    action: &'static str,
    to: String,
    reward: f64,
    q_value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PolicyEdgeDto {
    from: String,
    action: &'static str,
    to: String,
    reward: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TraceStepDto {
    step: usize,
    state: String,
    observation: String,
    action: &'static str,
    next_state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    rival_state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rival_action: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automaton_generation: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    automaton_cells: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    virus_cells: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cleansed_cells: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    infection_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    victory: Option<bool>,
    reward: f64,
    cumulative_reward: f64,
    value: f64,
    terminal: bool,
}

#[derive(Debug, Clone)]
struct SolvedGame {
    world: WorldSpec,
    values: BTreeMap<Pos, f64>,
    policy: BTreeMap<Pos, Action>,
    iterations: usize,
    evolution: Vec<EvolutionStage>,
}

#[derive(Debug, Clone)]
struct EvolutionStage {
    iteration: usize,
    label: String,
    values: BTreeMap<Pos, f64>,
    policy: BTreeMap<Pos, Action>,
    source: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvolutionDto {
    iteration: usize,
    label: String,
    changed_states: usize,
    cells: Vec<CellDto>,
    policy_edges: Vec<PolicyEdgeDto>,
    hymeko_source: String,
}

/// Solve, gate, and render the original small square game as JSON.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn solved_grid_game_json() -> Result<String, String> {
    solved_world_json("grid4")
}

/// Solve, gate, and render one of the available demo worlds as JSON.
///
/// Accepted ids are `grid4`, `grid6`, and `hex`. Unknown ids intentionally
/// fall back to the stable 4x4 witness so older browser demos keep working.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn solved_world_json(world_id: &str) -> Result<String, String> {
    let world = WorldSpec::from_id(world_id);
    world_json(world)
}

/// Solve a preset world after choosing a seeded random EGO start position.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn randomized_world_json(world_id: &str, seed: u32) -> Result<String, String> {
    let world = WorldSpec::from_id(world_id).randomized_start(seed);
    world_json(world)
}

/// Solve, gate, and render a deterministic generated world as JSON.
///
/// Square worlds use `width` and `height`; hex worlds use the larger dimension
/// as their radius. Dimensions are clamped to keep the browser demo responsive.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn generated_world_json(topology: &str, width: i32, height: i32) -> Result<String, String> {
    let world = WorldSpec::generated(topology, width, height);
    world_json(world)
}

/// Solve a generated world after choosing a seeded random EGO start position.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn generated_random_world_json(
    topology: &str,
    width: i32,
    height: i32,
    seed: u32,
) -> Result<String, String> {
    let world = WorldSpec::generated(topology, width, height).randomized_start(seed);
    world_json(world)
}

/// Solve and render a competitive Hikari-vs-Kage world with an evolving
/// cellular-automaton hazard layer.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn adversarial_world_json() -> Result<String, String> {
    let scenario = AdversarialScenario::new();
    adversarial_json(scenario)
}

/// Solve and render a seeded harder adversarial arena.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn adversarial_world_seed_json(seed: u32) -> Result<String, String> {
    let scenario = AdversarialScenario::seeded(seed);
    adversarial_json(scenario)
}

/// Solve and render the virus cellular-automaton gameplay arena.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn virus_world_json() -> Result<String, String> {
    virus_world_seed_json(0)
}

/// Solve and render a seeded virus cellular-automaton gameplay arena.
///
/// # Errors
/// Returns a string if the generated HyMeKo source unexpectedly fails to
/// compile or the DTO cannot serialize.
pub fn virus_world_seed_json(seed: u32) -> Result<String, String> {
    let scenario = VirusScenario::seeded(seed);
    virus_json(scenario)
}

fn adversarial_json(scenario: AdversarialScenario) -> Result<String, String> {
    let policy = adversarial_policy(&scenario);
    let values = adversarial_values(&scenario);
    let trace = simulate_adversarial(&scenario);
    let hymeko_source = render_adversarial_hymeko_source(&scenario, &policy, &values, &trace);
    let compiled = compile_source(&hymeko_source)?;
    let transitions = adversarial_transitions(&scenario, &values);
    let policy_edges = policy_edges_from(&scenario.world, &policy);
    let evolution = adversarial_evolution(&scenario, &values, &policy, &trace, &hymeko_source);
    let dto = GameDto {
        world_id: scenario.world.id.clone(),
        world_label: scenario.world.label.clone(),
        topology: scenario.world.topology.name(),
        width: scenario.world.width,
        height: scenario.world.height,
        start_state: scenario.world.state_id(scenario.world.start),
        gamma: GAMMA,
        iterations: trace.len(),
        gate: GateDto {
            accepted: true,
            node_count: compiled.node_count(),
            edge_count: compiled.edge_count(),
            arc_count: compiled.arc_count(),
        },
        counts: CountDto {
            playable_states: scenario.world.playable_positions().len(),
            action_edges: transitions.len(),
            transition_edges: transitions.len(),
            policy_edges: policy_edges.len(),
            concept_edges: concept_edge_count(&scenario.world) + 8,
            simulation_steps: trace.len(),
        },
        concepts: adversarial_concepts(),
        cells: adversarial_cells(&scenario, &values, &policy),
        transitions,
        policy_edges,
        trace,
        evolution,
        hymeko_source,
        mermaid: render_adversarial_mermaid(&scenario, &policy, &values),
    };
    serde_json::to_string_pretty(&dto).map_err(|e| e.to_string())
}

fn world_json(world: WorldSpec) -> Result<String, String> {
    let solved = solve_game(world);
    let hymeko_source = render_hymeko_source(&solved);
    let compiled = compile_source(&hymeko_source)?;
    let dto = to_dto(
        &solved,
        hymeko_source,
        GateDto {
            accepted: true,
            node_count: compiled.node_count(),
            edge_count: compiled.edge_count(),
            arc_count: compiled.arc_count(),
        },
    );
    serde_json::to_string_pretty(&dto).map_err(|e| e.to_string())
}

fn virus_json(scenario: VirusScenario) -> Result<String, String> {
    let policy = virus_policy(&scenario);
    let values = virus_values(&scenario);
    let trace = simulate_virus(&scenario);
    let hymeko_source = render_virus_hymeko_source(&scenario, &policy, &values, &trace);
    let compiled = compile_source(&hymeko_source)?;
    let transitions = virus_transitions(&scenario, &values);
    let policy_edges = policy_edges_from(&scenario.world, &policy);
    let evolution = virus_evolution(&scenario, &values, &policy, &trace, &hymeko_source);
    let dto = GameDto {
        world_id: scenario.world.id.clone(),
        world_label: scenario.world.label.clone(),
        topology: scenario.world.topology.name(),
        width: scenario.world.width,
        height: scenario.world.height,
        start_state: scenario.world.state_id(scenario.world.start),
        gamma: GAMMA,
        iterations: trace.len(),
        gate: GateDto {
            accepted: true,
            node_count: compiled.node_count(),
            edge_count: compiled.edge_count(),
            arc_count: compiled.arc_count(),
        },
        counts: CountDto {
            playable_states: scenario.world.playable_positions().len(),
            action_edges: transitions.len(),
            transition_edges: transitions.len(),
            policy_edges: policy_edges.len(),
            concept_edges: concept_edge_count(&scenario.world) + 10,
            simulation_steps: trace.len(),
        },
        concepts: virus_concepts(),
        cells: virus_cells(&scenario, &values, &policy, &scenario.virus_start),
        transitions,
        policy_edges,
        trace,
        evolution,
        hymeko_source,
        mermaid: render_virus_mermaid(&scenario, &policy, &values),
    };
    serde_json::to_string_pretty(&dto).map_err(|e| e.to_string())
}

#[derive(Debug, Clone)]
struct AdversarialScenario {
    world: WorldSpec,
    rival_start: Pos,
    automaton_start: BTreeSet<Pos>,
}

impl AdversarialScenario {
    fn new() -> Self {
        let width = 7;
        let height = 7;
        Self {
            world: WorldSpec {
                id: "adversarial_ca".to_string(),
                label: "adversarial arena with cellular automaton".to_string(),
                topology: Topology::Square,
                width,
                height,
                id_prefix: "c",
                start: (0, 6),
                display_positions: square_positions(width, height),
                walls: vec![(2, 2), (2, 3), (2, 4), (4, 1), (4, 2), (4, 5)],
                goals: vec![(6, 6)],
                pits: vec![(5, 3)],
            },
            rival_start: (0, 0),
            automaton_start: [(3, 1), (3, 3)].into_iter().collect(),
        }
    }

    fn seeded(seed: u32) -> Self {
        let mut scenario = Self::new();
        let mut rng = SeedRng::new(seed);
        let goal = scenario.world.goals[0];
        let mut candidates = scenario
            .world
            .playable_positions()
            .into_iter()
            .filter(|&pos| !scenario.world.is_terminal(pos))
            .filter(|&pos| manhattan(pos, goal) >= 4)
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|&pos| (manhattan(pos, goal), pos.1, pos.0));
        if !candidates.is_empty() {
            scenario.world.start = candidates[rng.index(candidates.len())];
        }

        let mut rival_candidates = scenario
            .world
            .playable_positions()
            .into_iter()
            .filter(|&pos| pos != scenario.world.start)
            .filter(|&pos| !scenario.world.is_terminal(pos))
            .filter(|&pos| manhattan(pos, scenario.world.start) <= 4)
            .collect::<Vec<_>>();
        rival_candidates.sort_unstable_by_key(|&pos| (manhattan(pos, scenario.world.start), pos.0));
        if !rival_candidates.is_empty() {
            scenario.rival_start = rival_candidates[rng.index(rival_candidates.len().min(8))];
        }

        let mut automaton_candidates = scenario
            .world
            .playable_positions()
            .into_iter()
            .filter(|&pos| pos != scenario.world.start && pos != scenario.rival_start)
            .filter(|&pos| !scenario.world.is_terminal(pos))
            .filter(|&pos| manhattan(pos, scenario.world.start) > 1 && manhattan(pos, goal) > 1)
            .collect::<Vec<_>>();
        automaton_candidates.sort_unstable_by_key(|&pos| {
            (
                manhattan(pos, scenario.world.start) + manhattan(pos, goal),
                pos.1,
                pos.0,
            )
        });
        let mut automata = BTreeSet::new();
        let automaton_count = 3 + (rng.index(3));
        for _ in 0..automaton_count {
            if automaton_candidates.is_empty() {
                break;
            }
            let index = rng.index(automaton_candidates.len());
            automata.insert(automaton_candidates.remove(index));
        }
        if !automata.is_empty() {
            scenario.automaton_start = automata;
        }
        scenario.world.id = format!("adversarial_ca_seed{}", seed % 10_000);
        scenario.world.label = "randomized adversarial arena with cellular automaton".to_string();
        scenario
    }
}

#[derive(Debug, Clone)]
struct VirusScenario {
    world: WorldSpec,
    virus_start: BTreeSet<Pos>,
}

impl VirusScenario {
    fn seeded(seed: u32) -> Self {
        let width = 8;
        let height = 8;
        let mut rng = SeedRng::new(seed);
        let start_candidates = [(0, 7), (1, 7), (0, 6), (2, 7), (1, 6)];
        let start = start_candidates[rng.index(start_candidates.len())];
        let mut world = WorldSpec {
            id: if seed == 0 {
                "virus_ca".to_string()
            } else {
                format!("virus_ca_seed{}", seed % 10_000)
            },
            label: if seed == 0 {
                "virus cellular-automaton arena".to_string()
            } else {
                "randomized virus cellular-automaton arena".to_string()
            },
            topology: Topology::Square,
            width,
            height,
            id_prefix: "c",
            start,
            display_positions: square_positions(width, height),
            walls: vec![(3, 1), (3, 2), (3, 3), (5, 4), (5, 5), (2, 5)],
            goals: vec![(7, 0)],
            pits: vec![(6, 6)],
        };
        if world.walls.contains(&world.start) {
            world.start = (0, 7);
        }

        let mut candidates = world
            .playable_positions()
            .into_iter()
            .filter(|&pos| pos != world.start && !world.is_terminal(pos))
            .filter(|&pos| manhattan(pos, world.start) >= 3 && manhattan(pos, world.goals[0]) >= 2)
            .collect::<Vec<_>>();
        candidates.sort_unstable_by_key(|&pos| (pos.1, pos.0));
        let mut virus_start = BTreeSet::new();
        for seed_cell in [(4, 2), (4, 3), (5, 2), (6, 3)] {
            if world.contains(seed_cell) && world.tile(seed_cell) != Tile::Wall {
                virus_start.insert(seed_cell);
            }
        }
        if seed != 0 {
            virus_start.clear();
            for _ in 0..5 {
                if candidates.is_empty() {
                    break;
                }
                let index = rng.index(candidates.len());
                virus_start.insert(candidates.remove(index));
            }
        }

        Self { world, virus_start }
    }
}

fn virus_policy(scenario: &VirusScenario) -> BTreeMap<Pos, Action> {
    scenario
        .world
        .playable_positions()
        .into_iter()
        .filter(|&pos| !scenario.world.is_terminal(pos))
        .map(|pos| {
            let action = Action::SQUARE
                .into_iter()
                .max_by(|&a, &b| {
                    virus_action_score(&scenario.world, pos, a, &scenario.virus_start)
                        .partial_cmp(&virus_action_score(
                            &scenario.world,
                            pos,
                            b,
                            &scenario.virus_start,
                        ))
                        .unwrap()
                        .then_with(|| b.name().cmp(a.name()))
                })
                .unwrap();
            (pos, action)
        })
        .collect()
}

fn virus_values(scenario: &VirusScenario) -> BTreeMap<Pos, f64> {
    scenario
        .world
        .playable_positions()
        .into_iter()
        .map(|pos| {
            (
                pos,
                virus_state_score(&scenario.world, pos, &scenario.virus_start),
            )
        })
        .collect()
}

fn virus_cells(
    scenario: &VirusScenario,
    values: &BTreeMap<Pos, f64>,
    policy: &BTreeMap<Pos, Action>,
    viruses: &BTreeSet<Pos>,
) -> Vec<CellDto> {
    scenario
        .world
        .display_positions
        .iter()
        .copied()
        .map(|pos| {
            let tile = if viruses.contains(&pos) {
                Tile::Virus
            } else {
                scenario.world.tile(pos)
            };
            CellDto {
                id: scenario.world.state_id(pos),
                x: pos.0,
                y: pos.1,
                q: pos.0,
                r: pos.1,
                tile,
                value: values.get(&pos).copied().unwrap_or(0.0),
                policy: policy.get(&pos).map(|action| ActionDto {
                    name: action.name(),
                    glyph: action.glyph(),
                }),
            }
        })
        .collect()
}

fn virus_transitions(scenario: &VirusScenario, values: &BTreeMap<Pos, f64>) -> Vec<TransitionDto> {
    scenario
        .world
        .playable_positions()
        .into_iter()
        .filter(|&pos| !scenario.world.is_terminal(pos))
        .flat_map(|pos| {
            Action::SQUARE.into_iter().map(move |action| {
                let next = transition(&scenario.world, pos, action);
                TransitionDto {
                    from: scenario.world.state_id(pos),
                    action: action.name(),
                    to: scenario.world.state_id(next),
                    reward: virus_reward(&scenario.world, next, &scenario.virus_start),
                    q_value: virus_action_score(
                        &scenario.world,
                        pos,
                        action,
                        &scenario.virus_start,
                    ) + values.get(&next).copied().unwrap_or(0.0) * 0.1,
                }
            })
        })
        .collect()
}

fn simulate_virus(scenario: &VirusScenario) -> Vec<TraceStepDto> {
    let mut out = Vec::new();
    let mut pos = scenario.world.start;
    let mut viruses = scenario.virus_start.clone();
    let mut cumulative_reward = 0.0;

    for step in 0..28 {
        if scenario.world.is_terminal(pos) {
            break;
        }
        let action = best_virus_action(&scenario.world, pos, &viruses);
        let next = transition(&scenario.world, pos, action);
        let cleansed = cleanse_virus(&scenario.world, next, &mut viruses);
        viruses = evolve_virus(&scenario.world, &viruses, next, step + 1);
        let infected = viruses.contains(&next);
        let victory = next == scenario.world.goals[0] && viruses.len() <= 14;
        let terminal = victory || infected || viruses.len() >= 34;
        let reward = if victory {
            25.0
        } else if infected || viruses.len() >= 34 {
            -18.0
        } else {
            virus_reward(&scenario.world, next, &viruses) + cleansed.len() as f64 * 0.9
        };
        cumulative_reward += reward;
        out.push(TraceStepDto {
            step,
            state: scenario.world.state_id(pos),
            observation: scenario.world.observation_id(pos),
            action: action.name(),
            next_state: scenario.world.state_id(next),
            rival_state: None,
            rival_action: None,
            automaton_generation: Some(step + 1),
            automaton_cells: None,
            virus_cells: Some(
                viruses
                    .iter()
                    .copied()
                    .map(|cell| scenario.world.state_id(cell))
                    .collect(),
            ),
            cleansed_cells: Some(
                cleansed
                    .iter()
                    .copied()
                    .map(|cell| scenario.world.state_id(cell))
                    .collect(),
            ),
            infection_count: Some(viruses.len()),
            victory: Some(victory),
            reward,
            cumulative_reward,
            value: virus_state_score(&scenario.world, pos, &viruses),
            terminal,
        });
        pos = next;
        if terminal {
            break;
        }
    }

    out
}

fn best_virus_action(world: &WorldSpec, pos: Pos, viruses: &BTreeSet<Pos>) -> Action {
    Action::SQUARE
        .into_iter()
        .max_by(|&a, &b| {
            virus_action_score(world, pos, a, viruses)
                .partial_cmp(&virus_action_score(world, pos, b, viruses))
                .unwrap()
                .then_with(|| b.name().cmp(a.name()))
        })
        .unwrap()
}

fn virus_action_score(world: &WorldSpec, pos: Pos, action: Action, viruses: &BTreeSet<Pos>) -> f64 {
    let next = transition(world, pos, action);
    let mut after_cleanse = viruses.clone();
    let cleansed = cleanse_virus(world, next, &mut after_cleanse).len() as f64;
    virus_state_score(world, next, &after_cleanse) + cleansed * 1.4
}

fn virus_state_score(world: &WorldSpec, pos: Pos, viruses: &BTreeSet<Pos>) -> f64 {
    let goal = world.goals[0];
    let mut score = 24.0 - manhattan(pos, goal) as f64 * 1.9;
    let nearest = viruses
        .iter()
        .map(|&cell| manhattan(pos, cell))
        .min()
        .unwrap_or(8);
    score += nearest.min(6) as f64 * 0.45;
    if viruses.contains(&pos) {
        score -= 35.0;
    }
    if nearest == 1 {
        score += 1.2;
    }
    if world.tile(pos) == Tile::Goal {
        score += 40.0;
    }
    score -= viruses.len() as f64 * 0.08;
    score
}

fn virus_reward(world: &WorldSpec, pos: Pos, viruses: &BTreeSet<Pos>) -> f64 {
    if world.tile(pos) == Tile::Goal {
        15.0
    } else if viruses.contains(&pos) || world.tile(pos) == Tile::Pit {
        -12.0
    } else {
        -0.2
    }
}

fn cleanse_virus(world: &WorldSpec, center: Pos, viruses: &mut BTreeSet<Pos>) -> Vec<Pos> {
    let mut cleansed = Vec::new();
    for candidate in virus_neighborhood(center).into_iter().chain([center]) {
        if world.contains(candidate) && viruses.remove(&candidate) {
            cleansed.push(candidate);
        }
    }
    cleansed
}

fn evolve_virus(
    world: &WorldSpec,
    viruses: &BTreeSet<Pos>,
    ego: Pos,
    generation: usize,
) -> BTreeSet<Pos> {
    let mut next = BTreeSet::new();
    for pos in world.playable_positions() {
        if pos == ego || pos == world.start || pos == world.goals[0] || world.tile(pos) == Tile::Pit
        {
            continue;
        }
        let neighbors = virus_neighborhood(pos)
            .into_iter()
            .filter(|cell| viruses.contains(cell))
            .count();
        let alive = viruses.contains(&pos);
        let birth_gate = (pos.0 + pos.1 + generation as i32).rem_euclid(2) == 0;
        if (alive && (1..=3).contains(&neighbors)) || (!alive && neighbors >= 2 && birth_gate) {
            next.insert(pos);
        }
    }
    next
}

fn virus_neighborhood(pos: Pos) -> Vec<Pos> {
    let mut out = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            out.push((pos.0 + dx, pos.1 + dy));
        }
    }
    out
}

fn adversarial_policy(scenario: &AdversarialScenario) -> BTreeMap<Pos, Action> {
    let automata = &scenario.automaton_start;
    scenario
        .world
        .playable_positions()
        .into_iter()
        .filter(|&pos| !scenario.world.is_terminal(pos))
        .map(|pos| {
            let action = Action::SQUARE
                .into_iter()
                .max_by(|&a, &b| {
                    adversarial_action_score(
                        &scenario.world,
                        pos,
                        a,
                        scenario.rival_start,
                        automata,
                    )
                    .partial_cmp(&adversarial_action_score(
                        &scenario.world,
                        pos,
                        b,
                        scenario.rival_start,
                        automata,
                    ))
                    .unwrap()
                    .then_with(|| b.name().cmp(a.name()))
                })
                .unwrap();
            (pos, action)
        })
        .collect()
}

fn adversarial_values(scenario: &AdversarialScenario) -> BTreeMap<Pos, f64> {
    let goal = scenario.world.goals[0];
    scenario
        .world
        .playable_positions()
        .into_iter()
        .map(|pos| {
            let goal_pull = 12.0 - manhattan(pos, goal) as f64;
            let rival_pressure = 0.25 * manhattan(pos, scenario.rival_start) as f64;
            let automaton_pressure = scenario
                .automaton_start
                .iter()
                .map(|&cell| manhattan(pos, cell))
                .min()
                .unwrap_or(8) as f64
                * 0.15;
            let value = if scenario.world.is_terminal(pos) {
                0.0
            } else {
                goal_pull + rival_pressure + automaton_pressure
            };
            (pos, value)
        })
        .collect()
}

fn adversarial_cells(
    scenario: &AdversarialScenario,
    values: &BTreeMap<Pos, f64>,
    policy: &BTreeMap<Pos, Action>,
) -> Vec<CellDto> {
    scenario
        .world
        .display_positions
        .iter()
        .copied()
        .map(|pos| {
            let tile = if pos == scenario.rival_start {
                Tile::Rival
            } else if scenario.automaton_start.contains(&pos) {
                Tile::Automaton
            } else {
                scenario.world.tile(pos)
            };
            let policy = policy.get(&pos).map(|action| ActionDto {
                name: action.name(),
                glyph: action.glyph(),
            });
            CellDto {
                id: scenario.world.state_id(pos),
                x: pos.0,
                y: pos.1,
                q: pos.0,
                r: pos.1,
                tile,
                value: values.get(&pos).copied().unwrap_or(0.0),
                policy,
            }
        })
        .collect()
}

fn adversarial_transitions(
    scenario: &AdversarialScenario,
    values: &BTreeMap<Pos, f64>,
) -> Vec<TransitionDto> {
    scenario
        .world
        .playable_positions()
        .into_iter()
        .filter(|&pos| !scenario.world.is_terminal(pos))
        .flat_map(|pos| {
            Action::SQUARE.into_iter().map(move |action| {
                let next = transition(&scenario.world, pos, action);
                TransitionDto {
                    from: scenario.world.state_id(pos),
                    action: action.name(),
                    to: scenario.world.state_id(next),
                    reward: adversarial_reward(
                        &scenario.world,
                        next,
                        scenario.rival_start,
                        &scenario.automaton_start,
                    ),
                    q_value: adversarial_action_score(
                        &scenario.world,
                        pos,
                        action,
                        scenario.rival_start,
                        &scenario.automaton_start,
                    ) + values.get(&next).copied().unwrap_or(0.0) * 0.1,
                }
            })
        })
        .collect()
}

fn simulate_adversarial(scenario: &AdversarialScenario) -> Vec<TraceStepDto> {
    let mut out = Vec::new();
    let mut hikari = scenario.world.start;
    let mut rival = scenario.rival_start;
    let mut automata = scenario.automaton_start.clone();
    let mut cumulative_reward = 0.0;
    let mut generation = 0;

    for step in 0..24 {
        if scenario.world.is_terminal(hikari) {
            break;
        }
        let action = best_adversarial_action(&scenario.world, hikari, rival, &automata);
        let next = transition(&scenario.world, hikari, action);
        let rival_action = if step % 2 == 0 {
            best_rival_action(&scenario.world, rival, next, &automata)
        } else {
            None
        };
        let rival_next =
            rival_action.map_or(rival, |action| transition(&scenario.world, rival, action));
        if step > 0 && step % 3 == 0 {
            generation += 1;
            automata = evolve_automaton(&scenario.world, &automata, generation);
        }

        let terminal = scenario.world.tile(next) == Tile::Goal
            || scenario.world.tile(next) == Tile::Pit
            || next == rival_next
            || automata.contains(&next);
        let reward = if scenario.world.tile(next) == Tile::Goal {
            15.0
        } else if next == rival_next {
            -12.0
        } else {
            adversarial_reward(&scenario.world, next, rival_next, &automata)
        };
        cumulative_reward += reward;
        out.push(TraceStepDto {
            step,
            state: scenario.world.state_id(hikari),
            observation: scenario.world.observation_id(hikari),
            action: action.name(),
            next_state: scenario.world.state_id(next),
            rival_state: Some(scenario.world.state_id(rival_next)),
            rival_action: rival_action.map(Action::name),
            automaton_generation: Some(generation),
            automaton_cells: Some(
                automata
                    .iter()
                    .copied()
                    .map(|pos| scenario.world.state_id(pos))
                    .collect(),
            ),
            virus_cells: None,
            cleansed_cells: None,
            infection_count: None,
            victory: None,
            reward,
            cumulative_reward,
            value: adversarial_state_score(&scenario.world, hikari, rival, &automata),
            terminal,
        });

        hikari = next;
        rival = rival_next;
        if terminal {
            break;
        }
    }

    out
}

fn best_adversarial_action(
    world: &WorldSpec,
    pos: Pos,
    rival: Pos,
    automata: &BTreeSet<Pos>,
) -> Action {
    Action::SQUARE
        .into_iter()
        .max_by(|&a, &b| {
            adversarial_action_score(world, pos, a, rival, automata)
                .partial_cmp(&adversarial_action_score(world, pos, b, rival, automata))
                .unwrap()
                .then_with(|| b.name().cmp(a.name()))
        })
        .unwrap()
}

fn best_rival_action(
    world: &WorldSpec,
    rival: Pos,
    target: Pos,
    automata: &BTreeSet<Pos>,
) -> Option<Action> {
    Action::SQUARE
        .into_iter()
        .filter(|&action| {
            let next = transition(world, rival, action);
            next != rival && !automata.contains(&next) && world.tile(next) != Tile::Goal
        })
        .min_by_key(|&action| {
            let next = transition(world, rival, action);
            manhattan(next, target)
        })
}

fn adversarial_action_score(
    world: &WorldSpec,
    pos: Pos,
    action: Action,
    rival: Pos,
    automata: &BTreeSet<Pos>,
) -> f64 {
    let next = transition(world, pos, action);
    adversarial_state_score(world, next, rival, automata)
}

fn adversarial_state_score(
    world: &WorldSpec,
    pos: Pos,
    rival: Pos,
    automata: &BTreeSet<Pos>,
) -> f64 {
    let goal = world.goals[0];
    let mut score = 20.0 - (manhattan(pos, goal) as f64 * 1.75);
    score += manhattan(pos, rival).min(8) as f64 * 0.55;
    if automata.contains(&pos) {
        score -= 30.0;
    }
    let nearest_automaton = automata
        .iter()
        .map(|&cell| manhattan(pos, cell))
        .min()
        .unwrap_or(8);
    if nearest_automaton <= 1 {
        score -= 4.0;
    }
    if world.tile(pos) == Tile::Goal {
        score += 50.0;
    }
    if world.tile(pos) == Tile::Pit || pos == rival {
        score -= 40.0;
    }
    score
}

fn adversarial_reward(world: &WorldSpec, pos: Pos, rival: Pos, automata: &BTreeSet<Pos>) -> f64 {
    if world.tile(pos) == Tile::Goal {
        15.0
    } else if world.tile(pos) == Tile::Pit || automata.contains(&pos) || pos == rival {
        -12.0
    } else {
        -0.15
    }
}

fn evolve_automaton(
    world: &WorldSpec,
    automata: &BTreeSet<Pos>,
    generation: usize,
) -> BTreeSet<Pos> {
    let mut next = automata.clone();
    for &cell in automata {
        for action in Action::SQUARE {
            let candidate = transition(world, cell, action);
            let protected = candidate == world.start
                || candidate == world.goals[0]
                || candidate.1 == world.height - 1
                || world.tile(candidate) == Tile::Wall;
            if !protected && (candidate.0 + candidate.1 + generation as i32).rem_euclid(3) == 0 {
                next.insert(candidate);
            }
        }
    }
    next
}

fn adversarial_evolution(
    scenario: &AdversarialScenario,
    values: &BTreeMap<Pos, f64>,
    policy: &BTreeMap<Pos, Action>,
    trace: &[TraceStepDto],
    source: &str,
) -> Vec<EvolutionDto> {
    let initial = EvolutionDto {
        iteration: 0,
        label: "adversarial scaffold".to_string(),
        changed_states: 0,
        cells: adversarial_cells(scenario, &BTreeMap::new(), &BTreeMap::new()),
        policy_edges: Vec::new(),
        hymeko_source: render_initial_adversarial_hymeko_source(scenario),
    };
    let mut automata = scenario.automaton_start.clone();
    let mut mid_cells = Vec::new();
    for step in trace.iter().take(4) {
        if let Some(cells) = &step.automaton_cells {
            automata = cells
                .iter()
                .filter_map(|id| parse_state_id(id))
                .collect::<BTreeSet<_>>();
        }
    }
    let mid_policy = policy.clone();
    for pos in scenario.world.display_positions.iter().copied() {
        let tile = if automata.contains(&pos) {
            Tile::Automaton
        } else {
            scenario.world.tile(pos)
        };
        mid_cells.push(CellDto {
            id: scenario.world.state_id(pos),
            x: pos.0,
            y: pos.1,
            q: pos.0,
            r: pos.1,
            tile,
            value: values.get(&pos).copied().unwrap_or(0.0),
            policy: mid_policy.get(&pos).map(|action| ActionDto {
                name: action.name(),
                glyph: action.glyph(),
            }),
        });
    }
    vec![
        initial,
        EvolutionDto {
            iteration: 3,
            label: "competitive pressure observed".to_string(),
            changed_states: automata.len(),
            cells: mid_cells,
            policy_edges: policy_edges_from(&scenario.world, policy),
            hymeko_source: render_adversarial_snapshot_source(scenario, trace),
        },
        EvolutionDto {
            iteration: trace.len(),
            label: "adversarial rollout".to_string(),
            changed_states: trace.len(),
            cells: adversarial_cells(scenario, values, policy),
            policy_edges: policy_edges_from(&scenario.world, policy),
            hymeko_source: source.to_string(),
        },
    ]
}

fn parse_state_id(id: &str) -> Option<Pos> {
    let rest = id.strip_prefix('c')?;
    let (x, y) = rest.split_once('_')?;
    Some((x.parse().ok()?, y.parse().ok()?))
}

fn manhattan(a: Pos, b: Pos) -> i32 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}

fn solve_game(world: WorldSpec) -> SolvedGame {
    let mut values = world
        .playable_positions()
        .into_iter()
        .map(|pos| (pos, 0.0))
        .collect::<BTreeMap<_, _>>();
    let mut evolution = vec![EvolutionStage {
        iteration: 0,
        label: "initial agent scaffold".to_string(),
        policy: BTreeMap::new(),
        source: render_initial_hymeko_source(&world),
        values: values.clone(),
    }];

    let mut iterations = 0;
    for iter in 1..=MAX_ITERS {
        let mut next = values.clone();
        let mut delta: f64 = 0.0;

        for pos in world.playable_positions() {
            if world.is_terminal(pos) {
                next.insert(pos, 0.0);
                continue;
            }
            let best = world
                .actions()
                .iter()
                .copied()
                .map(|action| action_value(&world, pos, action, &values))
                .fold(f64::NEG_INFINITY, f64::max);
            let old = values[&pos];
            next.insert(pos, best);
            delta = delta.max((best - old).abs());
        }

        values = next;
        iterations = iter;
        if iter <= 3 {
            let policy = policy_from_values(&world, &values);
            evolution.push(EvolutionStage {
                iteration: iter,
                label: format!("value sweep {iter}"),
                source: render_learning_hymeko_source(&world, iter, &values, &policy),
                values: values.clone(),
                policy,
            });
        }
        if delta < EPSILON {
            break;
        }
    }

    let policy = policy_from_values(&world, &values);
    if evolution
        .last()
        .is_none_or(|stage| stage.iteration != iterations)
    {
        evolution.push(EvolutionStage {
            iteration: iterations,
            label: "converged policy".to_string(),
            source: render_learning_hymeko_source(&world, iterations, &values, &policy),
            values: values.clone(),
            policy: policy.clone(),
        });
    }

    SolvedGame {
        world,
        values,
        policy,
        iterations,
        evolution,
    }
}

fn policy_from_values(world: &WorldSpec, values: &BTreeMap<Pos, f64>) -> BTreeMap<Pos, Action> {
    world
        .playable_positions()
        .into_iter()
        .filter(|&pos| !world.is_terminal(pos))
        .map(|pos| {
            let action = world
                .actions()
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    action_value(world, pos, a, values)
                        .partial_cmp(&action_value(world, pos, b, values))
                        .unwrap()
                        .then_with(|| b.name().cmp(a.name()))
                })
                .unwrap();
            (pos, action)
        })
        .collect()
}

fn to_dto(solved: &SolvedGame, hymeko_source: String, gate: GateDto) -> GameDto {
    let world = &solved.world;
    let cells = cells_from(world, &solved.values, &solved.policy);

    let transitions = world
        .playable_positions()
        .into_iter()
        .filter(|&pos| !world.is_terminal(pos))
        .flat_map(|pos| {
            world.actions().iter().copied().map(move |action| {
                let next = transition(world, pos, action);
                TransitionDto {
                    from: world.state_id(pos),
                    action: action.name(),
                    to: world.state_id(next),
                    reward: reward(world, next),
                    q_value: action_value(world, pos, action, &solved.values),
                }
            })
        })
        .collect::<Vec<_>>();

    let policy_edges = policy_edges_from(world, &solved.policy);
    let trace = simulate_policy(solved);
    let evolution = evolution_dtos(world, &solved.evolution);

    GameDto {
        world_id: world.id.clone(),
        world_label: world.label.clone(),
        topology: world.topology.name(),
        width: world.width,
        height: world.height,
        start_state: world.state_id(world.start),
        gamma: GAMMA,
        iterations: solved.iterations,
        gate,
        counts: CountDto {
            playable_states: world.playable_positions().len(),
            action_edges: transitions.len(),
            transition_edges: transitions.len(),
            policy_edges: policy_edges.len(),
            concept_edges: concept_edge_count(world),
            simulation_steps: trace.len(),
        },
        concepts: concepts(),
        cells,
        transitions,
        policy_edges,
        trace,
        evolution,
        hymeko_source,
        mermaid: render_mermaid_policy(solved),
    }
}

fn cells_from(
    world: &WorldSpec,
    values: &BTreeMap<Pos, f64>,
    policy: &BTreeMap<Pos, Action>,
) -> Vec<CellDto> {
    world
        .display_positions
        .iter()
        .copied()
        .map(|pos| {
            let value = values.get(&pos).copied().unwrap_or(0.0);
            let policy = policy.get(&pos).map(|action| ActionDto {
                name: action.name(),
                glyph: action.glyph(),
            });
            CellDto {
                id: world.state_id(pos),
                x: pos.0,
                y: pos.1,
                q: pos.0,
                r: pos.1,
                tile: world.tile(pos),
                value,
                policy,
            }
        })
        .collect()
}

fn policy_edges_from(world: &WorldSpec, policy: &BTreeMap<Pos, Action>) -> Vec<PolicyEdgeDto> {
    policy
        .iter()
        .map(|(&pos, &action)| {
            let next = transition(world, pos, action);
            PolicyEdgeDto {
                from: world.state_id(pos),
                action: action.name(),
                to: world.state_id(next),
                reward: reward(world, next),
            }
        })
        .collect()
}

fn evolution_dtos(world: &WorldSpec, stages: &[EvolutionStage]) -> Vec<EvolutionDto> {
    let mut previous: Option<&EvolutionStage> = None;
    let mut out = Vec::new();
    for stage in stages {
        let changed_states = previous.map_or(0, |prev| {
            world
                .playable_positions()
                .into_iter()
                .filter(|pos| {
                    let a = prev.values.get(pos).copied().unwrap_or(0.0);
                    let b = stage.values.get(pos).copied().unwrap_or(0.0);
                    (a - b).abs() > 1e-6
                        || prev.policy.get(pos).copied() != stage.policy.get(pos).copied()
                })
                .count()
        });
        out.push(EvolutionDto {
            iteration: stage.iteration,
            label: stage.label.clone(),
            changed_states,
            cells: cells_from(world, &stage.values, &stage.policy),
            policy_edges: policy_edges_from(world, &stage.policy),
            hymeko_source: stage.source.clone(),
        });
        previous = Some(stage);
    }
    out
}

fn simulate_policy(solved: &SolvedGame) -> Vec<TraceStepDto> {
    let world = &solved.world;
    let mut out = Vec::new();
    let mut pos = world.start;
    let mut cumulative_reward = 0.0;
    let max_steps = (world.playable_positions().len() * 3).max(32);

    for step in 0..max_steps {
        if world.is_terminal(pos) {
            break;
        }
        let Some(&action) = solved.policy.get(&pos) else {
            break;
        };
        let next = transition(world, pos, action);
        let step_reward = reward(world, next);
        cumulative_reward += step_reward;
        out.push(TraceStepDto {
            step,
            state: world.state_id(pos),
            observation: world.observation_id(pos),
            action: action.name(),
            next_state: world.state_id(next),
            rival_state: None,
            rival_action: None,
            automaton_generation: None,
            automaton_cells: None,
            virus_cells: None,
            cleansed_cells: None,
            infection_count: None,
            victory: None,
            reward: step_reward,
            cumulative_reward,
            value: solved.values[&pos],
            terminal: world.is_terminal(next),
        });
        pos = next;
    }

    out
}

fn action_value(world: &WorldSpec, pos: Pos, action: Action, values: &BTreeMap<Pos, f64>) -> f64 {
    let next = transition(world, pos, action);
    reward(world, next) + GAMMA * values.get(&next).copied().unwrap_or(0.0)
}

fn transition(world: &WorldSpec, pos: Pos, action: Action) -> Pos {
    if world.is_terminal(pos) {
        return pos;
    }
    let (dx, dy) = action.delta();
    let candidate = (pos.0 + dx, pos.1 + dy);
    if !world.contains(candidate) || world.tile(candidate) == Tile::Wall {
        pos
    } else {
        candidate
    }
}

fn reward(world: &WorldSpec, pos: Pos) -> f64 {
    match world.tile(pos) {
        Tile::Goal => 10.0,
        Tile::Pit => -10.0,
        _ => -0.25,
    }
}

fn tile_label(tile: Tile) -> &'static str {
    match tile {
        Tile::Start => "start",
        Tile::Empty => "empty",
        Tile::Wall => "wall",
        Tile::Goal => "goal",
        Tile::Pit => "pit",
        Tile::Rival => "rival",
        Tile::Automaton => "automaton",
        Tile::Virus => "virus",
    }
}

fn square_positions(width: i32, height: i32) -> Vec<Pos> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .collect()
}

fn hex_positions(radius: i32) -> Vec<Pos> {
    let mut positions = Vec::new();
    for r in -radius..=radius {
        for q in -radius..=radius {
            if q.abs().max(r.abs()).max((q + r).abs()) <= radius {
                positions.push((q, r));
            }
        }
    }
    positions
}

fn coord_token(value: i32) -> String {
    if value < 0 {
        format!("m{}", value.abs())
    } else {
        value.to_string()
    }
}

fn concepts() -> Vec<ConceptDto> {
    vec![
        ConceptDto {
            id: "emergent_agent",
            label: "agent",
            role: "entity whose behavior emerges from state, observation, goals, and actions",
        },
        ConceptDto {
            id: "agent_state",
            label: "state hypernode",
            role: "mutable internal state bound to world cells and observations",
        },
        ConceptDto {
            id: "observation_space",
            label: "observation space",
            role: "observations are linked to agent state, not stored as isolated facts",
        },
        ConceptDto {
            id: "agent_goal",
            label: "goal subnode",
            role: "sub-goals that bias the policy: reach, avoid, and minimize cost",
        },
        ConceptDto {
            id: "action_hyperedge",
            label: "action hyperedge",
            role: "available control is an edge over agent, state, observation, goal, and next state",
        },
        ConceptDto {
            id: "reward_signal",
            label: "reward",
            role: "scalar feedback attached to action/transition hyperedges",
        },
        ConceptDto {
            id: "value_field",
            label: "value",
            role: "solved discounted return over state observations",
        },
        ConceptDto {
            id: "policy_selector",
            label: "policy",
            role: "agent-owned selector that contains and chooses action hyperedges",
        },
        ConceptDto {
            id: "agent_trace",
            label: "trace",
            role: "simulated rollout of the agent following selected action hyperedges",
        },
        ConceptDto {
            id: "mdp_observation",
            label: "observation",
            role: "local percept node produced from the agent-state/world-state binding",
        },
    ]
}

fn adversarial_concepts() -> Vec<ConceptDto> {
    let mut out = concepts();
    out.extend([
        ConceptDto {
            id: "competitive_agent",
            label: "competitive agent",
            role: "another policy-bearing agent whose motion pressures Hikari's choices",
        },
        ConceptDto {
            id: "cellular_automaton",
            label: "cell automaton",
            role: "environment process that evolves hazard cells during the rollout",
        },
        ConceptDto {
            id: "adversarial_observation",
            label: "adversarial observation",
            role: "agent observation extended with rival position and automaton generation",
        },
        ConceptDto {
            id: "arena_generation",
            label: "arena generation",
            role: "time-indexed CA state linked into the agent trace",
        },
    ]);
    out
}

fn virus_concepts() -> Vec<ConceptDto> {
    let mut out = concepts();
    out.extend([
        ConceptDto {
            id: "virus_automaton",
            label: "virus automaton",
            role: "cellular automaton whose infection cells spread and survive by local rules",
        },
        ConceptDto {
            id: "cleanse_pulse",
            label: "cleanse pulse",
            role: "EGO-local action effect that removes nearby infected cells",
        },
        ConceptDto {
            id: "infection_front",
            label: "infection front",
            role: "time-varying observation layer produced by the virus CA",
        },
        ConceptDto {
            id: "victory_condition",
            label: "victory",
            role: "reach the objective while keeping the infection below the loss threshold",
        },
    ]);
    out
}

fn concept_edge_count(world: &WorldSpec) -> usize {
    // 6 schema/agent-policy/goal edges + 2 state/observation annotations per
    // playable state. Action and policy hyperedges are counted separately.
    6 + (world.playable_positions().len() * 2)
}

fn render_initial_hymeko_source(world: &WorldSpec) -> String {
    format!(
        r#"// Initial HyMeKo RL scaffold: no learned action values yet.
emergent_agent_demo {{
  phase "initial";
  world "{}";
  topology "{}";
}}

AgentMdpVocabulary {{
  meta_element;
  emergent_agent: + <isa> meta_element {{}}
  agent_state: + <isa> meta_element {{}}
  observation_space: + <isa> meta_element {{}}
  agent_goal: + <isa> meta_element {{}}
  policy_selector: + <isa> meta_element {{}}
}}

EmergentGridAgent {{
  agent_hikari: + <isa> AgentMdpVocabulary.emergent_agent {{}}
  hikari_state: + <isa> AgentMdpVocabulary.agent_state {{ gamma 0.95; }}
  hikari_policy: + <isa> AgentMdpVocabulary.policy_selector {{
    // Actions will be reified here as the agent learns.
  }}
  goal_reach_exit: + <isa> AgentMdpVocabulary.agent_goal {{}}
  goal_avoid_pit: + <isa> AgentMdpVocabulary.agent_goal {{}}
}}

@agent_policy_binding: + <isa> AgentMdpVocabulary.policy_selector {{
  (+ EmergentGridAgent.agent_hikari,
   + EmergentGridAgent.hikari_state,
   + EmergentGridAgent.hikari_policy);
}}
"#,
        world.id,
        world.topology.name()
    )
}

fn render_learning_hymeko_source(
    world: &WorldSpec,
    iteration: usize,
    values: &BTreeMap<Pos, f64>,
    policy: &BTreeMap<Pos, Action>,
) -> String {
    let mut out = format!(
        "// Learning snapshot after value-iteration sweep {iteration}.\n\
         // World: {} ({})\n\
         // Policy-owned actions are only shown for selected behavior.\n\
         emergent_agent_demo {{ phase \"learned\"; sweep {iteration}; topology \"{}\"; }}\n\n\
         EmergentGridAgent {{\n\
         \x20\x20agent_hikari {{}}\n\
         \x20\x20hikari_state {{ gamma 0.95; }}\n\
         \x20\x20hikari_policy {{\n",
        world.label,
        world.id,
        world.topology.name()
    );
    for (&pos, &action) in policy {
        let next = transition(world, pos, action);
        out.push_str(&format!(
            "    @action_{}_{} {{ value {:.6}; reward {:.2}; }}\n",
            world.state_id(pos),
            action.name(),
            values[&pos],
            reward(world, next)
        ));
        out.push_str(&format!(
            "    @select_{} {{ (~{}, ~{}, ~action_{}_{}); }}\n",
            world.state_id(pos),
            world.state_id(pos),
            world.observation_id(pos),
            world.state_id(pos),
            action.name()
        ));
    }
    out.push_str("  }\n}\n");
    out
}

fn render_hymeko_source(solved: &SolvedGame) -> String {
    let world = &solved.world;
    let mut out = format!(
        r#"// Emergent-agent reinforcement-learning witness.
// The solver computes values/policy, but HyMeKo describes the agent as a
// typed hypergraph: agent state + observation space + goals + action edges.
emergent_agent_demo {{
  author "HyMeKo WASM";
  world "{}";
  topology "{}";
}}

AgentMdpVocabulary {{
  // Meta layer: these declarations are templates/types used by the agent model.
  meta_element;
  emergent_agent: + <isa> meta_element {{}}
  agent_state: + <isa> meta_element {{}}
  environment_space: + <isa> meta_element {{}}
  environment_state: + <isa> meta_element {{}}
  observation_space: + <isa> meta_element {{}}
  mdp_observation: + <isa> meta_element {{}}
  agent_goal: + <isa> meta_element {{}}
  action_hyperedge: + <isa> meta_element {{}}
  reward_signal: + <isa> meta_element {{}}
  value_field: + <isa> meta_element {{}}
  policy_selector: + <isa> meta_element {{}}
  agent_trace: + <isa> meta_element {{}}
}}

EmergentGridAgent: + <isa> AgentMdpVocabulary.environment_space,
                   + <isa> AgentMdpVocabulary.observation_space
{{
  // Agent-space declarations.
  agent_hikari: + <isa> AgentMdpVocabulary.emergent_agent {{
    name "Hikari";
  }}
  hikari_state: + <isa> AgentMdpVocabulary.agent_state {{
    gamma 0.95;
  }}

  // Goal sub-hypernodes. The policy is interpreted against this goal stack.
  goal_reach_exit: + <isa> AgentMdpVocabulary.agent_goal {{ weight 1.0; }}
  goal_avoid_pit: + <isa> AgentMdpVocabulary.agent_goal {{ weight 1.0; }}
  goal_minimize_cost: + <isa> AgentMdpVocabulary.agent_goal {{ weight 0.25; }}

  // Concept aliases visible in the browser MDP tab.
"#,
        world.id,
        world.topology.name()
    );
    for concept in concepts() {
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.meta_element {{ role \"{}\"; }}\n",
            concept.id, concept.role
        ));
    }
    out.push_str("\n  // Environment states and their observation nodes.\n");
    for pos in world.playable_positions() {
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.environment_state {{ q \"{}\"; r \"{}\"; value {:.6}; tile \"{}\"; }}\n",
            world.state_id(pos),
            pos.0,
            pos.1,
            solved.values[&pos],
            tile_label(world.tile(pos))
        ));
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.mdp_observation {{ source \"{}\"; }}\n",
            world.observation_id(pos),
            world.state_id(pos)
        ));
    }
    out.push_str("\n  // Agent-owned policy space. Actions live inside the policy because\n");
    out.push_str("  // they are policy candidates, not global environment facts.\n");
    out.push_str("  hikari_policy: + <isa> AgentMdpVocabulary.policy_selector {\n");
    for pos in world.playable_positions() {
        if world.is_terminal(pos) {
            continue;
        }
        for &action in world.actions() {
            let next = transition(world, pos, action);
            out.push_str(&format!(
                "    @action_{}_{}: + <isa> AgentMdpVocabulary.action_hyperedge {{\n      // Candidate action owned by hikari_policy.\n      reward {:.2};\n      q_value {:.6};\n      (+ EmergentGridAgent.agent_hikari, + EmergentGridAgent.hikari_state, + EmergentGridAgent.{}, + EmergentGridAgent.{}, + EmergentGridAgent.goal_reach_exit, + EmergentGridAgent.goal_avoid_pit, + EmergentGridAgent.{}, + AgentMdpVocabulary.reward_signal);\n    }}\n",
                world.state_id(pos),
                action.name(),
                reward(world, next),
                action_value(world, pos, action, &solved.values),
                world.state_id(pos),
                world.observation_id(pos),
                world.state_id(next)
            ));
        }
        let policy = solved.policy[&pos];
        out.push_str(&format!(
            "    @select_{}: + <isa> AgentMdpVocabulary.policy_selector {{\n      // The selected behavior points to an action hyperedge owned by this policy.\n      (+ EmergentGridAgent.hikari_state, + EmergentGridAgent.{}, + EmergentGridAgent.{}, + EmergentGridAgent.hikari_policy.action_{}_{}, + AgentMdpVocabulary.value_field);\n    }}\n",
            world.state_id(pos),
            world.state_id(pos),
            world.observation_id(pos),
            world.state_id(pos),
            policy.name()
        ));
    }
    out.push_str("  }\n");
    out.push_str("}\n");

    out.push_str("\n// Schema edges: these are typed hyperedge templates over the vocabulary.\n");
    out.push_str("@agent_embodiment: + <isa> AgentMdpVocabulary.action_hyperedge { (+ EmergentGridAgent.agent_hikari, + EmergentGridAgent.hikari_state, + EmergentGridAgent.observation_space); }\n");
    out.push_str("@agent_policy_binding: + <isa> AgentMdpVocabulary.policy_selector { (+ EmergentGridAgent.agent_hikari, + EmergentGridAgent.hikari_state, + EmergentGridAgent.hikari_policy); }\n");
    out.push_str("@goal_stack: + <isa> AgentMdpVocabulary.action_hyperedge { (+ EmergentGridAgent.agent_hikari, + EmergentGridAgent.goal_reach_exit, + EmergentGridAgent.goal_avoid_pit, + EmergentGridAgent.goal_minimize_cost); }\n");
    out.push_str("@schema_action: + <isa> AgentMdpVocabulary.action_hyperedge { (+ AgentMdpVocabulary.emergent_agent, + AgentMdpVocabulary.agent_state, + AgentMdpVocabulary.mdp_observation, + AgentMdpVocabulary.agent_goal, + AgentMdpVocabulary.reward_signal); }\n");
    out.push_str("@schema_policy: + <isa> AgentMdpVocabulary.policy_selector { (+ AgentMdpVocabulary.agent_state, + AgentMdpVocabulary.mdp_observation, + AgentMdpVocabulary.action_hyperedge, + AgentMdpVocabulary.value_field); }\n");
    out.push_str("@schema_trace: + <isa> AgentMdpVocabulary.agent_trace { (+ AgentMdpVocabulary.emergent_agent, + AgentMdpVocabulary.agent_state, + AgentMdpVocabulary.mdp_observation, + AgentMdpVocabulary.action_hyperedge, + AgentMdpVocabulary.reward_signal); }\n");

    out.push_str("\n// Agent state is linked to environment cells and observations.\n");
    for pos in world.playable_positions() {
        out.push_str(&format!(
            "@state_bind_{}: + <isa> AgentMdpVocabulary.value_field {{ (+ EmergentGridAgent.hikari_state, + EmergentGridAgent.{}); }}\n",
            world.state_id(pos),
            world.state_id(pos)
        ));
        out.push_str(&format!(
            "@observe_{}: + <isa> AgentMdpVocabulary.mdp_observation {{ (+ EmergentGridAgent.hikari_state, + EmergentGridAgent.{}, + EmergentGridAgent.{}); }}\n",
            world.state_id(pos),
            world.state_id(pos),
            world.observation_id(pos)
        ));
        if world.is_terminal(pos) {
            continue;
        }
        let policy = solved.policy[&pos];
        out.push_str(&format!(
            "@policy_{}: + <isa> AgentMdpVocabulary.policy_selector {{\n  // Agent-to-policy relation for this state delegates to the owned action edge.\n  (+ EmergentGridAgent.agent_hikari, + EmergentGridAgent.hikari_policy, + EmergentGridAgent.hikari_state, + EmergentGridAgent.{}, + EmergentGridAgent.hikari_policy.action_{}_{}, + AgentMdpVocabulary.value_field);\n}}\n",
            world.state_id(pos),
            world.state_id(pos),
            world.state_id(pos),
            policy.name()
        ));
    }

    out.push_str("\n// Simulation trace: the policy-following rollout, step by step.\n");
    for step in simulate_policy(solved) {
        out.push_str(&format!(
            "@trace_{}: + <isa> AgentMdpVocabulary.agent_trace {{\n  step {};\n  cumulative_reward {:.2};\n  (+ EmergentGridAgent.agent_hikari, + EmergentGridAgent.hikari_state, + EmergentGridAgent.{}, + EmergentGridAgent.{}, + EmergentGridAgent.hikari_policy.action_{}_{}, + AgentMdpVocabulary.reward_signal);\n}}\n",
            step.step,
            step.step,
            step.cumulative_reward,
            step.state,
            step.observation,
            step.state,
            step.action
        ));
    }

    out
}

fn render_initial_adversarial_hymeko_source(scenario: &AdversarialScenario) -> String {
    format!(
        r#"// Initial adversarial HyMeKo scaffold: two agents and a CA arena.
adversarial_agent_demo {{
  phase "initial";
  world "{}";
}}

AgentMdpVocabulary {{
  meta_element;
  emergent_agent: + <isa> meta_element {{}}
  competitive_agent: + <isa> meta_element {{}}
  cell_automaton: + <isa> meta_element {{}}
  agent_state: + <isa> meta_element {{}}
  policy_selector: + <isa> meta_element {{}}
}}

AdversarialArena {{
  agent_hikari: + <isa> AgentMdpVocabulary.emergent_agent {{}}
  rival_kage: + <isa> AgentMdpVocabulary.competitive_agent {{}}
  arena_automaton: + <isa> AgentMdpVocabulary.cell_automaton {{}}
  hikari_state: + <isa> AgentMdpVocabulary.agent_state {{ start "{}"; }}
  hikari_policy: + <isa> AgentMdpVocabulary.policy_selector {{}}
}}
"#,
        scenario.world.id,
        scenario.world.state_id(scenario.world.start)
    )
}

fn render_adversarial_snapshot_source(
    scenario: &AdversarialScenario,
    trace: &[TraceStepDto],
) -> String {
    let mut out = String::from(
        "// Adversarial snapshot: trace binds Hikari, Kage, and CA generations.\n\
         adversarial_agent_demo { phase \"competitive_pressure\"; }\n\n\
         AdversarialArena {\n\
         \x20\x20agent_hikari {}\n\
         \x20\x20rival_kage {}\n\
         \x20\x20arena_automaton {}\n\
         }\n",
    );
    for step in trace.iter().take(4) {
        out.push_str(&format!(
            "@snapshot_{} {{ (+ AdversarialArena.agent_hikari, + AdversarialArena.rival_kage, + AdversarialArena.arena_automaton); step {}; hikari \"{}\"; rival \"{}\"; generation {}; }}\n",
            step.step,
            step.step,
            step.next_state,
            step.rival_state.as_deref().unwrap_or("unknown"),
            step.automaton_generation.unwrap_or(0)
        ));
    }
    if trace.is_empty() {
        out.push_str(&format!(
            "@snapshot_0 {{ (+ AdversarialArena.agent_hikari, + AdversarialArena.rival_kage, + AdversarialArena.arena_automaton); start \"{}\"; }}\n",
            scenario.world.state_id(scenario.world.start)
        ));
    }
    out
}

fn render_adversarial_hymeko_source(
    scenario: &AdversarialScenario,
    policy: &BTreeMap<Pos, Action>,
    values: &BTreeMap<Pos, f64>,
    trace: &[TraceStepDto],
) -> String {
    let world = &scenario.world;
    let mut out = format!(
        r#"// Adversarial emergent-agent witness.
// Hikari has a policy, Kage is a competitive agent, and the arena has a
// cellular automaton that changes the observation field over time.
adversarial_agent_demo {{
  author "HyMeKo WASM";
  world "{}";
  topology "{}";
  interaction "competitive";
}}

AgentMdpVocabulary {{
  meta_element;
  emergent_agent: + <isa> meta_element {{}}
  competitive_agent: + <isa> meta_element {{}}
  agent_state: + <isa> meta_element {{}}
  environment_space: + <isa> meta_element {{}}
  environment_state: + <isa> meta_element {{}}
  observation_space: + <isa> meta_element {{}}
  mdp_observation: + <isa> meta_element {{}}
  adversarial_observation: + <isa> meta_element {{}}
  agent_goal: + <isa> meta_element {{}}
  action_hyperedge: + <isa> meta_element {{}}
  reward_signal: + <isa> meta_element {{}}
  value_field: + <isa> meta_element {{}}
  policy_selector: + <isa> meta_element {{}}
  agent_trace: + <isa> meta_element {{}}
  cell_automaton: + <isa> meta_element {{}}
  automaton_state: + <isa> meta_element {{}}
}}

AdversarialArena: + <isa> AgentMdpVocabulary.environment_space,
                  + <isa> AgentMdpVocabulary.observation_space
{{
  agent_hikari: + <isa> AgentMdpVocabulary.emergent_agent {{ name "Hikari"; }}
  rival_kage: + <isa> AgentMdpVocabulary.competitive_agent {{ name "Kage"; start "{}"; }}
  arena_automaton: + <isa> AgentMdpVocabulary.cell_automaton {{
    rule "deterministic neighbor spread";
  }}
  hikari_state: + <isa> AgentMdpVocabulary.agent_state {{ gamma 0.95; }}
  goal_reach_exit: + <isa> AgentMdpVocabulary.agent_goal {{ weight 1.0; }}
  goal_avoid_kage: + <isa> AgentMdpVocabulary.agent_goal {{ weight 1.0; }}
  goal_avoid_automaton: + <isa> AgentMdpVocabulary.agent_goal {{ weight 1.0; }}

"#,
        world.id,
        world.topology.name(),
        world.state_id(scenario.rival_start)
    );

    for concept in adversarial_concepts() {
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.meta_element {{ role \"{}\"; }}\n",
            concept.id, concept.role
        ));
    }

    out.push_str("\n  // Environment states observed by both agents.\n");
    for pos in world.playable_positions() {
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.environment_state {{ x {}; y {}; value {:.6}; tile \"{}\"; }}\n",
            world.state_id(pos),
            pos.0,
            pos.1,
            values.get(&pos).copied().unwrap_or(0.0),
            tile_label(world.tile(pos))
        ));
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.adversarial_observation {{ source \"{}\"; }}\n",
            world.observation_id(pos),
            world.state_id(pos)
        ));
    }

    out.push_str("\n  hikari_policy: + <isa> AgentMdpVocabulary.policy_selector {\n");
    for (&pos, &selected) in policy {
        if world.is_terminal(pos) {
            continue;
        }
        for action in Action::SQUARE {
            let next = transition(world, pos, action);
            out.push_str(&format!(
                "    @action_{}_{}: + <isa> AgentMdpVocabulary.action_hyperedge {{ reward {:.2}; q_value {:.6}; (+ AdversarialArena.agent_hikari, + AdversarialArena.rival_kage, + AdversarialArena.arena_automaton, + AdversarialArena.{}, + AdversarialArena.{}, + AdversarialArena.{}); }}\n",
                world.state_id(pos),
                action.name(),
                adversarial_reward(world, next, scenario.rival_start, &scenario.automaton_start),
                adversarial_action_score(world, pos, action, scenario.rival_start, &scenario.automaton_start),
                world.state_id(pos),
                world.observation_id(pos),
                world.state_id(next)
            ));
        }
        out.push_str(&format!(
            "    @select_{}: + <isa> AgentMdpVocabulary.policy_selector {{ (+ AdversarialArena.hikari_state, + AdversarialArena.{}, + AdversarialArena.hikari_policy.action_{}_{}, + AgentMdpVocabulary.value_field); }}\n",
            world.state_id(pos),
            world.observation_id(pos),
            world.state_id(pos),
            selected.name()
        ));
    }
    out.push_str("  }\n}\n");

    out.push_str("\n@competitive_binding: + <isa> AgentMdpVocabulary.action_hyperedge { (+ AdversarialArena.agent_hikari, + AdversarialArena.rival_kage, + AdversarialArena.hikari_policy); }\n");
    out.push_str("@automaton_binding: + <isa> AgentMdpVocabulary.automaton_state { (+ AdversarialArena.arena_automaton, + AdversarialArena.hikari_state); }\n");
    out.push_str("@adversarial_goal_stack: + <isa> AgentMdpVocabulary.action_hyperedge { (+ AdversarialArena.agent_hikari, + AdversarialArena.goal_reach_exit, + AdversarialArena.goal_avoid_kage, + AdversarialArena.goal_avoid_automaton); }\n");

    let mut generations = BTreeMap::<usize, Vec<String>>::new();
    for step in trace {
        if let (Some(generation), Some(cells)) = (step.automaton_generation, &step.automaton_cells)
        {
            generations
                .entry(generation)
                .or_insert_with(|| cells.clone());
        }
    }
    for (generation, cells) in generations {
        out.push_str(&format!(
            "@automaton_generation_{}: + <isa> AgentMdpVocabulary.automaton_state {{ generation {}; cells \"{}\"; (+ AdversarialArena.arena_automaton, + AdversarialArena.hikari_state); }}\n",
            generation,
            generation,
            cells.join(",")
        ));
    }

    out.push_str("\n// Adversarial trace binds both agents and the evolving CA layer.\n");
    for step in trace {
        out.push_str(&format!(
            "@trace_{}: + <isa> AgentMdpVocabulary.agent_trace {{ step {}; cumulative_reward {:.2}; rival \"{}\"; automaton_generation {}; (+ AdversarialArena.agent_hikari, + AdversarialArena.rival_kage, + AdversarialArena.arena_automaton, + AdversarialArena.{}, + AdversarialArena.{}, + AdversarialArena.hikari_policy.action_{}_{}, + AgentMdpVocabulary.reward_signal); }}\n",
            step.step,
            step.step,
            step.cumulative_reward,
            step.rival_state.as_deref().unwrap_or("unknown"),
            step.automaton_generation.unwrap_or(0),
            step.state,
            step.observation,
            step.state,
            step.action
        ));
    }

    out
}

fn render_adversarial_mermaid(
    scenario: &AdversarialScenario,
    policy: &BTreeMap<Pos, Action>,
    values: &BTreeMap<Pos, f64>,
) -> String {
    let world = &scenario.world;
    let mut out = String::from("flowchart LR\n");
    out.push_str("  Hikari((Hikari))\n  Kage((Kage))\n  CA[[Cell automaton]]\n");
    for pos in world.playable_positions() {
        let label = if pos == scenario.rival_start {
            "K".to_string()
        } else if scenario.automaton_start.contains(&pos) {
            "CA".to_string()
        } else {
            match world.tile(pos) {
                Tile::Start => format!("S\\n{:.2}", values[&pos]),
                Tile::Goal => "G".to_string(),
                Tile::Pit => "X".to_string(),
                Tile::Empty => format!("{}\\n{:.2}", world.state_id(pos), values[&pos]),
                Tile::Wall | Tile::Rival | Tile::Automaton | Tile::Virus => unreachable!(),
            }
        };
        out.push_str(&format!("  {}[\"{}\"]\n", world.state_id(pos), label));
    }
    for (&pos, &action) in policy {
        let next = transition(world, pos, action);
        out.push_str(&format!(
            "  {} -->|{}| {}\n",
            world.state_id(pos),
            action.name(),
            world.state_id(next)
        ));
    }
    out.push_str("  Kage -. pressures .-> Hikari\n  CA -. evolves observations .-> Hikari\n");
    out
}

fn render_virus_hymeko_source(
    scenario: &VirusScenario,
    policy: &BTreeMap<Pos, Action>,
    values: &BTreeMap<Pos, f64>,
    trace: &[TraceStepDto],
) -> String {
    let world = &scenario.world;
    let mut out = format!(
        r#"// Virus cellular-automaton gameplay witness.
// Hikari is the EGO agent. The virus is a cellular automaton, and every
// movement emits a local cleanse pulse before the infection front evolves.
virus_agent_demo {{
  author "HyMeKo WASM";
  world "{}";
  topology "{}";
  gameplay "cleanse-and-reach";
}}

AgentMdpVocabulary {{
  meta_element;
  emergent_agent: + <isa> meta_element {{}}
  agent_state: + <isa> meta_element {{}}
  environment_space: + <isa> meta_element {{}}
  environment_state: + <isa> meta_element {{}}
  observation_space: + <isa> meta_element {{}}
  mdp_observation: + <isa> meta_element {{}}
  agent_goal: + <isa> meta_element {{}}
  action_hyperedge: + <isa> meta_element {{}}
  reward_signal: + <isa> meta_element {{}}
  value_field: + <isa> meta_element {{}}
  policy_selector: + <isa> meta_element {{}}
  agent_trace: + <isa> meta_element {{}}
  virus_automaton: + <isa> meta_element {{}}
  infection_front: + <isa> meta_element {{}}
  cleanse_pulse: + <isa> meta_element {{}}
  victory_condition: + <isa> meta_element {{}}
}}

VirusArena: + <isa> AgentMdpVocabulary.environment_space,
            + <isa> AgentMdpVocabulary.observation_space
{{
  agent_hikari: + <isa> AgentMdpVocabulary.emergent_agent {{ name "Hikari"; role "EGO"; }}
  hikari_state: + <isa> AgentMdpVocabulary.agent_state {{ gamma 0.95; }}
  arena_virus: + <isa> AgentMdpVocabulary.virus_automaton {{ rule "cleanse then life-like spread"; }}
  goal_reach_victory: + <isa> AgentMdpVocabulary.agent_goal {{ weight 1.0; }}
  goal_suppress_infection: + <isa> AgentMdpVocabulary.agent_goal {{ weight 0.8; }}

"#,
        world.id,
        world.topology.name()
    );

    for concept in virus_concepts() {
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.meta_element {{ role \"{}\"; }}\n",
            concept.id, concept.role
        ));
    }
    out.push_str("\n  // State observations include both terrain and infection-front context.\n");
    for pos in world.playable_positions() {
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.environment_state {{ x {}; y {}; value {:.6}; tile \"{}\"; }}\n",
            world.state_id(pos),
            pos.0,
            pos.1,
            values.get(&pos).copied().unwrap_or(0.0),
            tile_label(world.tile(pos))
        ));
        out.push_str(&format!(
            "  {}: + <isa> AgentMdpVocabulary.mdp_observation {{ source \"{}\"; }}\n",
            world.observation_id(pos),
            world.state_id(pos)
        ));
    }

    out.push_str("\n  hikari_policy: + <isa> AgentMdpVocabulary.policy_selector {\n");
    for (&pos, &selected) in policy {
        if world.is_terminal(pos) {
            continue;
        }
        for action in Action::SQUARE {
            let next = transition(world, pos, action);
            out.push_str(&format!(
                "    @action_{}_{}: + <isa> AgentMdpVocabulary.action_hyperedge {{ reward {:.2}; q_value {:.6}; (+ VirusArena.agent_hikari, + VirusArena.hikari_state, + VirusArena.arena_virus, + VirusArena.{}, + VirusArena.{}, + VirusArena.{}); }}\n",
                world.state_id(pos),
                action.name(),
                virus_reward(world, next, &scenario.virus_start),
                virus_action_score(world, pos, action, &scenario.virus_start),
                world.state_id(pos),
                world.observation_id(pos),
                world.state_id(next)
            ));
        }
        out.push_str(&format!(
            "    @select_{}: + <isa> AgentMdpVocabulary.policy_selector {{ (+ VirusArena.hikari_state, + VirusArena.{}, + VirusArena.hikari_policy.action_{}_{}, + AgentMdpVocabulary.value_field); }}\n",
            world.state_id(pos),
            world.observation_id(pos),
            world.state_id(pos),
            selected.name()
        ));
    }
    out.push_str("  }\n}\n");

    out.push_str("\n@virus_goal_stack: + <isa> AgentMdpVocabulary.action_hyperedge { (+ VirusArena.agent_hikari, + VirusArena.goal_reach_victory, + VirusArena.goal_suppress_infection); }\n");
    out.push_str("@virus_binding: + <isa> AgentMdpVocabulary.infection_front { (+ VirusArena.agent_hikari, + VirusArena.hikari_state, + VirusArena.arena_virus); }\n");

    for step in trace {
        out.push_str(&format!(
            "@infection_front_{}: + <isa> AgentMdpVocabulary.infection_front {{ generation {}; infection_count {}; cells \"{}\"; (+ VirusArena.arena_virus, + VirusArena.hikari_state); }}\n",
            step.step,
            step.automaton_generation.unwrap_or(step.step),
            step.infection_count.unwrap_or(0),
            step.virus_cells.clone().unwrap_or_default().join(",")
        ));
        out.push_str(&format!(
            "@cleanse_{}: + <isa> AgentMdpVocabulary.cleanse_pulse {{ cells \"{}\"; (+ VirusArena.agent_hikari, + VirusArena.arena_virus, + VirusArena.{}); }}\n",
            step.step,
            step.cleansed_cells.clone().unwrap_or_default().join(","),
            step.next_state
        ));
        out.push_str(&format!(
            "@trace_{}: + <isa> AgentMdpVocabulary.agent_trace {{ step {}; victory \"{}\"; cumulative_reward {:.2}; (+ VirusArena.agent_hikari, + VirusArena.hikari_state, + VirusArena.{}, + VirusArena.{}, + VirusArena.hikari_policy.action_{}_{}, + AgentMdpVocabulary.reward_signal); }}\n",
            step.step,
            step.step,
            step.victory.unwrap_or(false),
            step.cumulative_reward,
            step.state,
            step.observation,
            step.state,
            step.action
        ));
    }

    out
}

fn virus_evolution(
    scenario: &VirusScenario,
    values: &BTreeMap<Pos, f64>,
    policy: &BTreeMap<Pos, Action>,
    trace: &[TraceStepDto],
    source: &str,
) -> Vec<EvolutionDto> {
    let initial_source = format!(
        "// Initial virus arena scaffold.\nvirus_agent_demo {{ phase \"initial\"; world \"{}\"; }}\nVirusArena {{ agent_hikari {{ role \"EGO\"; }} arena_virus {{}} hikari_policy {{}} }}\n",
        scenario.world.id
    );
    let initial = EvolutionDto {
        iteration: 0,
        label: "virus scaffold".to_string(),
        changed_states: scenario.virus_start.len(),
        cells: virus_cells(
            scenario,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &scenario.virus_start,
        ),
        policy_edges: Vec::new(),
        hymeko_source: initial_source,
    };
    let mid_viruses = trace
        .get(2)
        .and_then(|step| step.virus_cells.as_ref())
        .map(|cells| cells.iter().filter_map(|id| parse_state_id(id)).collect())
        .unwrap_or_else(|| scenario.virus_start.clone());
    vec![
        initial,
        EvolutionDto {
            iteration: 3,
            label: "infection front responds".to_string(),
            changed_states: mid_viruses.len(),
            cells: virus_cells(scenario, values, policy, &mid_viruses),
            policy_edges: policy_edges_from(&scenario.world, policy),
            hymeko_source: render_virus_snapshot_source(scenario, trace),
        },
        EvolutionDto {
            iteration: trace.len(),
            label: "victory rollout".to_string(),
            changed_states: trace
                .last()
                .and_then(|step| step.infection_count)
                .unwrap_or(scenario.virus_start.len()),
            cells: virus_cells(scenario, values, policy, &scenario.virus_start),
            policy_edges: policy_edges_from(&scenario.world, policy),
            hymeko_source: source.to_string(),
        },
    ]
}

fn render_virus_snapshot_source(scenario: &VirusScenario, trace: &[TraceStepDto]) -> String {
    let mut out = format!(
        "// Virus snapshot: EGO cleanse pulses alter the CA infection front.\nvirus_agent_demo {{ phase \"infection_front\"; world \"{}\"; }}\n",
        scenario.world.id
    );
    for step in trace.iter().take(4) {
        out.push_str(&format!(
            "@virus_step_{} {{ ego \"{}\"; action \"{}\"; infection_count {}; cleansed \"{}\"; }}\n",
            step.step,
            step.next_state,
            step.action,
            step.infection_count.unwrap_or(0),
            step.cleansed_cells.clone().unwrap_or_default().join(",")
        ));
    }
    out
}

fn render_virus_mermaid(
    scenario: &VirusScenario,
    policy: &BTreeMap<Pos, Action>,
    values: &BTreeMap<Pos, f64>,
) -> String {
    let world = &scenario.world;
    let mut out = String::from("flowchart LR\n  Hikari((EGO Hikari))\n  Virus[[Virus CA]]\n");
    for pos in world.playable_positions() {
        let label = if scenario.virus_start.contains(&pos) {
            "V".to_string()
        } else {
            match world.tile(pos) {
                Tile::Start => format!("S\\n{:.2}", values[&pos]),
                Tile::Goal => "Victory".to_string(),
                Tile::Pit => "X".to_string(),
                Tile::Empty => format!("{}\\n{:.2}", world.state_id(pos), values[&pos]),
                Tile::Wall | Tile::Rival | Tile::Automaton | Tile::Virus => unreachable!(),
            }
        };
        out.push_str(&format!("  {}[\"{}\"]\n", world.state_id(pos), label));
    }
    for (&pos, &action) in policy {
        let next = transition(world, pos, action);
        out.push_str(&format!(
            "  {} -->|{} + cleanse| {}\n",
            world.state_id(pos),
            action.name(),
            world.state_id(next)
        ));
    }
    out.push_str("  Virus -. infection front .-> Hikari\n  Hikari -. cleanse pulse .-> Virus\n");
    out
}

fn render_mermaid_policy(solved: &SolvedGame) -> String {
    let world = &solved.world;
    let mut out = String::from("flowchart LR\n");
    for pos in world.playable_positions() {
        let label = match world.tile(pos) {
            Tile::Start => format!("S\\n{:.2}", solved.values[&pos]),
            Tile::Goal => "G".to_string(),
            Tile::Pit => "X".to_string(),
            Tile::Empty => format!("{}\\n{:.2}", world.state_id(pos), solved.values[&pos]),
            Tile::Wall | Tile::Rival | Tile::Automaton | Tile::Virus => unreachable!(),
        };
        out.push_str(&format!("  {}[\"{}\"]\n", world.state_id(pos), label));
    }
    for (&pos, &action) in &solved.policy {
        let next = transition(world, pos, action);
        out.push_str(&format!(
            "  {} -->|{} r={:+.2}| {}\n",
            world.state_id(pos),
            action.name(),
            reward(world, next),
            world.state_id(next)
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solved_grid_game_reaches_goal_from_start() {
        let solved = solve_game(WorldSpec::grid4());
        assert_eq!(solved.policy[&(0, 0)], Action::Down);

        let world = &solved.world;
        let mut pos = world.start;
        for _ in 0..16 {
            if world.is_terminal(pos) {
                break;
            }
            pos = transition(world, pos, solved.policy[&pos]);
        }
        assert_eq!(world.tile(pos), Tile::Goal);
    }

    #[test]
    fn solved_grid_game_json_contains_gated_representation() {
        let json = solved_grid_game_json().expect("game DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "grid4");
        assert_eq!(value["topology"], "square");
        assert_eq!(value["counts"]["transition_edges"], 48);
        assert_eq!(value["counts"]["policy_edges"], 12);
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@policy_c0_0")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@action_c0_0_down")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@agent_policy_binding")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("hikari_policy.action_c0_0_down")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("hikari_state")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@trace_0")
        );
        assert_eq!(value["counts"]["concept_edges"], 34);
        assert_eq!(value["counts"]["simulation_steps"], 6);
        assert_eq!(value["trace"][5]["terminal"], true);
        assert!(value["evolution"].as_array().unwrap().len() >= 5);
        assert!(
            value["evolution"][0]["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("Initial HyMeKo RL scaffold")
        );
        assert!(
            value["evolution"].as_array().unwrap().last().unwrap()["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@select_c0_0")
        );
    }

    #[test]
    fn larger_grid_world_is_gated_and_traced() {
        let json = solved_world_json("grid6").expect("larger grid DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "grid6");
        assert_eq!(value["topology"], "square");
        assert!(value["counts"]["playable_states"].as_u64().unwrap() > 16);
        assert!(value["counts"]["transition_edges"].as_u64().unwrap() > 48);
        assert_eq!(
            value["trace"].as_array().unwrap().last().unwrap()["terminal"],
            true
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("world \"grid6\"")
        );
    }

    #[test]
    fn hex_world_uses_hex_actions_and_signed_state_ids() {
        let json = solved_world_json("hex").expect("hex DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "hex");
        assert_eq!(value["topology"], "hex");
        assert!(value["counts"]["playable_states"].as_u64().unwrap() > 24);
        assert!(value["counts"]["transition_edges"].as_u64().unwrap() > 100);
        assert_eq!(
            value["trace"].as_array().unwrap().last().unwrap()["terminal"],
            true
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@action_hm3_0_south_east")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("topology \"hex\"")
        );
    }

    #[test]
    fn generated_square_world_uses_requested_dimensions() {
        let json =
            generated_world_json("square", 8, 5).expect("generated grid DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "custom_grid_8x5");
        assert_eq!(value["width"], 8);
        assert_eq!(value["height"], 5);
        assert_eq!(value["topology"], "square");
        assert_eq!(
            value["trace"].as_array().unwrap().last().unwrap()["terminal"],
            true
        );
    }

    #[test]
    fn generated_hex_world_uses_dimension_as_radius() {
        let json = generated_world_json("hex", 4, 2).expect("generated hex DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "custom_hex_r4");
        assert_eq!(value["width"], 9);
        assert_eq!(value["height"], 9);
        assert_eq!(value["topology"], "hex");
        assert!(value["counts"]["transition_edges"].as_u64().unwrap() > 200);
        assert_eq!(
            value["trace"].as_array().unwrap().last().unwrap()["terminal"],
            true
        );
    }

    #[test]
    fn randomized_single_agent_world_reports_seeded_start() {
        let json = randomized_world_json("grid6", 42).expect("randomized grid should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert!(value["world_id"].as_str().unwrap().contains("seed42"));
        assert_ne!(value["start_state"], "c0_0");
        assert_eq!(value["trace"][0]["state"], value["start_state"]);
    }

    #[test]
    fn seeded_adversarial_world_randomizes_pressure_points() {
        let json =
            adversarial_world_seed_json(77).expect("seeded adversarial DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "adversarial_ca_seed77");
        assert_ne!(value["start_state"], "c0_6");
        assert!(value["trace"][0]["rival_state"].as_str().is_some());
        assert!(
            value["trace"][0]["automaton_cells"]
                .as_array()
                .unwrap()
                .len()
                >= 3
        );
    }

    #[test]
    fn adversarial_world_models_competitor_and_automaton() {
        let json = adversarial_world_json().expect("adversarial DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "adversarial_ca");
        assert_eq!(value["topology"], "square");
        assert!(value["trace"][0]["rival_state"].as_str().is_some());
        assert!(value["trace"][0]["automaton_cells"].as_array().is_some());
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("rival_kage")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("arena_automaton")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@competitive_binding")
        );
    }

    #[test]
    fn virus_world_models_cleanse_and_infection_front() {
        let json = virus_world_json().expect("virus DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "virus_ca");
        assert!(value["trace"][0]["virus_cells"].as_array().is_some());
        assert!(value["trace"][0]["cleansed_cells"].as_array().is_some());
        assert!(value["trace"][0]["infection_count"].as_u64().is_some());
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("virus_automaton")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@infection_front_")
        );
        assert!(
            value["hymeko_source"]
                .as_str()
                .unwrap()
                .contains("@cleanse_")
        );
    }

    #[test]
    fn seeded_virus_world_changes_start_and_keeps_virus_layer() {
        let json = virus_world_seed_json(19).expect("seeded virus DTO should serialize");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(value["gate"]["accepted"], true);
        assert_eq!(value["world_id"], "virus_ca_seed19");
        assert!(value["trace"][0]["virus_cells"].as_array().unwrap().len() >= 1);
        assert_eq!(value["trace"][0]["state"], value["start_state"]);
    }
}
