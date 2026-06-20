//! Reinforcement-learning sketch across AKOIRE, HyMeKo, and HIVE.
//!
//! Run with:
//!     cargo run -p akoire --example rl_hive_policy
//!
//! The demo is intentionally small: AKOIRE proposes action refinements, HyMeKo
//! gate-keeps the proposed `.hymeko` source, and HIVE commits only accepted
//! experience transitions into the canonical signed-hypergraph store.

use std::collections::BTreeMap;

use akoire::{EvalOutcome, HymekoEngine, Refinement};
use hymeko_hive::{
    AttributeValue, HiveDelta, HiveNode, HiveQuery, HiveRelation, HiveStore, HiveTransaction,
    NodeId, Sign,
};

const ALPHA: f64 = 0.55;
const GAMMA: f64 = 0.90;

#[derive(Debug, Clone)]
struct Proposal {
    name: &'static str,
    state: &'static str,
    action: &'static str,
    next_state: &'static str,
    reward: f64,
    source: String,
}

fn main() {
    let mut engine = HymekoEngine::new();
    let mut hive = HiveStore::new();
    let mut hive_seq = 0_u64;
    let mut q = QTable::default();

    commit(
        &mut hive,
        &mut hive_seq,
        "rl-bootstrap",
        vec![
            node("s0", "rl_state"),
            node("s1", "rl_state"),
            node("s2_goal", "rl_state"),
            node("advance", "rl_action"),
            node("wait", "rl_action"),
        ],
    );

    println!("HyMeKo / AKOIRE / HIVE reinforcement-learning demo");
    println!("task: learn that advance -> advance reaches the goal\n");
    println!("initial HIVE hash: {}", hive.state_hash());

    let mut accepted_steps = 0_usize;
    for (round, proposal) in proposals().into_iter().enumerate() {
        println!(
            "\nround {} | AKOIRE proposes {:<14} {} --{}--> {}",
            round + 1,
            proposal.name,
            proposal.state,
            proposal.action,
            proposal.next_state
        );

        match engine.evaluate(&Refinement(proposal.source.clone())) {
            EvalOutcome::Accepted { generation } => {
                accepted_steps += 1;
                let old_q = q.value(proposal.state, proposal.action);
                let new_q = q.update(
                    proposal.state,
                    proposal.action,
                    proposal.reward,
                    proposal.next_state,
                );

                commit(
                    &mut hive,
                    &mut hive_seq,
                    format!("experience-{accepted_steps}"),
                    experience_deltas(accepted_steps, &proposal, new_q),
                );

                println!(
                    "  HyMeKo gate: accepted at ambience generation {generation}; reward {:+.2}",
                    proposal.reward
                );
                println!(
                    "  Q({}, {}) {:.3} -> {:.3}",
                    proposal.state, proposal.action, old_q, new_q
                );
                println!(
                    "  HIVE commit: generation {}, hash {}",
                    hive.generation(),
                    hive.state_hash()
                );
            }
            EvalOutcome::Rejected(feedback) => {
                println!("  HyMeKo gate: rejected; HIVE unchanged");
                println!("  feedback: {}", one_line(&feedback.message));
            }
        }
    }

    let experiences = hive.query(&HiveQuery::RelationsByType("rl_experience".to_string()));
    let q_values = hive.query(&HiveQuery::NodesWithAttribute("q_advance".to_string()));

    println!("\nsummary");
    println!(
        "  accepted experience transitions: {}",
        experiences.relations.len()
    );
    println!(
        "  states carrying learned q_advance: {}",
        q_values.nodes.len()
    );
    println!("  final best action at s0: {}", q.best_action("s0"));
    println!("  final Q table:\n{}", q.render());
}

fn proposals() -> Vec<Proposal> {
    let base = "RlTask {\n  s0;\n  s1;\n  s2_goal;\n  advance;\n  wait;\n}";
    vec![
        Proposal {
            name: "syntax-slip",
            state: "s0",
            action: "advance",
            next_state: "s1",
            reward: 0.0,
            source: format!("{base}\n@bad_step : {{ }}"),
        },
        Proposal {
            name: "stall",
            state: "s0",
            action: "wait",
            next_state: "s0",
            reward: -0.15,
            source: format!("{base}\n@step0 : s0, wait, s0 {{ }}"),
        },
        Proposal {
            name: "advance-1",
            state: "s0",
            action: "advance",
            next_state: "s1",
            reward: 0.20,
            source: format!("{base}\n@step1 : s0, advance, s1 {{ }}"),
        },
        Proposal {
            name: "advance-2",
            state: "s1",
            action: "advance",
            next_state: "s2_goal",
            reward: 1.00,
            source: format!("{base}\n@step2 : s1, advance, s2_goal {{ }}"),
        },
        Proposal {
            name: "exploit",
            state: "s0",
            action: "advance",
            next_state: "s1",
            reward: 0.20,
            source: format!("{base}\n@step3 : s0, advance, s1 {{ }}"),
        },
    ]
}

fn node(id: &str, ty: &str) -> HiveDelta {
    HiveDelta::AddNode(HiveNode::new(id, ty))
}

fn experience_deltas(step: usize, proposal: &Proposal, q_value: f64) -> Vec<HiveDelta> {
    let relation_id = format!("xp_{step}");
    vec![
        HiveDelta::AddRelation({
            let mut relation = HiveRelation::new(
                relation_id.clone(),
                "rl_experience",
                vec![
                    (Sign::Minus, NodeId::from(proposal.state)),
                    (Sign::Plus, NodeId::from(proposal.action)),
                    (Sign::Plus, NodeId::from(proposal.next_state)),
                ],
            );
            relation
                .attributes
                .insert("reward".to_string(), AttributeValue::Float(proposal.reward));
            relation
        }),
        HiveDelta::AttachNodeAttribute {
            id: NodeId::from(proposal.state),
            key: format!("q_{}", proposal.action),
            value: AttributeValue::Float(q_value),
        },
        HiveDelta::AttachRelationAttribute {
            id: relation_id.into(),
            key: "source".to_string(),
            value: AttributeValue::Text(proposal.name.to_string()),
        },
    ]
}

fn commit(hive: &mut HiveStore, seq: &mut u64, id: impl Into<String>, deltas: Vec<HiveDelta>) {
    let tx = HiveTransaction::new(
        id.into(),
        hive.state_hash(),
        "akoire-rl-policy",
        *seq,
        deltas,
    );
    hive.commit(tx).expect("demo transaction should commit");
    *seq += 1;
}

fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Default)]
struct QTable {
    values: BTreeMap<(&'static str, &'static str), f64>,
}

impl QTable {
    fn value(&self, state: &'static str, action: &'static str) -> f64 {
        *self.values.get(&(state, action)).unwrap_or(&0.0)
    }

    fn update(
        &mut self,
        state: &'static str,
        action: &'static str,
        reward: f64,
        next_state: &'static str,
    ) -> f64 {
        let old = self.value(state, action);
        let best_next = self
            .value(next_state, "advance")
            .max(self.value(next_state, "wait"));
        let new = old + ALPHA * (reward + GAMMA * best_next - old);
        self.values.insert((state, action), new);
        new
    }

    fn best_action(&self, state: &'static str) -> &'static str {
        if self.value(state, "advance") >= self.value(state, "wait") {
            "advance"
        } else {
            "wait"
        }
    }

    fn render(&self) -> String {
        ["s0", "s1", "s2_goal"]
            .into_iter()
            .map(|state| {
                format!(
                    "    {state:<7} advance={:.3} wait={:.3}",
                    self.value(state, "advance"),
                    self.value(state, "wait")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
