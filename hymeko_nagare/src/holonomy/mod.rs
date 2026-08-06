//! Holonomy local-learning subsystem.
//!
//! The library form of the Nagare holonomy learning experiments
//! (2026-07-01/02 report line): quaternion periodic feature lift over
//! point sets ([`features`]), global structural pooling ([`pooling`]),
//! holonomy-informed projection bases ([`projection`], stacked on the
//! generic [`crate::ops::project_alpha_mix`] kernel), shared metrics
//! ([`metrics`]), toy dataset generators ([`datasets`]), and the
//! entropy-pool local learner ([`learner`]).
//!
//! This package mirrors the planned `nagare-holonomy-learn` extraction
//! layout (`reports/2026-07-02-nagare-holonomy-learning-repo-report.md`)
//! so the eventual repository split is a move, not a rewrite. Experiment
//! orchestration (stress grids, CLI, JSON emission) stays in
//! `examples/`; only substrate lives here.

pub mod datasets;
pub mod features;
pub mod learner;
pub mod metrics;
pub mod pooling;
pub mod projection;

pub use datasets::{Dataset, Task, corrupt_dataset, gather_batch, make_dataset};
pub use features::{VERTEX_FEATURES, quaternion_periodic_features};
pub use learner::{EntropyPoolLocalLearner, GateMode, LOCAL_FEATURES, evaluate_local};
pub use metrics::{
    CrossEntropyEval, Metrics, Timing, binary_entropy, clifford_probability_error,
    cross_entropy_eval, entropy2, forward_timing, softmax2,
};
pub use pooling::{POOL_STATS, STRUCTURAL_FEATURES, structural_pool_features};
pub use projection::{
    PROJECTION_ALPHA, PROJECTION_RANK, ProjectionBasis, default_holonomy_basis,
    fit_class_mean_basis,
};
