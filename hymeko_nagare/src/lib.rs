//! # HymeKo-Nagare (流れ): paradigm-native dataflow ML for signed hypergraphs.
//!
//! A Rust ML framework where:
//! * the universal data type is a **SoA cycle pool**, not a dense tensor;
//! * every operator is a **(forward, backward) pair** with closed-form
//!   gradients, not an autograd-traced graph node;
//! * gradients are **multivector-valued** (Clifford algebra Cl(p, q))
//!   so the sign structure of signed graphs is first-class;
//! * parallelism is **MapReduce over cycles** (commutative, embarrassingly
//!   parallel) rather than tensor-batch.
//!
//! The mathematical foundation: signed-hypergraph forward computation
//! factors as
//!
//! ```text
//!   F(G, X) = Σ_c f_θ(c, X)
//!   ∂F/∂θ   = Σ_c ∂f_θ(c, X) / ∂θ
//! ```
//!
//! Both sums are commutative over the cycle pool `𝒞`, hence
//! evaluation order is irrelevant up to float-summation order
//! (mitigated with Kahan when needed). Gradient accumulation is a
//! commutative reduce; lockless atomic-add on shared parameters is
//! sound.
//!
//! See `docs/plans/2026-05-11-hymeko-nagare-flow/plan.{tex,pdf}` for
//! the full paradigm + mathematical write-up.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod holonomy;
pub mod ops;
pub mod optimizer;
pub mod runtime;

pub use ops::adam::{AdamState, adam_step};
pub use ops::catmull_rom::{
    CatmullRomBackward, CatmullRomCache, ChebyshevCrBackward, catmull_rom_backward,
    catmull_rom_forward, chebyshev_control_points, chebyshev_cr_backward, chebyshev_cr_forward,
    chebyshev_deploy_backward, chebyshev_deploy_forward, chebyshev_knot_basis,
};
pub use ops::cayley_rotor::{cayley_rotor_backward, cayley_rotor_forward};
pub use ops::clifford_fir::{CliffordFIR, clifford_fir_backward, clifford_fir_forward};
pub use ops::fsr_mixer::{FsrMixer, FsrMixerBackward, FsrMixerCache, FsrRoute};
pub use ops::fused_entropy_update::{
    FusedEntropyUpdateBackward, FusedEntropyUpdateShape, fused_entropy_update_backward,
    fused_entropy_update_forward,
};
pub use ops::linear::{LinearLayer, linear_backward, linear_forward};
pub use ops::loss::{bce_with_logits_backward, bce_with_logits_forward};
pub use ops::project_alpha_mix::{
    ProjectAlphaMixBackward, ProjectAlphaMixShape, project_alpha_mix_backward,
    project_alpha_mix_forward,
};
pub use ops::scatter::{scatter_mean_backward, scatter_mean_forward};
pub use ops::signed_scatter::{
    SignedScatterLanes, SignedScatterLayout, signed_scatter_backward, signed_scatter_forward,
};
pub use runtime::NagareRuntime;
