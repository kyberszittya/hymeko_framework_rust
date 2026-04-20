//! Gazebo world emission — T11 (Paper 2).
//!
//! As of 2026-04-19 the emitter is **template-driven**: output is
//! produced by rendering `<workspace>/transforms/gazebo/template.world.sdf`
//! against the query set in `<workspace>/transforms/gazebo/queries.hymeko`.
//! The Rust-side `push_str` emitter that lived here through the first
//! T11 slice has been retired — all format-shaping decisions now live
//! in the template file, not in Rust code.
//!
//! The public entry point [`generate_gazebo_world`] is preserved for
//! call-site compatibility (tests, docs, the launch-bundle builder in
//! `test_gazebo_sim_launch`); internally it composes a
//! [`TransformConfig`] with the requested `robot_name` + `world_name`
//! and hands off to [`TransformRegistry::render_from_templates`].

use std::path::{Path, PathBuf};

use hymeko::ir::ir::Ir;

use crate::traits::NameResolver;
use crate::transforms::{TransformConfig, TransformRegistry};

/// Resolve the workspace-level `transforms/` directory. This uses
/// `CARGO_MANIFEST_DIR`, which resolves at compile time to
/// `hymeko_query/`; the transforms dir is one level up.
///
/// For deployments where the binary ships without the templates
/// alongside, callers should use
/// [`TransformRegistry::render_from_templates`] directly and pass a
/// resolved path instead.
pub fn default_transforms_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("hymeko_query has a parent")
        .join("transforms")
}

/// Generate a full SDF 1.8 world document for a HyMeKo-described robot.
///
/// Data-driven — the output comes from
/// `transforms/gazebo/template.world.sdf` + `queries.hymeko`, not a
/// Rust string builder.
pub fn generate_gazebo_world<R: NameResolver>(
    ir: &Ir,
    resolver: &R,
    robot_name: &str,
    world_name: &str,
) -> String {
    let reg = TransformRegistry::default();
    let cfg = TransformConfig::default()
        .with_name(robot_name)
        .with_option("world_name", world_name);
    reg.render_from_templates("gazebo", ir, resolver, &cfg, &default_transforms_root())
        .expect("gazebo transform registered in TransformRegistry::default")
        .unwrap_or_else(|e| panic!("gazebo template render failed: {e}"))
}
