//! Compile GLSL `.comp` shaders in `shaders/` to SPIR-V `.spv` in the
//! crate's `OUT_DIR` at build time. The kernels include the resulting
//! `.spv` bytes via `include_bytes!` and hand them to `vulkano` at
//! runtime.
//!
//! Replaces the `vulkano_shaders::shader!` proc-macro path, which
//! had a name-resolution snag in this workspace.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shader_dir = manifest_dir.join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=shaders");

    if !shader_dir.exists() {
        return;
    }

    for entry in fs::read_dir(&shader_dir).expect("read shader dir") {
        let entry = entry.expect("shader entry");
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("comp") {
            continue;
        }
        println!("cargo:rerun-if-changed={}", path.display());

        let stem = path.file_stem().unwrap().to_str().unwrap();
        let out_path = out_dir.join(format!("{stem}.spv"));

        let status = Command::new("glslc")
            .arg("-fshader-stage=compute")
            .arg("-O")
            .arg(&path)
            .arg("-o")
            .arg(&out_path)
            .status()
            .expect("glslc not found on PATH; install Vulkan SDK / shaderc");
        if !status.success() {
            panic!("glslc failed for {}", path.display());
        }
    }
}
