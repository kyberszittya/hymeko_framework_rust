//! HyMeKo WASM-editor Axum server — static file serving + tiny REST layer
//! for reading/writing `.hymeko` / `.sysml` files from the browser canvas.
//!
//! The binary in `bin/hymeko_server.rs` binds to `127.0.0.1:3000` and
//! hosts everything under a single router; the same router is exposed
//! here as [`app`] so tests can invoke it via `tower::ServiceExt` without
//! actually listening on a socket.

use std::path::{Path, PathBuf};

use axum::{
    Json, Router,
    extract::{Path as AxPath, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
};
use serde::Serialize;
use tower_http::{cors::CorsLayer, services::ServeDir};

/// Application state handed to every route — currently just the
/// filesystem root that `/api/files/:name` is relative to. Defaults to
/// the process working directory.
#[derive(Clone, Debug)]
pub struct AppState {
    pub workspace_root: PathBuf,
}

impl AppState {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
        }
    }

    /// Join a user-supplied filename to the workspace root, rejecting any
    /// path that escapes it (basic path-traversal guard).
    fn safe_join(&self, name: &str) -> Result<PathBuf, &'static str> {
        let candidate = self.workspace_root.join(name);
        let root = self
            .workspace_root
            .canonicalize()
            .unwrap_or_else(|_| self.workspace_root.clone());
        match candidate.canonicalize() {
            Ok(c) if c.starts_with(&root) => Ok(c),
            // If the file doesn't exist yet (POST creating it), check by
            // component — reject `..` segments.
            _ if !name.split('/').any(|c| c == "..") => Ok(candidate),
            _ => Err("path escapes workspace root"),
        }
    }
}

#[derive(Serialize)]
struct WorkspaceListing {
    files: Vec<String>,
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/workspace", get(list_workspace))
        .route("/api/files/:name", get(read_file).post(write_file))
        .nest_service("/static", ServeDir::new(state.workspace_root.join("static")))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> &'static str {
    "OK"
}

async fn list_workspace(State(s): State<AppState>) -> Json<WorkspaceListing> {
    let mut files: Vec<String> = Vec::new();
    if let Ok(mut entries) = tokio::fs::read_dir(&s.workspace_root).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".hymeko") || name.ends_with(".sysml") {
                files.push(name);
            }
        }
    }
    files.sort();
    Json(WorkspaceListing { files })
}

async fn read_file(
    State(s): State<AppState>,
    AxPath(name): AxPath<String>,
) -> Result<String, (StatusCode, String)> {
    let path = s
        .safe_join(&name)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))
}

async fn write_file(
    State(s): State<AppState>,
    AxPath(name): AxPath<String>,
    body: String,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let path = s
        .safe_join(&name)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if let Some(parent) = Path::new(&path).parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    }
    tokio::fs::write(&path, body.as_bytes())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok((StatusCode::CREATED, "written"))
}
