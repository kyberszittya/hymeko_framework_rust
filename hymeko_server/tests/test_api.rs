//! In-memory integration tests for the Axum app — no socket bind.

use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use hymeko_server::{AppState, app};
use tower::ServiceExt;

fn tmpdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hymeko_server_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[tokio::test]
async fn health_returns_ok() {
    let state = AppState::new(tmpdir());
    let response = app(state)
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"OK");
}

#[tokio::test]
async fn workspace_listing_is_empty_for_fresh_dir() {
    let dir = tmpdir();
    let state = AppState::new(dir);
    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/api/workspace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["files"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn write_then_read_roundtrips() {
    let dir = tmpdir();
    let state = AppState::new(dir.clone());

    // POST /api/files/foo.hymeko
    let response = app(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/files/foo.hymeko")
                .body(Body::from("hello world"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 201);

    // GET /api/files/foo.hymeko
    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/api/files/foo.hymeko")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(&body[..], b"hello world");
}

#[tokio::test]
async fn workspace_listing_shows_hymeko_files() {
    let dir = tmpdir();
    std::fs::write(dir.join("a.hymeko"), "").unwrap();
    std::fs::write(dir.join("b.sysml"), "").unwrap();
    std::fs::write(dir.join("ignored.txt"), "").unwrap();

    let state = AppState::new(dir);
    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/api/workspace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let files: Vec<&str> = json["files"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(files.contains(&"a.hymeko"));
    assert!(files.contains(&"b.sysml"));
    assert!(!files.contains(&"ignored.txt"));
}

#[tokio::test]
async fn path_traversal_is_rejected() {
    let dir = tmpdir();
    let state = AppState::new(dir);
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/files/..%2Fescape")
                .body(Body::from("x"))
                .unwrap(),
        )
        .await
        .unwrap();
    // Either 400 (traversal caught at join) or 500 (filesystem rejected).
    // Both are acceptable as long as the write doesn't land outside root.
    assert!(
        response.status().is_client_error() || response.status().is_server_error(),
        "expected 4xx or 5xx for path-traversal attempt, got {}",
        response.status()
    );
}
