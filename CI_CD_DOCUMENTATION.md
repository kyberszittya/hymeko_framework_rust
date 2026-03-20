# CI/CD Pipeline Documentation

This project uses GitHub Actions for continuous integration and continuous deployment.

## Workflows

### 1. **CI Workflow** (`.github/workflows/ci.yml`)
Runs on every push to `master`, `main`, `develop` branches and on all pull requests.

**Jobs:**
- **Workspace Tests:** Runs `cargo test --workspace` across Linux, Windows, and macOS using both stable and nightly toolchains. Nightly additionally exercises `--all-features`.
- **Package Build & Test:** Ubuntu matrix that builds and tests each crate (`hymeko`, `hymeko_core`, `hymeko_daemon`, `hymeko_py`, `parser`) independently so regressions in one package cannot hide behind workspace defaults.
- **Coverage:** Uses Tarpaulin to create per-package XML + HTML reports, uploads each report to Codecov with dedicated flags, and publishes the HTML bundle as an artifact.
- **Build:** Produces release-mode artifacts after the test suites succeed.

**Coverage Job Details:**
- Iterates over every crate, running Tarpaulin twice (XML + HTML) so each package has its own `coverage/xml/<crate>.xml` and HTML viewer.
- Uploads all XML reports to Codecov with matching flags so the dashboards show per-crate deltas and status checks.
- Publishes the combined HTML directory (`coverage/html/`) as a downloadable artifact for offline inspection.
- Keeps the existing 300-second timeout and verbose logging per Tarpaulin invocation.

**Artifacts:**
- Release binaries are uploaded as GitHub Actions artifacts
- Coverage HTML directory saved for 30 days

### 2. **Release Workflow** (`.github/workflows/release.yml`)
Automatically creates releases and builds binaries when a tag is pushed (e.g., `v0.1.0`).

**Artifacts:**
- `parser-linux-x86_64` (Ubuntu)
- `parser-windows-x86_64.exe` (Windows)
- `parser-macos-x86_64` (macOS)

### 3. **Security Audit** (`.github/workflows/security-audit.yml`)
Runs daily and on every push to check for known security vulnerabilities in dependencies.

### 4. **Dependency Updates** (`.github/workflows/update-dependencies.yml`)
Runs every Monday to check for dependency updates and creates a pull request if updates are available.

## How to Use

### Running Tests Locally
```bash
cargo test --verbose --all
```

### Code Formatting
```bash
cargo fmt --all
```

### Linting
```bash
cargo clippy --all --all-targets -- -D warnings
```

### Creating a Release
1. Update version in `Cargo.toml` files
2. Create a git tag: `git tag v0.1.0`
3. Push the tag: `git push origin v0.1.0`
4. GitHub Actions will automatically build and create a release

### Code Coverage
Coverage reports are generated using `cargo-tarpaulin`. The CI job loops over every crate, stores XML outputs in `coverage/xml/`, HTML viewers in `coverage/html/`, and uploads each XML file to Codecov under a dedicated flag.

## Prerequisites

- GitHub repository with Actions enabled
- Codecov account (optional, for coverage reports)

## Security

All workflows use GitHub's default `GITHUB_TOKEN` for authentication. No additional secrets are required for basic functionality.

## Future Enhancements

- Add Docker image builds and pushes
- Add deployment to crates.io
- Add scheduled performance benchmarking
- Add integration tests with external services

## Python Packaging Integration

- CI/CD now builds and tests Python packages using maturin.
- `ci.yml` runs maturin build/test and uploads wheel artifacts.
- `release.yml` builds Python wheels and optionally publishes to PyPI.
- See `hymeko_py` crate for Python bindings and packaging details.
