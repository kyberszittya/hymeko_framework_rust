# CI/CD Pipeline Documentation

This project uses GitHub Actions for continuous integration and continuous deployment.

## Workflows

### 1. **CI Workflow** (`.github/workflows/ci.yml`)
Runs on every push to `master`, `main`, `develop` branches and on all pull requests.

**Jobs:**
- **Test**: Runs tests on Linux, Windows, and macOS with stable and nightly Rust
- **Rustfmt**: Checks code formatting
- **Clippy**: Runs linter with warnings as errors
- **Coverage**: Generates code coverage reports and uploads to Codecov
- **Build**: Creates release build artifacts

**Artifacts:**
- Release binaries are uploaded as GitHub Actions artifacts

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
Coverage reports are generated using `cargo-tarpaulin` and uploaded to Codecov.

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

