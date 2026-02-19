# Hymeko Framework - Development Guide

## Quick Start

### Prerequisites
- Rust 1.56+ (get it at https://rustup.rs/)
- Cargo (comes with Rust)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/hakiko/hymeko_framework.git
cd hymeko_framework

# Build the project
cargo build --release
```

## Development Workflow

### Using Development Scripts

#### On Windows (PowerShell):
```powershell
.\dev.ps1 test      # Run tests
.\dev.ps1 fmt       # Format code
.\dev.ps1 lint      # Run linter
.\dev.ps1 check     # Run all checks
```

#### On Linux/macOS (Bash):
```bash
./dev.sh test       # Run tests
./dev.sh fmt        # Format code
./dev.sh lint       # Run linter
./dev.sh check      # Run all checks
```

#### Using Make (Unix-like systems):
```bash
make test           # Run tests
make fmt            # Format code
make lint           # Run linter
make check          # Run all checks
```

#### Using Cargo directly:
```bash
cargo test --all                          # Run all tests
cargo fmt --all                           # Format code
cargo clippy --all --all-targets          # Run linter
cargo build --release                     # Build release binary
```

## CI/CD Pipeline

This project uses GitHub Actions for automated testing and releases.

### Workflows

#### 1. Continuous Integration (CI)
**Triggered:** On push to `master`/`main`/`develop` or pull requests

**Runs:**
- Tests on Linux, Windows, and macOS
- Tests with stable and nightly Rust
- Code formatting check (rustfmt)
- Linting (clippy)
- Code coverage report (tarpaulin → Codecov)
- Release build

#### 2. Security Audit
**Triggered:** Daily + on every push

**Runs:**
- Checks for known vulnerabilities in dependencies using `cargo audit`

#### 3. Dependency Updates
**Triggered:** Every Monday

**Runs:**
- Updates dependencies
- Runs tests with updated versions
- Creates a PR if updates are available

#### 4. Release
**Triggered:** When a tag is pushed (e.g., `git tag v0.1.0`)

**Builds and uploads:**
- Linux binary (`parser-linux-x86_64`)
- Windows binary (`parser-windows-x86_64.exe`)
- macOS binary (`parser-macos-x86_64`)

### Creating a Release

1. Update version numbers in all `Cargo.toml` files
2. Commit changes: `git commit -am "chore: bump version to v0.2.0"`
3. Create a tag: `git tag v0.2.0`
4. Push: `git push origin main && git push origin v0.2.0`
5. GitHub Actions will automatically build and create a release on GitHub

## Code Quality Standards

### Formatting
All code must be formatted with `rustfmt`:
```bash
cargo fmt --all
```

### Linting
All code must pass clippy with no warnings:
```bash
cargo clippy --all --all-targets -- -D warnings
```

### Testing
All tests must pass:
```bash
cargo test --all
```

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
set -e
cargo fmt --all -- --check
cargo clippy --all --all-targets -- -D warnings
cargo test --all
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Troubleshooting

### Build Issues

**LALRPOP errors:**
```bash
cargo clean
cargo build
```

**Dependency conflicts:**
```bash
cargo update
cargo test
```

### Test Failures

Run tests with backtrace:
```bash
RUST_BACKTRACE=1 cargo test --all
```

Run a specific test:
```bash
cargo test --lib test_name
```

## Project Structure

```
hymeko_framework/
├── .github/workflows/          # CI/CD workflows
│   ├── ci.yml                  # Main CI pipeline
│   ├── release.yml             # Release workflow
│   ├── security-audit.yml      # Security checks
│   └── update-dependencies.yml # Dependency updates
├── hymeko/
│   ├── parser/                 # Main parser crate
│   │   ├── src/
│   │   ├── tests/
│   │   ├── data/
│   │   └── Cargo.toml
│   └── ...
├── src/                        # Root package
├── Cargo.toml                  # Root workspace
├── Cargo.lock                  # Locked dependencies
├── dev.sh                      # Unix development script
├── dev.ps1                     # Windows development script
├── Makefile                    # Build automation
└── CI_CD_DOCUMENTATION.md      # Detailed CI/CD docs
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run all checks: `./dev.sh check` (or equivalent on Windows)
5. Commit and push: `git push origin feature/my-feature`
6. Create a pull request

## Additional Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Cargo Documentation](https://doc.rust-lang.org/cargo/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [CI/CD Details](./CI_CD_DOCUMENTATION.md)

