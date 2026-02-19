.PHONY: help build test fmt lint clean coverage release

help:
	@echo "Hymeko Framework - Development Tasks"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build release binary"
	@echo "  test        - Run all tests"
	@echo "  test-watch  - Run tests in watch mode"
	@echo "  fmt         - Format code"
	@echo "  fmt-check   - Check code formatting"
	@echo "  lint        - Run clippy linter"
	@echo "  clean       - Clean build artifacts"
	@echo "  coverage    - Generate code coverage report"
	@echo "  release     - Create a release build"
	@echo "  doc         - Generate and open documentation"
	@echo "  check       - Run all checks (test, fmt, lint)"
	@echo ""

build:
	cargo build --verbose

test:
	cargo test --verbose --all

test-watch:
	cargo watch -x "test --all"

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --all --all-targets -- -D warnings

clean:
	cargo clean

coverage:
	cargo tarpaulin --out Html --all

release:
	cargo build --release --verbose --all

doc:
	cargo doc --no-deps --open

check: fmt-check lint test
	@echo "✓ All checks passed!"

.DEFAULT_GOAL := help

