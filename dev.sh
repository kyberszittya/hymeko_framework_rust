#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default task
TASK="${1:-help}"

show_help() {
    echo -e "${CYAN}Hymeko Framework - Development Tasks${NC}"
    echo -e "${CYAN}=====================================${NC}"
    echo ""
    echo -e "${YELLOW}Usage: ./dev.sh [task]${NC}"
    echo ""
    echo -e "${YELLOW}Available tasks:${NC}"
    echo "  build       - Build release binary"
    echo "  test        - Run all tests"
    echo "  test-watch  - Run tests in watch mode"
    echo "  fmt         - Format code"
    echo "  fmt-check   - Check code formatting"
    echo "  lint        - Run clippy linter"
    echo "  clean       - Clean build artifacts"
    echo "  coverage    - Generate code coverage report"
    echo "  release     - Create a release build"
    echo "  doc         - Generate and open documentation"
    echo "  check       - Run all checks (test, fmt, lint)"
    echo "  help        - Show this help message"
    echo ""
}

run_task() {
    case "$TASK" in
        build)
            echo -e "${GREEN}Building project...${NC}"
            cargo build --workspace --all-targets --verbose
            ;;
        test)
            echo -e "${GREEN}Running tests...${NC}"
            cargo test --workspace --all-targets --verbose
            ;;
        test-watch)
            echo -e "${GREEN}Running tests in watch mode...${NC}"
            if command -v cargo-watch &> /dev/null; then
                cargo watch -x "test --workspace --all-targets"
            else
                echo -e "${YELLOW}cargo-watch not installed. Install with:${NC}"
                echo "cargo install cargo-watch"
                exit 1
            fi
            ;;
        fmt)
            echo -e "${GREEN}Formatting code...${NC}"
            cargo fmt --all
            echo -e "${GREEN}✓ Code formatted${NC}"
            ;;
        fmt-check)
            echo -e "${GREEN}Checking code formatting...${NC}"
            cargo fmt --all -- --check
            ;;
        lint)
            echo -e "${GREEN}Running clippy...${NC}"
            cargo clippy --workspace --all-targets -- -D warnings
            ;;
        clean)
            echo -e "${GREEN}Cleaning build artifacts...${NC}"
            cargo clean
            echo -e "${GREEN}✓ Clean complete${NC}"
            ;;
        coverage)
            echo -e "${GREEN}Generating code coverage...${NC}"
            if command -v cargo-tarpaulin &> /dev/null; then
                cargo tarpaulin --workspace --all-targets --out Html
            else
                echo -e "${YELLOW}cargo-tarpaulin not installed. Install with:${NC}"
                echo "cargo install cargo-tarpaulin"
                exit 1
            fi
            ;;
        release)
            echo -e "${GREEN}Building release binary...${NC}"
            cargo build --workspace --all-targets --release --verbose
            echo -e "${GREEN}✓ Release build complete${NC}"
            ;;
        doc)
            echo -e "${GREEN}Generating documentation...${NC}"
            cargo doc --no-deps --open
            ;;
        check)
            echo -e "${GREEN}Running all checks...${NC}"
            cargo fmt --all -- --check
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Formatting check failed${NC}"
                exit 1
            fi
            cargo clippy --workspace --all-targets -- -D warnings
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Lint check failed${NC}"
                exit 1
            fi
            cargo test --workspace --all-targets
            if [ $? -ne 0 ]; then
                echo -e "${RED}✗ Tests failed${NC}"
                exit 1
            fi
            echo -e "${GREEN}✓ All checks passed!${NC}"
            ;;
        help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown task: $TASK${NC}"
            show_help
            exit 1
            ;;
    esac
}

run_task
