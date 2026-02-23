param(
    [Parameter(Position = 0)]
    [string]$Task = "help"
)

function Show-Help {
    Write-Host "Hymeko Framework - Development Tasks (PowerShell)" -ForegroundColor Cyan
    Write-Host "=====================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\dev.ps1 [task]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available tasks:" -ForegroundColor Yellow
    Write-Host "  build       - Build release binary"
    Write-Host "  test        - Run all tests"
    Write-Host "  fmt         - Format code"
    Write-Host "  fmt-check   - Check code formatting"
    Write-Host "  lint        - Run clippy linter"
    Write-Host "  clean       - Clean build artifacts"
    Write-Host "  coverage    - Generate code coverage report"
    Write-Host "  release     - Create a release build"
    Write-Host "  doc         - Generate documentation"
    Write-Host "  check       - Run all checks (test, fmt, lint)"
    Write-Host "  help        - Show this help message"
    Write-Host ""
}

function Invoke-Task {
    param([string]$TaskName)

    switch ($TaskName) {
        "build" {
            Write-Host "Building project..." -ForegroundColor Green
            cargo build --workspace --all-targets --verbose
        }
        "test" {
            Write-Host "Running tests..." -ForegroundColor Green
            cargo test --workspace --all-targets --verbose
        }
        "fmt" {
            Write-Host "Formatting code..." -ForegroundColor Green
            cargo fmt --all
            Write-Host "✓ Code formatted" -ForegroundColor Green
        }
        "fmt-check" {
            Write-Host "Checking code formatting..." -ForegroundColor Green
            cargo fmt --all -- --check
        }
        "lint" {
            Write-Host "Running clippy..." -ForegroundColor Green
            cargo clippy --workspace --all-targets -- -D warnings
        }
        "clean" {
            Write-Host "Cleaning build artifacts..." -ForegroundColor Green
            cargo clean
            Write-Host "✓ Clean complete" -ForegroundColor Green
        }
        "coverage" {
            Write-Host "Generating code coverage..." -ForegroundColor Green
            if (Get-Command cargo-tarpaulin -ErrorAction SilentlyContinue) {
                cargo tarpaulin --workspace --all-targets --out Html
            } else {
                Write-Host "cargo-tarpaulin not installed. Install with:" -ForegroundColor Yellow
                Write-Host "cargo install cargo-tarpaulin" -ForegroundColor Yellow
            }
        }
        "release" {
            Write-Host "Building release binary..." -ForegroundColor Green
            cargo build --workspace --all-targets --release --verbose
            Write-Host "✓ Release build complete" -ForegroundColor Green
        }
        "doc" {
            Write-Host "Generating documentation..." -ForegroundColor Green
            cargo doc --no-deps --open
        }
        "check" {
            Write-Host "Running all checks..." -ForegroundColor Green
            cargo fmt --all -- --check
            if ($LASTEXITCODE -ne 0) {
                Write-Host "✗ Formatting check failed" -ForegroundColor Red
                exit 1
            }
            cargo clippy --workspace --all-targets -- -D warnings
            if ($LASTEXITCODE -ne 0) {
                Write-Host "✗ Lint check failed" -ForegroundColor Red
                exit 1
            }
            cargo test --workspace --all-targets
            if ($LASTEXITCODE -ne 0) {
                Write-Host "✗ Tests failed" -ForegroundColor Red
                exit 1
            }

            Write-Host "✓ All checks passed!" -ForegroundColor Green
        }
        "help" {
            Show-Help
        }
        default {
            Write-Host "Unknown task: $TaskName" -ForegroundColor Red
            Show-Help
            exit 1
        }
    }
}

# Main
Invoke-Task $Task
