# 📊 Code Coverage Documentation

This document describes the code coverage setup for the Hymeko Framework project.

## Overview

Code coverage is automatically tracked on every CI run using:
- **Tool:** `cargo-tarpaulin` - Fast code coverage tool for Rust
- **Platform:** Codecov.io - Coverage reporting and badges
- **Reporting:** XML (Codecov) and HTML (Artifacts) formats

## Coverage Configuration

### Codecov.yml Configuration

The project includes a `codecov.yml` file that defines:

**Coverage Targets:**
- **Default Target:** 60% coverage minimum for the aggregate project
- **hymeko_core Target:** 65% coverage minimum to protect the core engine APIs
- **parser Target:** 70% coverage minimum for the parser front-end
- **Other Flags:** `hymeko`, `hymeko_daemon`, and `hymeko_py` each inherit a 60% floor so regressions surface quickly.

**Flag Mapping:**
- Each crate uploads an XML file under `coverage/xml/<flag>.xml` (e.g., `coverage/xml/hymeko_core.xml`).
- The Codecov configuration uses matching flag names so dashboards and required checks line up with the CI job output.

**Ignored Paths:**
- Test files (`tests/`)
- Build scripts (`build.rs`)

**Reporting:**
- Project coverage tracking
- Patch coverage (PR changes only)
- Change coverage monitoring

### CI/CD Integration

The CI workflow includes enhanced coverage job with:
- ✅ Per-crate XML report generation saved under `coverage/xml/<crate>.xml` for Codecov uploads
- ✅ Per-crate HTML report generation saved under `coverage/html/<crate>.html` for artifact download
- ✅ Coverage artifact retention (30 days)
- ✅ Timeout handling (300 seconds)
- ✅ Verbose reporting

## Running Coverage Locally

### Generate Coverage Report

```bash
# Install cargo-tarpaulin (if not already installed)
cargo install cargo-tarpaulin

# Generate XML report for a single crate (matches CI flags)
cargo tarpaulin --package hymeko_core --all-targets --out Xml

# Generate HTML report for the same crate
cargo tarpaulin --package hymeko_core --all-targets --out Html

# Loop across all crates just like CI
declare -a crates=(hymeko hymeko_core hymeko_daemon hymeko_py parser)
for crate in "${crates[@]}"; do
  cargo tarpaulin --package "$crate" --all-targets --out Xml --output-dir coverage/xml
  cargo tarpaulin --package "$crate" --all-targets --out Html --output-dir coverage/html || true
done
```

### Using Development Script

```bash
# Run coverage using dev script
./dev.sh coverage    # Unix/macOS
.\dev.ps1 coverage   # Windows
make coverage        # Make
```

## CI/CD Coverage Workflow

### On Every Push/PR

1. **Test Job Runs**
   - Runs all tests on multiple platforms
   - Validates code quality

2. **Coverage Job Runs**
   - Generates per-crate XML reports (`coverage/xml/<crate>.xml`) for Codecov
   - Uploads each XML file with the matching flag so Codecov enforces per-crate gates
   - Generates per-crate HTML reports (`coverage/html/<crate>.html`)
   - Uploads the entire HTML directory as a GitHub Actions artifact

3. **Coverage Comments**
   - Codecov posts coverage comment on PR
   - Shows coverage change percentage
   - Shows if coverage targets are met

4. **Coverage Badge Updated**
   - Badge in README shows latest coverage
   - Updated after each successful run

## Understanding Coverage Reports

### HTML Report

Each Tarpaulin HTML report (local `tarpaulin-report.html`, CI `coverage/html/<crate>.html`) shows:

- **File-by-file coverage:** Each file with line coverage percentage
- **Color coding:**
  - 🟢 Green: Fully covered lines
  - 🟡 Yellow: Partially covered lines
  - 🔴 Red: Uncovered lines
- **Line numbers:** Click to see coverage details
- **Summary:** Overall coverage percentage

### XML Report (Cobertura Format)

Used by Codecov.io for:
- Tracking coverage trends over time
- Comparing branches
- Setting up coverage requirements
- Generating badges

### Codecov Dashboard

Available at: `https://codecov.io/gh/hakiko/hymeko_framework`

Shows:
- Coverage history by commit
- Branch coverage comparison
- File-by-file breakdown
- Coverage trends

## Coverage Targets

### Project Level

**Minimum Coverage:** 60%
- Ensures baseline quality
- Allows for incremental improvement

**Parser Module:** 70%
- Core parsing logic needs higher coverage
- More critical for functionality

### Patch Level (PR Changes)

**Minimum Coverage:** 60%
- New code in PR should have coverage
- Ensures new features are tested

### Change Detection

**Threshold:** 1%
- Alerts if coverage drops more than 1%
- Helps catch regressions

## Improving Coverage

### Steps to Improve Coverage

1. **Identify uncovered code:**
   ```bash
   cargo tarpaulin --out Html --all
   # Open tarpaulin-report.html and look for red lines
   ```

2. **Write tests for uncovered paths:**
   ```bash
   # Add tests in hymeko/parser/tests/
   ```

3. **Run coverage locally:**
   ```bash
   ./dev.sh coverage
   ```

4. **Commit and push:**
   ```bash
   git add tests/
   git commit -m "test: improve coverage for X module"
   git push origin <branch>
   ```

5. **Monitor on Codecov:**
   - Visit: https://codecov.io/gh/hakiko/hymeko_framework
   - Check coverage change on your PR

### Coverage Targets by Module

```
src/lexer/          Target: 80% (critical path)
src/ir/             Target: 75% (transformations)
src/resolve.rs      Target: 70% (symbol resolution)
tests/              Ignored (test code)
```

## Badge Integration

### Add Coverage Badge to README

The README already includes a coverage badge:

```markdown
[![codecov](https://codecov.io/gh/hakiko/hymeko_framework/branch/main/graph/badge.svg?token=YOUR_CODECOV_TOKEN)](https://codecov.io/gh/hakiko/hymeko_framework)
```

The badge shows:
- Current coverage percentage
- Color coded:
  - 🟢 Green: >80%
  - 🟡 Yellow: 60-80%
  - 🔴 Red: <60%

### Setup Codecov Token (Optional)

For private repositories or better integration:

1. Visit: https://codecov.io
2. Sign in with GitHub
3. Find your repository
4. Copy the token
5. Add to GitHub Secrets:
   - Settings → Secrets and variables → Actions
   - Name: `CODECOV_TOKEN`
   - Value: [Your Codecov token]
6. Update workflow to use token:
   ```yaml
   - uses: codecov/codecov-action@v4
     with:
       token: ${{ secrets.CODECOV_TOKEN }}
   ```

## Troubleshooting

### Coverage Report Not Uploading

**Issue:** Coverage upload fails to Codecov

**Solution:**
1. Check internet connection
2. Verify Codecov is accessible
3. Check GitHub Actions logs for error details
4. The workflow continues even if upload fails (fail_ci_if_error: false)

### HTML Report Not Generated

**Issue:** `tarpaulin-report.html` not created

**Solution:**
```bash
# Run tarpaulin manually to debug
cargo tarpaulin --out Html --all --verbose
```

### Coverage Lower Than Expected

**Issue:** Coverage metrics are lower than expected

**Solution:**
1. Check what's being excluded (see codecov.yml)
2. Add tests for uncovered code paths
3. Use `cargo tarpaulin --verbose` for detailed output
4. Check if tests are being run correctly

### Badge Not Showing

**Issue:** Codecov badge shows no data

**Solution:**
1. Ensure codecov.yml is committed
2. Ensure at least one successful coverage upload
3. Wait 5-10 minutes for badge to update
4. Use correct repository path in badge URL

## Best Practices

### Write Testable Code

```rust
// ✅ Good: Testable function
pub fn process_node(node: &Node) -> Result<Output, Error> {
    validate(node)?;
    transform(node)
}

// ❌ Avoid: Hard to test
pub fn process_node(node: &Node) -> Output {
    if let Ok(validated) = validate(node) {
        transform(&validated)
    } else {
        panic!("Invalid node")
    }
}
```

### Test Error Cases

```rust
#[test]
fn test_error_handling() {
    let invalid = create_invalid_input();
    assert!(process(invalid).is_err());
}

#[test]
fn test_success_case() {
    let valid = create_valid_input();
    assert!(process(valid).is_ok());
}
```

### Separate Integration and Unit Tests

```
tests/
├── minimal_tests/           (unit tests)
├── intermediate_tests/      (integration tests)
└── typical_graphs/          (graph-specific tests)
```

## Monitoring Coverage Over Time

### GitHub Actions

1. Go to: https://github.com/hakiko/hymeko_framework/actions
2. Click on "Code Coverage" job
3. View coverage summary

### Codecov Dashboard

1. Visit: https://codecov.io/gh/hakiko/hymeko_framework
2. See coverage trends
3. Compare branches
4. Track improvements

### Local Comparison

```bash
# Get coverage before changes
cargo tarpaulin --out Xml --all

# Make changes

# Get coverage after changes
cargo tarpaulin --out Xml --all
```

## Files Involved

### Coverage Configuration
- `codecov.yml` - Codecov configuration
- `.github/workflows/ci.yml` - CI coverage job

### Coverage Artifacts
- `coverage/xml/<crate>.xml` - Generated Cobertura reports uploaded to Codecov with matching flags
- `coverage/html/<crate>.html` - Generated HTML viewers published as GitHub Actions artifacts

### Documentation
- This file: `CODE_COVERAGE.md`
- `README.md` - Includes coverage badge
- `DEVELOPMENT.md` - References coverage commands

## Next Steps

1. ✅ Review `codecov.yml` for target settings
2. ✅ Commit and push to trigger coverage
3. ✅ Visit Codecov dashboard to verify
4. ✅ Add tests for uncovered code
5. ✅ Monitor coverage trends

## Resources

- **Codecov Docs:** https://docs.codecov.io/
- **Tarpaulin Docs:** https://github.com/xd009642/tarpaulin
- **Codecov.io:** https://codecov.io/
- **Coverage Best Practices:** https://docs.codecov.io/docs/goals

---

**Last Updated:** March 7, 2026
**Status:** ✅ Coverage tracking enabled

