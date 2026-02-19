# CI/CD Status and Badges

Add these badges to your repository's README.md to display CI/CD status.

## Markdown Badges

```markdown
[![CI](https://github.com/hakiko/hymeko_framework/actions/workflows/ci.yml/badge.svg)](https://github.com/hakiko/hymeko_framework/actions/workflows/ci.yml)
[![Security Audit](https://github.com/hakiko/hymeko_framework/actions/workflows/security-audit.yml/badge.svg)](https://github.com/hakiko/hymeko_framework/actions/workflows/security-audit.yml)
[![codecov](https://codecov.io/gh/hakiko/hymeko_framework/branch/master/graph/badge.svg)](https://codecov.io/gh/hakiko/hymeko_framework)
```

## Workflow Files Summary

### CI Workflow (`ci.yml`)
- **Trigger**: Push to master/main/develop, PRs
- **Jobs**:
  - Test (3 OS × 2 Rust versions = 6 parallel jobs)
  - Rustfmt (code formatting check)
  - Clippy (linting)
  - Coverage (Codecov integration)
  - Build (release binaries)
- **Cache**: Cargo registry, git, and build targets
- **Status**: ![CI](https://img.shields.io/badge/CI-Active-brightgreen)

### Release Workflow (`release.yml`)
- **Trigger**: Git tag push (v*)
- **Jobs**:
  - Create GitHub Release
  - Build for Linux, Windows, macOS
  - Upload binaries as release assets
- **Status**: ![Release](https://img.shields.io/badge/Release-Active-brightgreen)

### Security Audit (`security-audit.yml`)
- **Trigger**: Daily (2 AM UTC) + push events
- **Tool**: RustSec Audit
- **Status**: ![Security](https://img.shields.io/badge/Security-Active-brightgreen)

### Dependency Updates (`update-dependencies.yml`)
- **Trigger**: Weekly (Mondays 9 AM UTC)
- **Action**: Creates PR if updates available
- **Status**: ![Dependencies](https://img.shields.io/badge/Dependencies-Active-brightgreen)

## Setup Instructions for Your Repository

### Step 1: Enable GitHub Actions
1. Go to your repository Settings
2. Navigate to Actions → General
3. Ensure "Actions permissions" is set to "Allow all actions"

### Step 2: Add Codecov Token (Optional)
For code coverage reports:
1. Go to https://codecov.io
2. Sign up with GitHub
3. Add your repository
4. Codecov will automatically pick up reports from GitHub Actions

### Step 3: Add Branch Protection Rules (Recommended)
1. Go to Settings → Branches
2. Click "Add rule" for your main branch
3. Enable:
   - "Require status checks to pass before merging"
   - Select: CI test jobs, rustfmt, clippy
   - "Require code reviews before merging"
   - "Require branches to be up to date before merging"

### Step 4: Create Release Checklist
When ready to release:
```bash
# 1. Update version in Cargo.toml files
# 2. Update CHANGELOG.md
# 3. Commit: git commit -am "chore: release v0.2.0"
# 4. Tag: git tag v0.2.0
# 5. Push: git push origin main && git push origin v0.2.0
# GitHub Actions will automatically create the release
```

## Monitoring and Troubleshooting

### View Workflow Runs
https://github.com/hakiko/hymeko_framework/actions

### Common Issues

**Tests failing on specific OS:**
- Check OS-specific dependencies in `Cargo.toml`
- Verify code doesn't use OS-specific APIs without feature gates

**Clippy warnings:**
- Run locally: `cargo clippy --all --all-targets`
- Fix warnings or suppress with `#[allow(...)]`

**Coverage not uploading:**
- Verify Codecov integration in workflow
- Check GitHub token permissions

**Release build failing:**
- Verify Cargo.toml has correct metadata
- Check dependencies don't have platform-specific issues

## Performance Tips

1. **Cache optimization**: Current setup caches registry, git, and build targets
2. **Parallel jobs**: Test matrix runs 6 jobs in parallel for faster feedback
3. **Skip expensive checks on draft PRs**: You can add conditions if needed
4. **Artifact retention**: GitHub retains artifacts for 90 days by default

## Cost Considerations

- **Public repo**: GitHub Actions is free (3000 minutes/month included)
- **Private repo**: 2000 minutes/month free, then charged per minute
- **Storage**: Artifacts and caches have quotas but use GitHub's storage

## Next Steps

1. Commit all files: `git add .github/ *.md *.sh *.ps1 Makefile`
2. Push: `git push origin <branch>`
3. Create PR and let CI run
4. Monitor runs at https://github.com/hakiko/hymeko_framework/actions
5. Update README with badges once workflows are working

