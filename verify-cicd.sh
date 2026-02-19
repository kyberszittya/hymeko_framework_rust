#!/usr/bin/env bash
# Quick verification script for CI/CD setup
# Run this to verify all CI/CD files are in place

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  CI/CD Setup Verification Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

MISSING=0
FOUND=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((FOUND++))
    else
        echo -e "${RED}✗${NC} $1 (MISSING)"
        ((MISSING++))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        ((FOUND++))
    else
        echo -e "${RED}✗${NC} $1/ (MISSING)"
        ((MISSING++))
    fi
}

echo -e "${YELLOW}Workflow Files:${NC}"
check_file ".github/workflows/ci.yml"
check_file ".github/workflows/release.yml"
check_file ".github/workflows/security-audit.yml"
check_file ".github/workflows/update-dependencies.yml"

echo ""
echo -e "${YELLOW}Templates:${NC}"
check_file ".github/pull_request_template.md"
check_file ".github/ISSUE_TEMPLATE/bug_report.md"
check_file ".github/ISSUE_TEMPLATE/feature_request.md"

echo ""
echo -e "${YELLOW}Development Scripts:${NC}"
check_file "dev.sh"
check_file "dev.ps1"
check_file "Makefile"

echo ""
echo -e "${YELLOW}Documentation:${NC}"
check_file "DEVELOPMENT.md"
check_file "CI_CD_DOCUMENTATION.md"
check_file "CICD_STATUS.md"
check_file "CICD_SETUP_COMPLETE.md"
check_file "SETUP_CHECKLIST.md"

echo ""
echo -e "${YELLOW}Configuration:${NC}"
check_file ".gitignore"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ $MISSING -eq 0 ]; then
    echo -e "${GREEN}✓ All CI/CD files are in place! (${FOUND}/${FOUND})${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review the workflows: cat .github/workflows/ci.yml"
    echo "  2. Test locally: cargo test --all"
    echo "  3. Commit: git add .github/ *.md *.sh *.ps1 Makefile .gitignore"
    echo "  4. Commit: git commit -m 'ci: setup GitHub Actions CI/CD pipeline'"
    echo "  5. Push: git push origin <branch>"
    echo "  6. Monitor: https://github.com/hakiko/hymeko_framework/actions"
    exit 0
else
    echo -e "${RED}✗ Missing ${MISSING} file(s)!${NC}"
    exit 1
fi

