@echo off
REM Quick verification script for CI/CD setup (Windows)
REM Run this to verify all CI/CD files are in place

setlocal enabledelayedexpansion

cls
echo.
echo ═══════════════════════════════════════════════════════════
echo   CI/CD Setup Verification Script (Windows)
echo ═══════════════════════════════════════════════════════════
echo.

set MISSING=0
set FOUND=0

REM Workflow Files
echo Workflow Files:
if exist ".github\workflows\ci.yml" (
    echo [OK] .github\workflows\ci.yml
    set /a FOUND+=1
) else (
    echo [MISSING] .github\workflows\ci.yml
    set /a MISSING+=1
)

if exist ".github\workflows\release.yml" (
    echo [OK] .github\workflows\release.yml
    set /a FOUND+=1
) else (
    echo [MISSING] .github\workflows\release.yml
    set /a MISSING+=1
)

if exist ".github\workflows\security-audit.yml" (
    echo [OK] .github\workflows\security-audit.yml
    set /a FOUND+=1
) else (
    echo [MISSING] .github\workflows\security-audit.yml
    set /a MISSING+=1
)

if exist ".github\workflows\update-dependencies.yml" (
    echo [OK] .github\workflows\update-dependencies.yml
    set /a FOUND+=1
) else (
    echo [MISSING] .github\workflows\update-dependencies.yml
    set /a MISSING+=1
)

echo.
echo Templates:
if exist ".github\pull_request_template.md" (
    echo [OK] .github\pull_request_template.md
    set /a FOUND+=1
) else (
    echo [MISSING] .github\pull_request_template.md
    set /a MISSING+=1
)

if exist ".github\ISSUE_TEMPLATE\bug_report.md" (
    echo [OK] .github\ISSUE_TEMPLATE\bug_report.md
    set /a FOUND+=1
) else (
    echo [MISSING] .github\ISSUE_TEMPLATE\bug_report.md
    set /a MISSING+=1
)

if exist ".github\ISSUE_TEMPLATE\feature_request.md" (
    echo [OK] .github\ISSUE_TEMPLATE\feature_request.md
    set /a FOUND+=1
) else (
    echo [MISSING] .github\ISSUE_TEMPLATE\feature_request.md
    set /a MISSING+=1
)

echo.
echo Development Scripts:
if exist "dev.ps1" (
    echo [OK] dev.ps1
    set /a FOUND+=1
) else (
    echo [MISSING] dev.ps1
    set /a MISSING+=1
)

if exist "dev.sh" (
    echo [OK] dev.sh
    set /a FOUND+=1
) else (
    echo [MISSING] dev.sh
    set /a MISSING+=1
)

if exist "Makefile" (
    echo [OK] Makefile
    set /a FOUND+=1
) else (
    echo [MISSING] Makefile
    set /a MISSING+=1
)

echo.
echo Documentation:
if exist "DEVELOPMENT.md" (
    echo [OK] DEVELOPMENT.md
    set /a FOUND+=1
) else (
    echo [MISSING] DEVELOPMENT.md
    set /a MISSING+=1
)

if exist "CI_CD_DOCUMENTATION.md" (
    echo [OK] CI_CD_DOCUMENTATION.md
    set /a FOUND+=1
) else (
    echo [MISSING] CI_CD_DOCUMENTATION.md
    set /a MISSING+=1
)

if exist "CICD_STATUS.md" (
    echo [OK] CICD_STATUS.md
    set /a FOUND+=1
) else (
    echo [MISSING] CICD_STATUS.md
    set /a MISSING+=1
)

if exist "CICD_SETUP_COMPLETE.md" (
    echo [OK] CICD_SETUP_COMPLETE.md
    set /a FOUND+=1
) else (
    echo [MISSING] CICD_SETUP_COMPLETE.md
    set /a MISSING+=1
)

if exist "SETUP_CHECKLIST.md" (
    echo [OK] SETUP_CHECKLIST.md
    set /a FOUND+=1
) else (
    echo [MISSING] SETUP_CHECKLIST.md
    set /a MISSING+=1
)

echo.
echo Configuration:
if exist ".gitignore" (
    echo [OK] .gitignore
    set /a FOUND+=1
) else (
    echo [MISSING] .gitignore
    set /a MISSING+=1
)

echo.
echo ═══════════════════════════════════════════════════════════

if %MISSING% equ 0 (
    echo [SUCCESS] All CI/CD files are in place! (!FOUND!/!FOUND!)
    echo.
    echo Next steps:
    echo   1. Test locally: cargo test --all
    echo   2. Check scripts work: .\dev.ps1 help
    echo   3. Commit: git add .github/ *.md *.sh *.ps1 Makefile .gitignore
    echo   4. Commit: git commit -m "ci: setup GitHub Actions CI/CD pipeline"
    echo   5. Push: git push origin ^<branch^>
    echo   6. Monitor: https://github.com/hakiko/hymeko_framework/actions
) else (
    echo [ERROR] Missing !MISSING! file(s)!
    exit /b 1
)

endlocal

