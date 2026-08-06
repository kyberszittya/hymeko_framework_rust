# Disable AMD Vari-Bright / hand brightness control back to Windows.
# Requires elevation (admin). Backs up current values to a .reg first, then sets 3 DWORDs to 0.
# Reboot afterwards for the driver to pick up the change.
# Revert: double-click the generated backup .reg, or set the values back.

$ErrorActionPreference = 'Stop'
$log = Join-Path $PSScriptRoot 'varibright-run.log'
Start-Transcript -Path $log -Force | Out-Null
trap { "ERROR: $_" | Out-File $log -Append; Stop-Transcript | Out-Null; exit 1 }

$key  = 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000'
$reg  = 'HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000'
$names = 'KMD_EnableBrightnessInterface2','PP_UserVariBrightLevel','Dal_UserVariBrightLevel'

# Sanity: confirm this index is the AMD adapter (not the NVIDIA one).
$desc = (Get-ItemProperty $key -Name DriverDesc).DriverDesc
if ($desc -notmatch 'AMD|Radeon') {
    throw "Index 0000 is '$desc', not the AMD adapter. Aborting -- re-check the GPU index."
}
Write-Host "Target adapter: $desc" -ForegroundColor Cyan

# 1) Backup current values.
$stamp  = (Get-Date -Format 'yyyyMMdd-HHmmss')
$backup = Join-Path $PSScriptRoot "varibright-backup-$stamp.reg"
reg export $reg $backup /y | Out-Null
Write-Host "Backup written: $backup" -ForegroundColor Green

# 2) Show current, then set to 0.
foreach ($n in $names) {
    $cur = (Get-ItemProperty $key -Name $n -ErrorAction SilentlyContinue).$n
    Write-Host ("{0,-32} {1} -> 0" -f $n, $cur)
    New-ItemProperty -Path $key -Name $n -Value 0 -PropertyType DWord -Force | Out-Null
}

Write-Host "`nDone. Reboot for the driver to apply, then test the brightness slider / Fn keys." -ForegroundColor Yellow
Stop-Transcript | Out-Null
