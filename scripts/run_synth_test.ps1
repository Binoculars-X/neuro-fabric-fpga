# run_synth_test.ps1 -- Run Vivado batch synthesis test and check results.
#
# Usage:
#   cd neuro-fabric-fpga\scripts
#   .\run_synth_test.ps1
#
# Prerequisites:
#   1. Vivado 2023.x installed (script auto-detects common locations)
#   2. RecipSqrtTests (C# XSim tests) run at least once to generate recipsqrt_rom.mem

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Locate Vivado ─────────────────────────────────────────────────────────────
$vivadoExe = $null
$candidates = @(
    "vivado",   # if in PATH
    "C:\Xilinx\Vivado\2023.2\bin\vivado.bat",
    "C:\Xilinx\Vivado\2023.1\bin\vivado.bat",
    "C:\Xilinx\Vivado\2024.1\bin\vivado.bat",
    "C:\AMD\Vivado\2023.2\bin\vivado.bat",
    "C:\AMD\Vivado\2023.1\bin\vivado.bat"
)
foreach ($c in $candidates) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $vivadoExe = $c; break }
    if (Test-Path $c) { $vivadoExe = $c; break }
}
if (-not $vivadoExe) {
    Write-Error "Vivado not found. Add Vivado bin\ to PATH or edit candidates in this script."
    exit 1
}
Write-Host "Using Vivado: $vivadoExe"

# ── Run synthesis ─────────────────────────────────────────────────────────────
$tcl  = Join-Path $ScriptDir "synth_rom_test.tcl"
$log  = Join-Path $ScriptDir "synth_rom_test_out\vivado.log"
$jou  = Join-Path $ScriptDir "synth_rom_test_out\vivado.jou"

Push-Location $ScriptDir
try {
    & $vivadoExe -mode batch -source $tcl -log $log -journal $jou
    if ($LASTEXITCODE -ne 0) { Write-Error "Vivado exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}

# ── Check results ─────────────────────────────────────────────────────────────
$util = Join-Path $ScriptDir "synth_rom_test_out\utilization.rpt"

Write-Host ""
Write-Host "================================================================"
Write-Host " RESULTS CHECK"
Write-Host "================================================================"

# 1. ROM inferred as BRAM?
Write-Host ""
Write-Host "-- BRAM utilization (expect RAMB18E2 or RAMB36E2 >= 1) --"
if (Test-Path $util) {
    $bram = Select-String -Path $util -Pattern "RAMB\d+E2"
    if ($bram) { $bram | ForEach-Object { Write-Host "  $($_.Line.Trim())" } }
    else { Write-Warning "  No RAMB found in utilization.rpt -- ROM may not have been inferred as BRAM!" }
} else { Write-Warning "  utilization.rpt not found" }

# 2. $readmemh path error?
Write-Host ""
Write-Host "-- Synth 8-2898 check (expect: no matches) --"
if (Test-Path $log) {
    $readmemWarn = Select-String -Path $log -Pattern "8-2898|could not open.*readmem"
    if ($readmemWarn) {
        Write-Warning "  FOUND readmem path warning:"
        $readmemWarn | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
    } else { Write-Host "  OK -- no Synth 8-2898" }
} else { Write-Warning "  vivado.log not found" }

# 3. Critical warnings or errors?
Write-Host ""
Write-Host "-- CRITICAL WARNING / ERROR check (expect: none) --"
if (Test-Path $log) {
    $crits = Select-String -Path $log -Pattern "CRITICAL WARNING|^ERROR:"
    if ($crits) {
        Write-Warning "  Found critical issues:"
        $crits | ForEach-Object { Write-Host "  $($_.Line.Trim())" }
    } else { Write-Host "  OK -- no critical warnings or errors" }
}

Write-Host ""
Write-Host "Full reports in: $(Join-Path $ScriptDir 'synth_rom_test_out\')"
