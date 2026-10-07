# scripts/start_live_agent.ps1
# Optional Windows Live Telemetry Agent Launcher Script

[CmdletBinding()]
param(
    [string]$ApiUrl = "http://127.0.0.1:8000",
    [double]$Interval = 1.0,
    [string]$HostId = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
$agentScript = Join-Path $projectRoot "agents\windows_agent.py"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment python not found at '$venvPython'." -ForegroundColor Red
    Write-Host "Please run .\scripts\setup_windows.ps1 first to setup the virtual environment." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $agentScript)) {
    Write-Host "[ERROR] Windows agent script not found at '$agentScript'." -ForegroundColor Red
    exit 1
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " FGEAD Windows Telemetry Live Agent" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "[NOTICE] Baseline & Model Compatibility Information:" -ForegroundColor Yellow
Write-Host " Machine-specific live models are calibrated against target hardware baselines." -ForegroundColor White
Write-Host " If running on a new Windows machine, collect baseline telemetry before" -ForegroundColor White
Write-Host " using machine-specific live models for scientific evaluation." -ForegroundColor White
Write-Host ""
Write-Host " Connecting to API Gateway : $ApiUrl" -ForegroundColor Green
Write-Host " Telemetry Sample Interval : $Interval s" -ForegroundColor Green

$agentArgs = @("$agentScript", "--api-url", "$ApiUrl", "--interval", "$Interval")

if (-not [string]::IsNullOrWhiteSpace($HostId)) {
    Write-Host " Monitored Host ID        : $HostId" -ForegroundColor Green
    $agentArgs += "--host-id"
    $agentArgs += "$HostId"
} else {
    Write-Host " Monitored Host ID        : Auto-detecting local hostname" -ForegroundColor Green
}

Write-Host ""
Write-Host " Starting Windows Live Agent..." -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

& $venvPython $agentArgs
