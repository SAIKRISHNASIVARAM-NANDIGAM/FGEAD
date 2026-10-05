# ==============================================================================
# FGEAD - Production Startup Script (Windows PowerShell)
# Launches both FastAPI Backend and Streamlit Dashboard in Production Mode
# ==============================================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "          FGEAD - Production Stack Launcher (Windows)" -ForegroundColor Cyan
Write-Host "  Feature Graph Explainable Anomaly Detection Platform" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Load Environment Variables from .env if present
if (Test-Path ".env") {
    Write-Host "[INFO] Loading configuration from .env file..." -ForegroundColor Green
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Length -eq 2) {
                [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), [System.EnvironmentVariableTarget]::Process)
            }
        }
    }
}

$env:FGEAD_ENV = "PRODUCTION"
$HostAddr = if ($env:FGEAD_SERVER_HOST) { $env:FGEAD_SERVER_HOST } else { "127.0.0.1" }
$ApiPort = if ($env:FGEAD_SERVER_PORT) { $env:FGEAD_SERVER_PORT } else { "8000" }
$DashPort = if ($env:STREAMLIT_SERVER_PORT) { $env:STREAMLIT_SERVER_PORT } else { "8501" }

# 2. Virtual Environment Check
$PythonExe = "python"
if (Test-Path "venv\Scripts\python.exe") {
    $PythonExe = "$ScriptDir\venv\Scripts\python.exe"
}

# 3. Create required directories
New-Item -ItemType Directory -Force -Path "data", "checkpoints", "logs" | Out-Null

Write-Host "[INFO] Starting FastAPI Backend on ${HostAddr}:${ApiPort}..." -ForegroundColor Green
$ApiProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m", "uvicorn", "api.main:app", "--host", $HostAddr, "--port", $ApiPort -PassThru -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "[INFO] Starting Streamlit Dashboard on port ${DashPort}..." -ForegroundColor Green
$DashProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", $DashPort, "--server.address", $HostAddr, "--server.headless", "true" -PassThru -NoNewWindow

Write-Host ""
Write-Host "[SUCCESS] FGEAD Stack is running." -ForegroundColor Green
Write-Host "          - API Gateway: http://${HostAddr}:${ApiPort}" -ForegroundColor Yellow
Write-Host "          - Dashboard:   http://${HostAddr}:${DashPort}" -ForegroundColor Yellow
Write-Host "          - Health:      http://${HostAddr}:${ApiPort}/health" -ForegroundColor Yellow
Write-Host "          - API Docs:    http://${HostAddr}:${ApiPort}/docs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop both services." -ForegroundColor White

try {
    while ($true) {
        Start-Sleep -Seconds 1
        if ($ApiProcess.HasExited -or $DashProcess.HasExited) {
            break
        }
    }
}
finally {
    Write-Host "`n[INFO] Stopping FGEAD services..." -ForegroundColor Yellow
    if (-not $ApiProcess.HasExited) { Stop-Process -Id $ApiProcess.Id -Force }
    if (-not $DashProcess.HasExited) { Stop-Process -Id $DashProcess.Id -Force }
    Write-Host "[INFO] Services stopped." -ForegroundColor Green
}
