# scripts/start_demo.ps1
# One-Command Local Demo Launcher for FGEAD (FastAPI + Streamlit)

$ErrorActionPreference = "Stop"

# 1. Verify repository root location
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$apiMain = Join-Path $projectRoot "api\main.py"
$appStreamlit = Join-Path $projectRoot "app\streamlit_app.py"

if (-not (Test-Path $apiMain) -or -not (Test-Path $appStreamlit)) {
    Write-Host "[ERROR] Must be executed from within the FGEAD repository." -ForegroundColor Red
    exit 1
}

# 2. Verify Virtual Environment
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment python not found at '$venvPython'." -ForegroundColor Red
    Write-Host "Please run the automated setup script first:" -ForegroundColor Yellow
    Write-Host "  .\scripts\setup_windows.ps1" -ForegroundColor Cyan
    exit 1
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " FGEAD One-Command Local Demo Launcher" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# 3. Run Pre-Flight System Integrity Check
Write-Host "[INFO] Running system pre-flight verification..." -ForegroundColor Cyan
& $venvPython (Join-Path $projectRoot "scripts\preflight_check.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Pre-flight verification failed. Services will NOT be started." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[SUCCESS] Pre-flight verification passed." -ForegroundColor Green

# 4. Check if FastAPI is already online
$healthUrl = "http://127.0.0.1:8000/health/live"
$isFastApiRunning = $false

try {
    $checkResp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($checkResp -and ($checkResp.status -eq "alive" -or $checkResp.status -eq "healthy" -or $checkResp.status -eq "ok")) {
        $isFastApiRunning = $true
    }
} catch {
    $isFastApiRunning = $false
}

if (-not $isFastApiRunning) {
    Write-Host "[INFO] Launching FastAPI backend service in a dedicated process window..." -ForegroundColor Cyan
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$venvPython' -m uvicorn api.main:app --host 127.0.0.1 --port 8000"
} else {
    Write-Host "[INFO] FastAPI backend service is already running on http://127.0.0.1:8000" -ForegroundColor Green
}

# 5. Wait for FastAPI to become healthy
Write-Host "[INFO] Verifying FastAPI backend health at $healthUrl..." -ForegroundColor Cyan
$healthy = $false
$retryCount = 0
$maxRetries = 15

while (-not $healthy -and $retryCount -lt $maxRetries) {
    Start-Sleep -Seconds 1
    $retryCount++
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp -and ($resp.status -eq "alive" -or $resp.status -eq "healthy" -or $resp.status -eq "ok")) {
            $healthy = $true
        }
    } catch {
        # Retry silently until timeout
    }
}

if (-not $healthy) {
    Write-Host ""
    Write-Host "[ERROR] FastAPI backend failed health check at $healthUrl." -ForegroundColor Red
    Write-Host "Streamlit dashboard will NOT be started." -ForegroundColor Yellow
    exit 1
}

Write-Host "[SUCCESS] FastAPI backend service is online and healthy." -ForegroundColor Green

# 6. Launch Streamlit Dashboard
Write-Host "[INFO] Launching Streamlit Version 3 frontend dashboard..." -ForegroundColor Cyan
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$venvPython' -m streamlit run app\streamlit_app.py"

# 7. Print Final Readiness Information
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " FGEAD Demo Started" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " FastAPI:   http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host " Health:    http://127.0.0.1:8000/health/live" -ForegroundColor Cyan
Write-Host " Dashboard: http://localhost:8501" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "[NOTE] Windows telemetry live agent is an optional, machine-specific step." -ForegroundColor Yellow
Write-Host "To start live telemetry collection on this host, run:" -ForegroundColor White
Write-Host "  .\scripts\start_live_agent.ps1" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Green
