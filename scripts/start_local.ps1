# scripts/start_local.ps1
# FGEAD Local Development Launcher Script for Windows

$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment python not found at '$venvPython'." -ForegroundColor Red
    Write-Host "Please run .\scripts\setup_windows.ps1 first to setup the virtual environment." -ForegroundColor Yellow
    exit 1
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " FGEAD Local Development Launcher" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Launching Local FGEAD Services:" -ForegroundColor White
Write-Host "   1. FastAPI REST Backend     -> http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "      - Swagger Interactive    -> http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "      - Health Check Endpoint  -> http://127.0.0.1:8000/health/live" -ForegroundColor Green
Write-Host "   2. Streamlit Dashboard v3   -> http://127.0.0.1:8501" -ForegroundColor Green
Write-Host ""
Write-Host " Starting FastAPI backend service in a dedicated process window..." -ForegroundColor Cyan
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$venvPython' -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 2

Write-Host " Starting Streamlit Version 3 frontend dashboard..." -ForegroundColor Cyan
Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", "Set-Location '$projectRoot'; & '$venvPython' -m streamlit run app\streamlit_app.py"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Both processes have been launched successfully." -ForegroundColor Green
Write-Host " FastAPI endpoint  : http://127.0.0.1:8000" -ForegroundColor Green
Write-Host " Streamlit UI      : http://127.0.0.1:8501" -ForegroundColor Green
Write-Host " To stop services, close their respective process windows." -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Cyan
