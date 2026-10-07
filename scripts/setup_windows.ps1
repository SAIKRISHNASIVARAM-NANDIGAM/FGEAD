# scripts/setup_windows.ps1
# Automated Windows Environment Setup Script for FGEAD

$ErrorActionPreference = "Stop"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " FGEAD Windows Environment Setup" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Verify Python availability
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python command not found in PATH. Please install Python 3.10-3.12." -ForegroundColor Red
    exit 1
}

# 2. Check Python version
$verString = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$major,$minor = $verString.Split('.') | ForEach-Object { [int]$_ }

if ($major -ne 3 -or $minor -lt 10 -or $minor -gt 12) {
    Write-Host "[WARNING] Python $verString detected. FGEAD is validated on Python 3.10-3.12." -ForegroundColor Yellow
} else {
    Write-Host "[INFO] Python $verString detected and verified." -ForegroundColor Green
}

# 3. Create Virtual Environment if missing
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $projectRoot "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[INFO] Creating virtual environment at '$venvPath'..." -ForegroundColor Cyan
    & python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "[INFO] Existing virtual environment found at '$venvPath'." -ForegroundColor Green
}

# 4. Upgrade pip in venv
Write-Host "[INFO] Upgrading pip inside virtual environment..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Pip upgrade encountered an issue, proceeding with requirement installation." -ForegroundColor Yellow
}

# 5. Install requirements
$reqFile = Join-Path $projectRoot "requirements.txt"
if (Test-Path $reqFile) {
    Write-Host "[INFO] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    & $venvPython -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies from requirements.txt." -ForegroundColor Red
        exit 1
    }
    Write-Host "[SUCCESS] All dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "[ERROR] requirements.txt not found at '$reqFile'." -ForegroundColor Red
    exit 1
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Setup Complete! You can now run pre-flight check or start services." -ForegroundColor Green
Write-Host " Run Preflight Check : .\venv\Scripts\python scripts/preflight_check.py" -ForegroundColor Cyan
Write-Host " Run Start Launcher  : .\scripts\start_local.ps1" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
