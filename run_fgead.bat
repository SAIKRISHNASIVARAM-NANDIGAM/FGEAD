@echo off
title FGEAD - Full Pipeline Runner
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM ================================================================
REM  FGEAD - Feature Graph Explainable Anomaly Detection
REM  Automated Pipeline: Install - Generate - Train - Evaluate - Demo
REM ================================================================

echo.
echo ================================================================
echo          FGEAD - Automated Pipeline Runner
echo    Feature Graph Explainable Anomaly Detection
echo    Graph Autoencoder + LSTM + Correlation Intelligence
echo ================================================================
echo.

REM ================================================================
REM  STEP 0 - Check Python
REM ================================================================
echo [STEP 0/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Please install Python 3.10+ and add it to your PATH.
    pause
    exit /b 1
)
python --version
echo          OK - Python found.
echo.

REM ================================================================
REM  STEP 1 - Install Dependencies
REM ================================================================
echo [STEP 1/6] Installing dependencies from requirements.txt...
echo ----------------------------------------------------------------

if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        echo         Try running: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo          OK - All dependencies installed.
) else (
    echo          WARNING - requirements.txt not found, skipping.
)
echo.

REM ================================================================
REM  STEP 2 - Generate Synthetic Data
REM ================================================================
echo [STEP 2/6] Generating synthetic server-monitoring data...
echo ----------------------------------------------------------------

python data/synthetic_generator.py
if errorlevel 1 (
    echo [ERROR] Data generation failed!
    pause
    exit /b 1
)

if exist "data\synthetic_data.csv" (
    echo          OK - data/synthetic_data.csv created successfully.
) else (
    echo [ERROR] synthetic_data.csv was not created.
    pause
    exit /b 1
)
echo.

REM ================================================================
REM  STEP 3 - Train FGEAD Model
REM ================================================================
echo [STEP 3/6] Training FGEAD model (50 epochs, early stopping)...
echo ----------------------------------------------------------------
echo          This may take a few minutes. Please wait...
echo.

python train.py
if errorlevel 1 (
    echo [ERROR] Training failed!
    pause
    exit /b 1
)

if exist "checkpoints\best_model.pt" (
    echo.
    echo          OK - Model saved to checkpoints/best_model.pt
) else (
    echo [ERROR] best_model.pt was not created.
    pause
    exit /b 1
)
echo.

REM ================================================================
REM  STEP 4 - Evaluate Model + Baseline Comparison
REM ================================================================
echo [STEP 4/6] Evaluating FGEAD vs baselines (IF, LSTM-AE)...
echo ----------------------------------------------------------------

python evaluate.py
if errorlevel 1 (
    echo [ERROR] Evaluation failed!
    pause
    exit /b 1
)
echo.
echo          OK - Evaluation complete.
echo.

REM ================================================================
REM  STEP 5 - CLI Demo (Anomaly Explanation)
REM ================================================================
echo [STEP 5/6] Running CLI demo - listing detected anomalies...
echo ----------------------------------------------------------------

python demo_cli.py --list-anomalies
if errorlevel 1 (
    echo          WARNING - CLI demo encountered an issue (non-critical).
)
echo.

REM ================================================================
REM  STEP 6 - Launch Streamlit Dashboard
REM ================================================================
echo ================================================================
echo   PIPELINE COMPLETE - All stages passed!
echo ================================================================
echo.
echo   Summary:
echo     [OK] Dependencies installed
echo     [OK] Synthetic data generated  (data/synthetic_data.csv)
echo     [OK] FGEAD model trained       (checkpoints/best_model.pt)
echo     [OK] Evaluation complete        (FGEAD vs IF vs LSTM-AE)
echo     [OK] CLI demo executed
echo.
echo [STEP 6/6] Launching Streamlit dashboard...
echo ----------------------------------------------------------------
echo          Dashboard will open at: http://localhost:8501
echo          Press Ctrl+C to stop the server.
echo.

streamlit run app/streamlit_app.py --server.headless true
if errorlevel 1 (
    echo.
    echo [INFO] Streamlit server stopped.
)

echo.
echo ================================================================
echo          FGEAD Pipeline Finished. Goodbye!
echo ================================================================
echo.
pause
