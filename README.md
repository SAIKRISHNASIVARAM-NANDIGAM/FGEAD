# FGEAD — Feature Graph-based Explainable Anomaly Detection

> **Multivariate time-series anomaly detection framework featuring dynamic feature graph construction, forecasting z-score anomaly scoring, and multi-query Explainable AI (XAI) root-cause analysis.**

---

## 1. Project Title
**FGEAD — Feature Graph-based Explainable Anomaly Detection**

---

## 2. Short Description
FGEAD (Feature Graph-based Explainable Anomaly Detection) combines graph neural networks (GCN), self-attention graph representation learning, and recurrent temporal forecasting (LSTM) to detect and explain anomalies in high-dimensional multivariate system telemetry. It supports two primary deployment pathways:
1. **Windows Physical Host Live Monitoring**: Real-time telemetry collection and anomaly detection on Windows systems across a validated 22-feature schema.
2. **SMD (Server Machine Dataset) Benchmark Pipeline**: Offline benchmark evaluation across 38 telemetry channels across multi-machine datasets.

---

## 3. Architecture Overview

### Live Host Monitoring Architecture
```
Windows Live Agent
  └─► FastAPI Gateway (REST Endpoint)
        └─► Live Buffer (60-Sample Rolling Window)
              └─► Feature Preprocessing & Schema Validation (22 Features)
                    └─► FGEAD Neural Inference (GCN + LSTM + Graph Attention)
                          └─► Episode Tracking & State Classifier (NORMAL / SUSPICIOUS / ANOMALY)
                                └─► Explainability Engine (XAI Q1–Q5 Analysis)
                                      └─► Streamlit Version 3 Interactive Dashboard
```

### SMD Benchmark Pipeline
```
SMD Dataset (38 Channels)
  └─► SMD DataLoader
        └─► Feature Normalization & 60-Timestep Windowing
              └─► FGEAD Benchmark Neural Model (`fgead_smd_machine_1_1.pt`)
                    └─► Anomaly Scoring & z-Score Thresholding (τ = 2.073376)
                          └─► F1 / Precision / Recall Metric Evaluation
```

---

## 4. Repository Structure

```
FGEAD-main/
├── api/                        # FastAPI REST service, host registry, endpoints (/health/live, /hosts, /predict)
├── app/                        # Streamlit Version 3 interactive dashboard (streamlit_app.py)
├── checkpoints/                # Model weights (.pt), feature scalers (.joblib), thresholds (.json), and configs (.json)
├── config/                     # Central environment configuration management (settings.py)
├── core/                       # Core system utilities and structured logging (logger.py)
├── data/                       # Telemetry buffers, live feature schema (live_feature_schema.py), SMD data loaders
│   └── multihost_validation/   # Multi-host validation reports and regression test suites
├── agents/                     # Telemetry collection agents for Windows (windows_agent.py) and Linux (linux_agent.py)
├── research_paper/             # Academic paper documentation, LaTeX sources, literature reviews, and figures
├── scripts/                    # Automation scripts (preflight_check.py, setup_windows.ps1, start_local.ps1)
├── scratch/                    # Audit and test execution utilities
├── requirements.txt            # Python package dependencies
├── .env.example                # Environment configuration template
└── README.md                   # Project documentation
```

---

## 5. Prerequisites
- **Operating System**: Windows 10 / 11 (64-bit)
- **Python Version**: Python 3.10 – 3.12
- **Tools**: Git, PowerShell (5.1+)

---

## 6. Installation

### Option A: Automated PowerShell Setup (Recommended)
Open PowerShell as Administrator/User in the repository directory and run:
```powershell
.\scripts\setup_windows.ps1
```

### Option B: Manual Installation
```powershell
# 1. Clone the repository
git clone https://github.com/SAIKRISHNASIVARAM-NANDIGAM/FGEAD.git
cd FGEAD-main

# 2. Create Python virtual environment
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

---

## 7. Local Configuration
Copy `.env.example` to `.env` to configure your environment settings:
```powershell
Copy-Item .env.example .env
```
Key configuration properties in `.env`:
- `ENVIRONMENT`: Set to `DEVELOPMENT` for local testing or `PRODUCTION` for deployment.
- `FASTAPI_URL`: Backend service URL (default: `http://127.0.0.1:8000`).
- `FGEAD_API_SECRET_KEY`: Security token secret key (use safe local placeholder value).

> **SECURITY NOTE**: Never include real API keys, passwords, or machine secrets in version control or `.env.example`.

---

## 8. Running FastAPI REST Gateway
Start the FastAPI backend service using Python in the virtual environment:
```powershell
.\venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Interactive API documentation (Swagger UI) is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 9. Health Check Endpoint
Verify that the FastAPI service is healthy and online:
- URL: [http://127.0.0.1:8000/health/live](http://127.0.0.1:8000/health/live)
- Expected JSON response: `{"status": "healthy", ...}`

---

## 10. Running Streamlit Dashboard
Launch the Streamlit Version 3 frontend dashboard:
```powershell
.\venv\Scripts\python -m streamlit run app\streamlit_app.py
```
Access the dashboard UI at [http://127.0.0.1:8501](http://127.0.0.1:8501).

---

## 11. Running the Windows Live Agent
To start collecting live Windows system telemetry:
```powershell
.\venv\Scripts\python agents\windows_agent.py
```

> **IMPORTANT BASELINE NOTICE**:
> Machine-specific live models (such as `windows_sivachowdary_v2`) are trained and calibrated against specific hardware baselines. They are **not universal models**.
> If deploying on a new or different Windows machine, you must collect an appropriate baseline telemetry sample and calibrate feature normalization before using live inference for scientific evaluation.

---

## 12. SMD Benchmark Pipeline Usage
To evaluate FGEAD on the Server Machine Dataset (SMD 38-channel benchmark):
```powershell
.\venv\Scripts\python evaluate_smd_final.py
```
This script computes reconstruction z-scores, checks performance against the calibrated threshold ($\tau = 2.073376$), and prints F1-score, Precision, and Recall metrics.

---

## 13. The 60-Sample Rolling Window
FGEAD requires a continuous rolling temporal window of **60 consecutive timesteps** (collected at 1 Hz, representing a 60-second observation window).
- **Buffer Initialization**: During the first 59 seconds after host registration, the system accumulates initial telemetry.
- **Inference Trigger**: Once 60 timesteps are buffered, FGEAD constructs a 60×22 feature matrix, extracts dynamic feature graph attention weights, and computes temporal forecast z-scores.

---

## 14. Classification States
The system categorizes host condition into three distinct state levels:
- **`NORMAL`**: Forecast error z-score remains below the warning threshold ($\tau_{warn} < 1.4$). Telemetry exhibits expected baseline behavior.
- **`SUSPICIOUS`**: Forecast error z-score crosses moderate thresholds ($1.4 \le \tau < 1.859$) or transient spikes occur. Indicates potential early degradation.
- **`ANOMALY`**: Forecast error z-score exceeds the calibrated anomaly threshold ($\tau \ge 1.859450$) over sustained windows. Triggering episode tracking and XAI root-cause investigation.

---

## 15. Explainability & XAI Terminology
When an anomaly occurs, FGEAD's XAI explainer ranks individual feature deviations.
- **Terminology Standard**: Refer to identified feature deviations as **"top contributing features"** or **"root-cause candidates"**.
- **Causality Disclaimer**: Feature attribution identifies mathematical forecast error contributions and feature graph edge variations; it **does NOT prove causal relationships** in physical hardware systems.

---

## 16. Regression Testing & Validation
To execute the complete regression validation suite (75 tests across API, multi-host, model integrity, and database isolation invariants):
```powershell
.\venv\Scripts\python scratch/run_full_regression_audit.py
```

### Validated Test Result
```
================================================================================
REGRESSION AUDIT SUMMARY REPORT
================================================================================
  [PASS] Phase 7 Smoke Tests (7)                           : 7/7
  [PASS] Phase 8 Cloud Deployment Tests (4)                : 4/4
  [PASS] Host Registration Lifecycle Tests (8)             : 8/8
  [PASS] Operational DB Isolation Tests (4)                : 4/4
  [PASS] Phase 9 Accuracy Suite (18)                       : 18/18
  [PASS] Phase 10 Current-Machine Dedicated Model Suite (6): 7/7
  [PASS] MultiHost Platform Verification Suite (17)        : 17/17
  [PASS] Phase 6.1 Compatibility Suite (10)                : 10/10
--------------------------------------------------------------------------------
Original Regression Suite : 68 / 68 PASS (100.00%)
Phase 10 Test Suite       : 7 / 7 PASS (100.00%)
Combined Total Suite      : 75 / 75 PASS (100.00%)
================================================================================
```
**Status: 75 / 75 tests passed.**

---

## 17. Demo Scenarios

The framework includes four standardized demonstration test cases:

| Test Case | Scenario | Expected State | Description |
|---|---|---|---|
| **TC-01** | Normal Operation | `NORMAL` | Standard system workload, low z-scores across all 22 telemetry features. |
| **TC-02** | CPU Anomaly | `ANOMALY` | Synthetic or physical CPU stress workload; CPU features flagged as top contributing root-cause candidates. |
| **TC-03** | Disk-Read Anomaly | `ANOMALY` | High I/O read throughput spike (`disk_read_bytes_per_sec` deviation). |
| **TC-04** | System Recovery | `NORMAL` | Workload subsides, telemetry normalizes, state automatically transitions back to `NORMAL`. |

---

## 18. Cloudflare Quick Tunnel (Demo Only)
For temporary demonstration or remote UI evaluation, Cloudflare Quick Tunnel can expose local ports (`http://127.0.0.1:8000` or `http://127.0.0.1:8501`).
> **NOTICE**: Cloudflare Quick Tunnel is strictly a **temporary, demo-only tool**. It should **never** be used as a permanent production deployment architecture.

---

## 19. Security Guidelines
- **No Secrets in Git**: Never commit `.env`, passwords, API tokens, or cryptographic keys.
- **Agent Config Files**: Ensure `live_agent_config.json` is ignored if it contains machine secrets.
- **Database Safety**: Local runtime SQLite databases (`*.db`) and database backups must remain excluded from Git tracking.

---

## 20. System Limitations
1. **Machine-Dependent Calibration**: Live physical models require baseline calibration for the target host's specific hardware configuration and background workload.
2. **Real-World Scope**: High accuracy in validated test scenarios does not guarantee detection of all unobserved hardware or kernel anomaly patterns.
3. **Correlation vs Causality**: Graph attention weights and feature z-scores reflect statistical deviation patterns, not formal causal lineage.
