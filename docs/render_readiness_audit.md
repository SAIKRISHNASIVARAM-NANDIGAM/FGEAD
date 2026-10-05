# FGEAD Project — Render Deployment Readiness Audit Report

**Audit Date:** October 5, 2026  
**Audit Status:** **NOT READY (Pending Pre-Deployment Git Configuration & URL Discovery)**  
**Target Environment:** Render Cloud Platform (Ubuntu 22.04 LTS Linux Container)  
**Target Active Model:** `windows_sivachowdary_v2` ($\tau = 1.411807$, 22 Channels)  

---

## 1. Executive Summary & Verdict

| Assessment Area | Status | Critical Impact |
| :--- | :---: | :--- |
| **Overall Deployment Status** | **NOT READY** | Requires un-ignoring model checkpoints in `.gitignore` and dynamic `FASTAPI_URL` discovery before push. |
| **Core Application Code** | **READY** | FastAPI, Streamlit, Host Registry, XAI, and multi-host inference are fully operational. |
| **Model Checkpoint Integrity** | **READY (Local)** | All `.pt`, `.joblib`, and `.json` files exist and pass validation locally. |
| **Linux Path Compatibility** | **READY** | Zero hardcoded Windows drive letters; 100% `pathlib` and POSIX compliance. |
| **Regression Suite** | **READY** | **74 / 74 PASS (100.00%)** on multi-host test suite. |
| **Health Probes** | **READY** | `/health/live` (Liveness) and `/health/ready` (Readiness) return HTTP 200. |

---

## 2. Detailed 15-Point Verification Matrix

| # | Audit Item | Verification Requirement | Current Project State | Evaluation |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Git Repository Status** | Clean working tree on deployment branch | Branch: `main` (Ahead by 1 commit, 57 uncommitted/untracked modified files). | **ACTION REQD** |
| **2** | **`.gitignore` Configuration** | Essential production assets must not be ignored | `.gitignore` contains `checkpoints/*.pt`! Blocks model checkpoints from being pushed. | **BLOCKER** |
| **3** | **`requirements.txt`** | All production dependencies listed | Contains PyTorch, FastAPI, Uvicorn, Streamlit, Scikit-Learn, Plotly, Psutil, Pydantic. | **READY** |
| **4** | **Python Version Compatibility** | Compatible with Python 3.10, 3.11, 3.12 | Verified on Python 3.11.9. Fully compatible with Render default Python 3.11/3.12. | **READY** |
| **5** | **FastAPI Start Command ($PORT)** | Must bind to dynamic Render `$PORT` | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` directly handles Render `$PORT`. | **READY** |
| **6** | **Streamlit Start Command & URL** | Must bind to `$PORT` and discover API URL | `API_URL` in `app/streamlit_app.py` is hardcoded to `http://127.0.0.1:8000`. | **BLOCKER** |
| **7** | **Required v2 Model Checkpoints** | `fgead_live_windows_22ch_v2_current_machine.*` | Checkpoint (3.64 MB), Scaler (1.1 KB), Threshold (0.3 KB), Config (1.0 KB) all present. | **READY** |
| **8** | **Required Configuration Files** | `settings.py`, `.env.example`, agent configs | `config/settings.py`, `.env.example`, and `data/*_config.json` present. | **READY** |
| **9** | **Required Database Files** | Production SQLite DB `fgead_multihost.db` | Exists (44.0 KB, 2 hosts). `host_registry.py` auto-initializes if missing. | **READY** |
| **10** | **Environment Variables** | Zero hardcoded secrets, clean env defaults | All keys documented in `.env.example` with secure fallback defaults. | **READY** |
| **11** | **Linux Path Compatibility** | No Windows-only paths (`C:\`) or backslashes | 0 hardcoded drive letters; all path manipulations use `pathlib.Path`. | **READY** |
| **12** | **Files NOT to Upload to Render** | Omit heavy non-essential research files | `data/SMD/` (444 MB non-1-1 files), `research_paper/`, `scratch/`, `docs/`. | **OPTIMIZATION** |
| **13** | **SMD Dataset Requirement** | Is 463 MB SMD needed by Live FastAPI? | Only `machine-1-1` (18.63 MB) is used during startup; 444.84 MB can be omitted. | **OPTIMIZATION** |
| **14** | **PyTorch & RAM Constraints** | Fit within Render 512 MB (Free) / 2 GB (Starter) | Active runtime memory is ~280 MB RAM. CPU-only PyTorch recommended for build. | **READY** |
| **15** | **Health Endpoint Probes** | Liveness & Readiness probe availability | `/health/live` (200 OK) and `/health/ready` (200 OK). | **READY** |

---

## 3. Critical Blockers (Must Fix Before Render Deployment)

### Blocker 1: `.gitignore` Excludes Model Checkpoints (`checkpoints/*.pt`)
* **Finding:** Line 14 of `.gitignore` specifies `checkpoints/*.pt`.
* **Impact:** Git will ignore `checkpoints/fgead_live_windows_22ch_v2_current_machine.pt`, `checkpoints/fgead_live_windows_22ch.pt`, and `checkpoints/fgead_smd_machine_1_1.pt`. When Render clones the repository, the model files will be missing, causing FastAPI to crash on startup with `FileNotFoundError: Model checkpoint not found`.
* **Required Resolution:** Update `.gitignore` to allow required production model checkpoints (e.g., using `!checkpoints/fgead_live_windows_22ch_v2_current_machine.pt`, `!checkpoints/fgead_live_windows_22ch.pt`, `!checkpoints/fgead_smd_machine_1_1.pt`, `!checkpoints/best_model.pt`) or force-add them with `git add -f`.

### Blocker 2: Streamlit Dashboard Hardcoded Backend URL
* **Finding:** In `app/streamlit_app.py` line 84, `API_URL = "http://127.0.0.1:8000"`.
* **Impact:** When Streamlit is deployed to Render as an independent Web Service (e.g., `https://fgead-dashboard.onrender.com`), it will attempt to connect to localhost inside its own container and fail to communicate with the FastAPI service (`https://fgead-api.onrender.com`).
* **Required Resolution:** Update `API_URL` to read `os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")`.

### Blocker 3: Uncommitted Local Git State
* **Finding:** 57 files are modified or untracked locally.
* **Impact:** Deploying to Render via Git will build only the old commit without the latest Phase 10 v2 model, multi-host database isolation, and test suites.
* **Required Resolution:** Stage and commit the validated codebase after addressing Blockers 1 and 2.

---

## 4. Recommended Deployment Configurations & Optimizations

### 1. Render Service Architectures

#### Option A: Unified Single Web Service (FastAPI + Streamlit in One Container)
* **Start Command:**
  ```bash
  bash run_production.sh
  ```
* **Render Settings:**
  * **Environment:** Python 3
  * **Build Command:** `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt`
  * **Health Check Path:** `/health/live`

#### Option B: Decoupled Multi-Service (Recommended for Production Scale)
1. **Backend Web Service (`fgead-api`):**
   * **Build Command:** `pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt`
   * **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   * **Health Check Path:** `/health/live`
   * **Environment Variables:**
     * `FGEAD_ENV=PRODUCTION`
     * `FGEAD_API_SECRET_KEY=<secure-random-token>`
     * `FGEAD_CORS_ORIGINS=*`

2. **Frontend Web Service (`fgead-dashboard`):**
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false`
   * **Environment Variables:**
     * `FASTAPI_URL=https://fgead-api.onrender.com`

---

### 2. SMD Dataset Footprint Optimization (Save 444 MB)
* The entire `data/SMD/` folder contains 28 machines (`463.47 MB`).
* The live multi-host inference pipeline does **NOT** use SMD data during live telemetry scoring.
* Only `machine-1-1` (4 files totaling **`18.63 MB`**) is referenced during FastAPI startup (`SMDLoader.load_machine('1-1')`).
* Excluding machines 1-2 through 3-11 via `.dockerignore` or git reduces repository/slug upload size by **`444.84 MB` (96% reduction)** with zero functional impact on the API.

---

### 3. CPU-Only PyTorch Build Optimization
* Standard `torch>=2.0.0` wheel includes ~2 GB of CUDA/cuDNN binaries which are unnecessary on Render CPU instances.
* Installing CPU-only PyTorch:
  ```bash
  pip install --extra-index-url https://download.pytorch.org/whl/cpu torch>=2.0.0
  ```
  Reduces total disk usage from 2.5 GB to ~350 MB and cuts Render build time from 8 minutes to under 90 seconds.

---

## 5. Summary Action Checklist

- [ ] **1. Modify `.gitignore`** to un-ignore the 4 required `.pt` checkpoints.
- [ ] **2. Update `app/streamlit_app.py`** to support `os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")`.
- [ ] **3. Commit all validated files** to `git`.
- [ ] **4. Configure Render Environment Variables** (`FGEAD_ENV=PRODUCTION`, `FGEAD_API_SECRET_KEY`, `FASTAPI_URL`).
- [ ] **5. Set Render Health Check Path** to `/health/live`.
