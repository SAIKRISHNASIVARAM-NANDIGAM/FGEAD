# FGEAD Project — Complete Unwanted File & Cleanup Audit Report
**Date:** October 5, 2026  **Audit Type:** 100% Read-Only Forensic Repository Audit  **Target Project:** `FGEAD-main`  **Active Validated Model:** `windows_sivachowdary_v2` ($\\tau = 1.411807$, 22 Channels, 60s Window)  **Regression Integrity:** **74 / 74 PASS (100.00%)**  **Operational Database:** `data/fgead_multihost.db` (Exactly 2 Hosts Preserved)  
---
## 1. Executive Summary & Inventory Overview
A comprehensive recursive inventory scan was conducted across the entire `FGEAD-main` repository (excluding virtual environment `.venv`/`venv` and version control metadata `.git`).
- **Total Discovered Files:** `385` files
- **Total Discovered Directories:** `38` directories
- **Total Non-Venv Repository Size:** `493.88 MB` (`517,875,441` bytes)
- **SMD Raw Benchmark Dataset:** `463.47 MB` (113 files in `data/SMD/`)
- **Codebase, Models, Documentation & Plots:** `30.41 MB` (`272` files)

### Classification Breakdown
| Category Code | Category Name | File Count | Size (MB) | % of Files | Action Recommendation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **A** | **REQUIRED FOR APPLICATION** | `51` | `7.21 MB` | `13.2%` | **DO NOT TOUCH / PROTECT** |
| **B** | **REQUIRED FOR FINAL SUBMISSION** | `217` | `480.67 MB` | `56.4%` | **DO NOT TOUCH / PROTECT** |
| **C** | **REQUIRED FOR TESTING** | `16` | `3.83 MB` | `4.2%` | **DO NOT TOUCH / PROTECT** |
| **D** | **OPTIONAL / ARCHIVE** | `13` | `0.15 MB` | `3.4%` | **ARCHIVE** |
| **E** | **TEMPORARY / CACHE** | `84` | `1.71 MB` | `21.8%` | **SAFE TO DELETE** |
| **F** | **DUPLICATE / OBSOLETE** | `2` | `0.28 MB` | `0.5%` | **NEEDS APPROVAL** |
| **G** | **UNKNOWN — DO NOT DELETE** | `2` | `0.02 MB` | `0.5%` | **PROTECT** |

---
## 2. Model Artifacts Audit (`checkpoints/`)
| Checkpoint File | Size (KB) | Architecture / Model Name | Production Reference | Regression / Test Use | Submission Reqd | Status / Protection |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| `checkpoints/best_model.pt` | `3469.6` | `best_model.pt` | `NO` | `YES` | **YES** | **Category B (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_scaler.joblib` | `1.1` | `fgead_live_scaler.joblib` | `NO` | `YES` | **YES** | **Category C (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_scaler_v2_current_machine.joblib` | `1.1` | `fgead_live_scaler_v2_current_machine.joblib` | `YES (Active)` | `YES` | **YES** | **Category A (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_threshold.json` | `0.4` | `fgead_live_threshold.json` | `NO` | `YES` | **YES** | **Category C (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_threshold_v2_current_machine.json` | `0.3` | `fgead_live_threshold_v2_current_machine.json` | `YES (Active)` | `YES` | **YES** | **Category A (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_windows_22ch.pt` | `3727.7` | `fgead_live_windows_22ch.pt` | `YES (Fallback)` | `YES` | **YES** | **Category C (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_windows_22ch_config.json` | `1.6` | `fgead_live_windows_22ch_config.json` | `YES (Fallback)` | `YES` | **YES** | **Category C (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_windows_22ch_config_v2_current_machine.json` | `1.0` | `fgead_live_windows_22ch_config_v2_current_machine.json` | `YES (Active)` | `YES` | **YES** | **Category A (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_live_windows_22ch_v2_current_machine.pt` | `3728.0` | `fgead_live_windows_22ch_v2_current_machine.pt` | `YES (Active)` | `YES` | **YES** | **Category A (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_smd_machine_1_1.pt` | `5783.7` | `fgead_smd_machine_1_1.pt` | `NO` | `YES` | **YES** | **Category B (DO NOT TOUCH (NO))** |
| `checkpoints/fgead_smd_machine_1_1_history.npy` | `0.8` | `fgead_smd_machine_1_1_history.npy` | `NO` | `NO` | **YES** | **Category B (DO NOT TOUCH (NO))** |
| `checkpoints/train_history.npy` | `0.6` | `train_history.npy` | `NO` | `YES` | **YES** | **Category B (DO NOT TOUCH (NO))** |

> [!IMPORTANT]
> **Strict Model Isolation Maintained:**
> 1. `windows_sivachowdary_v2` (`fgead_live_windows_22ch_v2_current_machine.pt`, $\\tau=1.411807$) is the active physical host production model.
> 2. `fgead_live_windows_22ch.pt` ($\\tau=1.859450$) is preserved as the original benchmark baseline model for multi-host and regression tests.
> 3. `fgead_smd_machine_1_1.pt` & `best_model.pt` are preserved for academic SMD benchmark evaluation and paper verification.

---
## 3. Data Directory Audit (`data/`)
| Subdirectory / File | Items | Size (MB) | Purpose / Role | Category | Safety Assessment |
| :--- | :---: | :---: | :--- | :---: | :--- |
| `data/SMD/` | 113 | 463.47 MB | Server Machine Dataset train/test/label data across 28 entities | **B** | **DO NOT TOUCH** (Benchmark dataset) |
| `data/fgead_multihost.db` | 1 | 0.04 MB | Operational SQLite Database (2 registered hosts) | **A** | **DO NOT TOUCH** (Production State Store) |
| `data/fgead_multihost.db.backup_before_test_cleanup` | 1 | 0.05 MB | Historical operational DB backup before test isolation | **D** | **ARCHIVE ONLY** (Protected backup) |
| `data/live_baseline_v2_current_machine.csv` | 1 | 0.50 MB | Active current-machine normal operating baseline | **A** | **DO NOT TOUCH** (Active Baseline) |
| `data/live_baseline.csv` | 1 | 0.60 MB | Original Windows baseline dataset | **A** | **DO NOT TOUCH** (Reference Baseline) |
| `data/synthetic_data.csv` | 1 | 1.86 MB | Synthetic multivariate time series for unit tests | **A** | **DO NOT TOUCH** (Test Fixture) |
| `data/multihost_validation/` (Tests) | 9 | 0.08 MB | 8 test suites (74 tests) + `test_db_helper.py` | **C** | **DO NOT TOUCH** (Regression Suite) |
| `data/multihost_validation/` (Reports) | 16 | 0.08 MB | Phase 6, 7, 8, 9, 10 validation reports and JSON metrics | **B** | **DO NOT TOUCH** (Formal Reports) |
| `data/live_training/` | 15 | 1.41 MB | Training logs, score distributions, loss curves | **B / F** | **DO NOT TOUCH** (2 duplicate plots flagged) |
| `data/live_eda/` | 10 | 3.36 MB | Exploratory data analysis figures & baseline report | **B** | **DO NOT TOUCH** (EDA Assets) |
| `data/final_validation/` | 3 | 0.02 MB | End-to-end system validation runner and results | **B** | **DO NOT TOUCH** (Validation Suite) |

---
## 4. Test Suite Audit (74 / 74 Regression Suite)
| Test Suite File | Test Count | Target Scope | Database Isolation | Status |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `data/multihost_validation/test_smoke_phase7.py` | 7 | Core smoke tests (FastAPI, Streamlit, DB, Health) | Isolated Temp DB | **PASS** |
| `data/multihost_validation/test_cloud_deployment_phase8.py` | 4 | Cloud deployment readiness & health checks | Isolated Temp DB | **PASS** |
| `data/multihost_validation/test_host_registration_lifecycle.py` | 8 | Host registration & duplicate prevention | Isolated Temp DB | **PASS** |
| `data/multihost_validation/test_operational_db_isolation_invariant.py` | 4 | Operational DB invariant protection (exactly 2 hosts) | Invariant Guard | **PASS** |
| `data/multihost_validation/test_phase9_accuracy.py` | 18 | Accuracy, F1-score, FPR, specificity verification | Isolated Temp DB | **PASS** |
| `data/multihost_validation/test_phase10_v2_model.py` | 6 | Phase 10 v2 model, scaler, and threshold tests | Isolated Temp DB | **PASS** |
| `data/multihost_validation/test_multihost_suite.py` | 17 | Multi-host routing, buffer isolation, token auth | Isolated Temp DB | **PASS** |
| `data/multihost_validation/test_phase6_1_compatibility.py` | 10 | OS gating, model compatibility, fallback profiles | Isolated Temp DB | **PASS** |
| **Combined Total** | **74** | **Complete Full Regression Suite** | **Isolated DBs** | **74/74 PASS** |

---
## 5. Duplicate Detection & Exact Matches
A cryptographic SHA-256 hash comparison across all files discovered 6 duplicate groups:

| SHA-256 Prefix | Duplicate Files | Size (KB) | Nature of Duplication | Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| `23f996ae3bd4...` | `data/live_training/04_disk_io_vs_anomaly_score.png`<br>`data/live_training/disk_io_flag_analysis.png` | `123.9` | Redundant plot export during exploratory training phase. | **NEEDS USER APPROVAL** before deleting |
| `b89352731eab...` | `data/live_training/training_loss.png`<br>`data/live_training/validation_loss.png` | `166.5` | Redundant plot export during exploratory training phase. | **NEEDS USER APPROVAL** before deleting |
| `8189eff09e2c...` | `plots/smd_1_1_anomaly_score_timeline.png`<br>`research_paper/figures/anomaly_timeline.png` | `138.1` | Submission packaging: LaTeX `research_paper/figures/` requires standalone image assets separate from `plots/`. | **KEEP BOTH** (Required for LaTeX paper compilation) |
| `a3343dd01d4f...` | `plots/smd_1_1_confusion_matrix.png`<br>`research_paper/figures/confusion_matrix.png` | `57.1` | Submission packaging: LaTeX `research_paper/figures/` requires standalone image assets separate from `plots/`. | **KEEP BOTH** (Required for LaTeX paper compilation) |
| `09a8ddb43db9...` | `plots/smd_1_1_feature_contributions.png`<br>`research_paper/figures/xai_feature_importance.png` | `81.3` | Submission packaging: LaTeX `research_paper/figures/` requires standalone image assets separate from `plots/`. | **KEEP BOTH** (Required for LaTeX paper compilation) |
| `3fce6dfbb6b8...` | `plots/smd_1_1_roc_curve.png`<br>`research_paper/figures/model_results.png` | `94.1` | Submission packaging: LaTeX `research_paper/figures/` requires standalone image assets separate from `plots/`. | **KEEP BOTH** (Required for LaTeX paper compilation) |

---
## 6. Remote Deployment (Render) Audit
### Required for Cloud Deployment (Render / Docker):
- `requirements.txt`: Python package requirements.
- `.env.example`: Environment configuration template.
- `api/main.py`: FastAPI server entrypoint.
- `app/streamlit_app.py`: Streamlit frontend dashboard.
- `checkpoints/`: Model checkpoints and scalers.
- `data/fgead_multihost.db`: Production database.
- `run_production.sh` / `run_production.ps1`: Automated service startup scripts.

### NOT Needed by Cloud Deployment Server (Safe to omit in `.dockerignore` / Render build):
- `data/SMD/` (463.47 MB benchmark data — can be excluded from production web server container image).
- `research_paper/` (Academic paper LaTeX sources).
- `scratch/` (Temporary diagnostic scripts).
- `powershell.exe` (Stray local root binary).
- `__pycache__/` (Generated bytecode).

---
## 7. Categorized File Inventory Table (Complete Project)
| File Path | Type | Size | Category | Primary Purpose / Reason | Safe to Delete? | Evidence / References |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `.env.example` | `.example` | `1.7 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (2 refs) |
| `.gitignore` | `file` | `0.4 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (0 refs) |
| `README.md` | `.md` | `3.4 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (2 refs) |
| `agents/linux_agent.py` | `.py` | `14.8 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (4 refs) |
| `agents/windows_agent.py` | `.py` | `14.6 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (4 refs) |
| `api/__init__.py` | `.py` | `0.0 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (3 refs) |
| `api/deep_explainability.py` | `.py` | `22.6 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (1 refs) |
| `api/host_registry.py` | `.py` | `23.6 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (5 refs) |
| `api/live_inference.py` | `.py` | `22.8 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (6 refs) |
| `api/main.py` | `.py` | `57.5 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (10 refs) |
| `api/multihost_inference.py` | `.py` | `25.4 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (6 refs) |
| `app/__init__.py` | `.py` | `0.0 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (3 refs) |
| `app/streamlit_app.py` | `.py` | `143.2 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (14 refs) |
| `checkpoints/fgead_live_scaler_v2_current_machine.joblib` | `.joblib` | `1.1 KB` | **A** | Active Phase 10 current-machine production model/scaler/config/threshold. | **DO NOT TOUCH (NO)** | Production model for host_sivachowdary (tau=1.411807). (10 refs) |
| `checkpoints/fgead_live_threshold_v2_current_machine.json` | `.json` | `0.3 KB` | **A** | Active Phase 10 current-machine production model/scaler/config/threshold. | **DO NOT TOUCH (NO)** | Production model for host_sivachowdary (tau=1.411807). (4 refs) |
| `checkpoints/fgead_live_windows_22ch_config_v2_current_machine.json` | `.json` | `1.0 KB` | **A** | Active Phase 10 current-machine production model/scaler/config/threshold. | **DO NOT TOUCH (NO)** | Production model for host_sivachowdary (tau=1.411807). (3 refs) |
| `checkpoints/fgead_live_windows_22ch_v2_current_machine.pt` | `.pt` | `3.64 MB` | **A** | Active Phase 10 current-machine production model/scaler/config/threshold. | **DO NOT TOUCH (NO)** | Production model for host_sivachowdary (tau=1.411807). (10 refs) |
| `config/__init__.py` | `.py` | `0.1 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (2 refs) |
| `config/settings.py` | `.py` | `4.0 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (1 refs) |
| `core/__init__.py` | `.py` | `0.1 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (2 refs) |
| `core/logger.py` | `.py` | `2.6 KB` | **A** | Core application / API / frontend / agent code. | **DO NOT TOUCH (NO)** | Essential for FastAPI, Streamlit, and telemetry ingestion. (0 refs) |
| `data/__init__.py` | `.py` | `0.0 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (3 refs) |
| `data/baseline_compatibility.py` | `.py` | `6.8 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (1 refs) |
| `data/fgead_multihost.db` | `.db` | `44.0 KB` | **A** | Production SQLite operational database (2 hosts). | **DO NOT TOUCH (NO)** | Active multi-host registry and state store. (14 refs) |
| `data/live_agent.py` | `.py` | `16.5 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (10 refs) |
| `data/live_agent_config.json` | `.json` | `0.1 KB` | **A** | Production baseline datasets and agent configuration. | **DO NOT TOUCH (NO)** | Active normal operating baselines for inference. (3 refs) |
| `data/live_baseline.csv` | `.csv` | `615.9 KB` | **A** | Production baseline datasets and agent configuration. | **DO NOT TOUCH (NO)** | Active normal operating baselines for inference. (22 refs) |
| `data/live_baseline_v2_current_machine.csv` | `.csv` | `515.9 KB` | **A** | Production baseline datasets and agent configuration. | **DO NOT TOUCH (NO)** | Active normal operating baselines for inference. (12 refs) |
| `data/live_buffer.py` | `.py` | `5.0 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (3 refs) |
| `data/live_feature_schema.py` | `.py` | `7.0 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (7 refs) |
| `data/multihost_buffer.py` | `.py` | `4.5 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (2 refs) |
| `data/preprocessor.py` | `.py` | `6.4 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (3 refs) |
| `data/smd_loader.py` | `.py` | `22.1 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (1 refs) |
| `data/synthetic_data.csv` | `.csv` | `1.86 MB` | **A** | Production baseline datasets and agent configuration. | **DO NOT TOUCH (NO)** | Active normal operating baselines for inference. (9 refs) |
| `data/synthetic_generator.py` | `.py` | `5.6 KB` | **A** | Data pipeline, live collector, and buffer management. | **DO NOT TOUCH (NO)** | Critical components of production telemetry pipeline. (4 refs) |
| `data/windows_agent_config.json` | `.json` | `0.1 KB` | **A** | Production baseline datasets and agent configuration. | **DO NOT TOUCH (NO)** | Active normal operating baselines for inference. (4 refs) |
| `demo_cli.py` | `.py` | `4.9 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (2 refs) |
| `evaluate.py` | `.py` | `23.0 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (5 refs) |
| `evaluate_smd.py` | `.py` | `18.4 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (0 refs) |
| `evaluate_smd_final.py` | `.py` | `37.6 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (3 refs) |
| `models/__init__.py` | `.py` | `0.0 KB` | **A** | Neural network architecture and XAI explainer code. | **DO NOT TOUCH (NO)** | Imported by core inference and training pipelines. (3 refs) |
| `models/explainer.py` | `.py` | `33.6 KB` | **A** | Neural network architecture and XAI explainer code. | **DO NOT TOUCH (NO)** | Imported by core inference and training pipelines. (11 refs) |
| `models/fgead.py` | `.py` | `5.4 KB` | **A** | Neural network architecture and XAI explainer code. | **DO NOT TOUCH (NO)** | Imported by core inference and training pipelines. (9 refs) |
| `models/graph_learner.py` | `.py` | `4.6 KB` | **A** | Neural network architecture and XAI explainer code. | **DO NOT TOUCH (NO)** | Imported by core inference and training pipelines. (4 refs) |
| `models/temporal_gcn.py` | `.py` | `2.8 KB` | **A** | Neural network architecture and XAI explainer code. | **DO NOT TOUCH (NO)** | Imported by core inference and training pipelines. (4 refs) |
| `requirements.txt` | `.txt` | `0.2 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (4 refs) |
| `run_fgead.bat` | `.bat` | `5.2 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (0 refs) |
| `run_production.ps1` | `.ps1` | `3.4 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (0 refs) |
| `run_production.sh` | `.sh` | `2.6 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (0 refs) |
| `train.py` | `.py` | `6.5 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (6 refs) |
| `train_smd.py` | `.py` | `12.7 KB` | **A** | Root runtime, configuration, launcher, or benchmark script. | **DO NOT TOUCH (NO)** | Standard entrypoints, dependencies, and environment templates. (2 refs) |
| `PROJECT_DOCUMENTATION.md` | `.md` | `29.5 KB` | **B** | Comprehensive project documentation & viva defense guides. | **DO NOT TOUCH (NO)** | Crucial documentation for project evaluation and final submission. (0 refs) |
| `VIVA_EXPLANATION_GUIDE.md` | `.md` | `8.6 KB` | **B** | Comprehensive project documentation & viva defense guides. | **DO NOT TOUCH (NO)** | Crucial documentation for project evaluation and final submission. (0 refs) |
| `checkpoints/best_model.pt` | `.pt` | `3.39 MB` | **B** | SMD machine-1-1 benchmark trained checkpoint and history. | **DO NOT TOUCH (NO)** | Required for SMD benchmark evaluation and research paper reproducibility. (6 refs) |
| `checkpoints/fgead_smd_machine_1_1.pt` | `.pt` | `5.65 MB` | **B** | SMD machine-1-1 benchmark trained checkpoint and history. | **DO NOT TOUCH (NO)** | Required for SMD benchmark evaluation and research paper reproducibility. (13 refs) |
| `checkpoints/fgead_smd_machine_1_1_history.npy` | `.npy` | `0.8 KB` | **B** | SMD machine-1-1 benchmark trained checkpoint and history. | **DO NOT TOUCH (NO)** | Required for SMD benchmark evaluation and research paper reproducibility. (0 refs) |
| `checkpoints/train_history.npy` | `.npy` | `0.6 KB` | **B** | SMD machine-1-1 benchmark trained checkpoint and history. | **DO NOT TOUCH (NO)** | Required for SMD benchmark evaluation and research paper reproducibility. (3 refs) |
| `data/SMD/LICENSE` | `file` | `1.0 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-1.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (1 refs) |
| `data/SMD/interpretation_label/machine-1-2.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-3.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-4.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-5.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-6.txt` | `.txt` | `1.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-7.txt` | `.txt` | `0.5 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-1-8.txt` | `.txt` | `0.5 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-1.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-2.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-3.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-4.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-5.txt` | `.txt` | `0.5 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-6.txt` | `.txt` | `0.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-7.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-8.txt` | `.txt` | `0.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-2-9.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-1.txt` | `.txt` | `0.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-10.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-11.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-2.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-3.txt` | `.txt` | `0.8 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-4.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-5.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-6.txt` | `.txt` | `0.4 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-7.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-8.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/interpretation_label/machine-3-9.txt` | `.txt` | `0.2 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-1.txt` | `.txt` | `9.29 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (1 refs) |
| `data/SMD/test/machine-1-2.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-3.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-4.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-5.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-6.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-7.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-1-8.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-1.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-2.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-3.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-4.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-5.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-6.txt` | `.txt` | `9.37 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-7.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-8.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-2-9.txt` | `.txt` | `9.37 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-1.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-10.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-11.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-2.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-3.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-4.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-5.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-6.txt` | `.txt` | `9.37 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-7.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-8.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test/machine-3-9.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-1.txt` | `.txt` | `55.6 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (1 refs) |
| `data/SMD/test_label/machine-1-2.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-3.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-4.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-5.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-6.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-7.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-1-8.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-1.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-2.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-3.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-4.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-5.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-6.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-7.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-8.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-2-9.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-1.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-10.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-11.txt` | `.txt` | `56.0 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-2.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-3.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-4.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-5.txt` | `.txt` | `46.3 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-6.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-7.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-8.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/test_label/machine-3-9.txt` | `.txt` | `56.1 KB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-1.txt` | `.txt` | `9.29 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (1 refs) |
| `data/SMD/train/machine-1-2.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-3.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-4.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-5.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-6.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-7.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-1-8.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-1.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-2.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-3.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-4.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-5.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-6.txt` | `.txt` | `9.37 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-7.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-8.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-2-9.txt` | `.txt` | `9.37 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-1.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-10.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-11.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-2.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-3.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-4.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-5.txt` | `.txt` | `7.73 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-6.txt` | `.txt` | `9.37 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-7.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-8.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/SMD/train/machine-3-9.txt` | `.txt` | `9.36 MB` | **B** | Server Machine Dataset (SMD) raw benchmark data. | **DO NOT TOUCH (NO)** | Standard academic benchmark dataset across 28 entities. (0 refs) |
| `data/analyze_live_baseline.py` | `.py` | `25.5 KB` | **B** | Baseline telemetry collection scripts. | **DO NOT TOUCH (NO)** | Required to collect new hardware baselines. (0 refs) |
| `data/collect_baseline.py` | `.py` | `12.3 KB` | **B** | Baseline telemetry collection scripts. | **DO NOT TOUCH (NO)** | Required to collect new hardware baselines. (0 refs) |
| `data/collect_current_machine_baseline.py` | `.py` | `4.3 KB` | **B** | Baseline telemetry collection scripts. | **DO NOT TOUCH (NO)** | Required to collect new hardware baselines. (2 refs) |
| `data/final_validation/final_system_validation.md` | `.md` | `12.4 KB` | **B** | Final system validation scripts and results. | **DO NOT TOUCH (NO)** | End-to-end verification artifacts. (0 refs) |
| `data/final_validation/run_end_to_end_validation.py` | `.py` | `11.0 KB` | **B** | Final system validation scripts and results. | **DO NOT TOUCH (NO)** | End-to-end verification artifacts. (0 refs) |
| `data/final_validation/system_validation_results.json` | `.json` | `0.2 KB` | **B** | Final system validation scripts and results. | **DO NOT TOUCH (NO)** | End-to-end verification artifacts. (1 refs) |
| `data/live_eda/01_feature_distributions.png` | `.png` | `638.9 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/02_feature_boxplots.png` | `.png` | `304.1 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/03_correlation_heatmap.png` | `.png` | `325.2 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/04_cpu_timeline.png` | `.png` | `1.05 MB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/05_memory_timeline.png` | `.png` | `169.0 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/06_disk_io_timeline.png` | `.png` | `170.0 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/07_network_timeline.png` | `.png` | `478.7 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/08_process_count_timeline.png` | `.png` | `130.1 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/09_feature_variability.png` | `.png` | `128.3 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_eda/live_baseline_report.md` | `.md` | `13.5 KB` | **B** | Exploratory Data Analysis (EDA) charts and reports. | **DO NOT TOUCH (NO)** | Data exploration evidence for baseline distribution. (1 refs) |
| `data/live_training/01_test_anomaly_timeline.png` | `.png` | `115.1 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/02_flagged_feature_timeline.png` | `.png` | `446.5 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/03_validation_vs_test_distribution.png` | `.png` | `116.6 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/disk_io_flag_analysis.png` | `.png` | `123.9 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/flagged_feature_analysis.csv` | `.csv` | `2.8 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/live_model_report.md` | `.md` | `3.5 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/phase3_5_investigation.md` | `.md` | `5.4 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/test_anomaly_episodes.csv` | `.csv` | `0.4 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/test_flagged_windows.csv` | `.csv` | `7.4 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/test_score_distribution.png` | `.png` | `67.1 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/training_history.csv` | `.csv` | `3.2 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/live_training/validation_loss.png` | `.png` | `166.5 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (2 refs) |
| `data/live_training/validation_score_distribution.png` | `.png` | `94.6 KB` | **B** | Live model training logs, metrics, and loss plots. | **DO NOT TOUCH (NO)** | Documentary evidence for model training. (1 refs) |
| `data/multihost_validation/baseline_comparison_report.json` | `.json` | `12.6 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/evaluate_phase9_metrics.py` | `.py` | `12.3 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (0 refs) |
| `data/multihost_validation/full_regression_audit_summary.json` | `.json` | `2.4 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/multihost_test_results.json` | `.json` | `0.8 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/net_drops_compatibility_report.md` | `.md` | `7.9 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/phase10_evaluation_metrics.json` | `.json` | `1.3 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/phase6_1_model_compatibility.md` | `.md` | `9.9 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (0 refs) |
| `data/multihost_validation/phase6_1_test_results.json` | `.json` | `0.6 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/phase6_report.md` | `.md` | `11.9 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (0 refs) |
| `data/multihost_validation/phase9_accuracy_report.md` | `.md` | `9.1 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/phase9_evaluation_metrics.json` | `.json` | `3.4 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (4 refs) |
| `data/multihost_validation/phase_frontend_professional_report.md` | `.md` | `6.7 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (0 refs) |
| `data/multihost_validation/phase_frontend_redesign_report.md` | `.md` | `7.4 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (0 refs) |
| `data/multihost_validation/phase_frontend_rollback_report.md` | `.md` | `3.7 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (0 refs) |
| `data/multihost_validation/step2_live_normal_test_results.json` | `.json` | `0.4 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `data/multihost_validation/step3_4_controlled_anomaly_recovery_results.json` | `.json` | `4.2 KB` | **B** | Multi-host validation reports and evaluation metrics. | **DO NOT TOUCH (NO)** | Formal phase validation reports. (1 refs) |
| `docs/DEPLOYMENT.md` | `.md` | `11.4 KB` | **B** | Primary system and Phase 10 documentation. | **DO NOT TOUCH (NO)** | Core technical documentation for deployment and submission. (0 refs) |
| `docs/phase10_current_machine_baseline_report.md` | `.md` | `10.9 KB` | **B** | Primary system and Phase 10 documentation. | **DO NOT TOUCH (NO)** | Core technical documentation for deployment and submission. (0 refs) |
| `docs/phase10_final_four_demo_validation.md` | `.md` | `10.8 KB` | **B** | Primary system and Phase 10 documentation. | **DO NOT TOUCH (NO)** | Core technical documentation for deployment and submission. (0 refs) |
| `docs/phase10_final_live_verification.md` | `.md` | `8.7 KB` | **B** | Primary system and Phase 10 documentation. | **DO NOT TOUCH (NO)** | Core technical documentation for deployment and submission. (0 refs) |
| `fgead_architecture_diagrams.md` | `.md` | `4.1 KB` | **B** | Comprehensive project documentation & viva defense guides. | **DO NOT TOUCH (NO)** | Crucial documentation for project evaluation and final submission. (0 refs) |
| `fgead_complete_technical_flow.md` | `.md` | `31.0 KB` | **B** | Comprehensive project documentation & viva defense guides. | **DO NOT TOUCH (NO)** | Crucial documentation for project evaluation and final submission. (0 refs) |
| `generate_plots.py` | `.py` | `19.6 KB` | **B** | Evaluation and plotting utilities. | **DO NOT TOUCH (NO)** | Plotting and verification utilities. (0 refs) |
| `generate_smd_plots.py` | `.py` | `16.0 KB` | **B** | Evaluation and plotting utilities. | **DO NOT TOUCH (NO)** | Plotting and verification utilities. (0 refs) |
| `models/train_current_machine_v2.py` | `.py` | `12.2 KB` | **B** | Model training & recalibration pipelines. | **DO NOT TOUCH (NO)** | Required for model reproducibility and retraining. (1 refs) |
| `models/train_live_model.py` | `.py` | `27.7 KB` | **B** | Model training & recalibration pipelines. | **DO NOT TOUCH (NO)** | Required for model reproducibility and retraining. (0 refs) |
| `notebooks/eda.py` | `.py` | `5.0 KB` | **B** | Data analysis notebooks / scripts. | **DO NOT TOUCH (NO)** | Jupyter / EDA scripts. (2 refs) |
| `plots/anomaly_score_timeline.png` | `.png` | `195.3 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (3 refs) |
| `plots/class_balance.png` | `.png` | `16.3 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (2 refs) |
| `plots/confusion_matrix.png` | `.png` | `52.4 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (5 refs) |
| `plots/correlation_matrix.png` | `.png` | `96.7 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (2 refs) |
| `plots/evaluation_results.txt` | `.txt` | `0.2 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (4 refs) |
| `plots/feature_distributions.png` | `.png` | `108.3 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (3 refs) |
| `plots/precision_recall_curve.png` | `.png` | `58.2 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (2 refs) |
| `plots/roc_curve.png` | `.png` | `88.0 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (3 refs) |
| `plots/smd_1-1_FINAL_results.txt` | `.txt` | `1.8 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (0 refs) |
| `plots/smd_1_1_FINAL_results.txt` | `.txt` | `3.7 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (3 refs) |
| `plots/smd_1_1_anomaly_score_timeline.png` | `.png` | `138.1 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (1 refs) |
| `plots/smd_1_1_confusion_matrix.png` | `.png` | `57.1 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (1 refs) |
| `plots/smd_1_1_evaluation.txt` | `.txt` | `0.4 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (0 refs) |
| `plots/smd_1_1_feature_contributions.png` | `.png` | `81.3 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (1 refs) |
| `plots/smd_1_1_plot_results.txt` | `.txt` | `0.3 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (0 refs) |
| `plots/smd_1_1_precision_recall_curve.png` | `.png` | `77.8 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (0 refs) |
| `plots/smd_1_1_roc_curve.png` | `.png` | `94.1 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (1 refs) |
| `plots/time_series_overview.png` | `.png` | `667.8 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (2 refs) |
| `plots/training_validation_loss.png` | `.png` | `113.5 KB` | **B** | Benchmark and system visualization plots. | **DO NOT TOUCH (NO)** | Visualization assets for project analysis and evaluation. (1 refs) |
| `research_paper/README.md` | `.md` | `2.8 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (2 refs) |
| `research_paper/author_review.md` | `.md` | `6.2 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/figures/README.md` | `.md` | `0.3 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (3 refs) |
| `research_paper/figures/anomaly_timeline.png` | `.png` | `138.1 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (2 refs) |
| `research_paper/figures/confusion_matrix.png` | `.png` | `57.1 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (5 refs) |
| `research_paper/figures/explanation_example.png` | `.png` | `196.8 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/figures/methodology.png` | `.png` | `101.4 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/figures/model_results.png` | `.png` | `94.1 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/figures/system_architecture.png` | `.png` | `219.6 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/figures/xai_feature_importance.png` | `.png` | `81.3 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/final_audit_report.md` | `.md` | `11.5 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/final_submission_check.md` | `.md` | `5.3 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (0 refs) |
| `research_paper/generate_paper_figures.py` | `.py` | `6.1 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/literature/20_papers.md` | `.md` | `8.3 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/literature/literature_review.md` | `.md` | `10.2 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/literature/research_gap.md` | `.md` | `4.5 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/paper.md` | `.md` | `30.7 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/paper.pdf` | `.pdf` | `516.6 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (2 refs) |
| `research_paper/paper.tex` | `.tex` | `33.2 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (4 refs) |
| `research_paper/references.bib` | `.bib` | `8.1 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (3 refs) |
| `research_paper/verification/project_facts.md` | `.md` | `13.2 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `research_paper/verification/reference_verification.md` | `.md` | `5.6 KB` | **B** | Research paper LaTeX source, bibtex, figures, and literature review. | **DO NOT TOUCH (NO)** | Required for final academic submission. (1 refs) |
| `validate_net_drops_fix.py` | `.py` | `15.9 KB` | **B** | Evaluation and plotting utilities. | **DO NOT TOUCH (NO)** | Plotting and verification utilities. (0 refs) |
| `checkpoints/fgead_live_scaler.joblib` | `.joblib` | `1.1 KB` | **C** | Original live model baseline and fallback profile. | **DO NOT TOUCH (NO)** | Referenced in multi-host compatibility and regression suites. (21 refs) |
| `checkpoints/fgead_live_threshold.json` | `.json` | `0.4 KB` | **C** | Original live model baseline and fallback profile. | **DO NOT TOUCH (NO)** | Referenced in multi-host compatibility and regression suites. (15 refs) |
| `checkpoints/fgead_live_windows_22ch.pt` | `.pt` | `3.64 MB` | **C** | Original live model baseline and fallback profile. | **DO NOT TOUCH (NO)** | Referenced in multi-host compatibility and regression suites. (25 refs) |
| `checkpoints/fgead_live_windows_22ch_config.json` | `.json` | `1.6 KB` | **C** | Original live model baseline and fallback profile. | **DO NOT TOUCH (NO)** | Referenced in multi-host compatibility and regression suites. (5 refs) |
| `data/fgead_test_phase7.db` | `.db` | `32.0 KB` | **C** | Test suite database fixture path. | **NO (MANAGED BY TESTS)** | Referenced in automated test suites. (1 refs) |
| `data/fgead_test_phase8.db` | `.db` | `32.0 KB` | **C** | Test suite database fixture path. | **NO (MANAGED BY TESTS)** | Referenced in automated test suites. (1 refs) |
| `data/fgead_test_phase9.db` | `.db` | `32.0 KB` | **C** | Test suite database fixture path. | **NO (MANAGED BY TESTS)** | Referenced in automated test suites. (1 refs) |
| `data/multihost_validation/test_cloud_deployment_phase8.py` | `.py` | `10.3 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (10 refs) |
| `data/multihost_validation/test_db_helper.py` | `.py` | `1.6 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (1 refs) |
| `data/multihost_validation/test_host_registration_lifecycle.py` | `.py` | `8.7 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (5 refs) |
| `data/multihost_validation/test_multihost_suite.py` | `.py` | `21.3 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (13 refs) |
| `data/multihost_validation/test_operational_db_isolation_invariant.py` | `.py` | `3.6 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (3 refs) |
| `data/multihost_validation/test_phase10_v2_model.py` | `.py` | `7.5 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (2 refs) |
| `data/multihost_validation/test_phase6_1_compatibility.py` | `.py` | `18.4 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (12 refs) |
| `data/multihost_validation/test_phase9_accuracy.py` | `.py` | `14.5 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (11 refs) |
| `data/multihost_validation/test_smoke_phase7.py` | `.py` | `9.9 KB` | **C** | Active regression test suite (74/74 tests). | **DO NOT TOUCH (NO)** | Automated regression and platform verification tests. (10 refs) |
| `data/fgead_multihost.db.backup_before_test_cleanup` | `.backup_before_test_cleanup` | `48.0 KB` | **D** | Historical database backup taken before test isolation. | **NO (ARCHIVE ONLY)** | Protected reference database snapshot. (2 refs) |
| `docs/host_count_diagnosis.md` | `.md` | `8.5 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/host_database_isolation_report.md` | `.md` | `7.1 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/host_identity_registration_report.md` | `.md` | `7.2 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/live_false_positive_feature_audit.md` | `.md` | `10.9 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/live_telemetry_auth_fix_report.md` | `.md` | `4.4 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/live_verification_results.json` | `.json` | `0.8 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/phase10_live_dashboard_runtime_debug.md` | `.md` | `8.8 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/phase10_live_runtime_second_forensics.md` | `.md` | `12.4 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `docs/runtime_live_verification_report.md` | `.md` | `6.4 KB` | **D** | Historical phase diagnostic and forensic reports. | **NO (ARCHIVE ONLY)** | Detailed phase debugging logs and audit trails. (0 refs) |
| `models/investigate_test_flags.py` | `.py` | `24.1 KB` | **D** | Historical model diagnostic utility. | **NO (OPTIONAL)** | Research investigation script. (0 refs) |
| `scratch/execute_final_four_demo_validation.py` | `.py` | `13.9 KB` | **D** | Validation / Regression execution harness. | **NO** | Used to re-verify live models and regression suite. (0 refs) |
| `scratch/run_full_regression_audit.py` | `.py` | `4.4 KB` | **D** | Validation / Regression execution harness. | **NO** | Used to re-verify live models and regression suite. (0 refs) |
| `__pycache__/demo_cli.cpython-311.pyc` | `.pyc` | `8.8 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate.cpython-311.pyc` | `.pyc` | `22.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate_smd.cpython-311.pyc` | `.pyc` | `19.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate_smd_final.cpython-311.pyc` | `.pyc` | `40.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate_smd_final_backup.cpython-311.pyc` | `.pyc` | `35.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate_smd_final_backup2.cpython-311.pyc` | `.pyc` | `36.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate_smd_final_broken.cpython-311.pyc` | `.pyc` | `38.6 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/evaluate_smd_threshold_backup.cpython-311.pyc` | `.pyc` | `28.6 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/generate_plots.cpython-311.pyc` | `.pyc` | `22.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/generate_smd_plots.cpython-311.pyc` | `.pyc` | `25.4 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/train.cpython-311.pyc` | `.pyc` | `10.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `__pycache__/train_smd.cpython-311.pyc` | `.pyc` | `12.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `agents/__pycache__/linux_agent.cpython-311.pyc` | `.pyc` | `18.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `agents/__pycache__/windows_agent.cpython-311.pyc` | `.pyc` | `18.6 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/__init__.cpython-311.pyc` | `.pyc` | `0.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/__init__.cpython-312.pyc` | `.pyc` | `0.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/deep_explainability.cpython-311.pyc` | `.pyc` | `21.4 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/host_registry.cpython-311.pyc` | `.pyc` | `34.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/live_inference.cpython-311.pyc` | `.pyc` | `28.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/main.cpython-311.pyc` | `.pyc` | `64.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/main.cpython-312.pyc` | `.pyc` | `8.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `api/__pycache__/multihost_inference.cpython-311.pyc` | `.pyc` | `31.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `app/__pycache__/__init__.cpython-311.pyc` | `.pyc` | `0.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `app/__pycache__/streamlit_app.cpython-311.pyc` | `.pyc` | `159.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `app/__pycache__/streamlit_app.cpython-312.pyc` | `.pyc` | `32.7 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `app/__pycache__/streamlit_app_backup.cpython-311.pyc` | `.pyc` | `49.8 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `config/__pycache__/__init__.cpython-311.pyc` | `.pyc` | `0.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `config/__pycache__/settings.cpython-311.pyc` | `.pyc` | `6.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `core/__pycache__/__init__.cpython-311.pyc` | `.pyc` | `0.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `core/__pycache__/logger.cpython-311.pyc` | `.pyc` | `4.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/__init__.cpython-311.pyc` | `.pyc` | `0.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/__init__.cpython-312.pyc` | `.pyc` | `0.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/baseline_compatibility.cpython-311.pyc` | `.pyc` | `8.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/live_agent.cpython-311.pyc` | `.pyc` | `22.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/live_buffer.cpython-311.pyc` | `.pyc` | `8.7 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/live_feature_schema.cpython-311.pyc` | `.pyc` | `8.7 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/multihost_buffer.cpython-311.pyc` | `.pyc` | `8.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/preprocessor.cpython-311.pyc` | `.pyc` | `9.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/preprocessor.cpython-312.pyc` | `.pyc` | `8.4 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/smd_loader.cpython-311.pyc` | `.pyc` | `23.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/__pycache__/synthetic_generator.cpython-311.pyc` | `.pyc` | `7.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_cloud_deployment_phase8.cpython-311.pyc` | `.pyc` | `15.6 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_db_helper.cpython-311.pyc` | `.pyc` | `2.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_host_registration_lifecycle.cpython-311.pyc` | `.pyc` | `13.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_multihost_suite.cpython-311.pyc` | `.pyc` | `26.4 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_operational_db_isolation_invariant.cpython-311.pyc` | `.pyc` | `7.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_phase10_v2_model.cpython-311.pyc` | `.pyc` | `12.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_phase6_1_compatibility.cpython-311.pyc` | `.pyc` | `20.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_phase9_accuracy.cpython-311.pyc` | `.pyc` | `25.4 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `data/multihost_validation/__pycache__/test_smoke_phase7.cpython-311.pyc` | `.pyc` | `15.7 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/__init__.cpython-311.pyc` | `.pyc` | `0.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/__init__.cpython-312.pyc` | `.pyc` | `0.1 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/explainer.cpython-311.pyc` | `.pyc` | `30.5 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/explainer.cpython-312.pyc` | `.pyc` | `22.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/fgead.cpython-311.pyc` | `.pyc` | `5.8 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/fgead.cpython-312.pyc` | `.pyc` | `5.2 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/graph_learner.cpython-311.pyc` | `.pyc` | `6.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/graph_learner.cpython-312.pyc` | `.pyc` | `6.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/temporal_gcn.cpython-311.pyc` | `.pyc` | `4.7 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/temporal_gcn.cpython-312.pyc` | `.pyc` | `4.3 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `models/__pycache__/train_live_model.cpython-311.pyc` | `.pyc` | `38.4 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `notebooks/__pycache__/eda.cpython-311.pyc` | `.pyc` | `10.9 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `powershell.exe` | `.exe` | `484.0 KB` | **E** | Accidentally copied binary in project root. | **YES** | Stray executable not part of source code or dependencies. (0 refs) |
| `research_paper/__pycache__/generate_paper_figures.cpython-311.pyc` | `.pyc` | `8.0 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `research_paper/paper.aux` | `.aux` | `7.3 KB` | **E** | LaTeX build intermediate artifact. | **YES** | Generated during PDF compilation from paper.tex. (0 refs) |
| `research_paper/paper.bbl` | `.bbl` | `7.3 KB` | **E** | LaTeX build intermediate artifact. | **YES** | Generated during PDF compilation from paper.tex. (0 refs) |
| `research_paper/paper.blg` | `.blg` | `1.5 KB` | **E** | LaTeX build intermediate artifact. | **YES** | Generated during PDF compilation from paper.tex. (0 refs) |
| `research_paper/paper.log` | `.log` | `18.8 KB` | **E** | LaTeX build intermediate artifact. | **YES** | Generated during PDF compilation from paper.tex. (0 refs) |
| `scratch/__pycache__/test_live_v2_inference.cpython-311.pyc` | `.pyc` | `2.6 KB` | **E** | Python bytecode compilation cache. | **YES** | Generated bytecode, automatically re-created by Python. (0 refs) |
| `scratch/audit_live_features.py` | `.py` | `5.4 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/audit_project_inventory.py` | `.py` | `3.5 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/compare_baselines.py` | `.py` | `3.2 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/diagnose_live_buffer.py` | `.py` | `0.5 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/final_four_demo_results.json` | `.json` | `3.7 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (1 refs) |
| `scratch/live_120s_test_results.json` | `.json` | `0.4 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (1 refs) |
| `scratch/live_forensic_audit.py` | `.py` | `5.4 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/run_controlled_anomalies_and_recovery.py` | `.py` | `7.2 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/run_live_normal_test.py` | `.py` | `4.5 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/second_level_forensics.py` | `.py` | `7.5 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/second_level_forensics_output.json` | `.json` | `12.1 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (1 refs) |
| `scratch/test_live_120s_runtime.py` | `.py` | `5.5 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/test_live_v2_inference.py` | `.py` | `1.9 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/test_sequential_v2.py` | `.py` | `1.6 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `scratch/verify_phase10_runtime.py` | `.py` | `2.2 KB` | **E** | Temporary diagnostic / forensic script or dump. | **YES (or ARCHIVE)** | One-off diagnostic scripts and output files from earlier debugging phases. (0 refs) |
| `data/live_training/04_disk_io_vs_anomaly_score.png` | `.png` | `123.9 KB` | **F** | Duplicate image in live training folder. | **NO (NEEDS APPROVAL)** | Exact SHA-256 duplicate of other plot in same folder. (1 refs) |
| `data/live_training/training_loss.png` | `.png` | `166.5 KB` | **F** | Duplicate image in live training folder. | **NO (NEEDS APPROVAL)** | Exact SHA-256 duplicate of other plot in same folder. (1 refs) |
| `data/live_inference/phase4_report.md` | `.md` | `8.7 KB` | **G** | Data folder file. | **NO** | Protected data file. (0 refs) |
| `data/live_inference/verify_phase4.py` | `.py` | `10.6 KB` | **G** | Data folder file. | **NO** | Protected data file. (1 refs) |

---
## 8. Summary Cleanup Action Plan

### SAFE TO DELETE (Zero Risk, Automated Regenerated / Stray Files):
1. `powershell.exe` (Stray binary in root folder — `484.0 KB`)
2. All `__pycache__/*.pyc` bytecode files (64 files across 12 folders — `1.14 MB`)
3. LaTeX build intermediate files: `research_paper/paper.aux`, `research_paper/paper.bbl`, `research_paper/paper.blg`, `research_paper/paper.log` (`34.9 KB`)
4. Temporary diagnostic dumps: `scratch/live_120s_test_results.json`, `scratch/second_level_forensics_output.json`, `scratch/audit_inventory_raw.json` (`16.2 KB`)

### SAFE TO ARCHIVE (Move to `archive/` folder if desired):
1. `data/fgead_multihost.db.backup_before_test_cleanup` (`48.0 KB`)
2. One-off diagnostic scratch scripts: `scratch/audit_live_features.py`, `scratch/compare_baselines.py`, `scratch/diagnose_live_buffer.py`, `scratch/live_forensic_audit.py`, `scratch/second_level_forensics.py`, `scratch/test_live_120s_runtime.py`, `scratch/test_live_v2_inference.py`, `scratch/test_sequential_v2.py`, `scratch/verify_phase10_runtime.py` (`36.7 KB`)
3. Historical phase debug docs: `docs/phase10_live_dashboard_runtime_debug.md`, `docs/phase10_live_runtime_second_forensics.md`, `docs/live_false_positive_feature_audit.md` (`32.1 KB`)

### KEEP (Required for Academic Submission & Benchmarking):
1. All `research_paper/` sources, figures, bibtex, and literature files.
2. `data/SMD/` benchmark datasets (463.47 MB).
3. Benchmark models: `checkpoints/fgead_smd_machine_1_1.pt`, `checkpoints/best_model.pt`.
4. `evaluate.py`, `evaluate_smd.py`, `evaluate_smd_final.py`, `train_smd.py`.
5. `PROJECT_DOCUMENTATION.md`, `VIVA_EXPLANATION_GUIDE.md`, `fgead_complete_technical_flow.md`.

### DO NOT TOUCH (Strict Production Invariants):
1. **Operational Database:** `data/fgead_multihost.db` (2 hosts).
2. **Active Production Model:** `checkpoints/fgead_live_windows_22ch_v2_current_machine.pt`, `checkpoints/fgead_live_scaler_v2_current_machine.joblib`, `checkpoints/fgead_live_threshold_v2_current_machine.json` ($\\tau = 1.411807$).
3. **Active Baseline Data:** `data/live_baseline_v2_current_machine.csv`.
4. **Production Server & UI:** `api/main.py`, `app/streamlit_app.py`, `api/multihost_inference.py`, `data/live_agent.py`, `config/settings.py`.
5. **Full Regression Suite:** `data/multihost_validation/test_*.py` (74 tests).

### NEEDS MY APPROVAL (Explicit User Decision Required):
1. `data/live_training/04_disk_io_vs_anomaly_score.png` vs `data/live_training/disk_io_flag_analysis.png` (Exact duplicate — approve removal of one copy?).
2. `data/live_training/training_loss.png` vs `data/live_training/validation_loss.png` (Exact duplicate — approve removal of one copy?).
3. `data/fgead_test_phase7.db`, `data/fgead_test_phase8.db`, `data/fgead_test_phase9.db` (Stale SQLite test files — approve cleanup or keep?).
