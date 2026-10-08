# FGEAD V3 — DEVELOPER HANDOFF & SYSTEM OPERATIONS GUIDE

**Project:** FGEAD — Feature-Level Graph-Based Explainable Anomaly Detection  
**Target Host Profile:** `host_sivachowdary` (Windows Physical Host)  
**Active Model ID:** `windows_sivachowdary_v3`  
**Calibrated Anomaly Threshold ($\tau$):** `2.120169`  
**Evaluation Status:** 🟢 **DEMO READY WITH KNOWN LIMITATIONS**  

---

## CRITICAL DEVELOPER DIRECTIVES

> [!IMPORTANT]
> 1. **DO NOT retrain V3** unless a concrete validation failure or dataset corruption is empirically proven.
> 2. **DO NOT lower the threshold ($\tau = 2.120169$)** merely to make the disk-read detection test pass.
> 3. **DO NOT replace the real-machine baseline dataset** (`data/live_baseline_v3_current_machine.csv`) with synthetic data.
> 4. **DO NOT modify the 22-feature schema** or change feature ordering in `data/live_feature_schema.py`.
> 5. **DO NOT modify or overwrite legacy V2 artifacts** (`checkpoints/*v2*`).

---

## A. Project Overview

FGEAD is a graph neural network (GCN) + LSTM forecasting anomaly detection system designed to monitor high-dimensional multivariate physical telemetry. V3 was developed to resolve false-positive memory alerts present in V2 by recalibrating against a continuous 1,800-sample (30-minute) real-world telemetry baseline collected on `host_sivachowdary`.

---

## B. Current Validated Model

- **Model ID:** `windows_sivachowdary_v3`
- **Architecture:** 22-Channel Spatial-Temporal Graph Attention Neural Network (FGEAD)
  - Feature Embedding (dim=64)
  - Multi-Head Self-Attention Graph Learner (heads=4, sparsity threshold=0.3)
  - Dynamic Feature Graph + Temporal GCN (out=64)
  - LSTM Forecasting (hidden=128)
- **Training Epochs:** 20 (Final Train Loss: `0.1395`)
- **Validation FPR:** `0.00%` on unseen held-out normal test set (`v3_test_normal.csv`)
- **Specificity:** `100.00%`

---

## C. V3 Artifact Manifest

The complete set of production V3 deployment artifacts is checked into version control:

1. **PyTorch Model Checkpoint:** `checkpoints/fgead_live_windows_22ch_v3_current_machine.pt`
   - SHA256: `8e9e7c418491579f7832c1b1dfea2b881bd4c233cb521a219c7842a114e9739a`
2. **Standard Scaler:** `checkpoints/fgead_live_scaler_v3_current_machine.joblib`
   - SHA256: `d1d2746402df1db9c6bd1d3caf4075a23f03ca7403a2cb8f8daf8a21c7430953`
3. **Threshold Config:** `checkpoints/fgead_live_threshold_v3_current_machine.json`
   - SHA256: `acbd9117f936065d4a2b14d955eec51a73f1a71db443e41768601ccc716f9905`
4. **Model Config:** `checkpoints/fgead_live_windows_22ch_config_v3_current_machine.json`
   - SHA256: `889b099cabde6dcdbbf43e27abc05a45bb4e17a4d32d82e53e1e3d47871b8f91`
5. **Baseline Dataset:** `data/live_baseline_v3_current_machine.csv` (1,800 continuous rows)

---

## D. V3 Calibrated Threshold

- **Calibrated Threshold ($\tau$):** **`2.120169`**
- Computed as P99.5 + safety margin on Train + Calibration normal windows.
- Any rolling $60 \times 22$ window producing a maximum forecasting residual score $> 2.120169$ is flagged as an anomaly.

---

## E. 22-Feature Physical Telemetry Schema

Strict 22-channel schema defined in `data/live_feature_schema.py`:

```
 0: cpu_percent                 11: disk_read_bytes_per_sec
 1: cpu_freq_current            12: disk_write_bytes_per_sec
 2: cpu_user_time_percent       13: disk_read_count_per_sec
 3: cpu_system_time_percent     14: disk_write_count_per_sec
 4: cpu_ctx_switches_per_sec    15: net_bytes_sent_per_sec
 5: cpu_interrupts_per_sec      16: net_bytes_recv_per_sec
 6: memory_percent              17: net_packets_sent_per_sec
 7: memory_available_mb         18: net_packets_recv_per_sec
 8: memory_used_mb              19: net_errors_total
 9: swap_percent                20: net_drops_total
10: disk_usage_percent          21: process_count
```

---

## F. How to Start the FastAPI Backend Gateway

From PowerShell in virtual environment:

```powershell
.\venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Verify backend health:
```powershell
curl http://127.0.0.1:8000/health
```
Expected: `{"status":"ok", "model_loaded":true, ...}`

---

## G. How to Start the Streamlit Dashboard

In a separate terminal:

```powershell
.\venv\Scripts\python -m streamlit run app\streamlit_app.py
```

Dashboard loads automatically at `http://localhost:8501`.

---

## H. How to Start the Live Telemetry Agent

In a separate terminal:

```powershell
.\venv\Scripts\python data/live_agent.py
```

The agent automatically registers with `host_id = host_sivachowdary` and streams 22-channel telemetry every 1.0 second.

---

## I. How to Verify Host Assignment

Query the host registry endpoint:

```powershell
curl http://127.0.0.1:8000/hosts
```

Confirm:
- `host_id`: `host_sivachowdary`
- `model_id`: `windows_sivachowdary_v3`
- `model_compatibility`: `Compatible`

---

## J. How to Run "Scan My System"

1. Open Streamlit Dashboard (`http://localhost:8501`).
2. Navigate to **Single Host Monitoring** tab.
3. Select `host_sivachowdary` from the host dropdown.
4. Click **🔍 Scan My System**.
5. Observe the 4-stage execution pipeline (Telemetry Ingestion $\rightarrow$ Validation $\rightarrow$ 60s Analysis $\rightarrow$ Evidence Check).

---

## K. Controlled Anomaly Testing Instructions

To run automated physical workload injection tests:

```powershell
.\venv\Scripts\python data/live_training/run_v3_controlled_anomaly_eval.py
```

Tests executed:
- **Test [A]: CPU Stress Burst** $\rightarrow$ Peak Score `3.0671` (Detected 🟢)
- **Test [B]: Disk Write Burst** $\rightarrow$ Peak Score `5.9854` (Detected 🟢)
- **Test [C]: Disk Read Burst** $\rightarrow$ Peak Score `1.7776` (Undetected 🔴 - Known Limitation)
- **Test [D]: Network Burst** $\rightarrow$ Peak Score `99.6019` (Detected 🟢)
- **Test [E]: Combined CPU + Storage** $\rightarrow$ Peak Score `5.4778` (Detected 🟢)

---

## L. Known Disk-Read Limitation

Random disk-read bursts elevate the anomaly score from baseline ($\sim 0.31$) to `1.7776`, which remains below the calibrated threshold $\tau = 2.120169$. Disk-read burst is classified as undetected. Do **not** lower $\tau$ to force detection, as doing so will compromise normal-state specificity.

---

## M. V2 Rollback Information

If you ever need to fall back to V2 for legacy comparison:
- Model: `checkpoints/fgead_live_windows_22ch.pt`
- Scaler: `checkpoints/fgead_live_scaler_v2_current_machine.joblib`
- Threshold: `checkpoints/fgead_live_threshold_v2_current_machine.json` ($\tau = 1.411807$)

V2 hashes are preserved in `data/live_training/V3_MODEL_TRAINING_REPORT.md`.

---

## N. Important Documentation Files

- `README.md` — Project Overview & Quick Start
- `data/live_training/V3_MODEL_TRAINING_REPORT.md` — Model Training & Calibration details
- `data/live_training/V3_FINAL_LIVE_ACCEPTANCE_REPORT.md` — 20-Phase Live Integration Results
- `reports/V3_LIVE_MODEL_FINAL_REPORT.md` — Summary Executive Report
- `reports/V3_LIVE_MODEL_METRICS.json` — Machine-readable evaluation metrics

---

## O. Things NOT to Change

1. `LIVE_FEATURES` list in `data/live_feature_schema.py` (22 names & ordering).
2. `CUMULATIVE_COUNTER_BASELINE_ANCHORS` dictionary (anchors `net_drops_total` & `net_errors_total`).
3. Model hyperparameters in `models/fgead.py`.
4. `StandardScaler` fitting procedure (must ONLY fit on normal training data).

---

## P. Troubleshooting

- **FastAPI Port 8000 in use:**  
  `Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }`
- **Buffer size < 60:**  
  Wait 60 seconds after starting agent for initial window collection.
- **Model showing V2 instead of V3:**  
  Run `python -c "from api.host_registry import HostRegistry; hr=HostRegistry(); conn=hr._get_connection(); conn.cursor().execute('UPDATE hosts SET model_id=\'windows_sivachowdary_v3\' WHERE host_id=\'host_sivachowdary\''); conn.commit()"` then restart FastAPI.

---

## Q. Current Git Commit / Version

- **Branch:** `main`
- **Model State:** `windows_sivachowdary_v3` (Validated & Frozen)
- **Status:** 🟢 **DEMO READY WITH KNOWN LIMITATIONS**
