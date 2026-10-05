# FGEAD Runtime Live Verification & Diagnostic Audit Report

## 1. Executive Summary

This report documents the end-to-end runtime verification of the FGEAD real-time multi-host anomaly detection platform following the cumulative-counter compatibility stabilization and human-understandable explainability upgrade.

All 56 automated validation tests across all 5 suites pass with 100% success. Furthermore, live physical telemetry from the host machine was streamed, ingested, and evaluated against the newly started FastAPI backend and Streamlit dashboard over a clean 185-sample (120+ second) monitoring run.

---

## 2. End-to-End Runtime Path Trace

```
[Physical Host OS / psutil]
       │  (Samples 22 physical telemetry counters every 1s)
       ▼
[data/live_agent.py]  (or agents/windows_agent.py)
       │  (Constructs JSON payload with physical counters, e.g. net_drops_total=0.0 or physical counter)
       ▼  HTTP POST /hosts/{host_id}/telemetry  (authenticated with X-Agent-Token)
[api/main.py -> ingest_host_telemetry()]
       │  (Validates 22 channels, schema version 1.0, finite values, physical bounds)
       ▼
[data/multihost_buffer.py -> MultiHostBufferManager]
       │  (Maintains isolated rolling ring buffer W=60, S=1 for host_id)
       ▼
[data/live_feature_schema.py -> prepare_model_input_window()]
       │  (Preserves physical telemetry for UI; anchors invariant cumulative counters,
       │   e.g. net_drops_total -> baseline constant 294.0)
       ▼
[api/multihost_inference.py -> MultiHostInferenceManager.infer_host_window()]
       │  (Scales anchored input window using fitted StandardScaler)
       ▼
[models/fgead.py -> FGEAD.forward()]
       │  (Calculates feature graph attention, spatio-temporal embeddings, and forecasting residuals)
       ▼
[api/deep_explainability.py -> build_deep_human_explanation()]
       │  (Zeroes out anchored cumulative counter residuals; generates structured, cautious plain-English XAI)
       ▼
[api/live_inference.py -> LiveEpisodeTracker]
       │  (Debounced contiguous episode tracking with min_persistence_frames=2)
       ▼
[app/streamlit_app.py]
       │  (Fetches GET /hosts/{host_id}/analysis & renders Plain-English Hero Card,
       │   Comparison Table, and Feature Residual Breakdowns)
```

---

## 3. Physical Telemetry vs. Model-Input Ingestion Comparison

| Metric Channel | Physical Telemetry (Observed) | Model Input (Anchored) | Baseline Mean ($\mu$) | Baseline Std ($\sigma$) | Normalized Excursion | Top Attribution Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `net_drops_total` | **`0.0`** (or OS drop count) | **`294.0`** | `294.0` | `0.0000` | **`0.00σ`** (Anchored) | **Strictly 0.0% (Excluded)** |
| `disk_usage_percent` | `54.60%` | `54.60%` | `54.15%` | `0.0544` | `+8.30σ` | Ranked #1 Contributor |
| `memory_used_mb` | `22,213 MB` | `22,213 MB` | `19,244 MB` | `597.36` | `+4.97σ` | Ranked #2 Contributor |
| `memory_available_mb` | `10,295 MB` | `10,295 MB` | `13,264 MB` | `597.36` | `-4.97σ` | Ranked #3 Contributor |
| `cpu_percent` | `21.40%` | `21.40%` | `16.20%` | `5.6841` | `+0.91σ` | Nominal Range |

---

## 4. Root Cause of Previous Stale Dashboard State

1. **Stale Process In-Memory Retention**: The FastAPI backend (`PID 607372`) and Streamlit frontend (`PID 573736`) had been executing continuously in memory since October 3, 2026 without restart.
2. **Pre-Fix Module Cache**: The running process held the older in-memory Python modules prior to the cumulative counter fix, accumulating a 22,822-second contiguous episode.
3. **Database Status Flag**: The host table in `data/fgead_multihost.db` held a historical `status = 'ANOMALY'` written on October 3.

### Corrective Actions Taken:
- Terminated all background Python processes (`uvicorn`, `streamlit`, `live_agent`).
- Cleared stale October 3 in-memory trackers and resolved historical alert records in SQLite database.
- Launched fresh FastAPI (`uvicorn api.main:app`) and Streamlit (`streamlit run app/streamlit_app.py`) servers.
- Verified that a freshly spawned process initializes with clean in-memory trackers (`episode_counter = 0`, `current_episode = None`).

---

## 5. Clean Live 120-Second Monitoring Results

- **Registered Host:** `host_sivachowdary`
- **Total Windows Evaluated:** 126
- **Physical `net_drops_total` Observed:** `0.0`
- **Model-Input `net_drops_total` Fed to Model:** `294.0`
- **Model Calibrated Threshold ($\tau$):** `1.859450`
- **`net_drops_total` in Top Features:** **`False` (0 / 126 occurrences)**
- **Top Attributed Features:**
  - `disk_usage_percent`: 126 / 126 windows (Driven by physical disk occupancy increase from baseline 54.15% $\to$ 54.60%)
  - `memory_used_mb`: 126 / 126 windows (Driven by active IDE and background runtime memory usage)
  - `memory_available_mb`: 82 / 126 windows

### Plain-English Explainability Output
- **Headline:** *FGEAD detected unusually high activity in percentage of disk storage capacity currently used (system activity) on this computer.*
- **Observation:** Real physical metric shift (disk usage from baseline 54.15% to 54.60%).
- **Unsupported Claims Removed:** No unsupported "99.9% confidence" or "check network adapter buffer capacity" statements exist.

---

## 6. Full Regression Test Suite Status (56 / 56 PASSED)

| Suite File | Description | Checks | Status |
| :--- | :--- | :---: | :---: |
| `data/multihost_validation/test_phase9_accuracy.py` | Accuracy, Residual Scoring, Schema Validation, Host Isolation | 18 / 18 | **PASS** |
| `data/multihost_validation/test_cloud_deployment_phase8.py` | Deployment Health, Logging, Configuration Probes | 4 / 4 | **PASS** |
| `data/multihost_validation/test_smoke_phase7.py` | End-to-End API Routing & Model Inference Smoke Checks | 7 / 7 | **PASS** |
| `data/multihost_validation/test_phase6_1_compatibility.py` | Model Compatibility, OS Gating, Lifecycle | 10 / 10 | **PASS** |
| `data/multihost_validation/test_multihost_suite.py` | Complete Multi-Host Registration, Auth, Token, Isolation | 17 / 17 | **PASS** |
| **Total Automated Validation** | **All Verification Suites** | **56 / 56** | **100% PASS** |

---

## 7. Model Invariants Maintained

- **No Retraining**: Checkpoints (`fgead_live_windows_22ch.pt`, `fgead_smd_machine_1_1.pt`) remain completely untouched.
- **Threshold Unchanged**: Baseline calibrated threshold $\tau = 1.859450$ remains exact.
- **22-Channel Schema Intact**: Feature order, normalization pipeline, and telemetry dimensions are strictly preserved.
