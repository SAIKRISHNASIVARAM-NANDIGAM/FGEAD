# FGEAD LIVE MODEL V3 — MODEL TRAINING & EVALUATION REPORT

## Executive Summary

This report documents the training, threshold calibration, unseen normal held-out evaluation, and physical controlled anomaly testing for the **FGEAD Live Model V3** (`windows_sivachowdary_v3`). 

The V3 rebuild was undertaken to eliminate false-positive memory distribution anomaly alerts (scores ranging from $3.5779$ to $7.9249$) observed under V2 during normal laptop operation. The root cause of V2 false positives was identified as a distribution mismatch in the baseline dataset—V2 was calibrated against an artificially high snapshot of RAM usage (~74.6%, ~24.2 GB used) with narrow variance ($\sigma \approx 407\text{ MB}$), causing normal operational RAM levels (~16.7 GB used, ~51.5%) to register as an $18\sigma+$ normalized excursion.

V3 has been trained strictly on a fresh, continuous, 1,800-sample (30-minute) non-synthetic telemetry baseline collected on `host_sivachowdary` with zero sleep interruptions and zero missing data.

---

## 1. Baseline Dataset & Chronological Splits

| Dataset Split | Rows | Purpose |
| :--- | :---: | :--- |
| **Train Normal** (`v3_train_normal.csv`) | 1,260 | Model training & `StandardScaler` fitting |
| **Calibration Normal** (`v3_cal_normal.csv`) | 270 | Non-parametric threshold calibration |
| **Held-Out Test Normal** (`v3_test_normal.csv`) | 270 | Unseen false positive rate (FPR) & specificity evaluation |
| **Total Continuous Baseline** | **1,800** | Collected across 30.0 minutes (0.9998s avg interval) |

---

## 2. Model Architecture & Training Details

- **Model Identifier:** `windows_sivachowdary_v3`
- **Architecture:** 22-Channel Spatial-Temporal Graph Attention Neural Network (FGEAD)
  - Feature Embedding (dim=64)
  - Multi-Head Self-Attention Graph Learner (heads=4, sparsity threshold=0.3)
  - Dynamic Feature Graph + Temporal Graph Convolutional Network (GCN, out=64)
  - LSTM Forecasting (hidden=128)
  - Combined Reconstruction & 1-step Ahead Forecasting Residual Anomaly Scoring
- **Loss Function:** Joint MSE Forecasting/Reconstruction Loss + Sparsity Regularization ($\lambda = 0.01$)
- **Optimization:** Adam Optimizer ($lr = 1\times 10^{-3}$, weight decay = $1\times 10^{-5}$)
- **Epochs Trained:** 20
- **Final Training Loss:** 0.1395
- **Scaler:** `StandardScaler` fitted **strictly** on Train Normal split (never exposed to calibration, test, or anomaly data).

---

## 3. Threshold Calibration

- **Calibration Window Dataset:** 1,411 sliding windows ($W=60, S=1$) from Train + Calibration Normal splits.
- **Normal Score Statistics:**
  - Mean Score: `0.354122`
  - Median Score: `0.334051`
  - Std Deviation: `0.110294`
  - P95 Score: `0.548102`
  - P99.5 Score: `2.056914`
  - Max Score: `2.109618`
- **Calibrated Anomaly Threshold ($\tau$):** **`2.120169`** (Set to P99.5 + safety margin, strictly bounding normal variance).

---

## 4. Phase 6: Unseen Normal Validation (Held-Out Test Set)

Evaluated on 211 sliding windows from the chronological held-out normal test set (`v3_test_normal.csv`).

| Metric | Target / Gate | V3 Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **False Positive Rate (FPR)** | $\le 1.0\%$ | **`0.00%`** (0 / 211 windows) | 🟢 **PASS** |
| **Specificity** | $\ge 99.0\%$ | **`100.00%`** | 🟢 **PASS** |
| **Mean Anomaly Score** | — | `0.317394` | Nominal |
| **P95 Anomaly Score** | — | `0.410547` | Nominal |
| **P99 Anomaly Score** | — | `0.456636` | Nominal |
| **Max Anomaly Score** | $< \tau$ (`2.120169`) | `0.547307` | Nominal |

---

## 5. Phases 7, 8, & 9: Controlled Anomaly & Recovery Testing

Executed 5 physical controlled anomaly scenarios against the live model using actual system workload generators.

| Test Scenario | Detected? | Detection Delay | Peak Anomaly Score | Threshold $\tau$ | Recovery Time | Top Contributing Feature Candidate |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Test [A]: CPU-Intensive Workload** | 🟢 YES | **2.0s** | **3.0671** | 2.1202 | **0.0s** | `cpu_ctx_switches_per_sec` |
| **Test [B]: Disk-Write Burst** | 🟢 YES | **2.0s** | **5.9854** | 2.1202 | **0.0s** | `disk_write_bytes_per_sec` |
| **Test [C]: Disk-Read Burst** | 🔴 NO | -1.0s | 1.7776 | 2.1202 | 0.0s | `disk_read_count_per_sec` |
| **Test [D]: Network Ingress/Egress Burst** | 🟢 YES | **2.0s** | **99.6019** | 2.1202 | **0.0s** | `net_bytes_recv_per_sec` |
| **Test [E]: Combined CPU + Storage** | 🟢 YES | **2.0s** | **5.4778** | 2.1202 | **0.0s** | `disk_write_bytes_per_sec` |

### Confusion Matrix & Metrics Summary

```
======================================================================
V3 REAL LIVE CONFUSION MATRIX & ACCURACY SUMMARY
======================================================================
True Positives  (TP):   116  | False Positives (FP):     0
False Negatives (FN):    34  | True Negatives  (TN):   211
----------------------------------------------------------------------
Accuracy           : 90.76%
Precision          : 100.00%
Recall (Detection) : 77.33%
F1-Score           : 0.8722
Specificity        : 100.00%
FPR                : 0.00%
Average Detection Delay : 2.0 seconds (Target <= 5.0s)
Average Recovery Time   : 0.0 seconds
======================================================================
```

*Note: Test [C] (Disk-Read Burst) peak score reached 1.7776 (elevated above baseline mean 0.31), but did not breach $\tau = 2.1202$. High disk-write, CPU, and network bursts breached threshold immediately with 2.0s detection delay.*

---

## 6. Phase 10: V2 Artifact Preservation Verification

To ensure strict zero-regression standards, the SHA256 cryptographic digests of all legacy V2 artifacts were verified before and after V3 training:

| V2 Artifact File | Expected SHA256 Hash | Observed SHA256 Hash | Status |
| :--- | :--- | :--- | :---: |
| `checkpoints/fgead_live_windows_22ch.pt` | `2a1ba33ddee089af1c7356dc446a5712eee14806a264c10c62b66a20936c45ff` | `2a1ba33ddee089af1c7356dc446a5712eee14806a264c10c62b66a20936c45ff` | 🟢 **MATCH** |
| `checkpoints/fgead_live_scaler_v2_current_machine.joblib` | `440903ed188db3e5032b7c685a5678fd95408111a2476d4563125a06b1557dff` | `440903ed188db3e5032b7c685a5678fd95408111a2476d4563125a06b1557dff` | 🟢 **MATCH** |
| `checkpoints/fgead_live_threshold_v2_current_machine.json` | `1a456c37293bda66d4d0b6751899ecbbb80f5600bd32eb6ea0c6ff01f8ff2c28` | `1a456c37293bda66d4d0b6751899ecbbb80f5600bd32eb6ea0c6ff01f8ff2c28` | 🟢 **MATCH** |

---

## 7. Artifact Manifest

The complete set of V3 production artifacts is registered and ready for deployment:

- **PyTorch Model Checkpoint:** `checkpoints/fgead_live_windows_22ch_v3_current_machine.pt`
- **Scaler Artifact:** `checkpoints/fgead_live_scaler_v3_current_machine.joblib`
- **Threshold Config:** `checkpoints/fgead_live_threshold_v3_current_machine.json`
- **Model Config:** `checkpoints/fgead_live_windows_22ch_config_v3_current_machine.json`
- **Baseline Data:** `data/live_baseline_v3_current_machine.csv`
- **Chronological Train Split:** `data/live_training/v3_train_normal.csv`
- **Chronological Cal Split:** `data/live_training/v3_cal_normal.csv`
- **Chronological Test Split:** `data/live_training/v3_test_normal.csv`
- **Quality Audit Reports:** 
  - `data/live_training/V3_COLLECTION_FINAL_REPORT.md`
  - `reports/V3_LIVE_MODEL_FINAL_REPORT.md`
  - `reports/V3_LIVE_MODEL_METRICS.json`
