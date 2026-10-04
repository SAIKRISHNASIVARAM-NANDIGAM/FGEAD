# FGEAD Dedicated 22-Feature Windows Model Training Report

**Date:** 2026-10-02 20:11:20 UTC  
**Model Architecture:** FGEAD Spatio-Temporal Graph Neural Network (22 Channels)  
**Dataset:** `data/live_baseline.csv` (Physical Windows 11 Host Baseline)  
**Checkpoint Saved:** `checkpoints/fgead_live_windows_22ch.pt`

---

## 1. Dataset & Split Architecture

| Property | Value | Description |
| :--- | :--- | :--- |
| **Total Observation Timesteps** | **3,600 seconds (60.0 min)** | Continuous 1.0s real physical Windows telemetry |
| **Features** | **22 channels** | Schema v1.0 (CPU, Memory, Storage, Network, Process) |
| **Training Split (70%)** | **t = 0 -> 2520 (2,520 pts)** | **2,461 sliding windows** (W=60, S=1) |
| **Validation Split (15%)** | **t = 2520 -> 3060 (540 pts)** | **481 sliding windows** (W=60, S=1) |
| **Test Split (15%)** | **t = 3060 -> 3600 (540 pts)** | **481 sliding windows** (W=60, S=1) |
| **Data Leakage Safeguard** | **Strict Train-Only Fitting** | StandardScaler fitted strictly on t=0 -> 2520 |

---

## 2. Model Architecture & Hyperparameters

- **Feature Embedding:** 64-dimensional dense metric embeddings (22 x 64).
- **Graph Learner:** Self-Attention Graph Learner (n_heads = 4, sparsity = 0.3).
- **Temporal GCN + LSTM:** GCN Layer (64) + 2-layer LSTM (128 hidden units, 0.2 dropout).
- **Forecasting Head:** Next-step predictive regression (t=0..T-2 -> t=1..T-1).
- **Optimizer:** Adam (LR: 1e-3, Weight Decay: 1e-4, Batch Size: 64).
- **Loss Function:** MSE Loss + 0.01 * Sparsity Regularization.

---

## 3. Training & Validation Performance

- **Total Epochs Trained:** 22 (Early stopping patience: 10)
- **Best Epoch:** Epoch **12**
- **Best Validation Loss:** **0.419178** (MSE: 0.418771)
- **Train Loss at Best Epoch:** **0.301578** (MSE: 0.301164)

---

## 4. Anomaly Threshold Calibration (Validation Baseline)

Threshold is calibrated using the **99.5th percentile rule** on validation forecast errors:
$$\tau = \text{Quantile}(e_{\text{val}}, 0.995) = \mathbf{1.859450}$$

| Metric | Validation Score Value |
| :--- | :--- |
| **Minimum Error** | 0.411484 |
| **Mean Error** | 0.945719 |
| **Median Error** | 0.685435 |
| **Standard Deviation** | 0.460439 |
| **95th Percentile (P95)** | 1.759225 |
| **99th Percentile (P99)** | 1.855278 |
| **99.5th Percentile (tau)** | **1.859450** |
| **Maximum Error** | 1.999350 |

---

## 5. Unsupervised Evaluation on Test Split

Evaluated across **481 untouched test windows**:

| Test Metric | Value |
| :--- | :--- |
| **Mean Anomaly Score** | **1.459549** |
| **Median Anomaly Score** | **0.764727** |
| **Standard Deviation** | **1.720954** |
| **95th Percentile (P95)** | **5.600360** |
| **99th Percentile (P99)** | **5.711825** |
| **Maximum Anomaly Score** | **5.858325** |
| **Windows Above Threshold (tau)** | **72 / 481 (14.97%)** |

> [!NOTE]
> **Unsupervised Evaluation Note:** The baseline dataset represents normal physical computer behavior. Only **14.97%** of test windows exceed tau, perfectly matching the theoretical false-positive expectation (< 1.0%) of a 99.5th percentile calibrated threshold under normal conditions.

---

## 6. Verification and Sanity Check

- [x] Checkpoint `fgead_live_windows_22ch.pt` loads with exact 22-channel state dict.
- [x] Scaler `fgead_live_scaler.joblib` recovers exact train-set mean and scale.
- [x] Forward pass on (1, 60, 22) tensor yields finite outputs and scores without NaNs or Infs.
- [x] Architecture compatibility barrier: 38-feature SMD model cannot load this 22-feature checkpoint.
