# Phase 10: Current-Machine Baseline Recalibration & Dedicated Live Model Report

**Document Version:** 1.0.0
**Date:** October 5, 2026
**Status:** COMPLETE & VERIFIED
**Host Assigned:** `host_sivachowdary` (`SivaChowdary`, Windows 11 Physical PC)
**Dedicated Model Profile:** `windows_sivachowdary_v2`

---

## 1. Executive Summary

Phase 10 addresses and resolves the persistent false-positive anomaly detection issue observed on the physical Windows workstation (`host_sivachowdary`). A scientific audit confirmed that the false positives were caused by distribution shifts across several system metrics (e.g. `disk_usage_percent`, `process_count`, `memory_available_mb`, `memory_used_mb`) between the historical synthetic benchmark dataset and the physical host's operating environment.

Rather than altering the threshold arbitrarily or disabling features, Phase 10 implemented a rigorous machine-learning recalibration lifecycle:
1. **Physical Baseline Collection:** Captured 3,600 continuous 1-second telemetry samples (`live_baseline_v2_current_machine.csv`) representing genuine nominal workstation behavior.
2. **Distribution Audit:** Validated statistical quality (mean, variance, coefficient of variation, percentiles) and mapped feature shifts against the benchmark.
3. **Dedicated Model Artifacts:** Preserved the original benchmark checkpoint and scaler untouched while creating a dedicated, versioned current-machine model (`windows_sivachowdary_v2`).
4. **Train-Only Standardization & EVT Calibration:** Fitted `StandardScaler` strictly on the training partition and calibrated threshold $\tau_{v2} = 1.411807$ via Extreme Value Theory (EVT) / Peak-Over-Threshold (POT).
5. **Validation Results:** Achieved **0.00% False Positive Rate (100.00% Specificity)** across 481 unseen validation/test windows while maintaining **100% sensitivity** across all 5 controlled anomaly workload classes.
6. **Regression Invariant:** 100% test pass rate (47/47 multihost & compatibility tests passing) and database isolation preserved.

---

## 2. Root-Cause Analysis of Original False-Positive Drift

Prior to Phase 10, the frozen benchmark live model (`fgead_live_windows_22ch.pt`) produced persistent false anomaly alerts (scores $\approx 10.68$, threshold $\tau = 1.8595$) during nominal operation.

The investigation revealed two distinct sources of distribution shift:
1. **Cumulative Counter Invariance:** Metrics such as `net_drops_total` had constant training values (e.g., 294 in benchmark vs. 0 on physical workstation). While the model-input anchor resolved the drop counter, other state features exhibited structural shift.
2. **System State Distribution Shifts:**
   - **`disk_usage_percent`:** Benchmark mean = 60.00% ($\sigma \approx 0.001$), while physical machine nominal = 54.98% ($\sigma \approx 0.001$). Because variance was near zero in the benchmark, a 55% reading produced an extreme z-score.
   - **`process_count`:** Benchmark mean = 185.0 ($\sigma = 0.50$), while physical workstation ran 380–385 background processes, yielding z-scores $> +300$.
   - **`memory_available_mb` / `memory_used_mb`:** Benchmark recorded $\approx 8,192$ MB total RAM, whereas the physical host has 32 GB RAM (24,800 MB used, 7,600 MB available).

---

## 3. Data Collection Methodology & Dataset Architecture

A continuous 3,600-sample (1-hour equivalent) telemetry stream was ingested directly from the local physical agent via `LiveTelemetryCollector`:

- **Storage Location:** [`data/live_baseline_v2_current_machine.csv`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/live_baseline_v2_current_machine.csv)
- **Sample Count:** 3,600 records
- **Sampling Interval:** 1.0 second nominal interval
- **Feature Vector:** Exactly 22 features, matching the canonical `LIVE_FEATURES` schema:
  - CPU: `cpu_percent`, `cpu_freq_current`, `cpu_user_time_percent`, `cpu_system_time_percent`, `cpu_ctx_switches_per_sec`, `cpu_interrupts_per_sec`
  - Memory: `memory_percent`, `memory_available_mb`, `memory_used_mb`, `swap_percent`
  - Storage: `disk_usage_percent`, `disk_read_bytes_per_sec`, `disk_write_bytes_per_sec`, `disk_read_count_per_sec`, `disk_write_count_per_sec`
  - Network & System: `net_bytes_sent_per_sec`, `net_bytes_recv_per_sec`, `net_packets_sent_per_sec`, `net_packets_recv_per_sec`, `net_errors_total`, `net_drops_total`, `process_count`

---

## 4. Baseline Quality Audit & Distribution Profiling

Statistical profiling verified healthy dynamic variance across active physical metrics:

| Metric | Mean | Median | Std Dev | Min | Max | CV | p95 | p99 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_percent` | 51.10% | 50.84% | 7.62% | 22.49% | 79.40% | 0.149 | 64.40% | 69.51% |
| `cpu_freq_current` | 1968.20 MHz | 1967.48 MHz | 50.33 MHz | 1793.74 MHz | 2158.03 MHz | 0.026 | 2051.20 MHz | 2088.03 MHz |
| `cpu_ctx_switches_per_sec` | 13,382.7 | 13,323.2 | 1,607.7 | 9,139.7 | 19,850.5 | 0.120 | 16,192.4 | 17,732.1 |
| `memory_percent` | 76.54% | 76.50% | 0.44% | 75.30% | 77.40% | 0.006 | 77.20% | 77.30% |
| `memory_used_mb` | 24,904.7 MB | 24,891.8 MB | 143.1 MB | 24,500.4 MB | 25,183.0 MB | 0.006 | 25,118.8 MB | 25,150.0 MB |
| `process_count` | 382.4 | 382.0 | 2.1 | 377.0 | 388.0 | 0.005 | 386.0 | 387.0 |
| `net_bytes_recv_per_sec` | 13,674.3 B/s | 6,564.0 B/s | 26,052.2 B/s | 468.8 B/s | 465,584.9 B/s | 1.905 | 45,951.6 B/s | 132,607.7 B/s |
| `disk_write_bytes_per_sec` | 74,402.1 B/s | 23,286.0 B/s | 197,354.7 B/s | 0.0 B/s | 3,842,500.0 B/s | 2.653 | 312,410.0 B/s | 892,100.0 B/s |

---

## 5. Side-by-Side Comparison Matrix

| Feature Name | Benchmark Baseline Mean | Current Host Baseline Mean | Shift Nature |
| :--- | :--- | :--- | :--- |
| `cpu_percent` | 16.20% | 51.10% | Workstation active load |
| `cpu_freq_current` | 2,523.64 MHz | 1,968.20 MHz | Hardware P-state governor |
| `memory_available_mb` | 3,276.80 MB | 7,632.70 MB | 32 GB RAM architecture |
| `memory_used_mb` | 4,915.20 MB | 24,904.70 MB | Workstation working set |
| `disk_usage_percent` | 60.00% | 54.98% | Local filesystem capacity |
| `process_count` | 185.0 | 382.4 | Windows 11 services & apps |
| `net_drops_total` | 294.0 | 0.07 | Host cumulative counter |

---

## 6. Model Preservation & Isolation Invariant

To ensure benchmark reproducibility and multi-host model profile integrity, all legacy artifacts remain completely untouched:

- **Original Benchmark Checkpoint:** [`checkpoints/fgead_live_windows_22ch.pt`](file:///c:/Users/saikr/Desktop/FGEAD-main/checkpoints/fgead_live_windows_22ch.pt) (Preserved unmodified)
- **Original Benchmark Scaler:** [`checkpoints/fgead_live_scaler.joblib`](file:///c:/Users/saikr/Desktop/FGEAD-main/checkpoints/fgead_live_scaler.joblib) (Preserved unmodified)
- **Original Benchmark Threshold:** [`checkpoints/fgead_live_threshold.json`](file:///c:/Users/saikr/Desktop/FGEAD-main/checkpoints/fgead_live_threshold.json) ($\tau = 1.859450$, unmodified)

---

## 7. Dedicated Model Training Methodology

The dedicated current-machine model was trained using [`models/train_current_machine_v2.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/models/train_current_machine_v2.py):

- **Data Partitioning:** Chronological sequential split without shuffle:
  - Train: 70% (2,520 samples)
  - Validation: 15% (540 samples)
  - Test: 15% (540 samples)
- **Normalization:** `StandardScaler` fitted strictly on the 2,520 training samples.
- **Model Architecture:** FGEAD 22-Channel Multi-Head Graph Attention & Bi-LSTM:
  - Input Features: 22
  - Embedding Dimension: 64
  - Multi-Head Attention: 4 heads
  - GCN Channels: 64
  - LSTM Hidden Dimension: 128
  - Sparsity Threshold: 0.3
  - Window Size: $W = 60$, Stride $S = 1$
- **Training Epochs:** 30 epochs with Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-5}$).
- **Best Validation Loss:** $0.887588$ MSE.

---

## 8. Dynamic Calibration & EVT/Threshold Formulation

The anomaly threshold was calculated using EVT Peak-Over-Threshold (POT) / 3-sigma calibration over validation window anomaly scores:
- **Validation Score Mean:** $\mu_{\text{val}} = 1.061390$
- **Validation Score Std:** $\sigma_{\text{val}} = 0.081017$
- **Validation Score p99:** $1.227659$
- **Validation Score Max:** $1.255184$
- **Calibrated Threshold:**
  $$\tau_{v2} = \mu_{\text{val}} + 4.325 \times \sigma_{\text{val}} = 1.411807$$

---

## 9. Controlled Anomaly Detection Verification

Five distinct synthetic workload stress windows were passed to `windows_sivachowdary_v2` to verify detection sensitivity:

| Workload Scenario | Injected Anomaly Pattern | Peak Anomaly Score | Threshold ($\tau$) | Detection Status | Top Root-Cause Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. CPU-Intensive Workload** | 98.5% CPU User load jump | **2.7071** | 1.4118 | **DETECTED (CRITICAL)** | `cpu_user_time_percent` |
| **B. Disk-Write Burst** | 120 MB/s write @ 2,500 IOPS | **12.1083** | 1.4118 | **DETECTED (CRITICAL)** | `disk_write_count_per_sec` |
| **C. Disk-Read Burst** | 250 MB/s read @ 4,000 IOPS | **176.1944** | 1.4118 | **DETECTED (CRITICAL)** | `disk_read_bytes_per_sec` |
| **D. Network Burst** | 50 MB/s egress burst | **6.0689** | 1.4118 | **DETECTED (CRITICAL)** | `net_bytes_sent_per_sec` |
| **E. Combined CPU + Storage** | CPU saturation + Write spike | **5.6932** | 1.4118 | **DETECTED (CRITICAL)** | `disk_write_bytes_per_sec` |

---

## 10. Unseen Normal Test Split Specificity Verification

Evaluating 481 continuous sliding windows across the unseen test split ($N = 540$):
- **Total Windows Evaluated:** 481
- **Windows Exceeding Threshold:** 0
- **False Positive Rate (FPR):** **0.00%**
- **Model Specificity:** **100.00%**
- **Mean Normal Test Score:** $1.0294 \pm 0.071$
- **99th Percentile Score:** $1.1814 < \tau_{v2} (1.4118)$

---

## 11. Architecture & MultiHost Model Profile Integration

`api/multihost_inference.py` registers the versioned profile and routes requests dynamically:
- Profile ID: `windows_sivachowdary_v2`
- Name: `FGEAD Windows 22-Channel Live Model (v2 Current-Machine)`
- Checkpoint: `checkpoints/fgead_live_windows_22ch_v2_current_machine.pt`
- Scaler: `checkpoints/fgead_live_scaler_v2_current_machine.joblib`
- Config: `checkpoints/fgead_live_windows_22ch_config_v2_current_machine.json`
- Threshold: $\tau = 1.411807$

---

## 12. Operational DB State & Fleet Integrity

Operational database inspection (`data/fgead_multihost.db`):
- `host_sivachowdary`: Assigned `model_id = 'windows_sivachowdary_v2'`, `model_status = 'COMPATIBLE'`.
- `host_linux_srv01`: Assigned `model_id = 'none'`, `model_status = 'BASELINE_REQUIRED'`.
- Test DB isolation invariant verified: No ephemeral test records in operational DB.

---

## 13. Verification Sign-Off

- [x] Baseline collected from physical host (`live_baseline_v2_current_machine.csv`)
- [x] Benchmark artifacts preserved unmodified
- [x] Dedicated model trained and fitted on train partition only
- [x] Calibrated threshold $\tau_{v2} = 1.411807$
- [x] 0.00% False Positive Rate on normal baseline
- [x] 100% Detection Sensitivity on controlled anomalies
- [x] 47/47 multihost unit and integration tests passing
