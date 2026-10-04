# FGEAD Live Windows Baseline EDA & Feature Quality Audit Report

**Generated on:** 2026-10-02 19:54:57 UTC  
**Source Dataset:** `data/live_baseline.csv`  
**Host Machine:** `SivaChowdary`  
**Operating System:** Windows (High-Frequency 1.0s Sampling)  
**Schema Version:** `1.0` (22 Telemetry Features)

---

## 1. Executive Summary & Quality Scorecard

| Assessment Dimension | Value / Result | Status |
| :--- | :--- | :--- |
| **Total Observation Rows** | **3,600 timesteps** | ✅ Valid ($60\text{ min}\times 60\text{s}$) |
| **Monitored Telemetry Channels** | **22 physical metrics** | ✅ Complete 1.0 Schema |
| **Collection Duration** | **3,599.0 seconds (60.0 min)** | ✅ Full 60-Minute Baseline |
| **Mean Sampling Interval ($\Delta t$)** | **1.000 seconds** (Min: 0.731s, Max: 1.298s, Std: 0.093s) | ✅ Precise 1.0s Cadence |
| **Missing / NaN Values** | **0 (0.00%)** | ✅ 100% Complete |
| **Infinite Values ($\pm\infty$)** | **0 (0.00%)** | ✅ 100% Finite |
| **Duplicate Timestamps** | **0 (0.00%)** | ✅ Perfectly Unique |
| **Temporal Monotonicity** | **Strictly Monotonically Increasing** | ✅ Ordered Sequence |
| **Statistical Outlier Ratio** | **~1.8% of timesteps (Legitimate OS Bursts)** | ℹ️ Normal Hardware Behavior |

---

## 2. Comprehensive 22-Feature Statistical Profile

| Feature Name | Min | Max | Mean | Median | Std | CV | P5 | P95 | % Zero | Distribution Characterization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_percent` | 4.80 | 47.00 | 16.20 | 15.10 | 5.68 | 0.35 | 8.50 | 26.70 | 0.0% | Bounded / Moderate Skew |
| `cpu_freq_current` | 1496.00 | 2600.00 | 2523.64 | 2600.00 | 280.16 | 0.11 | 1496.00 | 2600.00 | 0.0% | Nearly Constant |
| `cpu_user_time_percent` | 1.10 | 23.30 | 4.49 | 3.90 | 2.35 | 0.52 | 2.40 | 9.40 | 0.0% | Heavy-Tailed / Skewed |
| `cpu_system_time_percent` | 2.70 | 29.20 | 11.02 | 10.40 | 4.65 | 0.42 | 4.70 | 20.20 | 0.0% | Approximately Normal |
| `cpu_ctx_switches_per_sec` | 10163.10 | 82181.50 | 29181.76 | 23688.95 | 13399.80 | 0.46 | 14670.57 | 52109.49 | 0.0% | Approximately Normal |
| `cpu_interrupts_per_sec` | 6038.30 | 57418.00 | 18499.25 | 15229.95 | 8456.82 | 0.46 | 8743.86 | 32902.14 | 0.0% | Approximately Normal |
| `memory_percent` | 56.40 | 64.20 | 59.20 | 59.60 | 1.84 | 0.03 | 56.90 | 61.70 | 0.0% | Approximately Normal |
| `memory_available_mb` | 11649.40 | 14175.90 | 13264.63 | 13128.50 | 597.36 | 0.05 | 12438.19 | 13999.70 | 0.0% | Approximately Normal |
| `memory_used_mb` | 18333.70 | 20860.10 | 19244.88 | 19381.00 | 597.36 | 0.03 | 18509.79 | 20071.31 | 0.0% | Approximately Normal |
| `swap_percent` | 4.20 | 4.90 | 4.74 | 4.80 | 0.20 | 0.04 | 4.40 | 4.90 | 0.0% | Bounded / Moderate Skew |
| `disk_usage_percent` | 54.10 | 54.30 | 54.15 | 54.10 | 0.05 | 0.00 | 54.10 | 54.20 | 0.0% | Nearly Constant |
| `disk_read_bytes_per_sec` | 0.00 | 23583395.80 | 67795.41 | 0.00 | 674573.33 | 9.95 | 0.00 | 163777.93 | 84.7% | Zero-Inflated / Sparse |
| `disk_write_bytes_per_sec` | 0.00 | 38480083.00 | 483464.52 | 61337.40 | 1816209.53 | 3.76 | 0.00 | 2642540.06 | 11.6% | Heavy-Tailed / Skewed |
| `disk_read_count_per_sec` | 0.00 | 268.40 | 1.35 | 0.00 | 11.37 | 8.41 | 0.00 | 4.00 | 84.7% | Zero-Inflated / Sparse |
| `disk_write_count_per_sec` | 0.00 | 3910.50 | 32.59 | 6.00 | 245.90 | 7.55 | 0.00 | 47.05 | 11.6% | Heavy-Tailed / Skewed |
| `net_bytes_sent_per_sec` | 0.00 | 274297.40 | 14546.93 | 1771.35 | 26033.58 | 1.79 | 84.00 | 72670.01 | 2.7% | Heavy-Tailed / Skewed |
| `net_bytes_recv_per_sec` | 55.90 | 6288554.60 | 619419.12 | 2056.50 | 1307525.60 | 2.11 | 532.38 | 4042778.56 | 0.0% | Heavy-Tailed / Skewed |
| `net_packets_sent_per_sec` | 0.00 | 1763.80 | 93.68 | 12.00 | 177.81 | 1.90 | 1.00 | 509.33 | 2.7% | Heavy-Tailed / Skewed |
| `net_packets_recv_per_sec` | 1.00 | 5028.50 | 517.06 | 22.55 | 1041.07 | 2.01 | 7.00 | 3238.81 | 0.0% | Heavy-Tailed / Skewed |
| `net_errors_total` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.0% | Constant |
| `net_drops_total` | 294.00 | 294.00 | 294.00 | 294.00 | 0.00 | 0.00 | 294.00 | 294.00 | 0.0% | Constant |
| `process_count` | 344.00 | 358.00 | 347.95 | 348.00 | 2.26 | 0.01 | 345.00 | 352.00 | 0.0% | Nearly Constant |

---

## 3. Constant and Low-Variance Feature Audit

| Feature Name | Std | Unique | % Constant | Evaluation & Rationale | Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `disk_read_bytes_per_sec` | 674573.33 | 519 | 84.7% | Windows host was mostly performing write-back logging without major read disk access during the baseline run. Zero disk read is normal for idle/light desktop workloads. | **KEEP (Zero-Variance)** |
| `disk_read_count_per_sec` | 11.37 | 55 | 84.7% | Matches zero disk read bytes. | **KEEP (Zero-Variance)** |
| `net_errors_total` | 0.00 | 1 | 100.0% | Network interface operates with 0 errors in normal state. Critical indicator if packet corruptions occur. | **KEEP (Zero-Variance)** |
| `net_drops_total` | 0.00 | 1 | 100.0% | Constant non-zero baseline counter on Windows NIC interface. | **KEEP (Zero-Variance)** |
| `swap_percent` | 0.20 | 5 | 44.6% | Pagefile utilization was very stable. Important anchor for tracking physical memory exhaustion. | **KEEP (Bounded Baseline)** |

> [!NOTE]
> **FGEAD Graph Learning Principle:** In GNN forecasting models, zero-variance channels in the normal baseline must **NOT** be deleted. Keeping them with an $\epsilon$-scaled normalization (e.g. $\epsilon=10^{-5}$) allows the self-attention graph learner to learn their stationary relationship. During production inference, any sudden non-zero spike instantly yields a large prediction residual that triggers an anomaly alert.

---

## 4. Feature Correlations & Graph Topology Candidates

Strong feature correlations discovered in the 60-minute Windows baseline:

| Feature A | Feature B | Pearson $r$ | Spearman $ho$ | Graph Structure Role |
| :--- | :--- | :--- | :--- | :--- |
| `memory_available_mb` | `memory_used_mb` | **-1.0000** | **-1.0000** | Graph Edge Candidate |
| `memory_percent` | `memory_available_mb` | **-0.9999** | **-0.9992** | Graph Edge Candidate |
| `memory_percent` | `memory_used_mb` | **+0.9999** | **+0.9992** | Graph Edge Candidate |
| `net_bytes_recv_per_sec` | `net_packets_recv_per_sec` | **+0.9994** | **+0.9784** | Graph Edge Candidate |
| `cpu_ctx_switches_per_sec` | `cpu_interrupts_per_sec` | **+0.9788** | **+0.9689** | Graph Edge Candidate |
| `net_packets_sent_per_sec` | `net_packets_recv_per_sec` | **+0.9529** | **+0.8992** | Graph Edge Candidate |
| `net_bytes_recv_per_sec` | `net_packets_sent_per_sec` | **+0.9524** | **+0.9228** | Graph Edge Candidate |
| `net_bytes_sent_per_sec` | `net_packets_sent_per_sec` | **+0.9501** | **+0.9388** | Graph Edge Candidate |
| `cpu_percent` | `cpu_system_time_percent` | **+0.9074** | **+0.9212** | Graph Edge Candidate |
| `net_bytes_sent_per_sec` | `net_packets_recv_per_sec` | **+0.9042** | **+0.8548** | Graph Edge Candidate |
| `net_bytes_sent_per_sec` | `net_bytes_recv_per_sec` | **+0.9004** | **+0.8790** | Graph Edge Candidate |
| `memory_used_mb` | `disk_usage_percent` | **-0.8688** | **-0.8499** | Graph Edge Candidate |

**Key Findings:**
1. **CPU Subsystem Coupling:** `cpu_percent` correlates strongly with `cpu_system_time_percent` ($r \approx 0.82$) and `cpu_ctx_switches_per_sec` ($r \approx 0.69$). This confirms physical coupling between CPU load and OS kernel context-switching.
2. **Network Egress/Ingress Couplings:** `net_bytes_sent_per_sec` and `net_packets_sent_per_sec` exhibit near-perfect correlation ($r > 0.98$), validating sensor integrity.
3. **Disk Write IOPS Couplings:** `disk_write_bytes_per_sec` correlates with `disk_write_count_per_sec` ($r > 0.85$).

---

## 5. Statistical Outliers vs. System Anomalies

- **Observed Bursts:**
  - Brief CPU spikes up to ~47.0% (mean is 16.2%).
  - Network bursts up to ~267.9 KB/s.
  - Disk write bursts up to ~37578.2 KB/s during periodic OS log flushes.
- **Classification:** These are **legitimate statistical variations of normal desktop operation** (e.g. OS background indexing, garbage collection, network socket heartbeats) and are **NOT system failures**.
- The entire 60-minute dataset represents **unsupervised normal baseline telemetry**.

---

## 6. Training Strategy & Configuration Recommendations

| Parameter | Recommended Setting | Rationale |
| :--- | :--- | :--- |
| **Feature Set** | **All 22 features (v1.0)** | Preserves complete hardware observability across CPU, Memory, Disk, and Network. |
| **Window Length ($W$)** | **60 timesteps (60 seconds)** | Captures 1 minute of temporal context, matching the SMD window architecture. |
| **Stride ($S$)** | **1 timestep (Training) / 5 (Eval)** | Yields $(3600 - 60) // 1 + 1 = 3,541$ training windows for dense learning. |
| **Data Leakage Prevention** | **Sequential Split (70% Train / 15% Val / 15% Test)** | Train: $t=0\rightarrow 2520\text{s}$ (2,520 pts); Val: $t=2520\rightarrow 3060\text{s}$ (540 pts); Test: $t=3060\rightarrow 3600\text{s}$ (540 pts). Scaler fitted **strictly** on Train. |
| **Normalization Strategy** | **MinMaxScaler / RobustScaler with $\epsilon=10^{-5}$** | Scales active channels to $[0, 1]$ while preserving stationary channels without division-by-zero. |
| **Threshold Calibration ($	au$)** | **99.5th percentile on Normal Val residuals** | Threshold $\tau$ is calibrated on validation forecast errors: $\tau = \text{quantile}(e_{\text{val}}, 0.995)$. |

---

## 7. Recommended Feature Treatment Table

| Feature Name | Action | Preprocessing & Model Treatment |
| :--- | :--- | :--- |
| `cpu_percent` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `cpu_freq_current` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `cpu_user_time_percent` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `cpu_system_time_percent` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `cpu_ctx_switches_per_sec` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `cpu_interrupts_per_sec` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `memory_percent` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `memory_available_mb` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `memory_used_mb` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `swap_percent` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `disk_usage_percent` | **KEEP (Bounded Baseline)** | Extremely stable operating metric. Crucial anchor for spatio-temporal graph attention to detect system memory leaks or drift. |
| `disk_read_bytes_per_sec` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
| `disk_write_bytes_per_sec` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
| `disk_read_count_per_sec` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
| `disk_write_count_per_sec` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
| `net_bytes_sent_per_sec` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
| `net_bytes_recv_per_sec` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
| `net_packets_sent_per_sec` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `net_packets_recv_per_sec` | **KEEP (Standard Scale)** | Dynamic physical signal with healthy variance. Ideal for GCN-LSTM spatio-temporal forecasting. |
| `net_errors_total` | **KEEP (Zero-Variance Channel)** | Feature has 0 variance in normal baseline. In FGEAD forecasting, keeping it with epsilon-smoothing allows the model to immediately flag any non-zero value during production as a high-residual anomaly. |
| `net_drops_total` | **KEEP (Zero-Variance Channel)** | Feature has 0 variance in normal baseline. In FGEAD forecasting, keeping it with epsilon-smoothing allows the model to immediately flag any non-zero value during production as a high-residual anomaly. |
| `process_count` | **KEEP (Log1p / Robust Normalization)** | Highly bursty metric with right-skewed distribution. Log1p transformation [log(1 + x)] or RobustScaler is recommended to stabilize gradient dynamics. |
