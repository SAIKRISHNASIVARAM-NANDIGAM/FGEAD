# FGEAD V3 Forensic Audit Report
**Date:** October 7, 2026  
**Host Environment:** Windows Physical Host (SivaChowdary Laptop)  
**Target Model Version:** V3 Current-Machine Production Rebuild  

---

## 1. Executive Summary & Root Cause Findings

The Windows Live Model V2 (`windows_sivachowdary_v2`, threshold $\tau = 1.411807$) experienced severe false-positive anomaly scores ($3.5779$ to $7.9249$) during normal laptop operation.

### Identified Root Causes:
1. **Synthetic Telemetry Snapshot Generator in V2 Baseline Collection:**  
   The script `data/collect_current_machine_baseline.py` collected only **50 live samples over 2.5 seconds** and then generated 3,600 synthetic AR(1) Gaussian samples centered around that transient 2.5-second mean.
2. **RAM & CPU Baseline Distribution Mismatch:**  
   When the 50-sample snapshot was taken for V2, the laptop RAM utilization was ~24.2 GB (74.6% used). In subsequent normal operation, RAM usage dropped to ~16.7 GB (51.5% used). Because the V2 `StandardScaler` was fitted with a narrow synthetic standard deviation ($\sigma \approx 407\text{ MB}$), a 7.5 GB shift in RAM produced a normalized deviation exceeding **18 standard deviations** ($z > 18.0$), dominating forecasting residuals and triggering false anomalies.
3. **Telemetry Pipeline Integrity:**  
   The underlying 22-channel hardware collection semantics in `data/live_agent.py` and `data/live_feature_schema.py` are sound and operate properly. The issue is strictly due to baseline distribution miscalibration caused by synthetic generation instead of real continuous live telemetry collection.

---

## 2. Telemetry & Model Pipeline Architectural Inspection

| Component | Target File | Verification Findings |
| :--- | :--- | :--- |
| **Feature Schema** | `data/live_feature_schema.py` | Exactly 22 features, fixed ordering (`cpu_percent` to `process_count`). Version 1.0. Units, formatting, and dict validation functions verified. |
| **Telemetry Agent** | `data/live_agent.py` | `LiveTelemetryCollector` uses `psutil` with stateful time delta (`dt = max(now - last_time, 0.001)`). Sampling interval: 1.0s. Non-blocking CPU per-second delta. Cumulative counter rates computed accurately for I/O and network. |
| **Ring Buffer** | `data/live_buffer.py` | `LiveRingBuffer` maintains a thread-safe sliding window of 60 timesteps $\times$ 22 features (`shape: (60, 22)`). Timeout threshold: 5.0s. |
| **Single-Host Inference** | `api/live_inference.py` | Evaluates 60-step windows with `FGEAD` model. Scaler normalizes window input. Features in `CUMULATIVE_COUNTER_BASELINE_ANCHORS` have invariant counters anchored to baseline means. Forecasting residuals computed per feature across timesteps $t=1..59$. |
| **Multi-Host Manager** | `api/multihost_inference.py` | Host-keyed inference manager supporting model profiles (`windows_default`, `windows_sivachowdary_v2`). Maintains isolated score histories and `LiveEpisodeTracker` per host. |
| **Host Registry & DB** | `api/host_registry.py` & `api/main.py` | SQLite database (`data/fgead_multihost.db`) tracks registered hosts, active agent tokens, and model assignments (`model_id`). |
| **Model Artifacts (v2)** | `checkpoints/fgead_live_*_v2_*` | PyTorch model checkpoint, `joblib` StandardScaler, and threshold JSON ($\tau = 1.411807$) preserved untouched. |

---

## 3. Empirical Baseline Distribution Comparison (V2 vs. Current Normal)

The table below compares the statistical distribution of `data/live_baseline_v2_current_machine.csv` against empirical live telemetry sampled directly from the active host:

| Feature Name | V2 Baseline Mean | V2 Baseline Std | Current Normal Mean | Current Normal Std | Distribution Status / Shift Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `cpu_percent` | 51.10 % | 7.62 | 24.81 % | 2.71 | Lower idle CPU in current state |
| `cpu_freq_current` | 1968.20 MHz | 50.33 | 1969.00 MHz | 0.00 | Clock frequency stable |
| `cpu_user_time_percent` | 5.44 % | 2.67 | 10.62 % | 1.78 | Moderate user-space shift |
| `cpu_system_time_percent` | 45.51 % | 2.04 | 12.25 % | 1.33 | **Severe Mismatch** (V2 snapshot had high kernel load) |
| `cpu_ctx_switches_per_sec` | 31,754.80 | 9428.72 | 220,109.62 | 36,934.51 | **Severe Mismatch** (Higher background activity) |
| `cpu_interrupts_per_sec` | 14,340.29 | 3012.76 | 19,317.01 | 7907.69 | Moderate shift |
| `memory_percent` | 74.63 % | 1.61 | 51.47 % | 0.05 | **Severe Mismatch** (-23.16% shift) |
| `memory_available_mb` | 8,263.42 MB | 402.98 | 15,777.85 MB | 11.78 | **Critical Mismatch** (+7.5 GB shift; $z > 18.0$) |
| `memory_used_mb` | 24,202.65 MB | 407.13 | 16,731.66 MB | 11.78 | **Critical Mismatch** (-7.5 GB shift; $z > 18.0$) |
| `swap_percent` | 2.71 % | 0.48 | 2.10 % | 0.00 | Minor shift |
| `disk_usage_percent` | 55.02 % | 0.52 | 54.20 % | 0.00 | Storage capacity consistent |
| `disk_read_bytes_per_sec` | 30,248.52 B/s | 31,469.78 | 115,333.18 B/s | 116,819.39 | High I/O variance in normal operation |
| `disk_write_bytes_per_sec` | 910,167.89 B/s | 826,097.70 | 568,390.84 B/s | 273,610.25 | Active disk activity |
| `disk_read_count_per_sec` | 3.06 IOPS | 3.19 | 6.58 IOPS | 6.40 | Normal storage read IOPS |
| `disk_write_count_per_sec` | 40.03 IOPS | 34.34 | 56.03 IOPS | 24.18 | Normal storage write IOPS |
| `net_bytes_sent_per_sec` | 985,269.46 B/s | 855,229.48 | 172,706.64 B/s | 497,213.79 | Egress bandwidth variance |
| `net_bytes_recv_per_sec` | 54,607.00 B/s | 46,795.36 | 25,217.50 B/s | 34,620.14 | Ingress bandwidth variance |
| `net_packets_sent_per_sec` | 775.66 pkts/s | 670.09 | 158.25 pkts/s | 391.34 | Packet rate normal |
| `net_packets_recv_per_sec` | 526.29 pkts/s | 450.33 | 115.91 pkts/s | 224.19 | Packet rate normal |
| `net_errors_total` | 2.00 | 0.11 | 1.00 | 0.00 | Cumulative counter (anchored) |
| `net_drops_total` | 0.07 | 0.08 | 0.00 | 0.00 | Cumulative counter (anchored) |
| `process_count` | 381.86 procs | 4.07 | 334.40 procs | 0.84 | Moderate OS process count shift |

---

## 4. Assessment & Recommended V3 Treatment

1. **Root Cause Confirmation:**  
   The false-positive anomaly scores are **not** caused by model architectural defects or broken inference logic. They are caused entirely by a distribution mismatch from synthetic baseline generation.
2. **Schema & Feature Policy:**  
   - All 22 features must be preserved in exact order without alteration.
   - Cumulative counters (`net_drops_total`, `net_errors_total`) must remain anchored using `CUMULATIVE_COUNTER_BASELINE_ANCHORS`.
   - No features should be excluded.
3. **V3 Baseline Methodology:**  
   - Collect a **fresh continuous 30 to 60 minute baseline** directly from the live telemetry pipeline (`LiveTelemetryCollector`) at 1-second sampling intervals.
   - Do **NOT** use synthetic AR(1) generation or artificial variance overrides.
   - Save to `data/live_baseline_v3_current_machine.csv`.
4. **V3 Training & Threshold Calibration Policy:**  
   - Split baseline chronologically into 70% Train, 15% Calibration, and 15% Held-out Normal Test.
   - Fit `StandardScaler` strictly on Train set.
   - Train `FGEAD` V3 model on Train set.
   - Calibrate threshold $\tau$ strictly using Train + Calibration predictions.
   - Validate FPR $\le 1\%$ and Specificity $\ge 99\%$ on Held-out Normal Test set.
