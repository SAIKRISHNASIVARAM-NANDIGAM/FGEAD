# FGEAD Final System Validation Report

**Project:** Feature Graph-based Explainable Anomaly Detector (FGEAD)  
**Date:** October 3, 2026  
**Status:** **PHASE 5 COMPLETE & VERIFIED (ALL TESTS PASSED)**  

---

## 1. System Architecture

The FGEAD platform operates as a dual-mode, end-to-end multivariate time series anomaly detection and root-cause explainability system.

```
+-----------------------------------------------------------------------------------+
|                              FGEAD SYSTEM ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+

           [SMD BENCHMARK MODE]                        [LIVE HOST MONITORING MODE]
                     │                                              │
        data/SMD/ (Machine 1-1)                       Physical Windows Host (SivaChowdary)
        38 Anonymized Features                                      │
                     │                                     data/live_agent.py (psutil)
                     │                                     22 Physical Hardware Metrics @ 1.0s
                     │                                              │
                     │                                       POST /telemetry
                     │                                              │
                     │                                    data/live_buffer.py (LiveRingBuffer)
                     │                                   [Bounded 60s Sliding Window deque]
                     │                                              │
                     │                               ┌──────────────┴──────────────┐
                     │                               │ (< 60s Buffer)              │ (≥ 60s Buffer)
                     │                               ▼                             ▼
                     │                       ⏳ Warm-Up Standby          api/live_inference.py
                     │                       [Buffering Telemetry]      StandardScaler.transform()
                     │                                                             │
                     ▼                                                             ▼
         FGEAD SMD Model (38-Ch)                                     FGEAD Live Model (22-Ch)
       checkpoints/fgead_smd_machine_1_1.pt                       checkpoints/fgead_live_windows_22ch.pt
       Threshold τ = 2.073376                                     Threshold τ = 1.859450
                     │                                                             │
                     ▼                                                             ▼
           POST /predict (SMD)                                      POST /predict/live (Live)
           models/explainer.py                                      Scaler.inverse_transform()
         (38-Channel Graph XAI)                                     (Physical Unit Reconstruction)
                     │                                                             │
                     │                                                    LiveEpisodeTracker
                     │                                                   [Debounce & Grouping]
                     │                                                             │
                     │                                                  5-Question XAI Narrative
                     │                                                Top Contributing Residuals
                     │                                                             │
                     └───────────────────────┬─────────────────────────────────────┘
                                             │
                                             ▼
                                  FastAPI REST Backend (:8000)
                                  app/streamlit_app.py (:8501)
                                 [Stitch Light Academic UI]
```

---

## 2. Core Components

1. **Telemetry Collector (`data/live_agent.py`):**
   - High-precision sampling of 22 physical OS hardware metrics at 1.0s intervals.
   - Computes true differential rates for I/O bytes, I/O operations, network throughput, and hardware interrupts.

2. **Ring Buffer (`data/live_buffer.py`):**
   - Thread-safe (`threading.RLock`) bounded circular buffer storing $60\text{s} \times 22$ telemetry samples.
   - Timeout detection automatically flags agent disconnection after $> 5.0\text{s}$.

3. **Dedicated Live Inference Service (`api/live_inference.py`):**
   - Pre-loads PyTorch checkpoint and `StandardScaler`.
   - Executes $9\text{ ms}$ GPU/CPU neural forecasting without runtime scaler refitting.
   - Performs physical unit reconstruction and 5-Question causal narrative synthesis.

4. **Live Episode Tracker (`LiveEpisodeTracker`):**
   - State machine grouping overlapping sliding windows ($W=60, S=1$) into coherent operational episodes with debounce hysteresis.

5. **FastAPI REST API (`api/main.py`):**
   - Serves both SMD benchmark and Windows live telemetry endpoints with strict schema validation.

6. **Streamlit Operations Center (`app/streamlit_app.py`):**
   - Google Stitch light academic UI providing dynamic mode switching, real-time rolling charts, warm-up progress bar, status hero card, and 5-Question root-cause XAI.

---

## 3. Model Details & Checkpoints

| Parameter | SMD Benchmark Model | Live Windows Host Model |
|---|---|---|
| **Checkpoint Path** | `checkpoints/fgead_smd_machine_1_1.pt` | `checkpoints/fgead_live_windows_22ch.pt` |
| **Input Shape** | $(B, 60, 38)$ | $(B, 60, 22)$ |
| **Embedding Dimension** | 64 | 64 |
| **Self-Attention Heads** | 4 | 4 |
| **Graph Top-K Sparsity** | 5 | 5 |
| **GCN Hidden Units** | 64 | 64 |
| **LSTM Architecture** | 2 Layers, 128 Hidden, Dropout 0.2 | 2 Layers, 128 Hidden, Dropout 0.2 |
| **Validation Threshold ($\tau$)** | **$2.073376$** | **$1.859450$** (99.5th percentile on validation) |
| **Normalization** | Train-only SMD scaler | `checkpoints/fgead_live_scaler.joblib` |

---

## 4. Live Telemetry Feature Schema (Version 1.0)

| Index | Feature ID | Category | Physical Unit | Description |
|---|---|---|---|---|
| 01 | `cpu_percent` | CPU | `%` | Total CPU utilization across all logical cores |
| 02 | `cpu_freq_current` | CPU | `MHz` | Current CPU clock frequency |
| 03 | `cpu_user_time_percent` | CPU | `%` | CPU execution time in user space |
| 04 | `cpu_system_time_percent` | CPU | `%` | CPU execution time in kernel space |
| 05 | `cpu_ctx_switches_per_sec` | CPU | `events/s` | System context switch rate |
| 06 | `cpu_interrupts_per_sec` | CPU | `events/s` | Hardware interrupt rate |
| 07 | `memory_percent` | Memory | `%` | Physical RAM utilization percentage |
| 08 | `memory_available_mb` | Memory | `MB` | Available physical memory |
| 09 | `memory_used_mb` | Memory | `MB` | Allocated physical memory |
| 10 | `swap_percent` | Memory | `%` | Windows page file / swap memory utilization |
| 11 | `disk_usage_percent` | Disk | `%` | Primary storage drive capacity utilized |
| 12 | `disk_read_bytes_per_sec` | Disk | `B/s` / `KB/s` / `MB/s` | Disk storage read throughput |
| 13 | `disk_write_bytes_per_sec` | Disk | `B/s` / `KB/s` / `MB/s` | Disk storage write throughput |
| 14 | `disk_read_count_per_sec` | Disk | `IOPS` | Disk read I/O operations rate |
| 15 | `disk_write_count_per_sec` | Disk | `IOPS` | Disk write I/O operations rate |
| 16 | `net_bytes_sent_per_sec` | Network | `B/s` / `KB/s` / `MB/s` | Outbound network egress bandwidth |
| 17 | `net_bytes_recv_per_sec` | Network | `B/s` / `KB/s` / `MB/s` | Inbound network ingress bandwidth |
| 18 | `net_packets_sent_per_sec` | Network | `pkts/s` | Network packets transmitted per second |
| 19 | `net_packets_recv_per_sec` | Network | `pkts/s` | Network packets received per second |
| 20 | `net_errors_total` | Network | `errors` | Cumulative network error count |
| 21 | `net_drops_total` | Network | `drops` | Cumulative network packet drop count |
| 22 | `process_count` | OS | `procs` | Total active operating system processes |

---

## 5. API Endpoints

- `GET /health`: Health and model readiness status.
- `GET /dataset`: SMD Machine 1-1 benchmark dataset statistics.
- `GET /machine`: SMD feature names and configuration.
- `POST /predict`: 38-channel SMD window inference and full graph XAI report.
- `POST /telemetry`: Live 22-channel snapshot ingestion from Windows agent (triggers automated background inference when buffer is full).
- `POST /predict/live`: Explicit 22-channel $60 \times 22$ live window inference and 5-Question XAI.
- `GET /telemetry/latest`: Most recent single physical telemetry snapshot.
- `GET /telemetry/live_analysis`: Full live neural inference result, anomaly score, threshold, episode status, and rolling score history.
- `GET /agent/status`: Live agent connectivity diagnostics and buffer state.
- `GET /telemetry/history`: Rolling history for dashboard charting.

---

## 6. Dashboard Interface Highlights

- **Header:** Clear machine name (`SivaChowdary`), live connection status badge, model architecture name, and sampling rate.
- **System Status:** 5 instantaneous cards for CPU, Memory, Disk, Network, and Process Count.
- **Model Status:** Dimensions ($60 \times 22$), Calibrated Threshold ($\tau = 1.859450$), Inference State (`Active / Buffering / Offline`), and Buffer Fullness ($X/60$).
- **Hero Status Card:**
  - `⏳ COLLECTING TELEMETRY` during warm-up ($< 60\text{s}$) with progress bar. (Never displays "Normal" during warm-up).
  - `🔴 TELEMETRY AGENT OFFLINE` when telemetry stops.
  - `🟢 NORMAL OPERATION` when score $\le 1.859450$.
  - `🔴 ANOMALY DETECTED` with score ratio and severity level when score $> 1.859450$.
- **Active Episode Box:** Displays active episode ID, start timestamp, duration, peak score, current score, flagged windows, and dominant features (or *"No active anomaly episode"*).
- **XAI Panel:** Displays 5 structured causal answers using neutral, cautious engineering terminology.
- **Top Contributing Features Table:** Ranked physical features with physical units (e.g. `MB/s`, `IOPS`, `MHz`, `%`).
- **Live Anomaly Timeline:** Rolling chart of Anomaly Scores vs $\tau = 1.859450$ with point color coding and automatic refresh.

---

## 7. Explainable AI (5-Question Framework)

1. **Question 1: WHAT HAPPENED?**
   - *Anomaly:* "A rolling 60-second telemetry window departed significantly from the learned normal baseline forecasting pattern."
   - *Normal:* "Nominal system operation — all 22 physical telemetry counters match baseline spatio-temporal forecasts."
2. **Question 2: WHEN DID IT HAPPEN?**
   - *Anomaly:* Exact UTC timestamp and active episode duration.
3. **Question 3: HOW SEVERE IS IT?**
   - *Anomaly:* Quantified score vs threshold (e.g. "$7.1748$ against calibrated threshold $\tau = 1.8595$ ($3.86\times$ above threshold)").
4. **Question 4: WHAT FEATURES CONTRIBUTED?**
   - *Anomaly:* Ranked physical features with highest forecast residuals (e.g. `net_bytes_sent_per_sec`, `cpu_ctx_switches_per_sec`).
5. **Question 5: WHICH SUBSYSTEM APPEARS INVOLVED?**
   - *Anomaly:* Storage I/O, Network, CPU Subsystem, Memory, or OS.

---

## 8. Performance & Latency

- **Inference Latency:** **$9.06\text{ ms}$ mean** ($19.2\text{ ms}$ p95) on CPU for a $60 \times 22$ window — well within the $< 50\text{ ms}$ operational budget.
- **Agent Overhead:** $< 0.5\%$ CPU utilization for 1.0s `psutil` sampling.
- **Memory Footprint:** Fixed-size bounded deques ($N \le 120$) prevent memory leaks during long-running execution.

---

## 9. Verification & Test Results

All 10 validation dimensions passed verification:

```json
{
  "smd_mode": true,
  "live_mode": true,
  "telemetry": true,
  "model_loading": true,
  "inference": true,
  "xai": true,
  "episode_tracking": true,
  "offline_detection": true,
  "recovery": true,
  "smd_isolation": true
}
```

---

## 10. Known Limitations

1. **Host-Specific Calibration:** The live model is calibrated to the baseline operating behavior of the host machine (`SivaChowdary`). Deploying on a new host requires collecting a baseline on that machine.
2. **Single Drive Monitoring:** Disk usage and throughput currently monitor the primary OS mount point.
3. **Warm-Up Requirement:** Accurate graph learning requires a minimum of 60 consecutive seconds of telemetry before issuing neural predictions.
