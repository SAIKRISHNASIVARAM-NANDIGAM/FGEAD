# FGEAD Phase 4 Report: Real-Time Live Inference & 5-Question Explainability Integration

**Project:** Feature Graph-based Explainable Anomaly Detector (FGEAD)  
**Target Platform:** Physical Windows Computer / Host Monitoring  
**Phase Completed:** Phase 4 — Real-Time Online Inference, Episode Tracking, and 5-Question Root-Cause XAI  
**Date:** October 3, 2026  

---

## 1. Executive Summary

Phase 4 successfully delivers production real-time online inference and dynamic 5-Question Explainable AI (XAI) for the dedicated 22-channel Windows physical host FGEAD model. 

The entire pipeline has been verified with **18/18 Critical Safety & Verification Checks passing**, maintaining strict mathematical isolation between the Server Machine Dataset (SMD) 38-feature benchmark pipeline and the Windows 22-channel live telemetry monitoring pipeline.

```
       PHYSICAL WINDOWS COMPUTER (Host)
                     │
                     ▼
          data/live_agent.py (psutil)
         [22 Physical Metrics @ 1.0s]
                     │
                     ▼
             POST /telemetry
                     │
                     ▼
       data/live_buffer.py (LiveRingBuffer)
      [Thread-Safe 60s Sliding Window]
                     │
     ┌───────────────┴───────────────┐
     │ (Buffer < 60s)                │ (Buffer = 60s Full)
     ▼                               ▼
⏳ Warm-Up Standby           api/live_inference.py
[Buffering Telemetry]        StandardScaler.transform()
                                     │
                                     ▼
                            FGEAD Model (22-Ch)
                            [GCN + LSTM Forecast]
                                     │
                                     ▼
                          Peak Anomaly Score Max
                          Score vs τ (1.859450)
                                     │
                                     ▼
                         Scaler.inverse_transform()
                        [Unscaled Physical Telemetry]
                                     │
                                     ▼
                          LiveEpisodeTracker
                         [Debounce & Grouping]
                                     │
                                     ▼
                       5-Question XAI Narrative
                     Top Contributing Residuals
                                     │
                                     ▼
                      Streamlit Live Operations
```

---

## 2. Architecture & Calibration Artifacts

| Component | Path / Location | Specification |
|---|---|---|
| **Dedicated Model Checkpoint** | `checkpoints/fgead_live_windows_22ch.pt` | 22 Features, Embed: 64, Heads: 4, LSTM: 128 |
| **Train-Fitted Scaler** | `checkpoints/fgead_live_scaler.joblib` | `StandardScaler` fitted strictly on train split ($N=2,461$) |
| **Calibrated Threshold** | `checkpoints/fgead_live_threshold.json` | $\tau = 1.859450$ (99.5th percentile on validation split) |
| **Model Configuration** | `checkpoints/fgead_live_windows_22ch_config.json` | $W=60, S=1, N=22$, Version: `1.0` |
| **Inference Service** | `api/live_inference.py` | Thread-safe singleton `LiveInferenceService` |
| **API Endpoints** | `api/main.py` | `POST /predict/live`, `GET /telemetry/live_analysis`, `GET /agent/status` |
| **Dashboard Interface** | `app/streamlit_app.py` | Live Machine Mode with Status Hero Card, 5-Q XAI, Residual Table |

---

## 3. Strict Pipeline Isolation Guarantee

To preserve scientific rigor, the system enforces complete physical and logical separation:

1. **SMD Benchmark Pipeline:**
   - Model: `fgead_smd_machine_1_1.pt` (38 features, stride 5).
   - Serves: `GET /dataset`, `GET /machine`, `POST /predict`.
   - Scaler: SMD train dataset normalization.
   - Status: **100% Intact & Unmodified**.

2. **Windows Physical Host Pipeline:**
   - Model: `fgead_live_windows_22ch.pt` (22 physical features, stride 1).
   - Serves: `POST /telemetry`, `POST /predict/live`, `GET /telemetry/live_analysis`, `GET /agent/status`.
   - Scaler: Dedicated `fgead_live_scaler.joblib`.
   - Status: **Fully Operational**.

---

## 4. Online Inference & Explainability Methodology

### 4.1. Preprocessing & Forecast Scoring
- Incoming raw window $\mathbf{X}_{\text{raw}} \in \mathbb{R}^{60 \times 22}$ is transformed via train-fitted scaler: $\mathbf{X}_{\text{norm}} = \text{scaler.transform}(\mathbf{X}_{\text{raw}})$. No refitting or data leakage occurs online.
- PyTorch tensor $(1, 60, 22)$ is passed through `FGEAD`.
- Anomaly score is computed via official max-step residual:
  $$\text{Score}(\mathbf{X}) = \max_{t \in [1, 59]} \text{scores\_step}(t)$$
- Decision Rule:
  $$\text{Is\_Anomaly} = (\text{Score} > \tau), \quad \tau = 1.859450$$

### 4.2. Physical Unit Unscaling for Intuitive XAI
To ensure operational clarity, predictions are inverse-transformed back to physical units (e.g., % CPU, MB/s Disk I/O, Interrupts/s) before presenting to the user:
- Current Physical Value vs Predicted Physical Value
- Feature Contribution Percentage:
  $$\text{Contrib}_i = \frac{|x_{t, i} - \hat{x}_{t, i}|}{\sum_{j=1}^{22} |x_{t, j} - \hat{x}_{t, j}|} \times 100\%$$

### 4.3. Neutral 5-Question Causal Narrative
The system formats explanations into 5 structured answers using cautious, neutral terminology:
1. **What happened?** (Forecast departure or nominal baseline tracking)
2. **When?** (Timestamp and active episode duration)
3. **How severe?** (Score relative to calibrated threshold $\tau$)
4. **What contributed?** (Top contributing residual features)
5. **Which subsystem?** (CPU, Memory, Storage I/O, Network, or Operating System)

### 4.4. Sliding Window Episode Debounce
Because sliding windows overlap across 59 timesteps, a single transient burst triggers multiple contiguous flagged windows. The `LiveEpisodeTracker` groups overlapping windows into a single coherent incident with start time, duration, peak score, and dominant features, using a 2-frame debounce to avoid alert flickering.

---

## 5. 18-Point Critical Safety Verification Results

All 18 checks were executed via `data/live_inference/verify_phase4.py` and passed with 100% compliance:

| Check # | Verification Item | Expected Criteria | Result |
|---|---|---|---|
| **01** | Checkpoint Existence & Architecture | Checkpoint exists, $N=22$ channels | ✅ **PASSED** |
| **02** | Scaler Existence & Calibration | Scaler exists, 22-dimensional mean vector | ✅ **PASSED** |
| **03** | Threshold Config Integrity | $\tau = 1.859450$ exactly | ✅ **PASSED** |
| **04** | Feature Schema & Ordering | Exact match against `live_feature_schema.py` | ✅ **PASSED** |
| **05** | SMD 38-Channel Isolation | SMD checkpoint exists and unmodified | ✅ **PASSED** |
| **06** | Service Initialization | `LiveInferenceService` loads cleanly | ✅ **PASSED** |
| **07** | Nominal Window Inference | Score $< 1.859450$, `is_anomaly = False` | ✅ **PASSED** |
| **08** | Invalid Window Length Rejection | Length $\ne 60$ raises `ValueError` (422) | ✅ **PASSED** |
| **09** | Invalid Feature Count Rejection | Features $\ne 22$ raises `ValueError` (422) | ✅ **PASSED** |
| **10** | NaN Value Rejection | NaN input rejected with `ValueError` | ✅ **PASSED** |
| **11** | Infinite Value Rejection | Inf input rejected with `ValueError` | ✅ **PASSED** |
| **12** | Synthetic Anomaly Detection | Anomaly injection triggers Score $> \tau$ | ✅ **PASSED** |
| **13** | Top Contributing Feature Ranking | Correct feature ranked #1 with high % | ✅ **PASSED** |
| **14** | Unscaled Physical Reconstructions | Real physical units in XAI response | ✅ **PASSED** |
| **15** | 5-Question Narrative Completeness | All 5 keys present, non-probabilistic tone | ✅ **PASSED** |
| **16** | Episode Tracker Debounce | Episode opened, debounced, cleanly closed | ✅ **PASSED** |
| **17** | Ring Buffer & Window Extraction | Thread-safe buffer extracts $(60, 22)$ | ✅ **PASSED** |
| **18** | Latency Budget Compliance | Mean latency $9.06\text{ ms} < 50\text{ ms}$ (p95: $9.66\text{ ms}$) | ✅ **PASSED** |

---

## 6. Live Machine Real-Time Execution Verification

- **Agent:** `data/live_agent.py` transmitting live Windows telemetry every 1.0s.
- **Buffer:** Dynamically transitions from `Warm-Up Standby` ($< 60\text{s}$) to `Active Neural Inference` ($\ge 60\text{s}$).
- **Inference Latency:** $\sim 9.1\text{ ms}$ per 60s sliding window on CPU.
- **Dashboard:** Streamlit UI automatically renders the Live Machine Hero Card, 5-Question XAI narrative, physical residual table, and rolling anomaly timeline.

---

## 7. Conclusion

Phase 4 is complete and verified. The FGEAD system is now fully capable of monitoring a real physical Windows machine in real time with scientifically defensible explainability and robust episode tracking.
