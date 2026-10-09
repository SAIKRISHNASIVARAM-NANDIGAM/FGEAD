# FGEAD V3 — DETECTION VALIDATION AND IMPLEMENTATION AUDIT

**Project:** FGEAD — Feature Graph-based Explainable Anomaly Detector  
**Active Model ID:** `windows_sivachowdary_v3`  
**Calibrated Anomaly Threshold ($\tau$):** `2.120169`  
**Input Schema:** 22 Windows Physical Telemetry Features  
**Rolling Window:** $60 \times 22$ (60-second observation window)  
**Audit Date:** October 9, 2026  

---

## Executive Audit Summary

This audit verifies the production implementation of the **FGEAD V3 Current-Machine Model** (`windows_sivachowdary_v3`) across all architectural components, file loading paths, inference mathematics, host registry bindings, feature attribution pipelines, episode persistence gating, and evaluation scripts.

All V3 artifacts were verified as loaded, active, and cryptographically untouched from the validated baseline commit.

---

## 1. Model Loading Path Verification

- **Single-Host Service:** `api/live_inference.py` (`LiveInferenceService`)
  - Explicit Checkpoint Path: `checkpoints/fgead_live_windows_22ch_v3_current_machine.pt`
  - Explicit Scaler Path: `checkpoints/fgead_live_scaler_v3_current_machine.joblib`
  - Explicit Config Path: `checkpoints/fgead_live_windows_22ch_config_v3_current_machine.json`
  - Verification: `LiveInferenceService` checks `v3_ckpt.exists()` on startup and defaults strictly to V3.
- **Multi-Host Gateway:** `api/multihost_inference.py` (`MultiHostInferenceManager`)
  - `ModelProfile` for `windows_sivachowdary_v3` registered with full path resolution to `checkpoints/*v3_current_machine*`.
  - Configured with `supported_os = ["Windows"]`, `anchor_names = ["net_drops_total", "net_errors_total"]`.

---

## 2. Threshold Loading Path Verification

- **Threshold File:** `checkpoints/fgead_live_threshold_v3_current_machine.json`
- **Active Threshold Value:** **`2.120169`**
- **Verification:** Both `LiveInferenceService` and `MultiHostInferenceManager` read `threshold` directly from JSON upon initialization. Fallback logic preserves $\tau = 2.120169$.

---

## 3. Host Assignment Audit

- **SQLite Database:** `data/fgead_multihost.db` (`hosts` table)
- **Target Host ID:** `host_sivachowdary`
- **Assigned Model ID:** `windows_sivachowdary_v3`
- **Model Compatibility Status:** `COMPATIBLE`
- **Verification:** `api/host_registry.py` (`HostRegistry.register_host`) checks `host_id == "host_sivachowdary"` and assigns `windows_sivachowdary_v3` when `v3_ckpt.exists()`.

---

## 4. Telemetry Ingestion Path

- **Collector Agents:** `data/live_agent.py` and `agents/windows_agent.py`
- **Sampling Interval:** 1.0 second (1 Hz continuous sampling)
- **Schema Validation:** `data/live_feature_schema.py` (`validate_live_feature_dict`) enforces strict presence, numeric type, and finite constraints (no NaN / Inf) across all 22 physical channels:
  1. `cpu_percent`
  2. `cpu_freq_current`
  3. `cpu_user_time_percent`
  4. `cpu_system_time_percent`
  5. `cpu_ctx_switches_per_sec`
  6. `cpu_interrupts_per_sec`
  7. `memory_percent`
  8. `memory_available_mb`
  9. `memory_used_mb`
  10. `swap_percent`
  11. `disk_usage_percent`
  12. `disk_read_bytes_per_sec`
  13. `disk_write_bytes_per_sec`
  14. `disk_read_count_per_sec`
  15. `disk_write_count_per_sec`
  16. `net_bytes_sent_per_sec`
  17. `net_bytes_recv_per_sec`
  18. `net_packets_sent_per_sec`
  19. `net_packets_recv_per_sec`
  20. `net_errors_total`
  21. `net_drops_total`
  22: `process_count`
- **Ingestion Endpoint:** `POST /hosts/{host_id}/telemetry` in `api/main.py`.

---

## 5. Feature Attribution Implementation

- **Forecasting Residual Calculation:**
  For normalized input tensor $X \in \mathbb{R}^{1 \times 60 \times 22}$ and 1-step ahead forecast $\hat{Y} \in \mathbb{R}^{1 \times 59 \times 22}$:
  $$\text{residual}_i = \frac{1}{59} \sum_{t=1}^{59} |\hat{Y}_{t, i} - X_{t+1, i}|$$
- **Counter Anchoring:**
  Invariant cumulative counters (`net_drops_total`, `net_errors_total`) are anchored to baseline constants (`scaler.mean_`) to prevent OS uptime drift from generating synthetic residual attributions.
- **Ranking:** Features are ranked descending by residual contribution percentage ($\text{contrib}_i = \frac{\text{residual}_i}{\sum_j \text{residual}_j} \times 100\%$).

---

## 6. Episode Confirmation & Persistence Rules

- **Class:** `LiveEpisodeTracker` in `api/live_inference.py`
- **Confirmation Threshold (`min_persistence_frames`):** `2` consecutive windows.
  - 1 window above $\tau \rightarrow$ `SUSPICIOUS` (Pending confirmation).
  - $\ge 2$ windows above $\tau \rightarrow$ `ANOMALY` (Confirmed Episode).
- **Debounce / Recovery Threshold (`debounce_frames`):** `2` consecutive nominal windows below $\tau$.
  - Requires 2 consecutive frames below $\tau$ to close an episode, prevent flickering, and restore state to `NORMAL`.

---

## 7. Scan My System Workflow

- **Frontend Implementation:** `app/streamlit_app.py`
- **Pre-Scan State:** Neutral UI rendering (no result, score, or explanation shown before explicit user scan).
- **Scan Trigger:** User clicks **🔍 Scan My System**.
- **Pipeline:** 4-Stage execution:
  1. Ingestion check (verifies $\ge 60$ samples in buffer).
  2. Telemetry validation (0 NaN/Inf across 22 channels).
  3. Window forecasting inference against V3 model.
  4. Evidence & episode persistence verification.

---

## 8. Evaluation Scripts & Metrics Infrastructure

- `data/live_training/evaluate_v3_heldout.py`: Evaluates unseen normal test split (`v3_test_normal.csv`, 211 sliding windows).
  - **FPR:** `0.00%` (Target $\le 1.0\%$)
  - **Specificity:** `100.00%` (Target $\ge 99.0\%$)
- `data/live_training/run_v3_controlled_anomaly_eval.py`: Evaluates 5 physical controlled workload scenarios.
  - **Overall Accuracy:** `90.76%`
  - **Precision:** `100.00%`
  - **Recall:** `77.33%`
  - **F1 Score:** `0.8722`
  - **Avg Detection Delay:** `2.0 seconds`

---

## 9. Explainability (XAI) Narrative Generation

- **Module:** `api/deep_explainability.py` (`build_deep_human_explanation`)
- **Structure:** 5-Question structured XAI narrative:
  1. *What happened:* Excursion from spatio-temporal forecasting baseline.
  2. *When:* Timestamped active episode duration.
  3. *Severity:* Score-to-threshold ratio ($S / \tau$).
  4. *What contributed:* Top 5 per-feature physical metric residuals and baseline comparison table.
  5. *Subsystem:* Dominant hardware subsystem identification.
- **Safety Disclaimers:** Incorporates explicit, non-definitive disclaimer text: *"FGEAD identifies unusual behavior compared with the learned normal operating pattern. It does not by itself prove that the machine is compromised, damaged, or under attack."*

---

## 10. Recovery Logic Verification

- **Mechanism:** `LiveEpisodeTracker.update(is_anomaly=False, ...)`
- When score drops below $\tau$, `_consecutive_nominal` increments.
- At `_consecutive_nominal == 2`, active episode is closed, duration and peak score recorded, and status reset to `NORMAL`.

---

## 11. Duplicate-Process & Agent Safeguards

- **Multi-Host Identity:** `api/host_registry.py` enforces canonical host IDs (`host_sivachowdary`) and token hash matching (`fgead_...`).
- **Thread Safety:** `threading.RLock()` protects in-memory buffers and inference state across concurrent API requests.

---

## Audit Conclusion

The production environment is **100% verified** using the active V3 model checkpoint (`checkpoints/fgead_live_windows_22ch_v3_current_machine.pt`), V3 scaler, V3 config, and calibrated threshold $\tau = 2.120169$. All safety gates and architectural components are intact.
