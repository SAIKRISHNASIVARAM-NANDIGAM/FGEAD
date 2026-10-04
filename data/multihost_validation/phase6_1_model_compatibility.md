# FGEAD Multi-Host Platform — Phase 6.1 Model Compatibility & Validation Report

**Date:** October 4, 2026  
**Status:** **PHASE 6.1 MODEL COMPATIBILITY: PASSED**  
**Phase 6.1 Test Suite Result:** 10 / 10 Checks Passed (100% Success Rate)  
**Phase 6 Regression Test Result:** 17 / 17 Checks Passed (100% Success Rate)  
**Trained Windows Model:** `checkpoints/fgead_live_windows_22ch.pt` ($\tau = 1.859450$, 22 Channels) — **Intact & Preserved**  
**SMD Benchmark Pipeline:** `checkpoints/fgead_smd_machine_1_1.pt` ($\tau = 2.073376$, 38 Channels) — **Intact & Isolated**

---

## 1. Executive Summary & Scientific Findings

During initial multi-host deployment, cross-host anomaly tests revealed that running the physical Windows-trained model on arbitrary machines (including Linux servers and different Windows environments) produced false anomaly scores dominated by `net_drops_total` ($\approx 21 \text{ to } 48$).

### Root Cause Analysis:
1. **Physical Baseline Specificity:** The original live Windows model was trained on 60 minutes of normal telemetry from one physical Windows machine where `net_drops_total` had $\text{min}=294, \text{max}=294, \text{std}=0$.
2. **Scaler Zero-Variance Distortion:** When standardizing on features with $\text{std} \approx 0$, any host operating at $\text{net\_drops\_total} = 0$ (or a different static hardware drop count) produced large normalized z-score excursions, causing the model forecast head to flag massive spurious residuals.
3. **Cross-OS Incompatibility:** Linux network subsystems, storage I/O, process count dynamics, and context switch scales differ systematically from Windows NT kernel dynamics.

**Phase 6.1 Resolution:** We implemented explicit model compatibility metadata, OS-level inference gating, constant feature distribution profiling, host-specific baseline preparation workflows, and reconnecting identity reuse.

---

## 2. Model Compatibility & Gating Architecture

```
                    ┌───────────────────────────────┐
                    │      TELEMETRY INGESTION      │
                    │      POST /hosts/.../telemetry│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   MODEL COMPATIBILITY CHECK   │
                    │   (OS, Schema, Baseline, Scaler)
                    └───────┬───────────────┬───────┘
                            │               │
             [Compatible OS]│               │[Incompatible OS / No Model]
             (e.g., Windows)│               │(e.g., Linux, FreeBSD, None)
                            ▼               ▼
          ┌───────────────────────────┐   ┌───────────────────────────┐
          │    FGEAD LIVE INFERENCE   │   │     TELEMETRY BUFFER ONLY │
          │  • Window 60 × 22         │   │  • Telemetry Stored       │
          │  • Anomaly Scoring        │   │  • Status: TELEMETRY_ONLY │
          │  • Debounced Episodes     │   │  • Score: None / "—"      │
          │  • 5-Question XAI         │   │  • No Spurious Alerts     │
          └───────────────────────────┘   └───────────────────────────┘
```

### 2.1 Model Profile Compatibility Metadata
Each FGEAD model profile declares explicit metadata:
- `model_id`: `fgead_live_windows_22ch` (alias: `windows_default`)
- `supported_os`: `["Windows"]`
- `training_host_type`: `Windows physical host`
- `training_baseline_id`: `live_baseline_60min_windows`
- `feature_schema_version`: `1.0` (22 channels)
- `scaler_id`: `fgead_live_scaler_joblib`
- `threshold_id`: `fgead_live_threshold_json`
- `threshold`: `1.859450`
- `is_universal`: `False` (explicitly not marked as universal)

### 2.2 Host Model Assignment Policy
- **Windows Hosts:** When schema version (`1.0`) and compatibility checks pass, the Windows model is assigned (`model_status: COMPATIBLE`, `status: ONLINE` / `ANOMALY`).
- **Linux Hosts:** In the absence of a trained Linux model, anomaly inference is strictly disabled (`model_status: BASELINE_REQUIRED`, `status: TELEMETRY_ONLY`, `latest_score: None`).
- **Unknown / Unsupported OS:** Model gating disables inference with explicit message.

---

## 3. Constant Feature & Distribution Difference Handling

A new analysis module ([`data/baseline_compatibility.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/baseline_compatibility.py)) performs distribution profiling across all 22 features:

1. **Feature Quality Policy:** Constant features (`net_drops_total`, `net_errors_total`) are retained in the schema and tensor shape $(60, 22)$ to preserve checkpoint weight dimensions.
2. **Baseline Distribution Difference Engine:** Computes $\text{mean}, \text{std}, \text{min}, \text{max}, \text{P95}, \text{P99}$ for the host and compares against the reference training baseline.
3. **Scientific Classification:** Distribution discrepancies (such as $\Delta \text{net\_drops} = 294 \rightarrow 0$) are categorized as **`"baseline compatibility difference"`**, explicitly preventing them from being misclassified as runtime anomalies.

---

## 4. Host-Specific Baseline Collection Workflow

Endpoints added for host baseline preparation:
- `POST /hosts/{host_id}/baseline/collect`: Captures host rolling buffers, validates 22 physical metrics, non-finite values, and computes statistical distribution profiles without modifying trained checkpoints.
- `GET /hosts/{host_id}/baseline/comparison`: Compares host baseline distributions against the model training reference.
- `GET /models/metadata`: Exposes model compatibility declarations.

---

## 5. Duplicate Host Reconnection & Identity Reuse

To prevent duplicate records (e.g. `SivaChowdary-PC` vs `SivaChowdary` vs random IDs):
1. **Deterministic Identity Matching:** `register_host` queries existing records by `hostname` + `operating_system` and reuses the established `host_id`.
2. **Local Credential Persistence:** Telemetry agents persist authorized tokens in local config files (`.fgead_agent_auth.json` / `windows_agent_config.json`).
3. **Reconciliation Utility:** `HostRegistry.reconcile_duplicate_hosts()` prunes historical duplicate entries.

---

## 6. Dashboard Display States (Fleet Operations Center)

| Host State | OS | Telemetry | Model | Compatibility | Status Badge | Score | Threshold |
|---|---|---|---|---|---|---|---|
| **Supported Windows (Normal)** | Windows | 🟢 Connected | `FGEAD Windows 22ch` | `Compatible` | `🟢 MODEL ACTIVE` | `1.2415` | `1.85945` |
| **Supported Windows (Excursion)** | Windows | 🟢 Connected | `FGEAD Windows 22ch` | `Compatible` | `🔴 ANOMALY` | `2.4180` | `1.85945` |
| **Linux Host (Pending Model)** | Linux | 🟢 Connected | `None` | `Baseline Required` | `🟡 TELEMETRY ONLY` | `—` | `—` |
| **Disconnected Node** | Any | ⚪ Inactive | Any | Any | `⚪ OFFLINE` | `—` | `—` |

---

## 7. Automated Test Suite Results

### Phase 6.1 Compatibility Suite (`test_phase6_1_compatibility.py`):
```
================================================================================
FGEAD PHASE 6.1 — MODEL COMPATIBILITY & CROSS-HOST VALIDATION SUITE
================================================================================
 [PASS] 1. Windows host + Windows model (Compatible & Allowed)
 [PASS] 2. Linux host + Windows model (Inference Rejected & Gated)
 [PASS] 3. Unknown OS (Inference Disabled)
 [PASS] 4. Schema mismatch (Inference Rejected)
 [PASS] 5. Model, Scaler & Threshold Verification
 [PASS] 6. Wrong model assignment (Inference Rejected)
 [PASS] 7. Constant feature difference (Classified as Baseline Compatibility Difference)
 [PASS] 8. Host A/B telemetry isolation
 [PASS] 9. Duplicate agent reconnect (Identity Reused & No Duplicate Hosts)
 [PASS] 10. Offline / recovery lifecycle
================================================================================
PHASE 6.1 SUMMARY: 10 / 10 CHECKS PASSED
================================================================================
```

### Phase 6 Full Regression Suite (`test_multihost_suite.py`):
```
================================================================================
FGEAD PHASE 6 — MULTI-HOST PLATFORM VALIDATION SUITE
================================================================================
 [PASS] 1. Register Windows host
 [PASS] 2. Register Linux host
 [PASS] 3. Authenticate agent
 [PASS] 4. Send telemetry
 [PASS] 5. Reject unauthenticated telemetry
 [PASS] 6. Reject unknown host
 [PASS] 7. Reject wrong feature count
 [PASS] 8. Reject NaN
 [PASS] 9. Reject Inf
 [PASS] 10. Maintain independent host buffers
 [PASS] 11. Run independent inference (Windows Active & Linux Gated)
 [PASS] 12. Track independent episodes
 [PASS] 13. Detect offline host
 [PASS] 14. Recover host
 [PASS] 15. Verify dashboard host selection
 [PASS] 16. Verify fleet dashboard
 [PASS] 17. Verify SMD isolation
================================================================================
SUMMARY: 17 / 17 CHECKS PASSED
================================================================================
```

---

## 8. Conclusion

All cross-host false alerts caused by constant feature shifts and OS distribution differences are resolved. The system guarantees that neural anomaly detection executes **only** when a scientifically compatible model and baseline exist.

```
================================================================================
PHASE 6.1 MODEL COMPATIBILITY: PASSED
================================================================================
```
