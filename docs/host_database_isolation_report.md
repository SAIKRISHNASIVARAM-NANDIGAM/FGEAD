# FGEAD Host Database Isolation & Clean Fleet Operations Audit Report

**Date:** 2026-10-05
**System:** FGEAD (Feature-Level Graph-Based Explainable Anomaly Detection)
**Status:** COMPLETED & VERIFIED

---

## 1. Executive Summary

This engineering audit details the implementation of strict test database isolation, the resolution of duplicate host registrations, the safety backup and cleanup of the operational multi-host database (`data/fgead_multihost.db`), and end-to-end verification across the FastAPI backend, Streamlit frontend, live agents, and regression test suites.

Prior to this fix:
- Automated regression tests (`test_phase6_1_compatibility.py`, `test_multihost_suite.py`, `test_smoke_phase7.py`, `test_cloud_deployment_phase8.py`, `test_phase9_accuracy.py`) executed against the default production database path, inserting 17 synthetic test fixture records.
- In-memory SQLite connections retained the default path, cluttering the Fleet Operations Center host selector.

After this implementation:
- **Test Database Isolation:** Test suites run against isolated temporary/per-test SQLite databases using `IsolatedTestDatabase` and `reset_host_registry()`. Zero records leak into the operational database.
- **Safety Backup Created:** Operational DB backed up bit-for-bit to `data/fgead_multihost.db.backup_before_test_cleanup` (49,152 bytes).
- **Clean Operational State:** Exactly 2 legitimate hosts exist in the production database:
  1. `host_sivachowdary` (`SivaChowdary`, Windows) — Active physical host.
  2. `host_linux_srv01` (`Ubuntu-Prod-Server-01`, Linux) — Reference Linux server for model gating.
- **Stable Identity Resolution:** 3-tier deterministic host identity matching (Explicit ID $\to$ Hardware MAC UUID $\to$ Normalized Hostname + OS) ensures live agent reconnects and cold restarts always reuse `host_sivachowdary` without creating duplicate records.
- **Verification:** Full regression test suites (68/68 automated checks) pass 100% with the operational database remaining strictly at 2 hosts.

---

## 2. Root Cause Analysis

### 2.1 Why Host Records Cluttered the Dashboard
1. **Direct Operational Database Ingestion During Automated Tests:**
   When automated test suites initialized `TestClient(app)`, `get_host_registry()` was invoked before the test context had redirected the SQLite database path. As a result, synthetic test fixtures (e.g., `test_host_win_01`, `host_machine_alpha_server`, `host_test_node_alpha`, `host_dual_os_node`) were inserted directly into `data/fgead_multihost.db`.
2. **Global Environment Variable Bleed:**
   Previous test teardowns popped `FGEAD_DB_PATH` from `os.environ`, causing subsequent test modules to fall back to the default operational database path.

---

## 3. Architecture & Implementation Details

### 3.1 3-Tier Deterministic Host Identity Resolution
In [`api/host_registry.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/api/host_registry.py):
1. **Tier 1 (Explicit Custom ID):** If `custom_host_id` is supplied and authorized, it is reused directly.
2. **Tier 2 (Hardware Machine Node UUID):** Telemetry agents extract the physical MAC address (`uuid.getnode()`) stored in `machine_info_json["machine_id"]`. If a host record exists with matching hardware identity and operating system, it is reused immediately.
3. **Tier 3 (Normalized Hostname + OS):** Normalized lowercase `(hostname, operating_system)` lookup via partial unique index:
   ```sql
   CREATE UNIQUE INDEX IF NOT EXISTS idx_hosts_unique_identity
   ON hosts (LOWER(hostname), LOWER(operating_system))
   WHERE is_enabled = 1;
   ```
4. **Deterministic Canonical ID Generation:** New hosts receive deterministic IDs (`host_{clean_hostname}`), with cross-OS disambiguation (`host_{clean_hostname}_{os.lower()}`).

### 3.2 Dynamic Database Path & Registry Reset
- `get_host_registry()` now dynamically evaluates `os.getenv("FGEAD_DB_PATH", "data/fgead_multihost.db")`.
- `reset_host_registry(db_path=...)` allows test frameworks and API lifecycle handlers to cleanly switch databases and clear active connection pools.

### 3.3 Test Database Isolation Helper
In [`data/multihost_validation/test_db_helper.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_db_helper.py):
- `IsolatedTestDatabase` provides a context manager creating a unique temporary SQLite file (e.g., `tempfile.NamedTemporaryFile(prefix="fgead_test_", suffix=".db")`), initializing tables, setting `FGEAD_DB_PATH`, calling `reset_host_registry()`, and cleanly tearing down upon completion.

---

## 4. Verification & Audit Results

### 4.1 Operational Database Audit
- **Path:** `data/fgead_multihost.db`
- **Backup:** `data/fgead_multihost.db.backup_before_test_cleanup`
- **Integrity Check:** `PRAGMA integrity_check` $\to$ `ok`
- **Current Host Records:**
  | Host ID | Hostname | OS | Status | Model Compatibility |
  | :--- | :--- | :--- | :--- | :--- |
  | `host_sivachowdary` | SivaChowdary | Windows | ANOMALY / ONLINE | Compatible (22 Channels) |
  | `host_linux_srv01` | Ubuntu-Prod-Server-01 | Linux | TELEMETRY_ONLY | Baseline Required |

### 4.2 Automated Test Suite Results
| Test Suite | File | Checks | Status |
| :--- | :--- | :---: | :---: |
| Host Registration Lifecycle | [`test_host_registration_lifecycle.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_host_registration_lifecycle.py) | 8/8 | **PASS** |
| Production Smoke Suite | [`test_smoke_phase7.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_smoke_phase7.py) | 7/7 | **PASS** |
| Cloud Deployment Readiness | [`test_cloud_deployment_phase8.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_cloud_deployment_phase8.py) | 4/4 | **PASS** |
| Accuracy & Decision Gate | [`test_phase9_accuracy.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_phase9_accuracy.py) | 18/18 | **PASS** |
| Model Compatibility & Gating | [`test_phase6_1_compatibility.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_phase6_1_compatibility.py) | 10/10 | **PASS** |
| Multi-Host Platform Suite | [`test_multihost_suite.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_multihost_suite.py) | 17/17 | **PASS** |
| Operational DB Isolation Invariant | [`test_operational_db_isolation_invariant.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_operational_db_isolation_invariant.py) | 4/4 | **PASS** |
| **Total** | | **68/68** | **100% PASS** |

### 4.3 Database Pollution Invariant Test
After running all 68 tests across the test discovery pipeline, `data/fgead_multihost.db` was verified:
- **Total Hosts:** Exactly 2 (`host_sivachowdary`, `host_linux_srv01`).
- **Synthetic Host Leakage:** 0 records.
- **Alert Table Leakage:** 0 records.

---

## 5. Summary of Model Preservations
- **Checkpoints Unmodified:**
  - `checkpoints/fgead_live_windows_22ch.pt` (MD5 preserved)
  - `checkpoints/fgead_smd_machine_1_1.pt` (MD5 preserved)
- **Scaler Unmodified:** `checkpoints/fgead_live_scaler.joblib`
- **Threshold Unmodified:** $\tau = 1.859450$ (Windows Live), $\tau = 2.073376$ (SMD Machine 1-1)
- **Feature Vector Unmodified:** 22 channels, exact order maintained.
