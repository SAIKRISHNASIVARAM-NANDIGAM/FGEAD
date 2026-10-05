# FGEAD Host Count Proliferation Diagnostic Audit Report

## 1. Executive Summary

A comprehensive read-only diagnostic investigation was conducted on the multi-host database [`data/fgead_multihost.db`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/fgead_multihost.db), the registration lifecycle code in [`api/host_registry.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/api/host_registry.py), and the automated test suites.

### Key Finding:
The physical live agent on your Windows PC (`SivaChowdary`) is **NOT** creating duplicate host records on restart or reconnect. It consistently resolves to `host_sivachowdary`.

The reason the Fleet Overview host count increased to **19 hosts** is that **automated regression test suites (`test_phase6_1_compatibility.py`, `test_multihost_suite.py`, `test_host_registration_lifecycle.py`) were executed against the default shared database (`data/fgead_multihost.db`)**, registering synthetic test fixtures directly into the operational database.

---

## 2. Complete Database Host Inventory (19 Records)

| # | `host_id` | `hostname` | `operating_system` | `status` | `model_id` | `created_at` | Classification |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `host_siva_windows` | `SivaChowdary-PC` | `Windows` | `ONLINE` | `windows_default` | 2026-10-03 15:00:11 | Historical Seed / Test Fixture |
| 2 | `host_linux_srv01` | `Ubuntu-Prod-Server-01` | `Linux` | `TELEMETRY_ONLY` | `none` | 2026-10-03 15:00:11 | Linux Gating Test Fixture |
| 3 | **`host_sivachowdary`** | **`SivaChowdary`** | **`Windows`** | **`ONLINE`** | **`windows_default`** | **2026-10-03 16:52:03** | **Actual Physical Windows Machine** |
| 4 | `test_host_win_01` | `Test-Win-Workstation` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 14:58:03 | Phase 6.1 Test Fixture |
| 5 | `test_host_lin_01` | `Test-Ubuntu-Server` | `Linux` | `TELEMETRY_ONLY` | `none` | 2026-10-04 14:58:04 | Phase 6.1 Test Fixture |
| 6 | `test_host_unk_01` | `Test-FreeBSD-Machine` | `FreeBSD` | `TELEMETRY_ONLY` | `none` | 2026-10-04 14:58:05 | Phase 6.1 Test Fixture |
| 7 | `test_host_bad_schema` | `Test-Win-OldSchema` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 14:58:05 | Phase 6.1 Test Fixture |
| 8 | `host_819af6a21c0c` | `Dedicated-PC-Chowdary` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 14:58:06 | Phase 6.1 Reconnect Test Fixture |
| 9 | `host_win_anom` | `Win-Anomaly-Host` | `Windows` | `ANOMALY` | `windows_default` | 2026-10-04 15:01:07 | Multi-Host Anomaly Test Fixture |
| 10 | `host_test_node_alpha` | `Test-Node-Alpha` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test A Fixture |
| 11 | `host_test_node_beta` | `Test-Node-Beta` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test B Fixture |
| 12 | `host_test_node_gamma` | `Test-Node-Gamma` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test C Fixture |
| 13 | `host_test_node_delta` | `Test-Node-Delta` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test D Fixture |
| 14 | `host_machine_alpha_server` | `Machine-Alpha-Server` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test E Fixture |
| 15 | `host_machine_beta_server` | `Machine-Beta-Server` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test E Fixture |
| 16 | `host_dual_os_node` | `Dual-OS-Node` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test F (Win) Fixture |
| 17 | `host_dual_os_node_linux` | `Dual-OS-Node` | `Linux` | `TELEMETRY_ONLY` | `none` | 2026-10-04 20:30:09 | Lifecycle Test F (Lin) Fixture |
| 18 | `host_reconnect_node_01` | `Reconnect-Node-01` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test G Fixture |
| 19 | `host_concurrent_node_01` | `Concurrent-Node-01` | `Windows` | `ONLINE` | `windows_default` | 2026-10-04 20:30:09 | Lifecycle Test H Fixture |

### Physical Machine Breakdown:
- **Actual Legitimate Physical Machine:** `1` (`host_sivachowdary`)
- **Historical Seed Machine:** `1` (`host_siva_windows`)
- **Automated Test Suite Fixtures:** `17` (Records 2, 4–19)

---

## 3. Investigation of Specific Potential Causes

| Diagnostic Question | Observed Behavior | Evidence / Verdict |
| :--- | :--- | :--- |
| **1. Is a new UUID generated on every registration?** | **NO.** Canonical deterministic IDs are used (`host_sivachowdary`). | Verified in `test_agent_restarts.py`: 3 cold restarts all produced `host_sivachowdary`. |
| **2. Is auto-register called repeatedly?** | **NO.** The agent only registers once when starting if no local config exists. | Reuses local config `data/windows_agent_config.json` or `live_agent_config.json`. |
| **3. Does reconnect create a new registration?** | **NO.** Reconnects send telemetry with `X-Agent-Token` to `POST /hosts/{host_id}/telemetry` without re-registering. | Verified across 5 consecutive reconnect calls. |
| **4. Does hostname/OS matching fail?** | **NO.** Case-insensitive lookup `LOWER(hostname) = LOWER(?) AND LOWER(operating_system) = LOWER(?)` matches reliably. | `api/host_registry.py` lines 170–178. |
| **5. Does machine identity change between requests?** | **NO.** Hardware MAC identity `0xf46d3f79e072` is invariant across boots. | Recorded in `machine_info_json`. |
| **6. Is the agent registering more than once?** | **NO.** Even if registration is explicitly called, it is idempotent and updates the existing record. | Verified in `test_host_registration_lifecycle.py` Test D. |
| **7. Are multiple agent processes running?** | **NO.** Stale processes were terminated. Only 1 agent instance runs. | Verified via `Get-CimInstance Win32_Process`. |
| **8. Is the dashboard itself creating hosts?** | **NO.** Streamlit frontend only issues `GET` requests (`/hosts`, `/fleet/overview`, `/hosts/{id}/analysis`). | Inspected all `requests.get` calls in `app/streamlit_app.py`. |
| **9. Is frontend polling triggering registration?** | **NO.** Polling only queries `GET /hosts/{host_id}/analysis`. | No `POST /hosts/register` is called by frontend. |
| **10. Does database INSERT happen instead of UPDATE?** | **NO.** Registration uses `INSERT ... ON CONFLICT(host_id) DO UPDATE SET`. | `api/host_registry.py` lines 208–226. |
| **11. Do concurrent registrations create duplicate rows?** | **NO.** Thread-safe `threading.RLock()` and SQLite unique index `idx_hosts_unique_identity` enforce single-row identity. | Verified in `test_host_registration_lifecycle.py` Test H (10 concurrent requests $\to$ 1 host). |
| **12. Are test suites polluting the operational database?** | **YES.** Test files initialize `TestClient(app)` using the operational database `data/fgead_multihost.db`, inserting test fixtures. | Confirmed: records 4–19 correspond exactly to test names in `test_phase6_1_compatibility.py` and `test_host_registration_lifecycle.py`. |

---

## 4. Summary & Conclusions

ROOT CAUSE:
The live agent on the physical machine is NOT duplicating hosts (it consistently resolves to `host_sivachowdary`). The increase in host count in the Fleet Overview is caused by automated test suites running against the shared operational database `data/fgead_multihost.db` instead of an isolated temporary test database or in-memory test database, and the Fleet Overview displaying all test fixture records.

EXPECTED FIX:
1. Configure automated test suites (`test_phase6_1_compatibility.py`, `test_multihost_suite.py`, `test_host_registration_lifecycle.py`, `test_phase9_accuracy.py`) to use an isolated temporary test database (e.g. `data/test_fgead_multihost.db` or `:memory:`) so test fixtures never pollute the operational database.
2. In `app/streamlit_app.py`, provide a clean host filter (or exclude synthetic test fixtures from normal operational view unless "Include Test Fixtures" is toggled).
3. Safely archive/clean up test fixture records from `data/fgead_multihost.db`.

FILES THAT MUST CHANGE:
- `api/host_registry.py` / `config/settings.py` (Support explicit test DB isolation via environment variable / parameter)
- `data/multihost_validation/test_host_registration_lifecycle.py` (Use isolated test database)
- `data/multihost_validation/test_phase6_1_compatibility.py` (Use isolated test database)
- `data/multihost_validation/test_multihost_suite.py` (Use isolated test database)
- `app/streamlit_app.py` (Clean separation of active production hosts vs test fixtures)

DATABASE RECORDS THAT SHOULD NOT BE DELETED:
- `host_sivachowdary` (The active legitimate Windows physical host)
- `host_linux_srv01` (The reference Linux server baseline model assignment)
