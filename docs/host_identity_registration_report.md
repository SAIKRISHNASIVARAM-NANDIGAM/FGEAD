# FGEAD Host Registration & Identity Lifecycle Audit Report

## 1. Executive Summary

This report provides the full architectural diagnosis and remediation of the host identity and auto-registration lifecycle in the FGEAD platform.

Previously, running agents or test suites without an explicit `custom_host_id` generated random hexadecimal suffixes (`f"host_{secrets.token_hex(6)}"`) whenever exact case-sensitive matching failed, resulting in host record proliferation in the SQLite database and Fleet Overview selector.

With the new multi-tier deterministic identity matching engine and SQLite unique identity index, **one physical machine strictly corresponds to one stable, persistent `host_id` across restarts, network disconnects, and repeated registrations.**

---

## 2. Root Cause Analysis

### A. Fallback to Random Hex Identifiers
- In `api/host_registry.py` -> `register_host()`, when a host registered without `custom_host_id` or when lookup failed, the system fell back to:
  ```python
  host_id = custom_host_id or existing_host_id or f"host_{secrets.token_hex(6)}"
  ```
- If an agent restarted with a cleared or missing local config file, a brand new `host_<hex>` was generated on every restart.

### B. Case-Sensitive Exact Match Failure
- SQLite query `WHERE hostname = ? AND operating_system = ?` was case-sensitive. Variations between `"Windows"` vs `"windows"` or `"SivaChowdary"` vs `"sivachowdary"` caused lookups to miss existing rows and trigger new random ID generation.

### C. Missing Hardware Identity Anchor
- Agents were not transmitting stable hardware UUIDs (`uuid.getnode()`), preventing unambiguous physical hardware identification when hostnames were modified.

### D. Missing Database Uniqueness Constraint
- The `hosts` table only enforced primary key uniqueness on `host_id`. There was no database-level unique index on `(LOWER(hostname), LOWER(operating_system))`, permitting multiple records for the same physical host.

---

## 3. Files & Functions Responsible

1. [`api/host_registry.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/api/host_registry.py):
   - `HostRegistry._init_db()`: Added SQLite partial unique index `idx_hosts_unique_identity`.
   - `HostRegistry.register_host()`: Implemented 3-tier deterministic identity matching, canonical ID synthesis, and idempotent updates.
2. [`agents/windows_agent.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/agents/windows_agent.py):
   - `WindowsTelemetryCollector.get_machine_info()`: Added `machine_id` and hardware MAC node `uuid.getnode()`.
3. [`agents/linux_agent.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/agents/linux_agent.py):
   - `LinuxTelemetryCollector.get_machine_info()`: Added `machine_id` and hardware MAC node `uuid.getnode()`.
4. [`data/live_agent.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/live_agent.py):
   - `LiveTelemetryCollector.get_machine_info()`: Added `machine_id` and hardware MAC node `uuid.getnode()`.

---

## 4. Identity Algorithm Comparison

### Before (Legacy):
1. If `custom_host_id` is supplied, query `WHERE host_id = ?`.
2. Otherwise, query `WHERE hostname = ? AND operating_system = ?` (case-sensitive).
3. If not found, generate `f"host_{secrets.token_hex(6)}"`.
4. Result: Proliferation of ephemeral host records across restarts.

### After (Deterministic & Robust):
```
[Agent Registration Request]
       │
       ├─► Tier 1: Explicit custom_host_id lookup
       │     `SELECT * FROM hosts WHERE host_id = ?`
       │
       ├─► Tier 2: Machine Hardware UUID Lookup
       │     `SELECT * FROM hosts WHERE json_extract(machine_info_json, '$.machine_id') = ? AND LOWER(operating_system) = LOWER(?)`
       │
       ├─► Tier 3: Case-Insensitive (Hostname, OS) Lookup
       │     `SELECT * FROM hosts WHERE LOWER(hostname) = LOWER(?) AND LOWER(operating_system) = LOWER(?)`
       │
       ▼
[Match Found?]
  ├─ YES: REUSE existing host_id and agent_id; update status, last_seen, and token hash.
  └─ NO : Synthesize stable canonical ID: `host_{clean_hostname}` (e.g. `host_sivachowdary`).
       │  Disambiguate with OS if collision occurs: `host_{clean_hostname}_{clean_os}`.
       ▼
[Idempotent Database Upsert]
  `INSERT INTO hosts (...) ON CONFLICT(host_id) DO UPDATE SET ...`
  Protected by unique index: `idx_hosts_unique_identity ON (LOWER(hostname), LOWER(operating_system))`
```

---

## 5. Host Inventory Audit

| Category | Count | Host IDs |
| :--- | :---: | :--- |
| **Legitimate Active Physical Hosts** | 2 | `host_sivachowdary` (Windows Physical Host), `host_linux_srv01` (Linux Production Server) |
| **Historical Seed Hosts** | 1 | `host_siva_windows` (Initial Windows Baseline Seed) |
| **Validation & Test Fixtures** | 16 | `test_host_win_01`, `test_host_lin_01`, `test_host_unk_01`, `test_host_bad_schema`, `host_win_anom`, `host_819af6a21c0c`, `host_test_node_alpha`, `host_test_node_beta`, `host_test_node_gamma`, `host_test_node_delta`, `host_machine_alpha_server`, `host_machine_beta_server`, `host_dual_os_node`, `host_dual_os_node_linux`, `host_reconnect_node_01`, `host_concurrent_node_01` |
| **Total Registered Records** | **19** | All preserved safely without data corruption |

---

## 6. Live Agent 3-Restart & Reconnect Verification

We executed a rigorous cold-restart test suite ([`scratch/test_agent_restarts.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/scratch/test_agent_restarts.py)) simulating 3 successive agent restarts where local cached config files were wiped between runs, followed by 5 live reconnect streams:

- **Restart 1/3 (Cold restart, config deleted):** Resulting `host_id` = **`host_sivachowdary`** (Status 200)
- **Restart 2/3 (Cold restart, config deleted):** Resulting `host_id` = **`host_sivachowdary`** (Status 200)
- **Restart 3/3 (Cold restart, config deleted):** Resulting `host_id` = **`host_sivachowdary`** (Status 200)
- **Reconnect Tests (5 warm reconnect transmissions):** All returned Status 200 to `host_sivachowdary`.
- **Host Count Stability:** Count before = **`19`**, Count after = **`19`** (**0 duplicate hosts created**).

---

## 7. Test Suite Validation (64 / 64 Checks Passed)

| Suite File | Checks | Status |
| :--- | :---: | :---: |
| [`data/multihost_validation/test_host_registration_lifecycle.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_host_registration_lifecycle.py) | **8 / 8** | **PASS** |
| [`data/multihost_validation/test_phase9_accuracy.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_phase9_accuracy.py) | **18 / 18** | **PASS** |
| [`data/multihost_validation/test_cloud_deployment_phase8.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_cloud_deployment_phase8.py) | **4 / 4** | **PASS** |
| [`data/multihost_validation/test_smoke_phase7.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_smoke_phase7.py) | **7 / 7** | **PASS** |
| [`data/multihost_validation/test_phase6_1_compatibility.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_phase6_1_compatibility.py) | **10 / 10** | **PASS** |
| [`data/multihost_validation/test_multihost_suite.py`](file:///c:/Users/saikr/Desktop/FGEAD-main/data/multihost_validation/test_multihost_suite.py) | **17 / 17** | **PASS** |
| **Total Automated Validation** | **64 / 64** | **100% PASS** |
