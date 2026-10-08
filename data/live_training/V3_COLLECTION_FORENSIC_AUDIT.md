# V3 Baseline Collection Forensic Audit Report

**Date:** October 7, 2026  
**Target File:** `data/live_baseline_v3_current_machine.csv`  
**Host Environment:** Windows Physical Host (SivaChowdary Laptop)  

---

## 1. Executive Summary & Audit Verdict

A forensic investigation was conducted into why the V3 baseline collection was flagged as `status: FAILED` with 1,160 samples saved instead of 1,800 expected samples, an 8.2966-second sampling gap, and zero variance in `cpu_freq_current` and `net_drops_total`.

### Key Findings:
1. **Partial File Reading during Background Execution:** The quality check was performed while `data/collect_v3_baseline.py` was actively streaming samples in the background. The file contained 1,160 rows at the moment it was inspected.
2. **PyTorch Test Execution CPU Contention (8.3s Gap):** The 8.2966-second gap between `18:06:18.845` and `18:06:27.141` UTC occurred when `test_multihost_suite.py` was launched concurrently, causing Python GIL locks and hardware I/O scheduling delays in `psutil`.
3. **Hardware & OS Counter Behavior:**
   - `cpu_freq_current`: `psutil.cpu_freq()` returns a static `1969.0 MHz` on this Windows ACPI driver interface under non-elevated user queries.
   - `net_drops_total`: Represents cumulative dropped packets (`dropin + dropout`), which is `0` when network health is pristine.
4. **Action Required:** The existing 1,160-sample file must **not** be used for training. A fresh, uninterrupted 1,800-sample (30-minute) baseline must be recollected after applying sampling drift compensation.

---

## 2. Forensic Investigation & Questions Addressed

### Q1: Why were only 1,160 samples saved when 1,800 were requested?
- **Root Cause:** The collection script `data/collect_v3_baseline.py` streams and flushes rows incrementally to disk every 10 samples. The quality report script `validate_v3_baseline.py` was run while the 30-minute background collection process was still in progress (at sample #1160 of 1800).
- **Verification:** Inspecting `task-65` process logs confirms the task remained `RUNNING` and reached sample #1290+ while the audit was initiated.

### Q2: Are the existing 1,160 samples usable for V3 model training?
- **Verdict:** **NO.**
- **Rationale:** 
  1. The dataset is incomplete (1,160 samples = 19.3 minutes vs. 1,800 samples = 30 minutes required).
  2. The dataset contains a severe 8.2966-second sampling gap caused by concurrent test execution, violating the 1.0-second time-series continuity requirement.

### Q3: What caused the 8.2966-second sampling gap?
- **Timestamp Analysis:**
  - Pre-gap sample: `2026-10-07T18:06:18.845379+00:00` (Index 670)
  - Post-gap sample: `2026-10-07T18:06:27.141931+00:00` (Index 671)
  - Delta: `8.296552` seconds
- **Process Activity Correlation:** At `18:06:18 UTC` (`23:36:18` local time), `venv\Scripts\python.exe data/multihost_validation/test_multihost_suite.py` was executed. The concurrent PyTorch model loading, matrix operations, and FastAPI TestClient requests created GIL lock contention and process preemption, delaying `psutil.disk_io_counters()` and `psutil.cpu_percent()` hardware calls.

### Q4: Assessment of `cpu_freq_current` (Constant Value: 1969.0 MHz)
- **Empirical Check:** Calling `psutil.cpu_freq()` directly returns `scpufreq(current=1969.0, min=0.0, max=2600.0)`.
- **System Diagnosis:** On Windows 10/11 platforms, `psutil` queries static WMI/ACPI performance counters unless high-resolution kernel frequency drivers are installed. A constant value of `1969.0 MHz` is normal, expected behavior for `psutil` on this physical host under standard operation.

### Q5: Assessment of `net_drops_total` (Constant Value: 0.0)
- **Empirical Check:** Calling `psutil.net_io_counters()` directly returns `dropin=0, dropout=0`.
- **System Diagnosis:** `net_drops_total` is a legitimate **cumulative network counter**. Zero value indicates zero dropped network packets during the monitoring window. Zero variance does **not** indicate a schema error, and the 22-feature schema must be strictly preserved.

---

## 3. Exact Fix Required for Baseline Collection

1. **Absolute Time Target Sampling Loop:**  
   Update `data/collect_v3_baseline.py` to use absolute time target scheduling (`target_time = t_start_epoch + i * interval_sec`) to prevent sleep drift accumulation:
   ```python
   next_target = t_start_epoch + i * interval_sec
   sleep_dur = max(0.001, next_target - time.time())
   time.sleep(sleep_dur)
   ```
2. **Process Isolation Guarantee:**  
   Ensure no background PyTorch model training, test suites, or heavy disk benchmark scripts are run concurrently during baseline collection.
3. **Quality Validation Rule Update:**  
   Update `validate_v3_baseline.py` so invariant cumulative counters (`net_drops_total`, `net_errors_total`) and static hardware clock readings (`cpu_freq_current`) are classified as expected baseline constants rather than quality failures.

---

## 4. Re-Collection Requirement

- **Fresh Baseline Required:** **YES.**
- **Plan:** Stop current collection task, apply sampling loop drift fix, and execute a fresh, uninterrupted 1,800-sample (30-minute) baseline collection.
