# V3 Baseline Collection & Post-Collection Validation Final Report

**Date:** October 8, 2026  
**Host Environment:** Windows Physical Host (SivaChowdary Laptop)  
**Output Baseline CSV:** `data/live_baseline_v3_current_machine.csv`  
**Output Metadata JSON:** `data/live_training/v3_baseline_metadata.json`  
**Quality Report JSON:** `data/live_training/v3_baseline_quality_report.json`  

---

## 1. Executive Summary & Final Verdict

The fresh 1,800-sample V3 live baseline collection was successfully executed under complete Windows sleep-prevention protection and monotonic absolute target scheduling. 

### Final Quality Validation Verdict: 🟢 **PASSED QUALITY GATE**

---

## 2. Comprehensive Collection & Quality Metrics

| Parameter / Metric | Target / Requirement | Measured Value | Validation Status |
| :--- | :---: | :---: | :---: |
| **Collection Start Time** | — | `2026-10-08T02:38:06.015Z` | Recorded |
| **Collection End Time** | — | `2026-10-08T03:08:07.045Z` | Recorded |
| **Requested Samples** | 1,800 samples | 1,800 samples | 🟢 PASSED |
| **Collected Samples** | Exactly 1,800 samples | **1,800 samples** | 🟢 PASSED |
| **NaN Count** | 0 | **0** | 🟢 PASSED |
| **Inf Count** | 0 | **0** | 🟢 PASSED |
| **Duplicate Timestamps** | 0 | **0** | 🟢 PASSED |
| **Mean Sampling Interval** | $\approx 1.0\text{s}$ | **0.9998 seconds** (std: 0.0131s) | 🟢 PASSED |
| **Maximum Sampling Gap** | $\le 5.0\text{s}$ | **1.0254 seconds** | 🟢 PASSED |
| **Sleep/Resume Interruptions** | 0 interruptions | **0 interruptions** | 🟢 PASSED |
| **Sleep Prevention** | Enabled | 🟢 **ENABLED (Win32 SetThreadExecutionState)** | 🟢 PASSED |
| **Feature Schema** | Exact 22 Features | Exact 22 Features | 🟢 PASSED |
| **Accepted Constant Channels** | Invariant OS channels | `cpu_freq_current`, `swap_percent`, `net_errors_total`, `net_drops_total` | 🟢 PASSED |
| **V3 Model Training** | **NOT STARTED** | **INTENTIONALLY NOT STARTED** | 🟢 ENFORCED |

---

## 3. Implementation Verification Summary

1. **Sleep & Suspend Protection:**  
   Windows API `SetThreadExecutionState` (`ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED`) successfully prevented system and display sleep for the entire 30.0-minute collection duration.
2. **Monotonic Target Scheduling:**  
   Sub-second monotonic timer (`time.monotonic()`) eliminated sleep drift and catch-up bursts, yielding a mean interval of **0.9998s** and maximum gap of **1.0254s**.
3. **Chronological Data Splits:**  
   The baseline was split chronologically without shuffling:
   - **Train Normal (70%):** 1,260 samples (`data/live_training/v3_train_normal.csv`)
   - **Calibration Normal (15%):** 270 samples (`data/live_training/v3_cal_normal.csv`)
   - **Held-Out Normal Test (15%):** 270 samples (`data/live_training/v3_test_normal.csv`)

---

## 4. Final Directive Compliance

- **No data was silently manufactured or repaired.**
- **The 30-sample pre-test and 1,800-sample real run completed cleanly.**
- **V3 Model Training was NOT started (stopped after validation as required).**
