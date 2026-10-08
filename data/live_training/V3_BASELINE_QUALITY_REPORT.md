# FGEAD V3 Baseline Quality Report
**Status:** 🟢 PASSED
**Total Telemetry Samples:** 1800 / 1800 expected

## 1. Quality & Integrity Checks
- **NaN Count:** 0
- **Inf Count:** 0
- **Duplicate Timestamps:** 0
- **Average Sampling Interval:** 0.9998s (std: 0.0131s)
- **Sampling Interval Range:** min = 0.5745s, max = 1.0254s
- **Max Sampling Gap Check:** 🟢 PASSED (<= 5.0s)

## 2. Chronological Dataset Split (No Shuffling)
- **Train Normal (70%):** 1260 samples (`v3_train_normal.csv`)
- **Calibration Normal (15%):** 270 samples (`v3_cal_normal.csv`)
- **Held-Out Normal Test (15%):** 270 samples (`v3_test_normal.csv`)

## 3. Per-Feature Statistics
| Feature Name | Mean | Std | Min | Max | P95 | P99 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `cpu_percent` | 10.8866 | 6.4558 | 0.3 | 73.1 | 17.3 | 42.034 |
| `cpu_freq_current` | 1969.0 | 0.0 | 1969.0 | 1969.0 | 1969.0 | 1969.0 |
| `cpu_user_time_percent` | 7.0935 | 4.1934 | 0.1 | 53.6 | 10.405 | 29.909 |
| `cpu_system_time_percent` | 3.2076 | 2.4397 | 0.2 | 24.6 | 6.405 | 14.201 |
| `cpu_ctx_switches_per_sec` | 20534.7253 | 15550.4207 | 3853.7 | 93920.4 | 42243.77 | 55046.049 |
| `cpu_interrupts_per_sec` | 14282.2442 | 9674.7374 | 2903.6 | 48996.4 | 28493.375 | 33604.444 |
| `memory_percent` | 45.8626 | 0.834 | 44.9 | 53.2 | 46.4 | 50.801 |
| `memory_available_mb` | 17599.8176 | 270.2941 | 15210.5 | 17914.2 | 17897.905 | 17911.3 |
| `memory_used_mb` | 14909.6936 | 270.2944 | 14595.3 | 17299.0 | 15092.8 | 16502.495 |
| `swap_percent` | 1.4 | 0.0 | 1.4 | 1.4 | 1.4 | 1.4 |
| `disk_usage_percent` | 54.3 | 0.0 | 54.3 | 54.3 | 54.3 | 54.3 |
| `disk_read_bytes_per_sec` | 1213689.7275 | 11199458.1414 | 0.0 | 239320812.1 | 386830.165 | 29184544.03 |
| `disk_write_bytes_per_sec` | 307939.9554 | 1424278.2409 | 0.0 | 25895164.7 | 664000.485 | 4313117.002 |
| `disk_read_count_per_sec` | 16.8593 | 122.5322 | 0.0 | 2451.7 | 13.9 | 484.603 |
| `disk_write_count_per_sec` | 18.84 | 83.033 | 0.0 | 2415.7 | 62.535 | 227.208 |
| `net_bytes_sent_per_sec` | 34348.5946 | 191329.6383 | 82.9 | 2721369.2 | 96492.74 | 1016477.182 |
| `net_bytes_recv_per_sec` | 8938.7902 | 43174.1247 | 87.1 | 1473681.6 | 34991.365 | 113097.584 |
| `net_packets_sent_per_sec` | 41.5719 | 157.5934 | 1.0 | 2107.5 | 118.15 | 889.44 |
| `net_packets_recv_per_sec` | 46.7279 | 112.3301 | 1.0 | 1377.1 | 192.505 | 527.01 |
| `net_errors_total` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `net_drops_total` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `process_count` | 324.8106 | 3.9508 | 320.0 | 368.0 | 330.0 | 342.0 |

## 4. Constant Channel Assessment
- **Constant Channels:** ['cpu_freq_current', 'swap_percent', 'net_errors_total', 'net_drops_total']
- **Accepted Invariant Baseline Channels:** ['cpu_freq_current', 'swap_percent', 'net_errors_total', 'net_drops_total']
- **Unexplained Constant Channels:** None
- **Validation Verdict:** 🟢 Baseline telemetry is pristine, continuous, non-synthetic, and PASSED quality gate.
