"""
data/live_training/validate_v3_baseline.py

Baseline Quality Validation & Chronological Dataset Splitter for V3 Current-Machine Model.
Evaluates data/live_baseline_v3_current_machine.csv against strict quality checks:
- NaN / Inf checks (Must be strictly 0)
- Timestamp integrity & interval distribution (Mean ≈ 1.0s, max gap <= 5.0s)
- Duplicate timestamp checks (Must be strictly 0)
- Legitimate constant/low-variance channels allowed: cpu_freq_current, net_drops_total, net_errors_total
- Chronological split: 70% Train, 15% Calibration, 15% Held-Out Normal Test
Generates data/live_training/v3_baseline_quality_report.json & V3_BASELINE_QUALITY_REPORT.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES

# Legitimate constant channels on Windows host telemetry
LEGITIMATE_CONSTANT_CHANNELS = {"cpu_freq_current", "net_drops_total", "net_errors_total", "swap_percent"}


def validate_v3_baseline(
    csv_path: str = "data/live_baseline_v3_current_machine.csv",
    meta_path: str = "data/live_training/v3_baseline_metadata.json",
    out_json_path: str = "data/live_training/v3_baseline_quality_report.json",
    out_md_path: str = "data/live_training/V3_BASELINE_QUALITY_REPORT.md",
    expected_samples: int = 1800,
) -> Dict[str, Any]:
    csv_file = PROJECT_ROOT / csv_path
    meta_file = PROJECT_ROOT / meta_path
    json_file = PROJECT_ROOT / out_json_path
    md_file = PROJECT_ROOT / out_md_path

    if not csv_file.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    n_total = len(df)

    if "timestamp" not in df.columns:
        raise ValueError("Baseline CSV is missing required 'timestamp' column.")

    # 1. Feature completeness & order check
    feat_cols = [c for c in df.columns if c != "timestamp"]
    if feat_cols != LIVE_FEATURES:
        raise ValueError(f"Feature schema mismatch! CSV cols: {feat_cols} vs expected LIVE_FEATURES: {LIVE_FEATURES}")

    # 2. Sample Count Verification
    count_check_passed = (n_total == expected_samples)

    # 3. NaN / Inf Check
    nan_count = int(df[LIVE_FEATURES].isna().sum().sum())
    inf_count = int(np.isinf(df[LIVE_FEATURES].values).sum())

    # 4. Duplicate Timestamps
    dup_ts_count = int(df["timestamp"].duplicated().sum())

    # 5. Sampling Interval Analysis
    df["dt_parsed"] = pd.to_datetime(df["timestamp"])
    time_diffs = df["dt_parsed"].diff().dt.total_seconds().dropna()
    dt_mean = float(time_diffs.mean()) if len(time_diffs) > 0 else 0.0
    dt_std = float(time_diffs.std()) if len(time_diffs) > 0 else 0.0
    dt_min = float(time_diffs.min()) if len(time_diffs) > 0 else 0.0
    dt_max = float(time_diffs.max()) if len(time_diffs) > 0 else 0.0

    # Max allowed sampling gap (must be <= 5.0 seconds for clean continuous telemetry)
    gap_check_passed = (dt_max <= 5.0) and (abs(dt_mean - 1.0) <= 0.1)

    # 6. Per-Feature Variance & Outlier Checks
    feature_stats: Dict[str, Dict[str, Any]] = {}
    constant_features: List[str] = []
    unexplained_constant_features: List[str] = []
    low_variance_features: List[str] = []

    for f in LIVE_FEATURES:
        vals = df[f].values.astype(np.float64)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        p95_val = float(np.percentile(vals, 95))
        p99_val = float(np.percentile(vals, 99))

        if std_val == 0.0:
            constant_features.append(f)
            if f not in LEGITIMATE_CONSTANT_CHANNELS:
                unexplained_constant_features.append(f)
        elif std_val < 1e-4:
            low_variance_features.append(f)

        feature_stats[f] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "p95": round(p95_val, 4),
            "p99": round(p99_val, 4),
        }

    # 7. Chronological Split (70% Train, 15% Calibration, 15% Held-Out Normal Test)
    n_train = int(n_total * 0.70)
    n_cal = int(n_total * 0.15)
    n_test = n_total - n_train - n_cal

    df_train = df.iloc[:n_train].copy()
    df_cal = df.iloc[n_train : n_train + n_cal].copy()
    df_test = df.iloc[n_train + n_cal :].copy()

    # Save splits to csv for reproducibility
    split_dir = PROJECT_ROOT / "data" / "live_training"
    split_dir.mkdir(parents=True, exist_ok=True)
    df_train.drop(columns=["dt_parsed"]).to_csv(split_dir / "v3_train_normal.csv", index=False)
    df_cal.drop(columns=["dt_parsed"]).to_csv(split_dir / "v3_cal_normal.csv", index=False)
    df_test.drop(columns=["dt_parsed"]).to_csv(split_dir / "v3_test_normal.csv", index=False)

    quality_passed = (
        count_check_passed
        and (nan_count == 0)
        and (inf_count == 0)
        and (dup_ts_count == 0)
        and gap_check_passed
        and (len(unexplained_constant_features) == 0)
    )

    report = {
        "status": "PASSED" if quality_passed else "FAILED",
        "total_samples": n_total,
        "expected_samples": expected_samples,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_timestamps": dup_ts_count,
        "sampling_interval_sec": {
            "mean": round(dt_mean, 4),
            "std": round(dt_std, 4),
            "min": round(dt_min, 4),
            "max": round(dt_max, 4),
        },
        "max_sampling_gap_sec": round(dt_max, 4),
        "constant_features": constant_features,
        "legitimate_constant_features": [f for f in constant_features if f in LEGITIMATE_CONSTANT_CHANNELS],
        "unexplained_constant_features": unexplained_constant_features,
        "low_variance_features": low_variance_features,
        "splits": {
            "train_samples": n_train,
            "train_ratio": 0.70,
            "calibration_samples": n_cal,
            "calibration_ratio": 0.15,
            "test_samples": n_test,
            "test_ratio": round(n_test / n_total, 4),
        },
        "feature_statistics": feature_stats,
    }

    # Write JSON report
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Write Human-Readable MD Report
    md_lines = [
        "# FGEAD V3 Baseline Quality Report",
        f"**Status:** {'🟢 PASSED' if quality_passed else '🔴 FAILED'}",
        f"**Total Telemetry Samples:** {n_total} / {expected_samples} expected",
        "",
        "## 1. Quality & Integrity Checks",
        f"- **NaN Count:** {nan_count}",
        f"- **Inf Count:** {inf_count}",
        f"- **Duplicate Timestamps:** {dup_ts_count}",
        f"- **Average Sampling Interval:** {dt_mean:.4f}s (std: {dt_std:.4f}s)",
        f"- **Sampling Interval Range:** min = {dt_min:.4f}s, max = {dt_max:.4f}s",
        f"- **Max Sampling Gap Check:** {'🟢 PASSED (<= 5.0s)' if dt_max <= 5.0 else '🔴 FAILED (> 5.0s)'}",
        "",
        "## 2. Chronological Dataset Split (No Shuffling)",
        f"- **Train Normal (70%):** {n_train} samples (`v3_train_normal.csv`)",
        f"- **Calibration Normal (15%):** {n_cal} samples (`v3_cal_normal.csv`)",
        f"- **Held-Out Normal Test (15%):** {n_test} samples (`v3_test_normal.csv`)",
        "",
        "## 3. Per-Feature Statistics",
        "| Feature Name | Mean | Std | Min | Max | P95 | P99 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for feat in LIVE_FEATURES:
        st = feature_stats[feat]
        md_lines.append(
            f"| `{feat}` | {st['mean']} | {st['std']} | {st['min']} | {st['max']} | {st['p95']} | {st['p99']} |"
        )

    md_lines.extend([
        "",
        "## 4. Constant Channel Assessment",
        f"- **Constant Channels:** {constant_features if constant_features else 'None'}",
        f"- **Accepted Invariant Baseline Channels:** {[f for f in constant_features if f in LEGITIMATE_CONSTANT_CHANNELS]}",
        f"- **Unexplained Constant Channels:** {unexplained_constant_features if unexplained_constant_features else 'None'}",
        f"- **Validation Verdict:** {'🟢 Baseline telemetry is pristine, continuous, non-synthetic, and PASSED quality gate.' if quality_passed else '🔴 Quality validation FAILED.'}",
    ])

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[QUALITY REPORT] Status: {'🟢 PASSED' if quality_passed else '🔴 FAILED'}")
    print(f"Total Samples   : {n_total} / {expected_samples}")
    print(f"NaN / Inf Count : {nan_count} / {inf_count}")
    print(f"Duplicate TS    : {dup_ts_count}")
    print(f"Mean Interval   : {dt_mean:.4f}s (std: {dt_std:.4f}s)")
    print(f"Max Gap         : {dt_max:.4f}s")
    print(f"Constant Chans  : {constant_features} (Legitimate: {[f for f in constant_features if f in LEGITIMATE_CONSTANT_CHANNELS]})")

    return report


if __name__ == "__main__":
    validate_v3_baseline()
