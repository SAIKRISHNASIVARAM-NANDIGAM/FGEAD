"""
data/collect_baseline.py

FGEAD Real Physical Windows Normal Baseline Telemetry Collector.
Collects continuous real hardware metrics from Windows host to build
a clean, uncorrupted normal operational baseline dataset for training
the dedicated 22-feature FGEAD graph model.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.live_agent import LiveTelemetryCollector
from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    LIVE_FEATURES,
    LIVE_FEATURE_VERSION,
    N_LIVE_FEATURES,
    validate_live_feature_dict,
)

CSV_COLUMNS: List[str] = ["timestamp", "machine_id"] + LIVE_FEATURES


def validate_collected_dataset(csv_path: Path, expected_duration_sec: float, interval_sec: float) -> Dict[str, Any]:
    """
    Perform comprehensive post-collection quality and statistical validation.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline file does not exist: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Column and dimension verification
    actual_cols = list(df.columns)
    missing_cols = [c for c in CSV_COLUMNS if c not in actual_cols]
    extra_cols = [c for c in actual_cols if c not in CSV_COLUMNS]

    # 2. Check rows
    n_rows = len(df)
    expected_rows = int(expected_duration_sec / interval_sec)

    # 3. Quality checks
    nan_counts = df[LIVE_FEATURES].isna().sum().to_dict()
    total_nans = int(df[LIVE_FEATURES].isna().sum().sum())
    total_infs = int(np.isinf(df[LIVE_FEATURES].to_numpy()).sum())

    # 4. Duplicate checks
    duplicate_rows = int(df.duplicated(subset=["timestamp"]).sum())

    # 5. Timestamp analysis
    try:
        ts_series = pd.to_datetime(df["timestamp"])
        is_monotonic = bool(ts_series.is_monotonic_increasing)
        deltas = ts_series.diff().dt.total_seconds().dropna()
        mean_dt = float(deltas.mean()) if len(deltas) > 0 else 0.0
        min_dt = float(deltas.min()) if len(deltas) > 0 else 0.0
        max_dt = float(deltas.max()) if len(deltas) > 0 else 0.0
        std_dt = float(deltas.std()) if len(deltas) > 0 else 0.0
    except Exception as e:
        is_monotonic = False
        mean_dt, min_dt, max_dt, std_dt = 0.0, 0.0, 0.0, 0.0

    # 6. Feature statistics
    stats: Dict[str, Dict[str, float]] = {}
    for feat in LIVE_FEATURES:
        if feat in df.columns:
            series = df[feat].dropna()
            stats[feat] = {
                "min": float(series.min()) if len(series) > 0 else 0.0,
                "max": float(series.max()) if len(series) > 0 else 0.0,
                "mean": float(series.mean()) if len(series) > 0 else 0.0,
                "std": float(series.std()) if len(series) > 0 else 0.0,
            }

    # Validation criteria:
    # - No missing required columns
    # - At least 1 row
    # - 0 NaNs and 0 Infs
    # - Monotonic timestamps
    # - 0 Duplicate timestamps
    is_valid = (
        len(missing_cols) == 0
        and n_rows > 0
        and total_nans == 0
        and total_infs == 0
        and duplicate_rows == 0
        and is_monotonic
    )

    return {
        "is_valid": is_valid,
        "n_rows": n_rows,
        "n_features": len(LIVE_FEATURES),
        "expected_rows": expected_rows,
        "missing_cols": missing_cols,
        "extra_cols": extra_cols,
        "total_nans": total_nans,
        "total_infs": total_infs,
        "duplicate_rows": duplicate_rows,
        "is_monotonic_timestamp": is_monotonic,
        "mean_interval_sec": mean_dt,
        "min_interval_sec": min_dt,
        "max_interval_sec": max_dt,
        "std_interval_sec": std_dt,
        "feature_stats": stats,
    }


def print_validation_report(results: Dict[str, Any], csv_path: Path):
    """Print structured, human-readable statistical validation report."""
    print()
    print("=" * 80)
    print("FGEAD BASELINE DATASET QUALITY & VALIDATION REPORT")
    print("=" * 80)
    print(f"File Path           : {csv_path.resolve()}")
    print(f"Total Samples (Rows): {results['n_rows']:,}")
    print(f"Expected Samples    : {results['expected_rows']:,}")
    print(f"Features Recorded   : {results['n_features']} channels")
    print(f"Missing Columns     : {results['missing_cols'] if results['missing_cols'] else 'None (100% Complete)'}")
    print(f"NaN / Null Values   : {results['total_nans']}")
    print(f"Infinite Values     : {results['total_infs']}")
    print(f"Duplicate Rows      : {results['duplicate_rows']}")
    print(f"Ordered Timestamps  : {'PASS (Monotonically Increasing)' if results['is_monotonic_timestamp'] else 'FAIL'}")
    print(f"Mean Sampling Rate  : {results['mean_interval_sec']:.2f}s (Min: {results['min_interval_sec']:.2f}s, Max: {results['max_interval_sec']:.2f}s, Std: {results['std_interval_sec']:.3f}s)")
    print("-" * 80)
    print(f"{'INDEX':<5} | {'FEATURE NAME':<26} | {'MIN':<9} | {'MAX':<10} | {'MEAN':<10} | {'STD':<10}")
    print("-" * 80)

    for idx, (feat, st) in enumerate(results["feature_stats"].items(), start=1):
        print(
            f"{idx:<5} | {feat:<26} | {st['min']:<9.2f} | {st['max']:<10.2f} | {st['mean']:<10.2f} | {st['std']:<10.2f}"
        )

    print("=" * 80)
    if results["is_valid"]:
        print("✅ VALIDATION STATUS: PASSED")
        print("   The collected dataset is clean, uncorrupted, and 100% ready for FGEAD baseline model training.")
    else:
        print("❌ VALIDATION STATUS: FAILED")
        print("   One or more validation constraints failed. Please inspect errors above.")
    print("=" * 80)
    print()


def collect_baseline(
    duration_sec: int = 3600,
    interval_sec: float = 1.0,
    output_path: str = "data/live_baseline.csv",
    append_mode: bool = False,
):
    """
    Collect continuous normal Windows telemetry snapshots and write directly to CSV.
    """
    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    collector = LiveTelemetryCollector()
    info = collector.get_machine_info()
    machine_id = info["machine_id"]

    total_expected = int(duration_sec / interval_sec)

    print()
    print("=" * 76)
    print(" [FGEAD BASELINE COLLECTOR] — REAL WINDOWS HARDWARE TELEMETRY")
    print("=" * 76)
    print(f" Machine Host ID  : {machine_id}")
    print(f" Operating System : {info['platform']} {info['os_version']}")
    print(f" CPU Cores        : {info['cpu_logical_cores']} Logical ({info['cpu_physical_cores']} Physical)")
    print(f" Target Duration  : {duration_sec:,} seconds ({duration_sec / 60.0:.1f} minutes)")
    print(f" Sampling Rate    : Every {interval_sec:.1f}s (Expected: {total_expected:,} samples)")
    print(f" Output CSV File  : {out_file}")
    print(f" Feature Schema   : v{info['schema_version']} ({len(LIVE_FEATURES)} physical metrics)")
    print("=" * 76)
    print()

    # Prime differentials
    print("Priming hardware differential counters (1.0s)...")
    time.sleep(1.0)
    print("Starting continuous live collection. Press Ctrl+C at any time to halt and save.\n")

    file_exists = out_file.exists() and not append_mode
    write_header = not (append_mode and out_file.exists() and out_file.stat().st_size > 0)

    # Open CSV in write/append mode with immediate flush
    file_mode = "a" if append_mode else "w"
    csv_f = open(out_file, mode=file_mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)

    if write_header:
        writer.writeheader()
        csv_f.flush()

    collected_count = 0
    invalid_count = 0
    t_start = time.time()
    t_next = t_start

    try:
        while True:
            t_now = time.time()
            elapsed = t_now - t_start
            if elapsed >= duration_sec:
                break

            try:
                features = collector.collect_features()
                iso_ts = datetime.now(timezone.utc).isoformat()

                # Validate feature values
                is_valid, err_msg = validate_live_feature_dict(features)
                if not is_valid:
                    invalid_count += 1
                    print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] Invalid sample skipped: {err_msg}")
                    continue

                row = {
                    "timestamp": iso_ts,
                    "machine_id": machine_id,
                    **features,
                }

                writer.writerow(row)
                collected_count += 1

                # Flush every 5 samples to guarantee persistence
                if collected_count % 5 == 0:
                    csv_f.flush()

                # Display real-time progress
                pct = min(100.0, (elapsed / duration_sec) * 100.0)
                net_kb = (features["net_bytes_sent_per_sec"] + features["net_bytes_recv_per_sec"]) / 1024.0
                disk_io_kb = (features["disk_read_bytes_per_sec"] + features["disk_write_bytes_per_sec"]) / 1024.0

                print(
                    f"\r[{pct:5.1f}%] Sample {collected_count:04d}/{total_expected:04d} | "
                    f"CPU: {features['cpu_percent']:5.1f}% | "
                    f"RAM: {features['memory_percent']:5.1f}% ({features['memory_used_mb']:5.0f}MB) | "
                    f"Disk: {disk_io_kb:6.1f} KB/s | "
                    f"Net: {net_kb:6.1f} KB/s | "
                    f"Elapsed: {int(elapsed)}s/{duration_sec}s",
                    end="",
                    flush=True,
                )

            except Exception as e:
                invalid_count += 1
                print(f"\n⚠️ Collection warning at t={int(elapsed)}s: {e}")

            # Precise timing interval
            t_next += interval_sec
            sleep_time = max(0.0, t_next - time.time())
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⏹️ Baseline collection interrupted by user (Ctrl+C). Finalizing dataset...")
    finally:
        csv_f.flush()
        csv_f.close()

    print(f"\n\nCollection finished! Total samples recorded: {collected_count:,} (Invalid skipped: {invalid_count})")

    # Run post-collection validation report
    actual_duration = time.time() - t_start
    validation_results = validate_collected_dataset(out_file, actual_duration, interval_sec)
    print_validation_report(validation_results, out_file)

    return validation_results


def main():
    parser = argparse.ArgumentParser(
        description="FGEAD Real Physical Windows Normal Baseline Telemetry Collector"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3600,
        help="Total collection duration in seconds (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/live_baseline.csv",
        help="Output CSV file path (default: data/live_baseline.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file rather than overwriting",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        default=None,
        help="Run validation on an existing CSV dataset without collecting new samples",
    )

    args = parser.parse_args()

    if args.validate_only:
        val_path = Path(args.validate_only)
        if not val_path.is_absolute():
            val_path = PROJECT_ROOT / val_path
        results = validate_collected_dataset(val_path, expected_duration_sec=3600, interval_sec=1.0)
        print_validation_report(results, val_path)
    else:
        collect_baseline(
            duration_sec=args.duration,
            interval_sec=args.interval,
            output_path=args.output,
            append_mode=args.append,
        )


if __name__ == "__main__":
    main()
