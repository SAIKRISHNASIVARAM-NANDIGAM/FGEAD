"""
data/collect_v3_baseline.py

Collects fresh live continuous physical host telemetry for V3 Current-Machine Model.
Uses exact LiveTelemetryCollector pipeline from data/live_agent.py.
Collects exactly 1,800 valid samples (30 minutes) at 1.0s interval under normal laptop operation.

Features:
- Windows Sleep/Suspend Prevention via SetThreadExecutionState Win32 API.
- Monotonic timing (time.monotonic()) for drift-free scheduling and interval tracking.
- Monotonic sleep/resume detection: Invalidates & restarts collection if gap > max_allowed_interval (5.0s).
- Heartbeat logging every 30 samples (or upon interruption).
- Streams and flushes safely to data/live_baseline_v3_current_machine.csv.
- Writes metadata to data/live_training/v3_baseline_metadata.json.
"""

from __future__ import annotations

import csv
import ctypes
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_agent import LiveTelemetryCollector
from data.live_feature_schema import LIVE_FEATURES, LIVE_FEATURE_VERSION, validate_live_feature_dict

# Win32 Power Management Constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def enable_windows_sleep_prevention() -> bool:
    """Prevent Windows system and display sleep during active collection."""
    if sys.platform == "win32":
        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            return res != 0
        except Exception as err:
            print(f"[POWER WARNING] Could not set Windows execution state: {err}", flush=True)
            return False
    return False


def disable_windows_sleep_prevention() -> None:
    """Restore normal Windows power management behavior."""
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        except Exception:
            pass


def collect_v3_baseline(
    n_samples: int = 1800,
    interval_sec: float = 1.0,
    max_allowed_interval_sec: float = 5.0,
    output_csv_path: str = "data/live_baseline_v3_current_machine.csv",
    metadata_json_path: str = "data/live_training/v3_baseline_metadata.json",
    max_auto_restarts: int = 3,
) -> Tuple[Path, Path]:
    csv_file = PROJECT_ROOT / output_csv_path
    meta_file = PROJECT_ROOT / metadata_json_path

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    collector = LiveTelemetryCollector()
    info = collector.get_machine_info()

    # Enable Windows Sleep Prevention
    sleep_prevented = enable_windows_sleep_prevention()

    start_time_iso = datetime.now(timezone.utc).isoformat()
    t_start_epoch = time.time()

    print("=" * 75, flush=True)
    print("FGEAD V3 LIVE BASELINE TELEMETRY COLLECTION (MONOTONIC & SLEEP-PROTECTED)", flush=True)
    print("=" * 75, flush=True)
    print(f"Target Machine      : {info['hostname']} ({info['platform']} {info['os_version']})", flush=True)
    print(f"Collection Start Time: {start_time_iso}", flush=True)
    print(f"Target Sample Count : {n_samples} samples", flush=True)
    print(f"Sampling Interval   : {interval_sec:.1f} second", flush=True)
    print(f"Expected Duration   : ≈ {round(n_samples * interval_sec / 60.0, 1)} minutes", flush=True)
    print(f"Sleep Prevention    : {'🟢 ENABLED (Win32 SetThreadExecutionState)' if sleep_prevented else '🟡 UNCHANGEABLE / OFF'}", flush=True)
    print(f"Output CSV Path     : {csv_file}", flush=True)
    print(f"Output Metadata Path: {meta_file}", flush=True)
    print("=" * 75, flush=True)

    try:
        # Prime differential counters (1 second pause)
        print("[INIT] Priming hardware differential counters...", flush=True)
        time.sleep(1.0)

        cols = ["timestamp"] + LIVE_FEATURES

        restart_count = 0
        total_interruptions = 0

        while restart_count <= max_auto_restarts:
            # Overwrite output CSV with fresh header
            with open(csv_file, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=cols)
                writer.writeheader()
                f_csv.flush()

                t_run_start_mono = time.monotonic()
                last_sample_mono = t_run_start_mono
                max_gap_observed = 0.0
                run_interrupted = False

                print(f"[START] Beginning live telemetry sampling (Attempt #{restart_count + 1})...", flush=True)

                for i in range(1, n_samples + 1):
                    # Monotonic absolute target time
                    target_mono = t_run_start_mono + i * interval_sec
                    now_mono = time.monotonic()
                    sleep_dur = max(0.0001, target_mono - now_mono)
                    time.sleep(sleep_dur)

                    now_after_sleep_mono = time.monotonic()
                    actual_interval = now_after_sleep_mono - last_sample_mono

                    # Check for Sleep / Resume or Abnormal Inter-Sample Delay
                    if i > 1 and actual_interval > max_allowed_interval_sec:
                        total_interruptions += 1
                        run_interrupted = True
                        print(f"\n[V3][SLEEP/RESUME DETECTED]", flush=True)
                        print(f"Previous interval: {actual_interval:.2f}s (exceeds max allowed {max_allowed_interval_sec:.1f}s)", flush=True)
                        print(f"Collection invalidated. Restarting fresh baseline collection...\n", flush=True)
                        break

                    if actual_interval > max_gap_observed and i > 1:
                        max_gap_observed = actual_interval

                    # Collect snapshot
                    try:
                        features = collector.collect_features()
                    except Exception as exc:
                        print(f"[ERROR] Telemetry collection exception at sample #{i}/{n_samples}: {exc}", flush=True)
                        raise RuntimeError(f"Telemetry collection failed at sample #{i}: {exc}") from exc

                    # Validate schema and non-null bounds
                    is_valid, msg = validate_live_feature_dict(features)
                    if not is_valid:
                        print(f"[ERROR] Telemetry validation failed at sample #{i}/{n_samples}: {msg}", flush=True)
                        raise ValueError(f"Telemetry validation failed at sample #{i}: {msg}")

                    ts_iso = datetime.now(timezone.utc).isoformat()
                    row_dict = {"timestamp": ts_iso}
                    for f in LIVE_FEATURES:
                        row_dict[f] = features[f]

                    writer.writerow(row_dict)
                    last_sample_mono = now_after_sleep_mono

                    if i % 10 == 0:
                        f_csv.flush()

                    # Heartbeat logging every 30 samples
                    if i % 30 == 0 or i == n_samples:
                        elapsed_m = round((time.monotonic() - t_run_start_mono) / 60.0, 1)
                        print(
                            f"[V3] Sample {i:04d}/{n_samples} ({elapsed_m:4.1f}m) | "
                            f"interval={actual_interval:.3f}s | "
                            f"max_gap={max_gap_observed:.3f}s | "
                            f"interruptions={total_interruptions} | "
                            f"CPU: {features['cpu_percent']:4.1f}% | "
                            f"RAM: {features['memory_percent']:4.1f}% ({features['memory_used_mb']:5.0f}MB)",
                            flush=True
                        )

                f_csv.flush()

            if not run_interrupted:
                # Successfully collected uninterrupted baseline
                break

            restart_count += 1
            if restart_count > max_auto_restarts:
                raise RuntimeError(f"Collection failed after {max_auto_restarts} restart attempts due to repeated OS sleep interruptions.")

        end_time_iso = datetime.now(timezone.utc).isoformat()
        total_elapsed_sec = time.time() - t_start_epoch

        # Read back saved CSV to confirm exact row count and schema
        df = pd.read_csv(csv_file)
        if len(df) != n_samples:
            raise RuntimeError(f"Persisted CSV row count mismatch: Expected {n_samples}, got {len(df)}")

        print("=" * 75, flush=True)
        print(f"[SUCCESS] Telemetry CSV written to {csv_file} ({len(df)} rows, {len(df.columns)} columns).", flush=True)
        print(f"[SUCCESS] Total Elapsed Time: {total_elapsed_sec:.2f}s | Max Gap Observed: {max_gap_observed:.3f}s", flush=True)
        print("=" * 75, flush=True)

        # Metadata record
        metadata = {
            "hostname": info["hostname"],
            "operating_system": info["platform"],
            "os_version": info["os_version"],
            "architecture": info["architecture"],
            "cpu_logical_cores": info["cpu_logical_cores"],
            "cpu_physical_cores": info["cpu_physical_cores"],
            "timestamp_start": start_time_iso,
            "timestamp_end": end_time_iso,
            "sample_count": len(df),
            "sampling_interval_sec": interval_sec,
            "actual_total_duration_sec": round(total_elapsed_sec, 2),
            "actual_mean_interval_sec": round(total_elapsed_sec / n_samples, 4),
            "max_gap_observed_sec": round(max_gap_observed, 4),
            "total_interruptions_detected": total_interruptions,
            "sleep_prevention_enabled": sleep_prevented,
            "duration_minutes": round(total_elapsed_sec / 60.0, 2),
            "feature_count": len(LIVE_FEATURES),
            "feature_names": LIVE_FEATURES,
            "collection_version": LIVE_FEATURE_VERSION,
            "model_version_target": "windows_sivachowdary_v3",
            "collection_mode": "continuous_live_telemetry_sleep_protected",
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SUCCESS] Metadata JSON written to {meta_file}.", flush=True)
        return csv_file, meta_file

    finally:
        # Restore normal power behavior upon exit
        disable_windows_sleep_prevention()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect V3 Baseline Telemetry")
    parser.add_argument("--samples", type=int, default=1800, help="Number of 1-second samples to collect (default: 1800)")
    parser.add_argument("--max-gap", type=float, default=5.0, help="Maximum allowed gap before sleep/resume detection triggers (default: 5.0s)")
    args = parser.parse_args()

    collect_v3_baseline(n_samples=args.samples, max_allowed_interval_sec=args.max_gap)
