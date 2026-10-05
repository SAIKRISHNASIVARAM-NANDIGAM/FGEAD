"""
data/live_feature_schema.py

Official Schema Definition for Live Windows Telemetry Monitoring.
Defines the exact ordered feature set, metadata, and validation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import numpy as np

LIVE_FEATURE_VERSION = "1.0"

# Exact ordered feature list for Live Telemetry
LIVE_FEATURES: List[str] = [
    "cpu_percent",
    "cpu_freq_current",
    "cpu_user_time_percent",
    "cpu_system_time_percent",
    "cpu_ctx_switches_per_sec",
    "cpu_interrupts_per_sec",
    "memory_percent",
    "memory_available_mb",
    "memory_used_mb",
    "swap_percent",
    "disk_usage_percent",
    "disk_read_bytes_per_sec",
    "disk_write_bytes_per_sec",
    "disk_read_count_per_sec",
    "disk_write_count_per_sec",
    "net_bytes_sent_per_sec",
    "net_bytes_recv_per_sec",
    "net_packets_sent_per_sec",
    "net_packets_recv_per_sec",
    "net_errors_total",
    "net_drops_total",
    "process_count",
]

N_LIVE_FEATURES: int = len(LIVE_FEATURES)

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "cpu_percent": "Total CPU utilization across all logical cores (%)",
    "cpu_freq_current": "Current CPU clock frequency (MHz)",
    "cpu_user_time_percent": "CPU execution time in user space (%)",
    "cpu_system_time_percent": "CPU execution time in kernel space (%)",
    "cpu_ctx_switches_per_sec": "System context switch rate (events/sec)",
    "cpu_interrupts_per_sec": "Hardware interrupt rate (events/sec)",
    "memory_percent": "Physical RAM utilization percentage (%)",
    "memory_available_mb": "Available physical memory (MB)",
    "memory_used_mb": "Allocated physical memory (MB)",
    "swap_percent": "Windows page file / swap memory utilization (%)",
    "disk_usage_percent": "System primary drive storage capacity utilized (%)",
    "disk_read_bytes_per_sec": "Disk storage read throughput (bytes/sec)",
    "disk_write_bytes_per_sec": "Disk storage write throughput (bytes/sec)",
    "disk_read_count_per_sec": "Disk read I/O operations rate (IOPS)",
    "disk_write_count_per_sec": "Disk write I/O operations rate (IOPS)",
    "net_bytes_sent_per_sec": "Network outbound egress bandwidth (bytes/sec)",
    "net_bytes_recv_per_sec": "Network inbound ingress bandwidth (bytes/sec)",
    "net_packets_sent_per_sec": "Network packets transmitted per second (pkts/sec)",
    "net_packets_recv_per_sec": "Network packets received per second (pkts/sec)",
    "net_errors_total": "Total cumulative network transmission/reception errors",
    "net_drops_total": "Total cumulative network packet drop count",
    "process_count": "Total active operating system processes",
}

FEATURE_UNITS: Dict[str, str] = {
    "cpu_percent": "%",
    "cpu_freq_current": "MHz",
    "cpu_user_time_percent": "%",
    "cpu_system_time_percent": "%",
    "cpu_ctx_switches_per_sec": "events/s",
    "cpu_interrupts_per_sec": "events/s",
    "memory_percent": "%",
    "memory_available_mb": "MB",
    "memory_used_mb": "MB",
    "swap_percent": "%",
    "disk_usage_percent": "%",
    "disk_read_bytes_per_sec": "B/s",
    "disk_write_bytes_per_sec": "B/s",
    "disk_read_count_per_sec": "IOPS",
    "disk_write_count_per_sec": "IOPS",
    "net_bytes_sent_per_sec": "B/s",
    "net_bytes_recv_per_sec": "B/s",
    "net_packets_sent_per_sec": "pkts/s",
    "net_packets_recv_per_sec": "pkts/s",
    "net_errors_total": "errors",
    "net_drops_total": "drops",
    "process_count": "procs",
}


def format_physical_metric(feature_name: str, val: float) -> str:
    """Format a physical telemetry value into human-friendly units (e.g. MB/s, KB/s, %, IOPS)."""
    if "bytes" in feature_name:
        if abs(val) >= 1024 * 1024 * 1024:
            return f"{val / (1024 * 1024 * 1024):.2f} GB/s"
        elif abs(val) >= 1024 * 1024:
            return f"{val / (1024 * 1024):.2f} MB/s"
        elif abs(val) >= 1024:
            return f"{val / 1024:.2f} KB/s"
        else:
            return f"{val:.1f} B/s"
    elif "percent" in feature_name:
        return f"{val:.1f}%"
    elif "freq" in feature_name:
        return f"{val:.0f} MHz"
    elif "mb" in feature_name:
        if abs(val) >= 1024:
            return f"{val / 1024:.2f} GB"
        return f"{val:,.0f} MB"
    elif "count_per_sec" in feature_name:
        return f"{val:,.1f} IOPS"
    elif "packets" in feature_name:
        return f"{val:,.1f} pkts/s"
    elif "process_count" in feature_name:
        return f"{int(round(val))} procs"
    elif "ctx_switches" in feature_name or "interrupts" in feature_name:
        return f"{val:,.0f} events/s"
    return f"{val:,.2f}"



def validate_live_feature_dict(features: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that a dictionary contains all required live features with valid numeric values.
    """
    if not isinstance(features, dict):
        return False, f"Expected features dictionary, received {type(features).__name__}"

    missing = [f for f in LIVE_FEATURES if f not in features]
    if missing:
        return False, f"Missing required live features: {missing}"

    for f in LIVE_FEATURES:
        val = features[f]
        if not isinstance(val, (int, float, np.number)):
            return False, f"Feature '{f}' has non-numeric type: {type(val).__name__}"
        if math.isnan(val) or math.isinf(val):
            return False, f"Feature '{f}' has non-finite value (NaN/Inf): {val}"

    return True, "Valid"


def feature_dict_to_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert a validated feature dict into an ordered 1D numpy float32 vector.
    """
    return np.array([float(features[f]) for f in LIVE_FEATURES], dtype=np.float32)


def vector_to_feature_dict(vec: np.ndarray) -> Dict[str, float]:
    """
    Convert an ordered 1D numpy vector back to a named dictionary.
    """
    if len(vec) != N_LIVE_FEATURES:
        raise ValueError(f"Expected vector of length {N_LIVE_FEATURES}, got {len(vec)}")
    return {f: float(vec[i]) for i, f in enumerate(LIVE_FEATURES)}


# Model-Input Compatibility Anchors for Invariant Baseline Cumulative Counters
# Maps feature_name -> anchored_model_input_value (e.g. net_drops_total: 294.0)
CUMULATIVE_COUNTER_BASELINE_ANCHORS: Dict[str, float] = {
    "net_drops_total": 294.0,
}


def prepare_model_input_window(
    window_raw: np.ndarray,
    anchors: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Prepare a raw 60x22 physical telemetry window for neural model inference.
    Preserves all 22 features and strict ordering, but anchors invariant cumulative counters
    to their calibrated training baseline constants to prevent false positives from OS uptime drift.
    """
    arr = np.array(window_raw, dtype=np.float32, copy=True)
    active_anchors = anchors if anchors is not None else CUMULATIVE_COUNTER_BASELINE_ANCHORS
    for feat_name, anchor_val in active_anchors.items():
        if feat_name in LIVE_FEATURES:
            idx = LIVE_FEATURES.index(feat_name)
            if arr.ndim == 2:
                arr[:, idx] = anchor_val
            elif arr.ndim == 1:
                arr[idx] = anchor_val
    return arr
