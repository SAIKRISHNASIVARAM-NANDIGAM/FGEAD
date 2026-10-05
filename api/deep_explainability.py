"""
api/deep_explainability.py

FGEAD Deep Human-Understandable Explainability Engine.
Translates technical neural forecasting residuals, attention weights, and multi-variate
telemetry deviations into structured, accessible, evidence-based human explanations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    FEATURE_UNITS,
    LIVE_FEATURES,
    format_physical_metric,
    CUMULATIVE_COUNTER_BASELINE_ANCHORS,
)

# Human-Readable Feature Names (Plain English)
FEATURE_HUMAN_NAMES: Dict[str, str] = {
    "cpu_percent": "Overall CPU usage",
    "cpu_freq_current": "Current CPU operating frequency",
    "cpu_user_time_percent": "CPU time used by applications",
    "cpu_system_time_percent": "CPU time used by the operating system",
    "cpu_ctx_switches_per_sec": "How frequently the operating system switches between running tasks",
    "cpu_interrupts_per_sec": "How frequently hardware/software interrupts are being handled",
    "memory_percent": "Percentage of system memory currently in use",
    "memory_available_mb": "Memory currently available for new work",
    "memory_used_mb": "Memory currently being used",
    "swap_percent": "Percentage of swap/page-file memory in use",
    "disk_usage_percent": "Percentage of disk storage capacity currently used",
    "disk_read_bytes_per_sec": "Amount of data read from storage each second",
    "disk_write_bytes_per_sec": "Amount of data written to storage each second",
    "disk_read_count_per_sec": "Number of storage read operations each second",
    "disk_write_count_per_sec": "Number of storage write operations each second",
    "net_bytes_sent_per_sec": "Amount of network data sent each second",
    "net_bytes_recv_per_sec": "Amount of network data received each second",
    "net_packets_sent_per_sec": "Number of network packets sent each second",
    "net_packets_recv_per_sec": "Number of network packets received each second",
    "net_errors_total": "Network errors recorded by the operating system",
    "net_drops_total": "Network packets reported as dropped by the operating system",
    "process_count": "Number of active processes running on the system",
}

# Subsystem Human Names
SUBSYSTEM_HUMAN_NAMES: Dict[str, str] = {
    "Storage I/O": "Disk Storage & File Activity",
    "CPU Subsystem": "Processor & Compute Workload",
    "Memory": "System RAM & Memory Allocation",
    "Network": "Network Traffic & Communications",
    "Operating System": "Operating System & Process Management",
}

# Practical Investigation Suggestions per Feature
FEATURE_INVESTIGATION_HINTS: Dict[str, List[str]] = {
    "cpu_percent": [
        "Check Task Manager or process monitor for applications consuming elevated CPU.",
        "Check whether background compilation, video encoding, or compute jobs were started.",
    ],
    "cpu_user_time_percent": [
        "Check running user applications, scripts, or browser tabs for high processor utilization.",
    ],
    "cpu_system_time_percent": [
        "Check for heavy kernel activity, device driver routines, or background antivirus file inspections.",
    ],
    "cpu_ctx_switches_per_sec": [
        "Check if a large number of concurrent threads or processes are actively competing for processor time.",
    ],
    "cpu_interrupts_per_sec": [
        "Check hardware peripherals, network adapters, or storage controllers generating high interrupt rates.",
    ],
    "memory_percent": [
        "Check which applications currently hold the largest memory allocations.",
        "Check if a process has an expanding working set indicating a memory leak.",
    ],
    "memory_used_mb": [
        "Inspect memory usage trends across currently running background processes.",
    ],
    "memory_available_mb": [
        "Verify whether remaining available RAM is adequate for ongoing workloads.",
    ],
    "swap_percent": [
        "Check if physical RAM is fully allocated and the OS is paging memory to storage.",
    ],
    "disk_usage_percent": [
        "Check if a large file download, database growth, or archive expansion recently occurred.",
    ],
    "disk_read_bytes_per_sec": [
        "Check which application is reading large files or scanning directories.",
        "Check whether an antivirus scan, backup read, or search indexer is currently active.",
    ],
    "disk_write_bytes_per_sec": [
        "Check which application is currently writing large amounts of data to disk.",
        "Check whether a large download, file transfer, database commit, or backup is running.",
        "Check whether Windows Update or an installer is actively writing files.",
    ],
    "disk_read_count_per_sec": [
        "Check for high-frequency random read operations from databases or indexing tools.",
    ],
    "disk_write_count_per_sec": [
        "Check for high-frequency small write operations, logging loops, or database journal writes.",
    ],
    "net_bytes_sent_per_sec": [
        "Check which application is uploading data, streaming media, or transferring files outbound.",
    ],
    "net_bytes_recv_per_sec": [
        "Check which application is downloading files or receiving large incoming data streams.",
        "Check whether a software update or remote file synchronization is in progress.",
    ],
    "net_packets_sent_per_sec": [
        "Check for high packet-rate traffic, socket communication loops, or port scanning activity.",
    ],
    "net_packets_recv_per_sec": [
        "Check for incoming packet bursts or high-frequency network requests.",
    ],
    "net_errors_total": [
        "Check network interface health, duplex settings, and cabling for physical transmission errors.",
    ],
    "net_drops_total": [
        "Network drop count is an OS-level counter; inspect network interface health and socket throughput if sustained drop events are reported.",
    ],
    "process_count": [
        "Check if an application spawned an unusually large number of child or worker processes.",
    ],
}

DISCLAIMER_TEXT = (
    "FGEAD identifies unusual behavior compared with the learned normal operating pattern. "
    "It does not by itself prove that the machine is compromised, damaged, or under attack."
)


def get_feature_human_name(feature: str) -> str:
    return FEATURE_HUMAN_NAMES.get(feature, feature.replace("_", " ").title())


def describe_feature_meaning(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, f"System metric tracking {feature}")


def compute_difference_description(feature: str, observed: float, expected: float) -> Tuple[str, str]:
    """Compute human-friendly difference and meaning."""
    diff = observed - expected
    obs_fmt = format_physical_metric(feature, observed)
    exp_fmt = format_physical_metric(feature, expected)

    if "bytes" in feature or "count" in feature or "packets" in feature or "switches" in feature or "interrupts" in feature:
        if expected > 0.001:
            multiplier = observed / expected
            if multiplier >= 2.0:
                diff_str = f"+{format_physical_metric(feature, diff)} (~{multiplier:.1f}× normal)"
                meaning = "Much higher than normal"
            elif multiplier <= 0.5 and multiplier > 0:
                diff_str = f"-{format_physical_metric(feature, abs(diff))} (~{1.0/multiplier:.1f}× lower)"
                meaning = "Much lower than normal"
            elif diff > 0:
                diff_str = f"+{format_physical_metric(feature, diff)}"
                meaning = "Slightly elevated"
            else:
                diff_str = f"-{format_physical_metric(feature, abs(diff))}"
                meaning = "Slightly lower"
        else:
            diff_str = f"+{obs_fmt}"
            meaning = "Elevated from near-zero baseline"
    elif "percent" in feature:
        pts = diff
        if pts >= 20.0:
            diff_str = f"+{pts:.1f} percentage points"
            meaning = "Substantially higher utilization"
        elif pts >= 5.0:
            diff_str = f"+{pts:.1f} percentage points"
            meaning = "Higher workload"
        elif pts <= -20.0:
            diff_str = f"{pts:.1f} percentage points"
            meaning = "Substantially lower utilization"
        elif pts <= -5.0:
            diff_str = f"{pts:.1f} percentage points"
            meaning = "Lower workload"
        else:
            diff_str = f"{pts:+.1f} percentage points"
            meaning = "Close to normal baseline"
    else:
        diff_str = f"{diff:+.1f}"
        meaning = "Different from baseline"

    return diff_str, meaning


def build_deep_human_explanation(
    state: str,  # "ANOMALY", "SUSPICIOUS", "NORMAL", "TELEMETRY_ONLY", "OFFLINE", "WARMING_UP"
    anomaly_score: float,
    threshold: float,
    top_features: List[Dict[str, Any]],
    timestamp_iso: str,
    episode_info: Optional[Dict[str, Any]] = None,
    warmup_count: int = 60,
    host_id: str = "host",
) -> Dict[str, Any]:
    """
    Construct a complete, deep human-understandable explanation adhering strictly to
    the 16 explainability principles.
    """
    ep_info = episode_info or {}
    ep_id = ep_info.get("active_episode_id", 1)
    ep_dur = ep_info.get("duration_sec", 1)
    ratio = round(anomaly_score / threshold, 2) if threshold > 0 else 1.0

    # 1. OFFLINE STATE
    if state == "OFFLINE":
        return {
            "state": "OFFLINE",
            "headline": "Host is currently offline",
            "what_happened": f"Host '{host_id}' is currently not sending telemetry.",
            "why_detected": (
                "FGEAD cannot determine whether the system is normal or experiencing an anomaly "
                "until telemetry transmission resumes."
            ),
            "what_changed_table": [],
            "top_contributors": [],
            "why_this_is_an_anomaly": "Telemetry stream is interrupted. No anomaly decision is produced.",
            "severity_text": "STATUS: OFFLINE",
            "investigation_suggestions": [
                "Verify that the physical host or server is powered on.",
                "Check whether the FGEAD monitoring agent process is active on the host.",
                "Check network connectivity and firewall rules between host and backend.",
            ],
            "disclaimer": DISCLAIMER_TEXT,
            "technical_details": {
                "host_id": host_id,
                "state": "OFFLINE",
                "timestamp": timestamp_iso,
            },
        }

    # 2. WARMING UP STATE
    if state == "WARMING_UP":
        return {
            "state": "WARMING_UP",
            "headline": f"Learning observation window ({warmup_count} / 60 measurements)",
            "what_happened": (
                f"FGEAD is buffering initial telemetry for '{host_id}'. "
                f"60 measurements are required before live anomaly inference can begin. "
                f"Current progress: {warmup_count} / 60."
            ),
            "why_detected": "The rolling 60-second window is warming up.",
            "what_changed_table": [],
            "top_contributors": [],
            "why_this_is_an_anomaly": "Inference starts automatically once 60 samples are collected.",
            "severity_text": f"STATUS: BUFFERING ({warmup_count}/60)",
            "investigation_suggestions": [
                "Allow the agent to finish collecting initial buffer telemetry (approx 60 seconds).",
            ],
            "disclaimer": DISCLAIMER_TEXT,
            "technical_details": {
                "host_id": host_id,
                "state": "WARMING_UP",
                "buffered_samples": warmup_count,
                "required_samples": 60,
                "timestamp": timestamp_iso,
            },
        }

    # 3. TELEMETRY ONLY STATE (Unsupported OS / Pending Model)
    if state == "TELEMETRY_ONLY":
        return {
            "state": "TELEMETRY_ONLY",
            "headline": "Telemetry-only monitoring active",
            "what_happened": (
                f"FGEAD is receiving telemetry from host '{host_id}', but it does not currently "
                f"have a compatible calibrated model for this operating system."
            ),
            "why_detected": (
                "The system is collecting and displaying telemetry metrics without producing "
                "an automated anomaly decision to prevent false alarms."
            ),
            "what_changed_table": [],
            "top_contributors": [],
            "why_this_is_an_anomaly": (
                "Anomaly inference is gated until a calibrated baseline model is trained for this platform."
            ),
            "severity_text": "STATUS: TELEMETRY_ONLY (GATED)",
            "investigation_suggestions": [
                "Telemetry counters (CPU, RAM, Disk, Network) are streaming normally.",
                "To enable anomaly detection, record an operating baseline and train a compatible model profile.",
            ],
            "disclaimer": DISCLAIMER_TEXT,
            "technical_details": {
                "host_id": host_id,
                "state": "TELEMETRY_ONLY",
                "model_gated": True,
                "timestamp": timestamp_iso,
            },
        }

    # 4. NORMAL STATE
    if state == "NORMAL":
        return {
            "state": "NORMAL",
            "headline": "System is operating within learned normal range",
            "what_happened": (
                "System behavior is within the learned normal operating range. "
                "No persistent unusual behavior has been detected."
            ),
            "why_detected": (
                f"The latest telemetry window produced an anomaly score of {anomaly_score:.4f}, "
                f"which remains safely below the calibrated baseline threshold of {threshold:.4f} ({ratio:.2f}× of threshold limit)."
            ),
            "what_changed_table": [],
            "top_contributors": [],
            "why_this_is_an_anomaly": (
                "No persistent unusual behavior has been detected in the latest telemetry window."
            ),
            "severity_text": "Severity: NOMINAL (Normal Operation)",
            "investigation_suggestions": [
                "No action required. All hardware subsystems (CPU, RAM, Disk, Network, and OS Processes) are operating within normal parameters.",
            ],
            "disclaimer": DISCLAIMER_TEXT,
            "technical_details": {
                "host_id": host_id,
                "state": "NORMAL",
                "anomaly_score": round(anomaly_score, 4),
                "threshold": round(threshold, 4),
                "ratio": ratio,
                "timestamp": timestamp_iso,
            },
        }

    # 5. SUSPICIOUS STATE
    if state == "SUSPICIOUS":
        valid_features = [f for f in top_features if f.get("feature") not in CUMULATIVE_COUNTER_BASELINE_ANCHORS]
        top_names = [get_feature_human_name(f["feature"]) for f in valid_features[:2]]
        feat_str = " and ".join(top_names) if top_names else "telemetry counters"
        return {
            "state": "SUSPICIOUS",
            "headline": "Temporary unusual activity observed (Monitoring)",
            "what_happened": (
                f"FGEAD noticed a temporary change in {feat_str}, but the behavior has not continued "
                f"long enough to confirm an anomaly."
            ),
            "why_detected": (
                f"A single telemetry window exceeded the threshold (Score: {anomaly_score:.4f} vs Threshold: {threshold:.4f}), "
                f"but multi-window persistence confirmation has not yet been reached."
            ),
            "what_changed_table": [],
            "top_contributors": [],
            "why_this_is_an_anomaly": (
                "The system requires persistent abnormal windows before alerting to prevent false-positive flickering. "
                "FGEAD is continuing to monitor subsequent measurements."
            ),
            "severity_text": "Severity: SUSPICIOUS (Pending Confirmation)",
            "investigation_suggestions": [
                "The system is currently observing whether this was a momentary spike or the start of a sustained workload.",
            ],
            "disclaimer": DISCLAIMER_TEXT,
            "technical_details": {
                "host_id": host_id,
                "state": "SUSPICIOUS",
                "anomaly_score": round(anomaly_score, 4),
                "threshold": round(threshold, 4),
                "ratio": ratio,
                "timestamp": timestamp_iso,
            },
        }

    # 6. CONFIRMED ANOMALY STATE
    valid_features = [f for f in top_features if f.get("feature") not in CUMULATIVE_COUNTER_BASELINE_ANCHORS]
    top3 = valid_features[:3] if valid_features else []
    dom_feature = top3[0]["feature"] if top3 else "system_metrics"
    dom_human = get_feature_human_name(dom_feature)
    dom_subsystem = SUBSYSTEM_HUMAN_NAMES.get(
        top3[0].get("subsystem", ""), top3[0].get("subsystem", "System Activity")
    ) if top3 else "System Activity"

    # A. WHAT HAPPENED?
    what_happened = f"FGEAD detected unusually high activity in {dom_human.lower()} ({dom_subsystem.lower()}) on this computer."

    # B. WHY WAS IT DETECTED?
    f0 = top3[0] if top3 else {}
    f0_name = f0.get("feature", "")
    f0_human = get_feature_human_name(f0_name)
    obs_val = float(f0.get("actual_value", f0.get("current_value", 0.0)))
    exp_val = float(f0.get("predicted_value", 0.0))
    obs_fmt = format_physical_metric(f0_name, obs_val)
    exp_fmt = format_physical_metric(f0_name, exp_val) if exp_val > 0 else "nominal baseline"

    if exp_val > 0.001 and obs_val > exp_val:
        mult = obs_val / exp_val
        why_detected = (
            f"Normally, this computer's {f0_human.lower()} is expected to be approximately {exp_fmt}. "
            f"During this event, it was observed at {obs_fmt}. "
            f"This is approximately {mult:.1f} times higher than the learned baseline forecast."
        )
    elif obs_val > exp_val:
        why_detected = (
            f"Normally, this computer's {f0_human.lower()} operates near {exp_fmt}. "
            f"During this event, it rose significantly to {obs_fmt}."
        )
    else:
        why_detected = (
            f"Normally, this computer operates with {f0_human.lower()} around {exp_fmt}. "
            f"During this event, it shifted unexpectedly to {obs_fmt}."
        )

    # C. WHAT CHANGED? (Comparison Table)
    what_changed_table = []
    for tf in top3:
        f_key = tf.get("feature", "")
        f_hname = get_feature_human_name(f_key)
        cur_v = float(tf.get("actual_value", tf.get("current_value", 0.0)))
        pred_v = float(tf.get("predicted_value", 0.0))
        cur_f = format_physical_metric(f_key, cur_v)
        pred_f = format_physical_metric(f_key, pred_v) if pred_v > 0 else "Baseline"
        diff_s, mean_s = compute_difference_description(f_key, cur_v, pred_v)

        what_changed_table.append({
            "measurement": f_hname,
            "technical_key": f_key,
            "normal_behavior": pred_f,
            "current_behavior": cur_f,
            "difference": diff_s,
            "meaning": mean_s,
        })

    # D & E. EXPLAIN EACH CONTRIBUTOR
    explained_contributors = []
    for rank, tf in enumerate(top3, start=1):
        f_key = tf.get("feature", "")
        f_hname = get_feature_human_name(f_key)
        f_desc = describe_feature_meaning(f_key)
        cur_v = float(tf.get("actual_value", tf.get("current_value", 0.0)))
        pred_v = float(tf.get("predicted_value", 0.0))
        cur_f = format_physical_metric(f_key, cur_v)
        pred_f = format_physical_metric(f_key, pred_v) if pred_v > 0 else "Baseline"
        diff_s, mean_s = compute_difference_description(f_key, cur_v, pred_v)

        explained_contributors.append({
            "rank": rank,
            "feature_human_name": f_hname,
            "technical_key": f_key,
            "what_it_means": f_desc,
            "normally_expected": pred_f,
            "observed_value": cur_f,
            "difference_factor": diff_s,
            "why_it_contributed": (
                f"{f_hname} was one of the strongest root-cause candidates because its observed value "
                f"({cur_f}) departed most significantly from the model's spatio-temporal baseline expectation ({pred_f})."
            ),
        })

    # F. WHY THIS IS AN ANOMALY
    why_anomaly = (
        f"FGEAD learned what normal behavior looks like from this computer's calibrated baseline telemetry. "
        f"During this window, multiple measurements (led by {dom_human.lower()}) changed significantly at the same time. "
        f"The changes were larger than the variation seen during normal operation, and the elevated behavior "
        f"persisted long enough ({ep_dur} seconds in Episode #{ep_id}) to satisfy the confirmation rule. "
        f"For these reasons, FGEAD confirmed this event as an anomaly."
    )

    # G. SEVERITY
    if ratio >= 2.5:
        sev_label = "CRITICAL"
        sev_desc = "substantially and severely outside the learned normal operating range"
    elif ratio >= 1.5:
        sev_label = "HIGH"
        sev_desc = "substantially outside the learned normal operating range"
    else:
        sev_label = "MODERATE"
        sev_desc = "moderately outside the learned normal operating range"

    severity_text = (
        f"Severity: {sev_label}\n"
        f"The anomaly score is {anomaly_score:.2f} while the calibrated threshold is {threshold:.2f}. "
        f"The score is approximately {ratio:.1f} times the threshold limit. "
        f"The event is therefore {sev_desc}."
    )

    # H. WHAT SHOULD THE USER CHECK?
    suggestions = []
    for tf in top3:
        f_key = tf.get("feature", "")
        hints = FEATURE_INVESTIGATION_HINTS.get(f_key, [])
        for h in hints:
            if h not in suggestions:
                suggestions.append(h)
    if not suggestions:
        suggestions = [
            "Check Task Manager for active processes consuming high system resources.",
            "Verify whether scheduled tasks, backups, or downloads were triggered.",
        ]

    return {
        "state": "ANOMALY",
        "headline": what_happened,
        "what_happened": what_happened,
        "why_detected": why_detected,
        "what_changed_table": what_changed_table,
        "top_contributors": explained_contributors,
        "why_this_is_an_anomaly": why_anomaly,
        "severity_text": severity_text,
        "investigation_suggestions": suggestions[:5],
        "disclaimer": DISCLAIMER_TEXT,
        "technical_details": {
            "host_id": host_id,
            "state": "ANOMALY",
            "anomaly_score": round(anomaly_score, 4),
            "threshold": round(threshold, 4),
            "ratio": ratio,
            "active_episode_id": ep_id,
            "episode_duration_sec": ep_dur,
            "detection_time": timestamp_iso,
            "raw_top_features": top_features,
        },
    }
