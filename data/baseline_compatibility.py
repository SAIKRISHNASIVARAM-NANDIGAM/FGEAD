"""
data/baseline_compatibility.py

Baseline Quality Audit, Distribution Profiling, and Cross-Host Compatibility Comparison.
Used to validate host telemetry baselines against model training baselines to detect
"baseline compatibility differences" before assigning or executing neural anomaly models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to original training baseline
DEFAULT_BASELINE_CSV = PROJECT_ROOT / "data" / "live_baseline.csv"

_CACHED_TRAINING_STATS: Optional[Dict[str, Dict[str, float]]] = None


def compute_feature_distribution(
    data: Union[np.ndarray, pd.DataFrame, List[Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute distribution statistics (mean, std, min, max, P95, P99, is_constant)
    for all 22 live features.
    """
    if isinstance(data, list):
        if not data:
            return {}
        arr = np.array([[float(d.get(f, 0.0)) for f in LIVE_FEATURES] for d in data], dtype=np.float64)
    elif isinstance(data, pd.DataFrame):
        arr = data[LIVE_FEATURES].to_numpy(dtype=np.float64)
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.float64)
    else:
        raise TypeError(f"Unsupported data type for distribution calculation: {type(data)}")

    if arr.ndim != 2 or arr.shape[1] != N_LIVE_FEATURES:
        raise ValueError(f"Expected 2D array with {N_LIVE_FEATURES} columns, got shape {arr.shape}")

    stats: Dict[str, Dict[str, float]] = {}
    for idx, feat in enumerate(LIVE_FEATURES):
        col = arr[:, idx]
        mean_val = float(np.mean(col))
        std_val = float(np.std(col))
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        p50_val = float(np.percentile(col, 50))
        p95_val = float(np.percentile(col, 95))
        p99_val = float(np.percentile(col, 99))
        is_const = bool(std_val < 1e-6 or (max_val - min_val) < 1e-6)

        stats[feat] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "p50": round(p50_val, 4),
            "p95": round(p95_val, 4),
            "p99": round(p99_val, 4),
            "is_constant": 1.0 if is_const else 0.0,
        }

    return stats


def get_reference_training_baseline_stats() -> Dict[str, Dict[str, float]]:
    """
    Load and cache baseline statistics for the original 60-minute Windows training dataset.
    """
    global _CACHED_TRAINING_STATS
    if _CACHED_TRAINING_STATS is not None:
        return _CACHED_TRAINING_STATS

    if not DEFAULT_BASELINE_CSV.exists():
        return {}

    try:
        df = pd.read_csv(DEFAULT_BASELINE_CSV)
        _CACHED_TRAINING_STATS = compute_feature_distribution(df)
        return _CACHED_TRAINING_STATS
    except Exception as exc:
        print(f"[BASELINE COMPATIBILITY] Warning: Failed to compute reference baseline stats: {exc}")
        return {}


def compare_baseline_distributions(
    host_stats: Dict[str, Dict[str, float]],
    reference_stats: Optional[Dict[str, Dict[str, float]]] = None,
    z_threshold: float = 3.0,
) -> Dict[str, Any]:
    """
    Compare a host's normal baseline distribution against a model's training baseline distribution.
    Identifies constant feature differences and distribution shifts.
    
    IMPORTANT: Flags discrepancies as "baseline compatibility difference", NOT as anomalies.
    """
    ref = reference_stats or get_reference_training_baseline_stats()
    if not ref:
        return {
            "is_compatible": False,
            "status": "REFERENCE_UNAVAILABLE",
            "differences": [],
            "constant_feature_shifts": [],
            "summary": "Reference training baseline statistics unavailable for comparison.",
        }

    differences = []
    constant_shifts = []

    for feat in LIVE_FEATURES:
        if feat not in host_stats or feat not in ref:
            continue

        h_feat = host_stats[feat]
        r_feat = ref[feat]

        h_mean, h_std = h_feat["mean"], h_feat["std"]
        r_mean, r_std = r_feat["mean"], r_feat["std"]
        h_is_const = bool(h_feat.get("is_constant", 0.0))
        r_is_const = bool(r_feat.get("is_constant", 0.0))

        # Check constant features (like net_drops_total or net_errors_total)
        if r_is_const:
            if abs(h_mean - r_mean) > 1e-4:
                shift_desc = (
                    f"Constant feature '{feat}' has different static value in host baseline "
                    f"(host mean: {h_mean:.2f} vs reference training: {r_mean:.2f})"
                )
                constant_shifts.append({
                    "feature": feat,
                    "host_mean": h_mean,
                    "ref_mean": r_mean,
                    "description": shift_desc,
                })
                differences.append({
                    "feature": feat,
                    "type": "constant_feature_shift",
                    "severity": "HIGH",
                    "description": shift_desc,
                    "host_stats": h_feat,
                    "ref_stats": r_feat,
                })
            continue

        # Check non-constant feature distribution shifts
        std_denom = r_std if r_std > 1e-4 else 1.0
        z_mean_diff = abs(h_mean - r_mean) / std_denom

        if z_mean_diff > z_threshold:
            diff_desc = (
                f"Feature '{feat}' mean deviates by {z_mean_diff:.2f} std from training baseline "
                f"(host mean: {h_mean:.2f}, ref mean: {r_mean:.2f})"
            )
            differences.append({
                "feature": feat,
                "type": "distribution_shift",
                "severity": "ELEVATED" if z_mean_diff < 5.0 else "HIGH",
                "z_score_diff": round(z_mean_diff, 2),
                "description": diff_desc,
                "host_stats": h_feat,
                "ref_stats": r_feat,
            })

    is_compatible = len(differences) == 0

    if is_compatible:
        summary = "Host baseline matches training baseline distribution within normal tolerance."
    else:
        summary = (
            f"Detected {len(differences)} baseline compatibility differences "
            f"({len(constant_shifts)} constant feature shifts, "
            f"{len(differences) - len(constant_shifts)} distribution deviations)."
        )

    return {
        "is_compatible": is_compatible,
        "status": "COMPATIBLE" if is_compatible else "COMPATIBILITY_DIFFERENCE_DETECTED",
        "difference_count": len(differences),
        "differences": differences,
        "constant_feature_shifts": constant_shifts,
        "summary": summary,
    }
