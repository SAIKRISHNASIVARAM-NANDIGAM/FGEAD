"""
data/multihost_validation/evaluate_phase9_metrics.py

Phase 9 Comprehensive Evaluation Script:
1. Normal Baseline Evaluation (3,541 windows from data/live_baseline.csv)
   - Score statistics (Min, Max, Mean, Std, Median, P95, P99)
   - False Positive Rate (FPR) at raw window level and confirmed episode level
2. Controlled Anomaly Scenarios (CPU, Disk Write, Disk Read, Network Burst, Combined)
   - Detection rate, max score, detection delay, feature attribution ranking
3. Accuracy Metrics (Precision, Recall, F1, FPR, FNR)
4. Decision Gate Verification
"""

import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    FEATURE_UNITS,
    LIVE_FEATURES,
    N_LIVE_FEATURES,
    format_physical_metric,
)
from models.fgead import FGEAD
from api.live_inference import LiveEpisodeTracker


def evaluate_system():
    print("=" * 80)
    print("FGEAD PHASE 9 — TRUSTWORTHY DECISION & FALSE-POSITIVE VALIDATION ENGINE")
    print("=" * 80)

    # 1. Load Checkpoint and Scaler
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch.pt"
    scaler_path = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler.joblib"
    threshold_path = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold.json"
    baseline_csv = PROJECT_ROOT / "data" / "live_baseline.csv"

    with open(threshold_path, "r", encoding="utf-8") as f:
        th_data = json.load(f)
        threshold = float(th_data["threshold"])

    print(f"Loaded Live Threshold τ = {threshold:.6f}")
    scaler = joblib.load(scaler_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = FGEAD(
        n_features=N_LIVE_FEATURES,
        embed_dim=64,
        n_heads=4,
        gcn_out=64,
        lstm_hidden=128,
        sparsity_threshold=0.3,
        dropout=0.2,
        sparsity_lambda=0.01,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Evaluate Normal Baseline Data
    print("\n[PART 1] Evaluating Normal Baseline Data (data/live_baseline.csv)...")
    df = pd.read_csv(baseline_csv)
    raw_feats = df[LIVE_FEATURES].values.astype(np.float32)
    total_samples = len(raw_feats)
    print(f"Total raw baseline samples: {total_samples}")

    scaled_feats = scaler.transform(raw_feats)

    window_size = 60
    windows = []
    for i in range(len(scaled_feats) - window_size + 1):
        windows.append(scaled_feats[i : i + window_size])
    windows_arr = np.array(windows, dtype=np.float32)  # (N, 60, 22)
    n_windows = len(windows_arr)
    print(f"Total evaluated sliding windows (W=60, S=1): {n_windows}")

    # Batched forward pass
    batch_size = 128
    scores = []
    t_start = time.perf_counter()

    with torch.no_grad():
        for b_start in range(0, n_windows, batch_size):
            b_end = min(b_start + batch_size, n_windows)
            batch_t = torch.tensor(windows_arr[b_start:b_end])
            _, _, step_scores = model(batch_t)  # (B, 59)
            max_scores = torch.max(step_scores, dim=1)[0].numpy()
            scores.extend(max_scores)

    eval_duration = time.perf_counter() - t_start
    scores = np.array(scores, dtype=np.float32)

    min_sc = float(np.min(scores))
    max_sc = float(np.max(scores))
    mean_sc = float(np.mean(scores))
    std_sc = float(np.std(scores))
    median_sc = float(np.median(scores))
    p95_sc = float(np.percentile(scores, 95))
    p99_sc = float(np.percentile(scores, 99))

    raw_above_threshold = int(np.sum(scores >= threshold))
    raw_fpr = (raw_above_threshold / n_windows) * 100.0

    # Evaluate with LiveEpisodeTracker (Hysteresis / Confirmation Gate)
    tracker = LiveEpisodeTracker(debounce_frames=2, min_persistence_frames=2)
    confirmed_episodes_count = 0
    suspicious_windows_count = 0

    for idx, sc in enumerate(scores):
        is_anom = bool(sc >= threshold)
        top_f = [{"feature": "nominal"}]
        ts = f"2026-10-02T18:{idx//60:02d}:{idx%60:02d}Z"
        tracker.update(is_anom, float(sc), threshold, ts, top_f)

    confirmed_episodes_count = len(tracker.completed_episodes)
    if tracker.current_episode and tracker.current_episode.get("is_confirmed", False):
        confirmed_episodes_count += 1

    episode_fpr = (confirmed_episodes_count / (n_windows / 60.0)) * 100.0  # per hour rate

    print("\n--- Normal Baseline Score Distribution ---")
    print(f"  Min Score      : {min_sc:.6f}")
    print(f"  Median Score   : {median_sc:.6f}")
    print(f"  Mean Score     : {mean_sc:.6f}")
    print(f"  Std Score      : {std_sc:.6f}")
    print(f"  P95 Score      : {p95_sc:.6f}")
    print(f"  P99 Score      : {p99_sc:.6f}")
    print(f"  Max Score      : {max_sc:.6f}")
    print(f"  Threshold τ    : {threshold:.6f}")
    print(f"  Raw Windows >= τ: {raw_above_threshold} / {n_windows} ({raw_fpr:.3f}%)")
    print(f"  Confirmed Anomaly Episodes: {confirmed_episodes_count}")
    print(f"  Confirmed Episode FPR: {confirmed_episodes_count / n_windows * 100:.3f}%")

    # 3. Controlled Anomaly Scenarios
    print("\n[PART 2] Evaluating Controlled Anomaly Scenarios...")
    scenarios = [
        {
            "name": "Scenario A: CPU Intensive Workload",
            "type": "CPU_STRESS",
            "feature_mods": {
                "cpu_percent": 96.5,
                "cpu_user_time_percent": 88.0,
                "cpu_system_time_percent": 8.5,
                "cpu_ctx_switches_per_sec": 85000.0,
            },
            "expected_features": ["cpu_percent", "cpu_user_time_percent", "cpu_ctx_switches_per_sec"],
        },
        {
            "name": "Scenario B: Disk Write Heavy Burst",
            "type": "DISK_WRITE_STRESS",
            "feature_mods": {
                "disk_write_bytes_per_sec": 250000000.0,  # 250 MB/s
                "disk_write_count_per_sec": 4500.0,
                "disk_usage_percent": 91.0,
            },
            "expected_features": ["disk_write_bytes_per_sec", "disk_write_count_per_sec", "disk_usage_percent"],
        },
        {
            "name": "Scenario C: Disk Read Flood",
            "type": "DISK_READ_STRESS",
            "feature_mods": {
                "disk_read_bytes_per_sec": 320000000.0,  # 320 MB/s
                "disk_read_count_per_sec": 5200.0,
            },
            "expected_features": ["disk_read_bytes_per_sec", "disk_read_count_per_sec"],
        },
        {
            "name": "Scenario D: Network Traffic Burst",
            "type": "NETWORK_BURST",
            "feature_mods": {
                "net_bytes_recv_per_sec": 85000000.0,  # 85 MB/s
                "net_packets_recv_per_sec": 65000.0,
            },
            "expected_features": ["net_bytes_recv_per_sec", "net_packets_recv_per_sec"],
        },
        {
            "name": "Scenario E: Combined CPU + Storage Contention",
            "type": "COMBINED_STRESS",
            "feature_mods": {
                "cpu_percent": 94.0,
                "cpu_user_time_percent": 75.0,
                "disk_write_bytes_per_sec": 180000000.0,
                "memory_percent": 89.0,
            },
            "expected_features": ["cpu_percent", "disk_write_bytes_per_sec", "memory_percent"],
        },
    ]

    scenario_results = []

    for sc_info in scenarios:
        # Construct nominal 60s background then inject 20s anomaly
        test_window_raw = np.tile(raw_feats[100:160], (1, 1))  # (60, 22)
        # Inject anomaly in last 20 timesteps
        for t in range(40, 60):
            for feat_name, mod_val in sc_info["feature_mods"].items():
                col_idx = LIVE_FEATURES.index(feat_name)
                test_window_raw[t, col_idx] = mod_val

        # Scale
        scaled_test = scaler.transform(test_window_raw)
        t_test = torch.tensor(scaled_test, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            preds_norm, attn, step_sc = model(t_test)
            sc_arr = step_sc.squeeze(0).numpy()
            max_sc = float(np.max(sc_arr))
            feat_residuals = np.mean(np.abs(preds_norm[0].numpy() - t_test[0, 1:].numpy()), axis=0)

        top_indices = np.argsort(feat_residuals)[::-1]
        top_feats = [LIVE_FEATURES[idx] for idx in top_indices[:3]]

        # Attribution consistency check
        overlap = set(sc_info["expected_features"]).intersection(set(top_feats))
        consistency = len(overlap) > 0

        detected = bool(max_sc >= threshold)
        delay_steps = int(np.argmax(sc_arr >= threshold)) + 1 if detected else -1

        res_dict = {
            "name": sc_info["name"],
            "type": sc_info["type"],
            "detected": detected,
            "max_score": round(max_sc, 4),
            "threshold": threshold,
            "detection_delay_steps": delay_steps,
            "top_contributing_features": top_feats,
            "expected_features": sc_info["expected_features"],
            "attribution_consistent": consistency,
        }
        scenario_results.append(res_dict)
        print(f"  -> {sc_info['name']}: Detected = {detected} | Score = {max_sc:.4f} (τ={threshold:.4f}) | Top Contributors = {top_feats} | Consistent = {consistency}")

    # 4. Accuracy & Confusion Matrix Metrics on Labeled Evaluation Dataset
    # Ground Truth:
    # - Normal baseline windows: Label = 0 (Normal)
    # - Controlled scenario windows: Label = 1 (Anomaly)
    y_true_normal = np.zeros(n_windows, dtype=int)
    y_pred_normal_raw = (scores >= threshold).astype(int)

    # For controlled anomalies (5 scenarios tested)
    n_anom_scenarios = len(scenarios)
    y_true_anom = np.ones(n_anom_scenarios, dtype=int)
    y_pred_anom = np.array([1 if r["detected"] else 0 for r in scenario_results], dtype=int)

    # Total Confusion Matrix
    TN = int(np.sum(y_pred_normal_raw == 0))
    FP = int(np.sum(y_pred_normal_raw == 1))
    TP = int(np.sum(y_pred_anom == 1))
    FN = int(np.sum(y_pred_anom == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    detection_rate = (TP / n_anom_scenarios) * 100.0

    print("\n--- Accuracy & Decision Metrics ---")
    print(f"  True Positives (TP) : {TP} / {n_anom_scenarios}")
    print(f"  False Positives (FP): {FP} / {n_windows}")
    print(f"  True Negatives (TN) : {TN} / {n_windows}")
    print(f"  False Negatives (FN): {FN} / {n_anom_scenarios}")
    print(f"  Detection Rate      : {detection_rate:.1f}%")
    print(f"  Precision           : {precision:.4f} (Raw Window Level)")
    print(f"  Recall              : {recall:.4f}")
    print(f"  F1 Score            : {f1:.4f}")
    print(f"  False Positive Rate : {fpr * 100:.3f}% (Raw Window Level)")
    print(f"  Confirmed FP Rate   : {confirmed_episodes_count / n_windows * 100:.3f}% (Episode Level)")

    # Save summary dictionary for report generation
    summary = {
        "normal_windows_count": n_windows,
        "normal_samples_count": total_samples,
        "score_statistics": {
            "min": min_sc,
            "median": median_sc,
            "mean": mean_sc,
            "std": std_sc,
            "p95": p95_sc,
            "p99": p99_sc,
            "max": max_sc,
            "threshold": threshold,
        },
        "raw_above_threshold_count": raw_above_threshold,
        "raw_fpr_percent": raw_fpr,
        "confirmed_episodes_count": confirmed_episodes_count,
        "confirmed_episode_fpr_percent": confirmed_episodes_count / n_windows * 100.0,
        "controlled_scenarios": scenario_results,
        "confusion_matrix": {
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr,
            "detection_rate": detection_rate,
        },
    }

    out_json = PROJECT_ROOT / "data" / "multihost_validation" / "phase9_evaluation_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Metrics saved to {out_json}")
    return summary


if __name__ == "__main__":
    evaluate_system()
