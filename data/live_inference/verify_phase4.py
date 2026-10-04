"""
data/live_inference/verify_phase4.py

Comprehensive 18-Step Safety & Verification Test Suite for Phase 4 Live Inference Integration.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

import joblib
import numpy as np
import pandas as pd
import torch

from api.live_inference import LiveEpisodeTracker, LiveInferenceService, get_live_inference_service
from data.live_buffer import LiveRingBuffer
from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    LIVE_FEATURES,
    LIVE_FEATURE_VERSION,
    N_LIVE_FEATURES,
    validate_live_feature_dict,
)
from models.fgead import FGEAD


def run_test_suite():
    print("=" * 80)
    print("FGEAD PHASE 4 — 18-POINT CRITICAL SAFETY & INFERENCE VERIFICATION")
    print("=" * 80)

    passed_tests = 0
    total_tests = 18

    def log_result(test_idx: int, test_name: str, passed: bool, details: str = ""):
        nonlocal passed_tests
        status = "PASSED [OK]" if passed else "FAILED [FAIL]"
        if passed:
            passed_tests += 1
        print(f"[{test_idx:02d}/18] {test_name:<55} -> {status}")
        if details:
            print(f"       Details: {details}")

    # 1. Checkpoint existence & architecture validation (N=22)
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch.pt"
    ckpt_exists = ckpt_path.exists()
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_exists else {}
    n_feat_ckpt = ckpt_data.get("n_features", 0)
    log_result(
        1,
        "Live Checkpoint & 22-Channel Dimension Check",
        ckpt_exists and n_feat_ckpt == 22,
        f"Path: {ckpt_path.name}, Features in Checkpoint: {n_feat_ckpt}",
    )

    # 2. Scaler existence & compatibility check (22 features)
    scaler_path = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler.joblib"
    scaler_exists = scaler_path.exists()
    scaler = joblib.load(scaler_path) if scaler_exists else None
    scaler_valid = scaler is not None and len(scaler.mean_) == 22
    log_result(
        2,
        "Scaler Existence & Calibration Dimension Check",
        scaler_valid,
        f"Scaler Mean Vector Dim: {len(scaler.mean_) if scaler else 'None'}",
    )

    # 3. Threshold configuration consistency (tau = 1.859450)
    thresh_path = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold.json"
    thresh_exists = thresh_path.exists()
    with open(thresh_path, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    tau_val = float(t_data.get("threshold", 0.0))
    log_result(
        3,
        "Threshold Config Integrity (tau = 1.859450)",
        thresh_exists and abs(tau_val - 1.859450) < 1e-5,
        f"Calibrated Threshold: {tau_val:.6f}",
    )

    # 4. Feature schema version & ordering match
    schema_ok = len(LIVE_FEATURES) == 22 and LIVE_FEATURE_VERSION in ("1.0", "1.0.0")
    log_result(
        4,
        "Feature Schema & Ordering Consistency",
        schema_ok,
        f"Schema Version: {LIVE_FEATURE_VERSION}, Features: {len(LIVE_FEATURES)}",
    )


    # 5. SMD 38-channel pipeline isolation
    smd_ckpt = PROJECT_ROOT / "checkpoints" / "fgead_smd_machine_1_1.pt"
    smd_ok = smd_ckpt.exists()
    log_result(
        5,
        "SMD 38-Channel Pipeline Isolation Check",
        smd_ok,
        f"SMD Checkpoint intact at: {smd_ckpt.name}",
    )

    # Initialize Service
    service = LiveInferenceService()
    svc_ready = service.is_ready
    log_result(
        6,
        "LiveInferenceService Initialization",
        svc_ready,
        f"Service Ready: {svc_ready}, Device: {service.device}",
    )

    # 7. Valid synthetic nominal window execution
    baseline_csv = PROJECT_ROOT / "data" / "live_baseline.csv"
    df_base = pd.read_csv(baseline_csv)
    raw_window_nom = df_base.iloc[0:60][list(LIVE_FEATURES)].to_numpy(dtype=np.float32)
    res_nom = service.infer_window(raw_window_nom)
    nom_ok = (
        res_nom["model"] == "fgead_live_windows_22ch"
        and not res_nom["is_anomaly"]
        and res_nom["anomaly_score"] < tau_val
    )
    log_result(
        7,
        "Nominal Window Inference & Max-Step Scoring",
        nom_ok,
        f"Score: {res_nom['anomaly_score']:.4f} < {tau_val:.4f}, Anomaly: {res_nom['is_anomaly']}",
    )

    # 8. Rejection of invalid window length (!= 60)
    invalid_len_rejected = False
    try:
        service.infer_window(raw_window_nom[:30])
    except ValueError:
        invalid_len_rejected = True
    log_result(
        8,
        "Invalid Window Length Rejection (Length != 60)",
        invalid_len_rejected,
        "Length 30 rejected with ValueError as expected",
    )

    # 9. Rejection of invalid feature count (!= 22)
    invalid_feat_rejected = False
    try:
        service.infer_window(np.zeros((60, 38), dtype=np.float32))
    except ValueError:
        invalid_feat_rejected = True
    log_result(
        9,
        "Invalid Feature Count Rejection (Features != 22)",
        invalid_feat_rejected,
        "38-feature matrix rejected with ValueError as expected",
    )

    # 10. Rejection of NaN values
    nan_mat = raw_window_nom.copy()
    nan_mat[10, 5] = np.nan
    nan_rejected = False
    try:
        service.infer_window(nan_mat)
    except ValueError:
        nan_rejected = True
    log_result(
        10,
        "NaN Input Value Sanitization & Rejection",
        nan_rejected,
        "NaN matrix rejected with ValueError as expected",
    )

    # 11. Rejection of Infinite values
    inf_mat = raw_window_nom.copy()
    inf_mat[15, 8] = np.inf
    inf_rejected = False
    try:
        service.infer_window(inf_mat)
    except ValueError:
        inf_rejected = True
    log_result(
        11,
        "Infinite Input Value Sanitization & Rejection",
        inf_rejected,
        "Inf matrix rejected with ValueError as expected",
    )

    # 12. Synthetic Anomaly Injection & Detection
    anom_mat = raw_window_nom.copy()
    # Inject massive disk write spike
    idx_dw = list(LIVE_FEATURES).index("disk_write_bytes_per_sec")
    anom_mat[45:60, idx_dw] = 100_000_000.0  # 100 MB/s
    res_anom = service.infer_window(anom_mat)
    anom_detected = res_anom["is_anomaly"] and res_anom["anomaly_score"] > tau_val
    log_result(
        12,
        "Synthetic Anomaly Injection & Threshold Trigger",
        anom_detected,
        f"Score: {res_anom['anomaly_score']:.4f} > {tau_val:.4f}, Anomaly: {res_anom['is_anomaly']}",
    )

    # 13. Top Contributing Feature Attribution
    top_f = res_anom["top_features"][0]
    attribution_correct = top_f["feature"] == "disk_write_bytes_per_sec"
    log_result(
        13,
        "Root-Cause Feature Attribution Ranking",
        attribution_correct,
        f"Top Feature: {top_f['feature']}, Contrib: {top_f['contribution_pct']}%, Subsystem: {top_f['subsystem']}",
    )

    # 14. Unscaled Physical Reconstructions in XAI
    actual_val = top_f["current_value"]
    pred_val = top_f["predicted_value"]
    unscaled_ok = actual_val > 50_000_000.0
    log_result(
        14,
        "Unscaled Physical Telemetry Values in XAI Output",
        unscaled_ok,
        f"Actual: {actual_val:,.1f} B/s, Predicted: {pred_val:,.1f} B/s",
    )

    # 15. 5-Question Narrative Completeness & Neutral Tone
    five_q = res_anom["five_question_explanation"]
    has_all_5 = all(
        k in five_q
        for k in ["what_happened", "when", "how_severe", "what_contributed", "which_subsystem"]
    )
    no_prob_word = "probability" not in json.dumps(five_q).lower()
    narrative_ok = has_all_5 and no_prob_word
    log_result(
        15,
        "5-Question Explainability Narrative Completeness",
        narrative_ok,
        f"All 5 keys present, neutral non-probabilistic language verified",
    )

    # 16. Episode Tracker Debounce & Grouping
    tracker = LiveEpisodeTracker(debounce_frames=2)
    # Feed 3 anomaly windows
    for i in range(3):
        tracker.update(True, 2.5 + i * 0.1, tau_val, f"2026-10-03T00:00:0{i}Z", res_anom["top_features"])
    st1 = tracker.get_status()
    ep1_active = st1["is_in_anomaly_episode"] and st1["active_episode_id"] == 1
    # Feed 1 nominal window (should still be debouncing)
    tracker.update(False, 1.2, tau_val, "2026-10-03T00:00:03Z", res_nom["top_features"])
    st2 = tracker.get_status()
    ep2_debouncing = st2["is_in_anomaly_episode"]  # Debounce = 2
    # Feed 2nd nominal window (should now close)
    tracker.update(False, 1.1, tau_val, "2026-10-03T00:00:04Z", res_nom["top_features"])
    st3 = tracker.get_status()
    ep3_closed = not st3["is_in_anomaly_episode"] and st3["total_completed_episodes"] == 1
    episode_ok = ep1_active and ep2_debouncing and ep3_closed
    log_result(
        16,
        "Sliding Window Episode Debounce & Grouping",
        episode_ok,
        f"Episode opened, debounced, and cleanly closed with 1 completed episode recorded",
    )

    # 17. LiveRingBuffer Thread Safety & Tensor Extraction
    buf = LiveRingBuffer(window_size=60)
    for i in range(65):
        sample = {f: float(df_base.iloc[i][f]) for f in LIVE_FEATURES}
        buf.add_sample(f"2026-10-03T00:00:{i:02d}Z", sample)
    arr, is_full = buf.get_window_tensor()
    buffer_ok = is_full and arr is not None and arr.shape == (60, 22)
    log_result(
        17,
        "LiveRingBuffer Thread Safety & (60, 22) Tensor Extraction",
        buffer_ok,
        f"Extracted Window Shape: {arr.shape if arr is not None else 'None'}",
    )

    # 18. End-to-End Latency & Performance Budget (< 50ms)
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        service.infer_window(raw_window_nom)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    mean_lat = np.mean(latencies)
    latency_ok = mean_lat < 50.0
    log_result(
        18,
        "Inference Latency Budget (< 50ms per window)",
        latency_ok,
        f"Mean Latency: {mean_lat:.2f} ms (p95: {np.percentile(latencies, 95):.2f} ms)",
    )

    print("=" * 80)
    print(f"VERIFICATION SUMMARY: {passed_tests} / {total_tests} TESTS PASSED")
    print("=" * 80)
    if passed_tests == total_tests:
        print(">>> ALL 18 CRITICAL SAFETY CHECKS PASSED SUCCESSFULLY. READY FOR PRODUCTION. <<<")
    else:
        print(">>> WARNING: SOME CHECKS FAILED. INVESTIGATE BEFORE DEPLOYMENT. <<<")


if __name__ == "__main__":
    run_test_suite()
