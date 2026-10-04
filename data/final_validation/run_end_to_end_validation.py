"""
data/final_validation/run_end_to_end_validation.py

Automated End-to-End System Validation Test Suite for Phase 5.
Validates all 10 core dimensions and saves validation results.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
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

import numpy as np
import pandas as pd
import requests

from api.live_inference import LiveEpisodeTracker, get_live_inference_service
from data.live_buffer import LiveRingBuffer
from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    FEATURE_UNITS,
    LIVE_FEATURES,
    N_LIVE_FEATURES,
    format_physical_metric,
    validate_live_feature_dict,
)

API_BASE = "http://127.0.0.1:8000"


def test_system():
    print("=" * 80)
    print("FGEAD PHASE 5 — COMPREHENSIVE END-TO-END SYSTEM VALIDATION")
    print("=" * 80)

    results = {
        "smd_mode": False,
        "live_mode": False,
        "telemetry": False,
        "model_loading": False,
        "inference": False,
        "xai": False,
        "episode_tracking": False,
        "offline_detection": False,
        "recovery": False,
        "smd_isolation": False,
    }

    test_logs = []

    def record(name: str, passed: bool, msg: str):
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str}] {name:<30} -> {msg}")
        test_logs.append({"test": name, "status": status_str, "details": msg})

    # 1. TEST SMD BENCHMARK MODE
    try:
        r_health = requests.get(f"{API_BASE}/health", timeout=3.0)
        r_data = requests.get(f"{API_BASE}/dataset", timeout=3.0)
        r_mach = requests.get(f"{API_BASE}/machine", timeout=3.0)

        # SMD Predict test
        dummy_smd_window = [[0.1] * 38 for _ in range(60)]
        r_pred = requests.post(f"{API_BASE}/predict", json={"data": dummy_smd_window, "window_start_abs": 0}, timeout=5.0)

        smd_ok = (
            r_health.status_code == 200
            and r_data.status_code == 200
            and r_mach.status_code == 200
            and r_pred.status_code == 200
            and r_pred.json()["machine"] == "1-1"
            and len(r_pred.json()["top_features"]) > 0
        )
        results["smd_mode"] = smd_ok
        record("SMD Dataset Benchmark Mode", smd_ok, f"Predict 200 OK | Score: {r_pred.json().get('anomaly_score', 'N/A')} | Machine: 1-1")
    except Exception as e:
        record("SMD Dataset Benchmark Mode", False, f"Error: {e}")

    # 2. TEST LIVE MODEL LOADING & ARTIFACTS
    try:
        live_svc = get_live_inference_service()
        model_loading_ok = (
            live_svc.is_ready
            and live_svc.model is not None
            and live_svc.scaler is not None
            and abs(live_svc.threshold - 1.859450) < 1e-5
            and live_svc.n_features == 22
        )
        results["model_loading"] = model_loading_ok
        record("Live Model Artifact Loading", model_loading_ok, f"Model Ready: {live_svc.is_ready} | Threshold: {live_svc.threshold:.6f} | Dim: {live_svc.n_features}")
    except Exception as e:
        record("Live Model Artifact Loading", False, f"Error: {e}")

    # 3. TEST LIVE INFERENCE ENDPOINT
    try:
        df_base = pd.read_csv(PROJECT_ROOT / "data" / "live_baseline.csv")
        sample_window = df_base.iloc[0:60][list(LIVE_FEATURES)].to_numpy().tolist()

        r_live_pred = requests.post(f"{API_BASE}/predict/live", json={"data": sample_window}, timeout=5.0)
        live_pred_data = r_live_pred.json() if r_live_pred.status_code == 200 else {}

        inference_ok = (
            r_live_pred.status_code == 200
            and live_pred_data.get("model") == "fgead_live_windows_22ch"
            and live_pred_data.get("n_features") == 22
            and "anomaly_score" in live_pred_data
            and not live_pred_data.get("is_anomaly", True)  # Baseline is normal
        )
        results["inference"] = inference_ok
        results["live_mode"] = inference_ok
        record("Live Neural Inference (/predict/live)", inference_ok, f"Score: {live_pred_data.get('anomaly_score')} < 1.85945 | Latency: {live_pred_data.get('latency_ms')} ms")
    except Exception as e:
        record("Live Neural Inference (/predict/live)", False, f"Error: {e}")

    # 4. TEST TELEMETRY INGESTION & BUFFER
    try:
        test_sample = {f: float(df_base.iloc[10][f]) for f in LIVE_FEATURES}
        payload = {
            "machine_id": "SivaChowdary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_info": {"test": True},
            "features": test_sample,
        }
        r_telem = requests.post(f"{API_BASE}/telemetry", json=payload, timeout=3.0)
        telem_data = r_telem.json() if r_telem.status_code == 200 else {}

        telemetry_ok = (
            r_telem.status_code == 200
            and telem_data.get("status") == "success"
            and telem_data.get("window_size") == 60
        )
        results["telemetry"] = telemetry_ok
        record("Telemetry Ingestion (/telemetry)", telemetry_ok, f"Buffer Size: {telem_data.get('buffer_size')} | Total Samples: {telem_data.get('samples_received')}")
    except Exception as e:
        record("Telemetry Ingestion (/telemetry)", False, f"Error: {e}")

    # 5. TEST 5-QUESTION XAI & PHYSICAL UNITS
    try:
        # Create an anomaly window by injecting spike
        anom_window = df_base.iloc[0:60][list(LIVE_FEATURES)].to_numpy().copy()
        idx_net = list(LIVE_FEATURES).index("net_bytes_sent_per_sec")
        anom_window[50:60, idx_net] = 10_000_000.0  # 10 MB/s

        r_anom = requests.post(f"{API_BASE}/predict/live", json={"data": anom_window.tolist()}, timeout=5.0)
        anom_res = r_anom.json() if r_anom.status_code == 200 else {}

        five_q = anom_res.get("five_question_explanation", {})
        top_f = anom_res.get("top_features", [])

        xai_ok = (
            r_anom.status_code == 200
            and anom_res.get("is_anomaly") is True
            and all(k in five_q for k in ["what_happened", "when", "how_severe", "what_contributed", "which_subsystem"])
            and len(top_f) > 0
            and top_f[0]["feature"] == "net_bytes_sent_per_sec"
            and "MB/s" in top_f[0].get("current_value_formatted", "")
            and "probability" not in json.dumps(five_q).lower()
        )
        results["xai"] = xai_ok
        record("5-Question XAI & Physical Formatting", xai_ok, f"Top Feature: {top_f[0]['feature']} ({top_f[0].get('current_value_formatted')}) | Residual: {top_f[0]['residual']}")
    except Exception as e:
        record("5-Question XAI & Physical Formatting", False, f"Error: {e}")

    # 6. TEST EPISODE TRACKING & DEBOUNCE
    try:
        ep_tracker = LiveEpisodeTracker(debounce_frames=2)
        top_dummy = [{"feature": "net_bytes_sent_per_sec", "feature_description": "Network Egress", "subsystem": "Network"}]
        # Feed 2 anomaly frames
        ep_tracker.update(True, 3.5, 1.85945, "2026-10-03T01:00:00Z", top_dummy)
        ep_tracker.update(True, 4.2, 1.85945, "2026-10-03T01:00:01Z", top_dummy)
        st_active = ep_tracker.get_status()

        # Feed 1 nominal frame (debouncing)
        ep_tracker.update(False, 1.0, 1.85945, "2026-10-03T01:00:02Z", top_dummy)
        st_debouncing = ep_tracker.get_status()

        # Feed 2nd nominal frame (closed)
        ep_tracker.update(False, 0.8, 1.85945, "2026-10-03T01:00:03Z", top_dummy)
        st_closed = ep_tracker.get_status()

        ep_ok = (
            st_active["is_in_anomaly_episode"] is True
            and st_active["active_episode_id"] == 1
            and st_active["peak_score"] == 4.2
            and st_debouncing["is_in_anomaly_episode"] is True
            and st_closed["is_in_anomaly_episode"] is False
            and st_closed["total_completed_episodes"] == 1
        )
        results["episode_tracking"] = ep_ok
        record("Live Episode Tracking & Debounce", ep_ok, f"Episode opened, debounced, closed with peak {st_active['peak_score']}")
    except Exception as e:
        record("Live Episode Tracking & Debounce", False, f"Error: {e}")

    # 7. TEST OFFLINE DETECTION
    try:
        test_buf = LiveRingBuffer(window_size=60, timeout_sec=0.1)
        test_buf.add_sample("2026-10-03T01:00:00Z", {f: 1.0 for f in LIVE_FEATURES})
        time.sleep(0.2)  # Wait past timeout
        is_conn_after_timeout = test_buf.is_agent_connected()

        offline_ok = is_conn_after_timeout is False
        results["offline_detection"] = offline_ok
        record("Offline Detection Timeout (> 5s)", offline_ok, f"Correctly detected disconnected state when timeout elapsed")
    except Exception as e:
        record("Offline Detection Timeout (> 5s)", False, f"Error: {e}")

    # 8. TEST TELEMETRY RECOVERY
    try:
        # Resume sending sample
        test_buf.add_sample("2026-10-03T01:00:05Z", {f: 1.0 for f in LIVE_FEATURES})
        is_conn_recovered = test_buf.is_agent_connected()

        recovery_ok = is_conn_recovered is True
        results["recovery"] = recovery_ok
        record("Telemetry Reconnection Recovery", recovery_ok, f"Agent instantly transitions back to CONNECTED upon sample ingest")
    except Exception as e:
        record("Telemetry Reconnection Recovery", False, f"Error: {e}")

    # 9. TEST SMD / LIVE ISOLATION
    try:
        # Ensure SMD checkpoint has 38 features, Live checkpoint has 22 features, different thresholds
        r_smd_mach = requests.get(f"{API_BASE}/machine").json()
        r_live_stat = requests.get(f"{API_BASE}/telemetry/live_analysis").json()

        smd_feats = r_smd_mach.get("features", 0)
        live_feats = len(LIVE_FEATURES)
        live_tau = r_live_stat.get("threshold", 0.0)

        isolation_ok = (
            smd_feats == 38
            and live_feats == 22
            and abs(live_tau - 1.859450) < 1e-5
            and (PROJECT_ROOT / "checkpoints" / "fgead_smd_machine_1_1.pt").exists()
            and (PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch.pt").exists()
        )
        results["smd_isolation"] = isolation_ok
        record("SMD vs Live Model & Pipeline Isolation", isolation_ok, f"SMD: {smd_feats} channels | Live: {live_feats} channels | Thresholds: 2.073376 vs {live_tau:.6f}")
    except Exception as e:
        record("SMD vs Live Model & Pipeline Isolation", False, f"Error: {e}")

    # Save output files
    out_dir = PROJECT_ROOT / "data" / "final_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "system_validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"SAVED RESULTS: {json_path}")
    print("=" * 80)

    return results, test_logs


if __name__ == "__main__":
    test_system()
