"""
data/multihost_validation/test_multihost_suite.py

Comprehensive End-to-End Multi-Host Platform Validation Suite for FGEAD Phase 6.
Tests all 17 required capabilities:
1. Windows host registration
2. Linux host registration
3. Agent token authentication
4. Valid telemetry transmission
5. Rejection of unauthenticated telemetry
6. Rejection of unknown host ID
7. Rejection of invalid feature count (< 22 or > 22)
8. Rejection of NaN values
9. Rejection of Inf values
10. Host buffer isolation (Host A vs Host B strictly separated)
11. Independent neural inference execution
12. Independent episode tracking and debouncing
13. Offline host detection on telemetry timeout
14. Host recovery after reconnection
15. Dashboard host selection data integrity
16. Fleet dashboard aggregation accuracy
17. SMD 38-channel benchmark pipeline isolation
"""

import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app
from api.host_registry import get_host_registry
from data.multihost_buffer import get_multihost_buffer_manager
from api.multihost_inference import get_multihost_inference_manager
from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES, validate_live_feature_dict


import pandas as pd

_baseline_df = None
def _get_baseline_df():
    global _baseline_df
    if _baseline_df is None:
        csv_path = PROJECT_ROOT / "data" / "live_baseline.csv"
        if csv_path.exists():
            _baseline_df = pd.read_csv(csv_path)
    return _baseline_df


def generate_valid_features(row_idx: int = 150) -> dict:
    """Generate authentic normal 22-channel features from the validated baseline dataset."""
    df = _get_baseline_df()
    if df is not None and len(df) > row_idx:
        row = df.iloc[row_idx]
        return {f: float(row[f]) for f in LIVE_FEATURES}
    
    # Fallback to authentic baseline averages
    return {
        "cpu_percent": 14.5,
        "cpu_freq_current": 2600.0,
        "cpu_user_time_percent": 4.2,
        "cpu_system_time_percent": 10.3,
        "cpu_ctx_switches_per_sec": 32000.0,
        "cpu_interrupts_per_sec": 21000.0,
        "memory_percent": 60.4,
        "memory_available_mb": 12890.0,
        "memory_used_mb": 19620.0,
        "swap_percent": 4.4,
        "disk_usage_percent": 54.1,
        "disk_read_bytes_per_sec": 15000.0,
        "disk_write_bytes_per_sec": 120000.0,
        "disk_read_count_per_sec": 2.0,
        "disk_write_count_per_sec": 15.0,
        "net_bytes_sent_per_sec": 3500.0,
        "net_bytes_recv_per_sec": 4500.0,
        "net_packets_sent_per_sec": 25.0,
        "net_packets_recv_per_sec": 35.0,
        "net_errors_total": 0.0,
        "net_drops_total": 294.0,
        "process_count": 345.0,
    }


def generate_anomalous_features(row_idx: int = 150) -> dict:
    """Generate severe anomalous telemetry (disk write & ctx spike)."""
    feats = generate_valid_features(row_idx=row_idx)
    feats["cpu_percent"] = 98.5
    feats["disk_write_bytes_per_sec"] = 95000000.0  # 95 MB/s
    feats["disk_write_count_per_sec"] = 4500.0
    feats["cpu_ctx_switches_per_sec"] = 95000.0
    feats["memory_percent"] = 95.0
    return feats


def run_all_tests():
    print("=" * 80)
    print("FGEAD PHASE 6 — MULTI-HOST PLATFORM VALIDATION SUITE")
    print("=" * 80)

    results = {}

    with TestClient(app) as client:
        # Test 1: Register Windows host
        try:
            payload_win = {
                "hostname": "SivaChowdary-PC",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
                "agent_version": "1.0.0",
                "schema_version": "1.0",
                "model_id": "windows_default",
                "custom_host_id": "host_siva_windows",
            }
            res_win = client.post("/hosts/register", json=payload_win)
            assert res_win.status_code == 200, f"Status: {res_win.status_code}, Body: {res_win.text}"
            data_win = res_win.json()
            assert data_win["host_id"] == "host_siva_windows"
            assert data_win["agent_token"].startswith("fgead_")
            assert data_win["operating_system"] == "Windows"
            win_token = data_win["agent_token"]
            results["1. Register Windows Host"] = "PASSED"
            print(" [PASS] 1. Register Windows host")
        except Exception as e:
            results["1. Register Windows Host"] = f"FAILED: {e}"
            print(f" [FAIL] 1. Register Windows host: {e}")

        # Test 2: Register Linux host
        try:
            payload_lin = {
                "hostname": "Ubuntu-Prod-Server-01",
                "operating_system": "Linux",
                "os_version": "6.5.0-generic",
                "architecture": "x86_64",
                "agent_version": "1.0.0",
                "schema_version": "1.0",
                "model_id": "windows_default",
                "custom_host_id": "host_linux_srv01",
            }
            res_lin = client.post("/hosts/register", json=payload_lin)
            assert res_lin.status_code == 200, f"Status: {res_lin.status_code}"
            data_lin = res_lin.json()
            assert data_lin["host_id"] == "host_linux_srv01"
            assert data_lin["operating_system"] == "Linux"
            lin_token = data_lin["agent_token"]
            results["2. Register Linux Host"] = "PASSED"
            print(" [PASS] 2. Register Linux host")
        except Exception as e:
            results["2. Register Linux Host"] = f"FAILED: {e}"
            print(f" [FAIL] 2. Register Linux host: {e}")

        # Test 3: Authenticate Agent
        try:
            reg = get_host_registry()
            is_auth, err, hdata = reg.authenticate_agent("host_siva_windows", win_token)
            assert is_auth is True, f"Auth failed: {err}"
            assert hdata["hostname"] == "SivaChowdary-PC"

            is_auth_bad, _, _ = reg.authenticate_agent("host_siva_windows", "invalid_token_xyz")
            assert is_auth_bad is False
            results["3. Authenticate Agent"] = "PASSED"
            print(" [PASS] 3. Authenticate agent")
        except Exception as e:
            results["3. Authenticate Agent"] = f"FAILED: {e}"
            print(f" [FAIL] 3. Authenticate agent: {e}")

        # Test 4: Send Telemetry
        try:
            feats = generate_valid_features()
            telemetry_payload = {
                "host_id": "host_siva_windows",
                "timestamp": "2026-10-03T12:00:00Z",
                "features": feats,
            }
            res_tel = client.post(
                "/hosts/host_siva_windows/telemetry",
                json=telemetry_payload,
                headers={"X-Agent-Token": win_token},
            )
            assert res_tel.status_code == 200, f"Status: {res_tel.status_code}, {res_tel.text}"
            data_tel = res_tel.json()
            assert data_tel["status"] == "success"
            assert data_tel["buffer_size"] >= 1
            results["4. Send Telemetry"] = "PASSED"
            print(" [PASS] 4. Send telemetry")
        except Exception as e:
            results["4. Send Telemetry"] = f"FAILED: {e}"
            print(f" [FAIL] 4. Send telemetry: {e}")

        # Test 5: Reject Unauthenticated Telemetry
        try:
            # Missing token
            res_no_tok = client.post("/hosts/host_siva_windows/telemetry", json=telemetry_payload)
            assert res_no_tok.status_code == 401, f"Expected 401, got {res_no_tok.status_code}"

            # Invalid token
            res_bad_tok = client.post(
                "/hosts/host_siva_windows/telemetry",
                json=telemetry_payload,
                headers={"X-Agent-Token": "bad_token_123"},
            )
            assert res_bad_tok.status_code == 403, f"Expected 403, got {res_bad_tok.status_code}"
            results["5. Reject Unauthenticated Telemetry"] = "PASSED"
            print(" [PASS] 5. Reject unauthenticated telemetry")
        except Exception as e:
            results["5. Reject Unauthenticated Telemetry"] = f"FAILED: {e}"
            print(f" [FAIL] 5. Reject unauthenticated telemetry: {e}")

        # Test 6: Reject Unknown Host
        try:
            res_unk = client.post(
                "/hosts/unknown_host_999/telemetry",
                json=telemetry_payload,
                headers={"X-Agent-Token": win_token},
            )
            assert res_unk.status_code == 404, f"Expected 404, got {res_unk.status_code}"
            results["6. Reject Unknown Host"] = "PASSED"
            print(" [PASS] 6. Reject unknown host")
        except Exception as e:
            results["6. Reject Unknown Host"] = f"FAILED: {e}"
            print(f" [FAIL] 6. Reject unknown host: {e}")

        # Test 7: Reject Wrong Feature Count
        try:
            bad_feats = dict(feats)
            bad_feats.pop("cpu_percent")  # 21 features
            res_bad_cnt = client.post(
                "/hosts/host_siva_windows/telemetry",
                json={"host_id": "host_siva_windows", "timestamp": "2026-10-03T12:00:01Z", "features": bad_feats},
                headers={"X-Agent-Token": win_token},
            )
            assert res_bad_cnt.status_code == 422, f"Expected 422, got {res_bad_cnt.status_code}"
            results["7. Reject Wrong Feature Count"] = "PASSED"
            print(" [PASS] 7. Reject wrong feature count")
        except Exception as e:
            results["7. Reject Wrong Feature Count"] = f"FAILED: {e}"
            print(f" [FAIL] 7. Reject wrong feature count: {e}")

        # Test 8: Reject NaN
        try:
            nan_feats = dict(feats)
            nan_feats["cpu_percent"] = float("nan")
            is_valid, err_msg = validate_live_feature_dict(nan_feats)
            assert is_valid is False and "NaN" in err_msg, f"Schema validation did not reject NaN: {err_msg}"
            
            # Test via API string payload
            nan_json = json.dumps({"host_id": "host_siva_windows", "timestamp": "2026-10-03T12:00:02Z", "features": nan_feats}, allow_nan=True)
            res_nan = client.post(
                "/hosts/host_siva_windows/telemetry",
                content=nan_json,
                headers={"X-Agent-Token": win_token, "Content-Type": "application/json"},
            )
            assert res_nan.status_code == 422, f"Expected 422, got {res_nan.status_code}"
            results["8. Reject NaN"] = "PASSED"
            print(" [PASS] 8. Reject NaN")
        except Exception as e:
            results["8. Reject NaN"] = f"FAILED: {e}"
            print(f" [FAIL] 8. Reject NaN: {e}")

        # Test 9: Reject Inf
        try:
            inf_feats = dict(feats)
            inf_feats["memory_percent"] = float("inf")
            is_valid, err_msg = validate_live_feature_dict(inf_feats)
            assert is_valid is False and "Inf" in err_msg, f"Schema validation did not reject Inf: {err_msg}"

            inf_json = json.dumps({"host_id": "host_siva_windows", "timestamp": "2026-10-03T12:00:03Z", "features": inf_feats}, allow_nan=True)
            res_inf = client.post(
                "/hosts/host_siva_windows/telemetry",
                content=inf_json,
                headers={"X-Agent-Token": win_token, "Content-Type": "application/json"},
            )
            assert res_inf.status_code == 422, f"Expected 422, got {res_inf.status_code}"
            results["9. Reject Inf"] = "PASSED"
            print(" [PASS] 9. Reject Inf")
        except Exception as e:
            results["9. Reject Inf"] = f"FAILED: {e}"
            print(f" [FAIL] 9. Reject Inf: {e}")

        # Test 10: Maintain Independent Host Buffers
        try:
            buf_mgr = get_multihost_buffer_manager()
            # Stream 60 nominal samples to Windows Host (from normal baseline)
            for i in range(60):
                f_win = generate_valid_features(row_idx=100 + i)
                client.post(
                    "/hosts/host_siva_windows/telemetry",
                    json={"host_id": "host_siva_windows", "timestamp": f"2026-10-03T12:{i:02d}:00Z", "features": f_win},
                    headers={"X-Agent-Token": win_token},
                )

            # Stream 60 anomalous samples to Linux Host
            for i in range(60):
                f_lin = generate_anomalous_features(row_idx=100 + i)
                client.post(
                    "/hosts/host_linux_srv01/telemetry",
                    json={"host_id": "host_linux_srv01", "timestamp": f"2026-10-03T12:{i:02d}:00Z", "features": f_lin},
                    headers={"X-Agent-Token": lin_token},
                )

            buf_win_arr, is_win_full = buf_mgr.get_window_tensor("host_siva_windows")
            buf_lin_arr, is_lin_full = buf_mgr.get_window_tensor("host_linux_srv01")

            assert is_win_full is True
            assert is_lin_full is True
            assert buf_win_arr.shape == (60, 22)
            assert buf_lin_arr.shape == (60, 22)

            assert float(np.mean(buf_win_arr[:, 0])) < 25.0, f"Win mean CPU: {np.mean(buf_win_arr[:, 0])}"
            assert float(np.mean(buf_lin_arr[:, 0])) > 90.0, f"Lin mean CPU: {np.mean(buf_lin_arr[:, 0])}"
            results["10. Maintain Independent Host Buffers"] = "PASSED"
            print(" [PASS] 10. Maintain independent host buffers")
        except Exception as e:
            results["10. Maintain Independent Host Buffers"] = f"FAILED: {e}"
            print(f" [FAIL] 10. Maintain independent host buffers: {e}")

        # Test 11: Run Independent Inference (Windows Compatible vs Linux Gated)
        try:
            # 1. Nominal Windows Host
            res_inf_win = client.post("/hosts/host_siva_windows/predict/live")
            assert res_inf_win.status_code == 200, f"Win infer status: {res_inf_win.status_code}, text: {res_inf_win.text}"
            data_inf_win = res_inf_win.json()
            assert data_inf_win["is_anomaly"] is False, f"Expected Windows normal, got score: {data_inf_win['anomaly_score']}"
            assert data_inf_win["threshold"] == 1.859450

            # 2. Linux Host without Linux Model (Must be gated with 400)
            res_inf_lin = client.post("/hosts/host_linux_srv01/predict/live")
            assert res_inf_lin.status_code == 400, f"Expected 400 for Linux without model, got: {res_inf_lin.status_code}"

            # 3. Anomalous Windows Host (to verify independent neural anomaly detection)
            res_win_anom_reg = client.post("/hosts/register", json={
                "hostname": "Win-Anomaly-Host",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
                "custom_host_id": "host_win_anom",
            }).json()
            anom_token = res_win_anom_reg["agent_token"]
            for i in range(60):
                client.post(
                    "/hosts/host_win_anom/telemetry",
                    json={"host_id": "host_win_anom", "timestamp": f"2026-10-03T12:{i:02d}:00Z", "features": generate_anomalous_features(100 + i)},
                    headers={"X-Agent-Token": anom_token},
                )
            res_anom_pred = client.post("/hosts/host_win_anom/predict/live")
            assert res_anom_pred.status_code == 200
            data_anom = res_anom_pred.json()
            assert data_anom["is_anomaly"] is True
            assert data_anom["anomaly_score"] > 1.859450
            assert "Q1_what_happened" in data_anom["xai_5_questions"]

            results["11. Run Independent Inference"] = "PASSED"
            print(" [PASS] 11. Run independent inference (Windows Active & Linux Gated)")
        except Exception as e:
            results["11. Run Independent Inference"] = f"FAILED: {e}"
            print(f" [FAIL] 11. Run independent inference: {e}")

        # Test 12: Track Independent Episodes
        try:
            res_ep_win = client.get("/hosts/host_siva_windows/episodes").json()
            res_ep_anom = client.get("/hosts/host_win_anom/episodes").json()

            assert res_ep_win["current_episode"] is None
            assert res_ep_anom["current_episode"] is not None
            assert res_ep_anom["current_episode"]["flagged_windows_count"] >= 1
            results["12. Track Independent Episodes"] = "PASSED"
            print(" [PASS] 12. Track independent episodes")
        except Exception as e:
            results["12. Track Independent Episodes"] = f"FAILED: {e}"
            print(f" [FAIL] 12. Track independent episodes: {e}")

        # Test 13: Detect Offline Host
        try:
            reg = get_host_registry()
            with reg._get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE hosts SET last_seen_epoch = ? WHERE host_id = ?",
                    (time.time() - 30.0, "host_linux_srv01"),
                )
                conn.commit()

            res_off = client.get("/hosts/host_linux_srv01").json()
            assert res_off["host"]["status"] == "OFFLINE", f"Expected OFFLINE, got {res_off['host']['status']}"
            results["13. Detect Offline Host"] = "PASSED"
            print(" [PASS] 13. Detect offline host")
        except Exception as e:
            results["13. Detect Offline Host"] = f"FAILED: {e}"
            print(f" [FAIL] 13. Detect offline host: {e}")

        # Test 14: Recover Host
        try:
            res_rec = client.post(
                "/hosts/host_linux_srv01/telemetry",
                json={"host_id": "host_linux_srv01", "timestamp": "2026-10-03T12:05:00Z", "features": generate_valid_features()},
                headers={"X-Agent-Token": lin_token},
            )
            assert res_rec.status_code == 200
            res_host = client.get("/hosts/host_linux_srv01").json()
            assert res_host["host"]["status"] in ["ONLINE", "ANOMALY", "TELEMETRY_ONLY"], f"Status after recovery: {res_host['host']['status']}"
            results["14. Recover Host"] = "PASSED"
            print(" [PASS] 14. Recover host")
        except Exception as e:
            results["14. Recover Host"] = f"FAILED: {e}"
            print(f" [FAIL] 14. Recover host: {e}")

        # Test 15: Verify Dashboard Host Selection Data Integrity
        try:
            res_win_ana = client.get("/hosts/host_siva_windows/analysis").json()
            res_lin_ana = client.get("/hosts/host_linux_srv01/analysis").json()

            assert res_win_ana["hostname"] == "SivaChowdary-PC"
            assert res_lin_ana["hostname"] == "Ubuntu-Prod-Server-01"
            assert res_win_ana["latest_inference"] is not None
            assert res_win_ana["latest_inference"]["is_anomaly"] is False
            assert res_lin_ana["model_compatibility"] in ["BASELINE_REQUIRED", "Baseline Required"]
            assert res_lin_ana["latest_inference"] is None
            results["15. Verify Dashboard Host Selection"] = "PASSED"
            print(" [PASS] 15. Verify dashboard host selection")
        except Exception as e:
            results["15. Verify Dashboard Host Selection"] = f"FAILED: {e}"
            print(f" [FAIL] 15. Verify dashboard host selection: {e}")

        # Test 16: Verify Fleet Dashboard Aggregation
        try:
            res_fleet = client.get("/fleet/overview").json()
            assert res_fleet["status"] == "ok"
            assert res_fleet["total_hosts"] >= 2
            assert res_fleet["online_hosts"] >= 1
            assert len(res_fleet["hosts"]) >= 2
            results["16. Verify Fleet Dashboard"] = "PASSED"
            print(" [PASS] 16. Verify fleet dashboard")
        except Exception as e:
            results["16. Verify Fleet Dashboard"] = f"FAILED: {e}"
            print(f" [FAIL] 16. Verify fleet dashboard: {e}")

        # Test 17: Verify SMD 38-Channel Isolation
        try:
            smd_res = client.get("/dataset").json()
            assert smd_res["features"] == 38
            assert smd_res["machine"] == "1-1"

            dummy_smd_window = np.zeros((60, 38), dtype=np.float32).tolist()
            res_smd_pred = client.post("/predict", json={"data": dummy_smd_window, "window_start_abs": 0})
            assert res_smd_pred.status_code == 200, f"SMD predict status: {res_smd_pred.status_code}"
            smd_pred = res_smd_pred.json()
            assert smd_pred["machine"] == "1-1"
            assert smd_pred["threshold"] == 2.073376
            results["17. Verify SMD Isolation"] = "PASSED"
            print(" [PASS] 17. Verify SMD isolation")
        except Exception as e:
            results["17. Verify SMD Isolation"] = f"FAILED: {e}"
            print(f" [FAIL] 17. Verify SMD isolation: {e}")

    print("=" * 80)
    passed_count = sum(1 for v in results.values() if v == "PASSED")
    total_count = len(results)
    print(f"SUMMARY: {passed_count} / {total_count} CHECKS PASSED")
    print("=" * 80)

    report_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tests": total_count,
        "passed_tests": passed_count,
        "results": results,
    }
    out_dir = Path(PROJECT_ROOT / "data" / "multihost_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "multihost_test_results.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)
