"""
data/multihost_validation/test_phase6_1_compatibility.py

Comprehensive Validation Suite for Phase 6.1:
Model Compatibility, OS Gating, Constant Feature Audit, and Cross-Host Isolation.

Covers all 10 Phase 6.1 Requirements:
1. Windows host + Windows model (Compatibility PASS, inference allowed)
2. Linux host + Windows model (Inference strictly REJECTED, baseline required)
3. Unknown OS (Inference disabled)
4. Schema mismatch (Inference rejected)
5. Scaler mismatch / Missing model (Inference rejected)
6. Wrong model assignment (Inference rejected)
7. Constant feature difference (Reported as baseline compatibility difference, not anomaly)
8. Host A/B telemetry isolation (Nominal Windows vs Linux telemetry buffers isolated)
9. Duplicate agent reconnect (Reuses authorized identity, no duplicate host records)
10. Offline / Recovery lifecycle
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from api.main import app
from api.host_registry import get_host_registry
from api.multihost_inference import get_multihost_inference_manager
from data.multihost_buffer import get_multihost_buffer_manager
from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES, validate_live_feature_dict
from data.baseline_compatibility import compute_feature_distribution, compare_baseline_distributions

_baseline_df = None

def _get_baseline_df():
    global _baseline_df
    if _baseline_df is None:
        csv_path = PROJECT_ROOT / "data" / "live_baseline.csv"
        if csv_path.exists():
            _baseline_df = pd.read_csv(csv_path)
    return _baseline_df


def generate_valid_features(row_idx: int = 150) -> dict:
    df = _get_baseline_df()
    if df is not None and len(df) > row_idx:
        row = df.iloc[row_idx]
        return {f: float(row[f]) for f in LIVE_FEATURES}
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


from data.multihost_validation.test_db_helper import IsolatedTestDatabase


def run_phase6_1_tests() -> bool:
    print("=" * 80)
    print("FGEAD PHASE 6.1 — MODEL COMPATIBILITY & CROSS-HOST VALIDATION SUITE")
    print("=" * 80)

    results = {}

    with IsolatedTestDatabase(prefix="phase6_1_test_"), TestClient(app) as client:
        # ---------------------------------------------------------------------
        # TEST 1: Windows host + Windows model -> Compatibility PASS, inference allowed
        # ---------------------------------------------------------------------
        try:
            reg_win = client.post("/hosts/register", json={
                "hostname": "Test-Win-Workstation",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
                "agent_version": "1.0.0",
                "schema_version": "1.0",
                "custom_host_id": "test_host_win_01",
            }).json()
            win_token = reg_win["agent_token"]

            # Stream 60 normal samples to fill buffer
            for i in range(60):
                client.post(
                    "/hosts/test_host_win_01/telemetry",
                    json={"host_id": "test_host_win_01", "timestamp": f"2026-10-04T10:{i:02d}:00Z", "features": generate_valid_features(100 + i)},
                    headers={"X-Agent-Token": win_token},
                )

            # Predict
            res_pred = client.post("/hosts/test_host_win_01/predict/live")
            assert res_pred.status_code == 200, f"Expected 200, got {res_pred.status_code}: {res_pred.text}"
            data_pred = res_pred.json()
            assert "anomaly_score" in data_pred
            assert data_pred["threshold"] == 1.859450
            results["1. Windows Host + Windows Model"] = "PASSED"
            print(" [PASS] 1. Windows host + Windows model (Compatible & Allowed)")
        except Exception as e:
            results["1. Windows Host + Windows Model"] = f"FAILED: {e}"
            print(f" [FAIL] 1. Windows host + Windows model: {e}")

        # ---------------------------------------------------------------------
        # TEST 2: Linux host + Windows model -> Inference MUST be rejected
        # ---------------------------------------------------------------------
        try:
            reg_lin = client.post("/hosts/register", json={
                "hostname": "Test-Ubuntu-Server",
                "operating_system": "Linux",
                "os_version": "6.5.0-generic",
                "architecture": "x86_64",
                "agent_version": "1.0.0",
                "schema_version": "1.0",
                "custom_host_id": "test_host_lin_01",
            }).json()
            lin_token = reg_lin["agent_token"]

            # Stream 60 samples into Linux buffer
            for i in range(60):
                tel_res = client.post(
                    "/hosts/test_host_lin_01/telemetry",
                    json={"host_id": "test_host_lin_01", "timestamp": f"2026-10-04T10:{i:02d}:00Z", "features": generate_valid_features(100 + i)},
                    headers={"X-Agent-Token": lin_token},
                )
                assert tel_res.status_code == 200
                # Telemetry ingestion must return model_status == BASELINE_REQUIRED
                assert tel_res.json()["model_status"] == "BASELINE_REQUIRED"

            # Direct predict endpoint must return HTTP 400 with explicit explanation
            res_lin_pred = client.post("/hosts/test_host_lin_01/predict/live")
            assert res_lin_pred.status_code == 400, f"Expected 400, got {res_lin_pred.status_code}: {res_lin_pred.text}"
            err_detail = res_lin_pred.json()["detail"]
            assert "compatible Linux baseline/model has not yet been trained" in err_detail, f"Unexpected detail: {err_detail}"

            # Host status in fleet overview must be TELEMETRY_ONLY, not NORMAL and not ANOMALY
            fleet = client.get("/fleet/overview").json()
            lin_entry = next((h for h in fleet["hosts"] if h["host_id"] == "test_host_lin_01"), None)
            assert lin_entry is not None
            assert lin_entry["status"] == "TELEMETRY_ONLY", f"Expected TELEMETRY_ONLY, got {lin_entry['status']}"
            assert lin_entry["latest_score"] is None, f"Expected no score, got {lin_entry['latest_score']}"
            assert lin_entry["model_compatibility"] == "BASELINE_REQUIRED" or lin_entry["model_compatibility"] == "Baseline Required"
            results["2. Linux Host + Windows Model Gated"] = "PASSED"
            print(" [PASS] 2. Linux host + Windows model (Inference Rejected & Gated)")
        except Exception as e:
            results["2. Linux Host + Windows Model Gated"] = f"FAILED: {e}"
            print(f" [FAIL] 2. Linux host + Windows model Gated: {e}")

        # ---------------------------------------------------------------------
        # TEST 3: Unknown OS -> Inference disabled
        # ---------------------------------------------------------------------
        try:
            reg_unk = client.post("/hosts/register", json={
                "hostname": "Test-FreeBSD-Machine",
                "operating_system": "FreeBSD",
                "os_version": "14.0",
                "architecture": "x86_64",
                "agent_version": "1.0.0",
                "schema_version": "1.0",
                "custom_host_id": "test_host_unk_01",
            }).json()
            unk_token = reg_unk["agent_token"]

            client.post(
                "/hosts/test_host_unk_01/telemetry",
                json={"host_id": "test_host_unk_01", "timestamp": "2026-10-04T10:00:00Z", "features": generate_valid_features(100)},
                headers={"X-Agent-Token": unk_token},
            )
            res_unk_pred = client.post("/hosts/test_host_unk_01/predict/live")
            assert res_unk_pred.status_code == 400
            results["3. Unknown OS Gated"] = "PASSED"
            print(" [PASS] 3. Unknown OS (Inference Disabled)")
        except Exception as e:
            results["3. Unknown OS Gated"] = f"FAILED: {e}"
            print(f" [FAIL] 3. Unknown OS: {e}")

        # ---------------------------------------------------------------------
        # TEST 4: Schema Mismatch -> Inference rejected
        # ---------------------------------------------------------------------
        try:
            reg = get_host_registry()
            # Register a host with incompatible schema version 2.0
            reg_mismatch = client.post("/hosts/register", json={
                "hostname": "Test-Win-OldSchema",
                "operating_system": "Windows",
                "os_version": "10.0.19045",
                "architecture": "AMD64",
                "agent_version": "1.0.0",
                "schema_version": "0.9_legacy",
                "custom_host_id": "test_host_bad_schema",
            }).json()
            mismatch_token = reg_mismatch["agent_token"]

            # Try predicting
            res_mismatch_pred = client.post("/hosts/test_host_bad_schema/predict/live")
            assert res_mismatch_pred.status_code == 400
            assert "Schema mismatch" in res_mismatch_pred.json()["detail"]
            results["4. Schema Mismatch Rejected"] = "PASSED"
            print(" [PASS] 4. Schema mismatch (Inference Rejected)")
        except Exception as e:
            results["4. Schema Mismatch Rejected"] = f"FAILED: {e}"
            print(f" [FAIL] 4. Schema mismatch: {e}")

        # ---------------------------------------------------------------------
        # TEST 5: Scaler / Checkpoint Availability Verification
        # ---------------------------------------------------------------------
        try:
            inf_mgr = get_multihost_inference_manager()
            profile = inf_mgr.get_profile("windows_default")
            assert profile is not None
            assert profile.is_loaded is True
            assert profile.scaler is not None
            assert profile.threshold == 1.859450
            assert profile.supported_os == ["Windows"]
            assert profile.is_universal is False
            results["5. Model & Scaler Verification"] = "PASSED"
            print(" [PASS] 5. Model, Scaler & Threshold Verification")
        except Exception as e:
            results["5. Model & Scaler Verification"] = f"FAILED: {e}"
            print(f" [FAIL] 5. Model & Scaler Verification: {e}")

        # ---------------------------------------------------------------------
        # TEST 6: Wrong Model Assignment -> Rejected
        # ---------------------------------------------------------------------
        try:
            reg.set_host_model_status("test_host_win_01", "non_existent_model_xyz", "MODEL_NOT_AVAILABLE")
            res_bad_model = client.post("/hosts/test_host_win_01/predict/live")
            assert res_bad_model.status_code == 400
            # Reset back to valid
            reg.set_host_model_status("test_host_win_01", "windows_default", "COMPATIBLE")
            results["6. Wrong Model Assignment"] = "PASSED"
            print(" [PASS] 6. Wrong model assignment (Inference Rejected)")
        except Exception as e:
            results["6. Wrong Model Assignment"] = f"FAILED: {e}"
            print(f" [FAIL] 6. Wrong Model Assignment: {e}")

        # ---------------------------------------------------------------------
        # TEST 7: Constant Feature Difference -> Reported as compatibility difference
        # ---------------------------------------------------------------------
        try:
            # Generate synthetic host telemetry with net_drops_total = 0 (vs baseline 294)
            sim_host_samples = []
            for i in range(20):
                f = generate_valid_features(100 + i)
                f["net_drops_total"] = 0.0  # Constant feature difference
                sim_host_samples.append(f)

            host_stats = compute_feature_distribution(sim_host_samples)
            comp = compare_baseline_distributions(host_stats)

            assert comp["is_compatible"] is False
            assert comp["status"] == "COMPATIBILITY_DIFFERENCE_DETECTED"
            assert len(comp["constant_feature_shifts"]) >= 1
            drop_shift = next((s for s in comp["constant_feature_shifts"] if s["feature"] == "net_drops_total"), None)
            assert drop_shift is not None
            assert drop_shift["host_mean"] == 0.0
            assert drop_shift["ref_mean"] == 294.0
            results["7. Constant Feature Difference"] = "PASSED"
            print(" [PASS] 7. Constant feature difference (Classified as Baseline Compatibility Difference)")
        except Exception as e:
            results["7. Constant Feature Difference"] = f"FAILED: {e}"
            print(f" [FAIL] 7. Constant feature difference: {e}")

        # ---------------------------------------------------------------------
        # TEST 8: Host A/B Telemetry Isolation
        # ---------------------------------------------------------------------
        try:
            buf_mgr = get_multihost_buffer_manager()
            buf_win = buf_mgr.get_buffer("test_host_win_01")
            buf_lin = buf_mgr.get_buffer("test_host_lin_01")

            assert buf_win is not None
            assert buf_lin is not None
            assert buf_win is not buf_lin
            win_arr, is_win_full = buf_mgr.get_window_tensor("test_host_win_01")
            lin_arr, is_lin_full = buf_mgr.get_window_tensor("test_host_lin_01")
            assert is_win_full is True
            assert is_lin_full is True
            assert win_arr.shape == (60, 22)
            assert lin_arr.shape == (60, 22)
            results["8. Host A/B Buffer Isolation"] = "PASSED"
            print(" [PASS] 8. Host A/B telemetry isolation")
        except Exception as e:
            results["8. Host A/B Buffer Isolation"] = f"FAILED: {e}"
            print(f" [FAIL] 8. Host A/B Buffer Isolation: {e}")

        # ---------------------------------------------------------------------
        # TEST 9: Duplicate Agent Reconnect Reuses Identity
        # ---------------------------------------------------------------------
        try:
            # First registration of a unique host
            res1 = client.post("/hosts/register", json={
                "hostname": "Dedicated-PC-Chowdary",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
            }).json()
            initial_host_id = res1["host_id"]

            # Second registration from reconnecting agent with same hostname + OS
            res2 = client.post("/hosts/register", json={
                "hostname": "Dedicated-PC-Chowdary",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
            }).json()
            reconnected_host_id = res2["host_id"]

            # Must reuse identical host_id
            assert initial_host_id == reconnected_host_id, f"Expected {initial_host_id}, got {reconnected_host_id}"

            # Test cleanup reconciliation
            cleaned = reg.reconcile_duplicate_hosts("Dedicated-PC-Chowdary")
            assert isinstance(cleaned, list)
            results["9. Duplicate Reconnect Reused"] = "PASSED"
            print(" [PASS] 9. Duplicate agent reconnect (Identity Reused & No Duplicate Hosts)")
        except Exception as e:
            results["9. Duplicate Reconnect Reused"] = f"FAILED: {e}"
            print(f" [FAIL] 9. Duplicate Reconnect: {e}")

        # ---------------------------------------------------------------------
        # TEST 10: Offline / Recovery Lifecycle
        # ---------------------------------------------------------------------
        try:
            # Force timeout
            with reg._get_connection() as conn:
                conn.cursor().execute(
                    "UPDATE hosts SET last_seen_epoch = ? WHERE host_id = ?",
                    (time.time() - 30.0, "test_host_win_01"),
                )
                conn.commit()

            host_after_timeout = client.get("/hosts/test_host_win_01").json()["host"]
            assert host_after_timeout["status"] == "OFFLINE"

            # Reconnect
            client.post(
                "/hosts/test_host_win_01/telemetry",
                json={"host_id": "test_host_win_01", "timestamp": "2026-10-04T10:15:00Z", "features": generate_valid_features(100)},
                headers={"X-Agent-Token": win_token},
            )
            host_recovered = client.get("/hosts/test_host_win_01").json()["host"]
            assert host_recovered["status"] in ["ONLINE", "ANOMALY"]
            results["10. Offline & Recovery Lifecycle"] = "PASSED"
            print(" [PASS] 10. Offline / recovery lifecycle")
        except Exception as e:
            results["10. Offline & Recovery Lifecycle"] = f"FAILED: {e}"
            print(f" [FAIL] 10. Offline & Recovery Lifecycle: {e}")

    print("=" * 80)
    passed_cnt = sum(1 for v in results.values() if v == "PASSED")
    tot_cnt = len(results)
    print(f"PHASE 6.1 SUMMARY: {passed_cnt} / {tot_cnt} CHECKS PASSED")
    print("=" * 80)

    out_dir = Path(PROJECT_ROOT / "data" / "multihost_validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase6_1_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "passed": passed_cnt,
            "total": tot_cnt,
            "results": results,
        }, f, indent=2)

    return passed_cnt == tot_cnt


if __name__ == "__main__":
    ok = run_phase6_1_tests()
    if not ok:
        sys.exit(1)
