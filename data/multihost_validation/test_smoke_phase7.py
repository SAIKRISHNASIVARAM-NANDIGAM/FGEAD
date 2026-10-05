"""
data/multihost_validation/test_smoke_phase7.py

Phase 7 Automated End-to-End Smoke Test Suite.
Validates:
1. Environment Configuration (DEVELOPMENT, TEST, PRODUCTION)
2. Structured Logging & Sanitization
3. Health Probes (/health, /health/live, /health/ready)
4. SMD Benchmark Mode APIs
5. Multi-Host Registration & Authentication
6. Windows Live Inference vs Linux Model Gating
7. Error Handling & JSON Responses
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ["FGEAD_ENV"] = "TEST"
os.environ["FGEAD_DB_PATH"] = str(PROJECT_ROOT / "data" / "fgead_test_phase7.db")

from fastapi.testclient import TestClient
from config.settings import Settings, settings
from core.logger import setup_logger, log_api_request
from api.main import app
from data.live_feature_schema import LIVE_FEATURES


from api.host_registry import reset_host_registry


class TestPhase7ProductionSmokeSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = PROJECT_ROOT / "data" / "fgead_test_phase7.db"
        os.environ["FGEAD_ENV"] = "TEST"
        os.environ["FGEAD_DB_PATH"] = str(cls.test_db)
        if cls.test_db.exists():
            try:
                cls.test_db.unlink()
            except Exception:
                pass
        reset_host_registry(db_path=cls.test_db)
        cls._client_context = TestClient(app)
        cls.client = cls._client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        try:
            cls._client_context.__exit__(None, None, None)
        except Exception:
            pass
        reset_host_registry()
        if hasattr(cls, "test_db") and cls.test_db.exists():
            try:
                cls.test_db.unlink()
            except Exception:
                pass

    def test_01_configuration_modes(self):
        """Verify settings load correctly and test mode properties."""
        test_settings = Settings()
        self.assertEqual(test_settings.ENVIRONMENT, "TEST")
        self.assertTrue(test_settings.is_test)
        self.assertFalse(test_settings.is_production)
        self.assertIsNotNone(test_settings.SMD_CHECKPOINT_PATH)
        self.assertIsNotNone(test_settings.LIVE_CHECKPOINT_PATH)
        print("  [PASS] Test 1: Centralized Configuration System")

    def test_02_structured_logger(self):
        """Verify structured logger initialization and formatting."""
        logger = setup_logger("test.fgead")
        self.assertIsNotNone(logger)
        # Verify log_api_request doesn't throw
        log_api_request(
            endpoint="/test/endpoint",
            method="GET",
            status_code=200,
            latency_ms=1.45,
            host_id="host_test_01",
        )
        print("  [PASS] Test 2: Structured Logger & Request Tracking")

    def test_03_root_and_health_probes(self):
        """Verify /, /health, /health/live, and /health/ready endpoints."""
        # Root endpoint
        resp_root = self.client.get("/")
        self.assertEqual(resp_root.status_code, 200)
        data_root = resp_root.json()
        self.assertIn("endpoints", data_root)
        self.assertIn("health_live", data_root["endpoints"])
        self.assertIn("health_ready", data_root["endpoints"])

        # Health endpoint
        resp_h = self.client.get("/health")
        self.assertEqual(resp_h.status_code, 200)
        data_h = resp_h.json()
        self.assertEqual(data_h["status"], "ok")
        self.assertTrue(data_h["model_loaded"])

        # Liveness probe
        resp_live = self.client.get("/health/live")
        self.assertEqual(resp_live.status_code, 200)
        data_live = resp_live.json()
        self.assertEqual(data_live["status"], "alive")

        # Readiness probe
        resp_ready = self.client.get("/health/ready")
        self.assertEqual(resp_ready.status_code, 200)
        data_ready = resp_ready.json()
        self.assertTrue(data_ready["ready"])
        self.assertEqual(data_ready["status"], "ready")
        self.assertTrue(data_ready["checks"]["database"])
        self.assertTrue(data_ready["checks"]["smd_model"])
        self.assertTrue(data_ready["checks"]["live_inference"])
        print("  [PASS] Test 3: Root and Health Probes (/health, /health/live, /health/ready)")

    def test_04_smd_benchmark_endpoints(self):
        """Verify SMD dataset and machine info metadata."""
        resp_mach = self.client.get("/machine")
        self.assertEqual(resp_mach.status_code, 200)
        data_mach = resp_mach.json()
        self.assertEqual(data_mach["features"], 38)
        self.assertEqual(len(data_mach["feature_names"]), 38)

        resp_ds = self.client.get("/dataset")
        self.assertEqual(resp_ds.status_code, 200)
        data_ds = resp_ds.json()
        self.assertGreater(data_ds["train_timesteps"], 0)
        self.assertGreater(data_ds["test_timesteps"], 0)
        print("  [PASS] Test 4: SMD Benchmark Mode API Endpoints")

    def test_05_host_registration_and_gating(self):
        """Verify Windows registration (COMPATIBLE) vs Linux registration (BASELINE_REQUIRED)."""
        # Register Windows host
        win_payload = {
            "hostname": "prod-win-01",
            "operating_system": "Windows",
            "os_version": "10.0.22631",
            "architecture": "x86_64",
            "agent_version": "1.0.0",
        }
        resp_win = self.client.post("/hosts/register", json=win_payload)
        self.assertEqual(resp_win.status_code, 200)
        data_win = resp_win.json()
        self.win_host_id = data_win["host_id"]
        self.win_token = data_win["agent_token"]
        self.assertEqual(data_win["model_id"], "windows_default")

        # Register Linux host
        lin_payload = {
            "hostname": "prod-linux-01",
            "operating_system": "Linux",
            "os_version": "6.5.0-generic",
            "architecture": "x86_64",
            "agent_version": "1.0.0",
        }
        resp_lin = self.client.post("/hosts/register", json=lin_payload)
        self.assertEqual(resp_lin.status_code, 200)
        data_lin = resp_lin.json()
        self.lin_host_id = data_lin["host_id"]
        self.lin_token = data_lin["agent_token"]
        self.assertEqual(data_lin["model_id"], "none")
        print("  [PASS] Test 5: Multi-Host Registration & Model Gating Assignment")

    def test_06_telemetry_and_inference_gating(self):
        """Verify telemetry ingestion and model gating for Windows vs Linux."""
        # 1. Register hosts
        win_resp = self.client.post("/hosts/register", json={
            "hostname": "srv-win", "operating_system": "Windows", "os_version": "11"
        }).json()
        lin_resp = self.client.post("/hosts/register", json={
            "hostname": "srv-lin", "operating_system": "Linux", "os_version": "Ubuntu 22.04"
        }).json()

        win_hid = win_resp["host_id"]
        win_tok = win_resp["agent_token"]
        lin_hid = lin_resp["host_id"]
        lin_tok = lin_resp["agent_token"]

        # 2. Ingest telemetry for both
        sample_feats = {f: 10.0 for f in LIVE_FEATURES}
        sample_feats["cpu_usage_pct"] = 15.5
        sample_feats["net_drops_total"] = 0.0

        headers_win = {"X-Agent-Token": win_tok}
        resp_w_tel = self.client.post(
            f"/hosts/{win_hid}/telemetry",
            json={"host_id": win_hid, "timestamp": "2026-10-04T20:00:00Z", "features": sample_feats},
            headers=headers_win,
        )
        self.assertEqual(resp_w_tel.status_code, 200)

        headers_lin = {"X-Agent-Token": lin_tok}
        resp_l_tel = self.client.post(
            f"/hosts/{lin_hid}/telemetry",
            json={"host_id": lin_hid, "timestamp": "2026-10-04T20:00:00Z", "features": sample_feats},
            headers=headers_lin,
        )
        self.assertEqual(resp_l_tel.status_code, 200)

        # 3. Attempt inference on Linux -> Should reject with 400 and clear message
        dummy_window = [[10.0] * 22 for _ in range(60)]
        resp_lin_inf = self.client.post(
            f"/hosts/{lin_hid}/predict/live",
            json={"data": dummy_window},
            headers=headers_lin,
        )
        self.assertEqual(resp_lin_inf.status_code, 400)
        self.assertIn("Anomaly inference is disabled", resp_lin_inf.json()["detail"])

        # 4. Attempt inference on Windows -> Should succeed (200)
        resp_win_inf = self.client.post(
            f"/hosts/{win_hid}/predict/live",
            json={"data": dummy_window},
            headers=headers_win,
        )
        self.assertEqual(resp_win_inf.status_code, 200)
        win_inf_data = resp_win_inf.json()
        self.assertIn("is_anomaly", win_inf_data)
        self.assertIn("anomaly_score", win_inf_data)
        self.assertIn("xai_5_questions", win_inf_data)
        self.assertIn("top_features", win_inf_data)
        self.assertIn("broken_pairs", win_inf_data)
        print("  [PASS] Test 6: Ingestion & Model Compatibility Gating Verification")

    def test_07_fleet_overview_and_error_handling(self):
        """Verify fleet overview aggregation and standardized error formats."""
        resp_fleet = self.client.get("/fleet/overview")
        self.assertEqual(resp_fleet.status_code, 200)
        fleet_data = resp_fleet.json()
        self.assertIn("total_hosts", fleet_data)
        self.assertIn("hosts", fleet_data)

        # Test 404 error format
        resp_404 = self.client.get("/hosts/non_existent_host_id_12345/analysis")
        self.assertEqual(resp_404.status_code, 404)
        err_json = resp_404.json()
        self.assertEqual(err_json["status"], "error")
        self.assertEqual(err_json["status_code"], 404)
        self.assertIn("detail", err_json)
        self.assertIn("timestamp", err_json)
        print("  [PASS] Test 7: Fleet Overview & Standardized Clean JSON Error Format")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FGEAD PHASE 7 — PRODUCTION SMOKE TEST SUITE")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
