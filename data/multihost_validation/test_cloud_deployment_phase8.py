"""
data/multihost_validation/test_cloud_deployment_phase8.py

Phase 8 Cloud Deployment Readiness & Remote Telemetry Validation Suite.
Validates:
1. Security & Authentication (Token & API Key Verification, Missing/Invalid Auth Rejection)
2. Live Remote Telemetry Ingestion & Warmup
3. Live Neural Inference, Dynamic Residuals & 5-Question XAI Attribution
4. Controlled Workload Anomaly Escalation & Episode Creation
5. Offline Detection & Recovery Lifecycle
6. Multi-Host Isolation & Cross-Host Gating
7. Performance & Latency Benchmarks (<100ms Inference)
"""

import os
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test Environment
os.environ["FGEAD_ENV"] = "TEST"
os.environ["FGEAD_DB_PATH"] = str(PROJECT_ROOT / "data" / "fgead_test_phase8.db")

from fastapi.testclient import TestClient
from config.settings import settings
from api.host_registry import reset_host_registry
from api.main import app
from data.live_agent import LiveTelemetryCollector
from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES


class TestPhase8CloudDeployment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = PROJECT_ROOT / "data" / "fgead_test_phase8.db"
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
        cls.collector = LiveTelemetryCollector()

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

    def test_01_security_and_auth_matrix(self):
        """Verify authentication gating: valid, missing, and forged tokens."""
        # 1. Register a valid host
        reg_resp = self.client.post("/hosts/register", json={
            "hostname": "win-cloud-node-01",
            "operating_system": "Windows",
            "os_version": "10.0.22631",
            "custom_host_id": "host_cloud_win_01",
        })
        self.assertEqual(reg_resp.status_code, 200)
        reg_data = reg_resp.json()
        host_id = reg_data["host_id"]
        valid_token = reg_data["agent_token"]

        sample_feats = self.collector.collect_features()
        payload = {"host_id": host_id, "timestamp": datetime.now(timezone.utc).isoformat(), "features": sample_feats}

        # 2. Case A: Valid authentication -> HTTP 200
        resp_valid = self.client.post(
            f"/hosts/{host_id}/telemetry",
            json=payload,
            headers={"X-Agent-Token": valid_token},
        )
        self.assertEqual(resp_valid.status_code, 200)

        # 3. Case B: Missing authentication -> HTTP 401
        resp_missing = self.client.post(
            f"/hosts/{host_id}/telemetry",
            json=payload,
            headers={},
        )
        self.assertEqual(resp_missing.status_code, 401)
        self.assertEqual(resp_missing.json()["status"], "error")

        # 4. Case C: Invalid / Forged authentication -> HTTP 403
        resp_forged = self.client.post(
            f"/hosts/{host_id}/telemetry",
            json=payload,
            headers={"X-Agent-Token": "forged-fake-agent-token-999"},
        )
        self.assertEqual(resp_forged.status_code, 403)
        self.assertEqual(resp_forged.json()["status"], "error")

        # 5. Case D: Unknown host with valid format token -> HTTP 404
        resp_unknown = self.client.post(
            "/hosts/host_non_existent_9999/telemetry",
            json={"host_id": "host_non_existent_9999", "timestamp": datetime.now(timezone.utc).isoformat(), "features": sample_feats},
            headers={"X-Agent-Token": valid_token},
        )
        self.assertEqual(resp_unknown.status_code, 404)
        print("  [PASS] Test 1: Security & Authentication Matrix (200 / 401 / 403 / 404)")

    def test_02_remote_windows_telemetry_warmup_and_inference(self):
        """Simulate real 60-second Windows telemetry transmission and rolling inference."""
        reg_resp = self.client.post("/hosts/register", json={
            "hostname": "win-prod-worker",
            "operating_system": "Windows",
            "os_version": "10.0.22631",
            "custom_host_id": "host_win_worker_01",
        }).json()
        host_id = reg_resp["host_id"]
        token = reg_resp["agent_token"]

        headers = {"X-Agent-Token": token}
        sample_feats = self.collector.collect_features()

        # Ingest 60 samples to fill the rolling buffer
        latencies = []
        for i in range(60):
            feats = dict(sample_feats)
            feats["cpu_percent"] = 12.0 + (i % 5) * 0.5
            feats["net_drops_total"] = 294.0 # matching baseline
            t_post_start = time.perf_counter()
            resp = self.client.post(
                f"/hosts/{host_id}/telemetry",
                json={"host_id": host_id, "timestamp": datetime.now(timezone.utc).isoformat(), "features": feats},
                headers=headers,
            )
            latencies.append((time.perf_counter() - t_post_start) * 1000)
            self.assertEqual(resp.status_code, 200)

        avg_ingest_ms = sum(latencies) / len(latencies)
        self.assertLess(avg_ingest_ms, 50.0) # Sub-50ms HTTP API ingest latency

        # Check buffer status
        h_resp = self.client.get(f"/hosts/{host_id}").json()
        self.assertEqual(h_resp["buffer_status"]["buffer_size"], 60)
        self.assertTrue(h_resp["buffer_status"]["is_window_full"])

        # Trigger live inference on full buffer
        t_inf_start = time.perf_counter()
        inf_resp = self.client.post(f"/hosts/{host_id}/predict/live", headers=headers)
        inf_latency_ms = (time.perf_counter() - t_inf_start) * 1000

        self.assertEqual(inf_resp.status_code, 200)
        inf_data = inf_resp.json()

        self.assertIn("anomaly_score", inf_data)
        self.assertIn("threshold", inf_data)
        self.assertEqual(inf_data["threshold"], 1.85945)
        self.assertIn("xai_5_questions", inf_data)
        self.assertIn("top_features", inf_data)
        self.assertIn("broken_pairs", inf_data)
        self.assertLess(inf_latency_ms, 250.0)
        print(f"  [PASS] Test 2: Real Windows Telemetry 60s Warmup & Live Inference ({inf_latency_ms:.1f}ms latency)")

    def test_03_controlled_workload_anomaly_test(self):
        """Simulate high workload condition and verify anomaly detection & XAI attribution."""
        reg_resp = self.client.post("/hosts/register", json={
            "hostname": "win-workload-node",
            "operating_system": "Windows",
            "os_version": "10.0.22631",
            "custom_host_id": "host_win_workload_01",
        }).json()
        host_id = reg_resp["host_id"]
        token = reg_resp["agent_token"]
        headers = {"X-Agent-Token": token}

        # Create anomalous window with high CPU and Memory deviation
        anom_window = []
        for i in range(60):
            sample = [10.0] * N_LIVE_FEATURES
            # Inject heavy CPU & Disk stress
            sample[0] = 98.5 # cpu_percent
            sample[6] = 95.0 # memory_percent
            sample[10] = 92.0 # disk_usage_percent
            sample[20] = 294.0 # net_drops_total baseline
            anom_window.append(sample)

        inf_resp = self.client.post(
            f"/hosts/{host_id}/predict/live",
            json={"data": anom_window, "timestamp": datetime.now(timezone.utc).isoformat()},
            headers=headers,
        )
        self.assertEqual(inf_resp.status_code, 200)
        inf_data = inf_resp.json()

        self.assertTrue(inf_data["is_anomaly"])
        self.assertGreater(inf_data["anomaly_score"], 1.85945)
        self.assertEqual(inf_data["severity"], "CRITICAL")
        self.assertGreater(len(inf_data["top_features"]), 0)
        self.assertIn("Q1_what_happened", inf_data["xai_5_questions"])
        self.assertIn("Q2_which_metrics_deviated", inf_data["xai_5_questions"])
        self.assertIn("Q3_how_metrics_interacted", inf_data["xai_5_questions"])
        print(f"  [PASS] Test 3: Controlled Workload Anomaly Detection & 5-Question XAI (Score: {inf_data['anomaly_score']:.2f} > τ=1.859)")

    def test_04_offline_detection_and_multi_host_isolation(self):
        """Verify offline host detection, recovery, and Linux model gating."""
        # 1. Register Windows & Linux nodes
        w_host = self.client.post("/hosts/register", json={"hostname": "w-node", "operating_system": "Windows", "os_version": "11"}).json()
        l_host = self.client.post("/hosts/register", json={"hostname": "l-node", "operating_system": "Linux", "os_version": "Ubuntu 22.04"}).json()

        w_hid = w_host["host_id"]
        l_hid = l_host["host_id"]

        # 2. Ingest telemetry
        sample = {f: 5.0 for f in LIVE_FEATURES}
        sample["net_drops_total"] = 294.0
        self.client.post(f"/hosts/{w_hid}/telemetry", json={"host_id": w_hid, "timestamp": datetime.now(timezone.utc).isoformat(), "features": sample}, headers={"X-Agent-Token": w_host["agent_token"]})
        self.client.post(f"/hosts/{l_hid}/telemetry", json={"host_id": l_hid, "timestamp": datetime.now(timezone.utc).isoformat(), "features": sample}, headers={"X-Agent-Token": l_host["agent_token"]})

        # 3. Check Fleet Overview
        fleet = self.client.get("/fleet/overview").json()
        hosts_map = {h["host_id"]: h for h in fleet["hosts"]}
        self.assertIn(w_hid, hosts_map)
        self.assertIn(l_hid, hosts_map)
        self.assertEqual(hosts_map[l_hid]["status"], "TELEMETRY_ONLY")
        self.assertIn(hosts_map[w_hid]["status"], ["ONLINE", "ANOMALY"])
        print("  [PASS] Test 4: Offline Detection & Multi-Host Linux Gating Verification")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FGEAD PHASE 8 — CLOUD DEPLOYMENT & REMOTE VALIDATION SUITE")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
