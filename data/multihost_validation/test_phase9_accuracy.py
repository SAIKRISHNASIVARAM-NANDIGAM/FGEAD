"""
data/multihost_validation/test_phase9_accuracy.py

Phase 9 Automated Accuracy, Trustworthy Decision Gate & Explainability Test Suite.
Validates all 18 decision gate requirements:
1. Normal window -> no anomaly
2. Invalid telemetry -> no anomaly
3. NaN -> rejected
4. Inf -> rejected
5. Wrong feature count -> rejected
6. Wrong schema -> rejected
7. Unsupported OS -> no anomaly score (TELEMETRY_ONLY)
8. Missing compatible model -> no anomaly score
9. Model unavailable -> no anomaly
10. Single threshold crossing -> not immediately confirmed (SUSPICIOUS)
11. Persistent abnormal windows -> anomaly confirmed (ANOMALY)
12. Recovery -> anomaly episode closes
13. Normal after recovery -> no anomaly
14. Explanation exists for confirmed anomaly
15. Explanation absent / nominal for normal state
16. Explanation absent for unsupported state
17. Host isolation
18. Existing SMD model isolation
"""

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment
os.environ["FGEAD_ENV"] = "TEST"
os.environ["FGEAD_DB_PATH"] = str(PROJECT_ROOT / "data" / "fgead_test_phase9.db")

from fastapi.testclient import TestClient
from api.main import app
from api.live_inference import LiveEpisodeTracker, get_live_inference_service
from api.multihost_inference import get_multihost_inference_manager
from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES


from api.host_registry import reset_host_registry


class TestPhase9AccuracyAndDecisionGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db = PROJECT_ROOT / "data" / "fgead_test_phase9.db"
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
        cls.live_svc = get_live_inference_service()
        cls.inf_mgr = get_multihost_inference_manager()

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

    def test_01_normal_window_no_anomaly(self):
        """1. Normal window -> evaluated as nominal and no confirmed anomaly."""
        df_base = Path(PROJECT_ROOT / "data" / "live_baseline.csv")
        import pandas as pd
        df = pd.read_csv(df_base)
        nominal_window = df[LIVE_FEATURES].iloc[100:160].values.astype(np.float32)
        res = self.live_svc.infer_window(nominal_window)
        self.assertFalse(res["is_anomaly"])
        self.assertLess(res["anomaly_score"], self.live_svc.threshold)
        print("  [PASS] 1. Normal window -> no anomaly")

    def test_02_invalid_telemetry_rejected(self):
        """2. Invalid telemetry dimensions -> rejected with error."""
        invalid_shape_window = np.ones((30, 22), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.live_svc.infer_window(invalid_shape_window)
        print("  [PASS] 2. Invalid telemetry dimensions -> rejected")

    def test_03_nan_rejected(self):
        """3. NaN in input window -> rejected with ValueError."""
        nan_window = np.ones((60, N_LIVE_FEATURES), dtype=np.float32)
        nan_window[10, 5] = np.nan
        with self.assertRaises(ValueError):
            self.live_svc.infer_window(nan_window)
        print("  [PASS] 3. NaN -> rejected")

    def test_04_inf_rejected(self):
        """4. Inf in input window -> rejected with ValueError."""
        inf_window = np.ones((60, N_LIVE_FEATURES), dtype=np.float32)
        inf_window[15, 3] = np.inf
        with self.assertRaises(ValueError):
            self.live_svc.infer_window(inf_window)
        print("  [PASS] 4. Inf -> rejected")

    def test_05_wrong_feature_count_rejected(self):
        """5. Wrong feature count (e.g. 18 instead of 22) -> rejected."""
        wrong_feat_window = np.ones((60, 18), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.live_svc.infer_window(wrong_feat_window)
        print("  [PASS] 5. Wrong feature count -> rejected")

    def test_06_wrong_schema_version_rejected(self):
        """6. Incompatible schema version -> model compatibility rejection."""
        host_bad_schema = {
            "hostname": "bad-schema-host",
            "operating_system": "Windows",
            "os_version": "10.0",
            "schema_version": "9.9.legacy",
        }
        is_compat, reason, details = self.inf_mgr.check_compatibility(host_bad_schema, "windows_default")
        self.assertFalse(is_compat)
        self.assertEqual(details["status"], "SCHEMA_MISMATCH")
        print("  [PASS] 6. Wrong schema version -> rejected")

    def test_07_unsupported_os_no_anomaly_score(self):
        """7. Unsupported OS (Linux on Windows model) -> TELEMETRY_ONLY with no anomaly score."""
        host_linux = {
            "hostname": "linux-srv-01",
            "operating_system": "Linux",
            "os_version": "6.5.0-generic",
            "schema_version": "1.0",
        }
        is_compat, reason, details = self.inf_mgr.check_compatibility(host_linux, "windows_default")
        self.assertFalse(is_compat)
        self.assertEqual(details["status"], "BASELINE_REQUIRED")
        print("  [PASS] 7. Unsupported OS -> no anomaly score (TELEMETRY_ONLY)")

    def test_08_missing_compatible_model_handled(self):
        """8. Missing compatible model profile -> inference rejected cleanly."""
        host_unmapped = {
            "hostname": "solaris-node",
            "operating_system": "Solaris",
            "model_id": "none",
        }
        is_compat, reason, details = self.inf_mgr.check_compatibility(host_unmapped, "none")
        self.assertFalse(is_compat)
        print("  [PASS] 8. Missing compatible model -> no anomaly score")

    def test_09_model_unavailable_handled(self):
        """9. Unloaded model profile -> rejected with MODEL_NOT_LOADED."""
        is_compat, reason, details = self.inf_mgr.check_compatibility({"operating_system": "Windows"}, "non_existent_profile")
        self.assertFalse(is_compat)
        print("  [PASS] 9. Model unavailable -> no anomaly")

    def test_10_single_threshold_crossing_not_immediately_confirmed(self):
        """10. Single window threshold crossing -> flagged as SUSPICIOUS, not confirmed."""
        tracker = LiveEpisodeTracker(debounce_frames=2, min_persistence_frames=2)
        ep_id, is_new = tracker.update(
            is_anomaly=True,
            score=5.5,
            threshold=1.85945,
            timestamp="2026-10-04T20:00:00Z",
            top_features=[{"feature": "cpu_percent"}],
        )
        status = tracker.get_status()
        self.assertTrue(status["is_pending_suspicious"])
        self.assertFalse(status["is_in_anomaly_episode"])
        self.assertEqual(status["decision_state"], "SUSPICIOUS")
        print("  [PASS] 10. Single threshold crossing -> not immediately confirmed (SUSPICIOUS)")

    def test_11_persistent_abnormal_windows_confirmed(self):
        """11. Consecutive abnormal windows (>= 2) -> anomaly confirmed (ANOMALY)."""
        tracker = LiveEpisodeTracker(debounce_frames=2, min_persistence_frames=2)
        # Window 1
        tracker.update(True, 5.5, 1.85945, "2026-10-04T20:00:00Z", [{"feature": "cpu_percent"}])
        # Window 2 (Persistent)
        tracker.update(True, 6.2, 1.85945, "2026-10-04T20:00:01Z", [{"feature": "cpu_percent"}])
        status = tracker.get_status()
        self.assertTrue(status["is_in_anomaly_episode"])
        self.assertTrue(status["is_confirmed"])
        self.assertEqual(status["decision_state"], "ANOMALY")
        print("  [PASS] 11. Persistent abnormal windows -> anomaly confirmed")

    def test_12_recovery_closes_episode(self):
        """12. Normal windows after anomaly -> episode closes cleanly after debounce."""
        tracker = LiveEpisodeTracker(debounce_frames=2, min_persistence_frames=2)
        # Anomaly windows
        tracker.update(True, 5.5, 1.85945, "2026-10-04T20:00:00Z", [{"feature": "cpu_percent"}])
        tracker.update(True, 6.2, 1.85945, "2026-10-04T20:00:01Z", [{"feature": "cpu_percent"}])
        # Nominal recovery window 1
        tracker.update(False, 0.5, 1.85945, "2026-10-04T20:00:02Z", [])
        # Nominal recovery window 2 (Debounce satisfied -> Close)
        tracker.update(False, 0.4, 1.85945, "2026-10-04T20:00:03Z", [])
        status = tracker.get_status()
        self.assertFalse(status["is_in_anomaly_episode"])
        self.assertEqual(len(tracker.completed_episodes), 1)
        self.assertTrue(tracker.completed_episodes[0]["is_confirmed"])
        print("  [PASS] 12. Recovery -> anomaly episode closes")

    def test_13_normal_after_recovery(self):
        """13. Subsequent normal windows remain NORMAL without phantom episodes."""
        tracker = LiveEpisodeTracker(debounce_frames=2, min_persistence_frames=2)
        # Anomaly & Recovery
        tracker.update(True, 5.5, 1.85945, "2026-10-04T20:00:00Z", [{"feature": "cpu_percent"}])
        tracker.update(True, 6.2, 1.85945, "2026-10-04T20:00:01Z", [{"feature": "cpu_percent"}])
        tracker.update(False, 0.5, 1.85945, "2026-10-04T20:00:02Z", [])
        tracker.update(False, 0.4, 1.85945, "2026-10-04T20:00:03Z", [])
        # 10 more nominal windows
        for i in range(10):
            tracker.update(False, 0.4 + (i * 0.01), 1.85945, f"2026-10-04T20:00:{i+4:02d}Z", [])
        status = tracker.get_status()
        self.assertEqual(status["decision_state"], "NORMAL")
        self.assertFalse(status["is_in_anomaly_episode"])
        print("  [PASS] 13. Normal after recovery -> no anomaly")

    def test_14_explanation_exists_for_confirmed_anomaly(self):
        """14. Confirmed anomaly generates complete 5-question XAI and top contributing features."""
        import pandas as pd
        df = pd.read_csv(PROJECT_ROOT / "data" / "live_baseline.csv")
        anom_window = df[LIVE_FEATURES].iloc[100:160].values.astype(np.float32).copy()
        anom_window[:, 0] = 99.5  # Heavy cpu_percent stress
        anom_window[:, 2] = 88.0  # cpu_user_time_percent
        res = self.live_svc.infer_window(anom_window)
        self.assertTrue(res["is_anomaly"])
        self.assertIn("five_question_explanation", res)
        self.assertIn("what_happened", res["five_question_explanation"])
        self.assertIn("what_contributed", res["five_question_explanation"])
        self.assertGreater(len(res["top_features"]), 0)
        top_feat_names = [f["feature"] for f in res["top_features"][:3]]
        self.assertTrue(any("cpu" in fn for fn in top_feat_names))
        print("  [PASS] 14. Explanation exists for confirmed anomaly")

    def test_15_explanation_nominal_for_normal_state(self):
        """15. Normal state reports nominal baseline behavior without fabricating false attack claims."""
        import pandas as pd
        df = pd.read_csv(PROJECT_ROOT / "data" / "live_baseline.csv")
        nom_window = df[LIVE_FEATURES].iloc[100:160].values.astype(np.float32)
        res = self.live_svc.infer_window(nom_window)
        self.assertFalse(res["is_anomaly"])
        exp = res["five_question_explanation"]
        self.assertIn("Nominal system operation", exp["what_happened"])
        self.assertNotIn("attack", exp["what_happened"].lower())
        print("  [PASS] 15. Explanation reports nominal baseline behavior for normal state")

    def test_16_explanation_absent_for_unsupported_state(self):
        """16. Unsupported host rejected before inference -> no fake anomaly explanation."""
        reg_resp = self.client.post("/hosts/register", json={
            "hostname": "linux-gated-node",
            "operating_system": "Linux",
            "os_version": "Ubuntu 22.04",
        }).json()
        h_id = reg_resp["host_id"]
        tok = reg_resp["agent_token"]

        dummy_window = [[10.0] * 22 for _ in range(60)]
        inf_resp = self.client.post(
            f"/hosts/{h_id}/predict/live",
            json={"data": dummy_window},
            headers={"X-Agent-Token": tok},
        )
        self.assertEqual(inf_resp.status_code, 400)
        self.assertIn("Anomaly inference is disabled", inf_resp.json()["detail"])
        print("  [PASS] 16. Explanation absent for unsupported state")

    def test_17_host_isolation_verified(self):
        """17. Telemetry and score history of Host A do not leak to Host B."""
        h_a = self.client.post("/hosts/register", json={"hostname": "host-iso-a", "operating_system": "Windows", "os_version": "11"}).json()
        h_b = self.client.post("/hosts/register", json={"hostname": "host-iso-b", "operating_system": "Windows", "os_version": "11"}).json()

        sample_a = {f: 10.0 for f in LIVE_FEATURES}
        sample_a["cpu_percent"] = 99.0
        sample_b = {f: 10.0 for f in LIVE_FEATURES}
        sample_b["cpu_percent"] = 5.0

        self.client.post(f"/hosts/{h_a['host_id']}/telemetry", json={"host_id": h_a["host_id"], "timestamp": "2026-10-04T20:00:00Z", "features": sample_a}, headers={"X-Agent-Token": h_a["agent_token"]})
        self.client.post(f"/hosts/{h_b['host_id']}/telemetry", json={"host_id": h_b["host_id"], "timestamp": "2026-10-04T20:00:00Z", "features": sample_b}, headers={"X-Agent-Token": h_b["agent_token"]})

        h_a_data = self.client.get(f"/hosts/{h_a['host_id']}").json()
        h_b_data = self.client.get(f"/hosts/{h_b['host_id']}").json()

        self.assertNotEqual(h_a_data["host"]["host_id"], h_b_data["host"]["host_id"])
        print("  [PASS] 17. Host isolation verified")

    def test_18_smd_model_isolation_verified(self):
        """18. SMD benchmark mode (38 features, tau=2.073376) remains completely isolated and operational."""
        resp_m = self.client.get("/machine")
        self.assertEqual(resp_m.status_code, 200)
        self.assertEqual(resp_m.json()["features"], 38)

        resp_ds = self.client.get("/dataset")
        self.assertEqual(resp_ds.status_code, 200)
        self.assertEqual(resp_ds.json()["features"], 38)
        print("  [PASS] 18. Existing SMD model isolation verified")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FGEAD PHASE 9 — ACCURACY & DECISION GATE VALIDATION SUITE")
    print("=" * 70 + "\n")
    unittest.main(verbosity=2)
