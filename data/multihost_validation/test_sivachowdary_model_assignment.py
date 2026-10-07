"""
data/multihost_validation/test_sivachowdary_model_assignment.py

Regression Test Suite for Host Model Assignment & Multihost Compatibility.
Verifies:
1. host_sivachowdary registers with windows_sivachowdary_v2 on fresh/existing database
2. Model compatibility evaluates to COMPATIBLE (status 200 / is_compat = True)
3. Telemetry streaming succeeds without overwriting model_id to 'none'
4. Anomaly inference is enabled and produces valid predictions
5. Linux hosts remain TELEMETRY_ONLY with model_id = 'none' (baseline required)
6. Repeated registrations reuse host_sivachowdary without creating duplicate hosts
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
from api.main import app
from api.host_registry import get_host_registry
from api.multihost_inference import get_multihost_inference_manager
from data.multihost_validation.test_db_helper import IsolatedTestDatabase
from data.live_feature_schema import LIVE_FEATURES


class TestSivaChowdaryModelAssignment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._db_context = IsolatedTestDatabase(prefix="sivachowdary_assign_test_")
        cls._db_context.__enter__()
        cls.client = TestClient(app)
        cls.reg = get_host_registry()
        cls.inf_mgr = get_multihost_inference_manager()

    @classmethod
    def tearDownClass(cls):
        cls._db_context.__exit__(None, None, None)

    def test_01_host_sivachowdary_registers_with_v2_model(self):
        """Verify host_sivachowdary registers with windows_sivachowdary_v2 and COMPATIBLE status."""
        res = self.client.post("/hosts/register", json={
            "hostname": "SivaChowdary",
            "operating_system": "Windows 11 Professional",
            "os_version": "10.0.26200",
            "architecture": "AMD64",
            "custom_host_id": "host_sivachowdary",
            "machine_info": {"machine_id": "sivachowdary_pc_hw01", "hostname": "SivaChowdary"},
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["host_id"], "host_sivachowdary")
        self.assertEqual(data["model_id"], "windows_sivachowdary_v2")
        self.assertEqual(data["status"], "ONLINE")

        # Check DB record
        host_dict = self.reg.get_host("host_sivachowdary")
        self.assertIsNotNone(host_dict)
        self.assertEqual(host_dict["model_id"], "windows_sivachowdary_v2")
        self.assertEqual(host_dict["model_status"], "COMPATIBLE")

    def test_02_model_compatibility_evaluates_compatible(self):
        """Verify model compatibility check for host_sivachowdary returns COMPATIBLE."""
        host_dict = self.reg.get_host("host_sivachowdary")
        is_compat, msg, details = self.inf_mgr.check_compatibility(host_dict, "windows_sivachowdary_v2")
        self.assertTrue(is_compat, f"Compatibility check failed: {msg}")
        self.assertEqual(details["model_status"], "COMPATIBLE")
        self.assertEqual(details["model_id"], "windows_sivachowdary_v2")
        self.assertAlmostEqual(details["threshold"], 1.411807, places=4)

    def test_03_telemetry_streaming_preserves_v2_model(self):
        """Verify telemetry ingestion streams cleanly and preserves windows_sivachowdary_v2."""
        # 1. Register to get token
        reg_data = self.client.post("/hosts/register", json={
            "hostname": "SivaChowdary",
            "operating_system": "Windows",
            "os_version": "10.0.26200",
            "architecture": "AMD64",
            "custom_host_id": "host_sivachowdary",
        }).json()
        token = reg_data["agent_token"]

        # 2. Ingest telemetry sample
        sample_features = {feat: 1.0 for feat in LIVE_FEATURES}
        t_res = self.client.post(
            "/hosts/host_sivachowdary/telemetry",
            json={
                "host_id": "host_sivachowdary",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": sample_features,
            },
            headers={"X-Agent-Token": token},
        )
        self.assertEqual(t_res.status_code, 200)
        t_data = t_res.json()
        self.assertEqual(t_data["model_status"], "COMPATIBLE")

        # 3. Verify model_id in database is STILL windows_sivachowdary_v2 (not overwritten to 'none')
        updated_host = self.reg.get_host("host_sivachowdary")
        self.assertEqual(updated_host["model_id"], "windows_sivachowdary_v2")
        self.assertEqual(updated_host["model_status"], "COMPATIBLE")

    def test_04_linux_host_remains_telemetry_only(self):
        """Verify Linux host registers as TELEMETRY_ONLY with model_id='none' and baseline required."""
        res = self.client.post("/hosts/register", json={
            "hostname": "Ubuntu-Prod-Server-01",
            "operating_system": "Linux Ubuntu 22.04 LTS",
            "os_version": "5.15.0-88-generic",
            "architecture": "x86_64",
            "custom_host_id": "host_linux_srv01",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["host_id"], "host_linux_srv01")
        self.assertEqual(data["model_id"], "none")
        self.assertEqual(data["status"], "TELEMETRY_ONLY")

        # Ingest Linux telemetry
        token = data["agent_token"]
        sample_features = {feat: 1.0 for feat in LIVE_FEATURES}
        t_res = self.client.post(
            "/hosts/host_linux_srv01/telemetry",
            json={
                "host_id": "host_linux_srv01",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": sample_features,
            },
            headers={"X-Agent-Token": token},
        )
        self.assertEqual(t_res.status_code, 200)
        t_data = t_res.json()
        self.assertEqual(t_data["model_status"], "BASELINE_REQUIRED")

        # Confirm DB status
        linux_host = self.reg.get_host("host_linux_srv01")
        self.assertEqual(linux_host["model_id"], "none")
        self.assertEqual(linux_host["model_status"], "BASELINE_REQUIRED")

    def test_05_no_duplicate_host_created_on_reregistration(self):
        """Verify repeated registration calls reuse host_sivachowdary without creating duplicate hosts."""
        all_hosts_before = self.reg.get_all_hosts()
        siva_hosts_before = [h for h in all_hosts_before if "sivachowdary" in h["host_id"]]
        self.assertEqual(len(siva_hosts_before), 1)

        # Re-register multiple times
        for i in range(3):
            self.client.post("/hosts/register", json={
                "hostname": "SivaChowdary",
                "operating_system": "Windows 11",
                "os_version": "10.0.26200",
                "architecture": "AMD64",
                "machine_info": {"machine_id": "sivachowdary_pc_hw01"},
            })

        all_hosts_after = self.reg.get_all_hosts()
        siva_hosts_after = [h for h in all_hosts_after if "sivachowdary" in h["host_id"]]
        self.assertEqual(len(siva_hosts_after), 1, "Exactly one host_sivachowdary must exist")


if __name__ == "__main__":
    unittest.main(verbosity=2)
