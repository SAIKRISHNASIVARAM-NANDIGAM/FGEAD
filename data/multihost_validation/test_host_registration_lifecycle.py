"""
data/multihost_validation/test_host_registration_lifecycle.py

Unit and Integration Test Suite for Robust Multi-Host Registration & Identity Lifecycle.
Validates:
A. First registration -> creates host
B. Same machine registration again -> reuses same host_id
C. Same machine after restart (e.g. fresh token / new agent instance) -> reuses same host_id
D. Repeated auto-registration -> no duplicates in database
E. Two different machines -> two different host_ids
F. Linux and Windows machines with same hostname -> remain separate
G. Reconnect after network/timeout -> reuses same host_id
H. Concurrent/repeated registration -> no duplicates, idempotent
"""

import os
import sys
import time
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
from api.main import app
from api.host_registry import get_host_registry
from data.multihost_validation.test_db_helper import IsolatedTestDatabase


class TestHostRegistrationLifecycle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._db_context = IsolatedTestDatabase(prefix="lifecycle_test_")
        cls._db_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._db_context.__exit__(None, None, None)

    def setUp(self):
        self.client = TestClient(app)
        self.reg = get_host_registry()

    def test_A_first_registration_creates_host(self):
        """A. First registration -> creates host with valid host_id and token."""
        res = self.client.post("/hosts/register", json={
            "hostname": "Test-Node-Alpha",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "alpha_hw_01", "hostname": "Test-Node-Alpha"},
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["host_id"].startswith("host_"))
        self.assertTrue(data["agent_token"].startswith("fgead_"))
        self.assertEqual(data["hostname"], "Test-Node-Alpha")
        self.assertEqual(data["status"], "ONLINE")

    def test_B_same_machine_registration_reuses_host_id(self):
        """B. Same machine registration again -> reuses same host_id."""
        res1 = self.client.post("/hosts/register", json={
            "hostname": "Test-Node-Beta",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "beta_hw_02", "hostname": "Test-Node-Beta"},
        }).json()
        hid1 = res1["host_id"]

        # Immediate second registration
        res2 = self.client.post("/hosts/register", json={
            "hostname": "Test-Node-Beta",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "beta_hw_02", "hostname": "Test-Node-Beta"},
        }).json()
        hid2 = res2["host_id"]

        self.assertEqual(hid1, hid2, "Host ID must be reused for the same machine")

    def test_C_same_machine_after_restart_reuses_host_id(self):
        """C. Same machine after restart (e.g. without saved config) -> reuses same host_id."""
        res1 = self.client.post("/hosts/register", json={
            "hostname": "Test-Node-Gamma",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "gamma_hw_03", "hostname": "Test-Node-Gamma"},
        }).json()
        hid1 = res1["host_id"]

        # Simulate agent restart where token is re-issued
        res2 = self.client.post("/hosts/register", json={
            "hostname": "Test-Node-Gamma",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "gamma_hw_03", "hostname": "Test-Node-Gamma"},
        }).json()
        hid2 = res2["host_id"]
        tok2 = res2["agent_token"]

        self.assertEqual(hid1, hid2)
        # Verify newly issued token authenticates successfully
        auth_ok, _, _ = self.reg.authenticate_agent(hid2, tok2)
        self.assertTrue(auth_ok)

    def test_D_repeated_auto_registration_no_duplicate(self):
        """D. Repeated auto-registration -> no duplicate hosts in database."""
        all_hosts_before = self.client.get("/hosts").json()["hosts"]
        count_before = len([h for h in all_hosts_before if h["hostname"].lower() == "test-node-delta"])

        for _ in range(5):
            self.client.post("/hosts/register", json={
                "hostname": "Test-Node-Delta",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
                "machine_info": {"machine_id": "delta_hw_04", "hostname": "Test-Node-Delta"},
            })

        all_hosts_after = self.client.get("/hosts").json()["hosts"]
        count_after = len([h for h in all_hosts_after if h["hostname"].lower() == "test-node-delta"])
        self.assertEqual(count_after, 1, "Exactly 1 record must exist despite 5 registrations")

    def test_E_two_different_machines_two_different_host_ids(self):
        """E. Two different machines -> two different host_ids."""
        res_a = self.client.post("/hosts/register", json={
            "hostname": "Machine-Alpha-Server",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "hw_mach_a"},
        }).json()
        res_b = self.client.post("/hosts/register", json={
            "hostname": "Machine-Beta-Server",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
            "machine_info": {"machine_id": "hw_mach_b"},
        }).json()

        self.assertNotEqual(res_a["host_id"], res_b["host_id"])

    def test_F_linux_and_windows_machines_remain_separate(self):
        """F. Linux and Windows machines with identical hostname -> remain separate."""
        res_w = self.client.post("/hosts/register", json={
            "hostname": "Dual-OS-Node",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
        }).json()
        res_l = self.client.post("/hosts/register", json={
            "hostname": "Dual-OS-Node",
            "operating_system": "Linux",
            "os_version": "6.5.0-generic",
            "architecture": "x86_64",
        }).json()

        self.assertNotEqual(res_w["host_id"], res_l["host_id"])
        self.assertEqual(res_w["status"], "ONLINE")
        self.assertEqual(res_l["status"], "TELEMETRY_ONLY")

    def test_G_reconnect_reuses_host_id(self):
        """G. Reconnect after offline period -> reuses same host_id and restores ONLINE status."""
        res1 = self.client.post("/hosts/register", json={
            "hostname": "Reconnect-Node-01",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
        }).json()
        hid = res1["host_id"]

        # Simulate host becoming OFFLINE
        self.reg.set_host_status(hid, "OFFLINE")
        host_status = self.reg.get_host(hid)["status"]
        self.assertEqual(host_status, "OFFLINE")

        # Reconnect registration
        res2 = self.client.post("/hosts/register", json={
            "hostname": "Reconnect-Node-01",
            "operating_system": "Windows",
            "os_version": "10.0.26100",
            "architecture": "AMD64",
        }).json()

        self.assertEqual(res2["host_id"], hid)
        self.assertEqual(res2["status"], "ONLINE")

    def test_H_concurrent_registration_no_duplicate(self):
        """H. Concurrent repeated registrations -> no duplicates created, idempotent."""
        def register_call(i):
            return self.client.post("/hosts/register", json={
                "hostname": "Concurrent-Node-01",
                "operating_system": "Windows",
                "os_version": "10.0.26100",
                "architecture": "AMD64",
                "machine_info": {"machine_id": "conc_hw_99"},
            }).json()

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(register_call, range(10)))

        host_ids = set(r["host_id"] for r in results)
        self.assertEqual(len(host_ids), 1, "All concurrent registrations must resolve to the identical host_id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
