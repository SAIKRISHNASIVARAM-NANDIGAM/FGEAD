"""
data/multihost_validation/test_operational_db_isolation_invariant.py

Safety Invariant Test Suite:
Validates that:
1. Operational database (data/fgead_multihost.db) contains strictly the 2 authorized hosts:
   - host_sivachowdary (Active physical Windows PC)
   - host_linux_srv01 (Reference Linux server)
2. Automated test suites execute in complete isolation without leaking records to the operational DB.
3. Deterministic host matching preserves the operational host count.
"""

import os
import sqlite3
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPERATIONAL_DB_PATH = PROJECT_ROOT / "data" / "fgead_multihost.db"
BACKUP_DB_PATH = PROJECT_ROOT / "data" / "fgead_multihost.db.backup_before_test_cleanup"


class TestOperationalDatabaseIsolationInvariant(unittest.TestCase):
    def test_01_backup_file_exists(self):
        """Verify safety backup file was created prior to synthetic test cleanup."""
        self.assertTrue(BACKUP_DB_PATH.exists(), f"Backup file not found at {BACKUP_DB_PATH}")
        self.assertGreater(BACKUP_DB_PATH.stat().st_size, 0)

    def test_02_operational_database_host_count_and_members(self):
        """Verify operational database contains exactly 2 legitimate hosts and no test fixture clutter."""
        self.assertTrue(OPERATIONAL_DB_PATH.exists(), f"Operational DB not found at {OPERATIONAL_DB_PATH}")
        conn = sqlite3.connect(OPERATIONAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT host_id, hostname, operating_system, is_enabled FROM hosts ORDER BY host_id ASC")
        rows = cursor.fetchall()
        conn.close()

        self.assertEqual(len(rows), 2, f"Expected exactly 2 hosts in operational DB, found {len(rows)}: {rows}")
        hosts_map = {r[0]: (r[1], r[2], r[3]) for r in rows}

        self.assertIn("host_sivachowdary", hosts_map)
        self.assertEqual(hosts_map["host_sivachowdary"][0], "SivaChowdary")
        self.assertEqual(hosts_map["host_sivachowdary"][1], "Windows")
        self.assertEqual(hosts_map["host_sivachowdary"][2], 1)

        self.assertIn("host_linux_srv01", hosts_map)
        self.assertEqual(hosts_map["host_linux_srv01"][0], "Ubuntu-Prod-Server-01")
        self.assertEqual(hosts_map["host_linux_srv01"][1], "Linux")
        self.assertEqual(hosts_map["host_linux_srv01"][2], 1)

    def test_03_no_synthetic_test_fixtures_in_operational_db(self):
        """Verify no test fixture host IDs exist in the operational database."""
        synthetic_prefixes = ("test_", "host_test_", "host_machine_", "host_dual_", "host_reconnect_", "host_concurrent_")
        conn = sqlite3.connect(OPERATIONAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT host_id FROM hosts")
        host_ids = [r[0] for r in cursor.fetchall()]
        conn.close()

        for hid in host_ids:
            for prefix in synthetic_prefixes:
                self.assertFalse(hid.startswith(prefix), f"Synthetic test host '{hid}' leaked into operational DB!")

    def test_04_alerts_table_cleanliness(self):
        """Verify alerts in operational DB only reference legitimate hosts."""
        conn = sqlite3.connect(OPERATIONAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT host_id FROM alerts")
        alert_hosts = [r[0] for r in cursor.fetchall()]
        conn.close()

        allowed_hosts = {"host_sivachowdary", "host_linux_srv01"}
        for hid in alert_hosts:
            self.assertIn(hid, allowed_hosts, f"Stale alert for '{hid}' in operational DB!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
