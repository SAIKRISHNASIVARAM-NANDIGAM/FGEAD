"""
data/multihost_validation/test_db_helper.py

Context manager for fully isolated, zero-leak test databases.
Ensures that tests NEVER write to data/fgead_multihost.db.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from api.host_registry import reset_host_registry, DEFAULT_DB_PATH


class IsolatedTestDatabase:
    """
    Context manager that creates a temporary isolated SQLite database for a test execution.
    Automatically redirects FGEAD_DB_PATH and resets the global HostRegistry singleton.
    Cleans up all temporary database files upon exit.
    """

    def __init__(self, prefix: str = "fgead_test_"):
        self.prefix = prefix
        self.temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self.test_db_path: Optional[Path] = None
        self.orig_env: Optional[str] = None

    def __enter__(self) -> Path:
        self.temp_dir = tempfile.TemporaryDirectory(prefix=self.prefix)
        self.test_db_path = Path(self.temp_dir.name) / "test_multihost.db"
        self.orig_env = os.environ.get("FGEAD_DB_PATH")
        os.environ["FGEAD_DB_PATH"] = str(self.test_db_path)
        reset_host_registry(db_path=self.test_db_path)
        return self.test_db_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.orig_env is not None:
            os.environ["FGEAD_DB_PATH"] = self.orig_env
        else:
            os.environ.pop("FGEAD_DB_PATH", None)
        reset_host_registry()
        if self.temp_dir is not None:
            try:
                self.temp_dir.cleanup()
            except Exception:
                pass
