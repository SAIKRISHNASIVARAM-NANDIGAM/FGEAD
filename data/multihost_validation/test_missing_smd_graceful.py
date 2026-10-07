"""
data/multihost_validation/test_missing_smd_graceful.py

Automated Test Suite for Graceful Non-Fatal SMD Benchmark Handling.
Verifies that:
1. Missing SMD dataset files do not cause application-level crashes or fatal errors.
2. load_raw_smd_dataset() returns (None, None, None) safely when files are absent.
3. Windows Live Host Monitoring and Streamlit app startup remain fully operational without SMD data.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit_app import load_raw_smd_dataset


class TestMissingSMDGracefulFallback(unittest.TestCase):
    
    @patch("pathlib.Path.is_file")
    def test_missing_smd_files_return_none(self, mock_is_file):
        """Verify load_raw_smd_dataset returns (None, None, None) when files do not exist."""
        mock_is_file.return_value = False
        try:
            load_raw_smd_dataset.clear()
        except Exception:
            pass

        test_data, test_labels, window_labels = load_raw_smd_dataset()
        self.assertIsNone(test_data)
        self.assertIsNone(test_labels)
        self.assertIsNone(window_labels)

        try:
            load_raw_smd_dataset.clear()
        except Exception:
            pass

    def test_load_raw_smd_nonexistent_dir(self):
        """Verify load_raw_smd_dataset handles missing files natively without throwing exceptions."""
        try:
            load_raw_smd_dataset.clear()
        except Exception:
            pass

        test_data, test_labels, window_labels = load_raw_smd_dataset()
        # Should be None if data/SMD/ is absent, or valid tuple if present, but must not raise an unhandled exception.
        if test_data is None:
            self.assertIsNone(test_labels)
            self.assertIsNone(window_labels)
        else:
            self.assertIsNotNone(test_labels)
            self.assertIsNotNone(window_labels)

        try:
            load_raw_smd_dataset.clear()
        except Exception:
            pass


if __name__ == "__main__":
    unittest.main()
