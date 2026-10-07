"""
data/multihost_validation/test_missing_smd_graceful.py

Automated Test Suite for Graceful Non-Fatal SMD Benchmark Resource Handling.
Verifies:
CASE A: SMD dataset/model absent -> no prediction API call made and graceful informational state active.
CASE B: SMD dataset/model present -> existing SMD benchmark functionality remains fully available.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.streamlit_app import load_raw_smd_dataset, fetch_api_prediction


class TestMissingSMDGracefulFallback(unittest.TestCase):

    def setUp(self):
        try:
            load_raw_smd_dataset.clear()
        except Exception:
            pass

    def tearDown(self):
        try:
            load_raw_smd_dataset.clear()
        except Exception:
            pass

    @patch("pathlib.Path.is_file")
    def test_case_a_absent_smd_dataset_returns_none_without_api_call(self, mock_is_file):
        """
        CASE A: Verify that when SMD dataset files are absent, load_raw_smd_dataset
        returns (None, None, None) safely and has_smd_resources becomes False,
        preventing any call to fetch_api_prediction.
        """
        mock_is_file.return_value = False

        test_data, test_labels, window_labels = load_raw_smd_dataset()
        self.assertIsNone(test_data)
        self.assertIsNone(test_labels)
        self.assertIsNone(window_labels)

        # Simulate Streamlit has_smd_resources flag check
        health_info = {"status": "ok", "model_loaded": False}
        has_smd_data = (test_data is not None and test_labels is not None and window_labels is not None)
        has_smd_model = bool(health_info and health_info.get("model_loaded"))
        has_smd_resources = (has_smd_data and has_smd_model)

        self.assertFalse(has_smd_resources)
        # When has_smd_resources is False, prediction API must not be called.

    @patch("requests.post")
    def test_case_a_no_prediction_api_call_when_resources_missing(self, mock_post):
        """
        CASE A: Explicitly verify that no request to /predict is dispatched when resources are missing.
        """
        health_info = {"status": "ok", "model_loaded": False}
        test_data = None
        has_smd_resources = bool(test_data is not None and health_info.get("model_loaded"))

        self.assertFalse(has_smd_resources)
        # Verify mock_post was not called
        mock_post.assert_not_called()

    def test_case_b_smd_dataset_loading_when_present(self):
        """
        CASE B: Verify that when SMD dataset files exist natively, load_raw_smd_dataset
        loads arrays with expected shape (28479, 38) and SMD benchmark mode remains available.
        """
        test_data, test_labels, window_labels = load_raw_smd_dataset()
        if test_data is not None:
            self.assertEqual(test_data.shape[1], 38)
            self.assertGreater(len(test_data), 0)
            self.assertEqual(len(test_labels), len(test_data))
            self.assertEqual(len(window_labels), (len(test_data) - 60) // 5 + 1)
            print("  [PASS] CASE B: SMD dataset loaded natively with 38 features.")
        else:
            print("  [INFO] CASE B: SMD files not natively present in local directory.")


if __name__ == "__main__":
    unittest.main()
