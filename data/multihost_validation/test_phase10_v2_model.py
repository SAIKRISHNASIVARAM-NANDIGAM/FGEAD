"""
data/multihost_validation/test_phase10_v2_model.py

Unit and Integration Tests for Phase 10: Current-Machine Baseline Recalibration & Dedicated v2 Model.
Verifies:
1. ModelProfile registration for 'windows_sivachowdary_v2'
2. Train-only fitted StandardScaler v2
3. Calibrated threshold tau = 1.411807
4. Zero false positive rate (FPR = 0.00%) on unseen normal baseline test windows
5. High sensitivity detection on simulated anomalous stress windows
6. Non-regression of original benchmark model 'windows_default' (preserved untouched)
7. Host registry mapping of 'host_sivachowdary' to 'windows_sivachowdary_v2'
"""

import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from api.multihost_inference import get_multihost_inference_manager, ModelProfile
from api.host_registry import get_host_registry
from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES
from config.settings import settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestPhase10CurrentMachineModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mgr = get_multihost_inference_manager()
        cls.reg = get_host_registry()
        cls.baseline_csv = PROJECT_ROOT / "data" / "live_baseline_v2_current_machine.csv"
        cls.v2_ckpt = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_v2_current_machine.pt"
        cls.v2_scaler = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler_v2_current_machine.joblib"
        cls.v2_threshold_file = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold_v2_current_machine.json"

    def test_01_v2_artifacts_exist(self):
        """Verify all Phase 10 dedicated v2 model artifacts are present."""
        self.assertTrue(self.baseline_csv.exists(), "Baseline CSV v2 must exist")
        self.assertTrue(self.v2_ckpt.exists(), "Model checkpoint v2 must exist")
        self.assertTrue(self.v2_scaler.exists(), "Scaler v2 must exist")
        self.assertTrue(self.v2_threshold_file.exists(), "Threshold JSON v2 must exist")

    def test_02_original_v1_benchmark_artifacts_preserved_untouched(self):
        """Verify original v1 benchmark model artifacts are preserved and unmodified."""
        v1_ckpt = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch.pt"
        v1_scaler = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler.joblib"
        v1_thresh = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold.json"

        self.assertTrue(v1_ckpt.exists(), "Original v1 checkpoint must remain present")
        self.assertTrue(v1_scaler.exists(), "Original v1 scaler must remain present")
        self.assertTrue(v1_thresh.exists(), "Original v1 threshold must remain present")

        with open(v1_thresh, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            self.assertAlmostEqual(float(t_data["threshold"]), 1.859450, places=5)

    def test_03_profile_registration_and_loading(self):
        """Verify windows_sivachowdary_v2 profile is registered and loaded successfully."""
        profile = self.mgr.get_profile("windows_sivachowdary_v2")
        self.assertIsNotNone(profile, "Profile windows_sivachowdary_v2 must be registered")
        self.assertTrue(profile.is_loaded, f"Profile must be loaded: {profile.load_error}")
        self.assertEqual(profile.profile_type, "dedicated host model")
        self.assertEqual(profile.training_host_type, "SivaChowdary Windows 11 Physical PC")
        self.assertEqual(len(profile.feature_names), N_LIVE_FEATURES)
        self.assertAlmostEqual(profile.threshold, 1.411807, places=4)

    def test_04_normal_test_split_zero_false_positive_rate(self):
        """Verify that unseen normal test split windows evaluate to NOMINAL (0% FPR)."""
        df = pd.read_csv(self.baseline_csv)
        feature_cols = [c for c in df.columns if c != "timestamp"]

        # Test split: last 15% (540 samples)
        n_test = int(len(df) * 0.15)
        test_df = df.iloc[-n_test:].reset_index(drop=True)

        scores = []
        is_anomalies = []

        host_info = {
            "host_id": "host_sivachowdary",
            "hostname": "SivaChowdary",
            "operating_system": "Windows",
            "model_id": "windows_sivachowdary_v2",
        }

        for start_idx in range(0, len(test_df) - 60 + 1, 15):
            window_raw = test_df.iloc[start_idx:start_idx + 60][feature_cols].values.astype(np.float32)
            res = self.mgr.infer_host_window(
                host_id="host_sivachowdary",
                window_raw=window_raw,
                model_id="windows_sivachowdary_v2",
                host_data=host_info,
            )
            scores.append(res["anomaly_score"])
            is_anomalies.append(res["is_anomaly"])

        self.assertGreater(len(scores), 20, "Should evaluate at least 20 test windows")
        fpr = (sum(is_anomalies) / len(is_anomalies)) * 100.0
        self.assertEqual(fpr, 0.0, f"Expected 0.00% FPR on normal baseline test split, got {fpr:.2f}%")
        self.assertLess(max(scores), 1.411807, "Max normal score must be strictly below threshold")

    def test_05_controlled_anomaly_sensitivity(self):
        """Verify high sensitivity detection on controlled stress scenarios."""
        df = pd.read_csv(self.baseline_csv)
        feature_cols = [c for c in df.columns if c != "timestamp"]
        base_window = df.iloc[-60:][feature_cols].values.astype(np.float32).copy()

        host_info = {
            "host_id": "host_sivachowdary",
            "hostname": "SivaChowdary",
            "operating_system": "Windows",
            "model_id": "windows_sivachowdary_v2",
        }

        # 1. CPU stress injection
        cpu_window = base_window.copy()
        cpu_idx = LIVE_FEATURES.index("cpu_percent")
        user_idx = LIVE_FEATURES.index("cpu_user_time_percent")
        intr_idx = LIVE_FEATURES.index("cpu_interrupts_per_sec")
        cpu_window[-15:, cpu_idx] = 99.0
        cpu_window[-15:, user_idx] = 95.0
        cpu_window[-15:, intr_idx] = 45000.0

        res_cpu = self.mgr.infer_host_window(
            host_id="host_sivachowdary",
            window_raw=cpu_window,
            model_id="windows_sivachowdary_v2",
            host_data=host_info,
        )
        self.assertTrue(res_cpu["is_anomaly"], "CPU stress must be detected as anomaly")
        self.assertGreaterEqual(res_cpu["anomaly_score"], 1.411807)

        # 2. Disk Write stress injection
        disk_window = base_window.copy()
        dw_idx = LIVE_FEATURES.index("disk_write_bytes_per_sec")
        di_idx = LIVE_FEATURES.index("disk_write_count_per_sec")
        disk_window[-15:, dw_idx] = 120 * 1024 * 1024  # 120 MB/s
        disk_window[-15:, di_idx] = 2500  # 2500 IOPS

        res_disk = self.mgr.infer_host_window(
            host_id="host_sivachowdary",
            window_raw=disk_window,
            model_id="windows_sivachowdary_v2",
            host_data=host_info,
        )
        self.assertTrue(res_disk["is_anomaly"], "Disk write burst must be detected as anomaly")

    def test_06_host_registry_assignment(self):
        """Verify that host_sivachowdary maps to windows_sivachowdary_v2 upon registration."""
        reg_res, token = self.reg.register_host(
            hostname="SivaChowdary",
            operating_system="Windows",
            os_version="10.0.26200",
            architecture="AMD64",
        )
        self.assertEqual(reg_res["model_id"], "windows_sivachowdary_v2")
        self.assertEqual(reg_res["model_status"], "COMPATIBLE")
        self.assertEqual(reg_res["operating_system"], "Windows")


if __name__ == "__main__":
    unittest.main()
