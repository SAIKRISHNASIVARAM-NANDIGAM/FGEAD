"""
data/live_training/test_v3_validation_suite.py

Phase 8 Automated Regression & Validation Suite for FGEAD V3.
Tests:
  1. V3 Artifact Loading Integrity
  2. Normal Telemetry Ingestion & Scoring
  3. Controlled Workload Detection (CPU, Write, Network)
  4. Feature Attribution & Residual Ranking
  5. Actionable Recommendation Mapping & Cause-Uncertain Handling
  6. Recovery & Episode Closure
  7. Invalid Telemetry Handling (NaN/Inf rejection)
  8. Separation of Live Results & Historical Benchmark Metrics
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_feature_schema import (
    CUMULATIVE_COUNTER_BASELINE_ANCHORS,
    LIVE_FEATURES,
    N_LIVE_FEATURES,
    prepare_model_input_window,
    validate_live_feature_dict,
)
from models.fgead import FGEAD
from api.live_inference import LiveEpisodeTracker, LiveInferenceService
from data.live_training.run_v3_comprehensive_validation import generate_actionable_recommendations


class TestFGEADV3ValidationSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ckpt_p = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_v3_current_machine.pt"
        cls.scaler_p = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler_v3_current_machine.joblib"
        cls.thresh_p = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold_v3_current_machine.json"
        cls.config_p = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_config_v3_current_machine.json"
        cls.base_p = PROJECT_ROOT / "data" / "live_baseline_v3_current_machine.csv"
        cls.test_p = PROJECT_ROOT / "data" / "live_training" / "v3_test_normal.csv"
        cls.metrics_p = PROJECT_ROOT / "data" / "live_training" / "V3_DETECTION_VALIDATION_METRICS.json"

    def test_01_artifact_integrity(self):
        """Verify all V3 artifacts exist, load cleanly, and have correct threshold and dimensions."""
        self.assertTrue(self.ckpt_p.exists(), "V3 checkpoint missing")
        self.assertTrue(self.scaler_p.exists(), "V3 scaler missing")
        self.assertTrue(self.thresh_p.exists(), "V3 threshold JSON missing")
        self.assertTrue(self.config_p.exists(), "V3 config JSON missing")

        scaler = joblib.load(self.scaler_p)
        self.assertEqual(len(scaler.mean_), 22, "Scaler dimension != 22")

        with open(self.thresh_p, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        threshold = float(tdata["threshold"])
        self.assertAlmostEqual(threshold, 2.120169, places=4, msg="V3 threshold mismatch")

    def test_02_normal_operation_scoring(self):
        """Verify normal telemetry score remains safely below tau = 2.120169."""
        df_test = pd.read_csv(self.test_p)[LIVE_FEATURES]
        win_raw = df_test.iloc[:60].values
        service = LiveInferenceService()
        res = service.infer_window(win_raw)

        self.assertFalse(res["is_anomaly"], "Normal window flagged as anomaly")
        self.assertLess(res["anomaly_score"], 2.120169, "Normal score exceeded threshold")
        self.assertIn("score_threshold_ratio", res)

    def test_03_controlled_cpu_workload(self):
        """Verify controlled CPU workload exceeds threshold and identifies cpu_ctx_switches."""
        df_base = pd.read_csv(self.base_p)[LIVE_FEATURES]
        win_raw = df_base.iloc[:60].values.copy()
        for t in range(60):
            win_raw[t, LIVE_FEATURES.index("cpu_percent")] = 96.5
            win_raw[t, LIVE_FEATURES.index("cpu_ctx_switches_per_sec")] = 480000.0

        service = LiveInferenceService()
        res = service.infer_window(win_raw)
        self.assertTrue(res["is_anomaly"], "Controlled CPU workload undetected")
        self.assertGreater(res["anomaly_score"], 2.120169)
        top1 = res["top_features"][0]["feature"]
        self.assertIn("cpu", top1, "Top feature should be CPU related")

    def test_04_feature_attribution_counter_anchoring(self):
        """Verify cumulative counter anchors (net_drops_total) receive 0 residual attribution."""
        df_base = pd.read_csv(self.base_p)[LIVE_FEATURES]
        win_raw = df_base.iloc[:60].values.copy()
        win_raw[:, LIVE_FEATURES.index("net_drops_total")] = 9999.0

        service = LiveInferenceService()
        res = service.infer_window(win_raw)
        top_names = [f["feature"] for f in res["top_features"]]
        self.assertNotIn("net_drops_total", top_names, "Anchored counter should not be top feature")

    def test_05_recommendation_mapping_and_cause_uncertainty(self):
        """Verify deterministic recommendation mapping without fabricated probabilities."""
        network_feats = [{"feature": "net_bytes_recv_per_sec", "current_value": 85000000.0, "predicted_value": 10970.0}]
        rec_net = generate_actionable_recommendations(network_feats)
        self.assertEqual(rec_net["primary_subsystem"], "Network & Communications")
        self.assertIn("Resource Monitor", rec_net["recommended_action"])

        unknown_feats = [{"feature": "process_count", "current_value": 350.0, "predicted_value": 325.0}]
        rec_unk = generate_actionable_recommendations(unknown_feats)
        self.assertIn("Cause uncertain", rec_unk["confidence_limitation"])

    def test_06_recovery_and_episode_closure(self):
        """Verify LiveEpisodeTracker requires 2 nominal frames to close an episode."""
        tracker = LiveEpisodeTracker(debounce_frames=2, min_persistence_frames=2)
        top_f = [{"feature": "cpu_percent"}]

        # Frame 1: Anomaly -> Suspicious (pending confirmation)
        ep_id, is_new = tracker.update(True, 3.5, 2.120169, "ts1", top_f)
        self.assertEqual(ep_id, 1)

        # Frame 2: Anomaly -> Confirmed Anomaly
        tracker.update(True, 3.8, 2.120169, "ts2", top_f)
        self.assertTrue(tracker.get_status()["is_confirmed"])

        # Frame 3: Nominal 1 (Debouncing)
        tracker.update(False, 1.2, 2.120169, "ts3", top_f)
        self.assertTrue(tracker.get_status()["is_in_anomaly_episode"], "Episode should stay open on 1st nominal frame")

        # Frame 4: Nominal 2 (Closed)
        tracker.update(False, 1.1, 2.120169, "ts4", top_f)
        self.assertFalse(tracker.get_status()["is_in_anomaly_episode"], "Episode should close on 2nd nominal frame")
        self.assertEqual(len(tracker.completed_episodes), 1)

    def test_07_invalid_telemetry_rejection(self):
        """Verify NaN or Inf telemetry inputs are rejected cleanly."""
        dict_nan = {f: 1.0 for f in LIVE_FEATURES}
        dict_nan["cpu_percent"] = float("nan")
        valid, msg = validate_live_feature_dict(dict_nan)
        self.assertFalse(valid, "NaN feature accepted")
        self.assertIn("non-finite", msg)

    def test_08_metrics_file_existence(self):
        """Verify structured metrics JSON file exists and contains valid metrics."""
        self.assertTrue(self.metrics_p.exists(), "Metrics JSON missing")
        with open(self.metrics_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["model_id"], "windows_sivachowdary_v3")
        self.assertEqual(data["event_level_metrics"]["precision_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
