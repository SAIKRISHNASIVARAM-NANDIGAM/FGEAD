"""
api/live_inference.py

FGEAD Dedicated Live Inference & Explainability Service for Windows Physical Host.
Performs real-time neural forecasting, anomaly scoring, episode tracking, and
5-Question root-cause XAI on 60-second rolling telemetry windows (60 × 22).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from data.live_feature_schema import (
    FEATURE_DESCRIPTIONS,
    FEATURE_UNITS,
    LIVE_FEATURES,
    LIVE_FEATURE_VERSION,
    N_LIVE_FEATURES,
    format_physical_metric,
    prepare_model_input_window,
    CUMULATIVE_COUNTER_BASELINE_ANCHORS,
)

from models.fgead import FGEAD
from api.deep_explainability import build_deep_human_explanation


class LiveEpisodeTracker:
    """
    Tracks contiguous anomalous sliding windows (W=60, S=1) as single coherent episodes.
    Uses debounce hysteresis and persistence gating to prevent false-positive alert flickering.
    """

    def __init__(self, debounce_frames: int = 2, min_persistence_frames: int = 2):
        self._lock = threading.RLock()
        self.debounce_frames = debounce_frames
        self.min_persistence_frames = min_persistence_frames
        self.episode_counter: int = 0
        self.current_episode: Optional[Dict[str, Any]] = None
        self._consecutive_nominal: int = 0
        self.completed_episodes: List[Dict[str, Any]] = []

    def update(
        self,
        is_anomaly: bool,
        score: float,
        threshold: float,
        timestamp: str,
        top_features: List[Dict[str, Any]],
    ) -> Tuple[Optional[int], bool]:
        """
        Update tracker with current window result.
        Returns: (active_episode_id, is_newly_opened)
        """
        with self._lock:
            if is_anomaly:
                self._consecutive_nominal = 0
                if self.current_episode is None:
                    # Open new episode (Pending confirmation or Confirmed)
                    self.episode_counter += 1
                    is_conf = (self.min_persistence_frames <= 1)
                    self.current_episode = {
                        "episode_id": self.episode_counter,
                        "start_time": timestamp,
                        "last_seen_time": timestamp,
                        "duration_sec": 1,
                        "peak_score": score,
                        "scores": [score],
                        "flagged_windows_count": 1,
                        "threshold": threshold,
                        "dominant_features": [f["feature"] for f in top_features[:3]],
                        "is_confirmed": is_conf,
                        "decision_state": "ANOMALY" if is_conf else "SUSPICIOUS",
                    }
                    return self.episode_counter, True
                else:
                    # Update active episode
                    ep = self.current_episode
                    ep["last_seen_time"] = timestamp
                    ep["scores"].append(score)
                    ep["flagged_windows_count"] += 1
                    ep["duration_sec"] += 1
                    if score > ep["peak_score"]:
                        ep["peak_score"] = score
                    # Update dominant features if higher score
                    ep["dominant_features"] = [f["feature"] for f in top_features[:3]]
                    if ep["flagged_windows_count"] >= self.min_persistence_frames:
                        ep["is_confirmed"] = True
                        ep["decision_state"] = "ANOMALY"
                    return ep["episode_id"], False
            else:
                # Nominal window
                if self.current_episode is not None:
                    self._consecutive_nominal += 1
                    if self._consecutive_nominal >= self.debounce_frames:
                        # Close episode
                        ep = self.current_episode
                        mean_sc = float(np.mean(ep["scores"]))
                        completed_ep = {
                            "episode_id": ep["episode_id"],
                            "start_time": ep["start_time"],
                            "end_time": ep["last_seen_time"],
                            "duration_sec": ep["duration_sec"],
                            "peak_score": round(ep["peak_score"], 4),
                            "mean_score": round(mean_sc, 4),
                            "flagged_windows_count": ep["flagged_windows_count"],
                            "dominant_features": ep["dominant_features"],
                            "is_confirmed": ep.get("is_confirmed", False),
                        }
                        self.completed_episodes.append(completed_ep)
                        if len(self.completed_episodes) > 50:
                            self.completed_episodes.pop(0)
                        self.current_episode = None
                        self._consecutive_nominal = 0

                return None, False

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            if self.current_episode is not None:
                ep = self.current_episode
                last_score = ep["scores"][-1] if ep["scores"] else ep["peak_score"]
                is_conf = ep.get("is_confirmed", ep["flagged_windows_count"] >= self.min_persistence_frames)
                return {
                    "is_in_anomaly_episode": is_conf,
                    "is_pending_suspicious": not is_conf,
                    "active_episode_id": ep["episode_id"],
                    "start_time": ep["start_time"],
                    "duration_sec": ep["duration_sec"],
                    "peak_score": round(ep["peak_score"], 4),
                    "current_score": round(last_score, 4),
                    "flagged_windows_count": ep["flagged_windows_count"],
                    "dominant_features": ep["dominant_features"],
                    "is_confirmed": is_conf,
                    "decision_state": "ANOMALY" if is_conf else "SUSPICIOUS",
                }
            return {
                "is_in_anomaly_episode": False,
                "is_pending_suspicious": False,
                "active_episode_id": None,
                "start_time": None,
                "duration_sec": 0,
                "peak_score": None,
                "current_score": None,
                "flagged_windows_count": 0,
                "dominant_features": [],
                "is_confirmed": False,
                "decision_state": "NORMAL",
                "total_completed_episodes": len(self.completed_episodes),
            }



class LiveInferenceService:
    """
    Dedicated production inference service for the 22-channel FGEAD Windows model.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/fgead_live_windows_22ch.pt",
        scaler_path: str = "checkpoints/fgead_live_scaler.joblib",
        threshold_path: str = "checkpoints/fgead_live_threshold.json",
        config_path: str = "checkpoints/fgead_live_windows_22ch_config.json",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()

        self.ckpt_path = Path(checkpoint_path)
        if not self.ckpt_path.is_absolute():
            self.ckpt_path = PROJECT_ROOT / self.ckpt_path

        self.scaler_path = Path(scaler_path)
        if not self.scaler_path.is_absolute():
            self.scaler_path = PROJECT_ROOT / self.scaler_path

        self.threshold_path = Path(threshold_path)
        if not self.threshold_path.is_absolute():
            self.threshold_path = PROJECT_ROOT / self.threshold_path

        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path

        self.model: Optional[FGEAD] = None
        self.scaler: Optional[Any] = None
        self.threshold: float = 1.859450
        self.config: Dict[str, Any] = {}
        self.is_ready: bool = False
        self.init_error: Optional[str] = None

        self.feature_names = list(LIVE_FEATURES)
        self.window_size: int = 60
        self.n_features: int = N_LIVE_FEATURES

        self.episode_tracker = LiveEpisodeTracker(debounce_frames=2)
        self.latest_result: Optional[Dict[str, Any]] = None
        self.score_history: List[Dict[str, Any]] = []

        self._load_artifacts()

    def _load_artifacts(self):
        """Strictly validate and load model artifacts."""
        try:
            # 1. Verify existence
            for p, name in [
                (self.ckpt_path, "Checkpoint"),
                (self.scaler_path, "Scaler"),
                (self.threshold_path, "Threshold JSON"),
                (self.config_path, "Config JSON"),
            ]:
                if not p.exists():
                    raise FileNotFoundError(f"{name} file not found at: {p}")

            # 2. Load and validate Config
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            cfg_feats = self.config.get("n_features", 0)
            if cfg_feats != N_LIVE_FEATURES:
                raise ValueError(f"Config n_features ({cfg_feats}) != {N_LIVE_FEATURES}")

            # 3. Load and validate Threshold
            with open(self.threshold_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            self.threshold = float(t_data.get("threshold", 1.859450))

            # 4. Load Scaler
            self.scaler = joblib.load(self.scaler_path)
            if not hasattr(self.scaler, "transform") or not hasattr(self.scaler, "mean_"):
                raise ValueError("Loaded scaler object is invalid or uncalibrated")
            if len(self.scaler.mean_) != N_LIVE_FEATURES:
                raise ValueError(f"Scaler features ({len(self.scaler.mean_)}) != {N_LIVE_FEATURES}")

            # 5. Load PyTorch Model Checkpoint
            ckpt_data = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            if ckpt_data.get("n_features") != N_LIVE_FEATURES:
                raise ValueError(
                    f"Checkpoint n_features ({ckpt_data.get('n_features')}) != {N_LIVE_FEATURES}"
                )

            self.model = FGEAD(
                n_features=N_LIVE_FEATURES,
                embed_dim=64,
                n_heads=4,
                gcn_out=64,
                lstm_hidden=128,
                sparsity_threshold=0.3,
                dropout=0.2,
                sparsity_lambda=0.01,
            ).to(self.device)

            self.model.load_state_dict(ckpt_data["model_state_dict"])
            self.model.eval()

            self.is_ready = True
            self.init_error = None
            print(f"[LIVE INFERENCE SERVICE] Initialized successfully. Model: 22 Channels | Threshold τ = {self.threshold:.6f}")

        except Exception as e:
            self.is_ready = False
            self.init_error = str(e)
            print(f"[LIVE INFERENCE SERVICE] Initialization ERROR: {e}")

    def generate_five_question_explanation(
        self,
        is_anomaly: bool,
        score: float,
        threshold: float,
        top_features: List[Dict[str, Any]],
        timestamp_iso: str,
        episode_info: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Construct structured 5-Question Explainable AI Narrative using cautious, scientifically defensible language.
        """
        ratio = round(score / threshold, 2)
        top_names = [f["feature_description"] for f in top_features[:3]]
        dominant_subsystems = list(set([f["subsystem"] for f in top_features[:3]]))
        subsystem_str = ", ".join(dominant_subsystems) if dominant_subsystems else "General OS Activity"

        if is_anomaly:
            q1 = "A rolling 60-second telemetry window departed significantly from the learned normal baseline forecasting pattern."
            q2 = f"Observed at {timestamp_iso} UTC (Active Anomaly Episode #{episode_info.get('active_episode_id', 1)}, ongoing for {episode_info.get('duration_sec', 1)}s)."
            q3 = f"High residual excursion: Anomaly Score is {score:.4f} against calibrated threshold τ = {threshold:.4f} ({ratio}x above threshold)."
            q4 = f"Top contributing forecast residuals observed in: {', '.join(top_names)}."
            q5 = f"Primary activity appears to be associated with {subsystem_str} operations."
        else:
            q1 = "Nominal system operation — all 22 physical telemetry counters match baseline spatio-temporal forecasts."
            q2 = f"Evaluated at {timestamp_iso} UTC."
            q3 = f"Nominal: Anomaly Score is {score:.4f}, remaining safely below threshold τ = {threshold:.4f} ({ratio}x of threshold limit)."
            q4 = "All hardware metrics (CPU, RAM, Disk I/O, Network, and OS Processes) are behaving within normal calibrated bounds."
            q5 = "All system subsystems are operating within nominal baseline bounds."

        return {
            "what_happened": q1,
            "when": q2,
            "how_severe": q3,
            "what_contributed": q4,
            "which_subsystem": q5,
        }

    def infer_window(
        self,
        window_raw: np.ndarray,
        timestamp_iso: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full inference on a single 60 × 22 raw physical telemetry window.
        """
        if not self.is_ready:
            raise RuntimeError(f"Live inference service is not ready: {self.init_error}")

        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()
        arr = np.asarray(window_raw, dtype=np.float32)

        # 1. Dimension & Quality Validation
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D window tensor, received shape {arr.shape}")
        if arr.shape[0] != self.window_size:
            raise ValueError(f"Window length must be exactly {self.window_size}, received {arr.shape[0]}")
        if arr.shape[1] != self.n_features:
            raise ValueError(f"Feature count must be exactly {self.n_features}, received {arr.shape[1]}")

        if np.isnan(arr).any():
            raise ValueError("Input window contains NaN values")
        if np.isinf(arr).any():
            raise ValueError("Input window contains Infinite values")

        with self._lock:
            # 2. Normalize using train-fitted scaler (with baseline anchor for invariant cumulative counters)
            model_input_arr = prepare_model_input_window(arr)
            norm_window = self.scaler.transform(model_input_arr).astype(np.float32)  # (60, 22)
            tensor_x = torch.from_numpy(norm_window).unsqueeze(0).to(self.device)  # (1, 60, 22)

            t0 = time.perf_counter()
            with torch.no_grad():
                preds_norm, attn_weights, anomaly_scores_step = self.model(tensor_x)
                # preds_norm: (1, 59, 22), attn_weights: (1, 22, 22), anomaly_scores_step: (1, 59)
                scores_arr = anomaly_scores_step.squeeze(0).cpu().numpy()
                window_anomaly_score = float(np.max(scores_arr))
                peak_t = int(np.argmax(scores_arr)) + 1  # 1..59

                # Actual vs Predicted at peak timestep
                preds_norm_peak = preds_norm[0, peak_t - 1].cpu().numpy().reshape(1, -1)
                actual_norm_peak = tensor_x[0, peak_t].cpu().numpy().reshape(1, -1)

                # Unscaled actual & predicted values for intuitive explainability
                preds_unscaled = self.scaler.inverse_transform(preds_norm_peak)[0]
                actual_unscaled = arr[peak_t]

                # Per-feature residuals across the window
                preds_all_unscaled = self.scaler.inverse_transform(preds_norm[0].cpu().numpy())
                actual_all_unscaled = arr[1:]
                feat_residuals = np.mean(np.abs(preds_norm[0].cpu().numpy() - tensor_x[0, 1:].cpu().numpy()), axis=0)

                # Zero out anchored cumulative counters so they cannot become anomaly evidence
                for anchor_feat in CUMULATIVE_COUNTER_BASELINE_ANCHORS:
                    if anchor_feat in self.feature_names:
                        a_idx = self.feature_names.index(anchor_feat)
                        feat_residuals[a_idx] = 0.0

            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            is_anomaly = bool(window_anomaly_score > self.threshold)
            ratio = round(window_anomaly_score / self.threshold, 4)

            # Top contributing features
            sorted_indices = np.argsort(feat_residuals)[::-1]
            total_residual_sum = float(np.sum(feat_residuals)) if np.sum(feat_residuals) > 0 else 1.0

            top_features: List[Dict[str, Any]] = []
            for rank, idx in enumerate(sorted_indices[:5], start=1):
                f_name = self.feature_names[idx]
                f_desc = FEATURE_DESCRIPTIONS.get(f_name, f_name)
                subsystem = (
                    "Storage I/O" if "disk" in f_name
                    else "CPU Subsystem" if "cpu" in f_name
                    else "Memory" if "mem" in f_name or "swap" in f_name
                    else "Network" if "net" in f_name
                    else "Operating System"
                )
                contrib_pct = round((float(feat_residuals[idx]) / total_residual_sum) * 100.0, 1)

                act_val = float(actual_unscaled[idx])
                pred_val = float(preds_unscaled[idx])
                unit_str = FEATURE_UNITS.get(f_name, "")
                act_fmt = format_physical_metric(f_name, act_val)
                pred_fmt = format_physical_metric(f_name, pred_val)

                top_features.append({
                    "rank": rank,
                    "feature": f_name,
                    "feature_description": f_desc,
                    "subsystem": subsystem,
                    "unit": unit_str,
                    "current_value": round(act_val, 2),
                    "predicted_value": round(pred_val, 2),
                    "current_value_formatted": act_fmt,
                    "predicted_value_formatted": pred_fmt,
                    "residual": round(float(feat_residuals[idx]), 4),
                    "contribution_pct": contrib_pct,
                })


            # Debug diagnostics
            drops_idx = self.feature_names.index("net_drops_total") if "net_drops_total" in self.feature_names else -1
            phys_drops = float(arr[-1, drops_idx]) if drops_idx >= 0 else 0.0
            model_drops = float(model_input_arr[-1, drops_idx]) if drops_idx >= 0 else 0.0
            b_mean = float(self.scaler.mean_[drops_idx]) if (self.scaler is not None and hasattr(self.scaler, "mean_") and drops_idx >= 0) else 294.0
            b_std = float(self.scaler.scale_[drops_idx]) if (self.scaler is not None and hasattr(self.scaler, "scale_") and drops_idx >= 0) else 1.0
            drops_res = float(feat_residuals[drops_idx]) if drops_idx >= 0 else 0.0

            # Episode Tracker Update
            active_ep_id, is_new_ep = self.episode_tracker.update(
                is_anomaly=is_anomaly,
                score=window_anomaly_score,
                threshold=self.threshold,
                timestamp=ts,
                top_features=top_features,
            )
            ep_status = self.episode_tracker.get_status()

            # Severity Categorization
            if not is_anomaly:
                severity = "🟢 NOMINAL"
            elif ratio >= 2.5:
                severity = "🔴 CRITICAL EXCURSION"
            elif ratio >= 1.5:
                severity = "🟠 HIGH EXCURSION"
            else:
                severity = "🟡 MODERATE EXCURSION"

            # 5-Question Narrative
            explanation = self.generate_five_question_explanation(
                is_anomaly=is_anomaly,
                score=window_anomaly_score,
                threshold=self.threshold,
                top_features=top_features,
                timestamp_iso=ts,
                episode_info=ep_status,
            )

            # Deep Human-Understandable Explainability
            decision_state = ep_status.get("decision_state", "ANOMALY" if is_anomaly else "NORMAL")
            deep_explanation = build_deep_human_explanation(
                state=decision_state,
                anomaly_score=window_anomaly_score,
                threshold=self.threshold,
                top_features=top_features,
                timestamp_iso=ts,
                episode_info=ep_status,
                host_id="live_windows_host",
            )

            result = {
                "model": "fgead_live_windows_22ch",
                "schema_version": LIVE_FEATURE_VERSION,
                "window_size": self.window_size,
                "n_features": self.n_features,
                "timestamp": ts,
                "is_anomaly": is_anomaly,
                "anomaly_score": round(window_anomaly_score, 6),
                "threshold": round(self.threshold, 6),
                "score_threshold_ratio": ratio,
                "severity": severity,
                "peak_timestep": peak_t,
                "latency_ms": latency_ms,
                "top_features": top_features,
                "five_question_explanation": explanation,
                "human_explanation": deep_explanation,
                "deep_explanation": deep_explanation,
                "episode_status": ep_status,
            }

            self.latest_result = result
            self.score_history.append({
                "timestamp": ts,
                "anomaly_score": round(window_anomaly_score, 4),
                "threshold": round(self.threshold, 4),
                "is_anomaly": is_anomaly,
                "severity": severity,
            })
            if len(self.score_history) > 120:
                self.score_history.pop(0)

            return result

    def get_latest_inference(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.latest_result

    def get_score_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.score_history)

    def reset_episodes(self) -> None:
        """Clear in-memory episode state and score history."""
        with self._lock:
            self.episode_tracker = LiveEpisodeTracker(debounce_frames=2)
            self.score_history = []
            self.latest_result = None


# Global singleton live inference instance
_live_inference_service = LiveInferenceService()


def get_live_inference_service() -> LiveInferenceService:
    return _live_inference_service
