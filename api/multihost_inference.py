"""
api/multihost_inference.py

FGEAD Multi-Host Inference, Model Profile & Root-Cause XAI Service.
Maintains isolated inference states, score histories, episode trackers,
and model profile associations keyed strictly by host_id.
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
)
from models.fgead import FGEAD
from api.live_inference import LiveEpisodeTracker
from api.host_registry import get_host_registry


class ModelProfile:
    """
    Encapsulates a loaded FGEAD model, scaler, threshold, and explicit compatibility metadata.
    """

    def __init__(
        self,
        profile_id: str,
        name: str,
        checkpoint_path: Path,
        scaler_path: Path,
        config_path: Path,
        threshold: float,
        supported_os: Optional[List[str]] = None,
        training_host_type: str = "Windows physical host",
        training_baseline_id: str = "live_baseline_60min_windows",
        feature_schema_version: str = "1.0",
        scaler_id: str = "fgead_live_scaler_joblib",
        threshold_id: str = "fgead_live_threshold_json",
        profile_type: str = "shared baseline model",
        is_universal: bool = False,
        device: str = "cpu",
    ):
        self.profile_id = profile_id
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.threshold = float(threshold)
        self.supported_os = supported_os or ["Windows"]
        self.training_host_type = training_host_type
        self.training_baseline_id = training_baseline_id
        self.feature_schema_version = feature_schema_version
        self.feature_names = list(LIVE_FEATURES)
        self.scaler_id = scaler_id
        self.threshold_id = threshold_id
        self.profile_type = profile_type
        self.is_universal = is_universal
        self.device = device

        self.model: Optional[FGEAD] = None
        self.scaler: Any = None
        self.config: Dict[str, Any] = {}
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

        self._load()

    def _load(self) -> None:
        try:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler checkpoint not found: {self.scaler_path}")

            # Load config
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)

            # Load scaler
            self.scaler = joblib.load(self.scaler_path)

            # Load model
            ckpt = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
            self.model = FGEAD(
                n_features=N_LIVE_FEATURES,
                embed_dim=64,
                n_heads=4,
                gcn_out=64,
                lstm_hidden=128,
                sparsity_threshold=0.3,
                dropout=0.2,
                sparsity_lambda=0.01,
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.load_error = None
        except Exception as exc:
            self.is_loaded = False
            self.load_error = str(exc)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.profile_id,
            "name": self.name,
            "supported_os": self.supported_os,
            "training_host_type": self.training_host_type,
            "training_baseline_id": self.training_baseline_id,
            "feature_schema_version": self.feature_schema_version,
            "n_features": len(self.feature_names),
            "scaler_id": self.scaler_id,
            "threshold_id": self.threshold_id,
            "threshold": self.threshold,
            "profile_type": self.profile_type,
            "is_universal": self.is_universal,
            "is_loaded": self.is_loaded,
        }


class MultiHostInferenceManager:
    """
    Manages model execution, per-host score history, episode tracking, and XAI for multiple hosts.
    Enforces strict OS, schema, and baseline model compatibility.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()
        self.profiles: Dict[str, ModelProfile] = {}
        
        # Per-host states
        self._episode_trackers: Dict[str, LiveEpisodeTracker] = {}
        self._score_histories: Dict[str, List[Dict[str, Any]]] = {}
        self._latest_inferences: Dict[str, Dict[str, Any]] = {}

        self._init_default_profiles()

    def _init_default_profiles(self) -> None:
        ckpt_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch.pt"
        scaler_path = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler.joblib"
        config_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_config.json"
        thresh_path = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold.json"

        threshold_val = 1.859450
        if thresh_path.exists():
            try:
                with open(thresh_path, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                    threshold_val = float(t_data.get("threshold", 1.859450))
            except Exception:
                pass

        # Windows Default Shared Baseline Model (NOT universal)
        win_profile = ModelProfile(
            profile_id="windows_default",
            name="FGEAD Windows 22-Channel Live Model",
            checkpoint_path=ckpt_path,
            scaler_path=scaler_path,
            config_path=config_path,
            threshold=threshold_val,
            supported_os=["Windows"],
            training_host_type="Windows physical host",
            training_baseline_id="live_baseline_60min_windows",
            feature_schema_version="1.0",
            scaler_id="fgead_live_scaler_joblib",
            threshold_id="fgead_live_threshold_json",
            profile_type="shared baseline model",
            is_universal=False,
            device=self.device,
        )
        self.profiles["windows_default"] = win_profile
        self.profiles["fgead_live_windows_22ch"] = win_profile

    def get_profile(self, profile_id: str) -> Optional[ModelProfile]:
        with self._lock:
            if not profile_id or profile_id.lower() == "none":
                return None
            return self.profiles.get(profile_id) or self.profiles.get("windows_default")

    def check_compatibility(
        self,
        host_data: Dict[str, Any],
        model_id: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate whether a host can safely execute inference against the assigned model profile.
        Returns: (is_compatible, reason_message, details_dict)
        """
        # 1. OS Compatibility Gating
        host_os = str(host_data.get("operating_system", "")).strip().lower()
        if host_os == "linux":
            return False, "Anomaly inference is disabled because a compatible Linux baseline/model has not yet been trained.", {
                "status": "BASELINE_REQUIRED",
                "model_status": "BASELINE_REQUIRED",
                "reason": "incompatible_os",
                "host_os": host_data.get("operating_system"),
                "supported_os": ["Linux (pending training)"],
            }
        elif not host_os or host_os in ["unknown", "other", "freebsd", "darwin"]:
            return False, f"Anomaly inference is disabled because host operating system '{host_data.get('operating_system', 'Unknown')}' is not supported.", {
                "status": "BASELINE_REQUIRED",
                "model_status": "BASELINE_REQUIRED",
                "reason": "incompatible_os",
                "host_os": host_data.get("operating_system"),
                "supported_os": ["Windows"],
            }

        # 2. Model Profile Lookup
        target_model_id = model_id or host_data.get("model_id", "windows_default")
        if not target_model_id or target_model_id.lower() == "none":
            return False, "Anomaly inference is disabled because no model is assigned (Baseline Required).", {
                "status": "BASELINE_REQUIRED",
                "model_status": "BASELINE_REQUIRED",
                "reason": "no_model_assigned",
            }

        if target_model_id not in self.profiles:
            return False, f"Model profile '{target_model_id}' is not registered.", {
                "status": "MODEL_NOT_AVAILABLE",
                "model_status": "MODEL_NOT_AVAILABLE",
                "reason": "profile_not_found",
            }

        profile = self.profiles[target_model_id]

        if not profile.is_loaded:
            return False, f"Model profile '{profile.name}' failed to load: {profile.load_error}", {
                "status": "MODEL_NOT_LOADED",
                "model_status": "MODEL_NOT_LOADED",
                "reason": profile.load_error,
            }

        supported = [s.lower() for s in profile.supported_os]
        if host_os not in supported:
            msg = (
                f"Anomaly inference is disabled: Host OS '{host_data.get('operating_system')}' "
                f"is not supported by model '{profile.name}' (supported: {', '.join(profile.supported_os)})."
            )
            return False, msg, {
                "status": "BASELINE_REQUIRED",
                "model_status": "BASELINE_REQUIRED",
                "reason": "incompatible_os",
                "host_os": host_data.get("operating_system"),
                "supported_os": profile.supported_os,
            }

        # 2. Schema Compatibility
        schema_ver = str(host_data.get("schema_version", "1.0"))
        if schema_ver != profile.feature_schema_version:
            return False, f"Schema mismatch: Host uses v{schema_ver}, model requires v{profile.feature_schema_version}", {
                "status": "SCHEMA_MISMATCH",
                "model_status": "SCHEMA_MISMATCH",
                "reason": "schema_version_mismatch",
            }

        return True, "Compatible", {
            "status": "COMPATIBLE",
            "model_status": "COMPATIBLE",
            "model_id": profile.profile_id,
            "model_name": profile.name,
            "threshold": profile.threshold,
        }

    def get_episode_tracker(self, host_id: str) -> LiveEpisodeTracker:
        with self._lock:
            if host_id not in self._episode_trackers:
                self._episode_trackers[host_id] = LiveEpisodeTracker(debounce_frames=2)
            return self._episode_trackers[host_id]

    def infer_host_window(
        self,
        host_id: str,
        window_raw: np.ndarray,
        timestamp_iso: Optional[str] = None,
        model_id: Optional[str] = None,
        host_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute FGEAD forecasting, residual scoring, episode tracking, and 5-question XAI for a host.
        Enforces model compatibility check before performing any inference.
        """
        ts_now = timestamp_iso or datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()

        # Enforce compatibility check
        reg = get_host_registry()
        h_info = host_data or reg.get_host(host_id) or {}
        is_compat, reason, details = self.check_compatibility(h_info, model_id)
        if not is_compat:
            raise ValueError(f"Inference rejected for host '{host_id}': {reason}")

        profile = self.get_profile(model_id or h_info.get("model_id", "windows_default"))
        if not profile or not profile.is_loaded:
            raise RuntimeError(f"Model profile '{model_id}' is not loaded: {profile.load_error if profile else 'Not found'}")

        if window_raw.ndim != 2 or window_raw.shape != (60, N_LIVE_FEATURES):
            raise ValueError(f"Expected window shape (60, {N_LIVE_FEATURES}), got {window_raw.shape}")

        # 1. Scale window
        try:
            window_scaled = profile.scaler.transform(window_raw)
        except Exception as sc_err:
            raise RuntimeError(f"StandardScaler transform failed: {sc_err}")

        # 2. PyTorch Tensor
        x_tensor = torch.tensor(window_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 60, 22)

        # 3. Model Forward Pass
        with torch.no_grad():
            preds_norm, attn_weights, anomaly_scores_step = profile.model(x_tensor)
            scores_arr = anomaly_scores_step.squeeze(0).cpu().numpy()
            anomaly_score = float(np.max(scores_arr))
            peak_t = int(np.argmax(scores_arr)) + 1
            
            # Per-feature residuals
            feat_residuals = np.mean(np.abs(preds_norm[0].cpu().numpy() - x_tensor[0, 1:].cpu().numpy()), axis=0)

        # 4. Threshold & Decision
        is_anomaly = anomaly_score >= profile.threshold
        ratio = anomaly_score / profile.threshold if profile.threshold > 0 else 1.0

        if ratio >= 2.0:
            severity = "CRITICAL"
        elif ratio >= 1.3:
            severity = "HIGH"
        elif ratio >= 1.0:
            severity = "MEDIUM"
        elif ratio >= 0.8:
            severity = "ELEVATED"
        else:
            severity = "NOMINAL"

        # 5. Feature Contributions (Q1 & Q2)
        top_indices = np.argsort(feat_residuals)[::-1]
        top_features = []
        for idx in top_indices:
            feat_name = LIVE_FEATURES[idx]
            feat_res = float(feat_residuals[idx])
            raw_val = float(window_raw[-1, idx])
            top_features.append({
                "feature": feat_name,
                "description": FEATURE_DESCRIPTIONS.get(feat_name, feat_name),
                "residual": round(feat_res, 4),
                "actual_value": round(raw_val, 2),
                "formatted_value": format_physical_metric(feat_name, raw_val),
                "unit": FEATURE_UNITS.get(feat_name, ""),
            })

        # 7. Broken Graph Pairs (Q3)
        broken_pairs = []
        if attn_weights is not None:
            try:
                adj_mat = attn_weights.squeeze(0).cpu().numpy()  # (22, 22)
                top_edges = []
                for i in range(N_LIVE_FEATURES):
                    for j in range(N_LIVE_FEATURES):
                        if i != j:
                            top_edges.append((adj_mat[i, j], i, j))
                top_edges.sort(key=lambda x: x[0], reverse=True)
                for weight, i, j in top_edges[:5]:
                    broken_pairs.append({
                        "source": LIVE_FEATURES[i],
                        "target": LIVE_FEATURES[j],
                        "weight": round(float(weight), 4),
                        "status": "DISRUPTED" if is_anomaly else "COHERENT",
                    })
            except Exception:
                pass

        # 8. Episode Tracker & Alerting
        tracker = self.get_episode_tracker(host_id)
        ep_id, is_newly_opened = tracker.update(
            is_anomaly=is_anomaly,
            score=anomaly_score,
            threshold=profile.threshold,
            timestamp=ts_now,
            top_features=top_features,
        )

        # Record alert if newly opened episode
        if is_newly_opened and is_anomaly:
            try:
                reg = get_host_registry()
                dom_feat = top_features[0]["feature"] if top_features else "unknown"
                msg = f"Anomaly episode #{ep_id} started on host {host_id}. Score: {anomaly_score:.2f} >= {profile.threshold:.2f}. Dominant metric: {dom_feat}"
                reg.record_alert(
                    host_id=host_id,
                    episode_id=ep_id or 1,
                    severity=severity,
                    anomaly_score=anomaly_score,
                    threshold=profile.threshold,
                    dominant_feature=dom_feat,
                    message=msg,
                )
            except Exception as alert_err:
                print(f"[ALERT WARNING] Could not persist alert for {host_id}: {alert_err}")

        # 9. 5-Question Root Cause Synthesis
        top_f1 = top_features[0] if top_features else {"feature": "N/A", "formatted_value": "N/A"}
        top_f2 = top_features[1] if len(top_features) > 1 else {"feature": "N/A", "formatted_value": "N/A"}
        
        q1_what = (
            f"Anomaly detected on {host_id} with score {anomaly_score:.4f} exceeding baseline threshold {profile.threshold:.4f} (Severity: {severity})."
            if is_anomaly
            else f"Telemetry on {host_id} is operating nominally (Score: {anomaly_score:.4f} < Threshold: {profile.threshold:.4f})."
        )
        q2_which = f"Top deviation driven by '{top_f1['feature']}' ({top_f1['formatted_value']}) and '{top_f2['feature']}' ({top_f2['formatted_value']})."
        q3_how = f"Cross-metric feature graph indicates correlation divergence in {top_f1['feature']} relative to its peer cluster."
        q4_when = f"Window span: 60s history ending at {ts_now}."
        q5_conf = f"{min(99.9, max(50.0, ratio * 55.0)):.1f}% confidence based on empirical residual deviations."

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "host_id": host_id,
            "timestamp": ts_now,
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 6),
            "threshold": round(profile.threshold, 6),
            "ratio": round(ratio, 4),
            "severity": severity,
            "model_id": profile.profile_id,
            "model_name": profile.name,
            "model_profile_type": profile.profile_type,
            "active_episode_id": ep_id,
            "is_newly_opened_episode": is_newly_opened,
            "top_features": top_features[:5],
            "broken_pairs": broken_pairs,
            "xai_5_questions": {
                "Q1_what_happened": q1_what,
                "Q2_which_metrics_deviated": q2_which,
                "Q3_how_metrics_interacted": q3_how,
                "Q4_when_did_it_occur": q4_when,
                "Q5_detection_confidence": q5_conf,
            },
            "latency_ms": round(latency_ms, 2),
        }

        # Store in host score history
        with self._lock:
            if host_id not in self._score_histories:
                self._score_histories[host_id] = []
            self._score_histories[host_id].append({
                "timestamp": ts_now,
                "score": round(anomaly_score, 4),
                "threshold": round(profile.threshold, 4),
                "is_anomaly": is_anomaly,
                "severity": severity,
                "top_feature": top_f1["feature"],
            })
            if len(self._score_histories[host_id]) > 120:
                self._score_histories[host_id].pop(0)

            self._latest_inferences[host_id] = result

        return result

    def get_latest_inference(self, host_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._latest_inferences.get(host_id)

    def get_score_history(self, host_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._score_histories.get(host_id, []))


# Singleton instance
_GLOBAL_INFERENCE_MANAGER: Optional[MultiHostInferenceManager] = None
_INF_MANAGER_LOCK = threading.RLock()


def get_multihost_inference_manager() -> MultiHostInferenceManager:
    global _GLOBAL_INFERENCE_MANAGER
    with _INF_MANAGER_LOCK:
        if _GLOBAL_INFERENCE_MANAGER is None:
            _GLOBAL_INFERENCE_MANAGER = MultiHostInferenceManager()
        return _GLOBAL_INFERENCE_MANAGER
