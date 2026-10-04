"""
data/live_buffer.py

Thread-safe Bounded Rolling Ring Buffer for Live Telemetry Ingestion.
Stores 60 timesteps × N live features for real-time monitoring and future model inference.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from data.live_feature_schema import (
    LIVE_FEATURES,
    N_LIVE_FEATURES,
    feature_dict_to_vector,
)


class LiveRingBuffer:
    """
    Thread-safe ring buffer storing up to `window_size` consecutive telemetry points.
    """

    def __init__(self, window_size: int = 60, timeout_sec: float = 5.0):
        self.window_size = window_size
        self.timeout_sec = timeout_sec
        self.n_features = N_LIVE_FEATURES
        self.feature_names = list(LIVE_FEATURES)

        self._lock = threading.RLock()
        self._buffer: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        self._raw_dicts: deque = deque(maxlen=window_size)

        self._machine_info: Dict[str, Any] = {}
        self._last_received_time: float = 0.0
        self._total_samples_received: int = 0

    def add_sample(
        self,
        timestamp: str,
        features: Dict[str, float],
        machine_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a validated telemetry point to the buffer."""
        vec = feature_dict_to_vector(features)
        now_epoch = time.time()

        with self._lock:
            self._buffer.append(vec)
            self._timestamps.append(timestamp)
            self._raw_dicts.append(dict(features))
            self._last_received_time = now_epoch
            self._total_samples_received += 1
            if machine_info:
                self._machine_info = dict(machine_info)

    def is_agent_connected(self) -> bool:
        """Check if telemetry was received recently within timeout_sec."""
        with self._lock:
            if self._last_received_time == 0.0:
                return False
            return (time.time() - self._last_received_time) <= self.timeout_sec

    def get_status(self) -> Dict[str, Any]:
        """Return connectivity and buffer diagnostics."""
        with self._lock:
            connected = self.is_agent_connected()
            seconds_ago = round(time.time() - self._last_received_time, 1) if self._last_received_time > 0 else None
            last_ts = self._timestamps[-1] if self._timestamps else None
            curr_size = len(self._buffer)

            return {
                "connected": connected,
                "status": "🟢 CONNECTED" if connected else "🔴 DISCONNECTED",
                "last_seen_epoch": self._last_received_time,
                "last_seen_iso": last_ts,
                "seconds_since_last_seen": seconds_ago,
                "buffer_size": curr_size,
                "window_size": self.window_size,
                "is_window_full": curr_size >= self.window_size,
                "total_samples_received": self._total_samples_received,
                "n_features": self.n_features,
                "machine_info": self._machine_info,
            }

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent single telemetry sample."""
        with self._lock:
            if not self._buffer:
                return None
            return {
                "timestamp": self._timestamps[-1],
                "features": dict(self._raw_dicts[-1]),
                "machine_info": dict(self._machine_info),
                "is_connected": self.is_agent_connected(),
            }

    def get_history(self) -> Dict[str, Any]:
        """Return full rolling history for dashboard visualization."""
        with self._lock:
            if not self._buffer:
                return {
                    "timestamps": [],
                    "features": {f: [] for f in self.feature_names},
                    "count": 0,
                }

            timestamps = list(self._timestamps)
            feat_series = {f: [] for f in self.feature_names}
            for d in self._raw_dicts:
                for f in self.feature_names:
                    feat_series[f].append(d.get(f, 0.0))

            return {
                "timestamps": timestamps,
                "features": feat_series,
                "count": len(timestamps),
            }

    def get_window_tensor(self) -> Tuple[Optional[np.ndarray], bool]:
        """
        Return the 2D window array of shape (window_size, n_features) if full.
        """
        with self._lock:
            if len(self._buffer) < self.window_size:
                return None, False
            arr = np.array(list(self._buffer), dtype=np.float32)
            return arr, True
