"""
data/multihost_buffer.py

Thread-safe Multi-Host Telemetry Buffer Manager for FGEAD.
Maintains strictly isolated 60-sample rolling ring buffers for each registered host.
Guarantees Host A telemetry is never mixed with Host B telemetry.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from data.live_buffer import LiveRingBuffer
from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES


class MultiHostBufferManager:
    """
    Manages a dictionary of LiveRingBuffer instances keyed strictly by `host_id`.
    Thread-safe and provides per-host isolation.
    """

    def __init__(self, window_size: int = 60, timeout_sec: float = 15.0):
        self.window_size = window_size
        self.timeout_sec = timeout_sec
        self._lock = threading.RLock()
        self._buffers: Dict[str, LiveRingBuffer] = {}

    def get_or_create_buffer(self, host_id: str) -> LiveRingBuffer:
        """Retrieve the dedicated buffer for a host, creating it if it doesn't exist."""
        with self._lock:
            if host_id not in self._buffers:
                self._buffers[host_id] = LiveRingBuffer(
                    window_size=self.window_size,
                    timeout_sec=self.timeout_sec,
                )
            return self._buffers[host_id]

    def get_buffer(self, host_id: str) -> Optional[LiveRingBuffer]:
        with self._lock:
            return self._buffers.get(host_id)

    def has_buffer(self, host_id: str) -> bool:
        with self._lock:
            return host_id in self._buffers

    def add_sample(
        self,
        host_id: str,
        timestamp: str,
        features: Dict[str, float],
        machine_info: Optional[Dict[str, Any]] = None,
    ) -> LiveRingBuffer:
        """Add a validated telemetry point to a host's isolated buffer."""
        buf = self.get_or_create_buffer(host_id)
        buf.add_sample(timestamp=timestamp, features=features, machine_info=machine_info)
        return buf

    def get_window_tensor(self, host_id: str) -> Tuple[Optional[np.ndarray], bool]:
        """Return (array, is_full) of shape (60, 22) for the specified host."""
        with self._lock:
            if host_id not in self._buffers:
                return None, False
            return self._buffers[host_id].get_window_tensor()

    def get_latest(self, host_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent telemetry sample for a specific host."""
        with self._lock:
            if host_id not in self._buffers:
                return None
            return self._buffers[host_id].get_latest()

    def get_history(self, host_id: str) -> Dict[str, Any]:
        """Return rolling history series for a specific host."""
        with self._lock:
            if host_id not in self._buffers:
                return {
                    "timestamps": [],
                    "features": {f: [] for f in LIVE_FEATURES},
                    "count": 0,
                }
            return self._buffers[host_id].get_history()

    def get_status(self, host_id: str) -> Dict[str, Any]:
        """Return buffer diagnostics for a specific host."""
        with self._lock:
            if host_id not in self._buffers:
                return {
                    "connected": False,
                    "status": "🔴 UNKNOWN / NO DATA",
                    "last_seen_epoch": 0.0,
                    "last_seen_iso": None,
                    "seconds_since_last_seen": None,
                    "buffer_size": 0,
                    "window_size": self.window_size,
                    "is_window_full": False,
                    "total_samples_received": 0,
                    "n_features": N_LIVE_FEATURES,
                    "machine_info": {},
                }
            return self._buffers[host_id].get_status()

    def get_all_host_ids(self) -> List[str]:
        """Return all active host IDs with buffers."""
        with self._lock:
            return list(self._buffers.keys())


# Singleton instance
_GLOBAL_BUFFER_MANAGER: Optional[MultiHostBufferManager] = None
_BUFFER_MANAGER_LOCK = threading.RLock()


def get_multihost_buffer_manager(window_size: int = 60, timeout_sec: float = 15.0) -> MultiHostBufferManager:
    global _GLOBAL_BUFFER_MANAGER
    with _BUFFER_MANAGER_LOCK:
        if _GLOBAL_BUFFER_MANAGER is None:
            _GLOBAL_BUFFER_MANAGER = MultiHostBufferManager(
                window_size=window_size,
                timeout_sec=timeout_sec,
            )
        return _GLOBAL_BUFFER_MANAGER
