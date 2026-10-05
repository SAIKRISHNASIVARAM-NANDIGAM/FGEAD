"""
api/host_registry.py

FGEAD Multi-Host Registry and Secure Agent Authentication Manager.
Provides persistent storage (SQLite + Thread-safe in-memory caching) for:
- Registered hosts and agent metadata
- Secure hashed agent tokens (SHA-256)
- Model profile assignments and baseline configurations
- Offline host detection and status tracking
- Anomaly episodes and alerts
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config.settings import settings
DEFAULT_DB_PATH = Path(settings.DATABASE_PATH)


class HostRegistry:
    """
    Thread-safe persistent registry for multi-host monitoring in FGEAD.
    """

    def __init__(self, db_path: Optional[Path | str] = None, offline_timeout_sec: Optional[float] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.offline_timeout_sec = offline_timeout_sec if offline_timeout_sec is not None else settings.OFFLINE_TIMEOUT_SEC
        self._lock = threading.RLock()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def is_healthy(self) -> bool:
        """Check if SQLite database connection and schema are healthy."""
        try:
            with self._lock, self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return cursor.fetchone() is not None
        except Exception:
            return False

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            # Hosts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hosts (
                    host_id TEXT PRIMARY KEY,
                    hostname TEXT NOT NULL,
                    operating_system TEXT NOT NULL,
                    os_version TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    agent_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OFFLINE',
                    model_id TEXT NOT NULL DEFAULT 'windows_default',
                    model_status TEXT NOT NULL DEFAULT 'COMPATIBLE',
                    baseline_id TEXT,
                    is_enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_seen TEXT,
                    last_seen_epoch REAL DEFAULT 0.0,
                    machine_info_json TEXT,
                    baseline_stats_json TEXT
                )
            """)
            # Migration check for existing databases
            cursor.execute("PRAGMA table_info(hosts)")
            existing_cols = [r["name"] for r in cursor.fetchall()]
            if "model_status" not in existing_cols:
                cursor.execute("ALTER TABLE hosts ADD COLUMN model_status TEXT DEFAULT 'COMPATIBLE'")
            if "baseline_stats_json" not in existing_cols:
                cursor.execute("ALTER TABLE hosts ADD COLUMN baseline_stats_json TEXT")

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id TEXT NOT NULL,
                    episode_id INTEGER NOT NULL,
                    severity TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    dominant_feature TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    is_resolved INTEGER DEFAULT 0,
                    FOREIGN KEY(host_id) REFERENCES hosts(host_id)
                )
            """)
            # Baselines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS baselines (
                    baseline_id TEXT PRIMARY KEY,
                    host_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    sample_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    notes TEXT,
                    stats_json TEXT,
                    FOREIGN KEY(host_id) REFERENCES hosts(host_id)
                )
            """)

            # Unique identity index: Prevents duplicate active registrations for the same physical host + OS
            cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_hosts_unique_identity
                ON hosts (LOWER(hostname), LOWER(operating_system))
                WHERE is_enabled = 1;
            """)
            conn.commit()

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()

    def register_host(
        self,
        hostname: str,
        operating_system: str,
        os_version: str,
        architecture: str = "x86_64",
        agent_version: str = "1.0.0",
        schema_version: str = "1.0",
        model_id: Optional[str] = None,
        machine_info: Optional[Dict[str, Any]] = None,
        custom_host_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Register a new host or re-register an existing host.
        Reuses existing host_id if matching hostname + OS / machine_id already registered to prevent duplicates.
        Enforces deterministic canonical host_id generation and model compatibility assignment.
        """
        import re
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            now_epoch = time.time()
            info_json = json.dumps(machine_info or {})
            is_windows = operating_system.strip().lower() == "windows"
            h_name = hostname.strip()
            os_name = operating_system.strip()

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Multi-tier deterministic identity matching:
                existing_row = None

                # Tier 1: Explicit custom_host_id match
                if custom_host_id:
                    cursor.execute("SELECT * FROM hosts WHERE host_id = ?", (custom_host_id,))
                    existing_row = cursor.fetchone()

                # Tier 2: Machine ID (Hardware Node) match
                if existing_row is None and machine_info and machine_info.get("machine_id"):
                    m_id = str(machine_info["machine_id"]).strip()
                    if m_id:
                        cursor.execute(
                            "SELECT * FROM hosts WHERE json_extract(machine_info_json, '$.machine_id') = ? AND LOWER(operating_system) = LOWER(?) AND is_enabled = 1 ORDER BY last_seen_epoch DESC",
                            (m_id, os_name),
                        )
                        existing_row = cursor.fetchone()

                # Tier 3: Case-insensitive (hostname, operating_system) match
                if existing_row is None:
                    cursor.execute(
                        "SELECT * FROM hosts WHERE LOWER(hostname) = LOWER(?) AND LOWER(operating_system) = LOWER(?) AND is_enabled = 1 ORDER BY last_seen_epoch DESC",
                        (h_name, os_name),
                    )
                    existing_row = cursor.fetchone()

                if existing_row:
                    host_id = existing_row["host_id"]
                    agent_id = existing_row["agent_id"]
                    # If host was OFFLINE, update to ONLINE / TELEMETRY_ONLY; otherwise preserve status
                    if existing_row["status"] == "OFFLINE":
                        status = "ONLINE" if is_windows else "TELEMETRY_ONLY"
                    else:
                        status = existing_row["status"]
                    created_at = existing_row["created_at"]
                else:
                    if custom_host_id:
                        host_id = custom_host_id
                    else:
                        # Deterministic canonical ID from normalized hostname
                        clean_host = re.sub(r"[^a-zA-Z0-9]", "_", h_name.lower()).strip("_")
                        if not clean_host:
                            clean_host = "node"
                        candidate_id = f"host_{clean_host}"
                        cursor.execute("SELECT host_id FROM hosts WHERE host_id = ?", (candidate_id,))
                        if cursor.fetchone():
                            clean_os = re.sub(r"[^a-zA-Z0-9]", "_", os_name.lower()).strip("_")
                            candidate_id = f"host_{clean_host}_{clean_os}"
                        host_id = candidate_id

                    agent_id = f"agent_{secrets.token_hex(6)}"
                    status = "ONLINE" if is_windows else "TELEMETRY_ONLY"
                    created_at = now_iso

                raw_token = f"fgead_{secrets.token_urlsafe(32)}"
                token_hash = self._hash_token(raw_token)

                # Assign model and initial status based on OS compatibility and host profile
                if is_windows:
                    if host_id == "host_sivachowdary" or h_name.lower() == "sivachowdary":
                        assigned_model_id = "windows_sivachowdary_v2"
                    elif model_id and model_id not in ("none", "windows_default"):
                        assigned_model_id = model_id
                    else:
                        assigned_model_id = "windows_default"
                    model_status = "COMPATIBLE"
                else:
                    assigned_model_id = "none"
                    model_status = "BASELINE_REQUIRED"

                cursor.execute(
                    """
                    INSERT INTO hosts (
                        host_id, hostname, operating_system, os_version, architecture,
                        agent_version, schema_version, agent_id, token_hash,
                        status, model_id, model_status, is_enabled, created_at, last_seen, last_seen_epoch,
                        machine_info_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(host_id) DO UPDATE SET
                        hostname=excluded.hostname,
                        operating_system=excluded.operating_system,
                        os_version=excluded.os_version,
                        architecture=excluded.architecture,
                        agent_version=excluded.agent_version,
                        schema_version=excluded.schema_version,
                        agent_id=excluded.agent_id,
                        token_hash=excluded.token_hash,
                        status=excluded.status,
                        model_id=excluded.model_id,
                        model_status=excluded.model_status,
                        last_seen=excluded.last_seen,
                        last_seen_epoch=excluded.last_seen_epoch,
                        machine_info_json=excluded.machine_info_json
                    """,
                    (
                        host_id,
                        h_name,
                        os_name,
                        os_version,
                        architecture,
                        agent_version,
                        schema_version,
                        agent_id,
                        token_hash,
                        status,
                        assigned_model_id,
                        model_status,
                        created_at,
                        now_iso,
                        now_epoch,
                        info_json,
                    ),
                )
                conn.commit()

            host_dict = self.get_host(host_id)
            return host_dict, raw_token

    def authenticate_agent(self, host_id: str, token: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Authenticate an agent request.
        Returns: (is_authenticated, error_reason, host_dict)
        """
        if not host_id or not token:
            return False, "Missing host ID or authentication token", None

        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hosts WHERE host_id = ?", (host_id,))
            row = cursor.fetchone()
            if not row:
                return False, f"Unknown host ID '{host_id}'", None

            host_dict = dict(row)
            if not host_dict.get("is_enabled", 1):
                return False, f"Host '{host_id}' is disabled", None

            input_hash = self._hash_token(token)
            if not secrets.compare_digest(input_hash, host_dict["token_hash"]):
                return False, "Invalid agent authentication token", None

            return True, None, host_dict

    def update_heartbeat(self, host_id: str, status: Optional[str] = None) -> None:
        """Update last_seen timestamp and epoch for a host."""
        now_iso = datetime.now(timezone.utc).isoformat()
        now_epoch = time.time()
        new_status = status or "ONLINE"
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE hosts
                SET last_seen = ?, last_seen_epoch = ?, status = ?
                WHERE host_id = ?
                """,
                (now_iso, now_epoch, new_status, host_id),
            )
            conn.commit()

    def set_host_status(self, host_id: str, status: str) -> None:
        """Explicitly set host status (e.g. ONLINE, OFFLINE, ANOMALY)."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE hosts SET status = ? WHERE host_id = ?", (status, host_id))
            conn.commit()

    def get_host(self, host_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve host details by host_id."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hosts WHERE host_id = ?", (host_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            # Remove secret hash
            res.pop("token_hash", None)
            if res.get("machine_info_json"):
                try:
                    res["machine_info"] = json.loads(res["machine_info_json"])
                except Exception:
                    res["machine_info"] = {}
            # Check offline status dynamically
            last_epoch = res.get("last_seen_epoch") or 0.0
            if (time.time() - last_epoch) > self.offline_timeout_sec and res.get("status") != "OFFLINE":
                res["status"] = "OFFLINE"
            return res

    def get_all_hosts(self) -> List[Dict[str, Any]]:
        """Retrieve all registered hosts with dynamically checked online/offline status."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hosts ORDER BY created_at ASC")
            rows = cursor.fetchall()
            hosts = []
            now_epoch = time.time()
            for r in rows:
                h = dict(r)
                h.pop("token_hash", None)
                if h.get("machine_info_json"):
                    try:
                        h["machine_info"] = json.loads(h["machine_info_json"])
                    except Exception:
                        h["machine_info"] = {}
                last_epoch = h.get("last_seen_epoch") or 0.0
                if (now_epoch - last_epoch) > self.offline_timeout_sec:
                    h["status"] = "OFFLINE"
                hosts.append(h)
            return hosts

    def record_alert(
        self,
        host_id: str,
        episode_id: int,
        severity: str,
        anomaly_score: float,
        threshold: float,
        dominant_feature: str,
        message: str,
    ) -> Dict[str, Any]:
        """Record an anomaly alert in persistent storage."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts (
                    host_id, episode_id, severity, anomaly_score,
                    threshold, dominant_feature, message, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    host_id,
                    episode_id,
                    severity,
                    anomaly_score,
                    threshold,
                    dominant_feature,
                    message,
                    now_iso,
                ),
            )
            conn.commit()
            alert_id = cursor.lastrowid
            return {
                "alert_id": alert_id,
                "host_id": host_id,
                "episode_id": episode_id,
                "severity": severity,
                "anomaly_score": round(anomaly_score, 4),
                "threshold": round(threshold, 4),
                "dominant_feature": dominant_feature,
                "message": message,
                "timestamp": now_iso,
            }

    def get_alerts(self, host_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent alerts."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if host_id:
                cursor.execute(
                    "SELECT * FROM alerts WHERE host_id = ? ORDER BY alert_id DESC LIMIT ?",
                    (host_id, limit),
                )
            else:
                cursor.execute("SELECT * FROM alerts ORDER BY alert_id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def register_baseline(
        self,
        host_id: str,
        model_id: str,
        threshold: float,
        sample_count: int,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Register a telemetry baseline for a host."""
        baseline_id = f"base_{secrets.token_hex(4)}"
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO baselines (
                    baseline_id, host_id, model_id, threshold, sample_count, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (baseline_id, host_id, model_id, threshold, sample_count, now_iso, notes),
            )
            cursor.execute("UPDATE hosts SET baseline_id = ? WHERE host_id = ?", (baseline_id, host_id))
            conn.commit()
            return {
                "baseline_id": baseline_id,
                "host_id": host_id,
                "model_id": model_id,
                "threshold": threshold,
                "sample_count": sample_count,
                "created_at": now_iso,
                "notes": notes,
            }

    def set_host_model_status(self, host_id: str, model_id: str, model_status: str) -> None:
        """Update model assignment and compatibility status for a host."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE hosts SET model_id = ?, model_status = ? WHERE host_id = ?",
                (model_id, model_status, host_id),
            )
            conn.commit()

    def save_host_baseline_stats(self, host_id: str, stats: Dict[str, Any]) -> None:
        """Save computed feature distribution statistics for a host."""
        stats_json = json.dumps(stats)
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE hosts SET baseline_stats_json = ? WHERE host_id = ?",
                (stats_json, host_id),
            )
            conn.commit()

    def get_host_baseline_stats(self, host_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve computed baseline distribution statistics for a host."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT baseline_stats_json FROM hosts WHERE host_id = ?", (host_id,))
            row = cursor.fetchone()
            if not row or not row["baseline_stats_json"]:
                return None
            try:
                return json.loads(row["baseline_stats_json"])
            except Exception:
                return None

    def reconcile_duplicate_hosts(self, target_hostname: Optional[str] = None) -> List[str]:
        """
        Identify and clean up duplicate host records for the same hostname.
        Keeps the latest active host record and removes stale historical duplicates.
        Returns list of cleaned host IDs.
        """
        cleaned = []
        with self._lock, self._get_connection() as conn:
            cursor = conn.cursor()
            if target_hostname:
                cursor.execute(
                    "SELECT host_id, hostname, operating_system, last_seen_epoch FROM hosts WHERE hostname = ? ORDER BY last_seen_epoch DESC",
                    (target_hostname,),
                )
            else:
                cursor.execute("SELECT host_id, hostname, operating_system, last_seen_epoch FROM hosts ORDER BY last_seen_epoch DESC")

            rows = cursor.fetchall()
            seen_keys = set()
            for r in rows:
                key = (r["hostname"].strip().lower(), r["operating_system"].strip().lower())
                if key in seen_keys:
                    # Duplicate found - disable/remove stale duplicate
                    stale_id = r["host_id"]
                    cursor.execute("DELETE FROM alerts WHERE host_id = ?", (stale_id,))
                    cursor.execute("DELETE FROM baselines WHERE host_id = ?", (stale_id,))
                    cursor.execute("DELETE FROM hosts WHERE host_id = ?", (stale_id,))
                    cleaned.append(stale_id)
                else:
                    seen_keys.add(key)
            conn.commit()
        return cleaned
_GLOBAL_REGISTRY: Optional[HostRegistry] = None
_REGISTRY_LOCK = threading.RLock()


def get_host_registry(db_path: Optional[Path | str] = None, force_new: bool = False) -> HostRegistry:
    """
    Retrieve the active thread-safe HostRegistry singleton.
    Honors FGEAD_DB_PATH environment variable and supports dynamic test database switching.
    """
    global _GLOBAL_REGISTRY
    with _REGISTRY_LOCK:
        if db_path is not None:
            return HostRegistry(db_path=db_path)

        current_env_db = os.getenv("FGEAD_DB_PATH")
        target_db = Path(current_env_db) if current_env_db else DEFAULT_DB_PATH

        if _GLOBAL_REGISTRY is None or force_new or _GLOBAL_REGISTRY.db_path != target_db:
            _GLOBAL_REGISTRY = HostRegistry(db_path=target_db)
        return _GLOBAL_REGISTRY


def reset_host_registry(db_path: Optional[Path | str] = None) -> HostRegistry:
    """Reset the global registry singleton to point to a fresh or specific database path."""
    global _GLOBAL_REGISTRY
    with _REGISTRY_LOCK:
        _GLOBAL_REGISTRY = None
        return get_host_registry(db_path=db_path)
