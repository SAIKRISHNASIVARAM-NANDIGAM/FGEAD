"""
data/live_agent.py

FGEAD Real-Time Physical Host Monitoring Agent.
Collects actual Windows telemetry counters every 1 second and transmits
structured feature vectors to the FGEAD FastAPI backend.
Supports both direct legacy /telemetry endpoint and multi-host /hosts/{host_id}/telemetry endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import psutil
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_feature_schema import (
    LIVE_FEATURES,
    LIVE_FEATURE_VERSION,
    validate_live_feature_dict,
)

DEFAULT_API_URL = os.getenv("FGEAD_API_URL", "http://127.0.0.1:8000")
CONFIG_PATH = PROJECT_ROOT / "data" / "live_agent_config.json"


class LiveTelemetryCollector:
    """
    Stateful hardware metrics collector for Windows systems using psutil.
    Computes accurate per-second differential rates for I/O and network counters.
    """

    def __init__(self):
        self.hostname = socket.gethostname()
        self.platform_name = platform.system()
        self.os_version = platform.version()
        self.architecture = platform.machine()
        self.python_version = platform.python_version()
        self.cpu_count_logical = psutil.cpu_count(logical=True) or 1
        self.cpu_count_physical = psutil.cpu_count(logical=False) or 1

        # Prime initial state for rate calculations
        self._last_time = time.time()
        self._last_disk_io = psutil.disk_io_counters()
        self._last_net_io = psutil.net_io_counters()
        self._last_cpu_stats = psutil.cpu_stats() if hasattr(psutil, "cpu_stats") else None

        # Prime CPU percent measurement
        psutil.cpu_percent(interval=None)

    def get_machine_info(self) -> Dict[str, Any]:
        """Return static hardware and environment metadata."""
        return {
            "machine_id": self.hostname,
            "hostname": self.hostname,
            "platform": self.platform_name,
            "operating_system": self.platform_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "cpu_logical_cores": self.cpu_count_logical,
            "cpu_physical_cores": self.cpu_count_physical,
            "python_version": self.python_version,
            "schema_version": LIVE_FEATURE_VERSION,
            "agent_version": "1.0.0",
        }

    def collect_features(self) -> Dict[str, float]:
        """
        Sample system hardware counters and compute one instantaneous telemetry snapshot.
        """
        now = time.time()
        dt = max(now - self._last_time, 0.001)

        # 1. CPU Metrics
        cpu_pct = float(psutil.cpu_percent(interval=None))
        cpu_times_pct = psutil.cpu_times_percent(interval=None)
        cpu_user = float(getattr(cpu_times_pct, "user", 0.0))
        cpu_system = float(getattr(cpu_times_pct, "system", 0.0))

        # CPU Frequency
        freq_info = psutil.cpu_freq()
        cpu_freq = float(freq_info.current) if freq_info and freq_info.current else 0.0

        # CPU Stats (Context switches & interrupts)
        curr_cpu_stats = psutil.cpu_stats() if hasattr(psutil, "cpu_stats") else None
        if curr_cpu_stats and self._last_cpu_stats:
            ctx_switches_rate = float(curr_cpu_stats.ctx_switches - self._last_cpu_stats.ctx_switches) / dt
            interrupts_rate = float(curr_cpu_stats.interrupts - self._last_cpu_stats.interrupts) / dt
        else:
            ctx_switches_rate = 0.0
            interrupts_rate = 0.0
        self._last_cpu_stats = curr_cpu_stats

        # 2. Memory & Swap Metrics
        vm = psutil.virtual_memory()
        mem_pct = float(vm.percent)
        mem_avail_mb = float(vm.available) / (1024.0 * 1024.0)
        mem_used_mb = float(vm.used) / (1024.0 * 1024.0)

        swap = psutil.swap_memory()
        swap_pct = float(swap.percent)

        # 3. Disk Metrics
        try:
            disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
            disk_pct = float(disk_usage.percent)
        except Exception:
            disk_pct = 0.0

        curr_disk_io = psutil.disk_io_counters()
        if curr_disk_io and self._last_disk_io:
            disk_read_bytes_sec = float(curr_disk_io.read_bytes - self._last_disk_io.read_bytes) / dt
            disk_write_bytes_sec = float(curr_disk_io.write_bytes - self._last_disk_io.write_bytes) / dt
            disk_read_count_sec = float(curr_disk_io.read_count - self._last_disk_io.read_count) / dt
            disk_write_count_sec = float(curr_disk_io.write_count - self._last_disk_io.write_count) / dt
        else:
            disk_read_bytes_sec = 0.0
            disk_write_bytes_sec = 0.0
            disk_read_count_sec = 0.0
            disk_write_count_sec = 0.0
        self._last_disk_io = curr_disk_io

        # 4. Network Metrics
        curr_net_io = psutil.net_io_counters()
        if curr_net_io and self._last_net_io:
            net_sent_sec = float(curr_net_io.bytes_sent - self._last_net_io.bytes_sent) / dt
            net_recv_sec = float(curr_net_io.bytes_recv - self._last_net_io.bytes_recv) / dt
            net_pkts_sent_sec = float(curr_net_io.packets_sent - self._last_net_io.packets_sent) / dt
            net_pkts_recv_sec = float(curr_net_io.packets_recv - self._last_net_io.packets_recv) / dt
            net_errs = float(curr_net_io.errin + curr_net_io.errout)
            net_drops = float(curr_net_io.dropin + curr_net_io.dropout)
        else:
            net_sent_sec = 0.0
            net_recv_sec = 0.0
            net_pkts_sent_sec = 0.0
            net_pkts_recv_sec = 0.0
            net_errs = 0.0
            net_drops = 0.0
        self._last_net_io = curr_net_io

        # 5. Process Count
        try:
            proc_count = float(len(psutil.pids()))
        except Exception:
            proc_count = 0.0

        self._last_time = now

        features: Dict[str, float] = {
            "cpu_percent": round(max(0.0, min(100.0, cpu_pct)), 2),
            "cpu_freq_current": round(cpu_freq, 2),
            "cpu_user_time_percent": round(max(0.0, min(100.0, cpu_user)), 2),
            "cpu_system_time_percent": round(max(0.0, min(100.0, cpu_system)), 2),
            "cpu_ctx_switches_per_sec": round(max(0.0, ctx_switches_rate), 1),
            "cpu_interrupts_per_sec": round(max(0.0, interrupts_rate), 1),
            "memory_percent": round(max(0.0, min(100.0, mem_pct)), 2),
            "memory_available_mb": round(max(0.0, mem_avail_mb), 1),
            "memory_used_mb": round(max(0.0, mem_used_mb), 1),
            "swap_percent": round(max(0.0, min(100.0, swap_pct)), 2),
            "disk_usage_percent": round(max(0.0, min(100.0, disk_pct)), 2),
            "disk_read_bytes_per_sec": round(max(0.0, disk_read_bytes_sec), 1),
            "disk_write_bytes_per_sec": round(max(0.0, disk_write_bytes_sec), 1),
            "disk_read_count_per_sec": round(max(0.0, disk_read_count_sec), 1),
            "disk_write_count_per_sec": round(max(0.0, disk_write_count_sec), 1),
            "net_bytes_sent_per_sec": round(max(0.0, net_sent_sec), 1),
            "net_bytes_recv_per_sec": round(max(0.0, net_recv_sec), 1),
            "net_packets_sent_per_sec": round(max(0.0, net_pkts_sent_sec), 1),
            "net_packets_recv_per_sec": round(max(0.0, net_pkts_recv_sec), 1),
            "net_errors_total": round(max(0.0, net_errs), 1),
            "net_drops_total": round(max(0.0, net_drops), 1),
            "process_count": round(max(0.0, proc_count), 0),
        }

        # Validation
        is_valid, msg = validate_live_feature_dict(features)
        if not is_valid:
            raise ValueError(f"Telemetry feature validation failed: {msg}")

        return features


def load_or_register_agent(
    api_url: str,
    collector: LiveTelemetryCollector,
    custom_host_id: Optional[str] = None,
) -> Tuple[str, str]:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("host_id") and data.get("agent_token"):
                    return data["host_id"], data["agent_token"]
        except Exception:
            pass

    info = collector.get_machine_info()
    reg_payload = {
        "hostname": info["hostname"],
        "operating_system": "Windows",
        "os_version": info["os_version"],
        "architecture": info["architecture"],
        "agent_version": info["agent_version"],
        "schema_version": info["schema_version"],
        "custom_host_id": custom_host_id or f"host_{socket.gethostname().lower()}",
        "machine_info": info,
    }

    resp = requests.post(f"{api_url.rstrip('/')}/hosts/register", json=reg_payload, timeout=5.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Host registration failed: {resp.status_code} {resp.text}")

    reg_data = resp.json()
    host_id = reg_data["host_id"]
    token = reg_data["agent_token"]

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"host_id": host_id, "agent_token": token, "api_url": api_url}, f, indent=2)

    return host_id, token


def run_agent(api_url: str = DEFAULT_API_URL, interval_sec: float = 1.0, max_iterations: Optional[int] = None, use_multihost: bool = True):
    """
    Main loop: collects telemetry every interval_sec and transmits to backend.
    """
    collector = LiveTelemetryCollector()
    info = collector.get_machine_info()

    print("=" * 70)
    print("FGEAD REAL-TIME WINDOWS TELEMETRY AGENT")
    print("=" * 70)
    print(f"Machine ID   : {info['machine_id']}")
    print(f"OS Platform  : {info['platform']} {info['os_version']} ({info['architecture']})")
    print(f"Logical Cores: {info['cpu_logical_cores']}")
    print(f"Target API   : {api_url}")
    print(f"Sampling     : Every {interval_sec}s")
    print(f"Schema Ver   : {info['schema_version']} ({len(LIVE_FEATURES)} features)")
    print("=" * 70)

    host_id = None
    token = None
    headers = {}

    if use_multihost:
        try:
            host_id, token = load_or_register_agent(api_url=api_url, collector=collector)
            endpoint = f"{api_url.rstrip('/')}/hosts/{host_id}/telemetry"
            headers = {"X-Agent-Token": token, "Content-Type": "application/json"}
            print(f"[AUTH] Multi-Host Registered: host_id = {host_id}")
        except Exception as e:
            print(f"[AUTH WARNING] Could not register multi-host: {e}. Falling back to legacy /telemetry.")
            endpoint = f"{api_url.rstrip('/')}/telemetry"
    else:
        endpoint = f"{api_url.rstrip('/')}/telemetry"

    # Allow 1 second for initial differential priming
    time.sleep(1.0)

    iteration = 0
    consecutive_failures = 0

    while True:
        iteration += 1
        t_start = time.time()

        try:
            features = collector.collect_features()
            payload = {
                "machine_id": info["machine_id"],
                "host_id": host_id or info["machine_id"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "machine_info": info,
                "features": features,
            }

            resp = requests.post(endpoint, json=payload, headers=headers, timeout=2.0)
            if resp.status_code == 200:
                consecutive_failures = 0
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Sample #{iteration:04d} -> POST OK | "
                    f"CPU: {features['cpu_percent']:5.1f}% | "
                    f"RAM: {features['memory_percent']:5.1f}% ({features['memory_used_mb']:6.0f}MB) | "
                    f"Disk: {features['disk_usage_percent']:4.1f}% | "
                    f"Net: {(features['net_bytes_recv_per_sec'] + features['net_bytes_sent_per_sec'])/1024.0:6.1f} KB/s"
                )
            else:
                consecutive_failures += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Server returned status {resp.status_code}: {resp.text}")

        except requests.exceptions.RequestException as e:
            consecutive_failures += 1
            if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for API at {endpoint}... ({e.__class__.__name__})")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Collection error: {e}")

        if max_iterations is not None and iteration >= max_iterations:
            print(f"Agent reached maximum iterations ({max_iterations}). Exiting cleanly.")
            break

        elapsed = time.time() - t_start
        sleep_time = max(0.05, interval_sec - elapsed)
        time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGEAD Real-Time Host Monitoring Agent")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="FastAPI backend base URL")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Collect and send a single sample then exit")
    parser.add_argument("--count", type=int, default=None, help="Number of samples to collect before exiting")
    parser.add_argument("--legacy", action="store_true", help="Use legacy unauthenticated /telemetry endpoint")
    args = parser.parse_args()

    iters = 1 if args.once else args.count
    run_agent(api_url=args.api_url, interval_sec=args.interval, max_iterations=iters, use_multihost=not args.legacy)
