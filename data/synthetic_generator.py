"""
data/synthetic_generator.py
Generates realistic multivariate time series with injected anomalies.
Run:  python data/synthetic_generator.py
"""

import numpy as np
import pandas as pd
import os
import sys

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

FEATURE_NAMES = [
    "cpu_usage", "cpu_temp", "memory_usage", "memory_free",
    "disk_io_read", "disk_io_write", "net_in", "net_out",
    "process_count", "context_switches", "cache_hits",
    "cache_misses", "load_avg_1m", "load_avg_5m", "load_avg_15m",
    "swap_usage", "iowait", "kernel_threads", "open_files", "network_errors",
]


def generate_synthetic_data(
    n_timesteps: int = 5000,
    n_features: int = 20,
    anomaly_prob: float = 0.005,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates realistic server-monitoring multivariate time series.

    Normal patterns include:
      - Diurnal trend (day/night cycle)
      - Correlated feature pairs (cpu↔cpu_temp, disk_read↔disk_write, etc.)
      - Low-frequency noise

    Anomaly types injected:
      - Spike: one feature jumps 4-6σ above normal
      - Correlation break: two normally-correlated features decouple
      - Gradual drift: feature slowly creeps outside normal range
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_timesteps)

    # ── Base signals with trend + seasonality ──────────────────────────────
    data = rng.standard_normal((n_timesteps, n_features)) * 0.3

    # Diurnal cycles
    day_cycle = np.sin(2 * np.pi * t / 1440)   # 1440-step "day"
    week_cycle = np.sin(2 * np.pi * t / 10080)  # week trend

    # Apply realistic baselines per feature
    baselines = np.linspace(0.2, 0.8, n_features)
    for i in range(n_features):
        data[:, i] += baselines[i] + 0.15 * day_cycle + 0.05 * week_cycle

    # ── Inject feature correlations (mimics real server metrics) ───────────
    # cpu_usage (0) ↔ cpu_temp (1)
    data[:, 1] = 0.85 * data[:, 0] + 0.15 * rng.standard_normal(n_timesteps)
    # memory_usage (2) ↔ memory_free (3) — inverse
    data[:, 3] = -0.90 * data[:, 2] + 0.10 * rng.standard_normal(n_timesteps)
    # disk_io_read (4) ↔ disk_io_write (5)
    data[:, 5] = 0.75 * data[:, 4] + 0.25 * rng.standard_normal(n_timesteps)
    # net_in (6) ↔ net_out (7)
    data[:, 7] = 0.80 * data[:, 6] + 0.20 * rng.standard_normal(n_timesteps)
    # load_avg_1m (12) ↔ 5m (13) ↔ 15m (14) — cascading averages
    data[:, 13] = 0.70 * data[:, 12] + 0.30 * rng.standard_normal(n_timesteps)
    data[:, 14] = 0.70 * data[:, 13] + 0.30 * rng.standard_normal(n_timesteps)

    # ── Inject anomalies ───────────────────────────────────────────────────
    n_anomalies =  20 #int(n_timesteps * anomaly_prob)
    # Space them out so we have distinct events
    # Force some anomalies into the test region

    # Match train/val/test split used in evaluate.py
    train_end = int(n_timesteps * 0.70)
    val_end   = int(n_timesteps * 0.85)

    # Distribute anomaly events across all splits
    train_events = int(n_anomalies * 0.70)
    val_events   = int(n_anomalies * 0.15)
    test_events  = n_anomalies - train_events - val_events

    train_indices = rng.choice(
        np.arange(100, train_end),
        size=train_events,
        replace=False
    )

    val_indices = rng.choice(
        np.arange(train_end, val_end),
        size=val_events,
        replace=False
    )

    test_indices = rng.choice(
        np.arange(val_end, n_timesteps - 100),
        size=test_events,
        replace=False
    )

    anomaly_indices = np.concatenate([
        train_indices,
        val_indices,
        test_indices
    ])
    
    labels = np.zeros(n_timesteps, dtype=int)

    for idx in sorted(anomaly_indices):
        anomaly_type = rng.integers(3)

        if anomaly_type == 0:
            # Spike: one or two features spike suddenly
            feat = rng.integers(n_features)
            spike_len = rng.integers(2, 5)
            data[idx:idx+spike_len, feat] += rng.uniform(4, 6)
            labels[idx:idx+spike_len] = 1

        elif anomaly_type == 1:
            # Correlation break: cpu ↔ cpu_temp decouple
            break_len = rng.integers(5, 12)
            data[idx:idx+break_len, 1] = rng.standard_normal(break_len) * 2
            labels[idx:idx+break_len] = 1

        else:
            # Gradual drift on memory
            drift_len = rng.integers(8, 15)
            drift = np.linspace(0, rng.uniform(3, 5), drift_len)
            data[idx:idx+drift_len, 2] += drift
            labels[idx:idx+drift_len] = 1

    # ── Save ───────────────────────────────────────────────────────────────
    names = FEATURE_NAMES if n_features == 20 else [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=names[:n_features])
    df["label"] = labels

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_data.csv", index=False)
    print(f"[✓] Saved data/synthetic_data.csv — {n_timesteps} timesteps, "
          f"{n_features} features, {labels.sum()} anomaly timesteps "
          f"({labels.mean():.1%})")
    return data, labels


if __name__ == "__main__":
    generate_synthetic_data()
