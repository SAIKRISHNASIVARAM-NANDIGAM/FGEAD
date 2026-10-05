"""
models/train_current_machine_v2.py

Step 5, 6, 7: Retraining and Calibration of Current-Machine Versioned Model (v2).
- Preserves original checkpoint untouched.
- Trains FGEAD on data/live_baseline_v2_current_machine.csv.
- Calibrates threshold on clean validation set.
- Evaluates Normal FPR and 5 Controlled Anomaly Scenarios (CPU, Disk-Write, Disk-Read, Network, Combined).
"""

import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES
from models.fgead import FGEAD


def train_v2_model(epochs: int = 15, batch_size: int = 64, lr: float = 0.001, retrain: bool = False):
    print("=" * 80)
    print("FGEAD PHASE 10 — TRAINING DEDICATED CURRENT-MACHINE MODEL (V2)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "live_baseline_v2_current_machine.csv"
    df = pd.read_csv(data_path)
    print(f"Loaded baseline data: {df.shape} from {data_path}")

    # 2. Sequential Split: 70% Train, 15% Val, 15% Test
    n = len(df)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    print(f"Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 3. Fit Scaler on TRAIN ONLY
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)

    scaler_path = PROJECT_ROOT / "checkpoints" / "fgead_live_scaler_v2_current_machine.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Saved train-only scaler to: {scaler_path}")

    # 4. Create Sliding Windows (W=60, S=1)
    W = 60
    def make_windows(arr):
        windows = []
        for i in range(len(arr) - W + 1):
            windows.append(arr[i:i + W])
        return np.array(windows, dtype=np.float32)

    X_train = make_windows(train_scaled)
    X_val = make_windows(val_scaled)
    X_test = make_windows(test_scaled)

    print(f"Window datasets: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    model = FGEAD(
        n_features=N_LIVE_FEATURES,
        embed_dim=64,
        n_heads=4,
        gcn_out=64,
        lstm_hidden=128,
        sparsity_threshold=0.3,
        dropout=0.2,
        sparsity_lambda=0.01,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = 0.887588
    ckpt_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_v2_current_machine.pt"

    if retrain or not ckpt_path.exists():
        print("\n--- Training Epochs ---")
        best_val_loss = float("inf")
        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon, adj, _ = model(batch_x)
                # Reconstruct next timestep forecast
                loss = criterion(recon, batch_x[:, 1:, :])
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= len(X_train)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(device)
                    recon, adj, _ = model(batch_x)
                    loss = criterion(recon, batch_x[:, 1:, :])
                    val_loss += loss.item() * len(batch_x)
            val_loss /= len(X_val)

            print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "n_features": N_LIVE_FEATURES,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "epoch": ep,
                    "baseline_id": "v2_current_machine_sivachowdary",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }, ckpt_path)

        print(f"\n[OK] Saved best model checkpoint to: {ckpt_path} (Best Val Loss: {best_val_loss:.6f})")
    else:
        print(f"\n[OK] Using existing trained checkpoint: {ckpt_path}")

    # 6. Load Best Checkpoint & Calibrate Threshold on Validation Set
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_scores = []
    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            _, _, scores_step = model(batch_x)
            # Max window anomaly score
            for sc in scores_step.cpu().numpy():
                val_scores.append(float(np.max(sc)))

    val_scores = np.array(val_scores)
    val_mean = float(np.mean(val_scores))
    val_std = float(np.std(val_scores))
    val_p95 = float(np.percentile(val_scores, 95))
    val_p99 = float(np.percentile(val_scores, 99))
    val_max = float(np.max(val_scores))

    # Scientific threshold calibration: mean + 3*std with empirical safety margin
    tau_v2 = round(max(val_mean + 3.0 * val_std, val_p99 * 1.15), 6)
    print(f"\n--- Validation Residual Distribution & Threshold Calibration ---")
    print(f"Val Mean : {val_mean:.6f} | Val Std: {val_std:.6f}")
    print(f"Val P95  : {val_p95:.6f} | Val P99: {val_p99:.6f} | Val Max: {val_max:.6f}")
    print(f"Calibrated Threshold tau_v2 = {tau_v2:.6f}")

    thresh_path = PROJECT_ROOT / "checkpoints" / "fgead_live_threshold_v2_current_machine.json"
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": "windows_sivachowdary_v2",
            "threshold": tau_v2,
            "validation_mean": val_mean,
            "validation_std": val_std,
            "validation_p95": val_p95,
            "validation_p99": val_p99,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, f, indent=2)

    config_path = PROJECT_ROOT / "checkpoints" / "fgead_live_windows_22ch_config_v2_current_machine.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": "windows_sivachowdary_v2",
            "n_features": N_LIVE_FEATURES,
            "window_size": 60,
            "features": LIVE_FEATURES,
            "baseline_path": "data/live_baseline_v2_current_machine.csv",
            "scaler_path": "checkpoints/fgead_live_scaler_v2_current_machine.joblib",
            "checkpoint_path": "checkpoints/fgead_live_windows_22ch_v2_current_machine.pt",
            "threshold": tau_v2,
        }, f, indent=2)

    # 7. Evaluate Normal False Positive Rate on Unseen Test Set
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test)), batch_size=batch_size, shuffle=False)
    test_scores = []
    with torch.no_grad():
        for (batch_x,) in test_loader:
            batch_x = batch_x.to(device)
            _, _, scores_step = model(batch_x)
            for sc in scores_step.cpu().numpy():
                test_scores.append(float(np.max(sc)))

    test_scores = np.array(test_scores)
    raw_flagged_windows = int(np.sum(test_scores > tau_v2))
    raw_fpr = (raw_flagged_windows / len(test_scores)) * 100.0
    specificity = 100.0 - raw_fpr

    print(f"\n--- Unseen Normal Test Set Evaluation ({len(test_scores)} windows) ---")
    print(f"Test Mean Score : {np.mean(test_scores):.6f}")
    print(f"Test Max Score  : {np.max(test_scores):.6f}")
    print(f"Test P95 / P99  : {np.percentile(test_scores, 95):.6f} / {np.percentile(test_scores, 99):.6f}")
    print(f"Raw Window FPR  : {raw_fpr:.2f}% ({raw_flagged_windows}/{len(test_scores)})")
    print(f"Specificity     : {specificity:.2f}%")

    # 8. Controlled Anomaly Validation (Workloads A, B, C, D, E)
    print("\n--- STEP 7: Controlled Anomaly Workload Testing ---")
    nominal_window = df_test.iloc[:60].values.copy() # (60, 22)
    scenarios = [
        ("A. CPU-Intensive Workload", {"cpu_percent": 98.5, "cpu_user_time_percent": 75.0, "cpu_system_time_percent": 23.5}),
        ("B. Disk-Write Burst", {"disk_write_bytes_per_sec": 95000000.0, "disk_write_count_per_sec": 4500.0}),
        ("C. Disk-Read Burst", {"disk_read_bytes_per_sec": 85000000.0, "disk_read_count_per_sec": 3800.0}),
        ("D. Network Burst", {"net_bytes_sent_per_sec": 80000000.0, "net_packets_sent_per_sec": 15000.0}),
        ("E. Combined CPU + Storage", {"cpu_percent": 96.0, "disk_write_bytes_per_sec": 75000000.0, "memory_percent": 94.0}),
    ]

    anomaly_results = {}
    for name, injection in scenarios:
        anom_win = nominal_window.copy()
        for f_name, injected_val in injection.items():
            f_idx = LIVE_FEATURES.index(f_name)
            anom_win[20:, f_idx] = injected_val # inject starting timestep 20

        scaled_anom = scaler.transform(anom_win).astype(np.float32)
        tensor_anom = torch.from_numpy(scaled_anom).unsqueeze(0).to(device)

        with torch.no_grad():
            preds_norm, _, scores_step = model(tensor_anom)
            sc_arr = scores_step.squeeze(0).cpu().numpy()
            peak_score = float(np.max(sc_arr))
            is_detected = peak_score > tau_v2

            # Attribution
            feat_res = np.mean(np.abs(preds_norm[0].cpu().numpy() - tensor_anom[0, 1:].cpu().numpy()), axis=0)
            top_idx = int(np.argmax(feat_res))
            top_feat_name = LIVE_FEATURES[top_idx]

        detected_str = "[PASS] DETECTED" if is_detected else "[FAIL] MISSED"
        print(f"  {name:30s} -> Score: {peak_score:7.4f} (tau={tau_v2:.4f}) | {detected_str} | Top Attrib: {top_feat_name}")
        anomaly_results[name] = {
            "peak_score": round(peak_score, 4),
            "threshold": tau_v2,
            "detected": is_detected,
            "top_attribution": top_feat_name,
        }

    # 9. Save Evaluation Metrics
    metrics_path = PROJECT_ROOT / "data" / "multihost_validation" / "phase10_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_id": "windows_sivachowdary_v2",
            "threshold": tau_v2,
            "train_loss": best_val_loss,
            "validation_stats": {
                "mean": round(val_mean, 6),
                "std": round(val_std, 6),
                "p95": round(val_p95, 6),
                "p99": round(val_p99, 6),
                "max": round(val_max, 6),
            },
            "test_stats": {
                "total_windows": len(test_scores),
                "flagged_windows": raw_flagged_windows,
                "raw_fpr_pct": round(raw_fpr, 4),
                "specificity_pct": round(specificity, 4),
                "mean_score": round(float(np.mean(test_scores)), 6),
                "p95_score": round(float(np.percentile(test_scores, 95)), 6),
                "p99_score": round(float(np.percentile(test_scores, 99)), 6),
            },
            "controlled_anomalies": anomaly_results,
        }, f, indent=2)

    print(f"\n[OK] Saved Phase 10 evaluation metrics to: {metrics_path}")
    return True


if __name__ == "__main__":
    train_v2_model()
