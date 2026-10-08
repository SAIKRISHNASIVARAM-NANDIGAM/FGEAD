"""
data/live_training/evaluate_v3_heldout.py

Phase 6: Unseen Normal Validation on Held-Out Test Split (v3_test_normal.csv).
Verifies that V3 model FPR <= 1.0% and Specificity >= 99.0% on never-before-seen normal data.
Saves evaluation metrics to data/live_training/v3_heldout_eval_report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.live_feature_schema import LIVE_FEATURES, N_LIVE_FEATURES, prepare_model_input_window
from models.fgead import FGEAD
from data.live_training.train_v3_model import create_sliding_windows


def evaluate_v3_heldout(
    test_csv_path: str = "data/live_training/v3_test_normal.csv",
    checkpoint_path: str = "checkpoints/fgead_live_windows_22ch_v3_current_machine.pt",
    scaler_path: str = "checkpoints/fgead_live_scaler_v3_current_machine.joblib",
    threshold_path: str = "checkpoints/fgead_live_threshold_v3_current_machine.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 6: UNSEEN NORMAL VALIDATION (HELD-OUT TEST SET)")
    print("=" * 70)

    test_file = PROJECT_ROOT / test_csv_path
    ckpt_file = PROJECT_ROOT / checkpoint_path
    scaler_file = PROJECT_ROOT / scaler_path
    thresh_file = PROJECT_ROOT / threshold_path

    for p, name in [(test_file, "Test CSV"), (ckpt_file, "Model Checkpoint"), (scaler_file, "Scaler"), (thresh_file, "Threshold JSON")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at: {p}")

    df_test = pd.read_csv(test_file)[LIVE_FEATURES]
    scaler = joblib.load(scaler_file)

    with open(thresh_file, "r", encoding="utf-8") as f:
        t_data = json.load(f)
    threshold = float(t_data["threshold"])

    # Transform held-out test data
    test_anchored = prepare_model_input_window(df_test.values)
    test_norm = scaler.transform(test_anchored).astype(np.float32)
    windows = create_sliding_windows(test_norm, window_size=60)

    print(f"Loaded held-out normal test set: {len(windows)} sliding windows.")

    # Load Model
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tensor_windows = torch.from_numpy(windows).to(device)

    with torch.no_grad():
        _, _, anomaly_scores_step = model(tensor_windows)
        scores = torch.max(anomaly_scores_step, dim=1)[0].cpu().numpy()

    # Statistics
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores))
    p95_score = float(np.percentile(scores, 95))
    p99_score = float(np.percentile(scores, 99))
    max_score = float(np.max(scores))

    # FPR & Specificity Calculation (Since all held-out data is ACTUAL NORMAL)
    n_total_windows = len(scores)
    false_positives = int(np.sum(scores > threshold))
    true_negatives = n_total_windows - false_positives

    fpr = (false_positives / n_total_windows) * 100.0
    specificity = (true_negatives / n_total_windows) * 100.0

    passed_fpr = fpr <= 1.0
    passed_spec = specificity >= 99.0
    passed_gate = passed_fpr and passed_spec

    result = {
        "status": "PASSED" if passed_gate else "FAILED",
        "heldout_test_windows": n_total_windows,
        "threshold": round(threshold, 6),
        "mean_score": round(mean_score, 6),
        "median_score": round(median_score, 6),
        "std_score": round(std_score, 6),
        "p95_score": round(p95_score, 6),
        "p99_score": round(p99_score, 6),
        "max_score": round(max_score, 6),
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_positive_rate_pct": round(fpr, 4),
        "specificity_pct": round(specificity, 4),
        "passed_fpr_gate": passed_fpr,
        "passed_specificity_gate": passed_spec,
    }

    out_json = PROJECT_ROOT / "data" / "live_training" / "v3_heldout_eval_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Mean Score   : {mean_score:.6f}")
    print(f"Median Score : {median_score:.6f}")
    print(f"Std Score    : {std_score:.6f}")
    print(f"P95 Score    : {p95_score:.6f}")
    print(f"P99 Score    : {p99_score:.6f}")
    print(f"Max Score    : {max_score:.6f}")
    print(f"Threshold τ  : {threshold:.6f}")
    print(f"Above Thresh : {false_positives} / {n_total_windows}")
    print(f"FPR          : {fpr:.2f}% (Target <= 1.0%) -> {'🟢 PASS' if passed_fpr else '🔴 FAIL'}")
    print(f"Specificity  : {specificity:.2f}% (Target >= 99.0%) -> {'🟢 PASS' if passed_spec else '🔴 FAIL'}")

    if not passed_gate:
        raise ValueError(f"Phase 6 Unseen Normal Validation Gate FAILED! FPR={fpr:.2f}%, Specificity={specificity:.2f}%")

    print(f"[SUCCESS] Phase 6 Unseen Normal Validation Gate PASSED!")
    return result


if __name__ == "__main__":
    evaluate_v3_heldout()
