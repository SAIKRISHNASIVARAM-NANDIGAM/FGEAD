"""
demo_cli.py
CLI demo — great for viva / live presentations.

Usage:
  python demo_cli.py --window 100        # explain window 100
  python demo_cli.py --window 200        # explain window 200
  python demo_cli.py --list-anomalies    # list all detected anomaly windows
"""

import os
import sys
import argparse
import numpy as np
import torch

from data.preprocessor import TimeSeriesPreprocessor
from models.fgead import FGEAD
from models.explainer import FGEADExplainer

FEATURE_NAMES = [
    "cpu_usage", "cpu_temp", "memory_usage", "memory_free",
    "disk_io_read", "disk_io_write", "net_in", "net_out",
    "process_count", "context_switches", "cache_hits",
    "cache_misses", "load_avg_1m", "load_avg_5m", "load_avg_15m",
    "swap_usage", "iowait", "kernel_threads", "open_files", "network_errors",
]


def load_everything():
    proc = TimeSeriesPreprocessor(window_size=60, stride=5)
    data, labels = proc.load_csv("data/synthetic_data.csv")
    data = proc.handle_missing(data)
    data, labels = proc.remove_duplicates(data, labels)

    n = len(data)
    t_end = int(n * 0.70)
    v_end = int(n * 0.85)

    train_norm, _, test_norm = proc.normalize(
        data[:t_end], data[t_end:v_end], data[v_end:]
    )
    test_win, test_lbl = proc.create_windows(test_norm, labels[v_end:])
    n_features = data.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = FGEAD(n_features=n_features).to(device)

    ckpt = "checkpoints/best_model.pt"
    if not os.path.exists(ckpt):
        print(f"[!] No checkpoint found at {ckpt}. Run python train.py first.")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    explainer = FGEADExplainer(model, FEATURE_NAMES[:n_features])
    return test_win, test_lbl, model, explainer, device


def explain_window(test_win, model, explainer, device, idx):
    print(f"\n{'='*60}")
    print(f"  Explaining window index {idx}")
    print(f"{'='*60}")

    window = torch.FloatTensor(test_win[idx:idx+1])
    report = explainer.explain(window, window_start_abs=idx, device=device)
    print(explainer.format_report(report))


def list_anomalies(test_win, test_lbl, model, device, k_sigma=2.0, top_n=10):
    print("\n[Scanning all test windows for anomalies...]")
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(len(test_win)):
            x = torch.FloatTensor(test_win[i:i+1]).to(device)
            _, _, s = model(x)
            scores.append(float(s.max().cpu()))

    scores = np.array(scores)
    thr    = scores.mean() + k_sigma * scores.std()
    anomaly_idxs = np.where(scores >= thr)[0]

    print(f"\nThreshold: {thr:.4f}  |  Detected: {len(anomaly_idxs)} windows")
    print(f"\n{'#':>5}  {'Window Idx':>12}  {'Score':>8}  {'True Label':>12}")
    print("-" * 45)
    for rank, idx in enumerate(anomaly_idxs[:top_n], 1):
        lbl = "ANOMALY ⚠️" if test_lbl[idx] == 1 else "normal"
        print(f"{rank:>5}  {idx:>12}  {scores[idx]:>8.4f}  {lbl:>12}")

    return anomaly_idxs, scores, thr


def main():
    parser = argparse.ArgumentParser(description="FGEAD CLI Demo")
    parser.add_argument("--window",          type=int, default=None,
                        help="Index of test window to explain")
    parser.add_argument("--list-anomalies",  action="store_true",
                        help="List all detected anomaly windows")
    parser.add_argument("--first-anomaly",   action="store_true",
                        help="Explain the first detected anomaly window")
    args = parser.parse_args()

    test_win, test_lbl, model, explainer, device = load_everything()

    if args.list_anomalies:
        anomaly_idxs, _, _ = list_anomalies(test_win, test_lbl, model, device)
        if len(anomaly_idxs) > 0:
            print(f"\nUse --window {anomaly_idxs[0]} to explain the first anomaly.")

    elif args.first_anomaly:
        anomaly_idxs, _, _ = list_anomalies(test_win, test_lbl, model, device)
        if len(anomaly_idxs) > 0:
            explain_window(test_win, model, explainer, device, anomaly_idxs[0])

    elif args.window is not None:
        if args.window >= len(test_win):
            print(f"[!] Window index {args.window} out of range (max {len(test_win)-1})")
            sys.exit(1)
        explain_window(test_win, model, explainer, device, args.window)

    else:
        # Default: explain the first true anomaly
        anomaly_idxs = np.where(test_lbl == 1)[0]
        if len(anomaly_idxs) > 0:
            explain_window(test_win, model, explainer, device, anomaly_idxs[0])
        else:
            print("[!] No anomaly windows found. Try --list-anomalies")


if __name__ == "__main__":
    main()
