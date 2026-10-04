"""
generate_smd_plots.py

Report-ready visualizations for FGEAD on the Server Machine Dataset (SMD).

Usage:
    python generate_smd_plots.py --machine 1-1

Outputs:
    plots/smd_<machine>_confusion_matrix.png
    plots/smd_<machine>_roc_curve.png
    plots/smd_<machine>_precision_recall_curve.png
    plots/smd_<machine>_anomaly_score_timeline.png
    plots/smd_<machine>_feature_contributions.png
    plots/smd_<machine>_evaluation_results.txt

Important:
- Uses the existing SMDLoader.
- Uses train-only normalization.
- Loads the existing SMD checkpoint.
- Uses the same 99.50th-percentile calibration used by evaluate_smd_final.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from data.smd_loader import SMDLoader
from models.fgead import FGEAD


WINDOW_SIZE = 60
STRIDE = 5
CALIBRATION_RATIO = 0.15
THRESHOLD_PERCENTILE = 99.50
PLOT_DIR = Path("plots")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate report-ready FGEAD SMD plots."
    )
    parser.add_argument(
        "--machine",
        default="1-1",
        help="SMD machine, e.g. 1-1",
    )
    return parser.parse_args()


def load_smd(machine: str):
    root = Path("data/SMD")

    loader = SMDLoader(
        root=str(root),
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    train, test, labels = loader.load_machine(machine)

    train_norm, test_norm = loader.normalize(
        train,
        test,
    )

    train_windows, _, _ = loader.create_windows(
        train_norm,
        np.zeros(len(train_norm), dtype=np.int64),
    )

    test_windows, test_window_labels, test_window_starts = loader.create_windows(
        test_norm,
        labels,
    )

    # Absolute start of the test timeline is zero for the SMD test file.
    test_start_abs = 0

    return {
        "loader": loader,
        "train": train,
        "test": test,
        "labels": labels,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "test_window_labels": test_window_labels,
        "test_window_starts": test_window_starts,
        "test_start_abs": test_start_abs,
        "feature_names": [
            f"feature_{i:02d}"
            for i in range(train.shape[1])
        ],
    }


def load_model(machine: str, n_features: int):
    checkpoint = Path(
        "checkpoints"
    ) / f"fgead_smd_machine_{machine.replace('-', '_')}.pt"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SMD checkpoint not found:\n{checkpoint}"
        )

    model = FGEAD(
        n_features=n_features
    ).to(DEVICE)

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]
    else:
        model_state = state

    model.load_state_dict(
        model_state,
        strict=True,
    )

    model.eval()

    print(f"[OK] Loaded checkpoint: {checkpoint}")

    return model


def calculate_scores(model, windows):
    scores = []

    print(
        f"[FGEAD] Calculating scores for {len(windows)} windows..."
    )

    with torch.no_grad():
        for i in range(len(windows)):
            x = torch.as_tensor(
                windows[i:i + 1],
                dtype=torch.float32,
                device=DEVICE,
            )

            _, _, anomaly_scores = model(x)

            scores.append(
                float(anomaly_scores.max().detach().cpu())
            )

            if (i + 1) % 1000 == 0:
                print(f"     {i + 1}/{len(windows)}")

    return np.asarray(
        scores,
        dtype=np.float32,
    )


def calibrate_threshold(train_scores):
    calibration_count = max(
        1,
        int(len(train_scores) * CALIBRATION_RATIO),
    )

    calibration_scores = train_scores[:calibration_count]

    threshold = float(
        np.percentile(
            calibration_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    return threshold, calibration_scores


def save_figure(fig, filename):
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = PLOT_DIR / filename

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[OK] Saved {path}")


def plot_confusion_matrix(labels, predictions, threshold, machine):
    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.imshow(cm, interpolation="nearest")

    ax.set_title(
        f"FGEAD — SMD Machine {machine}\n"
        f"Confusion Matrix | Threshold = {threshold:.4f}",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
            )

    fig.tight_layout()

    save_figure(
        fig,
        f"smd_{machine.replace('-', '_')}_confusion_matrix.png",
    )


def plot_roc(labels, scores, machine):
    fpr, tpr, _ = roc_curve(
        labels,
        scores,
    )

    score_auc = roc_auc_score(
        labels,
        scores,
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        label=f"FGEAD — AUC = {score_auc:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier",
    )

    ax.set_title(
        f"FGEAD — SMD Machine {machine}\n"
        "Receiver Operating Characteristic",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    save_figure(
        fig,
        f"smd_{machine.replace('-', '_')}_roc_curve.png",
    )


def plot_precision_recall(labels, scores, machine):
    precision, recall, _ = precision_recall_curve(
        labels,
        scores,
    )

    score_auc = auc(
        recall,
        precision,
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.plot(
        recall,
        precision,
        linewidth=2.5,
        label=f"FGEAD — AUC = {score_auc:.3f}",
    )

    ax.set_title(
        f"FGEAD — SMD Machine {machine}\n"
        "Precision-Recall Curve",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    save_figure(
        fig,
        f"smd_{machine.replace('-', '_')}_precision_recall_curve.png",
    )


def plot_anomaly_timeline(
    scores,
    labels,
    predictions,
    threshold,
    machine,
):
    positions = (
        np.arange(len(scores))
        * STRIDE
    )

    fig, ax = plt.subplots(
        figsize=(16, 7)
    )

    ax.plot(
        positions,
        scores,
        linewidth=1.6,
        label="FGEAD anomaly score",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.8,
        label=f"Threshold = {threshold:.3f}",
    )

    true_positions = positions[
        labels == 1
    ]

    true_scores = scores[
        labels == 1
    ]

    if len(true_positions):
        ax.scatter(
            true_positions,
            true_scores,
            marker="o",
            s=18,
            label="Ground-truth anomaly",
        )

    predicted_positions = positions[
        predictions == 1
    ]

    predicted_scores = scores[
        predictions == 1
    ]

    if len(predicted_positions):
        ax.scatter(
            predicted_positions,
            predicted_scores,
            marker="x",
            s=28,
            linewidths=1.4,
            label="FGEAD detection",
        )

    ax.set_title(
        f"FGEAD — SMD Machine {machine}\n"
        "Anomaly Score Timeline",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Absolute Test Timestep")
    ax.set_ylabel("Anomaly Score")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    save_figure(
        fig,
        f"smd_{machine.replace('-', '_')}_anomaly_score_timeline.png",
    )


def plot_feature_contributions(
    model,
    windows,
    feature_names,
    machine,
):
    # Select the highest-scoring test window.
    window_scores = calculate_scores(
        model,
        windows,
    )

    selected_index = int(
        np.argmax(window_scores)
    )

    x = torch.as_tensor(
        windows[selected_index:selected_index + 1],
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        prediction, _, _ = model(x)

    actual = x[0, -1].detach().cpu().numpy()
    predicted = prediction[0, -1].detach().cpu().numpy()
    deviations = np.abs(actual - predicted)
    deviations = np.asarray(deviations).reshape(-1)

    top_count = min(
        10,
        len(feature_names),
    )

    indices = np.argsort(
        deviations
    )[::-1][:top_count]

    feature_names = np.asarray(feature_names)

    names = [
        feature_names[i]
        for i in indices[::-1]
    ]

    values = [
        float(deviations[i])
        for i in indices[::-1]
    ]

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        names,
        values,
    )

    ax.set_title(
        f"FGEAD — SMD Machine {machine}\n"
        "Top Contributing Features",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_xlabel("Absolute Prediction Deviation")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    save_figure(
        fig,
        f"smd_{machine.replace('-', '_')}_feature_contributions.png",
    )

    return selected_index


def write_results(
    machine,
    threshold,
    precision,
    recall,
    f1,
    roc_auc,
    pr_auc,
    labels,
    predictions,
):
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        PLOT_DIR
        / f"smd_{machine.replace('-', '_')}_plot_results.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "FGEAD — SMD PLOT RESULTS\n"
        )
        f.write(
            "========================\n\n"
        )
        f.write(
            f"Machine       : {machine}\n"
        )
        f.write(
            f"Threshold     : {threshold:.6f}\n"
        )
        f.write(
            f"Precision     : {precision:.6f}\n"
        )
        f.write(
            f"Recall        : {recall:.6f}\n"
        )
        f.write(
            f"F1            : {f1:.6f}\n"
        )
        f.write(
            f"ROC-AUC       : {roc_auc:.6f}\n"
        )
        f.write(
            f"PR-AUC        : {pr_auc:.6f}\n"
        )
        f.write(
            f"True anomaly windows : "
            f"{int(labels.sum())}\n"
        )
        f.write(
            f"Detected anomaly windows : "
            f"{int(predictions.sum())}\n"
        )

    print(f"[OK] Saved {path}")


def main():
    args = parse_args()
    machine = args.machine

    print(
        "\n"
        + "=" * 70
    )
    print(
        "FGEAD — SMD REPORT VISUALIZATION"
    )
    print(
        "=" * 70
    )

    print(f"Device  : {DEVICE}")
    print(f"Machine : {machine}")

    dataset = load_smd(
        machine
    )

    test_windows = dataset[
        "test_windows"
    ]

    test_labels = dataset[
        "test_window_labels"
    ]

    train_windows = dataset[
        "train_windows"
    ]

    feature_names = dataset[
        "feature_names"
    ]

    n_features = dataset[
        "train"
    ].shape[1]

    print(
        "\nDataset:"
    )
    print(
        f"  Features       : {n_features}"
    )
    print(
        f"  Train windows  : {len(train_windows)}"
    )
    print(
        f"  Test windows   : {len(test_windows)}"
    )
    print(
        f"  Test anomalies : {int(test_labels.sum())}"
    )

    model = load_model(
        machine,
        n_features,
    )

    print(
        "\nCalculating calibration scores..."
    )

    train_scores = calculate_scores(
        model,
        train_windows,
    )

    threshold, calibration_scores = calibrate_threshold(
        train_scores
    )

    print(
        "\nThreshold calibration:"
    )
    print(
        f"  Calibration windows : {len(calibration_scores)}"
    )
    print(
        f"  Mean                : "
        f"{calibration_scores.mean():.6f}"
    )
    print(
        f"  Std                 : "
        f"{calibration_scores.std():.6f}"
    )
    print(
        f"  Max                 : "
        f"{calibration_scores.max():.6f}"
    )
    print(
        f"  Percentile          : "
        f"{THRESHOLD_PERCENTILE:.2f}%"
    )
    print(
        f"  Threshold           : "
        f"{threshold:.6f}"
    )

    print(
        "\nCalculating test scores..."
    )

    test_scores = calculate_scores(
        model,
        test_windows,
    )

    predictions = (
        test_scores >= threshold
    ).astype(np.int64)

    precision = precision_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        test_labels,
        test_scores,
    )

    pr_precision, pr_recall, _ = precision_recall_curve(
        test_labels,
        test_scores,
    )

    pr_auc = auc(
        pr_recall,
        pr_precision,
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "SMD PLOT METRICS"
    )
    print(
        "=" * 70
    )
    print(
        f"Threshold : {threshold:.6f}"
    )
    print(
        f"Precision : {precision:.3f}"
    )
    print(
        f"Recall    : {recall:.3f}"
    )
    print(
        f"F1        : {f1:.3f}"
    )
    print(
        f"ROC-AUC   : {roc_auc:.3f}"
    )
    print(
        f"PR-AUC    : {pr_auc:.3f}"
    )
    print(
        f"Ground-truth anomalies : "
        f"{int(test_labels.sum())}"
    )
    print(
        f"Detected anomalies     : "
        f"{int(predictions.sum())}"
    )

    print(
        "\nGenerating SMD report figures..."
    )

    plot_confusion_matrix(
        test_labels,
        predictions,
        threshold,
        machine,
    )

    plot_roc(
        test_labels,
        test_scores,
        machine,
    )

    plot_precision_recall(
        test_labels,
        test_scores,
        machine,
    )

    plot_anomaly_timeline(
        test_scores,
        test_labels,
        predictions,
        threshold,
        machine,
    )

    plot_feature_contributions(
        model,
        test_windows,
        feature_names,
        machine,
    )

    write_results(
        machine,
        threshold,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        test_labels,
        predictions,
    )

    print(
        "\n"
        + "=" * 70
    )
    print(
        "SMD PLOT GENERATION COMPLETE"
    )
    print(
        "=" * 70
    )
    print(
        "\nFigures are available inside:"
    )
    print(
        f"    {PLOT_DIR}\\"
    )


if __name__ == "__main__":
    main()










