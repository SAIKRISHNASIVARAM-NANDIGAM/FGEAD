"""
generate_plots.py

Report-ready FGEAD visualization.

IMPORTANT:
Uses the SAME validation-set F1 thresholding strategy as evaluate.py,
so the plotted predictions are consistent with the official evaluation.

Outputs:
    plots/confusion_matrix.png
    plots/roc_curve.png
    plots/precision_recall_curve.png
    plots/anomaly_score_timeline.png
    plots/training_validation_loss.png
"""

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

from data.preprocessor import TimeSeriesPreprocessor
from models.fgead import FGEAD


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = "data/synthetic_data.csv"
CHECKPOINT_PATH = "checkpoints/best_model.pt"
HISTORY_PATH = "checkpoints/train_history.npy"

PLOT_DIR = "plots"

WINDOW_SIZE = 60
STRIDE = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# DATA
# =============================================================================

def load_dataset():

    proc = TimeSeriesPreprocessor(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    data, labels = proc.load_csv(
        DATA_PATH
    )

    data = proc.handle_missing(data)

    data, labels = proc.remove_duplicates(
        data,
        labels,
    )

    n = len(data)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]

    train_norm, val_norm, test_norm = proc.normalize(
        train_data,
        val_data,
        test_data,
    )

    val_windows, val_window_labels = proc.create_windows(
        val_norm,
        val_labels,
    )

    test_windows, test_window_labels = proc.create_windows(
        test_norm,
        test_labels,
    )

    return {
        "data": data,
        "labels": labels,
        "val_windows": val_windows,
        "val_labels": val_window_labels,
        "test_windows": test_windows,
        "test_labels": test_window_labels,
        "test_start_abs": val_end,
        "proc": proc,
    }


# =============================================================================
# MODEL
# =============================================================================

def load_model(n_features):

    model = FGEAD(
        n_features=n_features
    ).to(DEVICE)

    if not os.path.exists(
        CHECKPOINT_PATH
    ):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    return model


# =============================================================================
# SCORE WINDOWS
# =============================================================================

def calculate_scores(
    model,
    windows,
):

    scores = []

    print(
        f"[FGEAD] Calculating scores "
        f"for {len(windows)} windows..."
    )

    with torch.no_grad():

        for i in range(
            len(windows)
        ):

            x = torch.FloatTensor(
                windows[i:i + 1]
            ).to(DEVICE)

            _, _, anomaly_scores = model(
                x
            )

            score = float(
                anomaly_scores.max().cpu()
            )

            scores.append(score)

    return np.asarray(
        scores,
        dtype=np.float32,
    )


# =============================================================================
# SAME THRESHOLD METHOD AS evaluate.py
# =============================================================================

def best_threshold_f1(
    scores,
    labels,
):

    best_f1 = 0.0
    best_threshold = 0.5

    score_min = float(
        scores.min()
    )

    score_max = float(
        scores.max()
    )

    for threshold in np.linspace(
        score_min,
        score_max,
        200,
    ):

        predictions = (
            scores >= threshold
        ).astype(int)

        if predictions.sum() == 0:
            continue

        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = float(
                threshold
            )

    return best_threshold


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def create_confusion_matrix(
    labels,
    predictions,
    threshold,
):

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(111)

    ax.imshow(
        cm,
        interpolation="nearest",
    )

    ax.set_title(
        "FGEAD Confusion Matrix",
        fontsize=17,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12,
    )

    ax.set_ylabel(
        "True Label",
        fontsize=12,
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Normal", "Anomaly"]
    )

    ax.set_yticklabels(
        ["Normal", "Anomaly"]
    )

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

    ax.set_title(
        f"FGEAD Confusion Matrix\n"
        f"Threshold = {threshold:.4f}",
        fontsize=16,
        fontweight="bold",
    )

    fig.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "confusion_matrix.png",
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[✓] Saved {path}"
    )


# =============================================================================
# ROC
# =============================================================================

def create_roc_curve(
    labels,
    scores,
):

    fpr, tpr, _ = roc_curve(
        labels,
        scores,
    )

    roc_auc = roc_auc_score(
        labels,
        scores,
    )

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        label=f"FGEAD — AUC = {roc_auc:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier",
    )

    ax.set_title(
        "FGEAD Receiver Operating Characteristic",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "roc_curve.png",
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[✓] Saved {path}"
    )


# =============================================================================
# PRECISION RECALL
# =============================================================================

def create_precision_recall_curve(
    labels,
    scores,
):

    precision, recall, _ = (
        precision_recall_curve(
            labels,
            scores,
        )
    )

    pr_auc = auc(
        recall,
        precision,
    )

    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        recall,
        precision,
        linewidth=2.5,
        label=f"FGEAD — AUC = {pr_auc:.3f}",
    )

    ax.set_title(
        "FGEAD Precision-Recall Curve",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "precision_recall_curve.png",
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[✓] Saved {path}"
    )


# =============================================================================
# ANOMALY TIMELINE
# =============================================================================

def create_anomaly_timeline(
    scores,
    labels,
    predictions,
    test_start_abs,
    threshold,
):

    window_positions = (
        test_start_abs
        + np.arange(len(scores))
        * STRIDE
    )

    fig = plt.figure(
        figsize=(15, 7)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        window_positions,
        scores,
        linewidth=2,
        label="FGEAD anomaly score",
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.8,
        label=f"Calibrated threshold = {threshold:.3f}",
    )

    true_positions = (
        window_positions[
            labels == 1
        ]
    )

    true_scores = (
        scores[
            labels == 1
        ]
    )

    if len(true_positions) > 0:

        ax.scatter(
            true_positions,
            true_scores,
            marker="o",
            s=40,
            label="Ground-truth anomaly",
        )

    predicted_positions = (
        window_positions[
            predictions == 1
        ]
    )

    predicted_scores = (
        scores[
            predictions == 1
        ]
    )

    if len(predicted_positions) > 0:

        ax.scatter(
            predicted_positions,
            predicted_scores,
            marker="x",
            s=55,
            linewidths=2,
            label="FGEAD detection",
        )

    ax.set_title(
        "FGEAD Anomaly Score Timeline",
        fontsize=17,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Absolute Timestep",
        fontsize=12,
    )

    ax.set_ylabel(
        "Anomaly Score",
        fontsize=12,
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "anomaly_score_timeline.png",
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[✓] Saved {path}"
    )


# =============================================================================
# TRAINING LOSS
# =============================================================================

def create_training_plot():

    if not os.path.exists(
        HISTORY_PATH
    ):

        print(
            "[!] train_history.npy not found."
        )

        return

    history = np.load(
        HISTORY_PATH,
        allow_pickle=True,
    ).item()

    train_loss = history[
        "train_loss"
    ]

    val_loss = history[
        "val_loss"
    ]

    epochs = np.arange(
        1,
        len(train_loss) + 1,
    )

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        epochs,
        train_loss,
        linewidth=2.5,
        label="Training Loss",
    )

    ax.plot(
        epochs,
        val_loss,
        linewidth=2.5,
        label="Validation Loss",
    )

    best_epoch = int(
        np.argmin(val_loss)
    ) + 1

    best_value = float(
        np.min(val_loss)
    )

    ax.scatter(
        [best_epoch],
        [best_value],
        s=60,
        zorder=5,
        label=(
            f"Best validation loss "
            f"({best_value:.4f})"
        ),
    )

    ax.set_title(
        "FGEAD Training and Validation Loss",
        fontsize=17,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    path = os.path.join(
        PLOT_DIR,
        "training_validation_loss.png",
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[✓] Saved {path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    os.makedirs(
        PLOT_DIR,
        exist_ok=True,
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FGEAD REPORT-READY VISUALIZATION"
    )

    print(
        "=" * 65
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    dataset = load_dataset()

    data = dataset["data"]

    val_windows = dataset[
        "val_windows"
    ]

    val_labels = dataset[
        "val_labels"
    ]

    test_windows = dataset[
        "test_windows"
    ]

    test_labels = dataset[
        "test_labels"
    ]

    test_start_abs = dataset[
        "test_start_abs"
    ]

    n_features = data.shape[1]

    print(
        f"\nTotal timesteps : {len(data)}"
    )

    print(
        f"Features        : {n_features}"
    )

    print(
        f"Validation windows : "
        f"{len(val_windows)}"
    )

    print(
        f"Test windows       : "
        f"{len(test_windows)}"
    )

    print(
        f"Test absolute start: "
        f"{test_start_abs}"
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = load_model(
        n_features
    )

    # -------------------------------------------------------------------------
    # Validation scores
    # -------------------------------------------------------------------------

    print(
        "\nCalculating validation scores..."
    )

    val_scores = calculate_scores(
        model,
        val_windows,
    )

    # -------------------------------------------------------------------------
    # Test scores
    # -------------------------------------------------------------------------

    print(
        "\nCalculating test scores..."
    )

    test_scores = calculate_scores(
        model,
        test_windows,
    )

    # -------------------------------------------------------------------------
    # EXACT SAME THRESHOLD CALIBRATION
    # -------------------------------------------------------------------------

    threshold = best_threshold_f1(
        val_scores,
        val_labels,
    )

    # Same 5% adjustment used in evaluate.py
    threshold *= 1.05

    predictions = (
        test_scores >= threshold
    ).astype(int)

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

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

    print(
        "\n"
        + "=" * 65
    )

    print(
        "OFFICIAL FGEAD PLOT METRICS"
    )

    print(
        "=" * 65
    )

    print(
        f"Threshold : {threshold:.4f}"
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
        f"Ground truth anomalies : "
        f"{int(test_labels.sum())}"
    )

    print(
        f"Detected anomalies     : "
        f"{int(predictions.sum())}"
    )

    # -------------------------------------------------------------------------
    # Generate plots
    # -------------------------------------------------------------------------

    print(
        "\nGenerating report figures..."
    )

    create_confusion_matrix(
        test_labels,
        predictions,
        threshold,
    )

    create_roc_curve(
        test_labels,
        test_scores,
    )

    create_precision_recall_curve(
        test_labels,
        test_scores,
    )

    create_anomaly_timeline(
        test_scores,
        test_labels,
        predictions,
        test_start_abs,
        threshold,
    )

    create_training_plot()

    # -------------------------------------------------------------------------
    # Save numerical results
    # -------------------------------------------------------------------------

    results_path = os.path.join(
        PLOT_DIR,
        "evaluation_results.txt",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FGEAD OFFICIAL EVALUATION RESULTS\n"
        )

        f.write(
            "=================================\n\n"
        )

        f.write(
            f"Threshold : {threshold:.6f}\n"
        )

        f.write(
            f"Precision : {precision:.6f}\n"
        )

        f.write(
            f"Recall    : {recall:.6f}\n"
        )

        f.write(
            f"F1        : {f1:.6f}\n"
        )

        f.write(
            f"ROC-AUC   : {roc_auc:.6f}\n"
        )

        f.write(
            f"Test anomalies : {int(test_labels.sum())}\n"
        )

        f.write(
            f"Detected anomalies : "
            f"{int(predictions.sum())}\n"
        )

    print(
        f"[✓] Saved {results_path}"
    )

    print(
        "\n"
        + "=" * 65
    )

    print(
        "DONE"
    )

    print(
        "=" * 65
    )

    print(
        "\nYour plots are now calibrated "
        "using the same threshold as evaluate.py."
    )


if __name__ == "__main__":
    main()