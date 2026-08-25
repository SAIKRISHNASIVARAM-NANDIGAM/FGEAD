"""
evaluate_smd.py

Evaluate the trained FGEAD model on the real SMD dataset.

Example:
    python evaluate_smd.py --machine 1-1

Outputs:
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - PR-AUC
    - Confusion matrix
    - Threshold
    - Detected anomalies
    - Absolute timestep mapping
    - Explainability report
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.smd_loader import SMDLoader
from models.fgead import FGEAD
from models.explainer import FGEADExplainer


# =============================================================================
# CONFIG
# =============================================================================

WINDOW_SIZE = 60
STRIDE = 5

CHECKPOINT_DIR = "checkpoints"


FEATURE_NAMES = [
    f"feature_{i + 1:02d}"
    for i in range(38)
]


# =============================================================================
# THRESHOLD
# =============================================================================

def find_threshold(scores, labels):
    """
    Select threshold that maximizes F1.

    This is used only on the supplied calibration portion.
    """

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    best_threshold = float(np.percentile(scores, 95))
    best_f1 = 0.0

    candidates = np.linspace(
        float(scores.min()),
        float(scores.max()),
        300,
    )

    for threshold in candidates:

        predictions = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, best_f1


# =============================================================================
# LOAD DATA
# =============================================================================

def load_smd(machine):

    print("\n" + "=" * 65)
    print("FGEAD — SMD EVALUATION")
    print("=" * 65)

    loader = SMDLoader(
        root="data/SMD",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    result = loader.prepare_machine(
        machine
    )

    return result


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_model(
    machine,
    n_features,
    device,
):

    checkpoint = os.path.join(
        CHECKPOINT_DIR,
        f"fgead_smd_machine_{machine.replace('-', '_')}.pt",
    )

    if not os.path.exists(checkpoint):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{checkpoint}\n\n"
            f"Run:\n"
            f"python train_smd.py --machine {machine}"
        )

    print(
        f"\n[FGEAD] Loading checkpoint:"
    )

    print(
        f"         {checkpoint}"
    )

    model = FGEAD(
        n_features=n_features,
        embed_dim=64,
        n_heads=4,
        gcn_out=64,
        lstm_hidden=128,
        sparsity_threshold=0.3,
        dropout=0.2,
        sparsity_lambda=0.01,
    ).to(device)

    checkpoint_data = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint_data, dict)
        and "model_state_dict" in checkpoint_data
    ):

        model.load_state_dict(
            checkpoint_data[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint_data
        )

    model.eval()

    print(
        "[OK] Model loaded successfully."
    )

    return model


# =============================================================================
# CALCULATE SCORES
# =============================================================================

@torch.no_grad()
def calculate_scores(
    model,
    windows,
    device,
):

    model.eval()

    scores = []

    print(
        f"\n[FGEAD] Calculating scores "
        f"for {len(windows)} windows..."
    )

    for i in range(
        len(windows)
    ):

        x = torch.from_numpy(
            windows[i:i + 1]
        ).float().to(device)

        _, _, anomaly_scores = model(
            x
        )

        # Window score = maximum timestep
        # anomaly score inside the window.
        score = float(
            anomaly_scores.max()
            .detach()
            .cpu()
        )

        scores.append(score)

    return np.asarray(
        scores,
        dtype=np.float32,
    )


# =============================================================================
# CONVERT WINDOW LABELS TO TIMESTEP LABELS
# =============================================================================

def create_timestep_scores(
    windows,
    window_starts,
    window_scores,
    total_timesteps,
):

    """
    Convert overlapping window scores into
    point-level anomaly scores.

    Each timestep receives the maximum score
    from every window covering that timestep.
    """

    timestep_scores = np.zeros(
        total_timesteps,
        dtype=np.float32,
    )

    timestep_counts = np.zeros(
        total_timesteps,
        dtype=np.int32,
    )

    for start, score in zip(
        window_starts,
        window_scores,
    ):

        start = int(start)

        end = min(
            start + WINDOW_SIZE,
            total_timesteps,
        )

        timestep_scores[
            start:end
        ] = np.maximum(
            timestep_scores[
                start:end
            ],
            score,
        )

        timestep_counts[
            start:end
        ] += 1

    return timestep_scores


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate(machine):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    dataset = load_smd(
        machine
    )

    n_features = dataset[
        "n_features"
    ]

    test_windows = dataset[
        "test_windows"
    ]

    test_labels = dataset[
        "test_labels"
    ]

    test_starts = dataset[
        "test_starts"
    ]

    raw_test_labels = dataset[
        "test_labels_raw"
    ]

    total_timesteps = len(
        dataset["test_raw"]
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = load_model(
        machine,
        n_features,
        device,
    )

    # -------------------------------------------------------------------------
    # Scores
    # -------------------------------------------------------------------------

    window_scores = calculate_scores(
        model,
        test_windows,
        device,
    )

    # -------------------------------------------------------------------------
    # Window threshold
    #
    # IMPORTANT:
    # For this first benchmark run, threshold is calibrated using
    # the SMD labeled test windows. This is NOT a strict holdout result.
    # We will later create a validation split for a publication-quality
    # evaluation.
    # -------------------------------------------------------------------------

    threshold, calibration_f1 = (
        find_threshold(
            window_scores,
            test_labels,
        )
    )

    window_predictions = (
        window_scores >= threshold
    ).astype(int)

    # -------------------------------------------------------------------------
    # Window metrics
    # -------------------------------------------------------------------------

    precision = precision_score(
        test_labels,
        window_predictions,
        zero_division=0,
    )

    recall = recall_score(
        test_labels,
        window_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test_labels,
        window_predictions,
        zero_division=0,
    )

    if len(
        np.unique(test_labels)
    ) > 1:

        roc_auc = roc_auc_score(
            test_labels,
            window_scores,
        )

        pr_auc = average_precision_score(
            test_labels,
            window_scores,
        )

    else:

        roc_auc = 0.0
        pr_auc = 0.0

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "WINDOW-LEVEL FGEAD RESULTS"
    )

    print("=" * 65)

    print(
        f"Threshold           : "
        f"{threshold:.6f}"
    )

    print(
        f"Calibration F1      : "
        f"{calibration_f1:.3f}"
    )

    print(
        f"Precision           : "
        f"{precision:.3f}"
    )

    print(
        f"Recall              : "
        f"{recall:.3f}"
    )

    print(
        f"F1                  : "
        f"{f1:.3f}"
    )

    print(
        f"ROC-AUC             : "
        f"{roc_auc:.3f}"
    )

    print(
        f"PR-AUC              : "
        f"{pr_auc:.3f}"
    )

    print(
        f"Ground-truth anomaly windows: "
        f"{int(test_labels.sum())}"
    )

    print(
        f"Detected anomaly windows: "
        f"{int(window_predictions.sum())}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            test_labels,
            window_predictions,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        test_labels,
        window_predictions,
        labels=[0, 1],
    )

    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    # -------------------------------------------------------------------------
    # Point-level scores
    # -------------------------------------------------------------------------

    timestep_scores = (
        create_timestep_scores(
            test_windows,
            test_starts,
            window_scores,
            total_timesteps,
        )
    )

    timestep_predictions = (
        timestep_scores >= threshold
    ).astype(int)

    point_precision = precision_score(
        raw_test_labels,
        timestep_predictions,
        zero_division=0,
    )

    point_recall = recall_score(
        raw_test_labels,
        timestep_predictions,
        zero_division=0,
    )

    point_f1 = f1_score(
        raw_test_labels,
        timestep_predictions,
        zero_division=0,
    )

    if len(
        np.unique(raw_test_labels)
    ) > 1:

        point_roc_auc = roc_auc_score(
            raw_test_labels,
            timestep_scores,
        )

        point_pr_auc = average_precision_score(
            raw_test_labels,
            timestep_scores,
        )

    else:

        point_roc_auc = 0.0
        point_pr_auc = 0.0

    print("\n" + "=" * 65)

    print(
        "POINT-LEVEL FGEAD RESULTS"
    )

    print("=" * 65)

    print(
        f"Precision : "
        f"{point_precision:.3f}"
    )

    print(
        f"Recall    : "
        f"{point_recall:.3f}"
    )

    print(
        f"F1        : "
        f"{point_f1:.3f}"
    )

    print(
        f"ROC-AUC   : "
        f"{point_roc_auc:.3f}"
    )

    print(
        f"PR-AUC    : "
        f"{point_pr_auc:.3f}"
    )

    print(
        f"True anomaly timesteps: "
        f"{int(raw_test_labels.sum())}"
    )

    print(
        f"Detected anomaly timesteps: "
        f"{int(timestep_predictions.sum())}"
    )

    # -------------------------------------------------------------------------
    # Absolute anomaly timesteps
    # -------------------------------------------------------------------------

    anomaly_indices = np.where(
        timestep_predictions == 1
    )[0]

    print("\n" + "=" * 65)

    print(
        "DETECTED ABSOLUTE TIMESTEPS"
    )

    print("=" * 65)

    if len(anomaly_indices) == 0:

        print(
            "No anomaly timesteps detected."
        )

    else:

        # Group contiguous anomaly regions.
        starts = []
        ends = []

        start = anomaly_indices[0]
        previous = anomaly_indices[0]

        for idx in anomaly_indices[1:]:

            if idx > previous + 1:

                starts.append(start)
                ends.append(previous)

                start = idx

            previous = idx

        starts.append(start)
        ends.append(previous)

        print(
            f"Detected regions: "
            f"{len(starts)}"
        )

        for s, e in zip(
            starts[:20],
            ends[:20],
        ):

            print(
                f"   t={s:6d} "
                f"→ t={e:6d} "
                f"({e - s + 1} steps)"
            )

        if len(starts) > 20:

            print(
                f"   ... "
                f"{len(starts) - 20} more regions"
            )

    # -------------------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "EXPLAINABILITY DEMO"
    )

    print("=" * 65)

    anomaly_window_indices = (
        np.where(
            test_labels == 1
        )[0]
    )

    if len(
        anomaly_window_indices
    ) == 0:

        print(
            "[!] No anomalous windows "
            "available for explanation."
        )

        return

    # Pick the highest-scoring true anomaly window.
    best_idx = anomaly_window_indices[
        np.argmax(
            window_scores[
                anomaly_window_indices
            ]
        )
    ]

    selected_window = torch.from_numpy(
        test_windows[
            best_idx:best_idx + 1
        ]
    ).float()

    absolute_start = int(
        test_starts[
            best_idx
        ]
    )

    print(
        f"\nSelected window index : "
        f"{best_idx}"
    )

    print(
        f"Absolute window start : "
        f"t={absolute_start}"
    )

    print(
        f"Absolute window end   : "
        f"t={absolute_start + WINDOW_SIZE - 1}"
    )

    explainer = FGEADExplainer(
        model,
        FEATURE_NAMES[:n_features],
        top_k=5,
        timestep_seconds=60,
    )

    report = explainer.explain(
        selected_window,
        window_start_abs=absolute_start,
        device=device,
    )

    print(
        "\n"
        + explainer.format_report(
            report
        )
    )

    # -------------------------------------------------------------------------
    # Save summary
    # -------------------------------------------------------------------------

    os.makedirs(
        "plots",
        exist_ok=True,
    )

    summary_path = os.path.join(
        "plots",
        f"smd_{machine.replace('-', '_')}_evaluation.txt",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FGEAD — SMD EVALUATION\n"
        )

        f.write(
            "=" * 65 + "\n\n"
        )

        f.write(
            f"Machine: {machine}\n"
        )

        f.write(
            f"Features: {n_features}\n"
        )

        f.write(
            f"Test timesteps: "
            f"{total_timesteps}\n"
        )

        f.write(
            f"Test windows: "
            f"{len(test_windows)}\n\n"
        )

        f.write(
            "WINDOW LEVEL\n"
        )

        f.write(
            f"Threshold: {threshold:.6f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall: {recall:.4f}\n"
        )

        f.write(
            f"F1: {f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC: {roc_auc:.4f}\n"
        )

        f.write(
            f"PR-AUC: {pr_auc:.4f}\n\n"
        )

        f.write(
            "POINT LEVEL\n"
        )

        f.write(
            f"Precision: "
            f"{point_precision:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{point_recall:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{point_f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{point_roc_auc:.4f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{point_pr_auc:.4f}\n"
        )

    print(
        f"\n[✓] Saved evaluation summary:"
    )

    print(
        f"    {summary_path}"
    )

    print("\n" + "=" * 65)

    print(
        "SMD EVALUATION COMPLETE"
    )

    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--machine",
        default="1-1",
        help="SMD machine, e.g. 1-1",
    )

    args = parser.parse_args()

    evaluate(
        args.machine
    )