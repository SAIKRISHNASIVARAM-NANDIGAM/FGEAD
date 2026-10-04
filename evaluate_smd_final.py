"""
evaluate_smd_final.py

Final FGEAD evaluation on the Server Machine Dataset (SMD).

Features:
- Uses the actual SMDLoader implementation
- Train-only normalization
- Sliding-window evaluation
- Window-level anomaly detection
- Point-level overlapping-window aggregation
- Temporal smoothing
- Train-derived threshold calibration
- Absolute timestep mapping
- Anomaly range detection
- Original-scale explainability
- Feature contribution analysis
- Feature relationship analysis
- Confidence score
- Final results report

Usage:
    python evaluate_smd_final.py --machine 1-1
"""

from __future__ import annotations

import argparse
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

import warnings
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from sklearn.preprocessing import StandardScaler

from data.smd_loader import SMDLoader
from models.fgead import FGEAD
from models.explainer import FGEADExplainer


# ============================================================================
# CONFIGURATION
# ============================================================================

WINDOW_SIZE = 60
STRIDE = 5

CALIBRATION_RATIO = 0.15

# Threshold is selected ONLY from training/calibration scores.
THRESHOLD_PERCENTILE = 99.5

SMOOTHING_WINDOW = 5

MIN_ANOMALY_DURATION = 5

MERGE_GAP = 10

TIMESTEP_SECONDS = 60


# ============================================================================
# UTILITY
# ============================================================================

def safe_auc(y_true, scores):
    try:
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        if len(np.unique(y_true)) < 2:
            return 0.0

        return float(roc_auc_score(y_true, scores))

    except Exception:
        return 0.0


def safe_pr_auc(y_true, scores):
    try:
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)

        if len(np.unique(y_true)) < 2:
            return 0.0

        return float(
            average_precision_score(
                y_true,
                scores,
            )
        )

    except Exception:
        return 0.0


# ============================================================================
# WINDOW CREATION
# ============================================================================

def create_windows(
    data: np.ndarray,
    labels: np.ndarray | None = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):
    """
    Create chronological sliding windows.

    Window label = 1 if ANY timestep inside the window is anomalous.
    """

    data = np.asarray(data)

    windows = []
    window_labels = []
    starts = []

    n = len(data)

    for start in range(
        0,
        n - window_size + 1,
        stride,
    ):
        end = start + window_size

        windows.append(
            data[start:end]
        )

        starts.append(start)

        if labels is not None:
            window_labels.append(
                int(
                    np.any(
                        labels[start:end] > 0
                    )
                )
            )

    windows = np.asarray(
        windows,
        dtype=np.float32,
    )

    starts = np.asarray(
        starts,
        dtype=np.int64,
    )

    if labels is not None:

        window_labels = np.asarray(
            window_labels,
            dtype=np.int64,
        )

        return (
            windows,
            window_labels,
            starts,
        )

    return (
        windows,
        starts,
    )


# ============================================================================
# SMD LOADING
# ============================================================================

def load_smd_machine(machine: str):
    """
    Load SMD using the actual SMDLoader implementation.

    IMPORTANT:

    The current SMDLoader.load_machine() returns:

        train, test, labels

    It does NOT return a dictionary.

    Therefore this function performs the remaining preparation here.
    """

    dataset_root = Path(
        "data"
    ) / "SMD"

    if not dataset_root.exists():

        raise FileNotFoundError(
            "\nSMD dataset directory not found:\n"
            f"{dataset_root.resolve()}\n"
        )

    print(
        f"\n[SMD] Loading machine {machine}..."
    )

    # Your actual SMDLoader uses root, not data_dir.
    loader = SMDLoader(
        root=str(dataset_root),
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    # ------------------------------------------------------------
    # RAW DATA
    # ------------------------------------------------------------

    train_raw, test_raw, test_labels = (
        loader.load_machine(machine)
    )

    train_raw = np.asarray(
        train_raw,
        dtype=np.float32,
    )

    test_raw = np.asarray(
        test_raw,
        dtype=np.float32,
    )

    test_labels = np.asarray(
        test_labels,
        dtype=np.int64,
    ).reshape(-1)

    n_features = train_raw.shape[1]

    # ------------------------------------------------------------
    # TRAIN-ONLY NORMALIZATION
    # ------------------------------------------------------------

    train_norm, test_norm = loader.normalize(
        train_raw,
        test_raw,
    )

    train_norm = np.asarray(
        train_norm,
        dtype=np.float32,
    )

    test_norm = np.asarray(
        test_norm,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # WINDOWS
    # ------------------------------------------------------------

    (
        train_windows,
        train_starts,
    ) = create_windows(
        train_norm,
        labels=None,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    (
        test_windows,
        test_window_labels,
        test_starts,
    ) = create_windows(
        test_norm,
        labels=test_labels,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    # ------------------------------------------------------------
    # PRINT DATA INFORMATION
    # ------------------------------------------------------------

    print()
    print(
        "[SMD] Windowed dataset"
    )

    print(
        f"      Train windows : "
        f"{len(train_windows)}"
    )

    print(
        f"      Test windows  : "
        f"{len(test_windows)}"
    )

    print(
        f"      Window size   : "
        f"{WINDOW_SIZE}"
    )

    print(
        f"      Stride        : "
        f"{STRIDE}"
    )

    print(
        f"      Test anomalies: "
        f"{int(test_window_labels.sum())}"
    )

    if len(test_starts) > 0:

        print(
            f"      Test timeline : "
            f"{int(test_starts[0])} → "
            f"{int(test_starts[-1])}"
        )

    print()
    print(
        "[SMD] Normalization"
    )

    print(
        "      Scaler fitted : TRAIN ONLY"
    )

    print(
        "      Test data transformed "
        "using training statistics."
    )

    return {
        "train": train_raw,
        "test": test_raw,
        "test_labels": test_labels,

        "train_norm": train_norm,
        "test_norm": test_norm,

        "train_windows": train_windows,
        "test_windows": test_windows,

        "train_starts": train_starts,
        "test_starts": test_starts,

        "test_window_labels": test_window_labels,

        "n_features": n_features,
    }


# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================

def smooth_scores(
    scores: np.ndarray,
    window: int = SMOOTHING_WINDOW,
):
    """
    Moving average smoothing.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if window <= 1:
        return scores.copy()

    if len(scores) < window:
        return scores.copy()

    kernel = (
        np.ones(
            window,
            dtype=np.float64,
        )
        / window
    )

    pad_left = window // 2

    pad_right = (
        window
        - 1
        - pad_left
    )

    padded = np.pad(
        scores,
        (
            pad_left,
            pad_right,
        ),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


# ============================================================================
# WINDOW → POINT SCORE AGGREGATION
# ============================================================================

def windows_to_point_scores(
    window_scores: np.ndarray,
    n_timesteps: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):
    """
    Convert window-level scores into point-level scores.

    Every timestep receives the mean score of all windows
    covering that timestep.
    """

    score_sum = np.zeros(
        n_timesteps,
        dtype=np.float64,
    )

    score_count = np.zeros(
        n_timesteps,
        dtype=np.float64,
    )

    for i, score in enumerate(
        window_scores
    ):

        start = i * stride

        end = min(
            start + window_size,
            n_timesteps,
        )

        if start >= n_timesteps:
            break

        score_sum[
            start:end
        ] += float(score)

        score_count[
            start:end
        ] += 1.0

    point_scores = np.zeros(
        n_timesteps,
        dtype=np.float64,
    )

    valid = score_count > 0

    point_scores[valid] = (
        score_sum[valid]
        / score_count[valid]
    )

    return point_scores


# ============================================================================
# MODEL SCORE CALCULATION
# ============================================================================

def calculate_model_scores(
    model,
    windows,
    device,
):

    model.eval()

    scores = []

    print(
        f"[FGEAD] Calculating "
        f"{len(windows)} window scores..."
    )

    with torch.no_grad():

        for i in range(
            len(windows)
        ):

            x = torch.tensor(
                windows[
                    i:i + 1
                ],
                dtype=torch.float32,
                device=device,
            )

            (
                _,
                _,
                anomaly_scores,
            ) = model(x)

            score = float(
                anomaly_scores
                .max()
                .detach()
                .cpu()
            )

            scores.append(
                score
            )

            if (
                len(windows) >= 1000
                and (i + 1) % 1000 == 0
            ):

                print(
                    f"     {i + 1}/"
                    f"{len(windows)}"
                )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


# ============================================================================
# TRAIN-ONLY THRESHOLD
# ============================================================================

def choose_threshold_from_train(
    calibration_scores,
):
    """
    Threshold is calculated exclusively from
    training/calibration scores.

    Test labels are NEVER used.
    """

    print()
    print(
        "=" * 70
    )
    print(
        "THRESHOLD CALIBRATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Calibration windows      : "
        f"{len(calibration_scores)}"
    )

    print()
    print(
        "Calibration score statistics:"
    )

    print(
        f"     Mean : "
        f"{calibration_scores.mean():.6f}"
    )

    print(
        f"     Std  : "
        f"{calibration_scores.std():.6f}"
    )

    print(
        f"     Max  : "
        f"{calibration_scores.max():.6f}"
    )

    threshold = float(
        np.percentile(
            calibration_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    print()
    print(
        f"     {THRESHOLD_PERCENTILE:.2f}th "
        f"percentile threshold: "
        f"{threshold:.6f}"
    )

    return {
        "percentile":
            THRESHOLD_PERCENTILE,
        "threshold":
            threshold,
    }


# ============================================================================
# ANOMALY RANGE DETECTION
# ============================================================================

def find_ranges(
    binary_scores: np.ndarray,
    absolute_start: int = 0,
    min_duration: int = MIN_ANOMALY_DURATION,
    merge_gap: int = MERGE_GAP,
):
    """
    Find continuous anomaly regions.

    Short regions are removed.
    Nearby regions are merged.
    """

    binary_scores = (
        np.asarray(
            binary_scores
        )
        .astype(np.int32)
    )

    padded = np.concatenate(
        [
            [0],
            binary_scores,
            [0],
        ]
    )

    diff = np.diff(
        padded
    )

    starts = np.where(
        diff == 1
    )[0]

    ends = (
        np.where(
            diff == -1
        )[0]
        - 1
    )

    raw_ranges = []

    for start, end in zip(
        starts,
        ends,
    ):

        duration = (
            end
            - start
            + 1
        )

        if duration < min_duration:
            continue

        raw_ranges.append(
            [
                int(start),
                int(end),
            ]
        )

    merged = []

    for start, end in raw_ranges:

        if not merged:

            merged.append(
                [start, end]
            )

            continue

        previous_start, previous_end = (
            merged[-1]
        )

        gap = (
            start
            - previous_end
            - 1
        )

        if gap <= merge_gap:

            merged[-1][1] = end

        else:

            merged.append(
                [start, end]
            )

    result = []

    for start, end in merged:

        duration = (
            end
            - start
            + 1
        )

        result.append(
            {
                "start":
                    absolute_start
                    + start,

                "end":
                    absolute_start
                    + end,

                "duration":
                    duration,
            }
        )

    return result


# ============================================================================
# EXPLAINABILITY
# ============================================================================

def run_explainability(
    model,
    test_windows,
    test_scores,
    absolute_test_start,
    feature_names,
    device,
    train_raw,
):

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL EXPLAINABILITY DEMO"
    )
    print(
        "=" * 70
    )

    # Select the highest-scoring test window.
    selected_idx = int(
        np.argmax(
            test_scores
        )
    )

    # IMPORTANT:
    # The selected window starts at:
    #
    # absolute test start
    # +
    # window index × stride
    #
    window_start_abs = (
        absolute_test_start
        + selected_idx * STRIDE
    )

    window_end_abs = (
        window_start_abs
        + WINDOW_SIZE
        - 1
    )

    print()
    print(
        f"Selected test window : "
        f"{selected_idx}"
    )

    print(
        f"Absolute start       : "
        f"t={window_start_abs}"
    )

    print(
        f"Absolute end         : "
        f"t={window_end_abs}"
    )

    # ------------------------------------------------------------
    # TRAIN-ONLY DISPLAY SCALER
    # ------------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "EXPLAINABILITY SCALE PREPARATION"
    )
    print(
        "=" * 70
    )

    print(
        "[OK] Explanation scaler fitted "
        "using TRAIN data only."
    )

    print(
        "     Actual and predicted values "
        "will be displayed in original SMD scale."
    )

    display_scaler = (
        StandardScaler()
    )

    display_scaler.fit(
        np.asarray(
            train_raw,
            dtype=np.float32,
        )
    )

    # ------------------------------------------------------------
    # MODEL WINDOW
    # ------------------------------------------------------------

    window = torch.tensor(
        test_windows[
            selected_idx:
            selected_idx + 1
        ],
        dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------
    # EXPLAINER
    # ------------------------------------------------------------

    feature_names = list(
        feature_names
    )

    explainer = FGEADExplainer(
        model=model,
        feature_names=feature_names,
        scaler=display_scaler,
        top_k=5,
        timestep_seconds=TIMESTEP_SECONDS,
    )

    report = explainer.explain(
        window,
        window_start_abs=window_start_abs,
        device=device,
    )

    # ------------------------------------------------------------
    # ORIGINAL SCALE CONVERSION
    # ------------------------------------------------------------

    try:

        feat = report[
            "Q1_Q2_feature_analysis"
        ]

        peak_t = int(
            feat[
                "peak_timestep"
            ]
        )

        # The explainer uses:
        #
        # actual at peak_t
        # prediction at peak_t - 1
        #
        actual_norm = (
            test_windows[
                selected_idx,
                peak_t,
                :
            ]
        )

        with torch.no_grad():

            predictions, _, _ = model(
                window
            )

        pred_norm = (
            predictions[
                0,
                peak_t - 1,
                :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        actual_raw = (
            display_scaler
            .inverse_transform(
                actual_norm.reshape(
                    1,
                    -1,
                )
            )[0]
        )

        pred_raw = (
            display_scaler
            .inverse_transform(
                pred_norm.reshape(
                    1,
                    -1,
                )
            )[0]
        )

        for item in feat[
            "top_features"
        ]:

            name = item[
                "feature"
            ]

            try:

                idx = feature_names.index(
                    name
                )

            except ValueError:

                continue

            item[
                "actual"
            ] = round(
                float(
                    actual_raw[idx]
                ),
                4,
            )

            item[
                "predicted"
            ] = round(
                float(
                    pred_raw[idx]
                ),
                4,
            )

            item[
                "deviation"
            ] = round(
                abs(
                    actual_raw[idx]
                    - pred_raw[idx]
                ),
                4,
            )

            item[
                "direction"
            ] = (
                "↑ spike"
                if actual_raw[idx]
                > pred_raw[idx]
                else "↓ drop"
            )

    except Exception as exc:

        print()
        print(
            "[WARNING] Original-scale "
            f"display conversion failed: {exc}"
        )

    print()
    print(
        explainer.format_report(
            report
        )
    )

    return report


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(
    machine,
    n_features,
    train_windows,
    test_windows,
    test_raw,
    test_labels,
    test_window_labels,
    best_threshold,
    threshold,
    window_precision,
    window_recall,
    window_f1,
    window_auc,
    window_pr_auc,
    point_precision,
    point_recall,
    point_f1,
    point_auc,
    point_pr_auc,
    point_predictions,
    ranges,
    calibration_scores,
):
    os.makedirs(
        "plots",
        exist_ok=True,
    )

    output_file = (
        Path("plots")
        / (
            f"smd_{machine}"
            "_FINAL_results.txt"
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FGEAD FINAL SMD EVALUATION\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Machine: {machine}\n"
        )

        f.write(
            f"Features: {n_features}\n"
        )

        f.write(
            f"Window size: "
            f"{WINDOW_SIZE}\n"
        )

        f.write(
            f"Window stride: "
            f"{STRIDE}\n"
        )

        f.write(
            f"Train windows: "
            f"{len(train_windows)}\n"
        )

        f.write(
            f"Test windows: "
            f"{len(test_windows)}\n"
        )

        f.write(
            f"Test timesteps: "
            f"{len(test_raw)}\n"
        )

        f.write(
            f"Test anomaly windows: "
            f"{int(test_window_labels.sum())}\n"
        )

        f.write(
            f"Test anomaly points: "
            f"{int(np.asarray(test_labels).sum())}\n\n"
        )

        f.write(
            "TRAIN-ONLY THRESHOLD\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Calibration windows: "
            f"{len(calibration_scores)}\n"
        )

        f.write(
            f"Percentile: "
            f"{best_threshold['percentile']:.2f}%\n"
        )

        f.write(
            f"Threshold: "
            f"{threshold:.6f}\n\n"
        )

        f.write(
            "WINDOW LEVEL\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Precision: "
            f"{window_precision:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{window_recall:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{window_f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{window_auc:.4f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{window_pr_auc:.4f}\n\n"
        )

        f.write(
            "POINT LEVEL\n"
        )

        f.write(
            "-" * 70
            + "\n"
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
            f"{point_auc:.4f}\n"
        )

        f.write(
            f"PR-AUC: "
            f"{point_pr_auc:.4f}\n"
        )

        f.write(
            f"True anomaly timesteps: "
            f"{int(np.asarray(test_labels).sum())}\n"
        )

        f.write(
            f"Detected anomaly timesteps: "
            f"{int(point_predictions.sum())}\n\n"
        )

        f.write(
            "CALIBRATION STATISTICS\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        f.write(
            f"Mean: "
            f"{calibration_scores.mean():.6f}\n"
        )

        f.write(
            f"Std: "
            f"{calibration_scores.std():.6f}\n"
        )

        f.write(
            f"Max: "
            f"{calibration_scores.max():.6f}\n\n"
        )

        f.write(
            "DETECTED RANGES\n"
        )

        f.write(
            "-" * 70
            + "\n"
        )

        for r in ranges:

            f.write(
                f"t={r['start']} -> "
                f"t={r['end']} "
                f"duration={r['duration']}\n"
            )

    return output_file


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Final FGEAD evaluation "
            "on SMD."
        )
    )

    parser.add_argument(
        "--machine",
        default="1-1",
        help=(
            "SMD machine identifier, "
            "for example 1-1"
        ),
    )

    args = parser.parse_args()

    machine = args.machine

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FGEAD — FINAL SMD EVALUATION"
    )
    print(
        "=" * 70
    )

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    data = load_smd_machine(
        machine
    )

    train_windows = data[
        "train_windows"
    ]

    test_windows = data[
        "test_windows"
    ]

    train_raw = data[
        "train"
    ]

    test_raw = data[
        "test"
    ]

    test_labels = data[
        "test_labels"
    ]

    test_window_labels = data[
        "test_window_labels"
    ]

    n_features = (
        train_windows.shape[-1]
    )

    print()
    print(
        "=" * 70
    )
    print(
        "DATASET INFORMATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Total test timesteps : "
        f"{len(test_raw)}"
    )

    print(
        f"Features             : "
        f"{n_features}"
    )

    print(
        f"Window size          : "
        f"{WINDOW_SIZE}"
    )

    print(
        f"Window stride        : "
        f"{STRIDE}"
    )

    print(
        f"Train windows        : "
        f"{len(train_windows)}"
    )

    print(
        f"Test windows         : "
        f"{len(test_windows)}"
    )

    print(
        f"Test anomaly windows : "
        f"{int(test_window_labels.sum())}"
    )

    print(
        f"Test anomaly points  : "
        f"{int(np.asarray(test_labels).sum())}"
    )

    # SMD test sequence starts at 0.
    absolute_test_start = 0

    # ========================================================================
    # CHECKPOINT
    # ========================================================================

    checkpoint = (
        Path("checkpoints")
        / (
            "fgead_smd_machine_"
            f"{machine.replace('-', '_')}.pt"
        )
    )

    print()
    print(
        "[FGEAD] Checkpoint:"
    )

    print(
        f"         {checkpoint}"
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{checkpoint}\n\n"
            "Train first using:\n"
            f"python train_smd.py "
            f"--machine {machine}"
        )

    # ========================================================================
    # MODEL
    # ========================================================================

    model = FGEAD(
        n_features=n_features
    ).to(device)

    # train_smd.py saves a complete checkpoint dictionary.
    # Extract the actual model weights before loading them.
    try:
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        state = torch.load(
            checkpoint,
            map_location=device,
        )

    if isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]

        print(
            f"     Checkpoint machine : {state.get('machine', 'unknown')}"
        )
        print(
            f"     Checkpoint features: {state.get('n_features', 'unknown')}"
        )
        print(
            f"     Checkpoint loss    : {state.get('best_loss', 'unknown')}"
        )
    else:
        # Compatibility with a raw state_dict checkpoint.
        model_state = state

    if not isinstance(model_state, dict):
        raise RuntimeError(
            "Invalid FGEAD checkpoint format. Expected a raw state_dict "
            "or a checkpoint containing 'model_state_dict'."
        )

    model.load_state_dict(
        model_state,
        strict=True,
    )

    model.eval()

    print(
        "[OK] Model loaded successfully."
    )

    # ========================================================================
    # TRAIN-ONLY CALIBRATION
    # ========================================================================

    print()
    print(
        "=" * 70
    )
    print(
        "THRESHOLD CALIBRATION"
    )
    print(
        "=" * 70
    )

    calibration_count = int(
        len(train_windows)
        * CALIBRATION_RATIO
    )

    calibration_count = max(
        calibration_count,
        1,
    )

    calibration_windows = (
        train_windows[
            -calibration_count:
        ]
    )

    print(
        f"Training windows      : "
        f"{len(train_windows)}"
    )

    print(
        f"Calibration windows   : "
        f"{len(calibration_windows)}"
    )

    print(
        f"Calibration ratio     : "
        f"{CALIBRATION_RATIO:.0%}"
    )

    calibration_scores = (
        calculate_model_scores(
            model,
            calibration_windows,
            device,
        )
    )

    print()
    print(
        "Calibration score statistics:"
    )

    print(
        f"     Mean : "
        f"{calibration_scores.mean():.6f}"
    )

    print(
        f"     Std  : "
        f"{calibration_scores.std():.6f}"
    )

    print(
        f"     Max  : "
        f"{calibration_scores.max():.6f}"
    )

    best_threshold = (
        choose_threshold_from_train(
            calibration_scores
        )
    )

    threshold = (
        best_threshold[
            "threshold"
        ]
    )

    # ========================================================================
    # TEST SCORES
    # ========================================================================

    test_scores = (
        calculate_model_scores(
            model,
            test_windows,
            device,
        )
    )

    # ========================================================================
    # WINDOW LEVEL
    # ========================================================================

    window_predictions = (
        test_scores >= threshold
    ).astype(
        np.int32
    )

    window_precision = (
        precision_score(
            test_window_labels,
            window_predictions,
            zero_division=0,
        )
    )

    window_recall = (
        recall_score(
            test_window_labels,
            window_predictions,
            zero_division=0,
        )
    )

    window_f1 = (
        f1_score(
            test_window_labels,
            window_predictions,
            zero_division=0,
        )
    )

    window_auc = safe_auc(
        test_window_labels,
        test_scores,
    )

    window_pr_auc = safe_pr_auc(
        test_window_labels,
        test_scores,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL SMD WINDOW-LEVEL RESULTS"
    )
    print(
        "=" * 70
    )

    print(
        f"Threshold : "
        f"{threshold:.6f}"
    )

    print(
        f"Precision : "
        f"{window_precision:.3f}"
    )

    print(
        f"Recall    : "
        f"{window_recall:.3f}"
    )

    print(
        f"F1        : "
        f"{window_f1:.3f}"
    )

    print(
        f"ROC-AUC   : "
        f"{window_auc:.3f}"
    )

    print(
        f"PR-AUC    : "
        f"{window_pr_auc:.3f}"
    )

    print(
        f"Ground-truth anomaly windows: "
        f"{int(test_window_labels.sum())}"
    )

    print(
        f"Detected anomaly windows: "
        f"{int(window_predictions.sum())}"
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            test_window_labels,
            window_predictions,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            test_window_labels,
            window_predictions,
        )
    )

    # ========================================================================
    # POINT LEVEL
    # ========================================================================

    print()
    print(
        "=" * 70
    )
    print(
        "POINT-LEVEL SCORE AGGREGATION"
    )
    print(
        "=" * 70
    )

    print(
        "[1/4] Aggregating overlapping "
        "windows using MEAN..."
    )

    raw_point_scores = (
        windows_to_point_scores(
            test_scores,
            len(test_raw),
            WINDOW_SIZE,
            STRIDE,
        )
    )

    print(
        "[2/4] Applying temporal smoothing..."
    )

    point_scores = smooth_scores(
        raw_point_scores,
        SMOOTHING_WINDOW,
    )

    print(
        "[3/4] Applying anomaly threshold..."
    )

    point_predictions = (
        point_scores >= threshold
    ).astype(
        np.int32
    )

    print(
        "[4/4] Removing short anomaly "
        "segments and merging close ranges..."
    )

    point_precision = (
        precision_score(
            test_labels,
            point_predictions,
            zero_division=0,
        )
    )

    point_recall = (
        recall_score(
            test_labels,
            point_predictions,
            zero_division=0,
        )
    )

    point_f1 = (
        f1_score(
            test_labels,
            point_predictions,
            zero_division=0,
        )
    )

    point_auc = safe_auc(
        test_labels,
        point_scores,
    )

    point_pr_auc = safe_pr_auc(
        test_labels,
        point_scores,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL SMD POINT-LEVEL RESULTS"
    )
    print(
        "=" * 70
    )

    print(
        f"Threshold : "
        f"{threshold:.6f}"
    )

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
        f"{point_auc:.3f}"
    )

    print(
        f"PR-AUC    : "
        f"{point_pr_auc:.3f}"
    )

    print(
        f"\nTrue anomaly timesteps: "
        f"{int(np.asarray(test_labels).sum())}"
    )

    print(
        f"Detected anomaly timesteps: "
        f"{int(point_predictions.sum())}"
    )

    # ========================================================================
    # RANGES
    # ========================================================================

    ranges = find_ranges(
        point_predictions,
        absolute_start=absolute_test_start,
        min_duration=MIN_ANOMALY_DURATION,
        merge_gap=MERGE_GAP,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL DETECTED ANOMALY RANGES"
    )
    print(
        "=" * 70
    )

    print(
        f"Detected regions: "
        f"{len(ranges)}"
    )

    for r in ranges:

        print(
            f"   t={r['start']:6d} "
            f"→ t={r['end']:6d} "
            f"duration={r['duration']} steps"
        )

    # ========================================================================
    # EXPLAINABILITY
    # ========================================================================

    feature_names = [
        f"feature_{i:02d}"
        for i in range(
            n_features
        )
    ]

    try:

        run_explainability(
            model=model,
            test_windows=test_windows,
            test_scores=test_scores,
            absolute_test_start=absolute_test_start,
            feature_names=feature_names,
            device=device,
            train_raw=train_raw,
        )

    except Exception as exc:

        print()
        print(
            "[WARNING] Explainability "
            "section failed:"
        )

        print(
            f"          {exc}"
        )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    output_file = save_results(
        machine=machine,
        n_features=n_features,
        train_windows=train_windows,
        test_windows=test_windows,
        test_raw=test_raw,
        test_labels=test_labels,
        test_window_labels=test_window_labels,
        best_threshold=best_threshold,
        threshold=threshold,
        window_precision=window_precision,
        window_recall=window_recall,
        window_f1=window_f1,
        window_auc=window_auc,
        window_pr_auc=window_pr_auc,
        point_precision=point_precision,
        point_recall=point_recall,
        point_f1=point_f1,
        point_auc=point_auc,
        point_pr_auc=point_pr_auc,
        point_predictions=point_predictions,
        ranges=ranges,
        calibration_scores=calibration_scores,
    )

    print()
    print(
        "[✓] Final results saved:"
    )

    print(
        f"    {output_file}"
    )

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL SMD EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
    )

    main()