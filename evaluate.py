"""
evaluate.py

Complete FGEAD evaluation pipeline.

Includes:
    1. FGEAD evaluation
    2. Isolation Forest baseline
    3. LSTM Autoencoder baseline
    4. Precision / Recall / F1 / ROC-AUC
    5. Validation threshold calibration
    6. Baseline comparison
    7. Explainability demonstration
    8. Correct absolute timestep mapping

Important:
    The test set starts at 85% of the original time series.
    Windows use:
        window_size = 60
        stride      = 5

    Therefore, an anomaly window index is converted to an
    absolute timestep using:

        absolute_start = test_start_abs + window_index * stride
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

from torch.utils.data import DataLoader

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from data.preprocessor import TimeSeriesPreprocessor, TimeSeriesDataset
from models.fgead import FGEAD
from models.explainer import FGEADExplainer


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_SIZE = 60
STRIDE = 5

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CHECKPOINT_PATH = "checkpoints/best_model.pt"


FEATURE_NAMES = [
    "cpu_usage",
    "cpu_temp",
    "memory_usage",
    "memory_free",
    "disk_io_read",
    "disk_io_write",
    "net_in",
    "net_out",
    "process_count",
    "context_switches",
    "cache_hits",
    "cache_misses",
    "load_avg_1m",
    "load_avg_5m",
    "load_avg_15m",
    "swap_usage",
    "iowait",
    "kernel_threads",
    "open_files",
    "network_errors",
]


# =============================================================================
# HELPERS
# =============================================================================

def best_threshold_f1(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Find the threshold that maximizes F1 score on validation data.
    """

    best_f1 = 0.0
    best_thr = 0.5

    if len(scores) == 0:
        return best_thr

    score_min = float(scores.min())
    score_max = float(scores.max())

    if score_min == score_max:
        return score_min

    for thr in np.linspace(score_min, score_max, 200):

        preds = (scores >= thr).astype(int)

        if preds.sum() == 0:
            continue

        f1 = f1_score(
            labels,
            preds,
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def compute_metrics(
    y_true,
    y_pred,
    y_scores=None,
):
    """
    Calculate classification metrics.
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    if (
        y_scores is not None
        and len(np.unique(y_true)) > 1
    ):
        auc = roc_auc_score(
            y_true,
            y_scores,
        )
    else:
        auc = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


# =============================================================================
# LOAD DATA
# =============================================================================

def load_test_data():
    """
    Load dataset and create chronological train/validation/test splits.

    Returns:
        train_norm
        train_win_clean
        val_win
        val_lbl
        test_win
        test_lbl
        n_features
        processor
        test_start_abs
    """

    proc = TimeSeriesPreprocessor(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    data, labels = proc.load_csv(
        "data/synthetic_data.csv"
    )

    data = proc.handle_missing(data)

    data, labels = proc.remove_duplicates(
        data,
        labels,
    )

    n = len(data)

    # -------------------------------------------------------------------------
    # Absolute split positions
    # -------------------------------------------------------------------------

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    test_start_abs = val_end

    # -------------------------------------------------------------------------
    # Split raw time series
    # -------------------------------------------------------------------------

    train_raw = data[:train_end]
    val_raw = data[train_end:val_end]
    test_raw = data[val_end:]

    train_labels = labels[:train_end]
    val_labels = labels[train_end:val_end]
    test_labels = labels[val_end:]

    # -------------------------------------------------------------------------
    # Normalize
    #
    # IMPORTANT:
    # scaler is fitted ONLY on training data.
    # -------------------------------------------------------------------------

    train_norm, val_norm, test_norm = proc.normalize(
        train_raw,
        val_raw,
        test_raw,
    )

    # -------------------------------------------------------------------------
    # Create sliding windows
    # -------------------------------------------------------------------------

    train_win, train_lbl = proc.create_windows(
        train_norm,
        train_labels,
    )

    val_win, val_lbl = proc.create_windows(
        val_norm,
        val_labels,
    )

    test_win, test_lbl = proc.create_windows(
        test_norm,
        test_labels,
    )

    # -------------------------------------------------------------------------
    # Training uses only normal windows
    # -------------------------------------------------------------------------

    normal_mask = train_lbl == 0

    train_win_clean = train_win[
        normal_mask
    ]

    print(
        f"[OK] Dataset split:"
    )

    print(
        f"     Total timesteps : {n}"
    )

    print(
        f"     Train           : 0 - {train_end - 1}"
    )

    print(
        f"     Validation      : {train_end} - {val_end - 1}"
    )

    print(
        f"     Test            : {test_start_abs} - {n - 1}"
    )

    print(
        f"     Test start abs  : {test_start_abs}"
    )

    print(
        f"     Window size     : {WINDOW_SIZE}"
    )

    print(
        f"     Window stride   : {STRIDE}"
    )

    return (
        train_norm,
        train_win_clean,
        val_win,
        val_lbl,
        test_win,
        test_lbl,
        data.shape[1],
        proc,
        test_start_abs,
    )


# =============================================================================
# FGEAD EVALUATION
# =============================================================================

def eval_fgead(
    val_win,
    val_lbl,
    test_win,
    test_lbl,
    n_features,
):
    """
    Evaluate the proposed FGEAD model.
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\n[FGEAD] Device: {device}"
    )

    model = FGEAD(
        n_features=n_features
    ).to(device)

    if not os.path.exists(
        CHECKPOINT_PATH
    ):
        print(
            f"[!] Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

        print(
            "[!] Run python train.py first."
        )

        return None, None, None

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    def get_scores(windows):

        scores = []

        with torch.no_grad():

            for i in range(
                len(windows)
            ):

                x = torch.FloatTensor(
                    windows[i:i + 1]
                ).to(device)

                _, _, anomaly_scores = model(
                    x
                )

                # Maximum anomaly score inside
                # the complete window.
                score = float(
                    anomaly_scores.max().cpu()
                )

                scores.append(score)

        return np.asarray(
            scores,
            dtype=np.float32,
        )

    # -------------------------------------------------------------------------
    # Validation and test scores
    # -------------------------------------------------------------------------

    val_scores = get_scores(
        val_win
    )

    test_scores = get_scores(
        test_win
    )

    # -------------------------------------------------------------------------
    # Threshold calibration
    # -------------------------------------------------------------------------

    threshold = best_threshold_f1(
        val_scores,
        val_lbl,
    )

    # Small safety margin to reduce false positives.
    threshold = threshold * 1.05

    test_preds = (
        test_scores >= threshold
    ).astype(int)

    metrics = compute_metrics(
        test_lbl,
        test_preds,
        test_scores,
    )

    print(
        f"\n[FGEAD] threshold={threshold:.4f}"
    )

    print(
        classification_report(
            test_lbl,
            test_preds,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    print(
        f"[FGEAD] Precision : "
        f"{metrics['precision']:.3f}"
    )

    print(
        f"[FGEAD] Recall    : "
        f"{metrics['recall']:.3f}"
    )

    print(
        f"[FGEAD] F1        : "
        f"{metrics['f1']:.3f}"
    )

    print(
        f"[FGEAD] ROC-AUC   : "
        f"{metrics['auc']:.3f}"
    )

    return (
        metrics,
        model,
        threshold,
    )


# =============================================================================
# ISOLATION FOREST BASELINE
# =============================================================================

def eval_isolation_forest(
    train_norm,
    test_win,
    test_lbl,
):
    """
    Isolation Forest baseline.

    Training data consists only of normal training
    windows.

    This is a baseline, not the proposed model.
    """

    print(
        "\n[Isolation Forest]"
    )

    window_size = test_win.shape[1]

    train_windows = []

    for i in range(
        0,
        len(train_norm) - window_size + 1,
        STRIDE,
    ):

        train_windows.append(
            train_norm[
                i:i + window_size
            ]
        )

    train_windows = np.asarray(
        train_windows,
        dtype=np.float32,
    )

    if len(train_windows) == 0:
        print(
            "[!] Not enough training data "
            "for Isolation Forest."
        )
        return None

    # Flatten:
    #
    # (windows, time, features)
    #
    # ->
    #
    # (windows, time * features)

    train_flat = train_windows.reshape(
        train_windows.shape[0],
        -1,
    )

    test_flat = test_win.reshape(
        test_win.shape[0],
        -1,
    )

    clf = IsolationForest(
        contamination="auto",
        random_state=42,
        n_estimators=200,
    )

    clf.fit(
        train_flat
    )

    raw_preds = clf.predict(
        test_flat
    )

    preds = (
        raw_preds == -1
    ).astype(int)

    scores = -clf.score_samples(
        test_flat
    )

    metrics = compute_metrics(
        test_lbl,
        preds,
        scores,
    )

    print(
        classification_report(
            test_lbl,
            preds,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    return metrics


# =============================================================================
# LSTM AUTOENCODER BASELINE
# =============================================================================

class LSTMAutoencoder(
    torch.nn.Module
):
    """
    Simple LSTM Autoencoder baseline.
    """

    def __init__(
        self,
        n_features,
        hidden=64,
    ):

        super().__init__()

        self.encoder = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )

        self.decoder = torch.nn.LSTM(
            input_size=hidden,
            hidden_size=n_features,
            batch_first=True,
        )

    def forward(
        self,
        x,
    ):

        _, (hidden, _) = self.encoder(
            x
        )

        context = (
            hidden[-1]
            .unsqueeze(1)
            .expand(
                -1,
                x.size(1),
                -1,
            )
        )

        decoded, _ = self.decoder(
            context
        )

        return decoded


def eval_lstm_ae(
    train_win,
    val_win,
    val_lbl,
    test_win,
    test_lbl,
    n_features,
):
    """
    Evaluate LSTM Autoencoder baseline.
    """

    print(
        "\n[LSTM Autoencoder]"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = LSTMAutoencoder(
        n_features
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.MSELoss()

    dataset = TimeSeriesDataset(
        train_win
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
    )

    model.train()

    for epoch in range(10):

        epoch_loss = 0.0

        for x in loader:

            x = x.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output,
                x,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        if (
            epoch == 0
            or epoch == 4
            or epoch == 9
        ):

            print(
                f"   Epoch "
                f"{epoch + 1:02d}/10 "
                f"loss="
                f"{epoch_loss / max(len(loader), 1):.6f}"
            )

    model.eval()

    def get_scores(windows):

        scores = []

        with torch.no_grad():

            for i in range(
                len(windows)
            ):

                x = torch.FloatTensor(
                    windows[i:i + 1]
                ).to(device)

                output = model(x)

                error = float(
                    torch.abs(
                        output - x
                    ).mean().cpu()
                )

                scores.append(error)

        return np.asarray(
            scores,
            dtype=np.float32,
        )

    val_scores = get_scores(
        val_win
    )

    test_scores = get_scores(
        test_win
    )

    threshold = best_threshold_f1(
        val_scores,
        val_lbl,
    )

    test_preds = (
        test_scores >= threshold
    ).astype(int)

    metrics = compute_metrics(
        test_lbl,
        test_preds,
        test_scores,
    )

    print(
        f"\n[LSTM Autoencoder] "
        f"threshold={threshold:.4f}"
    )

    print(
        classification_report(
            test_lbl,
            test_preds,
            labels=[0, 1],
            target_names=[
                "Normal",
                "Anomaly",
            ],
            zero_division=0,
        )
    )

    return metrics


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_comparison(
    results: dict
):
    """
    Print final model comparison.
    """

    print(
        "\n"
        + "=" * 75
    )

    print(
        "  BASELINE COMPARISON"
    )

    print(
        "=" * 75
    )

    print(
        f"{'Model':<25}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'AUC':>10}"
    )

    print(
        "-" * 75
    )

    for name, metrics in results.items():

        if metrics is None:
            continue

        print(
            f"{name:<25}"
            f"{metrics['precision']:>12.3f}"
            f"{metrics['recall']:>10.3f}"
            f"{metrics['f1']:>10.3f}"
            f"{metrics['auc']:>10.3f}"
        )

    print(
        "=" * 75
    )


# =============================================================================
# EXPLAINABILITY DEMO
# =============================================================================

def demo_explanation(
    model,
    test_win,
    test_lbl,
    n_features,
    test_start_abs,
):
    """
    Demonstrate the explainability layer.

    IMPORTANT:
        test_start_abs is the actual position of the test
        set in the original time series.

    The anomaly window index is converted to an absolute
    timestep using:

        absolute_start =
            test_start_abs + index * STRIDE
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "  EXPLAINABILITY DEMO "
        "(Enhancement 7.2)"
    )

    print(
        "=" * 60
    )

    # -------------------------------------------------------------------------
    # Feature names
    # -------------------------------------------------------------------------

    feature_names = FEATURE_NAMES[
        :n_features
    ]

    # -------------------------------------------------------------------------
    # Create explainer
    # -------------------------------------------------------------------------

    explainer = FGEADExplainer(
        model,
        feature_names,
        top_k=5,
        timestep_seconds=60,
    )

    # -------------------------------------------------------------------------
    # Find a true anomaly window
    # -------------------------------------------------------------------------

    anomaly_idxs = np.where(
        test_lbl == 1
    )[0]

    if len(anomaly_idxs) == 0:

        print(
            "[!] No anomaly windows "
            "found in test set."
        )

        return

    # Use first true anomaly window
    idx = int(
        anomaly_idxs[0]
    )

    # -------------------------------------------------------------------------
    # IMPORTANT TIMESTEP FIX
    # -------------------------------------------------------------------------

    # Position of this window inside the
    # original full dataset.

    window_start_abs = (
        test_start_abs
        + idx * STRIDE
    )

    # -------------------------------------------------------------------------
    # Extract window
    # -------------------------------------------------------------------------

    window = torch.FloatTensor(
        test_win[
            idx:idx + 1
        ]
    )

    # -------------------------------------------------------------------------
    # Generate explanation
    # -------------------------------------------------------------------------

    report = explainer.explain(
        window,
        window_start_abs=window_start_abs,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    # -------------------------------------------------------------------------
    # Print report
    # -------------------------------------------------------------------------

    print(
        explainer.format_report(
            report
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "[FGEAD Evaluation]"
    )

    # -------------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------------

    (
        train_norm,
        train_win,
        val_win,
        val_lbl,
        test_win,
        test_lbl,
        n_features,
        proc,
        test_start_abs,
    ) = load_test_data()

    # -------------------------------------------------------------------------
    # Dataset debug
    # -------------------------------------------------------------------------

    print(
        "\n=== DATASET SPLIT DEBUG ==="
    )

    print(
        "Train windows:",
        len(train_win),
    )

    print(
        "Val windows:",
        len(val_win),
    )

    print(
        "Test windows:",
        len(test_win),
    )

    print(
        "Val anomalies:",
        int(val_lbl.sum()),
    )

    print(
        "Test anomalies:",
        int(test_lbl.sum()),
    )

    print(
        "Test absolute start:",
        test_start_abs,
    )

    print(
        "Window size:",
        WINDOW_SIZE,
    )

    print(
        "Stride:",
        STRIDE,
    )

    print(
        "===========================\n"
    )

    # -------------------------------------------------------------------------
    # Results dictionary
    # -------------------------------------------------------------------------

    results = {}

    # -------------------------------------------------------------------------
    # 1. FGEAD
    # -------------------------------------------------------------------------

    (
        fgead_metrics,
        fgead_model,
        fgead_threshold,
    ) = eval_fgead(
        val_win,
        val_lbl,
        test_win,
        test_lbl,
        n_features,
    )

    results[
        "FGEAD (Proposed)"
    ] = fgead_metrics

    # -------------------------------------------------------------------------
    # 2. Isolation Forest
    # -------------------------------------------------------------------------

    isolation_metrics = (
        eval_isolation_forest(
            train_norm,
            test_win,
            test_lbl,
        )
    )

    results[
        "Isolation Forest"
    ] = isolation_metrics

    # -------------------------------------------------------------------------
    # 3. LSTM Autoencoder
    # -------------------------------------------------------------------------

    lstm_metrics = eval_lstm_ae(
        train_win,
        val_win,
        val_lbl,
        test_win,
        test_lbl,
        n_features,
    )

    results[
        "LSTM Autoencoder"
    ] = lstm_metrics

    # -------------------------------------------------------------------------
    # Final comparison
    # -------------------------------------------------------------------------

    print_comparison(
        results
    )

    # -------------------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------------------

    if fgead_model is not None:

        demo_explanation(
            fgead_model,
            test_win,
            test_lbl,
            n_features,
            test_start_abs,
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()