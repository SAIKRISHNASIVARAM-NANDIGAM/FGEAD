"""
data/smd_loader.py

FGEAD — Server Machine Dataset (SMD) Loader

Responsibilities:
    1. Load one SMD machine
    2. Load train/test telemetry
    3. Load test labels
    4. Fit StandardScaler ONLY on training data
    5. Normalize train and test consistently
    6. Create chronological sliding windows
    7. Preserve absolute test timesteps
    8. Provide inverse transformation for explainability
    9. Provide feature names
   10. Validate the complete dataset pipeline

Important:
    The FGEAD model works on normalized data.

    Explanations can use:
        normalized values -> model prediction
    OR
        inverse-transformed raw values -> human-readable explanation

Example:
    python data/smd_loader.py --machine 1-1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.preprocessing import StandardScaler


class SMDLoader:
    """
    Loader and preprocessing pipeline for the Server Machine Dataset.

    Example machine:
        machine-1-1
    """

    def __init__(
        self,
        root: str = "data/SMD",
        window_size: int = 60,
        stride: int = 5,
    ):
        self.root = Path(root)

        self.window_size = int(window_size)
        self.stride = int(stride)

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        if self.stride <= 0:
            raise ValueError("stride must be greater than 0")

        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"
        self.label_dir = self.root / "test_label"

        # IMPORTANT:
        # This scaler is fitted ONLY on training data.
        self.scaler = StandardScaler()
        self.fitted = False

        self.n_features: Optional[int] = None
        self.machine: Optional[str] = None

    # =====================================================================
    # FILE PATHS
    # =====================================================================

    def _machine_filename(self, machine: str) -> str:
        machine = str(machine).strip()

        if machine.startswith("machine-"):
            return f"{machine}.txt"

        return f"machine-{machine}.txt"

    def _get_paths(
        self,
        machine: str,
    ) -> Tuple[Path, Path, Path]:

        filename = self._machine_filename(machine)

        train_path = self.train_dir / filename
        test_path = self.test_dir / filename
        label_path = self.label_dir / filename

        return train_path, test_path, label_path

    # =====================================================================
    # LOAD TXT
    # =====================================================================

    @staticmethod
    def _load_txt(path: Path) -> np.ndarray:
        """
        Load an SMD comma-separated text file.

        Returns:
            float32 numpy array
        """

        if not path.exists():
            raise FileNotFoundError(
                f"\nSMD file not found:\n{path}\n"
            )

        try:
            data = np.loadtxt(
                path,
                delimiter=",",
                dtype=np.float32,
            )

        except Exception as exc:
            raise RuntimeError(
                f"\nCould not read SMD file:\n"
                f"{path}\n\n"
                f"Original error:\n{exc}"
            ) from exc

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.size == 0:
            raise ValueError(
                f"SMD file is empty:\n{path}"
            )

        if not np.isfinite(data).all():
            raise ValueError(
                f"SMD file contains NaN or infinite values:\n{path}"
            )

        return data.astype(np.float32)

    # =====================================================================
    # LOAD MACHINE
    # =====================================================================

    def load_machine(
        self,
        machine: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        train_path, test_path, label_path = self._get_paths(machine)

        train = self._load_txt(train_path)
        test = self._load_txt(test_path)
        labels = self._load_txt(label_path)

        # SMD labels normally contain one column.
        labels = labels.reshape(-1)

        # Convert labels to binary.
        labels = (labels > 0).astype(np.int64)

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------

        if len(test) != len(labels):
            raise ValueError(
                "\nTest data and labels have different lengths:\n"
                f"test   = {len(test)}\n"
                f"labels = {len(labels)}"
            )

        if train.shape[1] != test.shape[1]:
            raise ValueError(
                "\nTrain and test feature counts differ:\n"
                f"train = {train.shape[1]}\n"
                f"test  = {test.shape[1]}"
            )

        self.n_features = train.shape[1]
        self.machine = str(machine)

        print(
            f"[SMD] Machine {machine}"
        )

        print(
            f"      Train shape : {train.shape}"
        )

        print(
            f"      Test shape  : {test.shape}"
        )

        print(
            f"      Features    : {self.n_features}"
        )

        print(
            f"      Test anomalies: {int(labels.sum())}"
        )

        print(
            f"      Anomaly rate: {labels.mean():.2%}"
        )

        return train, test, labels

    # =====================================================================
    # NORMALIZATION
    # =====================================================================

    def normalize(
        self,
        train: np.ndarray,
        test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        """
        Normalize SMD data.

        CRITICAL:
            StandardScaler is fitted ONLY on training data.

        This prevents test-data information leakage.

        Returns:
            train_norm
            test_norm
        """

        if train.ndim != 2 or test.ndim != 2:
            raise ValueError(
                "Train and test data must be 2-dimensional."
            )

        if train.shape[1] != test.shape[1]:
            raise ValueError(
                "Train and test must contain the same number of features."
            )

        # ---------------------------------------------------------------
        # Fit ONLY on normal training telemetry.
        #
        # SMD training data is considered the normal reference set.
        # ---------------------------------------------------------------

        self.scaler.fit(train)

        train_norm = self.scaler.transform(train).astype(
            np.float32
        )

        test_norm = self.scaler.transform(test).astype(
            np.float32
        )

        self.fitted = True

        # Safety check
        if not np.isfinite(train_norm).all():
            raise ValueError(
                "Normalized training data contains NaN/Inf."
            )

        if not np.isfinite(test_norm).all():
            raise ValueError(
                "Normalized test data contains NaN/Inf."
            )

        return train_norm, test_norm

    # =====================================================================
    # NORMALIZED -> RAW
    # =====================================================================

    def inverse_transform(
        self,
        data: np.ndarray,
    ) -> np.ndarray:

        """
        Convert normalized values back to original SMD scale.

        This method is IMPORTANT for explainability.

        Example:

            normalized:
                3.72

            raw:
                68130.85

        The model should operate on normalized values,
        while human-readable explanations can use raw values.
        """

        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted yet. "
                "Call normalize() first."
            )

        original_shape = data.shape

        # ---------------------------------------------------------------
        # 2D: (N, M)
        # ---------------------------------------------------------------

        if data.ndim == 2:
            return self.scaler.inverse_transform(
                data
            ).astype(np.float32)

        # ---------------------------------------------------------------
        # 3D: (B, T, M)
        # ---------------------------------------------------------------

        if data.ndim == 3:
            B, T, M = data.shape

            flat = data.reshape(
                B * T,
                M
            )

            restored = self.scaler.inverse_transform(
                flat
            )

            return restored.reshape(
                B,
                T,
                M
            ).astype(np.float32)

        raise ValueError(
            "inverse_transform expects a 2D or 3D array. "
            f"Received shape={original_shape}"
        )

    # =====================================================================
    # SINGLE FEATURE TRANSFORMATION
    # =====================================================================

    def inverse_feature_value(
        self,
        value: float,
        feature_index: int,
    ) -> float:

        """
        Convert one normalized feature value to its original scale.
        """

        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        if self.n_features is None:
            raise RuntimeError(
                "Number of features is unknown."
            )

        if not 0 <= feature_index < self.n_features:
            raise IndexError(
                f"Invalid feature index: {feature_index}"
            )

        mean = float(
            self.scaler.mean_[feature_index]
        )

        scale = float(
            self.scaler.scale_[feature_index]
        )

        return float(
            value * scale + mean
        )

    # =====================================================================
    # FEATURE NAMES
    # =====================================================================

    def get_feature_names(self) -> list[str]:
        """
        Return stable feature names.

        SMD machine files do not contain column headers.
        Therefore the safest representation is feature_01 ... feature_N.
        """

        if self.n_features is None:
            raise RuntimeError(
                "Load a machine before requesting feature names."
            )

        return [
            f"feature_{i:02d}"
            for i in range(self.n_features)
        ]

    # =====================================================================
    # WINDOW CREATION
    # =====================================================================

    def create_windows(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        absolute_start: int = 0,
    ):
        """
        Create chronological sliding windows.

        Parameters
        ----------
        data:
            Normalized time series, shape (N, M)

        labels:
            Optional timestep labels, shape (N,)

        absolute_start:
            Absolute timestep represented by data[0].

        Returns
        -------
        With labels:
            windows
            window_labels
            window_starts

        Without labels:
            windows
            window_starts
        """

        if data.ndim != 2:
            raise ValueError(
                "data must have shape (timesteps, features)"
            )

        W = self.window_size
        S = self.stride
        n = len(data)

        if n < W:
            raise ValueError(
                f"Data length {n} is smaller than "
                f"window size {W}."
            )

        if labels is not None and len(labels) != n:
            raise ValueError(
                "Data and labels must have the same length."
            )

        windows = []
        window_labels = []
        window_starts = []

        for start in range(
            0,
            n - W + 1,
            S,
        ):

            end = start + W

            windows.append(
                data[start:end]
            )

            window_starts.append(
                int(absolute_start + start)
            )

            if labels is not None:
                window_labels.append(
                    int(
                        np.any(
                            labels[start:end] > 0
                        )
                    )
                )

        if not windows:
            raise ValueError(
                "No windows could be created."
            )

        windows = np.stack(
            windows,
            axis=0
        ).astype(np.float32)

        window_starts = np.asarray(
            window_starts,
            dtype=np.int64
        )

        if labels is not None:

            window_labels = np.asarray(
                window_labels,
                dtype=np.int64
            )

            return (
                windows,
                window_labels,
                window_starts,
            )

        return (
            windows,
            window_starts,
        )

    # =====================================================================
    # COMPLETE MACHINE PIPELINE
    # =====================================================================

    def prepare_machine(
        self,
        machine: str,
    ) -> Dict:

        train_raw, test_raw, test_labels_raw = (
            self.load_machine(machine)
        )

        # ---------------------------------------------------------------
        # NORMALIZATION
        # ---------------------------------------------------------------

        train_norm, test_norm = self.normalize(
            train_raw,
            test_raw,
        )

        # ---------------------------------------------------------------
        # TRAIN WINDOWS
        #
        # SMD training data is treated as the normal reference.
        # ---------------------------------------------------------------

        train_windows, train_starts = (
            self.create_windows(
                train_norm,
                labels=None,
                absolute_start=0,
            )
        )

        # ---------------------------------------------------------------
        # TEST WINDOWS
        # ---------------------------------------------------------------

        test_windows, test_labels, test_starts = (
            self.create_windows(
                test_norm,
                labels=test_labels_raw,
                absolute_start=0,
            )
        )

        print(
            "\n[SMD] Windowed dataset"
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
            f"{self.window_size}"
        )

        print(
            f"      Stride        : "
            f"{self.stride}"
        )

        print(
            f"      Test anomalies: "
            f"{int(test_labels.sum())}"
        )

        print(
            f"      Test timeline : "
            f"{test_starts[0]} → {test_starts[-1]}"
        )

        # ---------------------------------------------------------------
        # SCALER INFORMATION
        # ---------------------------------------------------------------

        print(
            "\n[SMD] Normalization"
        )

        print(
            "      Scaler fitted : TRAIN ONLY"
        )

        print(
            f"      Train mean[0] : "
            f"{self.scaler.mean_[0]:.6f}"
        )

        print(
            f"      Train scale[0]: "
            f"{self.scaler.scale_[0]:.6f}"
        )

        print(
            "      Test data transformed using "
            "training statistics."
        )

        return {
            # Raw data
            "train_raw": train_raw,
            "test_raw": test_raw,
            "test_labels_raw": test_labels_raw,

            # Normalized data
            "train_norm": train_norm,
            "test_norm": test_norm,

            # Windows
            "train_windows": train_windows,
            "train_starts": train_starts,

            "test_windows": test_windows,
            "test_labels": test_labels,
            "test_starts": test_starts,

            # Metadata
            "n_features": self.n_features,
            "feature_names": self.get_feature_names(),
            "machine": machine,

            "window_size": self.window_size,
            "stride": self.stride,

            # Absolute timeline
            "test_absolute_start": int(
                test_starts[0]
            ),
            "test_absolute_end": int(
                test_starts[-1] + self.window_size - 1
            ),

            # Scaler
            "scaler": self.scaler,
        }

    # =====================================================================
    # MACHINE LISTING
    # =====================================================================

    def list_machines(self) -> list[str]:

        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found:\n"
                f"{self.train_dir}"
            )

        machines = []

        for path in sorted(
            self.train_dir.glob(
                "machine-*.txt"
            )
        ):

            name = path.stem

            if name.startswith("machine-"):
                name = name[
                    len("machine-"):
                ]

            machines.append(name)

        return machines


# =========================================================================
# COMMAND LINE TEST
# =========================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Load and validate one SMD machine "
            "for FGEAD."
        )
    )

    parser.add_argument(
        "--machine",
        default="1-1",
        help="SMD machine identifier. Example: 1-1",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Sliding window size.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Sliding window stride.",
    )

    args = parser.parse_args()

    loader = SMDLoader(
        root="data/SMD",
        window_size=args.window,
        stride=args.stride,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FGEAD — SMD DATASET LOADER TEST"
    )

    print(
        "=" * 70
    )

    machines = loader.list_machines()

    print(
        f"\nAvailable machines: {len(machines)}"
    )

    print(
        "First machines:"
    )

    for machine in machines[:10]:
        print(
            f"   machine-{machine}"
        )

    print(
        f"\nLoading machine-{args.machine}..."
    )

    result = loader.prepare_machine(
        args.machine
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "[OK] SMD loader test successful."
    )

    print(
        "=" * 70
    )

    print(
        f"Machine       : "
        f"{result['machine']}"
    )

    print(
        f"Features      : "
        f"{result['n_features']}"
    )

    print(
        f"Feature names : "
        f"{result['feature_names'][:5]} ..."
    )

    print(
        f"Train windows : "
        f"{len(result['train_windows'])}"
    )

    print(
        f"Test windows  : "
        f"{len(result['test_windows'])}"
    )

    print(
        f"Test anomalies: "
        f"{int(result['test_labels'].sum())}"
    )

    print(
        f"First test window absolute start: "
        f"{result['test_starts'][0]}"
    )

    print(
        f"Last test window absolute start : "
        f"{result['test_starts'][-1]}"
    )

    print(
        f"Test timeline end               : "
        f"{result['test_absolute_end']}"
    )

    # ---------------------------------------------------------------
    # Inverse transformation smoke test
    # ---------------------------------------------------------------

    print(
        "\n[SMD] Inverse-transform smoke test"
    )

    sample_normalized = (
        result["test_windows"][0:1]
    )

    sample_raw = loader.inverse_transform(
        sample_normalized
    )

    print(
        f"      Normalized shape : "
        f"{sample_normalized.shape}"
    )

    print(
        f"      Raw shape        : "
        f"{sample_raw.shape}"
    )

    print(
        f"      Normalized value : "
        f"{sample_normalized[0, 0, 0]:.6f}"
    )

    print(
        f"      Raw value        : "
        f"{sample_raw[0, 0, 0]:.6f}"
    )

    print(
        "\n[OK] Normalization + inverse transformation "
        "are working."
    )


if __name__ == "__main__":
    main()