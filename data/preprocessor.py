"""
data/preprocessor.py
Handles normalization, sliding windows, train/val/test splits,
and the PyTorch Dataset wrapper.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch


class TimeSeriesPreprocessor:
    """All data prep steps in one place."""

    def __init__(self, window_size: int = 60, stride: int = 5):
        self.window_size = window_size
        self.stride = stride
        self.scaler = StandardScaler()
        self._fitted = False

    # ── Loading ─────────────────────────────────────────────────────────────

    def load_csv(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        if "label" in df.columns:
            labels = df["label"].values.astype(int)
            data = df.drop("label", axis=1).values.astype(np.float32)
        else:
            labels = np.zeros(len(df), dtype=int)
            data = df.values.astype(np.float32)
        print(f"[OK] Loaded {path}: {data.shape}, anomaly rate={labels.mean():.2%}")
        return data, labels

    # ── Cleaning ────────────────────────────────────────────────────────────

    def handle_missing(self, data: np.ndarray) -> np.ndarray:
        """Forward-fill then backward-fill NaNs."""
        df = pd.DataFrame(data)
        df = df.ffill().bfill()
        return df.values.astype(np.float32)

    def remove_duplicates(
        self, data: np.ndarray, labels: np.ndarray | None = None
    ) -> tuple:
        df = pd.DataFrame(data)
        mask = ~df.duplicated()
        data = data[mask]
        if labels is not None:
            labels = labels[mask]
        return (data, labels) if labels is not None else (data,)

    # ── Normalization ────────────────────────────────────────────────────────

    def normalize(
        self, train: np.ndarray, *others: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """
        Fit scaler ONLY on train data, transform all splits.
        CRITICAL: never fit on test data — this causes information leakage.
        """
        train_norm = self.scaler.fit_transform(train).astype(np.float32)
        self._fitted = True
        result = [train_norm]
        for arr in others:
            result.append(self.scaler.transform(arr).astype(np.float32))
        return tuple(result) if others else train_norm

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)

    # ── Windowing ────────────────────────────────────────────────────────────

    def create_windows(
        self, data: np.ndarray, labels: np.ndarray | None = None
    ) -> tuple:
        """
        Sliding window extraction.
        Window label = 1 if ANY timestep in window is anomalous.
        """
        W, S = self.window_size, self.stride
        n = len(data)
        windows, win_labels = [], []

        for start in range(0, n - W + 1, S):
            end = start + W
            windows.append(data[start:end])
            if labels is not None:
                win_labels.append(int(labels[start:end].any()))

        windows = np.stack(windows, axis=0)  # (N_windows, W, M)
        if labels is not None:
            win_labels = np.array(win_labels, dtype=int)
            return windows, win_labels
        return windows

    # ── Splitting ────────────────────────────────────────────────────────────

    def temporal_split(
        self,
        windows: np.ndarray,
        labels: np.ndarray | None = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> tuple:
        """
        Chronological split — NO shuffling (preserves temporal order).
        """
        N = len(windows)
        t_end = int(N * train_ratio)
        v_end = int(N * (train_ratio + val_ratio))

        X_train, X_val, X_test = windows[:t_end], windows[t_end:v_end], windows[v_end:]
        if labels is not None:
            return (X_train, X_val, X_test,
                    labels[:t_end], labels[t_end:v_end], labels[v_end:])
        return X_train, X_val, X_test


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset wrapper for windowed time series."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray | None = None):
        self.windows = torch.FloatTensor(windows)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return x


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    proc = TimeSeriesPreprocessor(window_size=60, stride=5)
    data, labels = proc.load_csv("data/synthetic_data.csv")
    data = proc.handle_missing(data)
    data, labels = proc.remove_duplicates(data, labels)

    n = len(data)
    train_raw, test_raw = data[: int(0.7 * n)], data[int(0.7 * n) :]
    train_lbl, test_lbl = labels[: int(0.7 * n)], labels[int(0.7 * n) :]

    train_norm, test_norm = proc.normalize(train_raw, test_raw)

    train_win, tr_lbl = proc.create_windows(train_norm, train_lbl)
    test_win, te_lbl = proc.create_windows(test_norm, test_lbl)

    train_ds = TimeSeriesDataset(train_win, tr_lbl)
    test_ds = TimeSeriesDataset(test_win, te_lbl)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}, Labels: {y.shape}")
