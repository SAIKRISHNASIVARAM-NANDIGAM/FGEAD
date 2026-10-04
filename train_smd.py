"""
train_smd.py
============================================================

Train FGEAD on the real Server Machine Dataset (SMD).

Pipeline:
    SMD train data
        ↓
    Training-only normalization
        ↓
    Sliding windows
        ↓
    Normal training windows only
        ↓
    Feature Embedding
        ↓
    Self-Attention Graph Learning
        ↓
    GCN + LSTM
        ↓
    Forecasting
        ↓
    Anomaly scoring
        ↓
    Best checkpoint

This script is separate from train.py so the existing
synthetic-data experiment remains reproducible.

Example:
    python train_smd.py --machine 1-1

"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
import torch.nn as nn
from torch.utils.data import DataLoader

from data.smd_loader import SMDLoader
from data.preprocessor import TimeSeriesDataset
from models.fgead import FGEAD


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "window_size": 60,
    "stride": 5,

    "embed_dim": 64,
    "n_heads": 4,
    "gcn_out": 64,
    "lstm_hidden": 128,

    "sparsity_threshold": 0.3,
    "dropout": 0.2,

    "n_epochs": 30,
    "batch_size": 64,

    "lr": 1e-3,
    "weight_decay": 1e-4,
    "sparsity_lambda": 0.01,

    "patience": 7,
    "max_norm": 1.0,

    "checkpoint_dir": "checkpoints",

    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
}


# =============================================================================
# DATA
# =============================================================================

def build_loaders(machine: str):
    print("\n" + "=" * 65)
    print("FGEAD — SMD DATA PREPARATION")
    print("=" * 65)

    loader = SMDLoader(
        root="data/SMD",
        window_size=CONFIG["window_size"],
        stride=CONFIG["stride"],
    )

    result = loader.prepare_machine(machine)

    train_windows = result["train_windows"]
    test_windows = result["test_windows"]

    train_starts = result["train_starts"]
    test_starts = result["test_starts"]

    test_labels = result["test_labels"]

    n_features = result["n_features"]

    # -------------------------------------------------------------------------
    # Important:
    #
    # SMD training data is considered normal for the semi-supervised
    # forecasting setup.
    #
    # We therefore train FGEAD using the training sequence only.
    # -------------------------------------------------------------------------

    train_ds = TimeSeriesDataset(
        train_windows
    )

    test_ds = TimeSeriesDataset(
        test_windows,
        test_labels
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    print("\n[OK] SMD loaders created")

    print(
        f"     Machine        : {machine}"
    )

    print(
        f"     Features       : {n_features}"
    )

    print(
        f"     Train windows  : {len(train_windows)}"
    )

    print(
        f"     Test windows   : {len(test_windows)}"
    )

    print(
        f"     Test anomalies : {int(test_labels.sum())}"
    )

    print(
        f"     Train batches  : {len(train_loader)}"
    )

    print(
        f"     Test batches   : {len(test_loader)}"
    )

    return (
        train_loader,
        test_loader,
        n_features,
        result,
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    reconstruction_loss = nn.MSELoss()

    for batch in loader:

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)

        optimizer.zero_grad()

        predictions, attn_weights, _ = model(x)

        # Predict t+1 from t
        target = x[:, 1:, :]

        recon_loss = reconstruction_loss(
            predictions,
            target,
        )

        sparse_loss = model.sparsity_loss(
            attn_weights
        )

        loss = (
            recon_loss
            + sparse_loss
        )

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CONFIG["max_norm"],
        )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):
    model.eval()

    total_loss = 0.0

    reconstruction_loss = nn.MSELoss()

    for batch in loader:

        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)

        predictions, attn_weights, _ = model(x)

        target = x[:, 1:, :]

        recon_loss = reconstruction_loss(
            predictions,
            target,
        )

        sparse_loss = model.sparsity_loss(
            attn_weights
        )

        loss = (
            recon_loss
            + sparse_loss
        )

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# =============================================================================
# TRAIN
# =============================================================================

def train(machine: str):

    device = CONFIG["device"]

    print(
        f"\n[FGEAD SMD Training]"
    )

    print(
        f"Device : {device}"
    )

    print(
        f"Machine: {machine}"
    )

    if device == "cuda":

        print(
            f"GPU    : "
            f"{torch.cuda.get_device_name(0)}"
        )

    os.makedirs(
        CONFIG["checkpoint_dir"],
        exist_ok=True,
    )

    (
        train_loader,
        test_loader,
        n_features,
        dataset,
    ) = build_loaders(machine)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = FGEAD(
        n_features=n_features,

        embed_dim=CONFIG["embed_dim"],

        n_heads=CONFIG["n_heads"],

        gcn_out=CONFIG["gcn_out"],

        lstm_hidden=CONFIG["lstm_hidden"],

        sparsity_threshold=CONFIG[
            "sparsity_threshold"
        ],

        dropout=CONFIG["dropout"],

        sparsity_lambda=CONFIG[
            "sparsity_lambda"
        ],
    ).to(device)

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),

        lr=CONFIG["lr"],

        weight_decay=CONFIG[
            "weight_decay"
        ],
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.5,
        )
    )

    # -------------------------------------------------------------------------
    # Training state
    # -------------------------------------------------------------------------

    best_loss = float("inf")

    patience_count = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # -------------------------------------------------------------------------
    # Checkpoint name
    # -------------------------------------------------------------------------

    checkpoint_path = os.path.join(
        CONFIG["checkpoint_dir"],
        f"fgead_smd_machine_{machine.replace('-', '_')}.pt",
    )

    history_path = os.path.join(
        CONFIG["checkpoint_dir"],
        f"fgead_smd_machine_{machine.replace('-', '_')}_history.npy",
    )

    # -------------------------------------------------------------------------
    # Training table
    # -------------------------------------------------------------------------

    print("\n" + "-" * 65)

    print(
        f"{'Epoch':>6}"
        f"{'Train Loss':>15}"
        f"{'Val Loss':>15}"
        f"{'Time':>10}"
    )

    print("-" * 65)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    for epoch in range(
        1,
        CONFIG["n_epochs"] + 1,
    ):

        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        # SMD does not provide a separate validation set here.
        #
        # We use training loss for scheduler/checkpoint monitoring.
        # The test set remains completely untouched during training.
        val_loss = train_loss

        elapsed = (
            time.time()
            - start_time
        )

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "val_loss"
        ].append(val_loss)

        scheduler.step(val_loss)

        marker = ""

        if val_loss < best_loss:

            best_loss = val_loss

            patience_count = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "machine":
                        machine,

                    "n_features":
                        n_features,

                    "config":
                        CONFIG,

                    "best_loss":
                        best_loss,
                },

                checkpoint_path,
            )

            marker = " [saved]"

        else:

            patience_count += 1

        print(
            f"{epoch:>6}"
            f"{train_loss:>15.6f}"
            f"{val_loss:>15.6f}"
            f"{elapsed:>9.1f}s"
            f"{marker}"
        )

        if patience_count >= CONFIG["patience"]:

            print(
                "\n[Early Stopping]"
            )

            print(
                f"No improvement for "
                f"{CONFIG['patience']} epochs."
            )

            break

    # -------------------------------------------------------------------------
    # Save training history
    # -------------------------------------------------------------------------

    np.save(
        history_path,
        history,
        allow_pickle=True,
    )

    print("\n" + "=" * 65)

    print(
        "SMD TRAINING COMPLETE"
    )

    print("=" * 65)

    print(
        f"Machine       : {machine}"
    )

    print(
        f"Features      : {n_features}"
    )

    print(
        f"Best loss     : {best_loss:.6f}"
    )

    print(
        f"Checkpoint    : {checkpoint_path}"
    )

    print(
        f"History       : {history_path}"
    )

    print("=" * 65)

    return (
        model,
        dataset,
        history,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train FGEAD on the "
            "Server Machine Dataset."
        )
    )

    parser.add_argument(
        "--machine",
        default="1-1",
        help=(
            "SMD machine identifier. "
            "Example: 1-1"
        ),
    )

    args = parser.parse_args()

    train(
        machine=args.machine
    )


if __name__ == "__main__":
    main()