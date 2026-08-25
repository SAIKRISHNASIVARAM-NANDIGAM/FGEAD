"""
train.py
Trains the FGEAD model on synthetic (or SMD) data.
Includes gradient clipping, early stopping, validation loss tracking.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.preprocessor import TimeSeriesPreprocessor, TimeSeriesDataset
from models.fgead import FGEAD

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_path":          "data/synthetic_data.csv",
    "window_size":        60,
    "stride":             5,
    "train_ratio":        0.70,
    "val_ratio":          0.15,
    "embed_dim":          64,
    "n_heads":            4,
    "gcn_out":            64,
    "lstm_hidden":        128,
    "sparsity_threshold": 0.3,
    "dropout":            0.2,
    "n_epochs":           50,
    "batch_size":         64,
    "lr":                 1e-3,
    "weight_decay":       1e-4,
    "sparsity_lambda":    0.01,
    "patience":           10,           # early stopping patience
    "max_norm":           1.0,          # gradient clipping
    "checkpoint_dir":     "checkpoints",
    "device":             "cuda" if torch.cuda.is_available() else "cpu",
}


def build_loaders():
    proc = TimeSeriesPreprocessor(
        window_size=CONFIG["window_size"],
        stride=CONFIG["stride"],
    )
    data, labels = proc.load_csv(CONFIG["data_path"])
    data = proc.handle_missing(data)
    data, labels = proc.remove_duplicates(data, labels)

    n = len(data)
    t_end = int(n * CONFIG["train_ratio"])
    v_end = int(n * (CONFIG["train_ratio"] + CONFIG["val_ratio"]))

    train_norm, val_norm, test_norm = proc.normalize(
        data[:t_end], data[t_end:v_end], data[v_end:]
    )

    tr_win, tr_lbl = proc.create_windows(train_norm, labels[:t_end])
    va_win, va_lbl = proc.create_windows(val_norm,   labels[t_end:v_end])
    te_win, te_lbl = proc.create_windows(test_norm,  labels[v_end:])

    normal_mask = tr_lbl == 0
    tr_win_clean = tr_win[normal_mask]
    tr_ds = TimeSeriesDataset(tr_win_clean)  # no labels needed
    va_ds = TimeSeriesDataset(va_win, va_lbl)
    te_ds = TimeSeriesDataset(te_win, te_lbl)

    tr_loader = DataLoader(tr_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    va_loader = DataLoader(va_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    n_features = data.shape[1]
    return tr_loader, va_loader, te_loader, n_features, proc


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    recon_criterion = nn.MSELoss()

    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        optimizer.zero_grad()
        predictions, attn_weights, _ = model(x)

        # Reconstruction loss: predict t+1 from t
        recon_loss = recon_criterion(predictions, x[:, 1:, :])
        sparse_loss = model.sparsity_loss(attn_weights)
        loss = recon_loss + sparse_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["max_norm"])
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    recon_criterion = nn.MSELoss()

    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        predictions, attn_weights, _ = model(x)
        recon_loss = recon_criterion(predictions, x[:, 1:, :])
        sparse_loss = model.sparsity_loss(attn_weights)
        total_loss += (recon_loss + sparse_loss).item()

    return total_loss / len(loader)


def train():
    print(f"[FGEAD Training]  device={CONFIG['device']}")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    tr_loader, va_loader, te_loader, n_features, proc = build_loaders()
    print(f"  n_features={n_features}, "
          f"train batches={len(tr_loader)}, val batches={len(va_loader)}")

    model = FGEAD(
        n_features=n_features,
        embed_dim=CONFIG["embed_dim"],
        n_heads=CONFIG["n_heads"],
        gcn_out=CONFIG["gcn_out"],
        lstm_hidden=CONFIG["lstm_hidden"],
        sparsity_threshold=CONFIG["sparsity_threshold"],
        dropout=CONFIG["dropout"],
        sparsity_lambda=CONFIG["sparsity_lambda"],
    ).to(CONFIG["device"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_loss  = float("inf")
    patience_count = 0
    history        = {"train_loss": [], "val_loss": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Time':>6}")
    print("-" * 45)

    for epoch in range(1, CONFIG["n_epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, tr_loader, optimizer, CONFIG["device"])
        va_loss = validate(model, va_loader, CONFIG["device"])
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        scheduler.step(va_loss)

        marker = ""
        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))
            marker = " [saved]"
        else:
            patience_count += 1

        print(f"{epoch:>6}  {tr_loss:>12.6f}  {va_loss:>10.6f}  {elapsed:>5.1f}s{marker}")

        if patience_count >= CONFIG["patience"]:
            print(f"\n[Early Stopping] No improvement for {CONFIG['patience']} epochs.")
            break

    print(f"\n[OK] Best val loss: {best_val_loss:.6f}")
    print(f"[OK] Model saved to {CONFIG['checkpoint_dir']}/best_model.pt")

    # Save history for plotting
    np.save(os.path.join(CONFIG["checkpoint_dir"], "train_history.npy"), history)
    return model, history


if __name__ == "__main__":
    train()
