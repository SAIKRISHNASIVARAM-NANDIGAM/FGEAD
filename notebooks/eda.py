"""
notebooks/eda.py
Exploratory Data Analysis — run standalone or adapt into a Jupyter notebook.

Usage:  python notebooks/eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

plt.style.use("dark_background")
os.makedirs("plots", exist_ok=True)


def load(path="data/synthetic_data.csv"):
    df     = pd.read_csv(path)
    labels = df["label"].values
    data   = df.drop("label", axis=1).values
    names  = [c for c in df.columns if c != "label"]
    return data, labels, names


def plot_time_series_overview(data, labels, feature_names, n_show=6):
    fig, axes = plt.subplots(n_show, 1, figsize=(16, 3 * n_show), sharex=True)
    anomaly_regions = np.where(labels == 1)[0]

    for i, ax in enumerate(axes):
        ax.plot(data[:, i], color="#6c8efb", linewidth=0.8, label=feature_names[i])
        for idx in anomaly_regions:
            ax.axvspan(idx - 1, idx + 1, alpha=0.3, color="red")
        ax.set_ylabel(feature_names[i], fontsize=9)
        ax.grid(alpha=0.2)

    axes[0].set_title("Time Series Overview  (red = anomaly region)", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/time_series_overview.png", dpi=120)
    print("[✓] Saved plots/time_series_overview.png")
    plt.close()


def plot_correlation_matrix(data, feature_names):
    df   = pd.DataFrame(data, columns=feature_names)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=feature_names, yticklabels=feature_names, linewidths=0.3)
    ax.set_title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png", dpi=120)
    print("[✓] Saved plots/correlation_matrix.png")

    # Top correlated pairs
    pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            pairs.append((feature_names[i], feature_names[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("\nTop 10 Correlated Feature Pairs:")
    for a, b, c in pairs[:10]:
        print(f"  {a:20s} ↔ {b:20s}  corr={c:.3f}")
    plt.close()


def plot_anomaly_distribution(data, labels, feature_names):
    normal    = data[labels == 0]
    anomalous = data[labels == 1]
    n_feat    = min(9, data.shape[1])

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, ax in enumerate(axes.flatten()):
        if i >= n_feat:
            break
        ax.hist(normal[:, i],    bins=50, alpha=0.7, color="#6c8efb", label="Normal")
        ax.hist(anomalous[:, i], bins=50, alpha=0.7, color="#f87171", label="Anomaly")
        ax.set_title(feature_names[i])
        ax.legend(fontsize=8)

    fig.suptitle("Normal vs. Anomaly Distributions per Feature", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png", dpi=120)
    print("[✓] Saved plots/feature_distributions.png")
    plt.close()


def plot_class_balance(labels):
    counts = [("Normal", (labels == 0).sum()), ("Anomaly", (labels == 1).sum())]
    names, vals = zip(*counts)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=["#6c8efb", "#f87171"], width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:,}\n({val/sum(vals):.1%})",
            ha="center",
            fontsize=11,
        )
    ax.set_title("Class Distribution")
    plt.tight_layout()
    plt.savefig("plots/class_balance.png", dpi=120)
    print("[✓] Saved plots/class_balance.png")
    plt.close()


def print_summary(data, labels, feature_names):
    print("\n" + "=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Timesteps   : {len(data):,}")
    print(f"  Features    : {data.shape[1]}")
    print(f"  Normal      : {(labels == 0).sum():,}  ({(labels == 0).mean():.1%})")
    print(f"  Anomaly     : {(labels == 1).sum():,}  ({(labels == 1).mean():.1%})")
    print(f"\n  Feature statistics:")
    for i, name in enumerate(feature_names[:5]):
        print(f"  {name:20s}  mean={data[:,i].mean():.3f}  std={data[:,i].std():.3f}")
    if len(feature_names) > 5:
        print(f"  ... ({len(feature_names)-5} more features)")
    print("=" * 55)


if __name__ == "__main__":
    data, labels, names = load()
    print_summary(data, labels, names)
    plot_time_series_overview(data, labels, names)
    plot_correlation_matrix(data, names)
    plot_anomaly_distribution(data, labels, names)
    plot_class_balance(labels)
    print("\n[✓] EDA complete — check the plots/ folder.")
