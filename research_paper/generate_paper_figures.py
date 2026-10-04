import os
import sys
import shutil
import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output on Windows (prevents UnicodeEncodeError with emoji/arrows)
if sys.stdout.encoding and sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stderr.reconfigure(encoding="utf-8")

# Set directories
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(script_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

plots_dir = os.path.abspath(os.path.join(script_dir, '..', 'plots'))

# Copy existing plots if they exist
plot_mappings = {
    'smd_1_1_roc_curve.png': 'model_results.png',
    'smd_1_1_feature_contributions.png': 'xai_feature_importance.png',
    'smd_1_1_confusion_matrix.png': 'confusion_matrix.png',
    'smd_1_1_anomaly_score_timeline.png': 'anomaly_timeline.png'
}

for src_name, dst_name in plot_mappings.items():
    src_path = os.path.join(plots_dir, src_name)
    dst_path = os.path.join(fig_dir, dst_name)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_name} -> {dst_name}")

# Generate Figure 1: system_architecture.png
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.axis('off')

boxes = [
    ("Multivariate Time Series Data\n(SMD Telemetry: 38 Features)", (0.15, 0.75), '#e1f5fe'),
    ("StandardScaler Preprocessing\n(Train-only Calibration)", (0.15, 0.50), '#e8eaf6'),
    ("Sliding Window Construction\n(Window Size = 60, Stride = 5)", (0.15, 0.25), '#e8eaf6'),
    ("Feature Embedding\n(Embedding Dim = 64)", (0.50, 0.75), '#fff3e0'),
    ("Self-Attention Graph Learner\n(4 Heads, Top-K=5 Sparsity)", (0.50, 0.50), '#fff3e0'),
    ("Spatio-Temporal GCN + LSTM\n(GCN Layer + 2-Layer LSTM)", (0.50, 0.25), '#fff3e0'),
    ("Forecasting Head & Scoring\n(Next-Step Prediction Error)", (0.85, 0.75), '#e8f5e9'),
    ("Custom 5-Question XAI Engine\n(Deviation & Graph Auditing)", (0.85, 0.50), '#fce4ec'),
    ("FastAPI Backend & Dashboard\n(Real-Time Alerts & Visuals)", (0.85, 0.25), '#f3e5f5')
]

for text, (x, y), color in boxes:
    ax.text(x, y, text, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.6', facecolor=color, edgecolor='#37474f', lw=1.5), fontsize=8.5, fontweight='bold')

arrows = [
    ((0.15, 0.68), (0.15, 0.57)),
    ((0.15, 0.43), (0.15, 0.32)),
    ((0.28, 0.25), (0.37, 0.75)),
    ((0.50, 0.68), (0.50, 0.57)),
    ((0.50, 0.43), (0.50, 0.32)),
    ((0.63, 0.25), (0.72, 0.75)),
    ((0.85, 0.68), (0.85, 0.57)),
    ((0.85, 0.43), (0.85, 0.32))
]

for p1, p2 in arrows:
    ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.5))

plt.title("Figure 1: FGEAD System Architecture and Data Pipeline", fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'system_architecture.png'), bbox_inches='tight')
plt.close()

# Generate Figure 2: methodology.png
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.axis('off')

m_boxes = [
    ("Raw Input Telemetry\n$X \\in \\mathbb{R}^{B \\times T \\times M}$", (0.10, 0.5), '#e1f5fe'),
    ("Graph Learner\n$A_{ij} = \\text{Softmax}(\\frac{Q K^T}{\\sqrt{d}})$", (0.30, 0.5), '#fff3e0'),
    ("Temporal GCN + LSTM\n$H^{(t)} = \\text{GCN}(A, X^{(t)})$\n$h_t = \\text{LSTM}(H)$", (0.52, 0.5), '#e8f5e9'),
    ("Forecast & Score\n$\\hat{X}_{T}, S = \\sum w_i |X - \\hat{X}|$", (0.74, 0.5), '#fff8e1'),
    ("5-Q XAI Engine\nZ-scores + Graph Audit", (0.92, 0.5), '#fce4ec')
]

for text, (x, y), color in m_boxes:
    ax.text(x, y, text, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='#37474f', lw=1.2), fontsize=8, fontweight='bold')

m_arrows = [
    ((0.19, 0.5), (0.21, 0.5)),
    ((0.39, 0.5), (0.41, 0.5)),
    ((0.63, 0.5), (0.65, 0.5)),
    ((0.83, 0.5), (0.85, 0.5))
]

for p1, p2 in m_arrows:
    ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color="#37474f", lw=1.5))

plt.title("Figure 2: FGEAD Technical Flowchart", fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'methodology.png'), bbox_inches='tight')
plt.close()

# Generate Figure 5: explanation_example.png
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
ax.axis('off')

exp_text = """
========================================================================
               FGEAD -- ANOMALY EXPLANATION REPORT
========================================================================
[*] Peak Anomaly: Timestep t = 17489 (Score = 1949.62)
[W] Analysis Window: t = 17485 to t = 17544 (Window Size = 60)

[>] Top Contributing Telemetry Features (Z-Score Deviation):
   1. feature_33  (Actual: 1.0000 | Pred: 0.0000)  -> Dev: +1.0000  (z = 3.7sig ^ spike)
   2. feature_28  (Actual: 1.0000 | Pred: 0.0001)  -> Dev: +0.9999  (z = 3.8sig ^ spike)
   3. feature_32  (Actual: 1.0000 | Pred: 0.0003)  -> Dev: +0.9997  (z = 3.7sig ^ spike)
   4. feature_24  (Actual: 1.0000 | Pred: 0.0600)  -> Dev: +0.9400  (z = 3.6sig ^ spike)
   5. feature_31  (Actual: 1.0000 | Pred: 0.0886)  -> Dev: +0.9114  (z = 3.7sig ^ spike)

[~] Key Graph Relationship Changes (Attention vs Observed Correlation):
   - feature_13 <-> feature_27 (Expected: 0.14 | Observed: 0.92) [RELATIONSHIP CHANGE]
   - feature_10 <-> feature_15 (Expected: 0.25 | Observed: 1.00) [RELATIONSHIP CHANGE]

[!] Alert Confidence: 72.1%  --->  HIGH ALERT LEVEL
[i] Diagnostic Hypothesis: Unusual multivariate pattern: feature_33 ^ spike,
   feature_28 ^ spike, feature_32 ^ spike, feature_13/feature_27 relationship change
========================================================================
"""

ax.text(0.5, 0.5, exp_text, ha='center', va='center', family='monospace', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#fafafa', edgecolor='#263238', lw=1.5))

plt.title("Figure 5: Example Diagnostic Hypothesis Output Generated by FGEAD (SMD Machine 1-1)", fontsize=9.5, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'explanation_example.png'), bbox_inches='tight')
plt.close()

print("All figure files generated successfully in research_paper/figures/")
