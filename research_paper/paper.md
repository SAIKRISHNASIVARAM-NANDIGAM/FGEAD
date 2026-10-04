# FGEAD: A Feature-Level Graph-Based Framework for Explainable Anomaly Detection in Multivariate Time Series

**Authors:**  
[AUTHOR NAME 1], [AUTHOR NAME 2], [AUTHOR NAME 3]  
[DEPARTMENT], [INSTITUTION]  
[EMAIL]

---

## Abstract
Detecting anomalies in high-dimensional multivariate server telemetry is critical for maintaining infrastructure reliability. However, deep learning anomaly detection models often function as opaque black boxes, failing to explain why an observation was flagged or which metric relationships altered. In this paper, we present **FGEAD** (Feature-level Graph-based Explainable Anomaly Detection), an end-to-end semi-supervised forecasting framework equipped with an integrated post-hoc explainability engine. FGEAD models evolving inter-metric topological dependencies using a self-attention graph learner with Top-$K$ sparsification ($K=5$), combined with Graph Convolutional Networks (GCN) and a 2-layer LSTM for spatio-temporal representations. Anomaly detection is performed via next-timestep forecasting error weighted by learnable feature importance parameters. To address the explainability gap, FGEAD implements a structured 5-question XAI framework: (1) metric deviation ranking via Z-scores, (2) scale-aware directional labeling ($\uparrow$ spike / $\downarrow$ drop), (3) Graph Relationship Auditing contrasting learned attention-derived dependency weights against observed Pearson correlations to isolate decoupling and sign-reversal events, (4) temporal range localization, and (5) composite confidence scoring. Evaluated on the benchmark Server Machine Dataset (SMD Machine 1-1, 38 metrics, 28,479 test timesteps), FGEAD achieves a window-level ROC-AUC of 0.9709, PR-AUC of 0.7981, Recall of 0.9039, and F1-score of 0.6916 (Point-level ROC-AUC 0.9708, Recall 0.9714). The high recall reflects a deliberate operational threshold calibration (99.5th percentile) favoring anomaly sensitivity. FGEAD is deployed via a FastAPI REST backend and a Streamlit observability dashboard prototype.

**Keywords:** Explainable AI (XAI), Anomaly Detection, Graph Neural Networks, Multivariate Time Series, Server Telemetry, Graph Relationship Auditing, Feature Attribution.

---

## I. Introduction
Modern cloud infrastructures and enterprise server clusters generate continuous streams of multivariate telemetry data, spanning CPU utilization, memory pressure, disk I/O, network throughput, and kernel performance metrics [20]. Rapid identification of operational anomalies is vital to mitigate system outages, service degradation, and hardware failures [12].

While deep learning architectures—such as Autoencoders, Recurrent Neural Networks (RNNs), Transformers, and Graph Neural Networks (GNNs)—have demonstrated strong detection capabilities [8, 13, 15], their operational utility is often hindered by the "black-box" problem [1, 4]. When an automated monitoring tool triggers an alert on a 38-metric server telemetry stream, site reliability engineers (SREs) and operations teams need to address critical diagnostic questions: *Which specific metrics deviated? What were their expected values? Have dependency patterns between components changed? How confident is the alert?*

Applying generic Explainable AI (XAI) feature-attribution tools such as LIME [2] and SHAP [3] to multivariate temporal telemetry presents notable challenges. In time-series server metrics, features are frequently strongly interdependent, temporal context is paramount, and local perturbation schemes can produce unrealistic metric states. Furthermore, applying iterative feature attribution over continuous sliding windows carries substantial computational overhead [16], while raw importance vectors do not automatically provide relationship-level context or operational alerts [11]. Importantly, Jain and Wallace [18] demonstrated that internal neural attention weights alone do not reliably constitute complete explanations. **FGEAD does not use SHAP or LIME**; rather, it implements a specialized, scale-aware post-hoc auditing pipeline tailored for graph-augmented temporal forecasting.

FGEAD unifies high-sensitivity anomaly detection with structured post-hoc diagnostics. FGEAD learns dynamic spatial dependencies between metrics using multi-head self-attention with Top-$K$ sparsification ($K=5$) and L1 regularization, coupled with Graph Convolution (GCN) and Long Short-Term Memory (LSTM) networks to capture temporal sequence dynamics. Anomaly scoring is formulated as a semi-supervised next-step forecasting error weighted by learnable metric importance parameters.

FGEAD integrates a post-hoc explainability engine based on a structured 5-Question Framework. Rather than relying solely on raw feature weights or generic perturbation models, FGEAD performs:
1. **Feature Deviation Analysis:** Standardizing forecast deviations into Z-scores on the original data scale to identify top contributing metrics.
2. **Graph Relationship Auditing:** Comparing the model's learned attention-derived dependency matrix against the empirical Pearson correlation matrix of the anomalous window to detect structural dependency breaks (decoupling, sign reversal).
3. **Temporal Localization & Confidence Calibration:** Isolating contiguous anomaly boundaries and computing a multi-factor composite alert confidence score.

The principal contributions of this work are summarized as follows:
- We present an integrated spatio-temporal architecture combining dynamic self-attention graph learning, GCN, and LSTM for semi-supervised multivariate time-series anomaly forecasting.
- We introduce a Graph Relationship Auditing mechanism that identifies discrepancies between learned dependency representations and observed empirical correlations in anomalous windows.
- We define a structured 5-question XAI engine that translates model predictions into actionable alert reports featuring Z-score rankings, directional indicators ($\uparrow/\downarrow$), relationship change types, and composite alert confidence levels.
- We conduct empirical evaluations on the benchmark Server Machine Dataset (SMD Machine 1-1) and a 20-feature synthetic telemetry dataset, demonstrating a window-level ROC-AUC of 0.9709 and Point-level Recall of 0.9714.
- We provide a prototype deployment implementation featuring a FastAPI REST backend and a Streamlit observability dashboard.

---

## II. Related Work

### A. Explainable Artificial Intelligence (XAI)
The field of XAI has evolved rapidly to address model interpretability in high-stakes domain deployments [1]. Model-agnostic post-hoc explanation methods, such as LIME [2] and SHAP [3], rely on local surrogate approximations or game-theoretic Shapley values. Influence functions [5] trace model predictions back to influential training instances. However, Rudin [4] argued that post-hoc explanations on black-box models can be misleading and advocate for domain-aware interpretable design. In multivariate time series, standard feature attribution tools suffer because metrics are strongly correlated and temporal context must be preserved.

### B. Machine Learning for Anomaly Detection
Unsupervised and semi-supervised deep learning models dominate time-series anomaly detection. OmniAnomaly [12] introduced stochastic recurrent neural networks with planar normalizing flows alongside the benchmark Server Machine Dataset (SMD). USAD [13] employed adversarial autoencoders to amplify reconstruction errors. MSCRED [14] constructed multi-scale signature matrices processed via ConvLSTM networks. Anomaly Transformer [15] introduced association discrepancy between prior-associations and series-associations in self-attention. While effective at detection, these methods do not explicitly map feature-to-feature graph topologies alongside structured diagnostic reports.

### C. Explainable Anomaly Detection
Recent research explicitly addresses root-cause localization in anomaly detection. MTAD-GAT [6] used dual graph attention networks (feature and temporal) to model metric dependencies. GDN [8] inferred directed feature dependency graphs using structure learning to spot relationship deviations. InterFusion [7] utilized hierarchical variational autoencoders with MCMC-based metric localization. GANF [9] integrated Bayesian graph discovery with normalizing flows. Exathlon [10] established a benchmark framework for evaluating time-series explainability. Tripathy et al. [11] applied autoencoders with post-hoc SHAP for industrial faults. However, existing approaches either treat explainability as an unvalidated byproduct of attention weights or rely on computationally heavy iterative sampling (MCMC).

### D. Graph Neural Networks and Feature Attribution for XAI
In graph neural networks, GNNExplainer [16] and PGExplainer [17] optimize node/edge masks to extract explanatory subgraphs. However, per-instance GNN optimization is computationally demanding for high-frequency sliding windows. Crucially, Jain and Wallace [18] proved that raw attention weights do not reliably equal explanation, demonstrating the necessity of dedicated post-hoc auditing. Recent 2024–2025 studies, such as Lee et al. [19] (masked latent generative XAD) and Wang et al. [20] (multivariate AD taxonomy), emphasize that combining graph representations with explicit deviation auditing represents an active frontier.

### E. Research Gap
Based on the literature reviewed in this study, existing approaches do not appear to provide a unified framework that simultaneously combines: (a) dynamic graph structure learning, (b) scale-aware feature deviation analysis, (c) auditing of learned dependency representations against observed window correlations, and (d) structured, multi-dimensional operational alert generation. FGEAD addresses this research gap by combining these capabilities into a single operationally oriented framework.

---

## III. Problem Statement

Let $\mathbf{X} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$ represent a multivariate time-series telemetry sequence, where each observation $\mathbf{x}_t \in \mathbb{R}^M$ contains $M$ continuous telemetry metric features recorded at timestep $t$.

Given a sliding window of length $T$ ending at timestep $t$, denoted as $\mathbf{W}_t = [\mathbf{x}_{t-T+1}, \mathbf{x}_{t-T+2}, \dots, \mathbf{x}_t] \in \mathbb{R}^{T \times M}$, the objectives are two-fold:

1. **Semi-Supervised Anomaly Detection:** Learn a forecasting function $f_{\theta}: \mathbb{R}^{(T-1) \times M} \to \mathbb{R}^M$ trained exclusively on normal telemetry data ($\mathbf{x}_t \in \mathcal{D}_{\text{train}}$) to predict the next timestep $\hat{\mathbf{x}}_t = f_{\theta}(\mathbf{W}_{t-1})$. Compute a scalar anomaly score $S_t \in \mathbb{R}_{\ge 0}$ such that an anomaly decision $y_t \in \{0, 1\}$ is flagged if $S_t > \tau$, where $\tau$ is a calibrated threshold.
2. **Structured Post-Hoc Explanation:** Upon flagging an anomaly ($y_t = 1$), generate a diagnostic explanation tuple $\mathcal{E}_t = (\mathcal{F}_{\text{top}}, \mathcal{G}_{\text{broken}}, \mathcal{T}_{\text{range}}, \mathcal{C}_{\text{alert}})$, where $\mathcal{F}_{\text{top}}$ ranks top deviated features with Z-scores and directional indicators, $\mathcal{G}_{\text{broken}}$ isolates changed inter-metric relationships, $\mathcal{T}_{\text{range}}$ defines temporal duration, and $\mathcal{C}_{\text{alert}}$ provides a calibrated alert confidence level.

---

## IV. Proposed Methodology

The architecture of FGEAD consists of four primary machine learning modules followed by the integrated 5-Question Explainability Layer.

```
+-----------------------------------------------------------------------+
|                       FGEAD Technical Pipeline                       |
+-----------------------------------------------------------------------+
|  [Raw Telemetry X] -> [StandardScaler] -> [Sliding Window (60x38)]    |
|                                 |                                     |
|  [Feature Embedding (d=64)] -> [Self-Attention Graph Learner (Top-K)]  |
|                                 |                                     |
|  [Temporal GCN Layer] -----> [2-Layer LSTM (Hidden=128)]              |
|                                 |                                     |
|  [Forecast Head (MLP)] ----> [Anomaly Score S = sum(w * |X - X_hat|)] |
|                                 |                                     |
|  [Score > Threshold?] -----> [Trigger 5-Question XAI Engine]          |
|                                 |                                     |
|  [FastAPI Backend] --------> [Streamlit Observability Dashboard]      |
+-----------------------------------------------------------------------+
```

### Module 1: Feature Embedding
Each metric index $i \in \{1, \dots, M\}$ is mapped to a dense trainable vector embedding $\mathbf{e}_i \in \mathbb{R}^{d}$ with dimension $d = 64$ using an embedding matrix $\mathbf{E} \in \mathbb{R}^{M \times d}$. The raw numerical metric values at timestep $t$ are projected into the embedding space, normalized via Layer Normalization, and passed through Dropout ($p=0.1$):
$$\mathbf{H}^{(0)}_{t, i} = \text{LayerNorm}(\mathbf{e}_i \cdot x_{t, i}) \in \mathbb{R}^d$$

### Module 2: Self-Attention Graph Learner
To capture inter-metric dependencies without requiring a static predefined topology, FGEAD dynamically constructs a feature adjacency matrix $\mathbf{A} \in \mathbb{R}^{M \times M}$ using multi-head self-attention.

Temporal pooling is applied over the window length $T$: $\bar{\mathbf{h}}_i = \frac{1}{T}\sum_{t=1}^T \mathbf{H}^{(0)}_{t, i}$. Linear query ($\mathbf{W}_Q$) and key ($\mathbf{W}_K$) projections compute attention affinity scores across $H=4$ attention heads:
$$\alpha_{ij}^{(h)} = \text{Softmax}\left( \frac{(\mathbf{W}_Q^{(h)} \bar{\mathbf{h}}_i) (\mathbf{W}_K^{(h)} \bar{\mathbf{h}}_j)^T}{\sqrt{d/H}} \right)$$

The attention scores are averaged across heads to form a dense adjacency matrix $\tilde{\mathbf{A}} \in \mathbb{R}^{M \times M}$. To enforce graph sparsity and reduce noisy connections, a Top-$K$ hard mask ($K=5$) is applied per metric node:
$$\mathbf{A}_{ij} = \begin{cases} \tilde{\mathbf{A}}_{ij}, & \text{if } \tilde{\mathbf{A}}_{ij} \in \text{Top-K}(\tilde{\mathbf{A}}_{i, :}) \\ 0, & \text{otherwise} \end{cases}$$

An L1 sparsity regularization loss $\mathcal{L}_{\text{sparsity}} = \lambda \|\mathbf{A}\|_1$ ($\lambda = 10^{-4}$) is added during training.

### Module 3: Spatio-Temporal GCN + LSTM
Spatial metric interactions are captured at each timestep $t$ via Graph Convolutional Networks (GCN):
$$\mathbf{H}^{(1)}_t = \text{ReLU}\left( \text{BatchNorm1D}\left( \hat{\mathbf{A}} \mathbf{H}^{(0)}_t \mathbf{W}_{\text{GCN}} \right) \right)$$
where $\hat{\mathbf{A}} = \mathbf{D}^{-1/2} (\mathbf{A} + \mathbf{I}_M) \mathbf{D}^{-1/2}$ is the symmetric normalized adjacency matrix.

The spatial node representations at timestep $t$ are flattened into a vector of dimension $M \cdot d$ and fed sequentially into a 2-layer LSTM network with hidden state dimension $h_{\text{lstm}} = 128$ and dropout rate $0.2$:
$$\mathbf{h}_t, (\mathbf{c}_t, \mathbf{s}_t) = \text{LSTM}\left( [\mathbf{H}^{(1)}_t], \mathbf{h}_{t-1} \right)$$

### Module 4: Forecasting Head & Anomaly Scoring
The final LSTM hidden state $\mathbf{h}_T$ is mapped through a 2-layer MLP to predict the telemetry metrics at next timestep $T+1$:
$$\hat{\mathbf{x}}_{T+1} = \mathbf{W}_2 \cdot \text{Dropout}\left( \text{ReLU}(\mathbf{W}_1 \mathbf{h}_T + \mathbf{b}_1) \right) + \mathbf{b}_2$$

The absolute forecast error vector is $\mathbf{e}_{T+1} = |\mathbf{x}_{T+1} - \hat{\mathbf{x}}_{T+1}| \in \mathbb{R}^M$. FGEAD learns a trainable metric weighting parameter $\mathbf{w} \in \mathbb{R}^M$ normalized via Softmax: $\boldsymbol{\gamma} = \text{Softmax}(\mathbf{w})$. The overall scalar anomaly score for window $\mathbf{W}$ is:
$$S_{\mathbf{W}} = \sum_{i=1}^M \gamma_i \cdot e_{T+1, i}$$

The model is trained to minimize the combined forecasting and graph sparsity loss:
$$\mathcal{L}_{\text{total}} = \frac{1}{B}\sum_{b=1}^B \|\mathbf{x}_b - \hat{\mathbf{x}}_b\|_2^2 + \lambda \|\mathbf{A}\|_1$$

---

## V. Dataset and Preprocessing

### Server Machine Dataset (SMD)
The primary benchmark evaluated is the Server Machine Dataset (SMD) [12], collected from a large Internet enterprise over 5 weeks. SMD consists of 28 server machines across 3 clusters. We evaluate machine-1-1, which comprises 38 continuous telemetry metrics.

| Characteristic | Value / Setting |
| :--- | :--- |
| Machine ID | machine-1-1 |
| Total Features ($M$) | 38 metrics |
| Train Timesteps | 28,479 (100% normal) |
| Test Timesteps | 28,479 |
| Test Anomaly Timesteps | 2,694 (Point anomaly rate: 9.46%) |
| Window Size ($T$) | 60 timesteps |
| Window Stride | 5 timesteps |
| Train Windows | 5,684 windows |
| Test Windows | 5,684 windows |
| Test Anomaly Windows | 635 windows |

### Preprocessing Pipeline
1. **Missing Value Imputation:** Forward-filling followed by backward-filling. Duplicate rows dropped.
2. **Standardization:** Normalized via `StandardScaler`. `fit()` is called exclusively on the training split to prevent leakage.
3. **Sliding Windowing:** Chronological sliding windows of size $T=60$ with stride $S=5$.
4. **Threshold Calibration:** Calibration subset (15% of train windows) determines anomaly threshold $\tau = 2.073376$ at the 99.5th percentile.

---

## VI. Machine Learning Model Setup

| Hyperparameter | Value |
| :--- | :--- |
| Embedding Dimension ($d$) | 64 |
| Self-Attention Heads ($H$) | 4 |
| Top-$K$ Graph Connections | 5 per metric |
| Sparsity Regularization ($\lambda$) | $10^{-4}$ |
| GCN Output Dimension | 64 |
| LSTM Layers / Hidden Dim | 2 layers / 128 units |
| LSTM Dropout | 0.2 |
| MLP Hidden Layer | 64 units |
| Optimizer | Adam ($\text{lr}=10^{-3}$, weight decay $10^{-4}$) |
| Learning Rate Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Batch Size / Epochs | 64 / 30 epochs |
| Gradient Clipping | Max norm = 1.0 |
| Early Stopping | Patience = 7 epochs |

---

## VII. Explainable AI Framework

When an anomaly is flagged ($S_{\mathbf{W}} > \tau$), FGEAD triggers its custom 5-Question Explainability Engine (`models/explainer.py`). Model inputs and predictions are inverse-transformed back to the original scale prior to evaluation.

### Q1: Metric Deviation Ranking
At peak anomaly timestep $t^* = \arg\max_{t} S_t$, absolute error for feature $i$ is $d_i = |x_{t^*, i} - \hat{x}_{t^*, i}|$. Deviations are standardized into Z-scores: $z_i = d_i / \sigma_{W, i}$. Top $K=5$ metrics are isolated.

### Q2: Directional Deviation Labeling
Comparing actual value $x_{t^*, i}$ against predicted $\hat{x}_{t^*, i}$ on the original scale yields a semantic directional label:
$$\text{Label}_i = \begin{cases} \text{"}\uparrow \text{ spike"}, & \text{if } x_{t^*, i} - \hat{x}_{t^*, i} > +0.1 \sigma_i \\ \text{"}\downarrow \text{ drop"}, & \text{if } x_{t^*, i} - \hat{x}_{t^*, i} < -0.1 \sigma_i \\ \text{"}\approx \text{ stable"}, & \text{otherwise} \end{cases}$$

### Q3: Graph Relationship Auditing
FGEAD isolates the window surrounding peak anomaly ($t^* \pm 10$ steps) and computes empirical Pearson correlation matrix $\mathbf{R}^{\text{obs}} \in \mathbb{R}^{M \times M}$. This is contrasted against learned expected dependency matrix $\mathbf{A} \in \mathbb{R}^{M \times M}$.

Graph Relationship Auditing measures discrepancies between the model's learned dependency representation and observed empirical co-movement. It does not perform causal discovery or claim causal proof. A relationship change is flagged if $|\mathbf{R}^{\text{obs}}_{ij} - \mathbf{A}_{ij}| \ge 0.20$. Break types:
- **Complete Decoupling:** $\mathbf{A}_{ij} \ge 0.70$ (strong expected dependency) but $\mathbf{R}^{\text{obs}}_{ij} < 0.10$.
- **Sign Reversal:** $\mathbf{A}_{ij} \ge 0.20$ but $\mathbf{R}^{\text{obs}}_{ij} < 0$.
- **Relationship Change:** Generic gap $|\mathbf{R}^{\text{obs}}_{ij} - \mathbf{A}_{ij}| \ge 0.20$.

### Q4: Temporal Range Localization
Contiguous anomalous timesteps within the window are identified to bound anomaly start time $t_{\text{start}}$, end time $t_{\text{end}}$, and duration.

### Q5: Composite Alert Confidence Calibration
Composite confidence score $C \in [0, 1.0]$ is computed using a weighted combination of four normalized signal components:
$$C = w_s \cdot S_{\text{signal}} + w_d \cdot D_{\text{signal}} + w_f \cdot F_{\text{signal}} + w_g \cdot G_{\text{signal}}$$
where $w_s = 0.40, w_d = 0.25, w_f = 0.20, w_g = 0.15$. The four normalized signal components are calculated in the implementation as:
- $S_{\text{signal}} = \min(1.0, \max(0.0, (S_{\text{peak}} - \tau) / (0.25 \tau)))$
- $D_{\text{signal}} = \min(1.0, \text{Duration} / (0.30 \cdot T))$
- $F_{\text{signal}} = \min(1.0, N_{\text{deviating}} / (0.30 \cdot M))$
- $G_{\text{signal}} = \min(1.0, N_{\text{broken}} / (0.20 \cdot E_{\text{total}}))$

Alert levels: CRITICAL ($\ge 80\%$), HIGH ($\ge 60\%$), MEDIUM ($\ge 40\%$), and LOW ($< 40\%$).

---

## VIII. System Architecture and Implementation

FGEAD backend is implemented using FastAPI (`api/main.py`) and Uvicorn, providing RESTful endpoints:
- `GET /health`: System status, active device (CPU/CUDA), loaded model configuration.
- `GET /dataset`: Dataset statistics, sample counts, baseline anomaly rates.
- `GET /machine`: Machine metadata and metric names.
- `POST /predict`: Accepts $60 \times 38$ telemetry window tensor, returns JSON payload containing anomaly score, top features, broken graph pairs, alert confidence level, and synthesized diagnostic root-cause hypothesis string. Execution latency is handled via lightweight HTTP REST calls.

Frontend is built using Streamlit (`app/streamlit_app.py`), delivering an observability dashboard prototype.

---

## IX. Experimental Results and Evaluation

### Quantitative Results on SMD Machine 1-1

| Evaluation Metric | Window-Level | Point-Level |
| :--- | :---: | :---: |
| Threshold ($\tau$) | 2.073376 | 2.073376 |
| Precision ($P$) | 0.5600 | 0.4891 |
| **Recall ($R$)** | **0.9039** | **0.9714** |
| F1-Score ($F_1$) | 0.6916 | 0.6506 |
| **ROC-AUC** | **0.9709** | **0.9708** |
| PR-AUC | 0.7981 | 0.7373 |
| True Anomaly Timesteps | — | 2,694 |
| Detected Anomaly Timesteps | — | 5,351 |

On a secondary 20-feature synthetic telemetry dataset (5,000 steps), FGEAD achieved Precision = 0.7273, Recall = 0.8571, F1 = 0.7869, and ROC-AUC = 0.9501.

### Discussion of Precision-Recall Trade-off
FGEAD exhibits strong discriminative performance (ROC-AUC 0.9709) and high recall (0.9039 window-level, 0.9714 point-level). The moderate precision (0.5600) and F1-score (0.6916) stem directly from the 99.5th percentile train-only threshold calibration strategy. In an operational monitoring setting where missed anomalies may be costly, a high-recall threshold can be a reasonable design choice prioritizing anomaly sensitivity over precision.

---

## X. XAI Case Study and Diagnostic Hypothesis Analysis

Actual anomalous test window output from SMD Machine 1-1 (peak timestep $t^* = 17489$, score $S = 1949.62$, threshold $\tau = 2.073$):

| Feature Name | Actual | Predicted | Deviation | Z-Score | Label |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `feature_33` | 1.0000 | 0.0000 | 1.0000 | $3.7\sigma$ | $\uparrow$ spike |
| `feature_28` | 1.0000 | 0.0001 | 0.9999 | $3.8\sigma$ | $\uparrow$ spike |
| `feature_32` | 1.0000 | 0.0003 | 0.9997 | $3.7\sigma$ | $\uparrow$ spike |
| `feature_24` | 1.0000 | 0.0600 | 0.9400 | $3.6\sigma$ | $\uparrow$ spike |
| `feature_31` | 1.0000 | 0.0886 | 0.9114 | $3.7\sigma$ | $\uparrow$ spike |

**Graph Auditing Findings:** 49 changed feature pairs detected. Top broken pairs:
- `feature_13` $\leftrightarrow$ `feature_27`: Expected dependency $= 0.14$, Observed correlation $= 0.92$ (`RELATIONSHIP CHANGE`).
- `feature_10` $\leftrightarrow$ `feature_15`: Expected dependency $= 0.25$, Observed correlation $= 1.00$ (`RELATIONSHIP CHANGE`).

**Alert & Diagnostic Hypothesis Synthesis:** Confidence = **72.1%** (**HIGH ALERT LEVEL**). Diagnostic hypothesis string:
*"Unusual multivariate pattern detected: feature_33 $\uparrow$ spike, feature_28 $\uparrow$ spike, feature_32 $\uparrow$ spike, feature_13/feature_27 relationship change."* Because SMD does not provide metric-level root-cause annotations, this output represents a candidate diagnostic hypothesis rather than confirmed physical causation.

---

## XI. Qualitative Literature Synthesis

Table XI provides a qualitative synthesis comparing the methodological characteristics of FGEAD against major published architectures reviewed in the literature. This table summarizes design traits reported in published works and is not a controlled benchmark under identical experimental setups.

| Study / Method | ML Architecture | Target Domain | XAI Mechanism | Feature Explanation | Graph Auditing | Deployment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OmniAnomaly [12] | Stochastic RNN + NormFlow | SMD, SMAP | Rec. Prob. | Score Ranking | No | Script |
| USAD [13] | Adversarial Autoencoder | SMD, SWaT | None | None | No | Script |
| MSCRED [14] | ConvLSTM + SigMatrix | Synthetic, Power | SigMatrix | Image Residuals | Dense Correlation | Script |
| GDN [8] | GNN + Structure Learning | SWaT, WADI | Learned Graph | Attention Weights | Implicit Deviation | Script |
| MTAD-GAT [6] | Dual Graph Attention | SMD, MSL | Dual Attention | Attention Weights | No | Script |
| InterFusion [7] | Hierarchical VAE | SMD, SMAP | MCMC Sampling | Metric Localization | No | Script |
| Anomaly Trans. [15] | Transformer + Association | SMD, SWaT | Assoc. Disc. | Temporal Attention | No | Script |
| **FGEAD (Ours)** | **Self-Attn GCN + LSTM** | **SMD Telemetry** | **5-Q Engine** | **Z-Score + Dir ($\uparrow/\downarrow$)** | **Explicit Graph Audit** | **FastAPI + Streamlit** |

---

## XII. Limitations
1. **Single-Machine Evaluation Scope:** Only SMD Machine 1-1 is evaluated in detail; multi-machine generalization across all 28 SMD servers remains to be established.
2. **Lack of Annotated Ground-Truth Root Causes:** SMD provides temporal anomaly labels but lacks feature-level root-cause annotations, preventing quantitative evaluation of explanation accuracy.
3. **Generic Metric Semantics:** SMD metrics are named generically (`feature_00` to `feature_37`), limiting natural language descriptions to keyword heuristics.
4. **Non-Causal Auditing Nature:** Graph Relationship Auditing measures discrepancy between dependency representations and empirical correlations; it should not be interpreted as proving physical causality.
5. **Heuristic Confidence Scoring:** The alert confidence score is an operational heuristic and has not been externally calibrated against probabilistic ground-truth outcomes.
6. **Quadratic Correlation Complexity:** Computing $M \times M$ Pearson correlation matrices for anomalous windows incurs $\mathcal{O}(M^2 T)$ complexity, creating potential bottlenecks for large metric counts ($M > 500$).
7. **Synthetic Dataset Scope:** The 20-feature synthetic dataset provides controlled validation but does not substitute for diverse real-world industrial benchmarks.

---

## XIII. Future Work
1. Evaluating FGEAD across all 28 SMD server machines and external industrial datasets.
2. Benchmarking explanation quality on datasets with labeled root causes such as Exathlon [10].
3. Calibrating confidence scores against operator feedback.
4. Incorporating adaptive/seasonal thresholding.
5. Optimizing graph auditing via sparse correlation computations.
6. Integrating semantic metric metadata for operational deployments.

---

## XIV. Conclusion
We presented **FGEAD**, a feature-level graph-based explainable anomaly detection framework for multivariate server telemetry. By combining self-attention graph learning, GCN, and LSTM with a structured 5-Question XAI engine, FGEAD bridges the gap between high-sensitivity anomaly forecasting and actionable operational diagnostics. Evaluated on SMD Machine 1-1, FGEAD achieved a window-level ROC-AUC of 0.9709 and Point-level Recall of 0.9714. The system provides operations teams with transparent Z-score rankings, directional indicators, graph relationship auditing, and calibrated alert confidence levels delivered through a FastAPI REST backend and Streamlit dashboard prototype.

---

## References

1. A. Barredo Arrieta et al., "Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI," *Information Fusion*, vol. 58, pp. 82–115, 2020.
2. M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why Should I Trust You?": Explaining the Predictions of Any Classifier," in *Proc. KDD '16*, 2016, pp. 1135–1144.
3. S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. NeurIPS 2017*, 2017, pp. 4765–4774.
4. C. Rudin, "Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead," *Nature Machine Intelligence*, vol. 1, no. 5, pp. 206–215, 2019.
5. P. W. Koh and P. Liang, "Understanding Black-box Predictions via Influence Functions," in *Proc. ICML 2017*, 2017, pp. 1885–1894.
6. H. Zhao et al., "Multivariate Time-series Anomaly Detection via Graph Attention Network," in *2020 IEEE ICDM*, 2020, pp. 841–850.
7. Z. Li et al., "Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding," in *Proc. KDD '21*, 2021, pp. 3220–3230.
8. A. Deng and B. Hooi, "Graph Neural Network-Based Anomaly Detection in Multivariate Time Series," in *Proc. AAAI-21*, 2021, pp. 4027–4035.
9. E. Dai and J. Chen, "Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series," in *Proc. ICLR*, 2022.
10. V. Jacob et al., "Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series," *Proc. VLDB Endow.*, vol. 14, no. 11, pp. 2613–2626, 2021.
11. S. M. Tripathy et al., "Explaining Anomalies in Industrial Multivariate Time-series Data with the help of eXplainable AI," in *2022 IEEE BigComp*, 2022, pp. 209–216.
12. Y. Su et al., "OmniAnomaly: Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network," in *Proc. KDD '19*, 2019, pp. 3228–3236.
13. J. Audibert et al., "USAD: UnSupervised Anomaly Detection on Multivariate Time Series," in *Proc. KDD '20*, 2020, pp. 3395–3404.
14. C. Zhang et al., "A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data," in *Proc. AAAI-19*, 2019, pp. 1409–1416.
15. J. Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy," in *Proc. ICLR*, 2022.
16. Z. Ying et al., "GNNExplainer: Generating Explanations for Graph Neural Networks," in *Proc. NeurIPS 2019*, 2019, pp. 9240–9251.
17. D. Luo et al., "Parameterized Explainer for Graph Neural Network," in *Proc. NeurIPS 2020*, 2020, pp. 19620–19631.
18. S. Jain and B. C. Wallace, "Attention is not Explanation," in *Proc. NAACL-HLT 2019*, 2019, pp. 3543–3556.
19. D. Lee, S. Malacarne, and E. Aune, "Explainable Time Series Anomaly Detection using Masked Latent Generative Modeling," *Pattern Recognition*, vol. 156, p. 110826, 2024.
20. F. Wang et al., "A Survey of Deep Anomaly Detection in Multivariate Time Series: Taxonomy, Applications, and Directions," *Sensors*, vol. 25, no. 1, p. 190, 2025.
