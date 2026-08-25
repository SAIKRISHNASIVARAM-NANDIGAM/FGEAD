# 🔍 FGEAD — Viva & Interview Explanation Guide

This guide is designed to help you confidently explain the **FGEAD** project, its architecture, the baseline comparison metrics, the dashboard layout, and handle tricky questions from examiners or interviewers.

---

## 🌟 1. Executive Summary (The Elevator Pitch)

> **"What is FGEAD?"**
> "FGEAD stands for **Feature Graph-based Explainable Anomaly Detector**. It is a semi-supervised anomaly detection system designed for multivariate time-series data (like server metrics). 
> 
> Unlike traditional models that either ignore feature relationships or require pre-defined dependency graphs, FGEAD **learns the relationship graph dynamically** using self-attention. It then processes these relationships through **Graph Convolution Networks (GCN)** and captures temporal trends with **LSTMs**. Finally, it provides **fully explainable alerts** by answering 5 structured questions (Q1–Q5) for every anomaly, giving operators immediate root-cause diagnostics instead of just a binary alarm."

---

## 📊 2. Explaining the Baseline Comparison (The Numbers)

When examiners look at the terminal output or the comparison table, they will ask you to explain **why the models performed the way they did**.

### The Comparison Table:
* **FGEAD**: ~0.958 F1-Score | ~0.932 AUC
* **LSTM Autoencoder**: ~0.918 F1-Score | ~0.927 AUC
* **Isolation Forest**: ~0.105 F1-Score | ~0.866 AUC

### How to Explain This in the Interview:

1. **Why is Isolation Forest (IF) so low (~10% F1-score)?**
   * *Answer*: "Isolation Forest is a non-deep learning, unsupervised model. To process a sliding window, we must flatten it into a single 1D vector (e.g., $60 \times 20 = 1200$ features). This flattening **destroys all temporal order and correlation structures**. If a feature spikes gradually over time or if two features swap correlations (a correlation break anomaly), Isolation Forest cannot detect it effectively because it treats every timestep as an independent feature. Hence, it suffers from a extremely low Recall."

2. **Why does LSTM Autoencoder do well (~92% F1-score) but still underperform FGEAD?**
   * *Answer*: "The LSTM Autoencoder is a strong baseline because it models temporal sequences. However, it treats all features as a flat array. It lacks an explicit **spatial or inter-feature dependency mechanism**. It has to learn both temporal causality and complex cross-feature interactions using just its hidden states, which is hard. FGEAD outperforms it because FGEAD has a dedicated Graph Convolution layer that explicitly groups and aggregates related features, letting the LSTM focus solely on temporal dynamics."

3. **Why does FGEAD achieve the best F1-score (~96%)?**
   * *Answer*: "FGEAD combines the best of both worlds. By using Multi-Head Self-Attention across features, it constructs a **dynamic dependency graph**. The GCN layer uses this graph to propagate message embeddings between correlated metrics (e.g., CPU and CPU Temperature). Finally, the LSTM captures the chronological sequence. This dual modeling allows FGEAD to easily identify both temporal spikes and complex **correlation breaks** (e.g., CPU usage drops but temperature stays high), which are invisible to simpler baselines."

---

## 🎨 3. Explaining the Streamlit Dashboard (The Visuals)

If you are asked to walk through the dashboard, structure your explanation into **four key sections**:

### Section A: Summary & Metrics
* **What to say**: "At the top, we display high-level system health: Total Windows processed, total features (20), number of anomalies detected, and the calculated anomaly rate. This gives an immediate operational view of system stability."

### Section B: Interactive Anomaly Score Plot
* **What to say**: "This chart shows the real-time anomaly scores calculated by FGEAD over time. The **red shaded regions** indicate the ground truth anomalies, and the **dashed horizontal line** represents the dynamically calibrated alert threshold ($Mean + k \cdot Std$). We can see FGEAD's score peaks precisely within the red anomaly windows, showing extreme precision and zero false alarms."

### Section C: The Anomaly Explanation (Q1–Q5 Breakdown)
Explain how the model answers the **5 structured questions** for any selected window:

* **Q1 & Q2: Feature Deviations (Bar Chart)**:
  * *What it is*: A horizontal bar chart ranking features by deviation (difference between actual and reconstructed/forecasted values), color-coded by Z-score severity.
  * *What to say*: "This shows *which* metrics are malfunctioning and *by how much*. For instance, if CPU spikes to $2.4\sigma$, it immediately stands out as the primary driver of the anomaly."
* **Q3: Feature Dependency Graph (Heatmap)**:
  * *What it is*: A correlation heatmap showing the learned multi-head self-attention weights between all 20 features.
  * *What to say*: "This is the brain of FGEAD. The model automatically learned that `cpu_usage` is closely coupled with `cpu_temp` and `load_avg`. If this relationship breaks (e.g. CPU temperature spikes while CPU usage is idle), the graph auditor detects a correlation gap and flags it as a severe system discrepancy."
* **Q4: Anomaly Time Range**:
  * *What to say*: "This tells operators exactly *when* the anomaly started, when it ended, and the total duration inside the sliding window."
* **Q5: Multi-Signal Confidence & Root Cause**:
  * *What to say*: "Instead of a simple alarm, FGEAD computes a weighted confidence score based on severity, duration, and broken graph links. It then uses a semantic rule engine to output an actionable root cause, such as *'Thermal issue — CPU load and temperature de-synced'*, allowing engineers to troubleshoot instantly."

---

## 🧠 4. Top 6 Hard Questions Examiners Ask (and how to crush them)

### Q1: What is "Semi-Supervised" learning in this context?
* **Your Answer**: "Semi-supervised anomaly detection means we **train the model exclusively on normal data**. The model learns to reconstruct and forecast normal system patterns. During inference, if we feed it an anomalous pattern, the model will fail to reconstruct it accurately, resulting in a high reconstruction/forecasting error. We treat this high error as the anomaly score."

### Q2: Why is it crucial to fit the `StandardScaler` only on the training set?
* **Your Answer**: "Fitting the scaler on the entire dataset (including validation and test sets) causes **information leakage**. It leaks the mean and variance of the test data into the training phase, artificially inflating evaluation metrics. We fit the scaler strictly on the 70% training data, and then apply that exact pre-fitted scaler to transform the validation, test, and real-time API inputs."

### Q3: How does the model dynamically learn the graph structure?
* **Your Answer**: "We use a **Self-Attention Graph Learner**. We project each feature into a high-dimensional embedding space. We then compute the dot-product similarity (self-attention) between the feature embeddings. The resulting attention weight matrix represents the strength of dependencies between features. We apply a sparsity threshold (e.g., $0.3$) to keep only the strongest, most stable connections, resulting in a sparse, highly interpretable graph."

### Q4: Why is there a sparsity regularization (L1 loss) on attention weights?
* **Your Answer**: "Without regularization, the attention matrix would be fully connected (dense), meaning every feature is connected to every other feature. This makes the graph uninterpretable and prone to noise. An $L_1$ penalty on the attention weights encourages **sparsity**, forcing the model to select only the most critical, genuine physical relationships (like `cpu_usage` and `cpu_temp`), which makes the explainability graph extremely clear and interpretable."

### Q5: What is the benefit of using "Weights Only" loading in PyTorch?
* **Your Answer**: "In PyTorch 2.6 and newer, standard `torch.load()` raises a security and deprecation warning because it can execute arbitrary pickled code. Since we only save and load the network weights (the `state_dict`), setting `weights_only=True` enforces a safe, restricted unpickling process, protecting the application from malicious code execution and aligning with modern PyTorch best practices."

### Q6: How does the `/predict` endpoint handle incoming raw data?
* **Your Answer**: "The `/predict` endpoint accepts raw, un-normalized sliding window data from clients. During startup, the API fits the preprocessor scaler on the training dataset. When `/predict` is called, it standardizes the incoming raw window using `scaler.transform()`, converts it to a FloatTensor, passes it through the model, and runs the explanation auditor to return a complete diagnostic report in under a few milliseconds."
