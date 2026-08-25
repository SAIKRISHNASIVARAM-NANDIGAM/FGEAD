# FGEAD — Project & Model Architecture Diagrams

> All architecture diagrams for the **Feature Graph Explainable Anomaly Detection** system.

---

## 1. Full System Architecture (End-to-End Pipeline)

The complete data flow from raw time series input to final anomaly report output, showing all 5 modules and their tensor shape transformations.

![FGEAD System Architecture](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/fgead_system_architecture_1785055629981.png)

---

## 2. Module 1 — Feature Embedding

Converts raw scalar feature values into 64-dimensional learned embeddings. Each feature (cpu_usage, memory_usage, etc.) gets a unique identity vector, scaled by the actual sensor reading.

**Key Transform:** `(B, T, M) → (B, T, M, 64)`

![Module 1 — Feature Embedding](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/module1_feature_embedding_1785055639861.png)

---

## 3. Module 2 — Self-Attention Graph Learner

The **core innovation** of FGEAD. Uses multi-head self-attention (4 heads) across the feature dimension to learn which features are correlated. The output is a sparse 20×20 adjacency matrix representing the feature dependency graph.

**Key Transform:** `(B, T, M, 64) → (B, T, M, 64) + (B, M, M)`

![Module 2 — Self-Attention Graph Learner](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/module2_graph_learner_1785055649442.png)

---

## 4. Module 3 — Graph Convolution + LSTM

Two-part spatio-temporal encoder. The **GCN** propagates information along the learned graph at each timestep (spatial). The **LSTM** captures temporal dynamics over the graph-enhanced features.

**Key Transform:** `(B, T, M, 64) + (B, M, M) → (B, T, 128)`

![Module 3 — GCN + LSTM](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/module3_gcn_lstm_1785055658831.png)

---

## 5. Module 4 — Forecasting Head + Anomaly Scoring

The forecasting head predicts `x(t+1)` from `x(t)`. The anomaly score is the **weighted mean absolute error** between predicted and actual values, using learned feature importance weights.

**Key Transform:** `(B, T, 128) → (B, T-1, 20) → (B, T-1)`

![Module 4 — Forecasting Head + Anomaly Scoring](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/module4_forecasting_scoring_1785055667383.png)

---

## 6. Module 5 — Explainability Layer (5-Question Framework)

The explainability layer answers 5 questions for every anomaly alert, producing a human-readable root cause analysis with confidence scoring.

| Question | Component | Output |
|----------|-----------|--------|
| Q1 + Q2 | Feature Deviation Analyzer | Top deviating features + actual vs predicted |
| Q3 | Graph Relationship Auditor | Broken feature correlations |
| Q4 | Anomaly Range Detector | Temporal start/end boundaries |
| Q5 | Confidence Scorer | Weighted confidence % + alert level |

![Module 5 — Explainability Layer](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/module5_explainability_1785055677006.png)

---

## 7. Training Pipeline

Shows the complete training loop including semi-supervised data filtering (normal windows only), dual loss function (MSE reconstruction + L1 sparsity), and optimization strategy.

![Training Pipeline](C:/Users/yuvak/.gemini/antigravity/brain/cfb51077-3004-49b1-bc6c-050d4990e179/training_pipeline_1785055685660.png)

---

## Quick Reference — Tensor Shape Summary

| Stage | Input | Output | Operation |
|-------|-------|--------|-----------|
| **Raw Data** | — | `(5000, 20)` | Generation |
| **Windowing** | `(N, 20)` | `(W, 60, 20)` | Sliding window |
| **Module 1** | `(B, 60, 20)` | `(B, 60, 20, 64)` | Embedding × value |
| **Module 2** | `(B, 60, 20, 64)` | `(B, 60, 20, 64)` + `(B, 20, 20)` | Self-attention |
| **Module 3 (GCN)** | `(B, 20, 64)` + `(B, 20, 20)` | `(B, 20, 64)` | Â·h·W per timestep |
| **Module 3 (LSTM)** | `(B, 60, 1280)` | `(B, 60, 128)` | 2-layer LSTM |
| **Module 4** | `(B, 59, 128)` | `(B, 59, 20)` → `(B, 59)` | MLP + weighted MAE |
| **Module 5** | All model outputs | Root cause report | 5-Q framework |
