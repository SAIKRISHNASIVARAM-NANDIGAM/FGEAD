# Author Review & Student Viva Guide
## FGEAD: Feature-Level Graph-Based Explainable Anomaly Detection

This document is prepared specifically for your personal review, enabling you to understand, verify, and defend your research paper during a viva exam or academic presentation.

---

## 1. Must Verify Personally Before Submission

- [ ] **Author Details:** Replace `[AUTHOR NAME 1]`, `[DEPARTMENT]`, `[INSTITUTION]`, `[EMAIL]` in `paper.tex` and `paper.md` with your actual personal details.
- [ ] **Execution Environment:** Confirm your local execution hardware (CPU model / GPU model if applicable) and Python version (e.g., Python 3.10 / 3.11).
- [ ] **Numerical Verification:** Confirm that all metrics cited in Table V (SMD Machine 1-1: Precision=0.5600, Recall=0.9039, F1=0.6916, ROC-AUC=0.9709, PR-AUC=0.7981) match your official evaluation log in `plots/smd_1_1_FINAL_results.txt`.

---

## 2. Key Technical Concepts to Understand for Viva

### A. Graph Learning & Topology
- **Why Graph Neural Networks (GNN)?** Server telemetry metrics (e.g., CPU load, memory utilization, disk I/O) are physically interconnected. Standard models treat features as isolated streams. GCN allows features to exchange spatial messages along learned graph edges.
- **Top-$K=5$ Sparsification:** Dense attention matrices contain noisy, spurious correlations. Keeping only the Top 5 strongest connections per metric node retains meaningful dependencies while maintaining computational efficiency.
- **Learned Dependency vs. Empirical Correlation:** The attention matrix represents *learned dependency representations* from training. Graph Relationship Auditing compares this against the *observed Pearson correlation* of the anomalous window ($t^* \pm 10$ steps) to spot relationship changes (decoupling, sign reversal). It is non-causal and measures structural discrepancy.

### B. Spatio-Temporal Modeling & Forecasting
- **GCN + LSTM:** GCN processes spatial feature interactions at each discrete timestep; the 2-layer LSTM (hidden size 128) processes temporal sequences across the sliding window ($T=60$).
- **Forecasting vs. Reconstruction:** Trained semi-supervised on normal data only. Anomalies trigger forecast deviations ($|x - \hat{x}|$) weighted by learnable feature importance parameters ($\boldsymbol{\gamma}$).

### C. 5-Question Explainability Framework (XAI)
1. **Q1 (Metric Deviation):** Standardizes forecast errors into Z-scores: $z_i = d_i / \sigma_{W, i}$.
2. **Q2 (Directional Labeling):** Inverse-transforms data to the original scale and compares actual vs. predicted values ($\uparrow$ spike, $\downarrow$ drop, $\approx$ stable).
3. **Q3 (Graph Relationship Auditing):** Flags changed relationships where $|\mathbf{R}^{\text{obs}}_{ij} - \mathbf{A}_{ij}| \ge 0.20$ (Complete Decoupling, Sign Reversal, Relationship Change).
4. **Q4 (Temporal Localization):** Identifies contiguous anomalous timesteps bounding start time, end time, and duration.
5. **Q5 (Alert Confidence):** Calculates a composite confidence score $C \in [0, 1.0]$:
   $$C = 0.40 \cdot S_{\text{signal}} + 0.25 \cdot D_{\text{signal}} + 0.20 \cdot F_{\text{signal}} + 0.15 \cdot G_{\text{signal}}$$
   where each component is normalized to $[0, 1.0]$:
   - $S_{\text{signal}} = \min(1.0, \max(0.0, (S_{\text{peak}} - \tau) / (0.25 \tau)))$
   - $D_{\text{signal}} = \min(1.0, \text{Duration} / (0.30 \cdot T))$
   - $F_{\text{signal}} = \min(1.0, N_{\text{deviating}} / (0.30 \cdot M))$
   - $G_{\text{signal}} = \min(1.0, N_{\text{broken}} / (0.20 \cdot E_{\text{total}}))$

---

## 3. Likely Viva Questions & Defensible Answers

### Q1: Why use semi-supervised forecasting instead of a supervised classifier?
**Answer:** Real-world server failures are rare, unpredictable, and diverse. Supervised classifiers require labeled positive samples for all possible fault types. Semi-supervised forecasting trains exclusively on normal telemetry to learn standard operating patterns. Any significant deviation signifies an anomaly, regardless of whether that specific failure mode was previously seen.

### Q2: Why not use standard XAI tools like SHAP or LIME?
**Answer:** SHAP and LIME assume feature independence, which fails on correlated server telemetry. Moreover, local perturbations in SHAP/LIME can produce unrealistic temporal states, fail to evaluate relationship changes between metrics, and carry high computational overhead over sliding windows. FGEAD uses a custom scale-aware post-hoc auditing engine designed specifically for time-series forecasting.

### Q3: Why is Precision (0.5600) lower than Recall (0.9039)?
**Answer:** Our threshold $\tau = 2.073376$ was set at the 99.5th percentile of normal prediction errors. In enterprise server monitoring, missed outages (false negatives) are far more damaging than false alarms (false positives). Calibrating for high recall ($90.39\%$ window-level, $97.14\%$ point-level) is a deliberate, operationally appropriate design choice prioritizing anomaly sensitivity.

### Q4: Does FGEAD prove physical root cause?
**Answer:** No. Because SMD does not provide ground-truth feature-level root-cause annotations, FGEAD's output represents a candidate **diagnostic root-cause hypothesis** based on Z-score rankings and graph relationship discrepancies. It assists domain engineers by pointing to likely metric deviations rather than proving physical causation.

### Q5: Is Graph Relationship Auditing causal?
**Answer:** No. Graph Relationship Auditing measures the statistical discrepancy between the model's learned dependency weights and the empirical Pearson correlation of an anomalous window. It does not perform causal discovery or claim causal proof.

---

## 4. Key Project Metrics Quick Reference

- **Machine Evaluated:** SMD machine-1-1 (38 features, 28,479 test timesteps)
- **Window Level:** Precision = 0.5600, Recall = 0.9039, F1 = 0.6916, ROC-AUC = 0.9709, PR-AUC = 0.7981
- **Point Level:** Precision = 0.4891, Recall = 0.9714, F1 = 0.6506, ROC-AUC = 0.9708, PR-AUC = 0.7373
- **True Anomaly Timesteps:** 2,694 | **Detected Anomaly Timesteps:** 5,351
- **Synthetic Dataset:** 5,000 timesteps, 20 features, Precision = 0.7273, Recall = 0.8571, F1 = 0.7869, ROC-AUC = 0.9501
- **Case Study Window:** Peak timestep $t^* = 17489$, Score = 1949.62, Threshold = 2.073, Confidence = 72.1% (HIGH)
