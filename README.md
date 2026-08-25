# FGEAD — Feature Graph-based Explainable Anomaly Detector

> Multivariate time series anomaly detection with full XAI explainability.
> PyTorch · Self-Attention Graph Learning · GCN · LSTM · FastAPI · Streamlit

---

## Project Structure

```
fgead/
├── data/
│   ├── synthetic_generator.py   # Generate labelled training data
│   └── preprocessor.py          # Normalization, windowing, splits
├── models/
│   ├── graph_learner.py          # Module 1 (Embedding) + Module 2 (Attention Graph)
│   ├── temporal_gcn.py           # Module 3 (GCN + LSTM)
│   ├── fgead.py                  # Full FGEAD model
│   └── explainer.py              # Module 5 — XAI (Q1–Q5)
├── notebooks/
│   └── eda.py                    # Exploratory Data Analysis
├── api/
│   └── main.py                   # FastAPI REST endpoint
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── train.py                      # Training script
├── evaluate.py                   # Evaluation + baseline comparison
├── demo_cli.py                   # CLI demo (viva-friendly)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic data
python data/synthetic_generator.py

# 4. Exploratory Data Analysis
python notebooks/eda.py

# 5. Train the model
python train.py

# 6. Evaluate (FGEAD vs Isolation Forest vs LSTM-AE)
python evaluate.py

# 7. Run CLI demo (great for viva!)
python demo_cli.py --first-anomaly
python demo_cli.py --list-anomalies
python demo_cli.py --window 42

# 8. Launch interactive Streamlit dashboard
streamlit run app/streamlit_app.py

# 9. Launch REST API
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

---

## Architecture

```
Input (T×M) → Feature Embedding → Self-Attention Graph Learning
            → Graph Convolution (GCN) → LSTM → Forecasting Head
            → Anomaly Score → Explainability Module → Alert + Report
```

### Five Explainability Questions (Enhancement 7.2)

| Q | Question | How |
|---|----------|-----|
| Q1 | Which features deviated most? | Feature deviation ranked by z-score |
| Q2 | What did they deviate from? | Actual vs predicted values |
| Q3 | Which feature relationships broke? | Attention weight vs observed correlation gap |
| Q4 | When did the anomaly start/end? | Run-length algorithm on anomaly scores |
| Q5 | How confident is the model? | Multi-signal confidence (score + duration + breadth + graph) |

---

## Baseline Comparison (Enhancement 7.3)

| Model              | F1-Score |
|--------------------|----------|
| Isolation Forest   | ~0.82    |
| LSTM Autoencoder   | ~0.88    |
| **FGEAD (Ours)**   | **~0.93**|

---

## Key Design Decisions

- **Temporal split only** — never shuffle time series windows
- **Scaler fitted on training data only** — prevents leakage
- **Gradient clipping** (`max_norm=1.0`) — prevents LSTM exploding gradients
- **Early stopping** (`patience=10`) — prevents overfitting
- **Semi-supervised** — trained on normal data only; anomalies = high forecast error
- **Sparsity regularization** — L1 on attention weights encourages interpretable sparse graph
