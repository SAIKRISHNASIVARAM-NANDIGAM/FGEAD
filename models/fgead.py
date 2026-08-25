"""
models/fgead.py
Full FGEAD model — assembles all 5 modules:
  1. Feature Embedding
  2. Self-Attention Graph Learning
  3. Graph Convolution + LSTM
  4. Forecasting Head + Anomaly Scoring
  5. (Explainability lives in explainer.py, driven by the attention weights here)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_learner import FeatureEmbedding, SelfAttentionGraphLearner
from models.temporal_gcn import TemporalGraphModel


class FGEAD(nn.Module):
    """
    Feature Graph-based Explainable Anomaly Detector.

    Training mode  : semi-supervised — trained to FORECAST the next timestep
                     using only normal data.  Anomalies = high forecast error.
    Inference mode : returns per-timestep anomaly scores + attention weights
                     for the explainability module.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        gcn_out: int = 64,
        lstm_hidden: int = 128,
        sparsity_threshold: float = 0.3,
        dropout: float = 0.2,
        sparsity_lambda: float = 0.01,
    ):
        super().__init__()
        self.n_features = n_features
        self.sparsity_lambda = sparsity_lambda

        # Module 1: Feature Embedding
        self.feature_embedding = FeatureEmbedding(n_features, embed_dim)

        # Module 2: Self-Attention Graph Learning
        self.graph_learner = SelfAttentionGraphLearner(
            embed_dim=embed_dim,
            n_heads=n_heads,
            sparsity_threshold=sparsity_threshold,
            dropout=dropout,
        )

        # Module 3: GCN + LSTM
        self.temporal_model = TemporalGraphModel(
            n_features=n_features,
            embed_dim=embed_dim,
            gcn_out=gcn_out,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
        )

        # Module 4a: Forecasting head — predict next timestep per feature
        self.forecast_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, n_features),
        )

        # Module 4b: Learned feature importance weights for aggregating scores
        self.score_weights = nn.Parameter(torch.ones(n_features) / n_features)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, M)
        Returns:
          predictions:   (B, T-1, M)  — forecasts for t+1
          attn_weights:  (B, M, M)    — feature dependency graph (for explainability)
          anomaly_scores:(B, T-1)     — per-timestep anomaly probability
        """
        # ── Module 1 ──────────────────────────────────────────────────────
        x_emb = self.feature_embedding(x)              # (B, T, M, embed_dim)

        # ── Module 2 ──────────────────────────────────────────────────────
        attended, attn_weights = self.graph_learner(x_emb)  # attended: (B,T,M,D), attn: (B,M,M)

        # ── Module 3 ──────────────────────────────────────────────────────
        lstm_out = self.temporal_model(attended, attn_weights)  # (B, T, lstm_hidden)

        # ── Module 4a: Forecast ───────────────────────────────────────────
        # Use t=0..T-2 to predict t=1..T-1
        predictions = self.forecast_head(lstm_out[:, :-1, :])  # (B, T-1, M)

        # ── Module 4b: Anomaly scoring ────────────────────────────────────
        actual = x[:, 1:, :]                          # (B, T-1, M)  ground truth
        errors = torch.abs(predictions - actual)      # (B, T-1, M)  per-feature error

        # Weighted aggregation with softmax-normalised weights
        weights = F.softmax(self.score_weights, dim=0)   # (M,)
        anomaly_scores = (errors * weights).sum(dim=-1)  # (B, T-1)

        return predictions, attn_weights, anomaly_scores

    # ── Sparsity regularisation loss (encourages sparse graph) ───────────────
    def sparsity_loss(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """L1 on attention weights to encourage a sparse feature graph."""
        return self.sparsity_lambda * attn_weights.abs().mean()

    # ── Convenience method for inference ─────────────────────────────────────
    @torch.no_grad()
    def predict_anomaly_scores(
        self, x: torch.Tensor, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (anomaly_scores, attn_weights) without gradients.
        x: (1, T, M) single window tensor.
        """
        self.eval()
        x = x.to(device)
        _, attn, scores = self(x)
        return scores.squeeze(0).cpu(), attn.squeeze(0).cpu()
