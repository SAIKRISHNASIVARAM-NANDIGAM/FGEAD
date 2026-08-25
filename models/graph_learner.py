"""
models/graph_learner.py
Module 1: Feature Embedding
Module 2: Self-Attention Graph Learning

Learns WHICH features are correlated using multi-head attention.
The attention matrix A[i,j] becomes the feature dependency graph.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbedding(nn.Module):
    """
    Embed each feature (metric) into a d-dimensional latent space.
    Like word embeddings in NLP — each metric gets a learned
    'personality vector' that captures what it represents.
    """

    def __init__(self, n_features: int, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_features, embed_dim)      # learnable identity emb
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, M)
        Returns: (B, T, M, embed_dim)
        """
        B, T, M = x.shape

        # Feature identity embeddings (one per feature index)
        feat_ids = torch.arange(M, device=x.device)          # (M,)
        feat_emb = self.embedding(feat_ids)                   # (M, embed_dim)
        feat_emb = feat_emb.unsqueeze(0).unsqueeze(0)         # (1, 1, M, embed_dim)
        feat_emb = feat_emb.expand(B, T, -1, -1)             # (B, T, M, embed_dim)

        # Scale feature values into embedding space
        val_emb = x.unsqueeze(-1) * feat_emb                  # (B, T, M, embed_dim)

        return self.dropout(self.layer_norm(val_emb))


class SelfAttentionGraphLearner(nn.Module):
    """
    Learn the graph (which features relate to which) using attention.
    Attention score A[i,j] = how much feature i is influenced by feature j.

    This is the KEY innovation: graph structure is LEARNED, not hand-designed!
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        sparsity_threshold: float = 0.05,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.threshold = sparsity_threshold

        # Q, K, V projections for self-attention across FEATURES (not time)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_emb: (B, T, M, embed_dim)
        Returns:
          attended:     (B, T, M, embed_dim) — graph-enhanced features
          attn_weights: (B, M, M)            — feature dependency graph
        """
        B, T, M, D = x_emb.shape

        # Pool over time to compute a single graph per sequence
        x_pool = x_emb.mean(dim=1)     # (B, M, D)

        Q = self.q_proj(x_pool)        # (B, M, D)
        K = self.k_proj(x_pool)
        V = self.v_proj(x_pool)

        # Reshape for multi-head attention
        Q = Q.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, M, d)
        K = K.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (scale * 0.1)          # (B, H, M, M)
        attn = F.softmax(scores/0.1, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                      # (B, H, M, d)
        out = out.transpose(1, 2).contiguous().view(B, M, D)             # (B, M, D)
        out = self.out_proj(out)

        # Average attention weights across heads → feature dependency graph
        attn_weights = attn.mean(dim=1)                                  # (B, M, M)

        # Sparsify: keep only strong connections
        k = 5

        topk_vals, topk_idx = torch.topk(attn_weights, k=k, dim=-1)

        mask = torch.zeros_like(attn_weights)

        mask.scatter_(-1, topk_idx, 1.0)

        sparse_graph = attn_weights * mask

        # Broadcast graph-enhanced representations back to temporal shape
        out_expanded = out.unsqueeze(1).expand(-1, T, -1, -1)           # (B, T, M, D)
        attended = x_emb + out_expanded                                  # residual

        return attended, sparse_graph
