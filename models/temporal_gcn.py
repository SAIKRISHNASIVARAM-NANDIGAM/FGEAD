"""
models/temporal_gcn.py
Module 3: Graph Convolution + Temporal LSTM

GCN propagates information along the learned feature graph.
LSTM captures temporal dynamics over graph-enhanced features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    One layer of graph convolution.
    h_new = σ(A_norm @ h @ W)
    where A_norm is the row-normalised attention-based adjacency.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h:   (B, M, in_dim)  — node features
        adj: (B, M, M)       — adjacency matrix (learned graph)
        Returns: (B, M, out_dim)
        """
        # Row-normalise adjacency so each node sums to 1
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        adj_norm = adj / row_sum

        # Aggregate: each node = weighted sum of neighbour features
        agg = torch.bmm(adj_norm, h)   # (B, M, in_dim)
        out = self.W(agg)              # (B, M, out_dim)
        out = F.relu(out)

        # BatchNorm over the node/feature dimension
        B, M, D = out.shape
        out = self.bn(out.view(B * M, D)).view(B, M, D)
        return out


class TemporalGraphModel(nn.Module):
    """
    Combines GCN with LSTM for spatio-temporal modelling.
    Per timestep: apply GCN → get graph-enhanced features → feed to LSTM.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        gcn_out: int = 64,
        lstm_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn = GraphConvolution(embed_dim, gcn_out)
        self.lstm = nn.LSTM(
            input_size=n_features * gcn_out,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.n_features = n_features
        self.gcn_out = gcn_out

    def forward(
        self, x_emb: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """
        x_emb: (B, T, M, embed_dim)
        adj:   (B, M, M)
        Returns: lstm_out (B, T, lstm_hidden)
        """
        B, T, M, D = x_emb.shape
        gcn_outs = []

        for t in range(T):
            h_t = x_emb[:, t, :, :]       # (B, M, D)
            g_t = self.gcn(h_t, adj)       # (B, M, gcn_out)
            gcn_outs.append(g_t)

        # Stack timesteps and flatten node features
        gcn_seq = torch.stack(gcn_outs, dim=1)                # (B, T, M, gcn_out)
        gcn_flat = gcn_seq.view(B, T, M * self.gcn_out)       # (B, T, M*gcn_out)

        lstm_out, _ = self.lstm(gcn_flat)                     # (B, T, lstm_hidden)
        return lstm_out
