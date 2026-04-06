import torch
import torch.nn as nn
import numpy as np


class MolecularStainingTransform(nn.Module):
    """
    MIST: Molecularly Informed Spatial Transcriptomics Transform

    Injects ST priors into patch embeddings via K learnable molecular prototypes
    that act as virtual molecular stains. Inserted between frozen patch encoder
    and MIL aggregator. No gene data required at inference.

    Args:
        in_dim:       input patch feature dimension
        out_dim:      output embedding dimension (= model working dim)
        n_prototypes: number of molecular prototypes (K)
        rank:         low-rank dim for per-prototype transform
    """
    def __init__(self, in_dim: int, out_dim: int, n_prototypes: int = 16, rank: int = 16):
        super().__init__()
        self.K = n_prototypes
        self.rank = rank

        self.prototypes = nn.Parameter(torch.randn(n_prototypes, in_dim))
        self.shared_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.V = nn.Linear(out_dim, rank, bias=False)
        self.U = nn.ModuleList([nn.Linear(rank, out_dim, bias=False) for _ in range(n_prototypes)])

    def init_prototypes(self, st_prototypes):
        """Load ST-derived prototype vectors. Shape: (K, in_dim) or np.ndarray."""
        if isinstance(st_prototypes, np.ndarray):
            st_prototypes = torch.FloatTensor(st_prototypes)
        st_prototypes = st_prototypes.to(self.prototypes.device)
        assert st_prototypes.shape == self.prototypes.shape, \
            f"Expected {self.prototypes.shape}, got {st_prototypes.shape}"
        with torch.no_grad():
            self.prototypes.copy_(st_prototypes)

    def forward(self, x):
        """
        Args:
            x: (B, N, D_in)
        Returns:
            x_out:    (B, N, D_out)
            affinity: (B, N, K)  — staining affinity scores for interpretability
        """
        x_low = self.shared_proj(x)                                            # (B, N, D_out)
        proto = self.shared_proj(self.prototypes)                              # (K, D_out)

        affinity = torch.sigmoid(torch.einsum('bnd,kd->bnk', x_low, proto))   # (B, N, K)
        affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-8)

        low = self.V(x_low)                                                    # (B, N, r)
        per_proto = torch.stack([self.U[k](low) for k in range(self.K)], dim=0)  # (K, B, N, D_out)
        views = x_low.unsqueeze(2) + per_proto.permute(1, 2, 0, 3)            # (B, N, K, D_out)

        stained = (affinity.unsqueeze(-1) * views).sum(dim=2)                 # (B, N, D_out)
        x_out = x_low + stained

        return x_out, affinity
