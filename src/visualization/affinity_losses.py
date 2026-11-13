"""
MIST affinity regularization losses.

Pluggable losses that operate on the staining_affinity tensor (B, N, K)
returned by MolecularStainingTransform. Can be added to the main training
loss to encourage more interpretable affinity distributions.

Usage in trainer.py:
    from visualization.affinity_losses import MISTRegularization
    mist_reg = MISTRegularization(entropy_weight=0.1, diversity_weight=0.1)
    ...
    if 'staining_affinity' in out:
        reg = mist_reg(out['staining_affinity'])
        total_loss = out['loss'] + reg['loss']
"""

import torch
import torch.nn as nn


class AffinityEntropyLoss(nn.Module):
    """
    Encourage sharp (low-entropy) affinity distributions per patch.

    Given affinity (B, N, K) already L1-normalized over K,
    minimising this loss pushes each patch toward high affinity
    to few prototypes rather than a uniform spread.

    Loss = mean over patches of H(affinity) = -sum_k a_k log(a_k)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, affinity: torch.Tensor) -> torch.Tensor:
        """affinity: (B, N, K), L1-normalised over K."""
        entropy = -(affinity * torch.log(affinity + self.eps)).sum(dim=-1)  # (B, N)
        return entropy.mean()


class AffinityDiversityLoss(nn.Module):
    """
    Encourage different prototypes to attend to different patches.

    Each prototype k has a usage vector u_k = affinity[:, :, k] of shape (N,).
    High cosine similarity between two usage vectors means the prototypes
    are redundantly covering the same patches. We penalise this.

    Loss = mean of upper-triangular pairwise cosine similarities.
    """

    def forward(self, affinity: torch.Tensor) -> torch.Tensor:
        """affinity: (B, N, K)"""
        # (B, K, N) — each row is a prototype's usage pattern
        usage = affinity.transpose(1, 2)
        usage_norm = usage / (usage.norm(dim=-1, keepdim=True) + 1e-8)
        sim = torch.bmm(usage_norm, usage_norm.transpose(1, 2))  # (B, K, K)

        K = sim.size(1)
        mask = torch.triu(torch.ones(K, K, device=sim.device, dtype=torch.bool), diagonal=1)
        return sim[:, mask].mean()


class MISTRegularization(nn.Module):
    """
    Combined MIST affinity regularisation.

    total = entropy_weight * entropy_loss + diversity_weight * diversity_loss

    Args:
        entropy_weight:   weight for sharpness loss (default 0.1)
        diversity_weight: weight for diversity loss (default 0.1)
    """

    def __init__(self, entropy_weight: float = 0.1, diversity_weight: float = 0.1):
        super().__init__()
        self.entropy_loss = AffinityEntropyLoss()
        self.diversity_loss = AffinityDiversityLoss()
        self.w_ent = entropy_weight
        self.w_div = diversity_weight

    def forward(self, affinity: torch.Tensor) -> dict:
        """
        Args:
            affinity: (B, N, K) from model output['staining_affinity']

        Returns:
            dict with 'loss', 'entropy_loss', 'diversity_loss'
        """
        ent = self.entropy_loss(affinity)
        div = self.diversity_loss(affinity)
        total = self.w_ent * ent + self.w_div * div
        return {'loss': total, 'entropy_loss': ent, 'diversity_loss': div}
