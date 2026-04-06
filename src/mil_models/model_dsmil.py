"""
DSMIL — Dual-stream Multiple Instance Learning.
Li et al., CVPR 2021.

Two parallel streams:
  1. Instance stream  : per-patch classifier; max-pooled logit → instance prediction.
  2. Bag stream       : cross-attention between instance embeddings and a query
                        formed from the max-critical instance → bag prediction.
Final logits are the average of the two streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import create_mlp, process_surv, process_clf
from .mist import MolecularStainingTransform


class FCLayer(nn.Module):
    """Per-patch fully-connected layer (instance-level classifier)."""
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.fc(x)


class BClassifier(nn.Module):
    """Bag-level classifier via attention from critical instance."""
    def __init__(self, embed_dim, n_classes, dropout=0.0):
        super().__init__()
        # Nonlinear query projection matching original DSMIL (Li et al., CVPR 2021)
        self.q = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.v = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim))
        self.fcc = nn.Linear(embed_dim, n_classes)

    def forward(self, x, x_q):
        """
        x   : (B, N, D) — all patch embeddings after instance projection
        x_q : (B, D)    — query = embedding of the max-critical instance
        """
        q = self.q(x_q).unsqueeze(1)        # (B, 1, 128)
        k = self.q(x)                       # (B, N, 128)
        A = torch.bmm(q, k.transpose(1, 2)) / (128 ** 0.5)  # (B, 1, N)
        A = F.softmax(A, dim=-1)

        v = self.v(x)                       # (B, N, D)
        M = torch.bmm(A, v).squeeze(1)     # (B, D)
        logits = self.fcc(M)
        return logits, A


class DSMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.mode = mode
        self.n_classes = config.n_classes
        self._use_mist = getattr(config, 'use_mist', False)

        if self._use_mist:
            self.mist = MolecularStainingTransform(
                in_dim=config.in_dim, out_dim=config.embed_dim,
                n_prototypes=config.n_prototypes, rank=config.mist_rank,
            )
        else:
            self.instance_proj = FCLayer(config.in_dim, config.embed_dim, dropout=config.dropout)

        self.instance_classifier = nn.Linear(config.embed_dim, config.n_classes)
        self.bag_classifier = BClassifier(config.embed_dim, config.n_classes, dropout=config.dropout)

    def forward_no_loss(self, h, attn_mask=None):
        if self._use_mist:
            h, affinity = self.mist(h)
        else:
            h = self.instance_proj(h)
            affinity = None

        inst_logits = self.instance_classifier(h)          # (B, N, n_classes)

        # Critical instance: patch with max score across classes
        max_scores = inst_logits.max(dim=-1).values        # (B, N)
        crit_idx = max_scores.argmax(dim=1)                # (B,)
        x_q = h[torch.arange(h.size(0)), crit_idx]        # (B, D)

        bag_logits, A = self.bag_classifier(h, x_q)       # (B, n_classes), (B, 1, N)

        # Slide-level logit = average of critical instance logit + bag logit
        crit_logits = inst_logits[torch.arange(h.size(0)), crit_idx]  # (B, n_classes)
        logits = (crit_logits + bag_logits) / 2.0

        out = {'logits': logits, 'attn': A, 'feats': h,
               'inst_logits': inst_logits, 'bag_logits': bag_logits}
        if affinity is not None:
            out['staining_affinity'] = affinity
        return out

    def forward(self, h, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        out = self.forward_no_loss(h)
        if self.mode == 'classification':
            results_dict, log_dict = process_clf(out['logits'], model_kwargs['label'], model_kwargs['loss_fn'])
        elif self.mode == 'survival':
            results_dict, log_dict = process_surv(
                out['logits'], model_kwargs['label'], model_kwargs['censorship'], model_kwargs['loss_fn']
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        return results_dict, log_dict
