"""
CLAM-SB (Clustering-constrained Attention Multiple Instance Learning, Single Branch).
Lu et al., Nature Biomedical Engineering 2021.

Instance-level supervision uses the top-k / bottom-k attention patches as
pseudo-positive / pseudo-negative instances for an auxiliary instance classifier.
For survival tasks the instance loss is optional (set k_sample=0 to disable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Attn_Net_Gated, create_mlp, process_surv, process_clf
from .mist import MolecularStainingTransform


class CLAM_SB(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.mode = mode
        self.n_classes = config.n_classes
        self.k_sample = getattr(config, 'k_sample', 8)
        self.instance_loss_weight = getattr(config, 'instance_loss_weight', 0.3)
        self._use_mist = getattr(config, 'use_mist', False)

        if self._use_mist:
            self.mist = MolecularStainingTransform(
                in_dim=config.in_dim, out_dim=config.embed_dim,
                n_prototypes=config.n_prototypes, rank=config.mist_rank,
            )
            self.mlp_post = create_mlp(
                in_dim=config.embed_dim,
                hid_dims=[config.embed_dim] * (config.n_fc_layers - 2),
                dropout=config.dropout, out_dim=config.embed_dim, end_with_fc=False,
            ) if config.n_fc_layers > 1 else nn.Identity()
        else:
            self.mlp = create_mlp(
                in_dim=config.in_dim,
                hid_dims=[config.embed_dim] * (config.n_fc_layers - 1),
                dropout=config.dropout, out_dim=config.embed_dim, end_with_fc=False,
            )

        self.attention_net = Attn_Net_Gated(
            L=config.embed_dim,
            D=config.attn_dim,
            dropout=config.dropout,
            n_classes=1,
        )

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)

        # Instance-level classifier (binary: positive vs. negative)
        if self.k_sample > 0:
            self.instance_classifier = nn.Linear(config.embed_dim, 2)

    def _instance_loss(self, h, A):
        """Compute bag-label-propagated instance loss on top-k/bottom-k patches."""
        if self.k_sample <= 0 or not self.training:
            return torch.tensor(0.0, device=h.device)

        # A: (B, 1, N) — use first batch only (batch_size=1 in MIL)
        A_squeezed = A.squeeze(1).squeeze(0)          # (N,)
        top_idx = A_squeezed.topk(self.k_sample).indices
        bot_idx = A_squeezed.topk(self.k_sample, largest=False).indices

        top_feats = h[0, top_idx]   # (k, D)
        bot_feats = h[0, bot_idx]   # (k, D)

        pos_logits = self.instance_classifier(top_feats)
        neg_logits = self.instance_classifier(bot_feats)

        pos_targets = torch.ones(self.k_sample, dtype=torch.long, device=h.device)
        neg_targets = torch.zeros(self.k_sample, dtype=torch.long, device=h.device)

        inst_loss = (
            F.cross_entropy(pos_logits, pos_targets)
            + F.cross_entropy(neg_logits, neg_targets)
        ) / 2.0
        return inst_loss

    def forward_no_loss(self, h, attn_mask=None):
        if self._use_mist:
            h, affinity = self.mist(h)
            h = self.mlp_post(h)
        else:
            h = self.mlp(h)
            affinity = None

        A = self.attention_net(h)                   # (B, N, 1)
        A = A.transpose(-2, -1)                     # (B, 1, N)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(1) * torch.finfo(A.dtype).min
        A_softmax = F.softmax(A, dim=-1)

        M = torch.bmm(A_softmax, h).squeeze(1)     # (B, D)
        logits = self.classifier(M)

        out = {'logits': logits, 'attn': A_softmax, 'feats': h, 'feats_agg': M}
        if affinity is not None:
            out['staining_affinity'] = affinity
        return out

    def forward(self, h, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        out = self.forward_no_loss(h, attn_mask=model_kwargs.get('attn_mask'))
        logits = out['logits']

        if self.mode == 'classification':
            results_dict, log_dict = process_clf(logits, model_kwargs['label'], model_kwargs['loss_fn'])
        elif self.mode == 'survival':
            results_dict, log_dict = process_surv(
                logits, model_kwargs['label'], model_kwargs['censorship'], model_kwargs['loss_fn']
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        # Add instance loss (only during training, only for classification where bag labels exist)
        if self.k_sample > 0 and self.mode == 'classification' and self.training:
            inst_loss = self._instance_loss(out['feats'], out['attn'])
            if 'loss' in results_dict:
                results_dict['loss'] = results_dict['loss'] + self.instance_loss_weight * inst_loss
                log_dict['inst_loss'] = inst_loss.item()

        return results_dict, log_dict
