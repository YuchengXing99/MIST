"""
Transformer-based MIL aggregator.
Wagner et al., Cancer Cell 2023; Vaswani et al., NeurIPS 2017.

Standard transformer encoder applied to patch embeddings.  A learned [CLS]
token is prepended; the model reads off its representation for slide-level
prediction — analogous to BERT-style classification.
"""

import torch
import torch.nn as nn

from .components import process_surv, process_clf
from .mist import MolecularStainingTransform


class TransformerMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.mode = mode
        self.n_classes = config.n_classes
        self.embed_dim = config.embed_dim
        self._use_mist = getattr(config, 'use_mist', False)

        if self._use_mist:
            self.mist = MolecularStainingTransform(
                in_dim=config.in_dim, out_dim=config.embed_dim,
                n_prototypes=config.n_prototypes, rank=config.mist_rank,
            )
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(config.in_dim, config.embed_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.n_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,          # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, config.n_classes)

    def forward_no_loss(self, h, attn_mask=None):
        B = h.size(0)
        if self._use_mist:
            h, affinity = self.mist(h)
        else:
            h = self.input_proj(h)
            affinity = None

        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        h = torch.cat([cls, h], dim=1)                  # (B, N+1, D)

        h = self.transformer(h)                         # (B, N+1, D)
        h = self.norm(h)

        cls_feat = h[:, 0]                              # (B, D)
        logits = self.classifier(cls_feat)

        out = {'logits': logits, 'feats': h[:, 1:], 'feats_agg': cls_feat}
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
