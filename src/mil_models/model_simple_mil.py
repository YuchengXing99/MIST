"""
MeanMIL and MaxMIL — simple non-parametric baselines from the MAMMOTH paper.

MeanMIL: mean-pool task-specific patch embeddings → classify.
MaxMIL:  per-patch classify → select the patch with the highest max logit.
"""

import torch
import torch.nn as nn

from .components import create_mlp, process_surv, process_clf
from .mist import MolecularStainingTransform


class MeanMIL(nn.Module):
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

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)

    def forward_no_loss(self, h, attn_mask=None):
        if self._use_mist:
            h, affinity = self.mist(h)
            h = self.mlp_post(h)
        else:
            h = self.mlp(h)
            affinity = None

        M = h.mean(dim=1)                       # (B, D)
        logits = self.classifier(M)             # (B, n_classes)
        out = {'logits': logits, 'feats': h, 'feats_agg': M}
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


class MaxMIL(nn.Module):
    """
    Per-patch classification; the patch with the highest max logit determines
    the slide-level prediction (equivalent to max-pooling over logits).
    """
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

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)

    def forward_no_loss(self, h, attn_mask=None):
        if self._use_mist:
            h, affinity = self.mist(h)
            h = self.mlp_post(h)
        else:
            h = self.mlp(h)
            affinity = None

        patch_logits = self.classifier(h)           # (B, N, n_classes)
        max_vals = patch_logits.max(dim=-1).values  # (B, N)
        best_idx = max_vals.argmax(dim=-1)          # (B,)
        logits = patch_logits[torch.arange(patch_logits.size(0)), best_idx]  # (B, n_classes)
        out = {'logits': logits, 'feats': h, 'patch_logits': patch_logits}
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
