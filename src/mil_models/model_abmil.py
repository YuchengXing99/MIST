import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_surv, process_clf
from .mist import MolecularStainingTransform


class ABMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.n_classes = config.n_classes
        self._use_mist = getattr(config, 'use_mist', False)

        if self._use_mist:
            self.mist = MolecularStainingTransform(
                in_dim=config.in_dim,
                out_dim=config.embed_dim,
                n_prototypes=config.n_prototypes,
                rank=config.mist_rank,
            )
            self.mlp_post = create_mlp(
                in_dim=config.embed_dim,
                hid_dims=[config.embed_dim] * (config.n_fc_layers - 2),
                dropout=config.dropout,
                out_dim=config.embed_dim,
                end_with_fc=False,
            ) if config.n_fc_layers > 1 else nn.Identity()
        else:
            self.mlp = create_mlp(
                in_dim=config.in_dim,
                hid_dims=[config.embed_dim] * (config.n_fc_layers - 1),
                dropout=config.dropout,
                out_dim=config.embed_dim,
                end_with_fc=False,
            )

        if config.gate:
            self.attention_net = Attn_Net_Gated(
                L=config.embed_dim, D=config.attn_dim,
                dropout=config.dropout, n_classes=1,
            )
        else:
            self.attention_net = Attn_Net(
                L=config.embed_dim, D=config.attn_dim,
                dropout=config.dropout, n_classes=1,
            )

        self.classifier = nn.Linear(config.embed_dim, config.n_classes)

    def forward_attention(self, h, attn_only=False):
        if self._use_mist:
            h, affinity = self.mist(h)
            h = self.mlp_post(h)
        else:
            h = self.mlp(h)
            affinity = None

        A = self.attention_net(h)
        A = torch.transpose(A, -2, -1)
        if attn_only:
            return A
        return h, A, affinity

    def forward_no_loss(self, h, attn_mask=None):
        h, A, affinity = self.forward_attention(h)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)
        M = torch.bmm(A, h).squeeze(dim=1)
        logits = self.classifier(M)

        out = {'logits': logits, 'attn': A, 'feats': h, 'feats_agg': M}
        if affinity is not None:
            out['staining_affinity'] = affinity
        return out

    def forward(self, h, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        if self.mode == 'classification':
            out = self.forward_no_loss(h, attn_mask=model_kwargs.get('attn_mask'))
            results_dict, log_dict = process_clf(out['logits'], model_kwargs['label'], model_kwargs['loss_fn'])
        elif self.mode == 'survival':
            out = self.forward_no_loss(h, attn_mask=model_kwargs.get('attn_mask'))
            results_dict, log_dict = process_surv(
                out['logits'], model_kwargs['label'], model_kwargs['censorship'], model_kwargs['loss_fn']
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        return results_dict, log_dict
