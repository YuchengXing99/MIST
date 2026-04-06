import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention

from .components import process_surv, process_clf
from .mist import MolecularStainingTransform


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, n_heads=8,
                 num_landmarks=256, pinv_iterations=6, dropout=0.25):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=max(1, dim // n_heads),
            heads=n_heads,
            num_landmarks=num_landmarks,
            pinv_iterations=pinv_iterations,
            residual=True,
            dropout=dropout,
        )

    def forward(self, x):
        return x + self.attn(self.norm(x))


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.proj  = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, h_size, w_size):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, h_size, w_size)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), x), dim=1)


class TransMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.n_classes = config.n_classes
        self.embed_dim = config.embed_dim
        self._use_mist = getattr(config, 'use_mist', False)

        if self._use_mist:
            self.mist = MolecularStainingTransform(
                in_dim=config.in_dim,
                out_dim=self.embed_dim,
                n_prototypes=config.n_prototypes,
                rank=config.mist_rank,
            )
        else:
            self._fc1 = nn.Sequential(nn.Linear(config.in_dim, self.embed_dim), nn.ReLU())

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_layer = PPEG(dim=self.embed_dim)
        self.layer1 = TransLayer(
            dim=self.embed_dim,
            n_heads=config.n_heads,
            num_landmarks=config.num_landmarks,
            pinv_iterations=config.pinv_iterations,
            dropout=config.dropout,
        )
        self.layer2 = TransLayer(
            dim=self.embed_dim,
            n_heads=config.n_heads,
            num_landmarks=config.num_landmarks,
            pinv_iterations=config.pinv_iterations,
            dropout=config.dropout,
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, self.n_classes)

    def forward_no_loss(self, h, attn_mask=None):
        B, N, _ = h.shape

        if self._use_mist:
            h, affinity = self.mist(h)
        else:
            h = self._fc1(h)
            affinity = None

        h_size = w_size = int(np.ceil(np.sqrt(N)))
        add_len = h_size * w_size - N
        if add_len > 0:
            h = torch.cat([h, h[:, :add_len]], dim=1)

        cls_token = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_token, h), dim=1)
        h = self.layer1(h)
        h = self.pos_layer(h, h_size, w_size)
        h = self.layer2(h)
        h = self.norm(h)

        cls_feat = h[:, 0]
        logits = self.classifier(cls_feat)

        out = {'logits': logits, 'attn': None, 'feats': h[:, 1:], 'feats_agg': cls_feat}
        if affinity is not None:
            out['staining_affinity'] = affinity
        return out

    def forward(self, h, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        out = self.forward_no_loss(h)
        logits = out['logits']

        if self.mode == 'classification':
            results_dict, log_dict = process_clf(logits, model_kwargs['label'], model_kwargs['loss_fn'])
        elif self.mode == 'survival':
            results_dict, log_dict = process_surv(
                logits, model_kwargs['label'], model_kwargs['censorship'], model_kwargs['loss_fn']
            )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")

        return results_dict, log_dict
