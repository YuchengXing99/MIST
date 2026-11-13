import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import process_surv, process_clf
from .mist import MolecularStainingTransform


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU()) if gated else None

    def forward(self, Q, K):
        Q0 = Q
        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)
        A, _ = self.multihead_attn(Q, K, V)
        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        if self.gate is not None:
            O = O.mul(self.gate(Q0))
        return O


class GAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.latent)
        self.project_forward  = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)
        return self.project_backward(X, H)


class NLP(nn.Module):
    def __init__(self, dim, num_heads, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, 1, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mha(self.S.repeat(X.size(0), 1, 1), X)


class ILRA(nn.Module):
    def __init__(self, config, mode='survival'):
        super().__init__()
        self.config = config
        self.mode = mode
        self._use_mist = getattr(config, 'use_mist', False)

        if self._use_mist:
            self.mist = MolecularStainingTransform(
                in_dim=config.in_dim,
                out_dim=config.hidden_feat,
                n_prototypes=config.n_prototypes,
                rank=config.mist_rank,
            )
            first_dim_in = config.hidden_feat
        else:
            first_dim_in = config.in_dim

        gab_blocks = []
        for i in range(config.num_layers):
            gab_blocks.append(GAB(
                dim_in=first_dim_in if i == 0 else config.hidden_feat,
                dim_out=config.hidden_feat,
                num_heads=config.num_heads,
                num_inds=config.topk,
                ln=config.ln,
            ))
        self.gab_blocks = nn.ModuleList(gab_blocks)
        self.pooling = NLP(dim=config.hidden_feat, num_heads=config.num_heads, ln=config.ln)
        self.classifier = nn.Linear(config.hidden_feat, config.n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_no_loss(self, x):
        if self._use_mist:
            x, affinity = self.mist(x)
        else:
            affinity = None

        for block in self.gab_blocks:
            x = block(x)

        feat = self.pooling(x)
        logits = self.classifier(feat).squeeze(1)

        out = {
            'logits': logits,
            'prob': F.softmax(logits, dim=1),
            'pred': torch.topk(logits, 1, dim=1)[1],
            'feat': feat,
        }
        if affinity is not None:
            out['staining_affinity'] = affinity
        return out

    def forward(self, x, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        out = self.forward_no_loss(x)
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
