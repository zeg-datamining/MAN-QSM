import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_dot,series_mask,beta, mask_flag=False, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.mask_dot = mask_dot
        self.series_mask = series_mask
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()

        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

        for i in range(window_size):
            for j in range(beta):  # beta
                if i <= 99 - beta:  # 99-beta
                    self.distances[i][i + j] = 0  # 100,100)
                if i >= beta:  # beta
                    self.distances[i][i - j] = 0  # 100,100)



    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        M1 = scores.max(-1)[0] - torch.div(scores.sum(-1), L)

        sparse = scores.reshape(-1, 100, 100)
        sparse_y = torch.split(sparse, 1, 0)
        sparse_l1 = list(sparse_y)
        sparse_a = []
        for a in range(len(sparse_l1)):
            i=sparse_l1[a]
            i = i.squeeze(0)
            i = i.detach().cpu().numpy()
            for p in range(self.window_size):
                i[p][p] = 0
                for q in range(self.mask_dot):  # beta
                    if p + q <= 99:  # 99-beta
                        i[p][p + q] = 0  # 100,100)
                    if p - q >= 0:  # beta
                        i[p][p - q] = 0  # 100,100)
            i = torch.from_numpy(i)
            sparse_a.append(i)


        sparse_B = torch.stack(sparse_a, 0)

        sparse_att = sparse_B.reshape(B, -1, 100, 100).cuda()

        M2 = sparse_att.max(-1)[0] - torch.div(sparse_att.sum(-1), (L-2*self.mask_dot))  ##torch.Size([32, 8, 100])
        M=M1-M2


        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))

        ##mask
        series = series.reshape(-1, 100, 100)
        y = torch.split(series, 1, 0)
        l1 = list(y)
        s = []
        for i in l1:
            i = i.squeeze(0)
            diag = torch.diag(i)
            a_diag = torch.diag_embed(diag)
            at = i - a_diag
            at = at.squeeze(0)
            s.append(at)
        att = torch.stack(s, 0)
        series = att.reshape(B, -1, 100, 100)


        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma,M)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma,M = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma,M

