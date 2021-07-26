import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weight
from model.module import Linear
from model.embedding import PositionalEncoding
from utils import init_weight

class RelativeMultiHeadAttention(nn.Module):
    """
    input:
    query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    """

    def __init__(self, d_model=512, n_heads=16, dropout_p=0.1):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_head = int(d_model / n_heads) # transformer 참고 ) attention dim = d_model / heads
        # 32
        self.n_heads = n_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.linear_q = Linear(d_model, d_model)
        self.linear_k = Linear(d_model, d_model)
        self.linear_v = Linear(d_model, d_model)
        self.linear_pos = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)

        # todo print this
        self.u_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.n_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.fc = Linear(d_model, d_model)

    def forward(self, q, k, v, pos_embedding, mask=None):
        batch_size = v.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.n_heads, self.d_head)
        k = self.linear_q(k).view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # [batch, n_heads, -1, d_head]
        v = self.linear_q(v).view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # [batch, n_heads, -1, d_head]
        pos_embedding = self.linear_pos(pos_embedding).view(batch_size, -1, self.n_heads, self.d_head)

        content_score = torch.matmul((q + self.u_bias).transpose(1, 2), k.transpose(2, 3))
        # [batch, n_heads, -1, d_head] * [batch, n_heads, d_head, -1]
        pos_score = torch.matmul((q + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        # [batch, n_heads, -1 , d_head] * [batch, n_heads, d_head, -1]
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1) # [batch, n_heads, -1, -1]
        attn = self.dropout(attn)

        context = torch.matmul(attn, v).transpose(1, 2)
        # [batch, n_heads, -1, d_head] -> [batch, -1, n_heads, d_head]
        context = context.contiguous().view(batch_size, -1, self.d_model) # d_model = n_heads x d_head

        output = self.fc(context)

        return output

    def _relative_shift(self, pos_score):
        batch_size, n_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, n_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, n_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score) # pos_score size 같이 변형(요소 같아야함)

        return pos_score



class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    """

    def __init__(self, args, d_model, n_heads, dropout_p=0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.args = args

        self.positional_encoding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, n_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs, mask=None):
        batch_size, seq_length, _ = inputs.size()

        pos_embedding = self.positional_encoding(seq_length).cuda(self.args.gpu)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding, mask)

        return self.dropout(outputs)

####################################################################
# Decoder Attention
####################################################################
class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim, dropout_p=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            # print("score: ", score.size())
            # print("mask: ", mask.size())
            score.masked_fill_(mask, -np.inf)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)

        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "attention dim = d_model / heads 해야 하기 때문에"

        self.attn_dim = int(d_model / n_heads) # default:64
        self.n_heads = n_heads

        # todo 뒤 사이즈가 조금 다름 원래 attn_dim만 들어갔는데
        # Projection
        self.Linear_Q = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_K = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_V = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        init_weight(self.Linear_Q)
        init_weight(self.Linear_K)
        init_weight(self.Linear_V)

        self.scaled_dot_attn = ScaledDotProductAttention(self.attn_dim) # sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = v.size(0)

        # [Batch, Length, N, D] = [Batch, Length, 8, 64]
        query = self.Linear_Q(q).view(batch_size, -1, self.n_heads, self.attn_dim)
        key = self.Linear_K(k).view(batch_size, -1, self.n_heads, self.attn_dim)
        value = self.Linear_V(v).view(batch_size, -1, self.n_heads, self.attn_dim)

        # [Batch * N, Length, Dim]
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)

        # mask
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.n_heads, batch_size, -1, self.attn_dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_heads * self.attn_dim)

        return context, attn








































