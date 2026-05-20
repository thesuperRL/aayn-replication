from torch import nn
import torch
from .scaled_dot_prod_attention import ScaledDotProdAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h_heads, mask=None):
        super().__init__()
        self.d_model = d_model
        self.h_heads = h_heads
        self.dk = d_model // self.h_heads
        self.dv = d_model // self.h_heads
        self.scaled_dot_prod_attention = ScaledDotProdAttention(self.d_model, mask)
        self.linear_q = nn.Linear(self.d_model, self.dk)
        self.linear_k = nn.Linear(self.d_model, self.dk)
        self.linear_v = nn.Linear(self.d_model, self.dv)
        self.linear_output = nn.Linear(self.d_model, self.d_model)

    def forward(self, Q, K, V):
        # "linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively."
        Q = self.linear_q(Q)
        K = self.linear_k(K)
        V = self.linear_v(V)

        # "On each of these projected versions of queries, keys and values"
        Q_heads = torch.split(Q, self.dk, dim=-1)
        K_heads = torch.split(K, self.dk, dim=-1)
        V_heads = torch.split(V, self.dv, dim=-1)

        # "we then perform the attention function in parallel, yielding dv-dimensional output values."
        heads = []
        for Q_i, K_i, V_i in zip(Q_heads, K_heads, V_heads):
            head_i = self.scaled_dot_prod_attention(Q_i, K_i, V_i)
            heads.append(head_i)
        
        # "These are concatenated and once again projected, resulting in the final values"
        concatenation = torch.cat(heads, dim=-1)
        return self.linear_output(concatenation)
