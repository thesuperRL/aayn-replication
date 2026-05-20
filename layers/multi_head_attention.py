from torch import nn
import torch
from .scaled_dot_prod_attention import ScaledDotProdAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h_heads, mask=None):
        super().__init__()
        self.d_model = d_model
        self.h_heads = h_heads
        self.scaled_dot_prod_attention = ScaledDotProdAttention(d_model, mask)

    def forward(self, Q, K, V):
        heads = []
        for i in range(self.h_heads):
            Q_i = Q[i]
            K_i = K[i]
            V_i = V[i]
            head_i = self.scaled_dot_prod_attention(Q_i, K_i, V_i)
            heads.append(head_i)
        attention = torch.stack(heads, dim=0)
        return attention
