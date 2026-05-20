import torch
from torch import nn
import math

class EncoderBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.multi_head_attention(x)
        x = self.layer_norm1(x)
        x = self.feed_forward(x)
        x = self.layer_norm2(x)
        return x