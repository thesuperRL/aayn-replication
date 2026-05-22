import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_output = self.multi_head_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        return x