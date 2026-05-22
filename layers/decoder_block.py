import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.masked_multi_head_attention = MultiHeadAttention(d_model, n_heads, mask=torch.triu(torch.ones(d_model, d_model), diagonal=1))
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output):
        x = self.masked_multi_head_attention(x)
        x = self.layer_norm1(x)
        x = self.multi_head_attention(x, encoder_output)
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.layer_norm3(x)
        return x