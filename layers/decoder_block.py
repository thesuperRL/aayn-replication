import torch
from torch import nn
from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.masked_multi_head_attention = MultiHeadAttention(d_model, n_heads, mask=True)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output):
        masked_attn = self.masked_multi_head_attention(x, x, x)
        x = self.layer_norm1(x + masked_attn)
        enc_dec_attn = self.multi_head_attention(x, encoder_output, encoder_output)
        x = self.layer_norm2(x + enc_dec_attn)
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(x + ff_output)
        return x