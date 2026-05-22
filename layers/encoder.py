from torch import nn
from .encoder_block import EncoderBlock
from .positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size):
        super().__init__()
        self.n_layers = n_layers
        self.input_embedding = PositionalEncoding(d_model, vocab_size=vocab_size)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        x = self.input_embedding(x)
        for block in self.encoder_blocks:
            x = block(x)
        return x