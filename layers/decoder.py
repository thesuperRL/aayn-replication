from torch import nn
from .decoder_block import DecoderBlock
from .positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, vocab_size):
        super().__init__()
        self.n_layers = n_layers
        self.input_embedding = PositionalEncoding(d_model, vocab_size=vocab_size)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
    def forward(self, x, encoder_output):
        x = self.input_embedding(x)
        for block in self.decoder_blocks:
            x = block(x, encoder_output)
        return x