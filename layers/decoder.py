from torch import nn
from .decoder_block import DecoderBlock
from .positional_encoding import PositionalEncoding

class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.input_embedding = PositionalEncoding(d_model)
        self.decoder_block = DecoderBlock(d_model, n_heads)
        
    def forward(self, x, encoder_output):
        x = self.input_embedding(x)
        for _ in range(self.n_layers):
            x = self.decoder_block(x, encoder_output)
        return x