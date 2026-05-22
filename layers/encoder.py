from torch import nn
from .encoder_block import EncoderBlock
from .positional_encoding import PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads):
        super().__init__()
        self.n_layers = n_layers
        self.input_embedding = PositionalEncoding(d_model)
        self.encoder_block = EncoderBlock(d_model, n_heads)
        
    def forward(self, x):
        x = self.input_embedding(x)
        for _ in range(self.n_layers):
            x = self.encoder_block(x)
        return x