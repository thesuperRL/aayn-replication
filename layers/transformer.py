import torch
from torch import nn
import math
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers_encoder, n_layers_decoder, n_heads, d_ff):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model, n_layers_encoder, n_heads, d_ff, src_vocab_size)
        self.decoder = Decoder(d_model, n_layers_decoder, n_heads, d_ff, tgt_vocab_size)
        self.linear_output = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)
        logits = self.linear_output(decoder_output)
        return self.softmax(logits)