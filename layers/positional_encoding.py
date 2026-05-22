import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    def token2vec(self, token, position):
        token_vec = []
        for i in range(self.d_model // 2):
            token_vec.append(token + math.sin(position / 10000 ** (2*i/self.d_model)))
            token_vec.append(token + math.cos(position / 10000 ** (2*i/self.d_model)))
        return token_vec
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        vecs = []
        for batch_idx in range(batch_size):
            batch_vecs = []
            for token_position in range(seq_len):
                token = x[batch_idx, token_position].item()
                vec = self.token2vec(token, token_position)
                batch_vecs.append(vec)
            vecs.append(batch_vecs)
        return torch.tensor(vecs, dtype=torch.float32)