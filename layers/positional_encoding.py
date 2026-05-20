import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def token2vec(self, token, position):
        token_vec = []
        for i in range(self.d_model // 2):
            token_vec.append(token + math.sin(position / 10000 ** (2*i/self.d_model)))
            token_vec.append(token + math.cos(position / 10000 ** (2*i/self.d_model)))
        return token_vec
    
    def forward(self, x):
        vecs = []
        for token_position in range(x.shape[0]):
            vec = self.token2vec(x[token_position], token_position)
            vecs.append(vec)
        return torch.tensor(vecs)