import torch
from torch import nn
import math

class ScaledDotProdAttention(nn.Module):
    def __init__(self, dk, mask=None):
        super().__init__()
        self.dk = dk
        self.mask = mask

    def forward(self, Q, K, V):
        QK = torch.matmul(Q, K.transpose(-2, -1))
        QK = QK / math.sqrt(self.dk)
        if self.mask is not None:
            QK = QK.masked_fill(self.mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        QK = torch.softmax(QK, dim=-1)
        return torch.matmul(QK, V)