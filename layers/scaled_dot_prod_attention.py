import torch
from torch import nn
import math

class ScaledDotProdAttention(nn.Module):
    def __init__(self, d_model, mask=None):
        super().__init__()
        self.d_model = d_model
        self.mask = mask

    def forward(self, Q, K, V):
        QK = torch.matmul(Q, K.transpose(-2, -1))
        QK = QK / math.sqrt(self.d_model)
        if self.mask is not None:
            QK = QK.masked_fill(~self.mask, float('-inf'))
        QK = torch.softmax(QK, dim=-1)
        return torch.matmul(QK, V)