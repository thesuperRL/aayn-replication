from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # "Another way of describing this is as two convolutions with kernel size 1."
        self.fc1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
    
    def forward(self, x):
        # "two linear transformations with a ReLU activation in between."
        # Conv1d expects (batch, channels, length), so transpose from (batch, length, channels)
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # Transpose back to (batch, length, channels)
        x = x.transpose(1, 2)
        return x